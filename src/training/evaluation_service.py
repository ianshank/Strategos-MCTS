"""
Evaluation Service for Dynamic Model Evaluation Metrics.

Provides:
- Real metrics computation from evaluation games
- Statistical significance testing with confidence intervals
- Moving averages and trend detection
- Anomaly detection for evaluation stability
- Multiple evaluation strategies (self-play, benchmark, A/B comparison)
- Configuration-driven thresholds (no hardcoded values)

Best Practices 2025:
- All thresholds from configuration
- Proper logging with correlation IDs
- Type-safe with comprehensive type hints
- Statistical rigor with confidence intervals
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import torch
import torch.nn as nn

from ..observability.logging import get_correlation_id, get_structured_logger

if TYPE_CHECKING:
    from ..config.settings import Settings
    from ..framework.mcts.neural_mcts import GameState, NeuralMCTS


# Type variable for model types
ModelT = TypeVar("ModelT", bound=nn.Module)


class EvaluationStrategy(str, Enum):
    """Supported evaluation strategies."""

    SELF_PLAY = "self_play"  # Evaluate against self
    BENCHMARK = "benchmark"  # Evaluate against fixed benchmark
    AB_COMPARISON = "ab_comparison"  # A/B test between two models


class GameOutcome(str, Enum):
    """Possible game outcomes."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation service.

    All values can be overridden from Settings or provided explicitly.
    No hardcoded defaults should bypass configuration.
    """

    # Core evaluation parameters
    num_games: int = 100
    min_games_for_significance: int = 30
    win_threshold: float = 0.55
    confidence_level: float = 0.95

    # MCTS parameters for evaluation
    mcts_iterations: int = 100
    temperature: float = 0.0  # Deterministic for evaluation

    # Moving average configuration
    moving_average_window: int = 50
    trend_detection_threshold: float = 0.05

    # Anomaly detection
    anomaly_std_threshold: float = 2.5  # Z-score threshold
    min_samples_for_anomaly: int = 10

    # Timeout and performance
    max_moves_per_game: int = 500
    game_timeout_seconds: float = 300.0

    @classmethod
    def from_settings(cls, settings: Settings) -> EvaluationConfig:
        """
        Create configuration from application settings.

        Uses settings values where available, with sensible defaults
        that can be overridden via environment variables.
        """
        return cls(
            num_games=getattr(settings, "EVAL_NUM_GAMES", 100),
            min_games_for_significance=getattr(settings, "EVAL_MIN_GAMES_SIGNIFICANCE", 30),
            win_threshold=getattr(settings, "EVAL_WIN_THRESHOLD", 0.55),
            confidence_level=getattr(settings, "EVAL_CONFIDENCE_LEVEL", 0.95),
            mcts_iterations=settings.MCTS_ITERATIONS,
            temperature=getattr(settings, "EVAL_TEMPERATURE", 0.0),
            moving_average_window=getattr(settings, "EVAL_MOVING_AVG_WINDOW", 50),
            trend_detection_threshold=getattr(settings, "EVAL_TREND_THRESHOLD", 0.05),
            anomaly_std_threshold=getattr(settings, "EVAL_ANOMALY_STD_THRESHOLD", 2.5),
            min_samples_for_anomaly=getattr(settings, "EVAL_MIN_SAMPLES_ANOMALY", 10),
            max_moves_per_game=getattr(settings, "EVAL_MAX_MOVES", 500),
            game_timeout_seconds=getattr(settings, "EVAL_TIMEOUT_SECONDS", 300.0),
        )


@dataclass
class GameResult:
    """Result of a single evaluation game."""

    outcome: GameOutcome
    game_length: int
    model1_avg_value: float
    model2_avg_value: float
    model1_started: bool
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Comprehensive evaluation result with statistical metrics.

    Provides confidence intervals and statistical significance testing
    for rigorous model comparison.
    """

    # Core metrics
    win_rate: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    draw_rate: float
    loss_rate: float
    avg_game_length: float
    games_played: int

    # Statistical significance
    is_statistically_significant: bool
    p_value: float | None = None
    effect_size: float | None = None

    # Raw counts
    wins: int = 0
    losses: int = 0
    draws: int = 0

    # Performance metrics
    avg_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0

    # Model comparison
    model_is_better: bool = False
    improvement_margin: float = 0.0

    # Metadata
    strategy: EvaluationStrategy = EvaluationStrategy.SELF_PLAY
    evaluation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            "win_rate": self.win_rate,
            "win_rate_ci_low": self.win_rate_ci_low,
            "win_rate_ci_high": self.win_rate_ci_high,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
            "avg_game_length": self.avg_game_length,
            "games_played": self.games_played,
            "is_statistically_significant": self.is_statistically_significant,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "avg_duration_seconds": self.avg_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "model_is_better": self.model_is_better,
            "improvement_margin": self.improvement_margin,
            "strategy": self.strategy.value,
            "evaluation_id": self.evaluation_id,
        }


@dataclass
class MetricsHistory:
    """
    Historical metrics for trend detection and anomaly analysis.

    Maintains rolling windows of evaluation metrics for
    detecting performance trends and anomalies.
    """

    win_rates: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    draw_rates: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    game_lengths: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def add_result(self, result: EvaluationResult, timestamp: float) -> None:
        """Add evaluation result to history."""
        self.win_rates.append(result.win_rate)
        self.draw_rates.append(result.draw_rate)
        self.game_lengths.append(result.avg_game_length)
        self.timestamps.append(timestamp)

    def get_moving_average(self, window: int) -> dict[str, float]:
        """Calculate moving averages for metrics."""
        if len(self.win_rates) == 0:
            return {"win_rate_ma": 0.0, "draw_rate_ma": 0.0, "game_length_ma": 0.0}

        recent_win = list(self.win_rates)[-window:]
        recent_draw = list(self.draw_rates)[-window:]
        recent_length = list(self.game_lengths)[-window:]

        return {
            "win_rate_ma": sum(recent_win) / len(recent_win) if recent_win else 0.0,
            "draw_rate_ma": sum(recent_draw) / len(recent_draw) if recent_draw else 0.0,
            "game_length_ma": sum(recent_length) / len(recent_length) if recent_length else 0.0,
        }


class InitialStateFn(Protocol):
    """Protocol for initial state factory function."""

    def __call__(self) -> GameState:
        """Return a new initial game state."""
        ...


class EvaluationService:
    """
    Service for comprehensive model evaluation.

    Provides:
    - Real metrics computation from evaluation games
    - Confidence interval calculation
    - Statistical significance testing
    - Moving average tracking
    - Trend detection
    - Anomaly detection
    - Multiple evaluation strategies

    Example:
        >>> settings = get_settings()
        >>> service = EvaluationService(settings)
        >>> result = await service.evaluate_models(
        ...     candidate_model=new_model,
        ...     baseline_model=current_best,
        ...     num_games=100,
        ... )
        >>> if result.model_is_better:
        ...     print(f"New model is better with {result.win_rate:.2%} win rate")
    """

    def __init__(
        self,
        settings: Settings,
        mcts: NeuralMCTS | None = None,
        initial_state_fn: InitialStateFn | None = None,
        config: EvaluationConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize evaluation service.

        Args:
            settings: Application settings for configuration
            mcts: Neural MCTS engine for game simulation
            initial_state_fn: Factory function for creating initial game states
            config: Optional explicit configuration (overrides settings)
            logger: Optional logger instance (creates structured logger if None)
        """
        self._settings = settings
        self._mcts = mcts
        self._initial_state_fn = initial_state_fn
        self._config = config or EvaluationConfig.from_settings(settings)
        self._logger = logger or get_structured_logger(__name__)
        self._history = MetricsHistory()
        self._device = getattr(settings, "device", "cpu") if hasattr(settings, "device") else "cpu"

    def set_mcts(self, mcts: NeuralMCTS) -> None:
        """Set the MCTS engine for evaluation."""
        self._mcts = mcts

    def set_initial_state_fn(self, fn: InitialStateFn) -> None:
        """Set the initial state factory function."""
        self._initial_state_fn = fn

    async def evaluate_models(
        self,
        candidate_model: nn.Module,
        baseline_model: nn.Module | None = None,
        num_games: int | None = None,
        strategy: EvaluationStrategy = EvaluationStrategy.SELF_PLAY,
    ) -> EvaluationResult:
        """
        Evaluate candidate model against baseline.

        Plays multiple games with alternating starting positions
        to ensure fair evaluation. Computes comprehensive metrics
        including confidence intervals and statistical significance.

        Args:
            candidate_model: Model to evaluate
            baseline_model: Baseline for comparison (uses candidate if None)
            num_games: Number of games to play (uses config default if None)
            strategy: Evaluation strategy to use

        Returns:
            EvaluationResult with comprehensive metrics

        Raises:
            ValueError: If MCTS or initial_state_fn not configured
        """
        import time
        import uuid

        correlation_id = get_correlation_id()
        evaluation_id = str(uuid.uuid4())[:8]

        if self._mcts is None:
            raise ValueError("MCTS engine not configured. Call set_mcts() first.")
        if self._initial_state_fn is None:
            raise ValueError("Initial state function not configured. Call set_initial_state_fn() first.")

        num_games = num_games or self._config.num_games
        baseline_model = baseline_model or candidate_model

        self._logger.info(
            f"Starting model evaluation: {num_games} games, strategy={strategy.value}",
            extra={
                "correlation_id": correlation_id,
                "evaluation_id": evaluation_id,
                "num_games": num_games,
                "strategy": strategy.value,
            },
        )

        start_time = time.time()
        game_results: list[GameResult] = []

        # Play games with alternating starting positions
        for game_idx in range(num_games):
            model1_starts = game_idx % 2 == 0

            try:
                result = await self._play_single_game(
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                    model1_starts=model1_starts,
                    game_index=game_idx,
                )
                game_results.append(result)

                if (game_idx + 1) % 10 == 0:
                    self._logger.debug(
                        f"Completed {game_idx + 1}/{num_games} evaluation games",
                        extra={
                            "correlation_id": correlation_id,
                            "games_completed": game_idx + 1,
                        },
                    )

            except Exception as e:
                self._logger.warning(
                    f"Game {game_idx} failed: {e}",
                    extra={
                        "correlation_id": correlation_id,
                        "game_index": game_idx,
                        "error": str(e),
                    },
                )
                continue

        total_duration = time.time() - start_time

        # Compute metrics from game results
        evaluation_result = self._compute_metrics(
            game_results=game_results,
            evaluation_id=evaluation_id,
            strategy=strategy,
            total_duration=total_duration,
        )

        # Update history for trend tracking
        self._history.add_result(evaluation_result, time.time())

        self._logger.info(
            f"Model evaluation completed: win_rate={evaluation_result.win_rate:.2%}",
            extra={
                "correlation_id": correlation_id,
                "evaluation_id": evaluation_id,
                "win_rate": evaluation_result.win_rate,
                "is_significant": evaluation_result.is_statistically_significant,
                "model_is_better": evaluation_result.model_is_better,
                "duration_seconds": total_duration,
            },
        )

        return evaluation_result

    async def _play_single_game(
        self,
        candidate_model: nn.Module,
        baseline_model: nn.Module,
        model1_starts: bool,
        game_index: int,
    ) -> GameResult:
        """
        Play a single evaluation game between two models.

        Args:
            candidate_model: Candidate model (model1)
            baseline_model: Baseline model (model2)
            model1_starts: Whether candidate model plays first

        Returns:
            GameResult with outcome and statistics
        """
        import time

        # These assertions are for type checking - the caller validates these
        assert self._mcts is not None
        assert self._initial_state_fn is not None

        start_time = time.time()
        state = self._initial_state_fn()
        models = [candidate_model, baseline_model] if model1_starts else [baseline_model, candidate_model]
        current_model_idx = 0
        move_count = 0
        model_values: list[list[float]] = [[], []]

        # Store original model
        original_network = self._mcts.network

        try:
            while not state.is_terminal() and move_count < self._config.max_moves_per_game:
                current_model = models[current_model_idx]

                # Temporarily set model for MCTS
                self._mcts.network = current_model

                # Run MCTS search - returns (action_probs dict, root_node)
                action_probs, root_node = await self._mcts.search(
                    state,
                    num_simulations=self._config.mcts_iterations,
                    temperature=self._config.temperature,
                )

                # Extract value from root node (average value/Q-value)
                root_value = root_node.value if root_node.visit_count > 0 else 0.0
                model_values[current_model_idx].append(root_value)

                # Select action deterministically for evaluation
                if self._config.temperature == 0:
                    action = max(action_probs.keys(), key=lambda a: action_probs[a])
                else:
                    actions = list(action_probs.keys())
                    probs = list(action_probs.values())
                    action_idx = int(torch.multinomial(torch.tensor(probs), num_samples=1).item())
                    action = actions[action_idx]

                state = state.apply_action(action)
                move_count += 1
                current_model_idx = 1 - current_model_idx

        finally:
            # Restore original model
            self._mcts.network = original_network

        duration = time.time() - start_time

        # Determine outcome
        reward = state.get_reward(player=0 if model1_starts else 1)
        if reward > 0:
            outcome = GameOutcome.WIN
        elif reward < 0:
            outcome = GameOutcome.LOSS
        else:
            outcome = GameOutcome.DRAW

        # Calculate average values
        model1_values = model_values[0] if model1_starts else model_values[1]
        model2_values = model_values[1] if model1_starts else model_values[0]

        return GameResult(
            outcome=outcome,
            game_length=move_count,
            model1_avg_value=sum(model1_values) / len(model1_values) if model1_values else 0.0,
            model2_avg_value=sum(model2_values) / len(model2_values) if model2_values else 0.0,
            model1_started=model1_starts,
            duration_seconds=duration,
            metadata={"game_index": game_index},
        )

    def _compute_metrics(
        self,
        game_results: list[GameResult],
        evaluation_id: str,
        strategy: EvaluationStrategy,
        total_duration: float,
    ) -> EvaluationResult:
        """
        Compute comprehensive metrics from game results.

        Includes confidence interval calculation and statistical
        significance testing.
        """
        if not game_results:
            return EvaluationResult(
                win_rate=0.0,
                win_rate_ci_low=0.0,
                win_rate_ci_high=0.0,
                draw_rate=0.0,
                loss_rate=0.0,
                avg_game_length=0.0,
                games_played=0,
                is_statistically_significant=False,
                evaluation_id=evaluation_id,
                strategy=strategy,
            )

        # Count outcomes
        wins = sum(1 for r in game_results if r.outcome == GameOutcome.WIN)
        losses = sum(1 for r in game_results if r.outcome == GameOutcome.LOSS)
        draws = sum(1 for r in game_results if r.outcome == GameOutcome.DRAW)
        games_played = len(game_results)

        # Calculate rates
        win_rate = (wins + 0.5 * draws) / games_played
        draw_rate = draws / games_played
        loss_rate = losses / games_played

        # Calculate confidence interval using Wilson score interval
        ci_low, ci_high = self._wilson_score_interval(
            successes=wins + draws // 2,
            total=games_played,
            confidence=self._config.confidence_level,
        )

        # Calculate average game length
        avg_game_length = sum(r.game_length for r in game_results) / games_played

        # Calculate duration metrics
        avg_duration = sum(r.duration_seconds for r in game_results) / games_played

        # Statistical significance testing
        is_significant, p_value = self._test_significance(
            win_rate=win_rate,
            games_played=games_played,
            null_hypothesis=0.5,  # Test against 50% (random)
        )

        # Effect size (Cohen's h for proportions)
        effect_size = self._cohens_h(win_rate, 0.5)

        # Determine if model is better
        model_is_better = (
            win_rate >= self._config.win_threshold
            and is_significant
            and games_played >= self._config.min_games_for_significance
        )

        improvement_margin = win_rate - self._config.win_threshold

        return EvaluationResult(
            win_rate=win_rate,
            win_rate_ci_low=ci_low,
            win_rate_ci_high=ci_high,
            draw_rate=draw_rate,
            loss_rate=loss_rate,
            avg_game_length=avg_game_length,
            games_played=games_played,
            is_statistically_significant=is_significant,
            p_value=p_value,
            effect_size=effect_size,
            wins=wins,
            losses=losses,
            draws=draws,
            avg_duration_seconds=avg_duration,
            total_duration_seconds=total_duration,
            model_is_better=model_is_better,
            improvement_margin=improvement_margin,
            strategy=strategy,
            evaluation_id=evaluation_id,
        )

    def _wilson_score_interval(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Calculate Wilson score confidence interval for a proportion.

        More accurate than normal approximation for small samples
        and extreme proportions.

        Args:
            successes: Number of successes (wins)
            total: Total trials (games)
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if total == 0:
            return 0.0, 0.0

        # Z-score for confidence level
        z = self._get_z_score(confidence)
        p_hat = successes / total
        n = total

        # Wilson score interval formula
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return lower, upper

    def _get_z_score(self, confidence: float) -> float:
        """Get z-score for given confidence level."""
        # Common z-scores (avoid scipy dependency)
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        return z_scores.get(confidence, 1.96)

    def _test_significance(
        self,
        win_rate: float,
        games_played: int,
        null_hypothesis: float = 0.5,
    ) -> tuple[bool, float | None]:
        """
        Test statistical significance of win rate against null hypothesis.

        Uses one-sample z-test for proportions.

        Args:
            win_rate: Observed win rate
            games_played: Number of games played
            null_hypothesis: Expected win rate under null (default 0.5)

        Returns:
            Tuple of (is_significant, p_value)
        """
        if games_played < self._config.min_games_for_significance:
            return False, None

        # Standard error under null hypothesis
        se = math.sqrt(null_hypothesis * (1 - null_hypothesis) / games_played)
        if se == 0:
            return False, None

        # Z-statistic
        z = (win_rate - null_hypothesis) / se

        # Two-tailed p-value (approximation)
        p_value = 2 * (1 - self._standard_normal_cdf(abs(z)))

        # Significance at configured confidence level
        alpha = 1 - self._config.confidence_level
        is_significant = p_value < alpha

        return is_significant, p_value

    def _standard_normal_cdf(self, x: float) -> float:
        """
        Approximate standard normal CDF using error function approximation.

        Avoids scipy dependency while maintaining reasonable accuracy.
        """
        # Approximation using tanh (Bowling et al.)
        return 0.5 * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    def _cohens_h(self, p1: float, p2: float) -> float:
        """
        Calculate Cohen's h effect size for two proportions.

        Args:
            p1: First proportion
            p2: Second proportion

        Returns:
            Cohen's h value (small: 0.2, medium: 0.5, large: 0.8)
        """
        # Arcsine transformation
        phi1 = 2 * math.asin(math.sqrt(p1))
        phi2 = 2 * math.asin(math.sqrt(p2))
        return abs(phi1 - phi2)

    def get_moving_average(self, window: int | None = None) -> dict[str, float]:
        """
        Get moving average of evaluation metrics.

        Args:
            window: Window size (uses config default if None)

        Returns:
            Dictionary with moving averages for win_rate, draw_rate, game_length
        """
        window = window or self._config.moving_average_window
        return self._history.get_moving_average(window)

    def detect_trend(self, window: int | None = None) -> dict[str, Any]:
        """
        Detect trends in evaluation metrics.

        Uses simple linear regression on recent values to
        determine if metrics are improving, declining, or stable.

        Args:
            window: Window size for trend analysis

        Returns:
            Dictionary with trend information for each metric
        """
        window = window or self._config.moving_average_window
        win_rates = list(self._history.win_rates)[-window:]

        if len(win_rates) < 3:
            return {
                "trend": "insufficient_data",
                "slope": 0.0,
                "direction": "stable",
            }

        # Simple linear regression
        n = len(win_rates)
        x_mean = (n - 1) / 2
        y_mean = sum(win_rates) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(win_rates))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0.0

        # Determine trend direction
        if abs(slope) < self._config.trend_detection_threshold:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "declining"

        return {
            "trend": direction,
            "slope": slope,
            "direction": direction,
            "window_size": len(win_rates),
        }

    def detect_anomaly(self, result: EvaluationResult) -> dict[str, Any]:
        """
        Detect if an evaluation result is anomalous.

        Uses z-score based detection against historical metrics.

        Args:
            result: Evaluation result to check

        Returns:
            Dictionary with anomaly detection results
        """
        win_rates = list(self._history.win_rates)

        if len(win_rates) < self._config.min_samples_for_anomaly:
            return {
                "is_anomaly": False,
                "reason": "insufficient_history",
                "z_score": 0.0,
            }

        # Calculate z-score
        mean = sum(win_rates) / len(win_rates)
        variance = sum((x - mean) ** 2 for x in win_rates) / len(win_rates)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return {
                "is_anomaly": False,
                "reason": "zero_variance",
                "z_score": 0.0,
            }

        z_score = (result.win_rate - mean) / std
        is_anomaly = abs(z_score) > self._config.anomaly_std_threshold

        return {
            "is_anomaly": is_anomaly,
            "z_score": z_score,
            "threshold": self._config.anomaly_std_threshold,
            "historical_mean": mean,
            "historical_std": std,
        }

    async def benchmark_evaluation(
        self,
        model: nn.Module,
        benchmark_models: list[nn.Module],
        games_per_benchmark: int | None = None,
    ) -> dict[str, EvaluationResult]:
        """
        Evaluate model against multiple benchmark models.

        Args:
            model: Model to evaluate
            benchmark_models: List of benchmark models
            games_per_benchmark: Games to play against each benchmark

        Returns:
            Dictionary mapping benchmark index to evaluation result
        """
        games = games_per_benchmark or self._config.num_games // len(benchmark_models)
        results = {}

        for idx, benchmark in enumerate(benchmark_models):
            result = await self.evaluate_models(
                candidate_model=model,
                baseline_model=benchmark,
                num_games=games,
                strategy=EvaluationStrategy.BENCHMARK,
            )
            results[f"benchmark_{idx}"] = result

        return results

    async def ab_comparison(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        num_games: int | None = None,
    ) -> dict[str, Any]:
        """
        Perform A/B comparison between two models.

        Plays games in both directions and provides comprehensive
        comparison metrics.

        Args:
            model_a: First model
            model_b: Second model
            num_games: Total games to play (split between directions)

        Returns:
            Dictionary with comparison results
        """
        num_games = num_games or self._config.num_games
        games_each = num_games // 2

        # A vs B
        result_a = await self.evaluate_models(
            candidate_model=model_a,
            baseline_model=model_b,
            num_games=games_each,
            strategy=EvaluationStrategy.AB_COMPARISON,
        )

        # B vs A
        result_b = await self.evaluate_models(
            candidate_model=model_b,
            baseline_model=model_a,
            num_games=games_each,
            strategy=EvaluationStrategy.AB_COMPARISON,
        )

        # Determine winner
        combined_a_wins = result_a.wins + result_b.losses
        combined_b_wins = result_b.wins + result_a.losses
        combined_draws = result_a.draws + result_b.draws
        total_games = games_each * 2

        a_win_rate = (combined_a_wins + 0.5 * combined_draws) / total_games
        b_win_rate = (combined_b_wins + 0.5 * combined_draws) / total_games

        if a_win_rate > b_win_rate + self._config.trend_detection_threshold:
            winner = "model_a"
        elif b_win_rate > a_win_rate + self._config.trend_detection_threshold:
            winner = "model_b"
        else:
            winner = "tie"

        return {
            "model_a_result": result_a.to_dict(),
            "model_b_result": result_b.to_dict(),
            "combined_a_wins": combined_a_wins,
            "combined_b_wins": combined_b_wins,
            "combined_draws": combined_draws,
            "a_win_rate": a_win_rate,
            "b_win_rate": b_win_rate,
            "winner": winner,
            "total_games": total_games,
        }

    def reset_history(self) -> None:
        """Reset evaluation history."""
        self._history = MetricsHistory()
        self._logger.info(
            "Evaluation history reset",
            extra={"correlation_id": get_correlation_id()},
        )

    def get_history_summary(self) -> dict[str, Any]:
        """Get summary of evaluation history."""
        win_rates = list(self._history.win_rates)

        if not win_rates:
            return {
                "total_evaluations": 0,
                "avg_win_rate": 0.0,
                "min_win_rate": 0.0,
                "max_win_rate": 0.0,
                "std_win_rate": 0.0,
            }

        mean = sum(win_rates) / len(win_rates)
        variance = sum((x - mean) ** 2 for x in win_rates) / len(win_rates)

        return {
            "total_evaluations": len(win_rates),
            "avg_win_rate": mean,
            "min_win_rate": min(win_rates),
            "max_win_rate": max(win_rates),
            "std_win_rate": math.sqrt(variance),
            "recent_trend": self.detect_trend(),
        }


# Factory function for creating evaluation service
def create_evaluation_service(
    settings: Settings,
    mcts: NeuralMCTS | None = None,
    initial_state_fn: InitialStateFn | None = None,
    logger: logging.Logger | None = None,
) -> EvaluationService:
    """
    Factory function to create an EvaluationService instance.

    Args:
        settings: Application settings
        mcts: Optional MCTS engine
        initial_state_fn: Optional initial state factory
        logger: Optional logger

    Returns:
        Configured EvaluationService instance
    """
    return EvaluationService(
        settings=settings,
        mcts=mcts,
        initial_state_fn=initial_state_fn,
        logger=logger,
    )


__all__ = [
    "EvaluationService",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationStrategy",
    "GameOutcome",
    "GameResult",
    "MetricsHistory",
    "create_evaluation_service",
]
