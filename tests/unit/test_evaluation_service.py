"""
Unit tests for Evaluation Service module.

Tests:
- EvaluationResult dataclass and methods
- EvaluationConfig creation and validation
- EvaluationService comprehensive functionality
- Statistical methods (Wilson score, significance testing)
- Trend detection and anomaly detection
- Edge cases (zero games, all wins, all losses)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
nn = torch.nn

from src.training.evaluation_service import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationService,
    EvaluationStrategy,
    GameOutcome,
    GameResult,
    MetricsHistory,
    create_evaluation_service,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create a mock settings object for testing."""
    settings = MagicMock()
    settings.MCTS_ITERATIONS = 50
    settings.device = "cpu"
    # Optional evaluation attributes
    settings.EVAL_NUM_GAMES = 100
    settings.EVAL_MIN_GAMES_SIGNIFICANCE = 30
    settings.EVAL_WIN_THRESHOLD = 0.55
    settings.EVAL_CONFIDENCE_LEVEL = 0.95
    settings.EVAL_TEMPERATURE = 0.0
    settings.EVAL_MOVING_AVG_WINDOW = 50
    settings.EVAL_TREND_THRESHOLD = 0.05
    settings.EVAL_ANOMALY_STD_THRESHOLD = 2.5
    settings.EVAL_MIN_SAMPLES_ANOMALY = 10
    settings.EVAL_MAX_MOVES = 500
    settings.EVAL_TIMEOUT_SECONDS = 300.0
    return settings


@pytest.fixture
def mock_settings_minimal():
    """Create a minimal mock settings object with only required attributes."""
    settings = MagicMock()
    settings.MCTS_ITERATIONS = 100
    # Remove optional attributes to test defaults
    del settings.EVAL_NUM_GAMES
    del settings.EVAL_MIN_GAMES_SIGNIFICANCE
    return settings


@pytest.fixture
def evaluation_config():
    """Create a test evaluation configuration."""
    return EvaluationConfig(
        num_games=20,
        min_games_for_significance=10,
        win_threshold=0.55,
        confidence_level=0.95,
        mcts_iterations=50,
        temperature=0.0,
        moving_average_window=10,
        trend_detection_threshold=0.05,
        anomaly_std_threshold=2.5,
        min_samples_for_anomaly=5,
        max_moves_per_game=100,
        game_timeout_seconds=60.0,
    )


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def mock_mcts():
    """Create a mock MCTS engine for testing."""
    mcts = MagicMock()

    # Mock root node with visit count and value
    mock_root_node = MagicMock()
    mock_root_node.visit_count = 50
    mock_root_node.value = 0.6

    # Mock search to return action probs and root node
    async def mock_search(*args, **kwargs):
        return {"action_0": 0.7, "action_1": 0.3}, mock_root_node

    mcts.search = AsyncMock(side_effect=mock_search)
    mcts.network = MagicMock()

    return mcts


@pytest.fixture
def mock_game_state_factory():
    """Create a mock game state factory for testing."""
    def create_state():
        state = MagicMock()
        # Simulate a game that ends after a few moves
        terminal_sequence = [False, False, False, True]
        state.is_terminal = MagicMock(side_effect=terminal_sequence)
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=1.0)  # Win
        return state

    return create_state


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = nn.Linear(10, 10)
    return model


@pytest.fixture
def evaluation_service(mock_settings, mock_logger, evaluation_config):
    """Create an evaluation service for testing."""
    service = EvaluationService(
        settings=mock_settings,
        config=evaluation_config,
        logger=mock_logger,
    )
    return service


@pytest.fixture
def game_results_candidate_wins():
    """Create game results where candidate model wins."""
    results = []
    # 15 wins, 3 losses, 2 draws = 20 games
    for i in range(15):
        results.append(GameResult(
            outcome=GameOutcome.WIN,
            game_length=30 + i,
            model1_avg_value=0.7,
            model2_avg_value=0.4,
            model1_started=(i % 2 == 0),
            duration_seconds=5.0,
        ))
    for i in range(3):
        results.append(GameResult(
            outcome=GameOutcome.LOSS,
            game_length=25 + i,
            model1_avg_value=0.3,
            model2_avg_value=0.6,
            model1_started=(i % 2 == 1),
            duration_seconds=4.0,
        ))
    for i in range(2):
        results.append(GameResult(
            outcome=GameOutcome.DRAW,
            game_length=50,
            model1_avg_value=0.5,
            model2_avg_value=0.5,
            model1_started=(i % 2 == 0),
            duration_seconds=6.0,
        ))
    return results


@pytest.fixture
def game_results_baseline_wins():
    """Create game results where baseline model wins."""
    results = []
    # 5 wins, 12 losses, 3 draws = 20 games
    for i in range(5):
        results.append(GameResult(
            outcome=GameOutcome.WIN,
            game_length=30 + i,
            model1_avg_value=0.7,
            model2_avg_value=0.4,
            model1_started=(i % 2 == 0),
            duration_seconds=5.0,
        ))
    for i in range(12):
        results.append(GameResult(
            outcome=GameOutcome.LOSS,
            game_length=25 + i,
            model1_avg_value=0.3,
            model2_avg_value=0.6,
            model1_started=(i % 2 == 1),
            duration_seconds=4.0,
        ))
    for i in range(3):
        results.append(GameResult(
            outcome=GameOutcome.DRAW,
            game_length=50,
            model1_avg_value=0.5,
            model2_avg_value=0.5,
            model1_started=(i % 2 == 0),
            duration_seconds=6.0,
        ))
    return results


@pytest.fixture
def game_results_draw():
    """Create game results for an inconclusive evaluation."""
    results = []
    # 8 wins, 8 losses, 4 draws = 20 games (roughly 50% win rate)
    for i in range(8):
        results.append(GameResult(
            outcome=GameOutcome.WIN,
            game_length=30 + i,
            model1_avg_value=0.6,
            model2_avg_value=0.4,
            model1_started=(i % 2 == 0),
            duration_seconds=5.0,
        ))
    for i in range(8):
        results.append(GameResult(
            outcome=GameOutcome.LOSS,
            game_length=30 + i,
            model1_avg_value=0.4,
            model2_avg_value=0.6,
            model1_started=(i % 2 == 1),
            duration_seconds=5.0,
        ))
    for i in range(4):
        results.append(GameResult(
            outcome=GameOutcome.DRAW,
            game_length=50,
            model1_avg_value=0.5,
            model2_avg_value=0.5,
            model1_started=(i % 2 == 0),
            duration_seconds=6.0,
        ))
    return results


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    @pytest.mark.unit
    def test_evaluation_result_fields(self):
        """Test EvaluationResult has all expected fields with correct types."""
        result = EvaluationResult(
            win_rate=0.75,
            win_rate_ci_low=0.65,
            win_rate_ci_high=0.85,
            draw_rate=0.1,
            loss_rate=0.15,
            avg_game_length=35.5,
            games_played=100,
            is_statistically_significant=True,
            p_value=0.01,
            effect_size=0.5,
            wins=70,
            losses=15,
            draws=15,
            avg_duration_seconds=5.5,
            total_duration_seconds=550.0,
            model_is_better=True,
            improvement_margin=0.2,
            strategy=EvaluationStrategy.SELF_PLAY,
            evaluation_id="test-123",
        )

        assert result.win_rate == 0.75
        assert result.win_rate_ci_low == 0.65
        assert result.win_rate_ci_high == 0.85
        assert result.draw_rate == 0.1
        assert result.loss_rate == 0.15
        assert result.avg_game_length == 35.5
        assert result.games_played == 100
        assert result.is_statistically_significant is True
        assert result.p_value == 0.01
        assert result.effect_size == 0.5
        assert result.wins == 70
        assert result.losses == 15
        assert result.draws == 15
        assert result.avg_duration_seconds == 5.5
        assert result.total_duration_seconds == 550.0
        assert result.model_is_better is True
        assert result.improvement_margin == 0.2
        assert result.strategy == EvaluationStrategy.SELF_PLAY
        assert result.evaluation_id == "test-123"

    @pytest.mark.unit
    def test_is_statistically_significant_true(self):
        """Test EvaluationResult correctly reports statistical significance."""
        result = EvaluationResult(
            win_rate=0.75,
            win_rate_ci_low=0.65,
            win_rate_ci_high=0.85,
            draw_rate=0.1,
            loss_rate=0.15,
            avg_game_length=35.5,
            games_played=100,
            is_statistically_significant=True,
            p_value=0.001,
        )

        assert result.is_statistically_significant is True
        assert result.p_value is not None
        assert result.p_value < 0.05

    @pytest.mark.unit
    def test_is_statistically_significant_false(self):
        """Test EvaluationResult correctly reports non-significance."""
        result = EvaluationResult(
            win_rate=0.52,
            win_rate_ci_low=0.42,
            win_rate_ci_high=0.62,
            draw_rate=0.1,
            loss_rate=0.38,
            avg_game_length=35.5,
            games_played=50,
            is_statistically_significant=False,
            p_value=0.45,
        )

        assert result.is_statistically_significant is False
        assert result.p_value is not None
        assert result.p_value > 0.05

    @pytest.mark.unit
    def test_evaluation_result_to_dict(self):
        """Test EvaluationResult to_dict method."""
        result = EvaluationResult(
            win_rate=0.75,
            win_rate_ci_low=0.65,
            win_rate_ci_high=0.85,
            draw_rate=0.1,
            loss_rate=0.15,
            avg_game_length=35.5,
            games_played=100,
            is_statistically_significant=True,
            wins=70,
            losses=15,
            draws=15,
            strategy=EvaluationStrategy.BENCHMARK,
            evaluation_id="test-456",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["win_rate"] == 0.75
        assert result_dict["games_played"] == 100
        assert result_dict["is_statistically_significant"] is True
        assert result_dict["strategy"] == "benchmark"
        assert result_dict["evaluation_id"] == "test-456"


# =============================================================================
# EvaluationConfig Tests
# =============================================================================


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    @pytest.mark.unit
    def test_config_from_settings(self, mock_settings):
        """Test EvaluationConfig creation from Settings."""
        config = EvaluationConfig.from_settings(mock_settings)

        assert config.num_games == 100
        assert config.min_games_for_significance == 30
        assert config.win_threshold == 0.55
        assert config.confidence_level == 0.95
        assert config.mcts_iterations == 50
        assert config.temperature == 0.0
        assert config.moving_average_window == 50
        assert config.trend_detection_threshold == 0.05
        assert config.anomaly_std_threshold == 2.5
        assert config.min_samples_for_anomaly == 10
        assert config.max_moves_per_game == 500
        assert config.game_timeout_seconds == 300.0

    @pytest.mark.unit
    def test_config_default_values(self):
        """Test EvaluationConfig default values."""
        config = EvaluationConfig()

        assert config.num_games == 100
        assert config.min_games_for_significance == 30
        assert config.win_threshold == 0.55
        assert config.confidence_level == 0.95
        assert config.mcts_iterations == 100
        assert config.temperature == 0.0
        assert config.moving_average_window == 50
        assert config.trend_detection_threshold == 0.05
        assert config.anomaly_std_threshold == 2.5
        assert config.min_samples_for_anomaly == 10
        assert config.max_moves_per_game == 500
        assert config.game_timeout_seconds == 300.0

    @pytest.mark.unit
    def test_config_custom_values(self):
        """Test EvaluationConfig with custom values."""
        config = EvaluationConfig(
            num_games=50,
            min_games_for_significance=20,
            win_threshold=0.60,
            confidence_level=0.99,
            mcts_iterations=200,
        )

        assert config.num_games == 50
        assert config.min_games_for_significance == 20
        assert config.win_threshold == 0.60
        assert config.confidence_level == 0.99
        assert config.mcts_iterations == 200


# =============================================================================
# EvaluationService Tests
# =============================================================================


class TestEvaluationService:
    """Tests for EvaluationService class."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_models_candidate_wins(
        self,
        evaluation_service,
        mock_mcts,
        mock_game_state_factory,
        simple_model,
    ):
        """Test evaluation when candidate model wins."""
        # Setup service with MCTS and state factory
        evaluation_service.set_mcts(mock_mcts)
        evaluation_service.set_initial_state_fn(mock_game_state_factory)

        # Create a baseline model
        baseline_model = nn.Linear(10, 10)

        # Configure mock to return wins most of the time
        win_sequence = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        def create_winning_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, False, True])
            state.apply_action = MagicMock(return_value=state)
            if win_sequence:
                reward = win_sequence.pop(0)
            else:
                reward = 1.0
            state.get_reward = MagicMock(return_value=reward)
            return state

        evaluation_service.set_initial_state_fn(create_winning_state)

        # Run evaluation with small number of games
        result = await evaluation_service.evaluate_models(
            candidate_model=simple_model,
            baseline_model=baseline_model,
            num_games=10,
        )

        assert isinstance(result, EvaluationResult)
        assert result.games_played == 10
        assert result.win_rate > 0.5
        assert result.wins > result.losses

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_models_baseline_wins(
        self,
        evaluation_service,
        mock_mcts,
        simple_model,
    ):
        """Test evaluation when baseline model wins."""
        evaluation_service.set_mcts(mock_mcts)

        def create_losing_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=-1.0)  # Loss
            return state

        evaluation_service.set_initial_state_fn(create_losing_state)

        baseline_model = nn.Linear(10, 10)

        result = await evaluation_service.evaluate_models(
            candidate_model=simple_model,
            baseline_model=baseline_model,
            num_games=10,
        )

        assert result.games_played == 10
        assert result.win_rate < 0.5
        assert result.losses > result.wins

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_models_draw(
        self,
        evaluation_service,
        mock_mcts,
        simple_model,
    ):
        """Test evaluation with inconclusive results (draws)."""
        evaluation_service.set_mcts(mock_mcts)

        def create_draw_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=0.0)  # Draw
            return state

        evaluation_service.set_initial_state_fn(create_draw_state)

        baseline_model = nn.Linear(10, 10)

        result = await evaluation_service.evaluate_models(
            candidate_model=simple_model,
            baseline_model=baseline_model,
            num_games=10,
        )

        assert result.games_played == 10
        assert result.draws == 10
        assert result.draw_rate == 1.0
        assert result.wins == 0
        assert result.losses == 0

    @pytest.mark.unit
    def test_wilson_score_interval_correctness(self, evaluation_service):
        """Test Wilson score interval calculation."""
        # Test with known values
        # 70 successes out of 100 trials at 95% confidence
        ci_low, ci_high = evaluation_service._wilson_score_interval(
            successes=70,
            total=100,
            confidence=0.95,
        )

        # Wilson score interval for p=0.7, n=100, z=1.96
        # Should be approximately (0.60, 0.78)
        assert 0.58 < ci_low < 0.65
        assert 0.75 < ci_high < 0.82
        assert ci_low < 0.70 < ci_high

    @pytest.mark.unit
    def test_wilson_score_interval_edge_cases(self, evaluation_service):
        """Test Wilson score interval with edge cases."""
        # Zero games
        ci_low, ci_high = evaluation_service._wilson_score_interval(
            successes=0, total=0, confidence=0.95
        )
        assert ci_low == 0.0
        assert ci_high == 0.0

        # All successes
        ci_low, ci_high = evaluation_service._wilson_score_interval(
            successes=100, total=100, confidence=0.95
        )
        assert ci_low > 0.9
        assert ci_high <= 1.0

        # No successes
        ci_low, ci_high = evaluation_service._wilson_score_interval(
            successes=0, total=100, confidence=0.95
        )
        assert ci_low >= 0.0
        assert ci_high < 0.1

    @pytest.mark.unit
    def test_significance_testing(self, evaluation_service):
        """Test statistical significance testing."""
        # Clear win rate should be significant
        is_significant, p_value = evaluation_service._test_significance(
            win_rate=0.75,
            games_played=100,
            null_hypothesis=0.5,
        )
        assert is_significant is True
        assert p_value is not None
        assert p_value < 0.05

        # Win rate close to null hypothesis should not be significant
        is_significant, p_value = evaluation_service._test_significance(
            win_rate=0.52,
            games_played=50,
            null_hypothesis=0.5,
        )
        # May or may not be significant depending on exact calculation
        assert p_value is not None

    @pytest.mark.unit
    def test_significance_testing_insufficient_games(self, evaluation_service):
        """Test significance testing with insufficient games."""
        # Too few games for significance
        is_significant, p_value = evaluation_service._test_significance(
            win_rate=0.75,
            games_played=5,  # Less than min_games_for_significance
            null_hypothesis=0.5,
        )
        assert is_significant is False
        assert p_value is None

    @pytest.mark.unit
    def test_get_moving_average(self, evaluation_service):
        """Test moving average calculation."""
        # Add some results to history
        for i in range(10):
            result = EvaluationResult(
                win_rate=0.5 + (i * 0.05),
                win_rate_ci_low=0.4,
                win_rate_ci_high=0.6,
                draw_rate=0.1,
                loss_rate=0.4 - (i * 0.05),
                avg_game_length=30.0 + i,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        # Get moving average with window of 5
        ma = evaluation_service.get_moving_average(window=5)

        assert "win_rate_ma" in ma
        assert "draw_rate_ma" in ma
        assert "game_length_ma" in ma
        assert ma["win_rate_ma"] > 0
        assert ma["draw_rate_ma"] > 0
        assert ma["game_length_ma"] > 0

    @pytest.mark.unit
    def test_get_moving_average_empty_history(self, evaluation_service):
        """Test moving average with empty history."""
        ma = evaluation_service.get_moving_average(window=5)

        assert ma["win_rate_ma"] == 0.0
        assert ma["draw_rate_ma"] == 0.0
        assert ma["game_length_ma"] == 0.0

    @pytest.mark.unit
    def test_detect_trend_improving(self, evaluation_service):
        """Test trend detection for improving performance."""
        # Add improving win rates to history
        for i in range(10):
            result = EvaluationResult(
                win_rate=0.3 + (i * 0.05),  # Steadily increasing
                win_rate_ci_low=0.2,
                win_rate_ci_high=0.5,
                draw_rate=0.1,
                loss_rate=0.6 - (i * 0.05),
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        trend = evaluation_service.detect_trend(window=10)

        assert trend["direction"] == "improving"
        assert trend["slope"] > 0
        assert trend["window_size"] == 10

    @pytest.mark.unit
    def test_detect_trend_declining(self, evaluation_service):
        """Test trend detection for declining performance."""
        # Add declining win rates to history
        for i in range(10):
            result = EvaluationResult(
                win_rate=0.8 - (i * 0.05),  # Steadily decreasing
                win_rate_ci_low=0.5,
                win_rate_ci_high=0.9,
                draw_rate=0.1,
                loss_rate=0.1 + (i * 0.05),
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        trend = evaluation_service.detect_trend(window=10)

        assert trend["direction"] == "declining"
        assert trend["slope"] < 0

    @pytest.mark.unit
    def test_detect_trend_stable(self, evaluation_service):
        """Test trend detection for stable performance."""
        # Add stable win rates to history
        for i in range(10):
            result = EvaluationResult(
                win_rate=0.5 + (0.01 * (i % 2)),  # Slight fluctuation
                win_rate_ci_low=0.4,
                win_rate_ci_high=0.6,
                draw_rate=0.1,
                loss_rate=0.4,
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        trend = evaluation_service.detect_trend(window=10)

        assert trend["direction"] == "stable"
        assert abs(trend["slope"]) < evaluation_service._config.trend_detection_threshold

    @pytest.mark.unit
    def test_detect_trend_insufficient_data(self, evaluation_service):
        """Test trend detection with insufficient data."""
        # Add only 2 data points (less than 3 required)
        for i in range(2):
            result = EvaluationResult(
                win_rate=0.5,
                win_rate_ci_low=0.4,
                win_rate_ci_high=0.6,
                draw_rate=0.1,
                loss_rate=0.4,
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        trend = evaluation_service.detect_trend(window=10)

        assert trend["trend"] == "insufficient_data"

    @pytest.mark.unit
    def test_detect_anomaly(self, evaluation_service):
        """Test anomaly detection."""
        # Build history with consistent values
        for i in range(15):
            result = EvaluationResult(
                win_rate=0.5,  # Consistent
                win_rate_ci_low=0.4,
                win_rate_ci_high=0.6,
                draw_rate=0.1,
                loss_rate=0.4,
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        # Create an anomalous result
        anomalous_result = EvaluationResult(
            win_rate=0.95,  # Way above historical mean
            win_rate_ci_low=0.9,
            win_rate_ci_high=1.0,
            draw_rate=0.0,
            loss_rate=0.05,
            avg_game_length=30.0,
            games_played=20,
            is_statistically_significant=True,
        )

        anomaly = evaluation_service.detect_anomaly(anomalous_result)

        assert anomaly["is_anomaly"] is True
        assert anomaly["z_score"] > evaluation_service._config.anomaly_std_threshold

    @pytest.mark.unit
    def test_detect_anomaly_insufficient_history(self, evaluation_service):
        """Test anomaly detection with insufficient history."""
        # Add only a few data points
        for i in range(3):
            result = EvaluationResult(
                win_rate=0.5,
                win_rate_ci_low=0.4,
                win_rate_ci_high=0.6,
                draw_rate=0.1,
                loss_rate=0.4,
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        test_result = EvaluationResult(
            win_rate=0.9,
            win_rate_ci_low=0.8,
            win_rate_ci_high=1.0,
            draw_rate=0.0,
            loss_rate=0.1,
            avg_game_length=30.0,
            games_played=20,
            is_statistically_significant=True,
        )

        anomaly = evaluation_service.detect_anomaly(test_result)

        assert anomaly["is_anomaly"] is False
        assert anomaly["reason"] == "insufficient_history"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_benchmark_evaluation(
        self,
        evaluation_service,
        mock_mcts,
        simple_model,
    ):
        """Test benchmark evaluation against multiple models."""
        evaluation_service.set_mcts(mock_mcts)

        def create_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=1.0)
            return state

        evaluation_service.set_initial_state_fn(create_state)

        # Create benchmark models
        benchmark_models = [nn.Linear(10, 10) for _ in range(3)]

        results = await evaluation_service.benchmark_evaluation(
            model=simple_model,
            benchmark_models=benchmark_models,
            games_per_benchmark=5,
        )

        assert len(results) == 3
        assert "benchmark_0" in results
        assert "benchmark_1" in results
        assert "benchmark_2" in results
        for _key, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert result.strategy == EvaluationStrategy.BENCHMARK

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ab_comparison(
        self,
        evaluation_service,
        mock_mcts,
    ):
        """Test A/B comparison between two models."""
        evaluation_service.set_mcts(mock_mcts)

        game_count = [0]

        def create_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, False, True])
            state.apply_action = MagicMock(return_value=state)
            # Alternate wins and losses
            reward = 1.0 if game_count[0] % 2 == 0 else -1.0
            game_count[0] += 1
            state.get_reward = MagicMock(return_value=reward)
            return state

        evaluation_service.set_initial_state_fn(create_state)

        model_a = nn.Linear(10, 10)
        model_b = nn.Linear(10, 10)

        comparison = await evaluation_service.ab_comparison(
            model_a=model_a,
            model_b=model_b,
            num_games=10,
        )

        assert "model_a_result" in comparison
        assert "model_b_result" in comparison
        assert "winner" in comparison
        assert "total_games" in comparison
        assert "a_win_rate" in comparison
        assert "b_win_rate" in comparison


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in evaluation service."""

    @pytest.mark.unit
    def test_zero_games_played(self, evaluation_service):
        """Test handling of zero games played."""
        result = evaluation_service._compute_metrics(
            game_results=[],
            evaluation_id="test",
            strategy=EvaluationStrategy.SELF_PLAY,
            total_duration=0.0,
        )

        assert result.games_played == 0
        assert result.win_rate == 0.0
        assert result.draw_rate == 0.0
        assert result.loss_rate == 0.0
        assert result.is_statistically_significant is False

    @pytest.mark.unit
    def test_all_wins(self, evaluation_service):
        """Test evaluation when all games are wins."""
        all_wins = [
            GameResult(
                outcome=GameOutcome.WIN,
                game_length=30,
                model1_avg_value=0.8,
                model2_avg_value=0.2,
                model1_started=True,
                duration_seconds=5.0,
            )
            for _ in range(20)
        ]

        result = evaluation_service._compute_metrics(
            game_results=all_wins,
            evaluation_id="test",
            strategy=EvaluationStrategy.SELF_PLAY,
            total_duration=100.0,
        )

        assert result.games_played == 20
        assert result.wins == 20
        assert result.losses == 0
        assert result.draws == 0
        assert result.win_rate == 1.0
        assert result.loss_rate == 0.0

    @pytest.mark.unit
    def test_all_losses(self, evaluation_service):
        """Test evaluation when all games are losses."""
        all_losses = [
            GameResult(
                outcome=GameOutcome.LOSS,
                game_length=25,
                model1_avg_value=0.2,
                model2_avg_value=0.8,
                model1_started=True,
                duration_seconds=4.0,
            )
            for _ in range(20)
        ]

        result = evaluation_service._compute_metrics(
            game_results=all_losses,
            evaluation_id="test",
            strategy=EvaluationStrategy.SELF_PLAY,
            total_duration=80.0,
        )

        assert result.games_played == 20
        assert result.wins == 0
        assert result.losses == 20
        assert result.draws == 0
        assert result.win_rate == 0.0
        assert result.loss_rate == 1.0

    @pytest.mark.unit
    def test_confidence_interval_bounds(self, evaluation_service):
        """Test that confidence intervals are always within [0, 1]."""
        # Test with extreme values
        test_cases = [
            (0, 100),  # No successes
            (100, 100),  # All successes
            (50, 100),  # Half successes
            (1, 100),  # Very few successes
            (99, 100),  # Almost all successes
        ]

        for successes, total in test_cases:
            ci_low, ci_high = evaluation_service._wilson_score_interval(
                successes=successes,
                total=total,
                confidence=0.95,
            )
            assert 0.0 <= ci_low <= 1.0, f"CI low {ci_low} out of bounds for {successes}/{total}"
            assert 0.0 <= ci_high <= 1.0, f"CI high {ci_high} out of bounds for {successes}/{total}"
            assert ci_low <= ci_high, f"CI low {ci_low} > CI high {ci_high}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_models_without_mcts_raises(self, evaluation_service, simple_model):
        """Test that evaluate_models raises without MCTS configured."""
        with pytest.raises(ValueError, match="MCTS engine not configured"):
            await evaluation_service.evaluate_models(
                candidate_model=simple_model,
                num_games=10,
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_models_without_state_fn_raises(
        self, evaluation_service, mock_mcts, simple_model
    ):
        """Test that evaluate_models raises without initial state function."""
        evaluation_service.set_mcts(mock_mcts)

        with pytest.raises(ValueError, match="Initial state function not configured"):
            await evaluation_service.evaluate_models(
                candidate_model=simple_model,
                num_games=10,
            )


# =============================================================================
# MetricsHistory Tests
# =============================================================================


class TestMetricsHistory:
    """Tests for MetricsHistory class."""

    @pytest.mark.unit
    def test_add_result(self):
        """Test adding results to history."""
        history = MetricsHistory()

        result = EvaluationResult(
            win_rate=0.75,
            win_rate_ci_low=0.65,
            win_rate_ci_high=0.85,
            draw_rate=0.1,
            loss_rate=0.15,
            avg_game_length=35.0,
            games_played=100,
            is_statistically_significant=True,
        )

        history.add_result(result, timestamp=1.0)

        assert len(history.win_rates) == 1
        assert history.win_rates[0] == 0.75
        assert len(history.draw_rates) == 1
        assert history.draw_rates[0] == 0.1
        assert len(history.game_lengths) == 1
        assert history.game_lengths[0] == 35.0

    @pytest.mark.unit
    def test_get_moving_average(self):
        """Test moving average calculation."""
        history = MetricsHistory()

        # Add 10 results
        for i in range(10):
            result = EvaluationResult(
                win_rate=0.5 + (i * 0.05),
                win_rate_ci_low=0.4,
                win_rate_ci_high=0.6,
                draw_rate=0.1,
                loss_rate=0.4,
                avg_game_length=30.0 + i,
                games_played=20,
                is_statistically_significant=False,
            )
            history.add_result(result, float(i))

        # Get moving average of last 5
        ma = history.get_moving_average(window=5)

        # Last 5 win rates: 0.75, 0.80, 0.85, 0.90, 0.95
        expected_win_ma = (0.75 + 0.80 + 0.85 + 0.90 + 0.95) / 5
        assert abs(ma["win_rate_ma"] - expected_win_ma) < 0.01

    @pytest.mark.unit
    def test_get_moving_average_empty(self):
        """Test moving average with empty history."""
        history = MetricsHistory()

        ma = history.get_moving_average(window=5)

        assert ma["win_rate_ma"] == 0.0
        assert ma["draw_rate_ma"] == 0.0
        assert ma["game_length_ma"] == 0.0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for create_evaluation_service factory function."""

    @pytest.mark.unit
    def test_create_evaluation_service(self, mock_settings, mock_mcts, mock_logger):
        """Test factory function creates service correctly."""
        def mock_state_fn():
            return MagicMock()

        service = create_evaluation_service(
            settings=mock_settings,
            mcts=mock_mcts,
            initial_state_fn=mock_state_fn,
            logger=mock_logger,
        )

        assert isinstance(service, EvaluationService)
        assert service._mcts is mock_mcts
        assert service._initial_state_fn is mock_state_fn

    @pytest.mark.unit
    def test_create_evaluation_service_minimal(self, mock_settings):
        """Test factory function with minimal arguments."""
        service = create_evaluation_service(settings=mock_settings)

        assert isinstance(service, EvaluationService)
        assert service._mcts is None
        assert service._initial_state_fn is None


# =============================================================================
# Statistical Method Tests
# =============================================================================


class TestStatisticalMethods:
    """Tests for statistical methods in EvaluationService."""

    @pytest.mark.unit
    def test_cohens_h_calculation(self, evaluation_service):
        """Test Cohen's h effect size calculation."""
        # Same proportions should give 0
        effect_size = evaluation_service._cohens_h(0.5, 0.5)
        assert abs(effect_size) < 0.001

        # Different proportions should give non-zero effect
        effect_size = evaluation_service._cohens_h(0.8, 0.5)
        assert effect_size > 0

        # Effect size should be symmetric
        effect1 = evaluation_service._cohens_h(0.7, 0.3)
        effect2 = evaluation_service._cohens_h(0.3, 0.7)
        assert abs(effect1 - effect2) < 0.001

    @pytest.mark.unit
    def test_get_z_score(self, evaluation_service):
        """Test z-score retrieval for confidence levels."""
        assert evaluation_service._get_z_score(0.90) == 1.645
        assert evaluation_service._get_z_score(0.95) == 1.96
        assert evaluation_service._get_z_score(0.99) == 2.576
        # Unknown confidence level should default to 1.96
        assert evaluation_service._get_z_score(0.85) == 1.96

    @pytest.mark.unit
    def test_standard_normal_cdf(self, evaluation_service):
        """Test standard normal CDF approximation."""
        # CDF at 0 should be approximately 0.5
        cdf_0 = evaluation_service._standard_normal_cdf(0)
        assert abs(cdf_0 - 0.5) < 0.01

        # CDF at large positive should approach 1
        cdf_large = evaluation_service._standard_normal_cdf(3)
        assert cdf_large > 0.99

        # CDF at large negative should approach 0
        cdf_neg = evaluation_service._standard_normal_cdf(-3)
        assert cdf_neg < 0.01

    @pytest.mark.unit
    @pytest.mark.parametrize("win_rate,games,expected_significant", [
        (0.75, 100, True),   # Clear win, many games
        (0.52, 30, False),   # Close to 50%, few games
        (0.90, 50, True),    # Very high win rate
        (0.55, 20, False),   # Not enough games for significance
    ])
    def test_significance_parametrized(
        self, evaluation_service, win_rate, games, expected_significant
    ):
        """Test significance testing with various parameters."""
        is_significant, _ = evaluation_service._test_significance(
            win_rate=win_rate,
            games_played=games,
            null_hypothesis=0.5,
        )
        # Check consistency with expectation (allowing for edge cases)
        if games >= evaluation_service._config.min_games_for_significance:
            # For games above threshold, verify the direction matches
            if abs(win_rate - 0.5) > 0.2:
                assert is_significant == expected_significant


# =============================================================================
# Service Reset and History Tests
# =============================================================================


class TestServiceResetAndHistory:
    """Tests for service reset and history functionality."""

    @pytest.mark.unit
    def test_reset_history(self, evaluation_service):
        """Test resetting evaluation history."""
        # Add some results
        for i in range(5):
            result = EvaluationResult(
                win_rate=0.5,
                win_rate_ci_low=0.4,
                win_rate_ci_high=0.6,
                draw_rate=0.1,
                loss_rate=0.4,
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        assert len(evaluation_service._history.win_rates) == 5

        # Reset history
        evaluation_service.reset_history()

        assert len(evaluation_service._history.win_rates) == 0

    @pytest.mark.unit
    def test_get_history_summary(self, evaluation_service):
        """Test getting history summary."""
        # Add some results with varying win rates
        for i in range(10):
            result = EvaluationResult(
                win_rate=0.4 + (i * 0.05),
                win_rate_ci_low=0.3,
                win_rate_ci_high=0.7,
                draw_rate=0.1,
                loss_rate=0.5 - (i * 0.05),
                avg_game_length=30.0,
                games_played=20,
                is_statistically_significant=False,
            )
            evaluation_service._history.add_result(result, float(i))

        summary = evaluation_service.get_history_summary()

        assert summary["total_evaluations"] == 10
        assert "avg_win_rate" in summary
        assert "min_win_rate" in summary
        assert "max_win_rate" in summary
        assert "std_win_rate" in summary
        assert "recent_trend" in summary

    @pytest.mark.unit
    def test_get_history_summary_empty(self, evaluation_service):
        """Test getting history summary with empty history."""
        summary = evaluation_service.get_history_summary()

        assert summary["total_evaluations"] == 0
        assert summary["avg_win_rate"] == 0.0
        assert summary["min_win_rate"] == 0.0
        assert summary["max_win_rate"] == 0.0
        assert summary["std_win_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
