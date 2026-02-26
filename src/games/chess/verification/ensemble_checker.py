"""
Ensemble Consistency Checker.

Provides verification for ensemble agent consistency,
checking agreement between HRM, TRM, and MCTS agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.games.chess.ensemble_agent import ChessEnsembleAgent

from src.games.chess.config import AgentType, GamePhase
from src.games.chess.constants import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE,
    DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD,
    DEFAULT_ROUTING_THRESHOLD,
    DEFAULT_VALUE_DIVERGENCE_THRESHOLD,
    STARTING_FEN,
    get_routing_scores,
    truncate_fen,
)
from src.games.chess.state import ChessGameState
from src.games.chess.verification.types import (
    EnsembleConsistencyResult,
    VerificationIssue,
    VerificationSeverity,
)
from src.observability.logging import StructuredLogger, get_structured_logger


@dataclass
class EnsembleCheckerConfig:
    """Configuration for ensemble consistency checking.

    All parameters are configurable via settings - no hardcoded values.
    Defaults are loaded from settings if available.
    """

    # Thresholds (defaults loaded from settings)
    agreement_threshold: float | None = None
    confidence_divergence_threshold: float | None = None
    value_divergence_threshold: float | None = None
    routing_threshold: float | None = None
    default_confidence: float | None = None

    # Routing expectations by phase
    opening_expected_agent: AgentType = AgentType.HRM
    middlegame_expected_agent: AgentType = AgentType.MCTS
    endgame_expected_agent: AgentType = AgentType.TRM

    # Analysis options
    compute_move_variance: bool = True
    compute_divergences: bool = True

    # Logging
    log_checks: bool = True

    def __post_init__(self) -> None:
        """Load defaults from settings if not explicitly provided."""
        try:
            from src.config.settings import get_settings

            settings = get_settings()

            if self.agreement_threshold is None:
                self.agreement_threshold = getattr(
                    settings,
                    "CHESS_VERIFICATION_AGREEMENT_THRESHOLD",
                    DEFAULT_AGREEMENT_THRESHOLD,
                )
            if self.confidence_divergence_threshold is None:
                self.confidence_divergence_threshold = getattr(
                    settings,
                    "CHESS_VERIFICATION_DIVERGENCE_THRESHOLD",
                    DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD,
                )
            if self.routing_threshold is None:
                self.routing_threshold = getattr(
                    settings,
                    "CHESS_VERIFICATION_ROUTING_THRESHOLD",
                    DEFAULT_ROUTING_THRESHOLD,
                )
            if self.value_divergence_threshold is None:
                self.value_divergence_threshold = getattr(
                    settings,
                    "CHESS_VERIFICATION_VALUE_DIVERGENCE_THRESHOLD",
                    DEFAULT_VALUE_DIVERGENCE_THRESHOLD,
                )
            if self.default_confidence is None:
                self.default_confidence = getattr(
                    settings,
                    "CHESS_VERIFICATION_DEFAULT_CONFIDENCE",
                    DEFAULT_CONFIDENCE,
                )
        except (ImportError, RuntimeError, OSError):
            # Fallback to defaults if settings unavailable
            # ImportError: settings module not found
            # RuntimeError: settings validation failed
            # OSError: .env file issues
            if self.agreement_threshold is None:
                self.agreement_threshold = DEFAULT_AGREEMENT_THRESHOLD
            if self.confidence_divergence_threshold is None:
                self.confidence_divergence_threshold = DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD
            if self.routing_threshold is None:
                self.routing_threshold = DEFAULT_ROUTING_THRESHOLD
            if self.value_divergence_threshold is None:
                self.value_divergence_threshold = DEFAULT_VALUE_DIVERGENCE_THRESHOLD
            if self.default_confidence is None:
                self.default_confidence = DEFAULT_CONFIDENCE


class EnsembleConsistencyChecker:
    """Checks consistency between ensemble agents.

    Verifies that HRM, TRM, and MCTS agents produce consistent
    results and that routing decisions are appropriate.

    Example:
        >>> checker = EnsembleConsistencyChecker(ensemble_agent)
        >>> state = ChessGameState.initial()
        >>> result = await checker.check_position_consistency(state)
        >>> print(result.agreement_rate)
    """

    def __init__(
        self,
        ensemble_agent: ChessEnsembleAgent | None = None,
        config: EnsembleCheckerConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize the consistency checker.

        Args:
            ensemble_agent: Optional ensemble agent (can be set later)
            config: Checker configuration
            logger: Optional logger instance
        """
        self._config = config or EnsembleCheckerConfig()
        self._ensemble_agent = ensemble_agent
        self._logger = logger or get_structured_logger("chess.verification.ensemble_checker")

    @property
    def config(self) -> EnsembleCheckerConfig:
        """Get the checker configuration."""
        return self._config

    @property
    def ensemble_agent(self) -> ChessEnsembleAgent | None:
        """Get the ensemble agent."""
        return self._ensemble_agent

    @ensemble_agent.setter
    def ensemble_agent(self, agent: ChessEnsembleAgent) -> None:
        """Set the ensemble agent."""
        self._ensemble_agent = agent

    def get_divergence_threshold(self) -> float:
        """Get the threshold for acceptable divergence.

        Returns:
            Float threshold value
        """
        threshold = self._config.confidence_divergence_threshold
        if threshold is None:
            return 0.3  # Default threshold
        return threshold

    async def check_position_consistency(
        self,
        state: ChessGameState,
    ) -> EnsembleConsistencyResult:
        """Check agent consistency for a single position.

        Args:
            state: Chess position to check

        Returns:
            EnsembleConsistencyResult with consistency analysis
        """
        start_time = time.perf_counter()
        issues: list[VerificationIssue] = []

        if self._ensemble_agent is None:
            issues.append(
                VerificationIssue(
                    code="NO_ENSEMBLE_AGENT",
                    message="Ensemble agent not configured",
                    severity=VerificationSeverity.ERROR,
                )
            )
            return EnsembleConsistencyResult(
                is_consistent=False,
                state_fen=state.fen,
                issues=issues,
            )

        # Get ensemble response with all agents
        try:
            ensemble_response = await self._ensemble_agent.get_best_move(
                state,
                temperature=0.0,
                use_ensemble=True,
            )
        except (ValueError, RuntimeError, AttributeError) as e:
            # Handle known error types specifically
            issues.append(
                VerificationIssue(
                    code="ENSEMBLE_EXECUTION_FAILED",
                    message=f"Failed to get ensemble response: {e}",
                    severity=VerificationSeverity.ERROR,
                    context={"error_type": type(e).__name__},
                )
            )
            return EnsembleConsistencyResult(
                is_consistent=False,
                state_fen=state.fen,
                issues=issues,
            )
        except Exception:
            # Log unexpected errors for debugging and re-raise
            # exception() automatically captures error details and traceback
            self._logger.exception(
                "Unexpected error in ensemble execution",
                fen=truncate_fen(state.fen),
            )
            raise

        # Extract agent responses
        agent_moves: dict[str, str] = {}
        agent_confidences: dict[str, float] = {}
        agent_values: dict[str, float] = {}

        for agent_name, response in ensemble_response.agent_responses.items():
            agent_moves[agent_name] = response.move
            agent_confidences[agent_name] = response.confidence
            agent_values[agent_name] = response.value_estimate

        # Calculate agreement rate
        agreement_rate = self._calculate_agreement_rate(agent_moves)

        # Calculate move variance
        move_variance: dict[str, float] = {}
        if self._config.compute_move_variance and len(agent_moves) > 1:
            move_variance = self._calculate_move_variance(
                agent_moves,
                ensemble_response.move_probabilities,
            )

        # Calculate divergences
        agent_divergences: dict[str, float] = {}
        if self._config.compute_divergences:
            agent_divergences = self._calculate_divergences(
                agent_moves,
                agent_confidences,
                ensemble_response.best_move,
            )

        # Check routing consistency
        routing_consistency = self._check_routing_consistency(
            state,
            ensemble_response.routing_decision.primary_agent,
        )

        # Add issues for low agreement
        agreement_threshold = self._config.agreement_threshold or DEFAULT_AGREEMENT_THRESHOLD
        if agreement_rate < agreement_threshold:
            issues.append(
                VerificationIssue(
                    code="LOW_AGREEMENT",
                    message=(
                        f"Agent agreement rate {agreement_rate:.2f} below threshold {self._config.agreement_threshold}"
                    ),
                    severity=VerificationSeverity.WARNING,
                    context={
                        "agreement_rate": agreement_rate,
                        "agent_moves": agent_moves,
                    },
                )
            )

        # Check for high divergences
        divergence_threshold = self._config.confidence_divergence_threshold or DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD
        for agent_name, divergence in agent_divergences.items():
            if divergence > divergence_threshold:
                issues.append(
                    VerificationIssue(
                        code="HIGH_DIVERGENCE",
                        message=(f"Agent {agent_name} divergence {divergence:.2f} exceeds threshold"),
                        severity=VerificationSeverity.WARNING,
                        context={
                            "agent": agent_name,
                            "divergence": divergence,
                        },
                    )
                )

        # Check routing appropriateness using configurable threshold
        routing_threshold = self._config.routing_threshold or DEFAULT_ROUTING_THRESHOLD
        if routing_consistency < routing_threshold:
            issues.append(
                VerificationIssue(
                    code="INAPPROPRIATE_ROUTING",
                    message=("Routing decision may not be optimal for game phase"),
                    severity=VerificationSeverity.WARNING,
                    context={
                        "primary_agent": ensemble_response.routing_decision.primary_agent.value,
                        "game_phase": state.get_game_phase().value,
                    },
                )
            )

        check_time_ms = (time.perf_counter() - start_time) * 1000

        # Determine overall consistency
        is_consistent = agreement_rate >= agreement_threshold and not any(
            i.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for i in issues
        )

        # Log if enabled
        if self._config.log_checks:
            self._logger.debug(
                "Position consistency checked",
                fen=truncate_fen(state.fen),
                is_consistent=is_consistent,
                agreement_rate=round(agreement_rate, 3),
                primary_agent=ensemble_response.routing_decision.primary_agent.value,
                duration_ms=round(check_time_ms, 2),
            )

        return EnsembleConsistencyResult(
            is_consistent=is_consistent,
            state_fen=state.fen,
            issues=issues,
            agreement_rate=agreement_rate,
            move_variance=move_variance,
            agent_divergences=agent_divergences,
            routing_consistency=routing_consistency,
            agent_moves=agent_moves,
            agent_confidences=agent_confidences,
            agent_values=agent_values,
            ensemble_move=ensemble_response.best_move,
            ensemble_confidence=ensemble_response.confidence,
            primary_agent=ensemble_response.routing_decision.primary_agent.value,
            check_time_ms=check_time_ms,
        )

    async def check_sequence_consistency(
        self,
        states: list[ChessGameState],
    ) -> list[EnsembleConsistencyResult]:
        """Check agent consistency across a sequence of positions.

        Args:
            states: List of chess positions

        Returns:
            List of EnsembleConsistencyResult for each position
        """
        results = []
        for state in states:
            result = await self.check_position_consistency(state)
            results.append(result)
        return results

    async def check_game_consistency(
        self,
        moves: list[str],
        initial_fen: str | None = None,
    ) -> list[EnsembleConsistencyResult]:
        """Check agent consistency throughout a game.

        Args:
            moves: Game moves in UCI format
            initial_fen: Optional starting position

        Returns:
            List of EnsembleConsistencyResult for each position
        """
        # Create states from moves using centralized starting position
        initial_fen = initial_fen or STARTING_FEN
        states: list[ChessGameState] = []

        current_state = ChessGameState.from_fen(initial_fen)
        states.append(current_state)

        for move in moves:
            try:
                current_state = current_state.apply_action(move)
                states.append(current_state)
            except ValueError:
                # Invalid move, stop here
                break

        return await self.check_sequence_consistency(states)

    def _calculate_agreement_rate(
        self,
        agent_moves: dict[str, str],
    ) -> float:
        """Calculate the agreement rate between agents.

        Args:
            agent_moves: Dictionary of agent names to moves

        Returns:
            Agreement rate between 0 and 1
        """
        if not agent_moves:
            return 0.0

        moves = list(agent_moves.values())
        if len(moves) < 2:
            return 1.0

        # Count agreements (pairs that agree)
        total_pairs = 0
        agreeing_pairs = 0

        move_list = list(moves)
        for i in range(len(move_list)):
            for j in range(i + 1, len(move_list)):
                total_pairs += 1
                if move_list[i] == move_list[j]:
                    agreeing_pairs += 1

        return agreeing_pairs / total_pairs if total_pairs > 0 else 1.0

    def _calculate_move_variance(
        self,
        agent_moves: dict[str, str],
        move_probabilities: dict[str, float],
    ) -> dict[str, float]:
        """Calculate move probability variance across agents.

        Args:
            agent_moves: Dictionary of agent names to moves
            move_probabilities: Ensemble move probabilities

        Returns:
            Dictionary of move to variance
        """
        variance: dict[str, float] = {}

        # Get unique moves selected by agents
        selected_moves = set(agent_moves.values())

        for move in selected_moves:
            prob = move_probabilities.get(move, 0.0)
            # Variance from uniform selection
            expected = 1.0 / len(move_probabilities) if move_probabilities else 0.0
            variance[move] = abs(prob - expected)

        return variance

    def _calculate_divergences(
        self,
        agent_moves: dict[str, str],
        agent_confidences: dict[str, float],
        ensemble_move: str,
    ) -> dict[str, float]:
        """Calculate divergence of each agent from ensemble decision.

        Args:
            agent_moves: Dictionary of agent names to moves
            agent_confidences: Dictionary of agent names to confidences
            ensemble_move: The ensemble's selected move

        Returns:
            Dictionary of agent names to divergence scores
        """
        divergences: dict[str, float] = {}

        for agent_name, agent_move in agent_moves.items():
            # Move agreement component
            move_matches = 1.0 if agent_move == ensemble_move else 0.0

            # Confidence component (use config default if not provided)
            default_conf = self._config.default_confidence or DEFAULT_CONFIDENCE
            confidence = agent_confidences.get(agent_name, default_conf)

            # Divergence: high when agent is confident but wrong
            if move_matches:
                divergence = 0.0
            else:
                divergence = confidence  # More confident + wrong = higher divergence

            divergences[agent_name] = divergence

        return divergences

    def _check_routing_consistency(
        self,
        state: ChessGameState,
        primary_agent: AgentType,
    ) -> float:
        """Check if routing decision is appropriate for position.

        Uses configurable routing scores from settings.

        Args:
            state: Chess position
            primary_agent: Selected primary agent

        Returns:
            Consistency score between 0 and 1
        """
        game_phase = state.get_game_phase()

        # Get routing scores from settings
        scores = get_routing_scores()

        # Get expected agent for phase
        expected_agent = self._get_expected_agent_for_phase(game_phase)

        # Score based on match
        if primary_agent == expected_agent:
            return scores["match"]

        # Partial credit for reasonable alternatives
        if game_phase == GamePhase.MIDDLEGAME:
            # Any agent is reasonable in middlegame
            return scores["middlegame_fallback"]
        elif game_phase == GamePhase.OPENING:
            # HRM or MCTS are reasonable
            if primary_agent in (AgentType.HRM, AgentType.MCTS):
                return scores["phase_appropriate"]
            return scores["phase_mismatch"]
        elif game_phase == GamePhase.ENDGAME:
            # TRM or MCTS are reasonable
            if primary_agent in (AgentType.TRM, AgentType.MCTS):
                return scores["phase_appropriate"]
            return scores["phase_mismatch"]

        return scores["default"]

    def _get_expected_agent_for_phase(
        self,
        phase: GamePhase,
    ) -> AgentType:
        """Get the expected agent for a game phase.

        Args:
            phase: Game phase

        Returns:
            Expected agent type
        """
        if phase == GamePhase.OPENING:
            return self._config.opening_expected_agent
        elif phase == GamePhase.MIDDLEGAME:
            return self._config.middlegame_expected_agent
        else:
            return self._config.endgame_expected_agent


def create_ensemble_checker(
    ensemble_agent: ChessEnsembleAgent | None = None,
    config: EnsembleCheckerConfig | None = None,
) -> EnsembleConsistencyChecker:
    """Factory function to create an EnsembleConsistencyChecker.

    Args:
        ensemble_agent: Optional ensemble agent
        config: Optional checker configuration

    Returns:
        Configured EnsembleConsistencyChecker instance
    """
    return EnsembleConsistencyChecker(
        ensemble_agent=ensemble_agent,
        config=config,
    )
