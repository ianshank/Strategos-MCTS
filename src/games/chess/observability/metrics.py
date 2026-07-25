"""
Chess Metrics Collector.

Provides a standalone metrics collector for chess verification and gameplay
(chess-specific counters; independent of src.observability.metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChessVerificationMetrics:
    """Chess-specific verification metrics."""

    # Verification counts
    games_verified: int = 0
    moves_validated: int = 0
    positions_verified: int = 0

    # Issue tracking
    invalid_moves_detected: int = 0
    encoding_errors: int = 0
    verification_errors: int = 0

    # Agent metrics
    agent_agreement_sum: float = 0.0
    agent_agreement_count: int = 0

    # Timing metrics (accumulative)
    total_verification_time_ms: float = 0.0
    total_move_validation_time_ms: float = 0.0

    # Edge case tracking
    edge_cases_tested: dict[str, int] = field(default_factory=dict)

    @property
    def avg_verification_time_ms(self) -> float:
        """Calculate average verification time."""
        if self.games_verified == 0:
            return 0.0
        return self.total_verification_time_ms / self.games_verified

    @property
    def avg_move_validation_time_ms(self) -> float:
        """Calculate average move validation time."""
        if self.moves_validated == 0:
            return 0.0
        return self.total_move_validation_time_ms / self.moves_validated

    @property
    def agent_agreement_rate(self) -> float:
        """Calculate average agent agreement rate."""
        if self.agent_agreement_count == 0:
            return 0.0
        return self.agent_agreement_sum / self.agent_agreement_count

    @property
    def error_rate(self) -> float:
        """Calculate overall error rate."""
        total = self.games_verified + self.moves_validated
        if total == 0:
            return 0.0
        errors = self.invalid_moves_detected + self.encoding_errors + self.verification_errors
        return errors / total

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "games_verified": self.games_verified,
            "moves_validated": self.moves_validated,
            "positions_verified": self.positions_verified,
            "invalid_moves_detected": self.invalid_moves_detected,
            "encoding_errors": self.encoding_errors,
            "verification_errors": self.verification_errors,
            "avg_verification_time_ms": round(self.avg_verification_time_ms, 2),
            "avg_move_validation_time_ms": round(self.avg_move_validation_time_ms, 2),
            "agent_agreement_rate": round(self.agent_agreement_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "edge_cases_tested": self.edge_cases_tested,
        }


@dataclass
class PhaseRoutingStats:
    """Statistics for agent routing by game phase."""

    total_routings: int = 0
    agent_selections: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    confidence_sum: float = 0.0

    def record_routing(self, agent: str, confidence: float) -> None:
        """Record a routing decision.

        Args:
            agent: Selected agent name
            confidence: Routing confidence
        """
        self.total_routings += 1
        self.agent_selections[agent] = self.agent_selections.get(agent, 0) + 1
        self.confidence_sum += confidence
        self.avg_confidence = self.confidence_sum / self.total_routings

    def get_most_selected_agent(self) -> str | None:
        """Get the most frequently selected agent."""
        if not self.agent_selections:
            return None
        return max(self.agent_selections.items(), key=lambda x: x[1])[0]


class ChessMetricsCollector:
    """Extended metrics collector for chess verification.

    Uses singleton pattern for consistent metrics collection across
    the application.

    Collects and aggregates chess-specific metrics including:
    - Verification statistics
    - Agent routing statistics by phase
    - Performance timing

    Example:
        >>> collector = ChessMetricsCollector.get_instance()
        >>> collector.record_game_verification("game_001", True, 150.0, 40)
        >>> report = collector.get_verification_report()
    """

    _instance: ChessMetricsCollector | None = None

    def __new__(cls) -> ChessMetricsCollector:
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the metrics collector (only once)."""
        if getattr(self, "_initialized", False):
            return
        self._verification_metrics = ChessVerificationMetrics()
        self._phase_routing_stats: dict[str, PhaseRoutingStats] = {}
        self._initialized = True

    @classmethod
    def get_instance(cls) -> ChessMetricsCollector:
        """Get the singleton instance.

        Returns:
            ChessMetricsCollector: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    @property
    def verification_metrics(self) -> ChessVerificationMetrics:
        """Get verification metrics."""
        return self._verification_metrics

    def record_game_verification(
        self,
        game_id: str,
        success: bool,
        duration_ms: float,
        moves_count: int,
    ) -> None:
        """Record game verification result.

        Args:
            game_id: Game identifier
            success: Whether verification succeeded
            duration_ms: Duration in milliseconds
            moves_count: Number of moves verified
        """
        self._verification_metrics.games_verified += 1
        self._verification_metrics.total_verification_time_ms += duration_ms

        if not success:
            self._verification_metrics.verification_errors += 1

    def record_move_validation(
        self,
        move: str,
        is_valid: bool,
        duration_ms: float,
        has_encoding_error: bool = False,
    ) -> None:
        """Record move validation result.

        Args:
            move: Move in UCI format
            is_valid: Whether move is valid
            duration_ms: Duration in milliseconds
            has_encoding_error: Whether encoding error occurred
        """
        self._verification_metrics.moves_validated += 1
        self._verification_metrics.total_move_validation_time_ms += duration_ms

        if not is_valid:
            self._verification_metrics.invalid_moves_detected += 1

        if has_encoding_error:
            self._verification_metrics.encoding_errors += 1

    def record_position_verification(
        self,
        fen: str,
        is_valid: bool,
    ) -> None:
        """Record position verification result.

        Args:
            fen: Position FEN
            is_valid: Whether position is valid
        """
        self._verification_metrics.positions_verified += 1

        if not is_valid:
            self._verification_metrics.verification_errors += 1

    def record_agent_routing(
        self,
        phase: str,
        agent_selected: str,
        confidence: float,
    ) -> None:
        """Record agent routing decision.

        Args:
            phase: Game phase (opening, middlegame, endgame)
            agent_selected: Selected agent name
            confidence: Routing confidence
        """
        if phase not in self._phase_routing_stats:
            self._phase_routing_stats[phase] = PhaseRoutingStats()

        self._phase_routing_stats[phase].record_routing(agent_selected, confidence)

    def record_agent_agreement(
        self,
        agreement_rate: float,
    ) -> None:
        """Record agent agreement rate.

        Args:
            agreement_rate: Agreement rate between 0 and 1
        """
        self._verification_metrics.agent_agreement_sum += agreement_rate
        self._verification_metrics.agent_agreement_count += 1

    def record_edge_case(
        self,
        case_type: str,
    ) -> None:
        """Record an edge case test.

        Args:
            case_type: Type of edge case (castling, en_passant, promotion)
        """
        edge_cases = self._verification_metrics.edge_cases_tested
        edge_cases[case_type] = edge_cases.get(case_type, 0) + 1

    def record_verification_error(
        self,
        error_type: str,
    ) -> None:
        """Record a verification error.

        Args:
            error_type: Type of error (execution_error, validation_error, etc.)
        """
        # Track errors using edge_cases dict for simplicity
        error_key = f"error_{error_type}"
        edge_cases = self._verification_metrics.edge_cases_tested
        edge_cases[error_key] = edge_cases.get(error_key, 0) + 1

    def get_verification_report(self) -> dict[str, Any]:
        """Generate verification report.

        Returns:
            Dictionary with verification statistics
        """
        return {
            "verification_metrics": self._verification_metrics.to_dict(),
            "phase_routing": {
                phase: {
                    "total_routings": stats.total_routings,
                    "agent_selections": stats.agent_selections,
                    "avg_confidence": round(stats.avg_confidence, 4),
                    "most_selected": stats.get_most_selected_agent(),
                }
                for phase, stats in self._phase_routing_stats.items()
            },
        }

    def get_phase_routing_stats(self, phase: str) -> PhaseRoutingStats | None:
        """Get routing statistics for a specific phase.

        Args:
            phase: Game phase

        Returns:
            PhaseRoutingStats or None if no data
        """
        return self._phase_routing_stats.get(phase)

    def reset(self) -> None:
        """Reset all metrics."""
        self._verification_metrics = ChessVerificationMetrics()
        self._phase_routing_stats = {}

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Summary string
        """
        m = self._verification_metrics
        return (
            f"Chess Verification Metrics:\n"
            f"  Games verified: {m.games_verified}\n"
            f"  Moves validated: {m.moves_validated}\n"
            f"  Error rate: {m.error_rate * 100:.2f}%\n"
            f"  Agent agreement: {m.agent_agreement_rate * 100:.2f}%\n"
            f"  Avg verification time: {m.avg_verification_time_ms:.2f}ms"
        )
