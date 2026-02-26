"""
Chess Verification Type Definitions.

Provides dataclasses and enums for verification results,
following the project's typing conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.games.chess.constants import STARTING_FEN


class MoveType(Enum):
    """Types of chess moves for validation."""

    NORMAL = "normal"
    CAPTURE = "capture"
    CASTLE_KINGSIDE = "castle_kingside"
    CASTLE_QUEENSIDE = "castle_queenside"
    EN_PASSANT = "en_passant"
    PROMOTION = "promotion"
    PROMOTION_CAPTURE = "promotion_capture"
    CHECK = "check"
    CHECKMATE = "checkmate"


class GameResult(Enum):
    """Possible game outcomes."""

    WHITE_WINS = "white_wins"
    BLACK_WINS = "black_wins"
    DRAW_STALEMATE = "draw_stalemate"
    DRAW_INSUFFICIENT_MATERIAL = "draw_insufficient_material"
    DRAW_FIFTY_MOVES = "draw_fifty_moves"
    DRAW_THREEFOLD_REPETITION = "draw_threefold_repetition"
    DRAW_AGREEMENT = "draw_agreement"
    IN_PROGRESS = "in_progress"


class VerificationSeverity(Enum):
    """Severity levels for verification issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class VerificationIssue:
    """A single verification issue found during validation."""

    code: str
    message: str
    severity: VerificationSeverity
    context: dict[str, Any] = field(default_factory=dict)
    move_number: int | None = None
    fen: str | None = None

    def __str__(self) -> str:
        """Human-readable representation."""
        prefix = f"[{self.severity.value.upper()}]"
        location = f" (move {self.move_number})" if self.move_number else ""
        return f"{prefix} {self.code}: {self.message}{location}"


@dataclass
class MoveValidationResult:
    """Result of validating a single chess move."""

    is_valid: bool
    move_uci: str
    move_type: MoveType
    encoded_index: int | None = None
    issues: list[VerificationIssue] = field(default_factory=list)
    extra_info: dict[str, Any] = field(default_factory=dict)

    # Move details
    from_square: str | None = None
    to_square: str | None = None
    piece_moved: str | None = None
    piece_captured: str | None = None
    promotion_piece: str | None = None

    # State checks
    is_check: bool = False
    is_checkmate: bool = False
    is_legal_in_position: bool = True

    @property
    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return any(
            issue.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "move_uci": self.move_uci,
            "move_type": self.move_type.value,
            "encoded_index": self.encoded_index,
            "issues": [
                {
                    "code": i.code,
                    "message": i.message,
                    "severity": i.severity.value,
                }
                for i in self.issues
            ],
            "from_square": self.from_square,
            "to_square": self.to_square,
            "piece_moved": self.piece_moved,
            "is_check": self.is_check,
            "is_checkmate": self.is_checkmate,
        }


@dataclass
class PositionVerificationResult:
    """Result of verifying a chess position."""

    is_valid: bool
    fen: str
    issues: list[VerificationIssue] = field(default_factory=list)
    extra_info: dict[str, Any] = field(default_factory=dict)

    # Position state
    is_terminal: bool = False
    game_result: GameResult | None = None
    legal_moves_count: int = 0
    material_balance: int = 0
    game_phase: str | None = None

    # Validity checks
    has_valid_king_positions: bool = True
    has_valid_pawn_positions: bool = True
    has_valid_castling_rights: bool = True
    has_valid_en_passant: bool = True

    @property
    def has_errors(self) -> bool:
        """Check if verification found any errors."""
        return any(
            issue.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for issue in self.issues
        )


@dataclass
class MoveSequenceResult:
    """Result of verifying a sequence of chess moves."""

    is_valid: bool
    initial_fen: str
    moves: list[str]
    final_fen: str | None = None
    issues: list[VerificationIssue] = field(default_factory=list)
    move_results: list[MoveValidationResult] = field(default_factory=list)

    # Sequence statistics
    total_moves: int = 0
    valid_moves: int = 0
    captures: int = 0
    checks: int = 0
    castles: int = 0
    promotions: int = 0

    # Timing (if measured)
    validation_time_ms: float = 0.0

    @property
    def has_errors(self) -> bool:
        """Check if verification found any errors."""
        return any(
            issue.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for issue in self.issues
        )

    @property
    def error_rate(self) -> float:
        """Calculate the error rate for the sequence."""
        if self.total_moves == 0:
            return 0.0
        return 1.0 - (self.valid_moves / self.total_moves)


@dataclass
class GameVerificationResult:
    """Result of verifying a complete chess game."""

    is_valid: bool
    game_id: str
    moves: list[str]
    result: GameResult
    issues: list[VerificationIssue] = field(default_factory=list)
    move_sequence_result: MoveSequenceResult | None = None

    # Game metadata
    initial_fen: str = STARTING_FEN
    final_fen: str | None = None
    total_moves: int = 0
    total_plies: int = 0

    # Verification details
    expected_result: GameResult | None = None
    result_matches_expected: bool = True

    # Timing
    verification_time_ms: float = 0.0

    @property
    def has_errors(self) -> bool:
        """Check if verification found any errors."""
        return any(
            issue.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for issue in self.issues
        )

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "VALID" if self.is_valid else "INVALID"
        errors = sum(
            1 for i in self.issues if i.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL)
        )
        warnings = sum(1 for i in self.issues if i.severity == VerificationSeverity.WARNING)
        return f"Game {self.game_id}: {status} ({self.total_moves} moves, {errors} errors, {warnings} warnings)"


@dataclass
class EnsembleConsistencyResult:
    """Result of checking ensemble agent consistency."""

    is_consistent: bool
    state_fen: str
    issues: list[VerificationIssue] = field(default_factory=list)

    # Agreement metrics
    agreement_rate: float = 0.0
    move_variance: dict[str, float] = field(default_factory=dict)
    agent_divergences: dict[str, float] = field(default_factory=dict)
    routing_consistency: float = 0.0

    # Agent responses
    agent_moves: dict[str, str] = field(default_factory=dict)
    agent_confidences: dict[str, float] = field(default_factory=dict)
    agent_values: dict[str, float] = field(default_factory=dict)

    # Ensemble decision
    ensemble_move: str | None = None
    ensemble_confidence: float = 0.0
    primary_agent: str | None = None

    # Timing
    check_time_ms: float = 0.0

    @property
    def all_agents_agree(self) -> bool:
        """Check if all agents selected the same move."""
        if not self.agent_moves:
            return True
        moves = list(self.agent_moves.values())
        return len(set(moves)) == 1

    def get_disagreeing_agents(self) -> list[str]:
        """Get list of agents that disagree with the ensemble decision."""
        if not self.ensemble_move:
            return []
        return [agent for agent, move in self.agent_moves.items() if move != self.ensemble_move]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_consistent": self.is_consistent,
            "state_fen": self.state_fen,
            "agreement_rate": self.agreement_rate,
            "agent_moves": self.agent_moves,
            "agent_confidences": self.agent_confidences,
            "ensemble_move": self.ensemble_move,
            "primary_agent": self.primary_agent,
            "all_agents_agree": self.all_agents_agree,
        }


@dataclass
class BatchVerificationResult:
    """Result of verifying multiple items in batch."""

    total_items: int
    valid_items: int
    invalid_items: int
    results: list[GameVerificationResult | MoveSequenceResult | EnsembleConsistencyResult]
    issues: list[VerificationIssue] = field(default_factory=list)

    # Timing
    total_time_ms: float = 0.0
    avg_time_per_item_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate the success rate."""
        if self.total_items == 0:
            return 0.0
        return self.valid_items / self.total_items

    def summary(self) -> str:
        """Generate a human-readable summary."""
        return (
            f"Batch verification: {self.valid_items}/{self.total_items} valid "
            f"({self.success_rate * 100:.1f}%), "
            f"total time: {self.total_time_ms:.1f}ms"
        )
