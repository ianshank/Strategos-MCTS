"""
Chess Verification Logger.

Provides specialized logging for chess verification events
with structured data and correlation ID tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.games.chess.ensemble_agent import AgentResponse
    from src.games.chess.meta_controller import RoutingDecision
    from src.games.chess.verification.types import (
        GameVerificationResult,
        MoveValidationResult,
    )

from src.games.chess.constants import truncate_fen
from src.observability.logging import (
    StructuredLogger,
)


class ChessVerificationLogger(StructuredLogger):
    """Specialized logger for chess verification events.

    Extends StructuredLogger with chess-specific logging methods
    for moves, games, and ensemble decisions.

    Example:
        >>> logger = ChessVerificationLogger("chess.verification")
        >>> logger.log_move_validation("rnbqkbnr/...", "e2e4", result)
    """

    def __init__(self, name: str = "chess.verification") -> None:
        """Initialize the chess verification logger.

        Args:
            name: Logger name
        """
        super().__init__(name)

    def log_game_verification(
        self,
        game_id: str,
        result: GameVerificationResult,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log game verification result.

        Args:
            game_id: Game identifier
            result: Verification result
            duration_ms: Verification duration in milliseconds
            **extra: Additional context
        """
        self.info(
            f"Game verification completed: {game_id}",
            game_id=game_id,
            is_valid=result.is_valid,
            total_moves=result.total_moves,
            game_result=result.result.value,
            issues_count=len(result.issues),
            duration_ms=round(duration_ms, 2),
            **extra,
        )

    def log_move_validation(
        self,
        state_fen: str,
        move: str,
        result: MoveValidationResult,
        **extra: Any,
    ) -> None:
        """Log move validation result.

        Args:
            state_fen: Position FEN (truncated for logging)
            move: Move in UCI format
            result: Validation result
            **extra: Additional context
        """
        # Truncate FEN for logging using centralized function
        fen_short = truncate_fen(state_fen)

        self.debug(
            f"Move validation: {move}",
            fen=fen_short,
            move=move,
            is_valid=result.is_valid,
            move_type=result.move_type.value,
            encoded_index=result.encoded_index,
            is_check=result.is_check,
            is_checkmate=result.is_checkmate,
            **extra,
        )

    def log_ensemble_decision(
        self,
        state_fen: str,
        routing_decision: RoutingDecision,
        agent_responses: dict[str, AgentResponse],
        **extra: Any,
    ) -> None:
        """Log ensemble decision.

        Args:
            state_fen: Position FEN
            routing_decision: Routing decision from meta-controller
            agent_responses: Responses from all agents
            **extra: Additional context
        """
        fen_short = truncate_fen(state_fen)

        agent_moves = {
            name: response.move
            for name, response in agent_responses.items()
        }
        agent_confidences = {
            name: round(response.confidence, 3)
            for name, response in agent_responses.items()
        }

        self.info(
            f"Ensemble decision: {routing_decision.primary_agent.value}",
            fen=fen_short,
            primary_agent=routing_decision.primary_agent.value,
            agent_moves=agent_moves,
            agent_confidences=agent_confidences,
            agent_weights=routing_decision.agent_weights,
            **extra,
        )

    def log_agent_divergence(
        self,
        state_fen: str,
        divergence_details: dict[str, Any],
        **extra: Any,
    ) -> None:
        """Log agent divergence warning.

        Args:
            state_fen: Position FEN
            divergence_details: Details about the divergence
            **extra: Additional context
        """
        fen_short = truncate_fen(state_fen)

        self.warning(
            "Agent divergence detected",
            fen=fen_short,
            divergence_details=divergence_details,
            **extra,
        )

    def log_verification_error(
        self,
        operation: str,
        error: Exception,
        context: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Log verification error.

        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
            context: Additional context about the error
            **extra: Additional fields
        """
        self.error(
            f"Verification error in {operation}: {error}",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            **extra,
        )

    def log_game_phase_transition(
        self,
        game_id: str,
        from_phase: str,
        to_phase: str,
        move_number: int,
        **extra: Any,
    ) -> None:
        """Log game phase transition.

        Args:
            game_id: Game identifier
            from_phase: Previous phase
            to_phase: New phase
            move_number: Move number at transition
            **extra: Additional context
        """
        self.debug(
            f"Game phase transition: {from_phase} -> {to_phase}",
            game_id=game_id,
            from_phase=from_phase,
            to_phase=to_phase,
            move_number=move_number,
            **extra,
        )

    def log_encoding_roundtrip(
        self,
        move: str,
        encoded_index: int,
        decoded_move: str,
        success: bool,
        **extra: Any,
    ) -> None:
        """Log encoding roundtrip result.

        Args:
            move: Original move
            encoded_index: Encoded action index
            decoded_move: Decoded move
            success: Whether roundtrip succeeded
            **extra: Additional context
        """
        level = logging.DEBUG if success else logging.WARNING

        self._log(
            level,
            f"Encoding roundtrip: {move}",
            original_move=move,
            encoded_index=encoded_index,
            decoded_move=decoded_move,
            success=success,
            **extra,
        )


def get_chess_logger(name: str = "chess") -> ChessVerificationLogger:
    """Get a chess verification logger.

    Args:
        name: Logger name (will be prefixed with 'chess.')

    Returns:
        ChessVerificationLogger instance
    """
    if not name.startswith("chess."):
        name = f"chess.{name}"
    return ChessVerificationLogger(name)
