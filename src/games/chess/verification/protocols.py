"""
Chess Verification Protocols.

Defines Protocol classes for verification components,
enabling loose coupling and easy testing/mocking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.games.chess.state import ChessGameState
    from src.games.chess.verification.types import (
        EnsembleConsistencyResult,
        GameResult,
        GameVerificationResult,
        MoveSequenceResult,
        MoveValidationResult,
        PositionVerificationResult,
    )


class MoveValidatorProtocol(Protocol):
    """Protocol for chess move validation.

    Implementations must provide methods for validating
    individual moves and special move types (castling, en passant, promotion).
    """

    def validate_move(
        self,
        state: ChessGameState,
        move_uci: str,
    ) -> MoveValidationResult:
        """Validate a move in UCI format.

        Args:
            state: Current chess position
            move_uci: Move in UCI format (e.g., 'e2e4', 'e7e8q')

        Returns:
            MoveValidationResult with validation details
        """
        ...

    def validate_castling(
        self,
        state: ChessGameState,
        kingside: bool,
    ) -> MoveValidationResult:
        """Validate a castling move.

        Args:
            state: Current chess position
            kingside: True for kingside (O-O), False for queenside (O-O-O)

        Returns:
            MoveValidationResult with castling-specific validation
        """
        ...

    def validate_en_passant(
        self,
        state: ChessGameState,
        move_uci: str,
    ) -> MoveValidationResult:
        """Validate an en passant capture.

        Args:
            state: Current chess position
            move_uci: En passant move in UCI format

        Returns:
            MoveValidationResult with en passant-specific validation
        """
        ...

    def validate_promotion(
        self,
        state: ChessGameState,
        move_uci: str,
    ) -> MoveValidationResult:
        """Validate a pawn promotion.

        Args:
            state: Current chess position
            move_uci: Promotion move in UCI format (e.g., 'e7e8q')

        Returns:
            MoveValidationResult with promotion-specific validation
        """
        ...

    def validate_encoding_roundtrip(
        self,
        state: ChessGameState,
        move_uci: str,
    ) -> MoveValidationResult:
        """Validate that a move survives encoding/decoding roundtrip.

        Args:
            state: Current chess position
            move_uci: Move in UCI format

        Returns:
            MoveValidationResult with encoding validation
        """
        ...


class ChessGameVerifierProtocol(Protocol):
    """Protocol for chess game verification.

    Implementations must provide methods for verifying
    complete games, positions, and move sequences.
    """

    async def verify_full_game(
        self,
        moves: list[str],
        expected_outcome: GameResult | None = None,
        game_id: str | None = None,
    ) -> GameVerificationResult:
        """Verify a complete chess game.

        Args:
            moves: List of moves in UCI format
            expected_outcome: Expected game result (if known)
            game_id: Optional game identifier

        Returns:
            GameVerificationResult with verification details
        """
        ...

    def verify_position(
        self,
        fen: str,
    ) -> PositionVerificationResult:
        """Verify a chess position.

        Args:
            fen: Position in FEN format

        Returns:
            PositionVerificationResult with position validation
        """
        ...

    def verify_move_sequence(
        self,
        initial_fen: str,
        moves: list[str],
    ) -> MoveSequenceResult:
        """Verify a sequence of moves from a given position.

        Args:
            initial_fen: Starting position in FEN format
            moves: List of moves in UCI format

        Returns:
            MoveSequenceResult with sequence validation
        """
        ...

    async def verify_game_playthrough(
        self,
        state: ChessGameState,
        max_moves: int | None = None,
    ) -> GameVerificationResult:
        """Verify a game by playing it through.

        Args:
            state: Starting position
            max_moves: Maximum number of moves to play

        Returns:
            GameVerificationResult with playthrough validation
        """
        ...


class EnsembleConsistencyCheckerProtocol(Protocol):
    """Protocol for ensemble agent consistency checking.

    Implementations must provide methods for checking
    consistency between HRM, TRM, and MCTS agents.
    """

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
        ...

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
        ...

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
            List of EnsembleConsistencyResult for each move
        """
        ...

    def get_divergence_threshold(self) -> float:
        """Get the threshold for acceptable divergence.

        Returns:
            Float threshold value
        """
        ...


class SubAgentVerifierProtocol(Protocol):
    """Protocol for sub-agent specific verification.

    Implementations verify HRM, TRM, or MCTS agent behavior.

    Note:
        This protocol is designed for future use when individual
        sub-agent verification is implemented. Currently reserved
        for forward compatibility.
    """

    async def verify_agent_response(
        self,
        state: ChessGameState,
        agent_name: str,
    ) -> MoveValidationResult:
        """Verify a sub-agent's response for a position.

        Args:
            state: Chess position
            agent_name: Name of the agent (hrm, trm, mcts)

        Returns:
            MoveValidationResult with agent-specific validation
        """
        ...

    async def verify_routing_decision(
        self,
        state: ChessGameState,
    ) -> bool:
        """Verify that routing decision is appropriate.

        Args:
            state: Chess position

        Returns:
            True if routing is appropriate for the position
        """
        ...
