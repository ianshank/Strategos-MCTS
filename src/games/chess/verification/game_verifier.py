"""
Chess Game Verifier.

Provides comprehensive verification for complete chess games,
positions, and move sequences.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import chess

from src.games.chess.constants import (
    EN_PASSANT_RANKS,
    INVALID_PAWN_RANKS,
    STARTING_FEN,
    UUID_SHORT_LENGTH,
    get_piece_values,
    truncate_fen,
)
from src.games.chess.state import ChessGameState
from src.games.chess.verification.move_validator import MoveValidator, MoveValidatorConfig
from src.games.chess.verification.types import (
    GameResult,
    GameVerificationResult,
    MoveSequenceResult,
    MoveType,
    PositionVerificationResult,
    VerificationIssue,
    VerificationSeverity,
)
from src.observability.logging import StructuredLogger, get_structured_logger


@dataclass
class GameVerifierConfig:
    """Configuration for game verification.

    All parameters are configurable - no hardcoded values.
    """

    # Verification options
    verify_encoding: bool = True
    verify_terminal_state: bool = True
    verify_result: bool = True
    stop_on_first_error: bool = False

    # Limits
    max_moves: int = 500
    max_repetitions: int = 10

    # Logging
    log_verifications: bool = True

    # Move validator config
    move_validator_config: MoveValidatorConfig = field(default_factory=MoveValidatorConfig)


class ChessGameVerifier:
    """Verifies complete chess games and positions.

    Provides comprehensive verification including:
    - Complete game playthrough verification
    - Position validity checking
    - Move sequence validation
    - Terminal state verification

    Example:
        >>> verifier = ChessGameVerifier()
        >>> result = await verifier.verify_full_game(
        ...     ["e2e4", "e7e5", "g1f3", "b8c6"],
        ... )
        >>> print(result.is_valid)
    """

    # Starting FEN uses centralized constant
    DEFAULT_STARTING_FEN: str = STARTING_FEN

    def __init__(
        self,
        move_validator: MoveValidator | None = None,
        config: GameVerifierConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize the game verifier.

        Args:
            move_validator: Optional move validator (created with defaults if None)
            config: Verifier configuration
            logger: Optional logger instance
        """
        self._config = config or GameVerifierConfig()
        self._move_validator = move_validator or MoveValidator(config=self._config.move_validator_config)
        self._logger = logger or get_structured_logger("chess.verification.game_verifier")

    @property
    def config(self) -> GameVerifierConfig:
        """Get the verifier configuration."""
        return self._config

    @property
    def move_validator(self) -> MoveValidator:
        """Get the move validator."""
        return self._move_validator

    async def verify_full_game(
        self,
        moves: list[str],
        expected_outcome: GameResult | None = None,
        game_id: str | None = None,
        initial_fen: str | None = None,
    ) -> GameVerificationResult:
        """Verify a complete chess game.

        Args:
            moves: List of moves in UCI format
            expected_outcome: Expected game result (if known)
            game_id: Optional game identifier
            initial_fen: Starting position (defaults to standard)

        Returns:
            GameVerificationResult with verification details
        """
        start_time = time.perf_counter()
        game_id = game_id or str(uuid.uuid4())[:UUID_SHORT_LENGTH]
        initial_fen = initial_fen or self.DEFAULT_STARTING_FEN
        issues: list[VerificationIssue] = []

        # Verify the starting position
        position_result = self.verify_position(initial_fen)
        if not position_result.is_valid:
            issues.extend(position_result.issues)
            if self._config.stop_on_first_error:
                return GameVerificationResult(
                    is_valid=False,
                    game_id=game_id,
                    moves=moves,
                    result=GameResult.IN_PROGRESS,
                    issues=issues,
                    initial_fen=initial_fen,
                    verification_time_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Verify move sequence
        sequence_result = self.verify_move_sequence(initial_fen, moves)
        if not sequence_result.is_valid:
            issues.extend(sequence_result.issues)

        # Determine game result
        game_result = self._determine_game_result(sequence_result.final_fen)

        # Check expected outcome
        result_matches = True
        if expected_outcome is not None and self._config.verify_result:
            result_matches = game_result == expected_outcome
            if not result_matches:
                issues.append(
                    VerificationIssue(
                        code="RESULT_MISMATCH",
                        message=(f"Expected {expected_outcome.value}, got {game_result.value}"),
                        severity=VerificationSeverity.ERROR,
                        context={
                            "expected": expected_outcome.value,
                            "actual": game_result.value,
                        },
                    )
                )

        verification_time_ms = (time.perf_counter() - start_time) * 1000

        # Log verification
        if self._config.log_verifications:
            self._logger.info(
                "Game verified",
                game_id=game_id,
                is_valid=sequence_result.is_valid and result_matches,
                total_moves=len(moves),
                result=game_result.value,
                issues_count=len(issues),
                duration_ms=round(verification_time_ms, 2),
            )

        return GameVerificationResult(
            is_valid=sequence_result.is_valid and result_matches,
            game_id=game_id,
            moves=moves,
            result=game_result,
            issues=issues,
            move_sequence_result=sequence_result,
            initial_fen=initial_fen,
            final_fen=sequence_result.final_fen,
            total_moves=len(moves),
            total_plies=len(moves),
            expected_result=expected_outcome,
            result_matches_expected=result_matches,
            verification_time_ms=verification_time_ms,
        )

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
        issues: list[VerificationIssue] = []
        extra_info: dict[str, Any] = {}

        # Try to create a board from the FEN
        try:
            board = chess.Board(fen)
        except ValueError as e:
            issues.append(
                VerificationIssue(
                    code="INVALID_FEN",
                    message=f"Invalid FEN string: {e}",
                    severity=VerificationSeverity.CRITICAL,
                    context={"fen": fen},
                )
            )
            return PositionVerificationResult(
                is_valid=False,
                fen=fen,
                issues=issues,
            )

        # Validate position
        validation_checks = self._validate_position_checks(board)
        for check_name, (is_valid, message) in validation_checks.items():
            if not is_valid:
                issues.append(
                    VerificationIssue(
                        code=f"POSITION_{check_name.upper()}",
                        message=message,
                        severity=VerificationSeverity.ERROR,
                        fen=fen,
                    )
                )

        # Check terminal state
        is_terminal = board.is_game_over()
        game_result = None
        if is_terminal:
            game_result = self._get_game_result_from_board(board)

        # Count legal moves
        legal_moves_count = len(list(board.legal_moves))

        # Material balance
        material_balance = self._calculate_material_balance(board)

        # Game phase
        state = ChessGameState.from_fen(fen)
        game_phase = state.get_game_phase().value

        return PositionVerificationResult(
            is_valid=not any(i.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for i in issues),
            fen=fen,
            issues=issues,
            extra_info=extra_info,
            is_terminal=is_terminal,
            game_result=game_result,
            legal_moves_count=legal_moves_count,
            material_balance=material_balance,
            game_phase=game_phase,
            has_valid_king_positions=validation_checks.get("king_positions", (True, ""))[0],
            has_valid_pawn_positions=validation_checks.get("pawn_positions", (True, ""))[0],
            has_valid_castling_rights=validation_checks.get("castling_rights", (True, ""))[0],
            has_valid_en_passant=validation_checks.get("en_passant", (True, ""))[0],
        )

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
        start_time = time.perf_counter()
        issues: list[VerificationIssue] = []
        move_results = []

        # Statistics
        total_moves = len(moves)
        valid_moves = 0
        captures = 0
        checks = 0
        castles = 0
        promotions = 0

        # Create starting position
        try:
            state = ChessGameState.from_fen(initial_fen)
        except ValueError as e:
            # FEN parsing error
            issues.append(
                VerificationIssue(
                    code="INVALID_INITIAL_FEN",
                    message=f"Invalid FEN format: {e}",
                    severity=VerificationSeverity.CRITICAL,
                    fen=initial_fen,
                    context={"error_type": "ValueError"},
                )
            )
            return MoveSequenceResult(
                is_valid=False,
                initial_fen=initial_fen,
                moves=moves,
                issues=issues,
                total_moves=total_moves,
            )

        # Play through the game
        current_state = state
        for move_number, move_uci in enumerate(moves, 1):
            # Validate the move
            move_result = self._move_validator.validate_move(current_state, move_uci)
            move_results.append(move_result)

            if not move_result.is_valid:
                issues.append(
                    VerificationIssue(
                        code="INVALID_MOVE",
                        message=f"Move {move_uci} is invalid: {move_result.issues}",
                        severity=VerificationSeverity.ERROR,
                        move_number=move_number,
                        fen=current_state.fen,
                        context={"move": move_uci},
                    )
                )
                if self._config.stop_on_first_error:
                    break
                continue

            valid_moves += 1

            # Count move types
            if move_result.move_type == MoveType.CAPTURE:
                captures += 1
            elif move_result.move_type in (
                MoveType.CASTLE_KINGSIDE,
                MoveType.CASTLE_QUEENSIDE,
            ):
                castles += 1
            elif move_result.move_type in (MoveType.PROMOTION, MoveType.PROMOTION_CAPTURE):
                promotions += 1
                if move_result.move_type == MoveType.PROMOTION_CAPTURE:
                    captures += 1
            elif move_result.move_type == MoveType.EN_PASSANT:
                captures += 1

            if move_result.is_check:
                checks += 1

            # Apply the move
            try:
                current_state = current_state.apply_action(move_uci)
            except ValueError as e:
                issues.append(
                    VerificationIssue(
                        code="MOVE_APPLICATION_FAILED",
                        message=f"Failed to apply move {move_uci}: {e}",
                        severity=VerificationSeverity.ERROR,
                        move_number=move_number,
                        fen=current_state.fen,
                    )
                )
                if self._config.stop_on_first_error:
                    break

            # Check for excessive repetitions
            if current_state.is_repetition(self._config.max_repetitions):
                issues.append(
                    VerificationIssue(
                        code="EXCESSIVE_REPETITIONS",
                        message=(f"Position repeated {self._config.max_repetitions} times"),
                        severity=VerificationSeverity.WARNING,
                        move_number=move_number,
                        fen=current_state.fen,
                    )
                )

        validation_time_ms = (time.perf_counter() - start_time) * 1000

        return MoveSequenceResult(
            is_valid=valid_moves == total_moves
            and not any(i.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for i in issues),
            initial_fen=initial_fen,
            moves=moves,
            final_fen=current_state.fen,
            issues=issues,
            move_results=move_results,
            total_moves=total_moves,
            valid_moves=valid_moves,
            captures=captures,
            checks=checks,
            castles=castles,
            promotions=promotions,
            validation_time_ms=validation_time_ms,
        )

    async def verify_game_playthrough(
        self,
        state: ChessGameState,
        max_moves: int | None = None,
    ) -> GameVerificationResult:
        """Verify a game by playing it through from current position.

        This validates that the game state can be played to completion
        without any illegal moves or errors.

        Args:
            state: Starting position
            max_moves: Maximum number of moves to play

        Returns:
            GameVerificationResult with playthrough validation
        """
        max_moves = max_moves or self._config.max_moves
        start_time = time.perf_counter()
        game_id = str(uuid.uuid4())[:UUID_SHORT_LENGTH]
        issues: list[VerificationIssue] = []
        moves_played: list[str] = []

        current_state = state
        move_count = 0

        while not current_state.is_terminal() and move_count < max_moves:
            # Get legal moves
            legal_moves = current_state.get_legal_actions()
            if not legal_moves:
                break

            # Pick a move (first legal move for verification)
            move = legal_moves[0]
            moves_played.append(move)

            # Apply the move
            try:
                current_state = current_state.apply_action(move)
                move_count += 1
            except ValueError as e:
                issues.append(
                    VerificationIssue(
                        code="PLAYTHROUGH_MOVE_FAILED",
                        message=f"Move {move} failed: {e}",
                        severity=VerificationSeverity.ERROR,
                        move_number=move_count + 1,
                        fen=current_state.fen,
                    )
                )
                break

        # Determine game result
        game_result = self._get_game_result_from_board(current_state.board)

        verification_time_ms = (time.perf_counter() - start_time) * 1000

        return GameVerificationResult(
            is_valid=not any(i.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for i in issues),
            game_id=game_id,
            moves=moves_played,
            result=game_result,
            issues=issues,
            initial_fen=state.fen,
            final_fen=current_state.fen,
            total_moves=move_count,
            total_plies=move_count,
            verification_time_ms=verification_time_ms,
        )

    def _validate_position_checks(
        self,
        board: chess.Board,
    ) -> dict[str, tuple[bool, str]]:
        """Run validation checks on a position.

        Args:
            board: Chess board to validate

        Returns:
            Dictionary of check name to (is_valid, message) tuples
        """
        checks: dict[str, tuple[bool, str]] = {}

        # Check king positions
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))
        if white_kings != 1 or black_kings != 1:
            checks["king_positions"] = (
                False,
                f"Invalid king count: white={white_kings}, black={black_kings}",
            )
        else:
            checks["king_positions"] = (True, "")

        # Check pawn positions (no pawns on 1st or 8th rank)
        pawns_on_invalid_ranks = False
        for color in [chess.WHITE, chess.BLACK]:
            for square in board.pieces(chess.PAWN, color):
                rank = chess.square_rank(square)
                if rank in INVALID_PAWN_RANKS:
                    pawns_on_invalid_ranks = True
                    break

        if pawns_on_invalid_ranks:
            checks["pawn_positions"] = (False, "Pawns on first or eighth rank")
        else:
            checks["pawn_positions"] = (True, "")

        # Check en passant square validity
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            is_white = board.turn == chess.WHITE
            expected_rank = EN_PASSANT_RANKS[is_white]
            if ep_rank != expected_rank:
                checks["en_passant"] = (
                    False,
                    f"En passant square on wrong rank: {chess.square_name(board.ep_square)}",
                )
            else:
                checks["en_passant"] = (True, "")
        else:
            checks["en_passant"] = (True, "")

        # Check castling rights consistency
        checks["castling_rights"] = (True, "")
        if board.castling_rights:
            # Check if the required pieces are in place
            if board.castling_rights & chess.BB_H1 and (
                board.piece_at(chess.E1) != chess.Piece(chess.KING, chess.WHITE)
                or board.piece_at(chess.H1) != chess.Piece(chess.ROOK, chess.WHITE)
            ):
                checks["castling_rights"] = (
                    False,
                    "White kingside castling rights but pieces not in place",
                )
            if board.castling_rights & chess.BB_A1 and (
                board.piece_at(chess.E1) != chess.Piece(chess.KING, chess.WHITE)
                or board.piece_at(chess.A1) != chess.Piece(chess.ROOK, chess.WHITE)
            ):
                checks["castling_rights"] = (
                    False,
                    "White queenside castling rights but pieces not in place",
                )

        return checks

    def _determine_game_result(
        self,
        final_fen: str | None,
    ) -> GameResult:
        """Determine game result from final position.

        Args:
            final_fen: Final position in FEN format

        Returns:
            GameResult enum value
        """
        if final_fen is None:
            return GameResult.IN_PROGRESS

        try:
            state = ChessGameState.from_fen(final_fen)
            return self._get_game_result_from_board(state.board)
        except ValueError:
            # Invalid FEN format
            self._logger.warning(
                "Invalid FEN when determining game result",
                fen=truncate_fen(final_fen) if final_fen else "None",
            )
            return GameResult.IN_PROGRESS

    def _get_game_result_from_board(
        self,
        board: chess.Board,
    ) -> GameResult:
        """Get game result from a chess board.

        Args:
            board: Chess board

        Returns:
            GameResult enum value
        """
        if not board.is_game_over():
            return GameResult.IN_PROGRESS

        outcome = board.outcome()
        if outcome is None:
            return GameResult.IN_PROGRESS

        if outcome.winner == chess.WHITE:
            return GameResult.WHITE_WINS
        elif outcome.winner == chess.BLACK:
            return GameResult.BLACK_WINS
        else:
            # Draw - determine type
            if board.is_stalemate():
                return GameResult.DRAW_STALEMATE
            elif board.is_insufficient_material():
                return GameResult.DRAW_INSUFFICIENT_MATERIAL
            elif board.is_fifty_moves():
                return GameResult.DRAW_FIFTY_MOVES
            elif board.can_claim_threefold_repetition():
                return GameResult.DRAW_THREEFOLD_REPETITION
            else:
                return GameResult.DRAW_AGREEMENT

    def _calculate_material_balance(
        self,
        board: chess.Board,
    ) -> int:
        """Calculate material balance from white's perspective.

        Uses piece values from settings for configurability.

        Args:
            board: Chess board

        Returns:
            Material balance in centipawns
        """
        # Get piece values from centralized configuration
        piece_values = get_piece_values()

        balance = 0
        for piece_type, value in piece_values.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            balance += (white_count - black_count) * value

        return balance


def create_game_verifier(
    config: GameVerifierConfig | None = None,
    move_validator: MoveValidator | None = None,
) -> ChessGameVerifier:
    """Factory function to create a ChessGameVerifier.

    Args:
        config: Optional verifier configuration
        move_validator: Optional move validator

    Returns:
        Configured ChessGameVerifier instance
    """
    return ChessGameVerifier(
        move_validator=move_validator,
        config=config,
    )
