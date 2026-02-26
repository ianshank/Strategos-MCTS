"""
Chess Move Validator.

Provides comprehensive validation for chess moves including
special moves (castling, en passant, promotion) and edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from src.games.chess.action_space import ChessActionEncoder
from src.games.chess.config import ChessActionSpaceConfig
from src.games.chess.constants import CASTLING_MOVES
from src.games.chess.state import ChessGameState
from src.games.chess.verification.types import (
    MoveType,
    MoveValidationResult,
    VerificationIssue,
    VerificationSeverity,
)
from src.observability.logging import StructuredLogger, get_structured_logger


@dataclass
class MoveValidatorConfig:
    """Configuration for move validation.

    All parameters are configurable - no hardcoded values.
    """

    # Validation options
    validate_encoding: bool = True
    validate_legality: bool = True
    validate_san_format: bool = False

    # Tolerance settings
    encoding_mismatch_severity: VerificationSeverity = VerificationSeverity.ERROR

    # Logging
    log_validations: bool = False


class MoveValidator:
    """Validates chess moves comprehensively.

    Includes validation for:
    - Basic move legality
    - Special moves (castling, en passant, promotion)
    - Action encoding roundtrip
    - Edge cases

    Example:
        >>> validator = MoveValidator()
        >>> state = ChessGameState.initial()
        >>> result = validator.validate_move(state, "e2e4")
        >>> assert result.is_valid
    """

    def __init__(
        self,
        action_encoder: ChessActionEncoder | None = None,
        config: MoveValidatorConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize the move validator.

        Args:
            action_encoder: Optional action encoder (created with defaults if None)
            config: Validation configuration
            logger: Optional logger instance
        """
        self._config = config or MoveValidatorConfig()
        self._encoder = action_encoder or ChessActionEncoder(ChessActionSpaceConfig())
        self._logger = logger or get_structured_logger("chess.verification.move_validator")

    @property
    def config(self) -> MoveValidatorConfig:
        """Get the validator configuration."""
        return self._config

    @property
    def encoder(self) -> ChessActionEncoder:
        """Get the action encoder."""
        return self._encoder

    def validate_move(
        self,
        state: ChessGameState,
        move_uci: str,
    ) -> MoveValidationResult:
        """Validate a chess move.

        Args:
            state: Current chess position
            move_uci: Move in UCI format (e.g., 'e2e4')

        Returns:
            MoveValidationResult with validation details
        """
        extra_info: dict[str, Any] = {}

        # Parse the move using centralized helper
        move, issues = self._parse_uci_move(move_uci, MoveType.NORMAL)
        if move is None:
            return MoveValidationResult(
                is_valid=False,
                move_uci=move_uci,
                move_type=MoveType.NORMAL,
                issues=issues,
            )

        # Determine move type
        move_type = self._determine_move_type(state.board, move)

        # Check legality
        is_legal = move in state.board.legal_moves
        if not is_legal and self._config.validate_legality:
            issues.append(
                VerificationIssue(
                    code="ILLEGAL_MOVE",
                    message=f"Move {move_uci} is not legal in this position",
                    severity=VerificationSeverity.ERROR,
                    context={"fen": state.fen, "move": move_uci},
                )
            )

        # Validate encoding roundtrip
        encoded_index = None
        if self._config.validate_encoding and is_legal:
            encoding_result = self._validate_encoding(state, move_uci)
            encoded_index = encoding_result.get("encoded_index")
            if not encoding_result.get("roundtrip_valid", True):
                issues.append(
                    VerificationIssue(
                        code="ENCODING_MISMATCH",
                        message=f"Encoding roundtrip failed: {encoding_result.get('error')}",
                        severity=self._config.encoding_mismatch_severity,
                        context=encoding_result,
                    )
                )
            extra_info["encoding"] = encoding_result

        # Get move details
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        piece_moved = state.board.piece_at(move.from_square)
        piece_captured = state.board.piece_at(move.to_square)
        promotion_piece = chess.piece_name(move.promotion) if move.promotion else None

        # Check for check/checkmate after move
        is_check = False
        is_checkmate = False
        if is_legal:
            test_board = state.board.copy()
            test_board.push(move)
            is_check = test_board.is_check()
            is_checkmate = test_board.is_checkmate()

        # Log validation if enabled
        if self._config.log_validations:
            self._logger.debug(
                "Move validated",
                move=move_uci,
                is_valid=is_legal and not issues,
                move_type=move_type.value,
                issues_count=len(issues),
            )

        return MoveValidationResult(
            is_valid=is_legal
            and not any(i.severity in (VerificationSeverity.ERROR, VerificationSeverity.CRITICAL) for i in issues),
            move_uci=move_uci,
            move_type=move_type,
            encoded_index=encoded_index,
            issues=issues,
            extra_info=extra_info,
            from_square=from_square,
            to_square=to_square,
            piece_moved=piece_moved.symbol() if piece_moved else None,
            piece_captured=piece_captured.symbol() if piece_captured else None,
            promotion_piece=promotion_piece,
            is_check=is_check,
            is_checkmate=is_checkmate,
            is_legal_in_position=is_legal,
        )

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
        issues: list[VerificationIssue] = []
        extra_info: dict[str, Any] = {"castling_type": "kingside" if kingside else "queenside"}

        # Determine the castling move using centralized constants
        # Key is (is_white, is_kingside)
        is_white = state.board.turn == chess.WHITE
        move_uci = CASTLING_MOVES[(is_white, kingside)]
        move = chess.Move.from_uci(move_uci)

        # Check if castling rights exist
        if kingside:
            has_rights = bool(
                state.board.castling_rights & (chess.BB_H1 if state.board.turn == chess.WHITE else chess.BB_H8)
            )
        else:
            has_rights = bool(
                state.board.castling_rights & (chess.BB_A1 if state.board.turn == chess.WHITE else chess.BB_A8)
            )

        if not has_rights:
            issues.append(
                VerificationIssue(
                    code="NO_CASTLING_RIGHTS",
                    message="Castling rights not available",
                    severity=VerificationSeverity.ERROR,
                    context=extra_info,
                )
            )

        # Check if castling is legal
        is_legal = move in state.board.legal_moves

        if not is_legal and has_rights:
            # Determine why castling is blocked
            if state.board.is_check():
                issues.append(
                    VerificationIssue(
                        code="CASTLING_IN_CHECK",
                        message="Cannot castle while in check",
                        severity=VerificationSeverity.ERROR,
                        context=extra_info,
                    )
                )
            else:
                issues.append(
                    VerificationIssue(
                        code="CASTLING_BLOCKED",
                        message="Castling is blocked (pieces in the way or squares attacked)",
                        severity=VerificationSeverity.ERROR,
                        context=extra_info,
                    )
                )

        move_type = MoveType.CASTLE_KINGSIDE if kingside else MoveType.CASTLE_QUEENSIDE

        return MoveValidationResult(
            is_valid=is_legal,
            move_uci=move_uci,
            move_type=move_type,
            issues=issues,
            extra_info=extra_info,
            is_legal_in_position=is_legal,
        )

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
        extra_info: dict[str, Any] = {"move_type": "en_passant"}

        # Parse the move using centralized helper
        move, issues = self._parse_uci_move(move_uci, MoveType.EN_PASSANT)
        if move is None:
            return MoveValidationResult(
                is_valid=False,
                move_uci=move_uci,
                move_type=MoveType.EN_PASSANT,
                issues=issues,
            )

        # Check if this is actually an en passant move
        is_ep = state.board.is_en_passant(move)
        if not is_ep:
            issues.append(
                VerificationIssue(
                    code="NOT_EN_PASSANT",
                    message="Move is not an en passant capture",
                    severity=VerificationSeverity.WARNING,
                    context={"move": move_uci, "ep_square": str(state.board.ep_square)},
                )
            )

        # Check en passant square
        if state.board.ep_square is None:
            issues.append(
                VerificationIssue(
                    code="NO_EP_SQUARE",
                    message="No en passant square available",
                    severity=VerificationSeverity.ERROR,
                )
            )

        # Check if move is legal
        is_legal = move in state.board.legal_moves

        extra_info["ep_square"] = chess.square_name(state.board.ep_square) if state.board.ep_square else None

        return MoveValidationResult(
            is_valid=is_legal and is_ep,
            move_uci=move_uci,
            move_type=MoveType.EN_PASSANT,
            issues=issues,
            extra_info=extra_info,
            is_legal_in_position=is_legal,
        )

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
        extra_info: dict[str, Any] = {"move_type": "promotion"}

        # Parse the move using centralized helper
        move, issues = self._parse_uci_move(move_uci, MoveType.PROMOTION)
        if move is None:
            return MoveValidationResult(
                is_valid=False,
                move_uci=move_uci,
                move_type=MoveType.PROMOTION,
                issues=issues,
            )

        # Check if this is a promotion move
        if move.promotion is None:
            issues.append(
                VerificationIssue(
                    code="NOT_PROMOTION",
                    message="Move does not specify a promotion piece",
                    severity=VerificationSeverity.ERROR,
                    context={"move": move_uci},
                )
            )

        # Validate promotion piece
        if move.promotion is not None:
            valid_promotions = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            if move.promotion not in valid_promotions:
                issues.append(
                    VerificationIssue(
                        code="INVALID_PROMOTION_PIECE",
                        message=f"Invalid promotion piece: {move.promotion}",
                        severity=VerificationSeverity.ERROR,
                    )
                )
            extra_info["promotion_piece"] = chess.piece_name(move.promotion)

        # Check if move is legal
        is_legal = move in state.board.legal_moves

        # Check if source square has a pawn
        piece = state.board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            issues.append(
                VerificationIssue(
                    code="NOT_PAWN",
                    message="Source square does not contain a pawn",
                    severity=VerificationSeverity.ERROR,
                )
            )

        # Check if move captures
        captures = state.board.piece_at(move.to_square) is not None
        move_type = MoveType.PROMOTION_CAPTURE if captures else MoveType.PROMOTION

        return MoveValidationResult(
            is_valid=is_legal,
            move_uci=move_uci,
            move_type=move_type,
            issues=issues,
            extra_info=extra_info,
            promotion_piece=chess.piece_name(move.promotion) if move.promotion else None,
            is_legal_in_position=is_legal,
        )

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
        issues: list[VerificationIssue] = []
        encoding_result = self._validate_encoding(state, move_uci)

        if not encoding_result.get("roundtrip_valid", True):
            issues.append(
                VerificationIssue(
                    code="ENCODING_ROUNDTRIP_FAILED",
                    message=encoding_result.get("error", "Unknown encoding error"),
                    severity=VerificationSeverity.ERROR,
                    context=encoding_result,
                )
            )

        return MoveValidationResult(
            is_valid=encoding_result.get("roundtrip_valid", False),
            move_uci=move_uci,
            move_type=self._determine_move_type(
                state.board, self._parse_move_safe(move_uci) or state.board.parse_uci(move_uci)
            ),
            encoded_index=encoding_result.get("encoded_index"),
            issues=issues,
            extra_info={"encoding": encoding_result},
        )

    def validate_all_legal_moves(
        self,
        state: ChessGameState,
    ) -> list[MoveValidationResult]:
        """Validate all legal moves in a position.

        Args:
            state: Chess position

        Returns:
            List of MoveValidationResult for each legal move
        """
        results = []
        for move in state.board.legal_moves:
            result = self.validate_move(state, move.uci())
            results.append(result)
        return results

    def _validate_encoding(
        self,
        state: ChessGameState,
        move_uci: str,
    ) -> dict[str, Any]:
        """Validate move encoding roundtrip.

        Args:
            state: Chess position
            move_uci: Move in UCI format

        Returns:
            Dictionary with encoding validation details
        """
        result: dict[str, Any] = {
            "original_move": move_uci,
            "roundtrip_valid": False,
        }

        try:
            # Encode the move
            from_black = state.current_player == -1
            encoded_index = self._encoder.encode_move(move_uci, from_black)
            result["encoded_index"] = encoded_index

            # Decode back
            decoded_move = self._encoder.decode_move(encoded_index, from_black)
            result["decoded_move"] = decoded_move

            # Check if they match
            result["roundtrip_valid"] = decoded_move == move_uci

            if not result["roundtrip_valid"]:
                result["error"] = f"Mismatch: {move_uci} -> {encoded_index} -> {decoded_move}"

        except ValueError as e:
            result["error"] = str(e)
            result["roundtrip_valid"] = False

        return result

    def _determine_move_type(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> MoveType:
        """Determine the type of a chess move.

        Args:
            board: Chess board
            move: Chess move

        Returns:
            MoveType enum value
        """

        # Check for castling
        if board.is_castling(move):
            if board.is_kingside_castling(move):
                return MoveType.CASTLE_KINGSIDE
            return MoveType.CASTLE_QUEENSIDE

        # Check for en passant
        if board.is_en_passant(move):
            return MoveType.EN_PASSANT

        # Check for promotion
        if move.promotion is not None:
            if board.piece_at(move.to_square) is not None:
                return MoveType.PROMOTION_CAPTURE
            return MoveType.PROMOTION

        # Check for capture
        if board.is_capture(move):
            return MoveType.CAPTURE

        # Normal move
        return MoveType.NORMAL

    def _parse_move_safe(self, move_uci: str) -> chess.Move | None:
        """Safely parse a UCI move string.

        Args:
            move_uci: Move in UCI format

        Returns:
            Chess move or None if invalid
        """
        try:
            return chess.Move.from_uci(move_uci)
        except ValueError:
            return None

    def _parse_uci_move(
        self,
        move_uci: str,
        move_type: MoveType = MoveType.NORMAL,
    ) -> tuple[chess.Move | None, list[VerificationIssue]]:
        """Parse a UCI move string with error handling.

        Centralizes UCI parsing logic to avoid code duplication.

        Args:
            move_uci: Move in UCI format
            move_type: Expected move type for error context

        Returns:
            Tuple of (parsed move or None, list of issues)
        """
        issues: list[VerificationIssue] = []
        try:
            move = chess.Move.from_uci(move_uci)
            return move, issues
        except ValueError as e:
            issues.append(
                VerificationIssue(
                    code="INVALID_UCI_FORMAT",
                    message=f"Invalid UCI format: {e}",
                    severity=VerificationSeverity.ERROR,
                    context={"move": move_uci, "expected_type": move_type.value},
                )
            )
            return None, issues


def create_move_validator(
    action_config: ChessActionSpaceConfig | None = None,
    validator_config: MoveValidatorConfig | None = None,
) -> MoveValidator:
    """Factory function to create a MoveValidator.

    Args:
        action_config: Optional action space configuration
        validator_config: Optional validator configuration

    Returns:
        Configured MoveValidator instance
    """
    encoder = ChessActionEncoder(action_config or ChessActionSpaceConfig())
    return MoveValidator(
        action_encoder=encoder,
        config=validator_config,
    )
