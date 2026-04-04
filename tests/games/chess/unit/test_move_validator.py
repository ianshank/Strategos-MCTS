"""
Unit Tests for MoveValidator.

Tests move validation functionality including:
- Basic move validation
- Castling edge cases
- En passant edge cases
- Promotion edge cases
- Encoding roundtrip validation
"""

from __future__ import annotations

import pytest

from src.games.chess.state import ChessGameState
from src.games.chess.verification import (
    MoveType,
    MoveValidator,
    MoveValidatorConfig,
    create_move_validator,
)
from tests.games.chess.builders import (
    ChessPositionBuilder,
    initial_position,
)
from tests.games.chess.fixtures import (
    CASTLING_EDGE_CASES,
    EN_PASSANT_EDGE_CASES,
    PROMOTION_EDGE_CASES,
)


class TestMoveValidator:
    """Unit tests for MoveValidator."""

    @pytest.fixture
    def validator(self) -> MoveValidator:
        """Create a move validator for testing."""
        return create_move_validator()

    @pytest.fixture
    def initial_state(self) -> ChessGameState:
        """Create initial position for testing."""
        return initial_position()

    @pytest.mark.unit
    def test_validate_legal_move(self, validator: MoveValidator, initial_state: ChessGameState) -> None:
        """Test that legal moves are validated correctly."""
        result = validator.validate_move(initial_state, "e2e4")

        assert result.is_valid is True
        assert result.move_uci == "e2e4"
        assert result.move_type == MoveType.NORMAL
        assert result.is_legal_in_position is True
        assert not result.has_errors

    @pytest.mark.unit
    def test_validate_illegal_move(self, validator: MoveValidator, initial_state: ChessGameState) -> None:
        """Test that illegal moves are flagged correctly."""
        result = validator.validate_move(initial_state, "e1e2")

        assert result.is_valid is False
        assert result.is_legal_in_position is False
        assert any(issue.code == "ILLEGAL_MOVE" for issue in result.issues)

    @pytest.mark.unit
    def test_validate_invalid_uci_format(self, validator: MoveValidator, initial_state: ChessGameState) -> None:
        """Test that invalid UCI format is handled."""
        result = validator.validate_move(initial_state, "invalid")

        assert result.is_valid is False
        assert any(issue.code == "INVALID_UCI_FORMAT" for issue in result.issues)

    @pytest.mark.unit
    def test_validate_capture_move(self, validator: MoveValidator) -> None:
        """Test that capture moves are identified correctly."""
        # Position with a capture available
        state = ChessPositionBuilder().with_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2").build()

        result = validator.validate_move(state, "e4d5")

        assert result.is_valid is True
        assert result.move_type == MoveType.CAPTURE
        assert result.piece_captured is not None

    @pytest.mark.unit
    def test_validate_check_move(self, validator: MoveValidator) -> None:
        """Test that check moves are identified correctly."""
        # Position where Qh5 gives check
        state = (
            ChessPositionBuilder().with_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3").build()
        )

        result = validator.validate_move(state, "d1h5")

        assert result.is_valid is True
        assert result.is_check is True

    @pytest.mark.unit
    def test_validate_checkmate_move(self, validator: MoveValidator) -> None:
        """Test that checkmate moves are identified correctly."""
        # Scholar's mate position
        state = (
            ChessPositionBuilder()
            .with_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
            .build()
        )

        result = validator.validate_move(state, "h5f7")

        assert result.is_valid is True
        assert result.is_checkmate is True

    @pytest.mark.unit
    def test_validate_move_details(self, validator: MoveValidator, initial_state: ChessGameState) -> None:
        """Test that move details are extracted correctly."""
        result = validator.validate_move(initial_state, "g1f3")

        assert result.from_square == "g1"
        assert result.to_square == "f3"
        assert result.piece_moved == "N"  # Knight

    @pytest.mark.unit
    def test_encoding_roundtrip_success(self, validator: MoveValidator, initial_state: ChessGameState) -> None:
        """Test that encoding roundtrip succeeds for valid moves."""
        result = validator.validate_encoding_roundtrip(initial_state, "e2e4")

        assert result.is_valid is True
        assert result.encoded_index is not None
        assert "encoding" in result.extra_info

    @pytest.mark.unit
    def test_validate_all_legal_moves(self, validator: MoveValidator, initial_state: ChessGameState) -> None:
        """Test validation of all legal moves in a position."""
        results = validator.validate_all_legal_moves(initial_state)

        # Initial position has 20 legal moves
        assert len(results) == 20
        assert all(r.is_valid for r in results)

    @pytest.mark.unit
    @pytest.mark.parametrize("case", CASTLING_EDGE_CASES, ids=lambda c: c["name"])
    def test_castling_validation(self, validator: MoveValidator, case: dict) -> None:
        """Test castling move validation for various edge cases."""
        state = ChessPositionBuilder().with_fen(case["fen"]).build()

        # Use specific castling validation for castling moves
        if "castling_type" in case:
            kingside = case["castling_type"] == "kingside"
            result = validator.validate_castling(state, kingside)
        else:
            result = validator.validate_move(state, case["move"])

        assert (
            result.is_valid == case["should_be_legal"]
        ), f"Expected is_valid={case['should_be_legal']} for {case['name']}, got {result.is_valid}"

    @pytest.mark.unit
    @pytest.mark.parametrize("case", EN_PASSANT_EDGE_CASES, ids=lambda c: c["name"])
    def test_en_passant_validation(self, validator: MoveValidator, case: dict) -> None:
        """Test en passant move validation for various edge cases."""
        state = ChessPositionBuilder().with_fen(case["fen"]).build()

        if case.get("is_en_passant", False):
            result = validator.validate_en_passant(state, case["move"])
        else:
            result = validator.validate_move(state, case["move"])

        assert (
            result.is_valid == case["should_be_legal"]
        ), f"Expected is_valid={case['should_be_legal']} for {case['name']}, got {result.is_valid}"

    @pytest.mark.unit
    @pytest.mark.parametrize("case", PROMOTION_EDGE_CASES, ids=lambda c: c["name"])
    def test_promotion_validation(self, validator: MoveValidator, case: dict) -> None:
        """Test promotion move validation for various edge cases."""
        state = ChessPositionBuilder().with_fen(case["fen"]).build()

        result = validator.validate_promotion(state, case["move"])

        assert (
            result.is_valid == case["should_be_legal"]
        ), f"Expected is_valid={case['should_be_legal']} for {case['name']}, got {result.is_valid}"

        if result.is_valid and "promotion_piece" in case:
            assert result.promotion_piece is not None

    @pytest.mark.unit
    def test_config_disable_encoding_validation(self, initial_state: ChessGameState) -> None:
        """Test that encoding validation can be disabled."""
        config = MoveValidatorConfig(validate_encoding=False)
        validator = MoveValidator(config=config)

        result = validator.validate_move(initial_state, "e2e4")

        assert result.is_valid is True
        # Should not have encoding info
        assert result.encoded_index is None or "encoding" not in result.extra_info

    @pytest.mark.unit
    def test_config_log_validations(self, initial_state: ChessGameState) -> None:
        """Test that logging can be enabled."""
        config = MoveValidatorConfig(log_validations=True)
        validator = MoveValidator(config=config)

        # Should not raise
        result = validator.validate_move(initial_state, "e2e4")
        assert result.is_valid is True

    @pytest.mark.unit
    def test_move_result_to_dict(self, validator: MoveValidator, initial_state: ChessGameState) -> None:
        """Test MoveValidationResult serialization."""
        result = validator.validate_move(initial_state, "e2e4")
        result_dict = result.to_dict()

        assert "is_valid" in result_dict
        assert "move_uci" in result_dict
        assert "move_type" in result_dict
        assert result_dict["move_uci"] == "e2e4"

    @pytest.mark.unit
    def test_factory_function(self) -> None:
        """Test the create_move_validator factory function."""
        validator = create_move_validator()

        assert isinstance(validator, MoveValidator)
        assert validator.config is not None
        assert validator.encoder is not None


class TestMoveValidatorEdgeCases:
    """Edge case tests for MoveValidator."""

    @pytest.fixture
    def validator(self) -> MoveValidator:
        """Create a move validator for testing."""
        return create_move_validator()

    @pytest.mark.unit
    def test_empty_move_string(self, validator: MoveValidator) -> None:
        """Test handling of empty move string."""
        state = initial_position()
        result = validator.validate_move(state, "")

        assert result.is_valid is False

    @pytest.mark.unit
    def test_terminal_position(self, validator: MoveValidator) -> None:
        """Test validation in a terminal position."""
        # Checkmate position
        state = ChessPositionBuilder().with_checkmate().build()
        results = validator.validate_all_legal_moves(state)

        # No legal moves in checkmate
        assert len(results) == 0

    @pytest.mark.unit
    def test_stalemate_position(self, validator: MoveValidator) -> None:
        """Test validation in a stalemate position."""
        state = ChessPositionBuilder().with_stalemate().build()
        results = validator.validate_all_legal_moves(state)

        # No legal moves in stalemate
        assert len(results) == 0

    @pytest.mark.unit
    def test_long_algebraic_notation(self, validator: MoveValidator) -> None:
        """Test handling of various move formats."""
        state = initial_position()

        # Standard UCI
        result = validator.validate_move(state, "e2e4")
        assert result.is_valid is True

        # With promotion
        promo_state = ChessPositionBuilder().with_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").build()
        result = validator.validate_move(promo_state, "e7e8q")
        assert result.is_valid is True
