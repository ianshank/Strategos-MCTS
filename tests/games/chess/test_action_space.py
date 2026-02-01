"""Unit tests for chess action space encoder/decoder."""

from __future__ import annotations

import numpy as np
import pytest

from src.games.chess.action_space import (
    ChessActionEncoder,
    KnightMove,
    MoveDirection,
    PromotionDirection,
    PromotionPiece,
)


class TestChessActionEncoder:
    """Tests for ChessActionEncoder."""

    @pytest.fixture
    def encoder(self) -> ChessActionEncoder:
        """Create encoder fixture."""
        return ChessActionEncoder()

    def test_initialization(self, encoder: ChessActionEncoder) -> None:
        """Test encoder initialization."""
        assert encoder.action_size == 4672  # 73 * 64
        assert encoder._knight_plane_offset == 56
        assert encoder._promotion_plane_offset == 64

    def test_encode_simple_pawn_move(self, encoder: ChessActionEncoder) -> None:
        """Test encoding a simple pawn move."""
        # e2-e4 (1 square forward then 2 squares = pawn double push)
        action_idx = encoder.encode_move("e2e4")
        assert 0 <= action_idx < encoder.action_size
        # Decode and verify
        decoded = encoder.decode_move(action_idx)
        assert decoded == "e2e4"

    def test_encode_knight_move(self, encoder: ChessActionEncoder) -> None:
        """Test encoding knight moves."""
        # Ng1-f3
        action_idx = encoder.encode_move("g1f3")
        assert 0 <= action_idx < encoder.action_size
        decoded = encoder.decode_move(action_idx)
        assert decoded == "g1f3"

    def test_encode_bishop_move(self, encoder: ChessActionEncoder) -> None:
        """Test encoding bishop (diagonal) moves."""
        # Bf1-c4
        action_idx = encoder.encode_move("f1c4")
        assert 0 <= action_idx < encoder.action_size
        decoded = encoder.decode_move(action_idx)
        assert decoded == "f1c4"

    def test_encode_rook_move(self, encoder: ChessActionEncoder) -> None:
        """Test encoding rook moves."""
        # Ra1-a8 (vertical)
        action_idx = encoder.encode_move("a1a8")
        assert 0 <= action_idx < encoder.action_size
        decoded = encoder.decode_move(action_idx)
        assert decoded == "a1a8"

    def test_encode_queen_move(self, encoder: ChessActionEncoder) -> None:
        """Test encoding queen moves."""
        # Qd1-h5 (diagonal)
        action_idx = encoder.encode_move("d1h5")
        assert 0 <= action_idx < encoder.action_size
        decoded = encoder.decode_move(action_idx)
        assert decoded == "d1h5"

    def test_encode_underpromotion(self, encoder: ChessActionEncoder) -> None:
        """Test encoding underpromotion moves."""
        # e7-e8=n (knight promotion)
        action_idx = encoder.encode_move("e7e8n")
        assert 0 <= action_idx < encoder.action_size
        decoded = encoder.decode_move(action_idx)
        assert decoded == "e7e8n"

    def test_encode_decode_roundtrip(self, encoder: ChessActionEncoder) -> None:
        """Test that encode-decode is a roundtrip for various moves."""
        test_moves = [
            "e2e4",  # Pawn double push
            "g1f3",  # Knight move
            "f1b5",  # Bishop diagonal
            "a1a4",  # Rook vertical
            "d1h5",  # Queen diagonal
            "e1g1",  # King move (castling squares)
        ]
        for move in test_moves:
            action_idx = encoder.encode_move(move)
            decoded = encoder.decode_move(action_idx)
            assert decoded == move, f"Roundtrip failed for {move}"

    def test_encode_with_black_perspective(self, encoder: ChessActionEncoder) -> None:
        """Test encoding from black's perspective."""
        # Same move encoded from different perspectives should differ
        white_idx = encoder.encode_move("e2e4", from_black_perspective=False)
        black_idx = encoder.encode_move("e2e4", from_black_perspective=True)
        assert white_idx != black_idx

    def test_decode_invalid_index_negative(self, encoder: ChessActionEncoder) -> None:
        """Test decoding negative index raises error."""
        with pytest.raises(ValueError, match="out of range"):
            encoder.decode_move(-1)

    def test_decode_invalid_index_too_large(self, encoder: ChessActionEncoder) -> None:
        """Test decoding too large index raises error."""
        with pytest.raises(ValueError, match="out of range"):
            encoder.decode_move(encoder.action_size)

    def test_encode_batch(self, encoder: ChessActionEncoder) -> None:
        """Test batch encoding."""
        moves = ["e2e4", "d2d4", "g1f3"]
        indices = encoder.encode_moves_batch(moves)
        assert len(indices) == 3
        assert all(0 <= idx < encoder.action_size for idx in indices)

    def test_get_action_info(self, encoder: ChessActionEncoder) -> None:
        """Test getting action information."""
        action_idx = encoder.encode_move("e2e4")
        info = encoder.get_action_info(action_idx)
        assert info is not None
        assert "from" in info
        assert "to" in info
        assert "type" in info

    def test_get_legal_action_mask(self, encoder: ChessActionEncoder) -> None:
        """Test getting legal action mask."""
        pytest.importorskip("chess")
        import chess

        board = chess.Board()
        mask = encoder.get_legal_action_mask(board)
        assert mask.shape == (encoder.action_size,)
        assert mask.dtype == np.bool_
        # Initial position has 20 legal moves
        assert mask.sum() == 20

    def test_filter_policy_to_legal(self, encoder: ChessActionEncoder) -> None:
        """Test filtering policy to legal moves."""
        pytest.importorskip("chess")
        import chess

        board = chess.Board()
        policy = np.random.randn(encoder.action_size)

        move_probs = encoder.filter_policy_to_legal(policy, board)

        assert len(move_probs) == 20  # 20 legal moves in starting position
        assert abs(sum(move_probs.values()) - 1.0) < 1e-6  # Should sum to 1

    def test_filter_policy_with_temperature(self, encoder: ChessActionEncoder) -> None:
        """Test policy filtering with different temperatures."""
        pytest.importorskip("chess")
        import chess

        board = chess.Board()
        policy = np.random.randn(encoder.action_size)

        # Low temperature should be more deterministic
        probs_low = encoder.filter_policy_to_legal(policy, board, temperature=0.1)
        probs_high = encoder.filter_policy_to_legal(policy, board, temperature=2.0)

        # Higher temperature should have more entropy
        entropy_low = -sum(p * np.log(p + 1e-10) for p in probs_low.values())
        entropy_high = -sum(p * np.log(p + 1e-10) for p in probs_high.values())
        assert entropy_high > entropy_low

    def test_repr(self, encoder: ChessActionEncoder) -> None:
        """Test string representation."""
        repr_str = repr(encoder)
        assert "ChessActionEncoder" in repr_str
        assert "action_size=4672" in repr_str


class TestMoveDirections:
    """Tests for move direction enums."""

    def test_queen_directions(self) -> None:
        """Test queen move directions."""
        assert len(MoveDirection) == 8
        assert MoveDirection.N.value == 0
        assert MoveDirection.NW.value == 7

    def test_knight_moves(self) -> None:
        """Test knight move types."""
        assert len(KnightMove) == 8
        assert KnightMove.NNE.value == 0
        assert KnightMove.NNW.value == 7

    def test_promotion_directions(self) -> None:
        """Test promotion directions."""
        assert len(PromotionDirection) == 3
        assert PromotionDirection.LEFT.value == 0
        assert PromotionDirection.STRAIGHT.value == 1
        assert PromotionDirection.RIGHT.value == 2

    def test_promotion_pieces(self) -> None:
        """Test promotion piece types."""
        assert len(PromotionPiece) == 3
        assert PromotionPiece.KNIGHT.value == 0
        assert PromotionPiece.BISHOP.value == 1
        assert PromotionPiece.ROOK.value == 2
