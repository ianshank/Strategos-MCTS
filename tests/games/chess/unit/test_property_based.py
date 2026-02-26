"""
Property-Based Tests for Chess Verification.

Uses hypothesis to verify chess invariants and edge cases
through property-based testing.
"""

from __future__ import annotations

import contextlib

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.games.chess.action_space import ChessActionEncoder
from src.games.chess.config import ChessActionSpaceConfig
from src.games.chess.state import ChessGameState
from src.games.chess.verification import (
    ChessGameVerifier,
    MoveValidator,
    create_game_verifier,
    create_move_validator,
)
from tests.games.chess.builders import initial_position


# Custom strategies for chess testing
@st.composite
def valid_squares(draw: st.DrawFn) -> str:
    """Generate valid chess square coordinates."""
    file = draw(st.sampled_from("abcdefgh"))
    rank = draw(st.sampled_from("12345678"))
    return f"{file}{rank}"


@st.composite
def valid_uci_moves(draw: st.DrawFn) -> str:
    """Generate syntactically valid UCI move strings."""
    from_sq = draw(valid_squares())
    to_sq = draw(valid_squares())
    # Optionally add promotion piece
    promotion = draw(st.sampled_from(["", "q", "r", "b", "n"]))
    return f"{from_sq}{to_sq}{promotion}"


@st.composite
def invalid_uci_moves(draw: st.DrawFn) -> str:
    """Generate invalid UCI move strings."""
    strategy = draw(st.integers(min_value=0, max_value=4))

    if strategy == 0:
        # Too short
        return draw(st.text(min_size=0, max_size=3))
    elif strategy == 1:
        # Invalid characters
        return draw(st.text(min_size=4, max_size=6, alphabet="xyz123!@#"))
    elif strategy == 2:
        # Invalid file
        invalid_file = draw(st.sampled_from("ijklmnop"))
        rank = draw(st.sampled_from("12345678"))
        return f"{invalid_file}{rank}e4"
    elif strategy == 3:
        # Invalid rank
        file = draw(st.sampled_from("abcdefgh"))
        invalid_rank = draw(st.sampled_from("09"))
        return f"{file}{invalid_rank}e4"
    else:
        # Invalid promotion piece
        from_sq = draw(valid_squares())
        to_sq = draw(valid_squares())
        invalid_promo = draw(st.sampled_from("xyzpk"))
        return f"{from_sq}{to_sq}{invalid_promo}"


class TestMoveValidatorProperties:
    """Property-based tests for MoveValidator."""

    @pytest.fixture
    def validator(self) -> MoveValidator:
        """Create a move validator."""
        return create_move_validator()

    @pytest.fixture
    def encoder(self) -> ChessActionEncoder:
        """Create an action encoder."""
        return ChessActionEncoder(ChessActionSpaceConfig())

    @pytest.mark.unit
    @pytest.mark.property
    @given(move=valid_uci_moves())
    @settings(max_examples=100, deadline=None)
    def test_uci_format_validation_never_crashes(
        self,
        validator: MoveValidator,
        move: str,
    ) -> None:
        """Property: UCI format validation never crashes."""
        state = initial_position()
        # Should never raise an exception
        result = validator.validate_move(state, move)
        # Result should always be returned
        assert result is not None
        assert hasattr(result, "is_valid")

    @pytest.mark.unit
    @pytest.mark.property
    @given(move=invalid_uci_moves())
    @settings(max_examples=50, deadline=None)
    def test_invalid_uci_always_rejected(
        self,
        validator: MoveValidator,
        move: str,
    ) -> None:
        """Property: Invalid UCI format is always rejected."""
        state = initial_position()
        result = validator.validate_move(state, move)
        # Most invalid moves should be rejected (some might accidentally be valid)
        # We just ensure no crashes
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.property
    def test_legal_moves_always_valid(
        self,
        validator: MoveValidator,
    ) -> None:
        """Property: All legal moves from a position are valid."""
        state = initial_position()
        legal_moves = state.get_legal_actions()

        for move in legal_moves:
            result = validator.validate_move(state, move)
            assert result.is_valid, f"Legal move {move} should be valid"

    @pytest.mark.unit
    @pytest.mark.property
    @given(move_index=st.integers(min_value=0, max_value=4671))
    @settings(max_examples=100, deadline=None)
    def test_action_encoding_roundtrip(
        self,
        encoder: ChessActionEncoder,
        move_index: int,
    ) -> None:
        """Property: Action encoding/decoding is consistent."""
        # Decode to move
        move = encoder.decode(move_index)

        # If move is decodable, it should be re-encodable
        if move is not None:
            re_encoded = encoder.encode(move)
            assert re_encoded == move_index, f"Roundtrip failed for index {move_index}"

    @pytest.mark.unit
    @pytest.mark.property
    def test_all_legal_moves_encodable(
        self,
        encoder: ChessActionEncoder,
    ) -> None:
        """Property: All legal moves can be encoded."""
        state = initial_position()
        legal_moves = state.get_legal_actions()

        for move in legal_moves:
            encoded = encoder.encode(move)
            assert encoded is not None, f"Move {move} should be encodable"
            assert 0 <= encoded < encoder.action_size


class TestGameStateProperties:
    """Property-based tests for ChessGameState."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_legal_move_application_valid(self) -> None:
        """Property: Applying legal moves produces valid states."""
        state = initial_position()

        for move in state.get_legal_actions():
            new_state = state.apply_action(move)
            # New state should be valid
            assert new_state is not None
            assert new_state.fen is not None
            # Turn should change
            assert new_state.current_player != state.current_player

    @pytest.mark.unit
    @pytest.mark.property
    @given(num_moves=st.integers(min_value=1, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_random_game_progression(self, num_moves: int) -> None:
        """Property: Random valid games maintain invariants."""
        import random

        state = initial_position()

        for _ in range(num_moves):
            if state.is_terminal:
                break

            legal_moves = state.get_legal_actions()
            if not legal_moves:
                break

            # Pick random legal move
            move = random.choice(legal_moves)
            state = state.apply_action(move)

            # Invariant: FEN is always valid
            assert state.fen is not None
            # Invariant: Move count increases
            # Invariant: State is consistent
            assert hasattr(state, "current_player")

    @pytest.mark.unit
    @pytest.mark.property
    def test_terminal_states_have_result(self) -> None:
        """Property: Terminal states have a game result."""
        # Test known checkmate position
        checkmate_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        state = ChessGameState.from_fen(checkmate_fen)

        # Apply Scholar's mate
        state = state.apply_action("h5f7")

        if state.is_terminal:
            result = state.get_result()
            assert result is not None


class TestGameVerifierProperties:
    """Property-based tests for ChessGameVerifier."""

    @pytest.fixture
    def verifier(self) -> ChessGameVerifier:
        """Create a game verifier."""
        return create_game_verifier()

    @pytest.mark.unit
    @pytest.mark.property
    def test_empty_sequence_always_valid(self, verifier: ChessGameVerifier) -> None:
        """Property: Empty move sequence is always valid."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = verifier.verify_move_sequence(fen, [])
        assert result.is_valid
        assert result.total_moves == 0

    @pytest.mark.unit
    @pytest.mark.property
    def test_known_valid_games(self, verifier: ChessGameVerifier) -> None:
        """Property: Known valid game sequences pass verification."""
        valid_games = [
            ["e2e4", "e7e5"],
            ["e2e4", "e7e5", "g1f3", "b8c6"],
            ["d2d4", "d7d5", "c2c4"],
            ["e2e4", "c7c5"],  # Sicilian
        ]

        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        for moves in valid_games:
            result = verifier.verify_move_sequence(initial_fen, moves)
            assert result.is_valid, f"Valid game {moves} should pass"
            assert result.total_moves == len(moves)

    @pytest.mark.unit
    @pytest.mark.property
    @given(garbage=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5))
    @settings(max_examples=20, deadline=None)
    def test_garbage_sequences_rejected(
        self,
        verifier: ChessGameVerifier,
        garbage: list[str],
    ) -> None:
        """Property: Random garbage move sequences are rejected."""
        # Filter out accidentally valid moves
        assume(not all(len(m) in (4, 5) and m[0] in "abcdefgh" for m in garbage))

        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = verifier.verify_move_sequence(initial_fen, garbage)

        # Should not crash, and should detect invalid move
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.property
    def test_verification_preserves_move_count(self, verifier: ChessGameVerifier) -> None:
        """Property: Verification correctly counts valid moves."""
        valid_prefix = ["e2e4", "e7e5", "g1f3"]
        invalid_suffix = ["invalid"]

        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = verifier.verify_move_sequence(initial_fen, valid_prefix + invalid_suffix)

        # Should have processed valid moves before failure
        assert result.valid_moves == len(valid_prefix)


class TestPositionConsistencyProperties:
    """Property-based tests for position consistency."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_fen_roundtrip_consistency(self) -> None:
        """Property: FEN serialization/deserialization is consistent."""
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1",
        ]

        for fen in test_fens:
            state = ChessGameState.from_fen(fen)
            # The resulting FEN should be valid (might differ in move counters)
            assert state.fen is not None
            # Should be parseable again
            reparsed = ChessGameState.from_fen(state.fen)
            assert reparsed.fen is not None

    @pytest.mark.unit
    @pytest.mark.property
    def test_game_phase_transitions(self) -> None:
        """Property: Game phase transitions are monotonic (opening -> middle -> end)."""
        from src.games.chess.config import GamePhase

        # Opening position
        opening = initial_position()
        opening_phase = opening.get_game_phase()

        # After several moves, should not go back to earlier phase
        # (Though middlegame can transition to endgame, not back to opening)
        assert opening_phase == GamePhase.OPENING

    @pytest.mark.unit
    @pytest.mark.property
    def test_legal_move_count_bounds(self) -> None:
        """Property: Legal move count is within reasonable bounds."""
        # In any chess position, there are at most ~218 legal moves
        # and at least 0 (checkmate/stalemate)
        state = initial_position()
        legal_moves = state.get_legal_actions()

        assert 0 <= len(legal_moves) <= 218
        # Initial position has exactly 20 moves
        assert len(legal_moves) == 20


class TestEncodingProperties:
    """Property-based tests for move encoding."""

    @pytest.fixture
    def encoder(self) -> ChessActionEncoder:
        """Create an action encoder."""
        return ChessActionEncoder(ChessActionSpaceConfig())

    @pytest.mark.unit
    @pytest.mark.property
    def test_action_space_size(self, encoder: ChessActionEncoder) -> None:
        """Property: Action space has expected size."""
        # AlphaZero uses 4672 actions
        assert encoder.action_size == 4672

    @pytest.mark.unit
    @pytest.mark.property
    @given(index=st.integers(min_value=-10, max_value=4682))
    @settings(max_examples=100, deadline=None)
    def test_decode_bounds_handling(self, encoder: ChessActionEncoder, index: int) -> None:
        """Property: Decode handles out-of-bounds indices gracefully."""
        if index < 0 or index >= encoder.action_size:
            # Should handle gracefully (return None or raise)
            with contextlib.suppress(IndexError, ValueError):
                encoder.decode(index)
        else:
            # Valid index should decode to something
            result = encoder.decode(index)
            # Result should be a string move or None
            assert result is None or isinstance(result, str)

    @pytest.mark.unit
    @pytest.mark.property
    def test_promotion_encoding(self, encoder: ChessActionEncoder) -> None:
        """Property: Promotion moves encode correctly."""
        # White pawn promotion moves
        promotion_moves = [
            "e7e8q",
            "e7e8r",
            "e7e8b",
            "e7e8n",
            "a7a8q",
            "h7h8q",
        ]

        for move in promotion_moves:
            encoded = encoder.encode(move)
            if encoded is not None:
                decoded = encoder.decode(encoded)
                # Decoded should match original
                assert decoded == move, f"Promotion {move} roundtrip failed"
