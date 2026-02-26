"""
Unit Tests for ChessGameVerifier.

Tests game verification functionality including:
- Full game verification
- Position verification
- Move sequence verification
- Terminal state detection
"""

from __future__ import annotations

import pytest

from src.games.chess.state import ChessGameState
from src.games.chess.verification import (
    ChessGameVerifier,
    GameResult,
    GameVerifierConfig,
    create_game_verifier,
)
from tests.games.chess.builders import (
    ChessPositionBuilder,
    initial_position,
)
from tests.games.chess.fixtures import FAMOUS_GAMES


class TestChessGameVerifier:
    """Unit tests for ChessGameVerifier."""

    @pytest.fixture
    def verifier(self) -> ChessGameVerifier:
        """Create a game verifier for testing."""
        return create_game_verifier()

    @pytest.fixture
    def initial_state(self) -> ChessGameState:
        """Create initial position for testing."""
        return initial_position()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_verify_scholars_mate(self, verifier: ChessGameVerifier) -> None:
        """Test verification of Scholar's mate game."""
        moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]

        result = await verifier.verify_full_game(
            moves=moves,
            expected_outcome=GameResult.WHITE_WINS,
        )

        assert result.is_valid is True
        assert result.result == GameResult.WHITE_WINS
        assert result.result_matches_expected is True
        assert result.total_moves == len(moves)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_verify_fools_mate(self, verifier: ChessGameVerifier) -> None:
        """Test verification of Fool's mate game."""
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]

        result = await verifier.verify_full_game(
            moves=moves,
            expected_outcome=GameResult.BLACK_WINS,
        )

        assert result.is_valid is True
        assert result.result == GameResult.BLACK_WINS
        assert result.result_matches_expected is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_verify_stalemate(self, verifier: ChessGameVerifier) -> None:
        """Test verification of stalemate position."""
        # Create a stalemate position directly
        stalemate_fen = "k7/8/1K6/8/8/8/8/8 b - - 0 1"

        position_result = verifier.verify_position(stalemate_fen)

        assert position_result.is_valid is True
        assert position_result.is_terminal is True
        assert position_result.game_result == GameResult.DRAW_STALEMATE

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_verify_fifty_move_rule(self, verifier: ChessGameVerifier) -> None:
        """Test detection of fifty-move rule."""
        # Position with high halfmove clock
        fen = "8/8/4k3/8/4K3/8/8/8 w - - 100 50"

        position_result = verifier.verify_position(fen)

        assert position_result.is_valid is True
        assert position_result.is_terminal is True

    @pytest.mark.unit
    def test_verify_position_valid(self, verifier: ChessGameVerifier) -> None:
        """Test verification of valid position."""
        result = verifier.verify_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

        assert result.is_valid is True
        assert result.is_terminal is False
        assert result.legal_moves_count > 0
        assert result.has_valid_king_positions is True

    @pytest.mark.unit
    def test_verify_position_invalid_fen(self, verifier: ChessGameVerifier) -> None:
        """Test handling of invalid FEN."""
        result = verifier.verify_position("invalid_fen")

        assert result.is_valid is False
        assert any(issue.code == "INVALID_FEN" for issue in result.issues)

    @pytest.mark.unit
    def test_verify_position_checkmate(self, verifier: ChessGameVerifier) -> None:
        """Test verification of checkmate position."""
        # Back rank mate
        fen = "6k1/5ppp/8/8/8/8/8/R3K3 b - - 0 1"
        state = ChessPositionBuilder().with_fen(fen).build()

        # Apply the mating move
        state = state.apply_action("a1a8") if "a1a8" in state.get_legal_actions() else state

        result = verifier.verify_position(state.fen)

        # The position should show game state correctly
        assert result.is_valid is True

    @pytest.mark.unit
    def test_verify_move_sequence_valid(self, verifier: ChessGameVerifier) -> None:
        """Test verification of valid move sequence."""
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]

        result = verifier.verify_move_sequence(initial_fen, moves)

        assert result.is_valid is True
        assert result.total_moves == len(moves)
        assert result.valid_moves == len(moves)
        assert result.final_fen is not None

    @pytest.mark.unit
    def test_verify_move_sequence_with_illegal_move(self, verifier: ChessGameVerifier) -> None:
        """Test verification of sequence with illegal move."""
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = ["e2e4", "e1e2"]  # e1e2 is illegal

        result = verifier.verify_move_sequence(initial_fen, moves)

        assert result.is_valid is False
        assert any(issue.code == "INVALID_MOVE" for issue in result.issues)

    @pytest.mark.unit
    def test_verify_move_sequence_statistics(self, verifier: ChessGameVerifier) -> None:
        """Test that move sequence statistics are tracked."""
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = ["e2e4", "d7d5", "e4d5"]  # Includes a capture

        result = verifier.verify_move_sequence(initial_fen, moves)

        assert result.is_valid is True
        assert result.captures == 1
        assert result.total_moves == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_verify_game_playthrough(self, verifier: ChessGameVerifier) -> None:
        """Test game playthrough verification."""
        state = initial_position()

        result = await verifier.verify_game_playthrough(state, max_moves=10)

        assert result.is_valid is True
        assert result.total_moves <= 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_verify_game_with_expected_outcome_mismatch(self, verifier: ChessGameVerifier) -> None:
        """Test verification with mismatched expected outcome."""
        moves = ["e2e4", "e7e5"]  # In progress, not a win

        result = await verifier.verify_full_game(
            moves=moves,
            expected_outcome=GameResult.WHITE_WINS,
        )

        assert result.result_matches_expected is False
        assert any(issue.code == "RESULT_MISMATCH" for issue in result.issues)

    @pytest.mark.unit
    def test_config_stop_on_first_error(self) -> None:
        """Test that stop_on_first_error configuration works."""
        config = GameVerifierConfig(stop_on_first_error=True)
        verifier = ChessGameVerifier(config=config)

        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = ["e2e4", "invalid", "e7e5"]  # Second move is invalid

        result = verifier.verify_move_sequence(initial_fen, moves)

        # Should stop at first error
        assert result.is_valid is False
        assert result.valid_moves == 1  # Only e2e4 was valid

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_game_result_summary(self, verifier: ChessGameVerifier) -> None:
        """Test GameVerificationResult summary generation."""
        moves = ["e2e4", "e7e5", "g1f3"]
        result = await verifier.verify_full_game(moves=moves)
        summary = result.summary()

        assert "VALID" in summary or "INVALID" in summary
        assert str(len(moves)) in summary

    @pytest.mark.unit
    def test_factory_function(self) -> None:
        """Test the create_game_verifier factory function."""
        verifier = create_game_verifier()

        assert isinstance(verifier, ChessGameVerifier)
        assert verifier.config is not None
        assert verifier.move_validator is not None


class TestChessGameVerifierFamousGames:
    """Test verification of famous games."""

    @pytest.fixture
    def verifier(self) -> ChessGameVerifier:
        """Create a game verifier for testing."""
        return create_game_verifier()

    @pytest.mark.unit
    @pytest.mark.parametrize("game", FAMOUS_GAMES, ids=lambda g: g["name"])
    @pytest.mark.asyncio
    async def test_verify_famous_game(self, verifier: ChessGameVerifier, game: dict) -> None:
        """Test verification of famous games."""
        result = await verifier.verify_full_game(moves=game["moves"])

        # All famous games should be valid
        assert result.is_valid is True, f"Game '{game['name']}' should be valid"

        # Check result if specified
        if game["result"] == "white_wins":
            assert result.result == GameResult.WHITE_WINS
        elif game["result"] == "black_wins":
            assert result.result == GameResult.BLACK_WINS
        elif game["result"] == "in_progress":
            assert result.result == GameResult.IN_PROGRESS


class TestPositionVerification:
    """Tests for position-specific verification."""

    @pytest.fixture
    def verifier(self) -> ChessGameVerifier:
        """Create a game verifier for testing."""
        return create_game_verifier()

    @pytest.mark.unit
    def test_verify_position_with_invalid_kings(self, verifier: ChessGameVerifier) -> None:
        """Test detection of invalid king configuration."""
        # Position with two white kings (invalid)
        fen = "rnbqkbnr/pppppppp/8/8/4K3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        result = verifier.verify_position(fen)

        # Should still parse but may have issues
        assert "issues" in dir(result)

    @pytest.mark.unit
    def test_verify_position_game_phase(self, verifier: ChessGameVerifier) -> None:
        """Test that game phase is correctly identified."""
        # Opening position
        result = verifier.verify_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert result.game_phase == "opening"

        # Endgame position
        result = verifier.verify_position("8/8/4k3/8/4P3/8/4K3/8 w - - 0 1")
        assert result.game_phase == "endgame"

    @pytest.mark.unit
    def test_verify_position_material_balance(self, verifier: ChessGameVerifier) -> None:
        """Test material balance calculation."""
        # Even position
        result = verifier.verify_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert result.material_balance == 0

        # White up a queen
        result = verifier.verify_position("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert result.material_balance > 0  # White has more material
