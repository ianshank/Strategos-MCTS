"""
Unit tests for chess constants module.

Tests cover constant values, helper functions (truncate_fen, get_fen_truncate_length,
get_piece_values, get_routing_scores, get_stockfish_executables).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestChessConstantValues:
    """Test that all constant values are correct."""

    def test_starting_fen(self) -> None:
        from src.games.chess.constants import STARTING_FEN

        assert STARTING_FEN == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_default_fen_log_truncate_length(self) -> None:
        from src.games.chess.constants import DEFAULT_FEN_LOG_TRUNCATE_LENGTH

        assert DEFAULT_FEN_LOG_TRUNCATE_LENGTH == 40

    def test_invalid_pawn_ranks(self) -> None:
        from src.games.chess.constants import INVALID_PAWN_RANKS

        assert frozenset({0, 7}) == INVALID_PAWN_RANKS
        assert isinstance(INVALID_PAWN_RANKS, frozenset)

    def test_default_stockfish_executables(self) -> None:
        from src.games.chess.constants import DEFAULT_STOCKFISH_EXECUTABLES

        assert isinstance(DEFAULT_STOCKFISH_EXECUTABLES, tuple)
        assert "stockfish" in DEFAULT_STOCKFISH_EXECUTABLES
        assert "stockfish.exe" in DEFAULT_STOCKFISH_EXECUTABLES
        assert len(DEFAULT_STOCKFISH_EXECUTABLES) == 6

    def test_castling_moves(self) -> None:
        from src.games.chess.constants import CASTLING_MOVES

        assert CASTLING_MOVES[(True, True)] == "e1g1"   # White kingside
        assert CASTLING_MOVES[(True, False)] == "e1c1"  # White queenside
        assert CASTLING_MOVES[(False, True)] == "e8g8"  # Black kingside
        assert CASTLING_MOVES[(False, False)] == "e8c8"  # Black queenside

    def test_en_passant_ranks(self) -> None:
        from src.games.chess.constants import EN_PASSANT_RANKS

        assert EN_PASSANT_RANKS[True] == 5   # White captures on rank 6 (0-indexed)
        assert EN_PASSANT_RANKS[False] == 2  # Black captures on rank 3 (0-indexed)

    def test_uuid_short_length(self) -> None:
        from src.games.chess.constants import UUID_SHORT_LENGTH

        assert UUID_SHORT_LENGTH == 8

    def test_default_thresholds(self) -> None:
        from src.games.chess.constants import (
            DEFAULT_AGREEMENT_THRESHOLD,
            DEFAULT_CONFIDENCE,
            DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD,
            DEFAULT_ROUTING_THRESHOLD,
            DEFAULT_VALUE_DIVERGENCE_THRESHOLD,
        )

        assert DEFAULT_AGREEMENT_THRESHOLD == 0.6
        assert DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD == 0.3
        assert DEFAULT_ROUTING_THRESHOLD == 0.5
        assert DEFAULT_VALUE_DIVERGENCE_THRESHOLD == 0.2
        assert DEFAULT_CONFIDENCE == 0.5


@pytest.mark.unit
class TestTruncateFen:
    """Test truncate_fen function."""

    def test_short_fen_not_truncated(self) -> None:
        from src.games.chess.constants import truncate_fen

        short = "abc"
        assert truncate_fen(short, length=10) == "abc"

    def test_exact_length_not_truncated(self) -> None:
        from src.games.chess.constants import truncate_fen

        exact = "a" * 10
        assert truncate_fen(exact, length=10) == exact

    def test_long_fen_truncated(self) -> None:
        from src.games.chess.constants import truncate_fen

        long_fen = "a" * 50
        result = truncate_fen(long_fen, length=10)
        assert result == "a" * 10 + "..."
        assert len(result) == 13

    def test_truncate_with_default_length(self) -> None:
        from src.games.chess.constants import DEFAULT_FEN_LOG_TRUNCATE_LENGTH, truncate_fen

        long_fen = "x" * 100
        # When settings unavailable, falls back to default
        with patch("src.games.chess.constants.get_fen_truncate_length", return_value=DEFAULT_FEN_LOG_TRUNCATE_LENGTH):
            result = truncate_fen(long_fen)
            assert result == "x" * DEFAULT_FEN_LOG_TRUNCATE_LENGTH + "..."

    def test_truncate_with_explicit_length_overrides_settings(self) -> None:
        from src.games.chess.constants import truncate_fen

        fen = "x" * 100
        result = truncate_fen(fen, length=5)
        assert result == "xxxxx..."


@pytest.mark.unit
class TestGetFenTruncateLength:
    """Test get_fen_truncate_length function."""

    def test_with_settings_having_attribute(self) -> None:
        from src.games.chess.constants import get_fen_truncate_length

        mock_settings = MagicMock()
        mock_settings.CHESS_FEN_LOG_TRUNCATE_LENGTH = 60
        assert get_fen_truncate_length(mock_settings) == 60

    def test_with_settings_missing_attribute(self) -> None:
        from src.games.chess.constants import DEFAULT_FEN_LOG_TRUNCATE_LENGTH, get_fen_truncate_length

        mock_settings = MagicMock(spec=[])  # No attributes
        result = get_fen_truncate_length(mock_settings)
        assert result == DEFAULT_FEN_LOG_TRUNCATE_LENGTH

    def test_without_settings_falls_back(self) -> None:
        from src.games.chess.constants import DEFAULT_FEN_LOG_TRUNCATE_LENGTH, get_fen_truncate_length

        with patch("src.config.settings.get_settings", side_effect=RuntimeError("no settings")):
            result = get_fen_truncate_length(None)
            assert result == DEFAULT_FEN_LOG_TRUNCATE_LENGTH


@pytest.mark.unit
class TestGetPieceValues:
    """Test get_piece_values function."""

    def test_defaults_without_settings(self) -> None:
        try:
            import chess
        except ImportError:
            pytest.skip("python-chess not installed")

        from src.games.chess.constants import get_piece_values

        with patch("src.config.settings.get_settings", side_effect=RuntimeError("no settings")):
            values = get_piece_values(None)
            assert len(values) == 5
            assert values[chess.PAWN] == 100
            assert values[chess.QUEEN] == 900

    def test_with_settings(self) -> None:
        try:
            import chess
        except ImportError:
            pytest.skip("python-chess not installed")

        from src.games.chess.constants import get_piece_values

        mock_settings = MagicMock()
        mock_settings.CHESS_PIECE_VALUE_PAWN = 110
        mock_settings.CHESS_PIECE_VALUE_KNIGHT = 330
        mock_settings.CHESS_PIECE_VALUE_BISHOP = 340
        mock_settings.CHESS_PIECE_VALUE_ROOK = 510
        mock_settings.CHESS_PIECE_VALUE_QUEEN = 950
        values = get_piece_values(mock_settings)
        assert values[chess.PAWN] == 110
        assert values[chess.QUEEN] == 950


@pytest.mark.unit
class TestGetRoutingScores:
    """Test get_routing_scores function."""

    def test_defaults_without_settings(self) -> None:
        from src.games.chess.constants import get_routing_scores

        with patch("src.config.settings.get_settings", side_effect=RuntimeError("no settings")):
            scores = get_routing_scores(None)
            assert scores["match"] == 1.0
            assert scores["middlegame_fallback"] == 0.7
            assert scores["phase_appropriate"] == 0.8
            assert scores["phase_mismatch"] == 0.4
            assert scores["default"] == 0.5

    def test_with_settings(self) -> None:
        from src.games.chess.constants import get_routing_scores

        mock_settings = MagicMock()
        mock_settings.CHESS_ROUTING_SCORE_MATCH = 0.9
        mock_settings.CHESS_ROUTING_SCORE_MIDDLEGAME_FALLBACK = 0.6
        mock_settings.CHESS_ROUTING_SCORE_PHASE_APPROPRIATE = 0.7
        mock_settings.CHESS_ROUTING_SCORE_PHASE_MISMATCH = 0.3
        mock_settings.CHESS_ROUTING_SCORE_DEFAULT = 0.4
        scores = get_routing_scores(mock_settings)
        assert scores["match"] == 0.9
        assert scores["default"] == 0.4


@pytest.mark.unit
class TestGetStockfishExecutables:
    """Test get_stockfish_executables function."""

    def test_defaults_without_settings(self) -> None:
        from src.games.chess.constants import DEFAULT_STOCKFISH_EXECUTABLES, get_stockfish_executables

        with patch("src.config.settings.get_settings", side_effect=RuntimeError("no settings")):
            result = get_stockfish_executables(None)
            assert result == DEFAULT_STOCKFISH_EXECUTABLES

    def test_custom_executables_from_settings(self) -> None:
        from src.games.chess.constants import get_stockfish_executables

        mock_settings = MagicMock()
        mock_settings.STOCKFISH_EXECUTABLES = "sf1, sf2, sf3"
        result = get_stockfish_executables(mock_settings)
        assert result == ("sf1", "sf2", "sf3")

    def test_no_custom_executables_returns_defaults(self) -> None:
        from src.games.chess.constants import DEFAULT_STOCKFISH_EXECUTABLES, get_stockfish_executables

        mock_settings = MagicMock(spec=[])  # No STOCKFISH_EXECUTABLES attribute
        result = get_stockfish_executables(mock_settings)
        assert result == DEFAULT_STOCKFISH_EXECUTABLES

    def test_import_error_returns_defaults(self) -> None:
        from src.games.chess.constants import DEFAULT_STOCKFISH_EXECUTABLES, get_stockfish_executables

        with patch("src.config.settings.get_settings", side_effect=ImportError):
            result = get_stockfish_executables(None)
            assert result == DEFAULT_STOCKFISH_EXECUTABLES


@pytest.mark.unit
class TestModuleExports:
    """Test __all__ exports."""

    def test_all_exports(self) -> None:
        from src.games.chess import constants

        for name in constants.__all__:
            assert hasattr(constants, name), f"Missing export: {name}"
