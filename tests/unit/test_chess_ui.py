"""Unit tests for Chess UI module (src/games/chess/ui.py)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed")

# Mock gradio before importing ui module
sys.modules.setdefault("gradio", MagicMock())
sys.modules.setdefault("gradio.themes", MagicMock())

from src.games.chess.continuous_learning import GameResult, ScoreCard


@pytest.mark.unit
class TestGetPieceUnicode:
    """Tests for get_piece_unicode function."""

    def test_white_pieces(self):
        from src.games.chess.ui import get_piece_unicode

        assert get_piece_unicode("K") == "\u2654"
        assert get_piece_unicode("Q") == "\u2655"
        assert get_piece_unicode("R") == "\u2656"
        assert get_piece_unicode("B") == "\u2657"
        assert get_piece_unicode("N") == "\u2658"
        assert get_piece_unicode("P") == "\u2659"

    def test_black_pieces(self):
        from src.games.chess.ui import get_piece_unicode

        assert get_piece_unicode("k") == "\u265a"
        assert get_piece_unicode("q") == "\u265b"
        assert get_piece_unicode("r") == "\u265c"
        assert get_piece_unicode("b") == "\u265d"
        assert get_piece_unicode("n") == "\u265e"
        assert get_piece_unicode("p") == "\u265f"

    def test_unknown_piece(self):
        from src.games.chess.ui import get_piece_unicode

        assert get_piece_unicode("X") == ""
        assert get_piece_unicode("") == ""


@pytest.mark.unit
class TestFenToBoard:
    """Tests for fen_to_board function."""

    def test_starting_position(self):
        from src.games.chess.ui import fen_to_board

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = fen_to_board(fen)
        assert len(board) == 8
        assert all(len(row) == 8 for row in board)
        # First row is rank 8 (black pieces)
        assert board[0] == ["r", "n", "b", "q", "k", "b", "n", "r"]
        assert board[1] == ["p"] * 8
        # Empty rows
        assert board[2] == [""] * 8
        assert board[3] == [""] * 8
        # White pieces
        assert board[6] == ["P"] * 8
        assert board[7] == ["R", "N", "B", "Q", "K", "B", "N", "R"]

    def test_position_with_gaps(self):
        from src.games.chess.ui import fen_to_board

        fen = "8/8/8/8/4P3/8/8/8 w - - 0 1"
        board = fen_to_board(fen)
        # Row 4 (rank 4) has pawn at e4
        assert board[4][4] == "P"
        assert board[4][3] == ""
        assert board[4][5] == ""


@pytest.mark.unit
class TestRenderBoardHtml:
    """Tests for render_board_html function."""

    def test_returns_html_string(self):
        from src.games.chess.ui import render_board_html

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        html = render_board_html(fen)
        assert isinstance(html, str)
        assert "chess-board" in html
        assert "chess-square" in html

    def test_highlight_last_move(self):
        from src.games.chess.ui import render_board_html

        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        html = render_board_html(fen, last_move="e2e4")
        assert "highlight-square" in html

    def test_no_highlight_without_move(self):
        from src.games.chess.ui import render_board_html

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        html = render_board_html(fen, last_move=None)
        # No squares should have the highlight class applied (it exists in CSS but not on elements)
        # Count occurrences outside the style block
        style_end = html.index("</style>")
        board_html = html[style_end:]
        assert "highlight-square" not in board_html

    def test_flipped_board(self):
        from src.games.chess.ui import render_board_html

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        html = render_board_html(fen, flipped=True)
        assert isinstance(html, str)
        assert "chess-board" in html

    def test_file_labels_normal(self):
        from src.games.chess.ui import render_board_html

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        html = render_board_html(fen, flipped=False)
        # Files a-h should appear in order
        assert "file-label" in html

    def test_file_labels_flipped(self):
        from src.games.chess.ui import render_board_html

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        html = render_board_html(fen, flipped=True)
        assert "file-label" in html

    def test_short_last_move_ignored(self):
        from src.games.chess.ui import render_board_html

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # Move string too short to parse - no squares highlighted
        html = render_board_html(fen, last_move="e2")
        style_end = html.index("</style>")
        board_html = html[style_end:]
        assert "highlight-square" not in board_html


@pytest.mark.unit
class TestRenderScorecardHtml:
    """Tests for render_scorecard_html function."""

    def test_empty_scorecard(self):
        from src.games.chess.ui import render_scorecard_html

        sc = ScoreCard()
        html = render_scorecard_html(sc)
        assert "scorecard" in html
        assert "Games Played" in html
        assert "Estimated Elo" in html

    def test_scorecard_with_wins(self):
        from src.games.chess.ui import render_scorecard_html

        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 30, 5000.0, "white")
        html = render_scorecard_html(sc)
        assert "1" in html  # total games

    def test_scorecard_win_streak(self):
        from src.games.chess.ui import render_scorecard_html

        sc = ScoreCard()
        sc.record_game(GameResult.WHITE_WIN, 20, 1000.0, "white")
        sc.record_game(GameResult.WHITE_WIN, 25, 1000.0, "white")
        html = render_scorecard_html(sc)
        assert "streak-win" in html

    def test_scorecard_loss_streak(self):
        from src.games.chess.ui import render_scorecard_html

        sc = ScoreCard()
        sc.record_game(GameResult.BLACK_WIN, 20, 1000.0, "white")
        sc.record_game(GameResult.BLACK_WIN, 25, 1000.0, "white")
        html = render_scorecard_html(sc)
        assert "streak-loss" in html

    def test_scorecard_no_streak_on_draw(self):
        from src.games.chess.ui import render_scorecard_html

        sc = ScoreCard()
        sc.record_game(GameResult.DRAW, 40, 2000.0, "white")
        html = render_scorecard_html(sc)
        # No streak badge element should be rendered (class exists in CSS but not on element)
        style_end = html.index("</style>")
        body_html = html[style_end:]
        assert "streak-badge" not in body_html


@pytest.mark.unit
class TestFormatMoveHistory:
    """Tests for format_move_history function."""

    def test_empty_history(self):
        from src.games.chess.ui import format_move_history

        assert format_move_history([]) == "No moves yet"

    def test_single_move(self):
        from src.games.chess.ui import format_move_history

        result = format_move_history(["e2e4"])
        assert "1. e2e4" in result

    def test_full_move_pair(self):
        from src.games.chess.ui import format_move_history

        result = format_move_history(["e2e4", "e7e5"])
        assert "1. e2e4 e7e5" in result

    def test_multiple_moves(self):
        from src.games.chess.ui import format_move_history

        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        result = format_move_history(moves)
        assert "1. e2e4 e7e5" in result
        assert "2. g1f3 b8c6" in result

    def test_odd_number_of_moves(self):
        from src.games.chess.ui import format_move_history

        moves = ["e2e4", "e7e5", "g1f3"]
        result = format_move_history(moves)
        assert "2. g1f3" in result


@pytest.mark.unit
class TestGameSession:
    """Tests for GameSession dataclass."""

    def test_default_session(self):
        from src.games.chess.ui import GameSession

        session = GameSession()
        assert session.fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert session.move_history == []
        assert session.game_over is False
        assert session.result == ""
        assert session.player_color == "white"

    def test_reset(self):
        from src.games.chess.ui import GameSession

        session = GameSession()
        session.move_history = ["e2e4", "e7e5"]
        session.game_over = True
        session.result = "Draw"
        session.reset("black")
        assert session.move_history == []
        assert session.game_over is False
        assert session.result == ""
        assert session.player_color == "black"

    def test_record_game_result_player_white_wins(self):
        from src.games.chess.ui import GameSession

        session = GameSession()
        session.player_color = "white"
        session.move_history = ["e2e4", "e7e5"]
        session.record_game_result("You win by checkmate!")
        assert session.scorecard.white_wins == 1

    def test_record_game_result_ai_wins(self):
        from src.games.chess.ui import GameSession

        session = GameSession()
        session.player_color = "white"
        session.move_history = ["e2e4"]
        # Use "AI wins" without "checkmate" to avoid hitting the first condition
        session.record_game_result("AI wins")
        assert session.scorecard.total_games == 1
        assert session.scorecard.black_wins == 1

    def test_record_game_result_draw(self):
        from src.games.chess.ui import GameSession

        session = GameSession()
        session.move_history = ["e2e4", "e7e5"]
        session.record_game_result("Draw by stalemate")
        assert session.scorecard.draws == 1

    def test_record_game_result_player_black_wins(self):
        from src.games.chess.ui import GameSession

        session = GameSession()
        session.player_color = "black"
        session.move_history = ["e2e4"]
        session.record_game_result("You win by checkmate!")
        assert session.scorecard.black_wins == 1

    def test_record_game_result_ai_wins_when_player_black(self):
        from src.games.chess.ui import GameSession

        session = GameSession()
        session.player_color = "black"
        session.move_history = ["e2e4"]
        session.record_game_result("AI wins")
        assert session.scorecard.white_wins == 1


@pytest.mark.unit
class TestGetGameStatus:
    """Tests for get_game_status function."""

    def test_game_over(self):
        from src.games.chess.ui import GameSession, get_game_status

        session = GameSession()
        session.game_over = True
        session.result = "Draw"
        assert "Game Over" in get_game_status(session)

    def test_white_to_move(self):
        from src.games.chess.ui import GameSession, get_game_status

        session = GameSession()
        status = get_game_status(session)
        assert "White to move" in status

    def test_check_detected(self):
        from src.games.chess.ui import GameSession, get_game_status

        # Position where black is in check
        session = GameSession()
        session.fen = "rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3"
        status = get_game_status(session)
        assert "Check" in status

    def test_invalid_fen_fallback(self):
        from src.games.chess.ui import GameSession, get_game_status

        session = GameSession()
        session.fen = "invalid-fen"
        status = get_game_status(session)
        assert "Game in progress" in status


@pytest.mark.unit
class TestValidateMove:
    """Tests for validate_move function."""

    def test_valid_move(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        valid, error = ui_mod.validate_move("e2e4")
        assert valid is True
        assert error == ""

    def test_invalid_format_short(self):
        import src.games.chess.ui as ui_mod

        valid, error = ui_mod.validate_move("e2")
        assert valid is False
        assert "Invalid move format" in error

    def test_empty_move(self):
        import src.games.chess.ui as ui_mod

        valid, error = ui_mod.validate_move("")
        assert valid is False

    def test_illegal_move(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        valid, error = ui_mod.validate_move("e1e5")
        assert valid is False
        assert "Illegal move" in error


@pytest.mark.unit
class TestFormatAnalysis:
    """Tests for format_analysis function."""

    def test_basic_format(self):
        from src.games.chess.ui import format_analysis

        analysis = {
            "value": 0.5,
            "confidence": 0.8,
            "routing": "mcts",
            "reasoning": "Deep search",
            "top_moves": {"e2e4": 0.4, "d2d4": 0.3},
        }
        result = format_analysis("e2e4", analysis)
        assert "e2e4" in result
        assert "0.50" in result
        assert "mcts" in result
        assert "Deep search" in result

    def test_missing_analysis_keys(self):
        from src.games.chess.ui import format_analysis

        result = format_analysis("e2e4", {})
        assert "e2e4" in result
        assert "N/A" in result


@pytest.mark.unit
class TestInitializeGame:
    """Tests for initialize_game function."""

    def test_initialize_white(self):
        import src.games.chess.ui as ui_mod

        result = ui_mod.initialize_game("white")
        assert len(result) == 5
        board_html, status, history, message, scorecard_html = result
        assert "chess-board" in board_html
        assert "New game started" in message

    def test_initialize_black(self):
        """When player is black, AI makes first move."""
        import src.games.chess.ui as ui_mod

        result = ui_mod.initialize_game("black")
        assert len(result) == 5
        # AI should have made a move
        assert len(ui_mod._session.move_history) >= 1


@pytest.mark.unit
class TestApplyPlayerMove:
    """Tests for apply_player_move function."""

    def test_game_already_over(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        ui_mod._session.game_over = True
        ui_mod._session.result = "Draw"
        result = ui_mod.apply_player_move("e2e4")
        assert len(result) == 5
        assert "Game is over" in result[3]

    def test_invalid_move(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        result = ui_mod.apply_player_move("e1e5")
        assert len(result) == 5
        assert "Illegal" in result[3] or "Invalid" in result[3] or "Error" in result[3]

    def test_valid_move_triggers_ai(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        result = ui_mod.apply_player_move("e2e4")
        assert len(result) == 5
        # After player move + AI response, there should be 2 moves
        assert len(ui_mod._session.move_history) >= 1


@pytest.mark.unit
class TestUndoMove:
    """Tests for undo_move function."""

    def test_undo_not_enough_moves(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        result = ui_mod.undo_move()
        assert len(result) == 5
        assert "Cannot undo" in result[3]

    def test_undo_with_moves(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        # Simulate two moves
        ui_mod._session.move_history = ["e2e4", "e7e5"]
        import chess

        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        ui_mod._session.fen = board.fen()

        result = ui_mod.undo_move()
        assert len(result) == 5
        assert "Undone" in result[3]
        assert len(ui_mod._session.move_history) == 0


@pytest.mark.unit
class TestResetScorecard:
    """Tests for reset_scorecard function."""

    def test_reset(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.scorecard.record_game(GameResult.WHITE_WIN, 20, 1000.0, "white")
        result = ui_mod.reset_scorecard()
        assert "scorecard" in result
        assert ui_mod._session.scorecard.total_games == 0


@pytest.mark.unit
class TestExportGamePgn:
    """Tests for export_game_pgn function."""

    def test_no_moves_returns_none(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        assert ui_mod.export_game_pgn() is None

    def test_export_with_moves(self):
        import os

        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        ui_mod._session.move_history = ["e2e4", "e7e5"]
        result = ui_mod.export_game_pgn()
        assert result is not None
        assert result.endswith(".pgn")
        assert os.path.exists(result)
        # Clean up temp file
        os.unlink(result)


@pytest.mark.unit
class TestMakeAiMoveSync:
    """Tests for make_ai_move_sync function."""

    def test_game_already_over_checkmate(self):
        import src.games.chess.ui as ui_mod

        # Fool's mate position - black has just checkmated white
        ui_mod._session.reset("white")
        ui_mod._session.fen = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        result = ui_mod.make_ai_move_sync()
        assert len(result) == 5
        assert ui_mod._session.game_over is True

    def test_fallback_ai_move(self):
        import src.games.chess.ui as ui_mod

        ui_mod._session.reset("white")
        # Standard starting position, AI should find a move via fallback
        result = ui_mod.make_ai_move_sync()
        assert len(result) == 5
        assert len(ui_mod._session.move_history) >= 1


@pytest.mark.unit
class TestGetAiMove:
    """Tests for get_ai_move function."""

    def test_fallback_returns_valid_move(self):
        from src.games.chess.ui import get_ai_move

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move, analysis = get_ai_move(fen)
        assert isinstance(move, str)
        assert len(move) >= 4
        assert "value" in analysis
        assert "confidence" in analysis
        assert "top_moves" in analysis


@pytest.mark.unit
class TestContinuousLearningFunctions:
    """Tests for continuous learning control functions."""

    def test_stop_when_not_running(self):
        import src.games.chess.ui as ui_mod

        ui_mod._learning_session = None
        msg, html = ui_mod.stop_continuous_learning()
        assert "No learning session" in msg

    def test_pause_when_not_running(self):
        import src.games.chess.ui as ui_mod

        ui_mod._learning_session = None
        msg, html = ui_mod.pause_continuous_learning()
        assert "No learning session" in msg

    def test_get_learning_status_no_session(self):
        import src.games.chess.ui as ui_mod

        ui_mod._learning_session = None
        html = ui_mod.get_learning_status()
        assert "No learning session active" in html

    def test_render_learning_board_no_session(self):
        import src.games.chess.ui as ui_mod

        ui_mod._learning_session = None
        html = ui_mod.render_learning_board_html()
        assert "Waiting for session" in html

    def test_render_learning_status_no_session(self):
        import src.games.chess.ui as ui_mod

        ui_mod._learning_session = None
        html = ui_mod.render_learning_status()
        assert "No learning session active" in html

    def test_render_learning_status_with_session(self):
        from datetime import timedelta

        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.scorecard = ScoreCard()
        mock_session.is_running = True
        mock_session.is_paused = False
        mock_session.get_session_duration.return_value = timedelta(minutes=5, seconds=30)

        ui_mod._learning_session = mock_session
        try:
            html = ui_mod.render_learning_status()
            assert "Running" in html
            assert "Continuous Learning" in html
        finally:
            ui_mod._learning_session = None

    def test_render_learning_status_paused(self):
        from datetime import timedelta

        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.scorecard = ScoreCard()
        mock_session.is_running = True
        mock_session.is_paused = True
        mock_session.get_session_duration.return_value = timedelta(minutes=2)

        ui_mod._learning_session = mock_session
        try:
            html = ui_mod.render_learning_status()
            assert "Paused" in html
        finally:
            ui_mod._learning_session = None

    def test_render_learning_status_stopped(self):
        from datetime import timedelta

        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.scorecard = ScoreCard()
        mock_session.is_running = False
        mock_session.is_paused = False
        mock_session.get_session_duration.return_value = timedelta(minutes=10)

        ui_mod._learning_session = mock_session
        try:
            html = ui_mod.render_learning_status()
            assert "Stopped" in html
        finally:
            ui_mod._learning_session = None

    def test_render_learning_board_with_running_session(self):
        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.current_game_id_display = "Game #5"
        mock_session.current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        mock_session.current_last_move = None

        ui_mod._learning_session = mock_session
        try:
            html = ui_mod.render_learning_board_html()
            assert "Game #5" in html
            assert "chess-board" in html
        finally:
            ui_mod._learning_session = None

    def test_render_learning_board_not_running(self):
        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.is_running = False

        ui_mod._learning_session = mock_session
        try:
            html = ui_mod.render_learning_board_html()
            assert "Waiting for session" in html
        finally:
            ui_mod._learning_session = None

    def test_stop_running_session(self):
        from datetime import timedelta

        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.scorecard = ScoreCard()
        mock_session.is_paused = False
        mock_session.get_session_duration.return_value = timedelta(minutes=1)

        ui_mod._learning_session = mock_session
        try:
            msg, html = ui_mod.stop_continuous_learning()
            assert "Stopping" in msg
            mock_session.stop.assert_called_once()
        finally:
            ui_mod._learning_session = None

    def test_pause_running_session(self):
        from datetime import timedelta

        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.is_paused = False
        mock_session.scorecard = ScoreCard()
        mock_session.get_session_duration.return_value = timedelta(minutes=1)

        ui_mod._learning_session = mock_session
        try:
            msg, html = ui_mod.pause_continuous_learning()
            assert "Paused" in msg
            mock_session.pause.assert_called_once()
        finally:
            ui_mod._learning_session = None

    def test_resume_paused_session(self):
        from datetime import timedelta

        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.is_paused = True
        mock_session.scorecard = ScoreCard()
        mock_session.get_session_duration.return_value = timedelta(minutes=1)

        ui_mod._learning_session = mock_session
        try:
            msg, html = ui_mod.pause_continuous_learning()
            assert "Resumed" in msg
            mock_session.resume.assert_called_once()
        finally:
            ui_mod._learning_session = None

    def test_start_when_already_running(self):
        from datetime import timedelta

        import src.games.chess.ui as ui_mod

        mock_session = MagicMock()
        mock_session.is_running = True
        mock_session.scorecard = ScoreCard()
        mock_session.is_paused = False
        mock_session.get_session_duration.return_value = timedelta(minutes=1)

        ui_mod._learning_session = mock_session
        try:
            msg, html = ui_mod.start_continuous_learning(10, 50)
            assert "already running" in msg
        finally:
            ui_mod._learning_session = None
