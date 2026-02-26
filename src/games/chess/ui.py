"""
Chess Web UI with Gradio.

Provides a web interface for playing chess against the AlphaZero-style
ensemble AI with real-time board visualization, scorecard tracking,
and continuous learning mode.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import chess.pgn
import gradio as gr

from src.games.chess.continuous_learning import (
    ContinuousLearningConfig,
    ContinuousLearningSession,
    GameResult,
    ScoreCard,
)

logger = logging.getLogger(__name__)


@dataclass
class GameSession:
    """Maintains state for a chess game session."""

    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move_history: list[str] = field(default_factory=list)
    game_over: bool = False
    result: str = ""
    player_color: str = "white"
    ai_thinking: bool = False
    last_ai_analysis: dict[str, Any] = field(default_factory=dict)

    # Scorecard
    scorecard: ScoreCard = field(default_factory=ScoreCard)

    # Learning mode
    learning_session: ContinuousLearningSession | None = None
    learning_thread: threading.Thread | None = None

    def reset(self, player_color: str = "white") -> None:
        """Reset the game to initial state."""
        self.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.move_history = []
        self.game_over = False
        self.result = ""
        self.player_color = player_color
        self.ai_thinking = False
        self.last_ai_analysis = {}

    def record_game_result(self, result: str) -> None:
        """Record a game result to the scorecard."""
        if "You win" in result or "checkmate" in result.lower():
            if self.player_color == "white":
                game_result = GameResult.WHITE_WIN
            else:
                game_result = GameResult.BLACK_WIN
        elif "AI wins" in result:
            if self.player_color == "white":
                game_result = GameResult.BLACK_WIN
            else:
                game_result = GameResult.WHITE_WIN
        else:
            game_result = GameResult.DRAW

        self.scorecard.record_game(
            game_result,
            len(self.move_history),
            0,  # No time tracking in manual mode
            self.player_color,
        )


# Global session
_session = GameSession()


def get_piece_unicode(piece: str) -> str:
    """Convert piece character to Unicode chess symbol."""
    pieces = {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
        "P": "♙",
        "k": "♚",
        "q": "♛",
        "r": "♜",
        "b": "♝",
        "n": "♞",
        "p": "♟",
    }
    return pieces.get(piece, "")


def fen_to_board(fen: str) -> list[list[str]]:
    """Convert FEN string to 8x8 board array."""
    board = []
    rows = fen.split()[0].split("/")

    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend([""] * int(char))
            else:
                board_row.append(char)
        board.append(board_row)

    return board


def render_board_html(fen: str, last_move: str | None = None, flipped: bool = False) -> str:
    """Render chess board as HTML with CSS styling."""
    board = fen_to_board(fen)

    # Parse last move for highlighting
    highlight_squares = set()
    if last_move and len(last_move) >= 4:
        from_sq = last_move[:2]
        to_sq = last_move[2:4]
        from_file = ord(from_sq[0]) - ord("a")
        from_rank = 8 - int(from_sq[1])
        to_file = ord(to_sq[0]) - ord("a")
        to_rank = 8 - int(to_sq[1])
        highlight_squares.add((from_rank, from_file))
        highlight_squares.add((to_rank, to_file))

    html_parts = [
        """
    <style>
        .chess-board {
            display: inline-block;
            border: 3px solid #333;
            border-radius: 4px;
            font-family: 'Segoe UI Symbol', 'DejaVu Sans', sans-serif;
        }
        .chess-row { display: flex; height: 60px; }
        .chess-square {
            width: 60px; height: 60px;
            display: flex; align-items: center; justify-content: center;
            font-size: 42px; cursor: pointer; user-select: none;
        }
        .chess-square:hover { outline: 3px solid #007bff; outline-offset: -3px; }
        .light-square { background-color: #f0d9b5; }
        .dark-square { background-color: #b58863; }
        .highlight-square { background-color: #aaa23a !important; }
        .rank-label, .file-label { font-size: 12px; color: #666; font-weight: bold; }
        .rank-labels { display: flex; flex-direction: column; justify-content: space-around; padding-right: 5px; height: 480px; }
        .file-labels { display: flex; justify-content: space-around; padding-left: 25px; width: 480px; }
        .board-container { display: flex; align-items: center; }
        .board-wrapper { display: flex; flex-direction: column; }
    </style>
    <div class="board-wrapper">
        <div class="board-container">
            <div class="rank-labels">
    """
    ]

    ranks = list(range(8, 0, -1)) if not flipped else list(range(1, 9))
    for rank in ranks:
        html_parts.append(f'<span class="rank-label">{rank}</span>')

    html_parts.append('</div><div class="chess-board" id="chess-board">')

    row_indices = range(8) if not flipped else range(7, -1, -1)
    col_indices = range(8) if not flipped else range(7, -1, -1)

    for _row_idx, board_row in enumerate(row_indices):
        html_parts.append('<div class="chess-row">')
        for _col_idx, board_col in enumerate(col_indices):
            is_light = (board_row + board_col) % 2 == 0
            square_class = "light-square" if is_light else "dark-square"
            if (board_row, board_col) in highlight_squares:
                square_class += " highlight-square"
            piece = board[board_row][board_col]
            piece_symbol = get_piece_unicode(piece)
            file_char = chr(ord("a") + board_col)
            rank_num = 8 - board_row
            square_name = f"{file_char}{rank_num}"
            html_parts.append(
                f'<div class="chess-square {square_class}" '
                f'data-square="{square_name}" data-piece="{piece}">'
                f"{piece_symbol}</div>"
            )
        html_parts.append("</div>")

    html_parts.append("</div></div>")
    html_parts.append('<div class="file-labels">')
    files = "abcdefgh" if not flipped else "hgfedcba"
    for f in files:
        html_parts.append(f'<span class="file-label">{f}</span>')
    html_parts.append("</div></div>")

    return "".join(html_parts)


def render_scorecard_html(scorecard: ScoreCard) -> str:
    """Render scorecard as HTML."""
    win_rate = scorecard.win_rate * 100
    scorecard.draw_rate * 100

    # Determine streak info
    if scorecard.streak_type == "win":
        streak_text = f"🔥 {scorecard.current_streak} win streak"
        streak_class = "streak-win"
    elif scorecard.streak_type == "loss":
        streak_text = f"❄️ {scorecard.current_streak} loss streak"
        streak_class = "streak-loss"
    else:
        streak_text = ""
        streak_class = ""

    return f"""
    <style>
        .scorecard {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 20px;
            color: white;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .scorecard-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            color: #e94560;
        }}
        .score-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
        }}
        .score-label {{ color: #a0a0a0; }}
        .score-value {{ font-weight: bold; color: #00d9ff; }}
        .score-wins {{ color: #4ade80; }}
        .score-losses {{ color: #f87171; }}
        .score-draws {{ color: #fbbf24; }}
        .elo-badge {{
            text-align: center;
            margin-top: 15px;
            padding: 10px;
            background: rgba(233, 69, 96, 0.2);
            border-radius: 8px;
            border: 1px solid #e94560;
        }}
        .elo-value {{ font-size: 24px; font-weight: bold; color: #e94560; }}
        .streak-badge {{
            text-align: center;
            margin-top: 10px;
            padding: 8px;
            border-radius: 6px;
            font-weight: bold;
        }}
        .streak-win {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; }}
        .streak-loss {{ background: rgba(248, 113, 113, 0.2); color: #f87171; }}
        .learning-stats {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }}
    </style>
    <div class="scorecard">
        <div class="scorecard-title">📊 Score Card</div>

        <div class="score-row">
            <span class="score-label">Games Played</span>
            <span class="score-value">{scorecard.total_games}</span>
        </div>

        <div class="score-row">
            <span class="score-label">Wins (White)</span>
            <span class="score-value score-wins">{scorecard.white_wins}</span>
        </div>

        <div class="score-row">
            <span class="score-label">Wins (Black)</span>
            <span class="score-value score-wins">{scorecard.black_wins}</span>
        </div>

        <div class="score-row">
            <span class="score-label">Draws</span>
            <span class="score-value score-draws">{scorecard.draws}</span>
        </div>

        <div class="score-row">
            <span class="score-label">Win Rate</span>
            <span class="score-value">{win_rate:.1f}%</span>
        </div>

        <div class="score-row">
            <span class="score-label">Avg Game Length</span>
            <span class="score-value">{scorecard.avg_game_length:.1f} moves</span>
        </div>

        <div class="elo-badge">
            <div style="color: #a0a0a0; font-size: 12px;">Estimated Elo</div>
            <div class="elo-value">{scorecard.elo_estimate:.0f}</div>
        </div>

        {f'<div class="streak-badge {streak_class}">{streak_text}</div>' if streak_text else ""}

        <div class="learning-stats">
            <div class="score-row">
                <span class="score-label">Positions Learned</span>
                <span class="score-value">{scorecard.total_positions_learned:,}</span>
            </div>
            <div class="score-row">
                <span class="score-label">Last Loss</span>
                <span class="score-value">{scorecard.last_loss:.4f}</span>
            </div>
        </div>
    </div>
    """


def format_move_history(moves: list[str]) -> str:
    """Format move history as PGN-like notation."""
    if not moves:
        return "No moves yet"

    lines = []
    for i in range(0, len(moves), 2):
        move_num = i // 2 + 1
        white_move = moves[i]
        black_move = moves[i + 1] if i + 1 < len(moves) else ""
        lines.append(f"{move_num}. {white_move} {black_move}".strip())

    return "\n".join(lines)


def get_game_status(session: GameSession) -> str:
    """Get current game status message."""
    if session.game_over:
        return f"Game Over: {session.result}"

    try:
        import chess

        board = chess.Board(session.fen)
        if board.is_check():
            return "Check!"
        turn = "White" if board.turn else "Black"
        return f"{turn} to move"
    except Exception:
        return "Game in progress"


def initialize_game(player_color: str) -> tuple[str, str, str, str, str]:
    """Initialize a new game."""
    global _session
    _session.reset(player_color)

    board_html = render_board_html(_session.fen, flipped=(player_color == "black"))
    status = get_game_status(_session)
    history = format_move_history(_session.move_history)
    scorecard_html = render_scorecard_html(_session.scorecard)

    if player_color == "black":
        return make_ai_move_sync()

    return board_html, status, history, "New game started. Your turn!", scorecard_html


def validate_move(move_uci: str) -> tuple[bool, str]:
    """Validate a move in UCI format."""
    if not move_uci or len(move_uci) < 4:
        return False, "Invalid move format. Use UCI notation (e.g., e2e4)"

    try:
        import chess

        board = chess.Board(_session.fen)
        move = chess.Move.from_uci(move_uci.lower().strip())
        if move not in board.legal_moves:
            legal = [m.uci() for m in list(board.legal_moves)[:10]]
            return False, f"Illegal move. Legal moves include: {', '.join(legal)}..."
        return True, ""
    except Exception as e:
        return False, f"Error validating move: {str(e)}"


def apply_player_move(move_uci: str) -> tuple[str, str, str, str, str]:
    """Apply player's move and get AI response."""
    global _session

    if _session.game_over:
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            "Game is over. Start a new game.",
            render_scorecard_html(_session.scorecard),
        )

    is_valid, error = validate_move(move_uci)
    if not is_valid:
        return (
            render_board_html(
                _session.fen,
                _session.move_history[-1] if _session.move_history else None,
                flipped=(_session.player_color == "black"),
            ),
            get_game_status(_session),
            format_move_history(_session.move_history),
            error,
            render_scorecard_html(_session.scorecard),
        )

    try:
        import chess

        board = chess.Board(_session.fen)
        move = chess.Move.from_uci(move_uci.lower().strip())
        board.push(move)

        _session.fen = board.fen()
        _session.move_history.append(move_uci.lower())

        if board.is_game_over():
            _session.game_over = True
            if board.is_checkmate():
                _session.result = "You win by checkmate!"
            elif board.is_stalemate():
                _session.result = "Draw by stalemate"
            else:
                _session.result = "Draw"

            _session.record_game_result(_session.result)

            return (
                render_board_html(_session.fen, move_uci, flipped=(_session.player_color == "black")),
                get_game_status(_session),
                format_move_history(_session.move_history),
                _session.result,
                render_scorecard_html(_session.scorecard),
            )

        return make_ai_move_sync()

    except Exception as e:
        logger.exception("Error applying move")
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            f"Error: {str(e)}",
            render_scorecard_html(_session.scorecard),
        )


def make_ai_move_sync() -> tuple[str, str, str, str, str]:
    """Make AI move synchronously."""
    global _session

    try:
        import chess

        board = chess.Board(_session.fen)

        if board.is_game_over():
            _session.game_over = True
            if board.is_checkmate():
                _session.result = "AI wins by checkmate!"
            else:
                _session.result = "Draw"

            _session.record_game_result(_session.result)

            return (
                render_board_html(_session.fen, flipped=(_session.player_color == "black")),
                get_game_status(_session),
                format_move_history(_session.move_history),
                _session.result,
                render_scorecard_html(_session.scorecard),
            )

        ai_move, analysis = get_ai_move(_session.fen)

        move = chess.Move.from_uci(ai_move)
        board.push(move)

        _session.fen = board.fen()
        _session.move_history.append(ai_move)
        _session.last_ai_analysis = analysis

        if board.is_game_over():
            _session.game_over = True
            if board.is_checkmate():
                _session.result = "AI wins by checkmate!"
            elif board.is_stalemate():
                _session.result = "Draw by stalemate"
            else:
                _session.result = "Draw"

            _session.record_game_result(_session.result)

        return (
            render_board_html(_session.fen, ai_move, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            format_analysis(ai_move, analysis),
            render_scorecard_html(_session.scorecard),
        )

    except Exception as e:
        logger.exception("Error making AI move")
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            f"AI Error: {str(e)}",
            render_scorecard_html(_session.scorecard),
        )


def get_ai_move(fen: str) -> tuple[str, dict[str, Any]]:
    """Get AI's move for the position."""
    import random

    import chess

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    try:
        from src.games.chess import ChessEnsembleAgent, ChessGameState, get_chess_small_config

        config = get_chess_small_config()
        config.mcts.num_simulations = 50

        state = ChessGameState.from_fen(fen)
        agent = ChessEnsembleAgent(config)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(agent.get_best_move(state, temperature=0.1, use_ensemble=False))
            return response.best_move, {
                "value": response.value_estimate,
                "confidence": response.confidence,
                "routing": response.routing_decision.primary_agent.value,
                "reasoning": response.routing_decision.reasoning,
                "top_moves": dict(list(response.move_probabilities.items())[:5]),
            }
        finally:
            loop.close()

    except Exception as e:
        logger.warning(f"Ensemble agent failed, using fallback: {e}")

    # Fallback
    scored_moves = []
    for move in legal_moves:
        score = random.random() * 0.1
        if board.is_capture(move):
            score += 0.5
        to_sq = move.to_square
        file, rank = to_sq % 8, to_sq // 8
        center_dist = abs(file - 3.5) + abs(rank - 3.5)
        score += (7 - center_dist) * 0.05
        board.push(move)
        if board.is_check():
            score += 0.3
        board.pop()
        scored_moves.append((move, score))

    scored_moves.sort(key=lambda x: -x[1])
    best_move = scored_moves[0][0]

    return best_move.uci(), {
        "value": 0.0,
        "confidence": 0.5,
        "routing": "fallback",
        "reasoning": "Using simple heuristic (ensemble agent not available)",
        "top_moves": {m.uci(): s for m, s in scored_moves[:5]},
    }


def format_analysis(move: str, analysis: dict[str, Any]) -> str:
    """Format AI analysis for display."""
    lines = [
        f"AI plays: **{move}**",
        "",
        f"Evaluation: {analysis.get('value', 0):.2f}",
        f"Confidence: {analysis.get('confidence', 0):.1%}",
        f"Agent: {analysis.get('routing', 'unknown')}",
        "",
        f"Reasoning: {analysis.get('reasoning', 'N/A')}",
        "",
        "Top moves considered:",
    ]
    top_moves = analysis.get("top_moves", {})
    for m, score in list(top_moves.items())[:5]:
        lines.append(f"  {m}: {score:.1%}")
    return "\n".join(lines)


def undo_move() -> tuple[str, str, str, str, str]:
    """Undo the last two moves (player + AI)."""
    global _session

    if len(_session.move_history) < 2:
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            "Cannot undo: not enough moves",
            render_scorecard_html(_session.scorecard),
        )

    try:
        import chess

        board = chess.Board()
        moves_to_keep = _session.move_history[:-2]
        for move_uci in moves_to_keep:
            board.push(chess.Move.from_uci(move_uci))

        _session.fen = board.fen()
        _session.move_history = moves_to_keep
        _session.game_over = False
        _session.result = ""

        last_move = moves_to_keep[-1] if moves_to_keep else None

        return (
            render_board_html(_session.fen, last_move, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            "Undone last two moves",
            render_scorecard_html(_session.scorecard),
        )

    except Exception as e:
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            f"Error undoing: {str(e)}",
            render_scorecard_html(_session.scorecard),
        )


def reset_scorecard() -> str:
    """Reset the scorecard."""
    global _session
    _session.scorecard.reset()
    return render_scorecard_html(_session.scorecard)


def export_game_pgn() -> str | None:
    """Export the current game to a PGN file."""
    global _session

    if not _session.move_history:
        return None

    try:
        # Create PGN game
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess vs AlphaZero AI"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "Player" if _session.player_color == "white" else "AI"
        game.headers["Black"] = "AI" if _session.player_color == "white" else "Player"
        game.headers["Result"] = _session.result if _session.game_over else "*"

        # Replay moves
        node = game
        board = chess.Board()
        for move_uci in _session.move_history:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                node = node.add_variation(move)
                board.push(move)
            else:
                break

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as tmp:
            print(game, file=tmp, end="\n\n")
            return tmp.name

    except Exception as e:
        logger.error(f"Error exporting PGN: {e}")
        return None


# Continuous learning functions
_learning_session: ContinuousLearningSession | None = None
_learning_stop_event = threading.Event()


def start_continuous_learning(
    duration_minutes: int,
    max_games: int,
) -> tuple[str, str]:
    """Start continuous learning mode."""
    global _learning_session, _learning_stop_event

    if _learning_session is not None and _learning_session.is_running:
        return "Learning already running!", render_learning_status()

    _learning_stop_event.clear()

    from src.games.chess import get_chess_small_config

    config = get_chess_small_config()
    learning_config = ContinuousLearningConfig(
        max_session_minutes=duration_minutes,
        max_games=max_games,
        learn_every_n_games=3,
        min_games_before_learning=5,
    )

    _learning_session = ContinuousLearningSession(config, learning_config)

    def run_learning():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _learning_session.run_session(
                    max_minutes=duration_minutes,
                    max_games=max_games,
                )
            )
        finally:
            loop.close()

    thread = threading.Thread(target=run_learning, daemon=True)
    thread.start()

    return f"Started learning: {duration_minutes} min, max {max_games} games", render_learning_status()


def stop_continuous_learning() -> tuple[str, str]:
    """Stop continuous learning mode."""
    global _learning_session

    if _learning_session is None or not _learning_session.is_running:
        return "No learning session running", render_learning_status()

    _learning_session.stop()
    return "Stopping learning session...", render_learning_status()


def pause_continuous_learning() -> tuple[str, str]:
    """Pause/resume continuous learning."""
    global _learning_session

    if _learning_session is None or not _learning_session.is_running:
        return "No learning session running", render_learning_status()

    if _learning_session.is_paused:
        _learning_session.resume()
        return "Resumed learning", render_learning_status()
    else:
        _learning_session.pause()
        return "Paused learning", render_learning_status()


def get_learning_status() -> str:
    """Get current learning status."""
    return render_learning_status()


def render_learning_status() -> str:
    """Render learning status as HTML."""
    global _learning_session

    if _learning_session is None:
        return """
        <div style="padding: 20px; text-align: center; color: #666;">
            <p>No learning session active</p>
            <p>Start a session to train the AI through self-play</p>
        </div>
        """

    sc = _learning_session.scorecard
    is_running = _learning_session.is_running
    is_paused = _learning_session.is_paused
    duration = _learning_session.get_session_duration()

    status_color = "#4ade80" if is_running and not is_paused else "#fbbf24" if is_paused else "#f87171"
    status_text = "Running" if is_running and not is_paused else "Paused" if is_paused else "Stopped"

    return f"""
    <style>
        .learning-panel {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 12px;
            padding: 20px;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }}
        .learning-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 12px;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .stat-box {{
            background: rgba(255,255,255,0.1);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #00d9ff;
        }}
        .stat-label {{
            font-size: 11px;
            color: #a0a0a0;
            text-transform: uppercase;
        }}
        .progress-bar {{
            height: 6px;
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
            margin-top: 15px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #e94560);
            border-radius: 3px;
        }}
    </style>
    <div class="learning-panel">
        <div class="learning-header">
            <span style="font-size: 16px; font-weight: bold;">🤖 Continuous Learning</span>
            <span class="status-badge" style="background: {status_color}; color: black;">{status_text}</span>
        </div>

        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-value">{sc.total_games}</div>
                <div class="stat-label">Games</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{duration.seconds // 60}:{duration.seconds % 60:02d}</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{sc.white_wins}</div>
                <div class="stat-label">White Wins</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{sc.black_wins}</div>
                <div class="stat-label">Black Wins</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{sc.draws}</div>
                <div class="stat-label">Draws</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{sc.total_positions_learned:,}</div>
                <div class="stat-label">Positions Learned</div>
            </div>
        </div>

        <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: #a0a0a0; font-size: 12px;">Training Loss</span>
                <span style="color: #00d9ff; font-weight: bold;">{sc.last_loss:.4f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #a0a0a0; font-size: 12px;">Avg Game Length</span>
                <span style="color: #00d9ff; font-weight: bold;">{sc.avg_game_length:.1f}</span>
            </div>
        </div>
    </div>
    """


def render_learning_board_html() -> str:
    """Render the live learning board."""
    global _learning_session

    if _learning_session is None or not _learning_session.is_running:
        return "<div style='text-align:center; padding:50px; color:#666;'>Waiting for session to start...</div>"

    game_info = f"<div style='text-align:center; margin-bottom:10px; font-weight:bold; color:#00d9ff;'>Playing: {_learning_session.current_game_id_display}</div>"

    return game_info + render_board_html(
        _learning_session.current_fen, _learning_session.current_last_move, flipped=False
    )


def create_chess_ui() -> gr.Blocks:
    """Create the Gradio chess UI with scorecard and learning mode."""
    with gr.Blocks(
        title="Chess vs AlphaZero AI",
        theme=gr.themes.Soft(),
        css="""
        .chess-container { max-width: 1400px; margin: auto; }
        .move-input { font-family: monospace; }
        .tab-content { min-height: 400px; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # ♟️ Chess vs AlphaZero-Style AI

            Play chess against an AI using HRM, TRM, and Neural MCTS ensemble agents.
            Watch the AI learn and improve through continuous self-play!
            """
        )

        with gr.Tabs():
            # Tab 1: Play Mode
            with gr.TabItem("🎮 Play", elem_id="play-tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        board_display = gr.HTML(
                            value=render_board_html(_session.fen),
                            label="Chess Board",
                            elem_id="board-container",
                        )

                        with gr.Row():
                            move_input = gr.Textbox(
                                label="Your Move (UCI format)",
                                placeholder="e.g., e2e4",
                                elem_classes=["move-input"],
                                elem_id="move-input",
                            )
                            move_btn = gr.Button("Make Move", variant="primary", elem_id="move-btn")

                        with gr.Row():
                            color_select = gr.Radio(
                                choices=["white", "black"],
                                value="white",
                                label="Play as",
                                elem_id="color-select",
                            )
                            new_game_btn = gr.Button("New Game", variant="secondary", elem_id="new-game-btn")
                            undo_btn = gr.Button("Undo", elem_id="undo-btn")

                    with gr.Column(scale=1):
                        scorecard_display = gr.HTML(
                            value=render_scorecard_html(_session.scorecard),
                            label="Score Card",
                            elem_id="scorecard-display",
                        )

                        reset_score_btn = gr.Button("Reset Score", size="sm", elem_id="reset-score-btn")

                        status_display = gr.Textbox(
                            label="Game Status",
                            value="White to move",
                            interactive=False,
                            elem_id="status-display",
                        )

                        history_display = gr.Textbox(
                            label="Move History",
                            value="No moves yet",
                            interactive=False,
                            lines=6,
                            elem_id="history-display",
                        )

                        with gr.Row():
                            export_pgn_btn = gr.Button("Export PGN", size="sm", elem_id="export-pgn-btn")
                            pgn_file = gr.File(
                                label="Download PGN",
                                file_count="single",
                                type="filepath",
                                interactive=False,
                                elem_id="pgn-file-output",
                                height=80,
                            )

                        analysis_display = gr.Markdown(
                            value="*AI analysis will appear here*",
                            label="AI Analysis",
                            elem_id="analysis-display",
                        )

            # Tab 2: Continuous Learning Mode
            with gr.TabItem("🧠 Continuous Learning", elem_id="learning-tab"):
                gr.Markdown(
                    """
                    ### AlphaZero-Style Continuous Learning

                    Watch the AI play against itself and learn in real-time!
                    The AI generates training data through self-play and updates its neural network.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Session Configuration")

                        duration_slider = gr.Slider(
                            minimum=1,
                            maximum=120,
                            value=10,
                            step=1,
                            label="Duration (minutes)",
                            elem_id="duration-slider",
                        )

                        max_games_slider = gr.Slider(
                            minimum=5,
                            maximum=500,
                            value=50,
                            step=5,
                            label="Maximum Games",
                            elem_id="max-games-slider",
                        )

                        with gr.Row():
                            start_learning_btn = gr.Button(
                                "▶️ Start Learning",
                                variant="primary",
                                elem_id="start-learning-btn",
                            )
                            pause_learning_btn = gr.Button(
                                "⏸️ Pause",
                                elem_id="pause-learning-btn",
                            )
                            stop_learning_btn = gr.Button(
                                "⏹️ Stop",
                                variant="stop",
                                elem_id="stop-learning-btn",
                            )

                        learning_message = gr.Textbox(
                            label="Status Message",
                            interactive=False,
                            elem_id="learning-message",
                        )

                    with gr.Column(scale=1):
                        learning_status_display = gr.HTML(
                            value=render_learning_status(),
                            label="Learning Status",
                            elem_id="learning-status-display",
                        )

                # Live Board Display
                with gr.Row():
                    learning_board_display = gr.HTML(
                        value=render_learning_board_html(),
                        label="Live Training Games",
                        elem_id="learning-board-display",
                    )

                    refresh_btn = gr.Button(
                        "🔄 Refresh Status",
                        elem_id="refresh-btn",
                    )

        # Event handlers for Play mode
        move_btn.click(
            fn=apply_player_move,
            inputs=[move_input],
            outputs=[board_display, status_display, history_display, analysis_display, scorecard_display],
        ).then(fn=lambda: "", outputs=[move_input])

        move_input.submit(
            fn=apply_player_move,
            inputs=[move_input],
            outputs=[board_display, status_display, history_display, analysis_display, scorecard_display],
        ).then(fn=lambda: "", outputs=[move_input])

        new_game_btn.click(
            fn=initialize_game,
            inputs=[color_select],
            outputs=[board_display, status_display, history_display, analysis_display, scorecard_display],
        )

        undo_btn.click(
            fn=undo_move,
            outputs=[board_display, status_display, history_display, analysis_display, scorecard_display],
        )

        reset_score_btn.click(
            fn=reset_scorecard,
            outputs=[scorecard_display],
        )

        export_pgn_btn.click(
            fn=export_game_pgn,
            outputs=[pgn_file],
        )

        # Event handlers for Learning mode
        start_learning_btn.click(
            fn=start_continuous_learning,
            inputs=[duration_slider, max_games_slider],
            outputs=[learning_message, learning_status_display],
        )

        pause_learning_btn.click(
            fn=pause_continuous_learning,
            outputs=[learning_message, learning_status_display],
        )

        stop_learning_btn.click(
            fn=stop_continuous_learning,
            outputs=[learning_message, learning_status_display],
        )

        # Auto-refresh learning tab (status + board) every 1 second
        demo.load(
            fn=lambda: (render_learning_status(), render_learning_board_html()),
            outputs=[learning_status_display, learning_board_display],
            every=1.0,
        )

        refresh_btn.click(
            fn=get_learning_status,
            outputs=[learning_status_display],
        )

        # Initialize Play tab state on load
        demo.load(
            fn=lambda: (
                render_board_html(_session.fen),
                get_game_status(_session),
                format_move_history(_session.move_history),
                "*AI analysis will appear here*",
                render_scorecard_html(_session.scorecard),
            ),
            outputs=[board_display, status_display, history_display, analysis_display, scorecard_display],
        )

    return demo


def main() -> None:
    """Launch the chess UI."""
    import argparse

    parser = argparse.ArgumentParser(description="Chess Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    demo = create_chess_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
