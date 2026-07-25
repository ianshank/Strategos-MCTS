"""
Chess Demo Web Application.

Flask-based web app with chessboard.js frontend for interactive
multi-agent chess gameplay with learning dashboard.

Best Practices 2025:
- Async endpoint support
- Real-time updates via SSE
- Comprehensive error handling
"""

from __future__ import annotations

import asyncio
import uuid

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from .chess_ensemble import ChessEnsemble

# Local imports
from .chess_state import CHESS_AVAILABLE, ChessState
from .learning_dashboard import LearningDashboard

# Create Flask app
app = Flask(__name__)
CORS(app)

# Global state
_ensembles: dict[str, ChessEnsemble] = {}
_dashboard = LearningDashboard()


def get_or_create_ensemble(game_id: str | None = None) -> tuple[str, ChessEnsemble]:
    """Get existing or create new ensemble for a game."""
    if game_id and game_id in _ensembles:
        return game_id, _ensembles[game_id]

    new_id = game_id or str(uuid.uuid4())[:8]
    ensemble = ChessEnsemble(game_id=new_id)
    _ensembles[new_id] = ensemble
    return new_id, ensemble


@app.route("/")
def index():
    """Serve main chess interface."""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard_page():
    """Serve learning dashboard."""
    return render_template("dashboard.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    """Start a new game."""
    if not CHESS_AVAILABLE:
        return jsonify({"error": "python-chess not installed"}), 500

    data = request.json or {}
    game_id, ensemble = get_or_create_ensemble()

    # Initialize starting position or custom FEN
    fen = data.get("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    try:
        state = ChessState.from_fen(fen)
    except Exception as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    return jsonify(
        {
            "game_id": game_id,
            "fen": state.get_fen(),
            "turn": "white" if state.board.turn else "black",
            "legal_moves": state.get_legal_actions(),
            "phase": state.get_phase(),
            "evaluation": state.evaluate(),
        }
    )


@app.route("/api/move", methods=["POST"])
def make_move():
    """Make a move in the game."""
    if not CHESS_AVAILABLE:
        return jsonify({"error": "python-chess not installed"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    game_id = data.get("game_id")
    fen = data.get("fen")
    move = data.get("move")  # UCI format

    if not fen:
        return jsonify({"error": "FEN required"}), 400

    try:
        state = ChessState.from_fen(fen)

        if move:
            # User made a move
            if move not in state.get_legal_actions():
                return jsonify({"error": f"Illegal move: {move}"}), 400

            state = state.apply_action(move)

        return jsonify(
            {
                "game_id": game_id,
                "fen": state.get_fen(),
                "turn": "white" if state.board.turn else "black",
                "legal_moves": state.get_legal_actions(),
                "is_check": state.board.is_check(),
                "is_checkmate": state.board.is_checkmate(),
                "is_stalemate": state.board.is_stalemate(),
                "is_game_over": state.is_terminal(),
                "phase": state.get_phase(),
                "evaluation": state.evaluate(),
                "result": state.board.result() if state.is_terminal() else None,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ai_move", methods=["POST"])
def ai_move():
    """Get AI move using multi-agent ensemble."""
    if not CHESS_AVAILABLE:
        return jsonify({"error": "python-chess not installed"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    game_id = data.get("game_id")
    fen = data.get("fen")
    time_limit = data.get("time_limit_ms", 5000)

    if not fen:
        return jsonify({"error": "FEN required"}), 400

    game_id, ensemble = get_or_create_ensemble(game_id)

    try:
        state = ChessState.from_fen(fen)

        if state.is_terminal():
            return (
                jsonify(
                    {
                        "error": "Game is over",
                        "result": state.board.result(),
                    }
                ),
                400,
            )

        # Run ensemble asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            selected_move, metadata = loop.run_until_complete(ensemble.select_move(state, time_limit))
        finally:
            loop.close()

        if not selected_move:
            return jsonify({"error": "No move found"}), 500

        # Apply the move
        new_state = state.apply_action(selected_move)

        return jsonify(
            {
                "game_id": game_id,
                "move": selected_move,
                "fen": new_state.get_fen(),
                "turn": "white" if new_state.board.turn else "black",
                "legal_moves": new_state.get_legal_actions(),
                "is_check": new_state.board.is_check(),
                "is_checkmate": new_state.board.is_checkmate(),
                "is_stalemate": new_state.board.is_stalemate(),
                "is_game_over": new_state.is_terminal(),
                "phase": new_state.get_phase(),
                "evaluation": new_state.evaluate(),
                "result": new_state.board.result() if new_state.is_terminal() else None,
                "ai_analysis": metadata,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_position():
    """Analyze a position without making a move."""
    if not CHESS_AVAILABLE:
        return jsonify({"error": "python-chess not installed"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "FEN required"}), 400

    try:
        state = ChessState.from_fen(fen)

        # Quick analysis
        analysis = {
            "fen": state.get_fen(),
            "turn": "white" if state.board.turn else "black",
            "phase": state.get_phase(),
            "evaluation": state.evaluate(),
            "is_check": state.board.is_check(),
            "is_checkmate": state.board.is_checkmate(),
            "is_stalemate": state.board.is_stalemate(),
            "legal_moves_count": len(state.get_legal_actions()),
            "threats": state.get_threats(),
        }

        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dashboard/summary")
def dashboard_summary():
    """Get dashboard summary."""
    return jsonify(_dashboard.get_summary())


@app.route("/api/dashboard/agents")
def dashboard_agents():
    """Get agent comparison data."""
    return jsonify(_dashboard.get_agent_comparison())


@app.route("/api/dashboard/trend")
def dashboard_trend():
    """Get confidence trend data."""
    return jsonify(_dashboard.get_confidence_trend())


@app.route("/api/dashboard/game/<game_id>")
def dashboard_game(game_id: str):
    """Get detailed game analysis."""
    return jsonify(_dashboard.get_game_details(game_id))


@app.route("/api/dashboard/export")
def dashboard_export():
    """Export dashboard data."""
    try:
        path = _dashboard.export_json()
        return jsonify({"status": "ok", "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/html")
def dashboard_html():
    """Get HTML dashboard report."""
    html = _dashboard.generate_html_report()
    return Response(html, mimetype="text/html")


@app.route("/api/game/<game_id>/learning")
def game_learning(game_id: str):
    """Get learning summary for a game."""
    if game_id not in _ensembles:
        return jsonify({"error": "Game not found"}), 404

    ensemble = _ensembles[game_id]
    return jsonify(ensemble.get_learning_summary())


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run the Flask server."""
    print(f"Starting Chess Demo server at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess Demo Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    run_server(args.host, args.port, args.debug)
