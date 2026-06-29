"""
Chess MCP (Model Context Protocol) Tools.

Provides chess-specific tools for the MCP server, enabling external
callers to analyse positions, suggest moves, and query game status
through the Multi-Agent MCTS framework.

Uses Pydantic validation for inputs (same pattern as tools/mcp/server.py).
"""

from __future__ import annotations

import re
from typing import Any

try:
    from pydantic import BaseModel, Field, field_validator

    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Lightweight stubs when pydantic is not installed
    PYDANTIC_AVAILABLE = False

    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            return {"type": "object"}

    def Field(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
        return kwargs.get("default")

    def field_validator(*args: Any, **kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator


# ---------------------------------------------------------------------------
# Input models with Pydantic validation
# ---------------------------------------------------------------------------

FEN_INITIAL = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_PATTERN = re.compile(r"^[rnbqkpRNBQKP1-8/]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+$")
UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

DEFAULT_ANALYSIS_DEPTH = 8
DEFAULT_SUGGEST_COUNT = 5
MAX_SUGGEST_COUNT = 20
MAX_ANALYSIS_DEPTH = 50


class AnalyzePositionInput(BaseModel):
    """Input for analysing a chess position."""

    fen: str = Field(
        default=FEN_INITIAL,
        min_length=10,
        max_length=200,
        description="FEN string of the position to analyse",
    )
    depth: int = Field(
        default=DEFAULT_ANALYSIS_DEPTH,
        ge=1,
        le=MAX_ANALYSIS_DEPTH,
        description="MCTS depth / analysis iterations",
    )

    @field_validator("fen")
    @classmethod
    def validate_fen(cls, v: str) -> str:
        if not FEN_PATTERN.match(v.strip()):
            raise ValueError(f"Invalid FEN string: {v!r}")
        return v.strip()


class SuggestMovesInput(BaseModel):
    """Input for suggesting top N moves."""

    fen: str = Field(
        default=FEN_INITIAL,
        min_length=10,
        max_length=200,
        description="FEN string",
    )
    num_moves: int = Field(
        default=DEFAULT_SUGGEST_COUNT,
        ge=1,
        le=MAX_SUGGEST_COUNT,
        description="Number of moves to suggest",
    )

    @field_validator("fen")
    @classmethod
    def validate_fen(cls, v: str) -> str:
        if not FEN_PATTERN.match(v.strip()):
            raise ValueError(f"Invalid FEN string: {v!r}")
        return v.strip()


class EvaluateMoveInput(BaseModel):
    """Input for evaluating a specific move."""

    fen: str = Field(..., min_length=10, max_length=200, description="FEN string")
    move: str = Field(
        ...,
        min_length=4,
        max_length=5,
        description="Move in UCI format (e.g., e2e4)",
    )

    @field_validator("fen")
    @classmethod
    def validate_fen(cls, v: str) -> str:
        if not FEN_PATTERN.match(v.strip()):
            raise ValueError(f"Invalid FEN string: {v!r}")
        return v.strip()

    @field_validator("move")
    @classmethod
    def validate_move(cls, v: str) -> str:
        if not UCI_PATTERN.match(v.strip()):
            raise ValueError(f"Invalid UCI move: {v!r}")
        return v.strip()


class GameStatusInput(BaseModel):
    """Input for checking game status."""

    fen: str = Field(..., min_length=10, max_length=200, description="FEN string")

    @field_validator("fen")
    @classmethod
    def validate_fen(cls, v: str) -> str:
        if not FEN_PATTERN.match(v.strip()):
            raise ValueError(f"Invalid FEN string: {v!r}")
        return v.strip()


class PositionFeaturesInput(BaseModel):
    """Input for extracting position features."""

    fen: str = Field(..., min_length=10, max_length=200, description="FEN string")

    @field_validator("fen")
    @classmethod
    def validate_fen(cls, v: str) -> str:
        if not FEN_PATTERN.match(v.strip()):
            raise ValueError(f"Invalid FEN string: {v!r}")
        return v.strip()


# ---------------------------------------------------------------------------
# Tool definitions (for MCP server registration)
# ---------------------------------------------------------------------------


def get_chess_tool_definitions() -> list[dict[str, Any]]:
    """Return MCP tool definitions for chess tools."""
    return [
        {
            "name": "chess_analyze_position",
            "description": "Analyse a chess position using multi-agent MCTS framework",
            "inputSchema": AnalyzePositionInput.model_json_schema(),
        },
        {
            "name": "chess_suggest_moves",
            "description": "Suggest top N candidate moves for a chess position",
            "inputSchema": SuggestMovesInput.model_json_schema(),
        },
        {
            "name": "chess_evaluate_move",
            "description": "Evaluate a specific move in a chess position",
            "inputSchema": EvaluateMoveInput.model_json_schema(),
        },
        {
            "name": "chess_game_status",
            "description": "Check the status of a chess game (check/checkmate/stalemate/draw)",
            "inputSchema": GameStatusInput.model_json_schema(),
        },
        {
            "name": "chess_position_features",
            "description": "Extract features from a chess position (phase, material, etc.)",
            "inputSchema": PositionFeaturesInput.model_json_schema(),
        },
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def handle_analyze_position(args: dict[str, Any], engine: Any) -> dict[str, Any]:
    """Handle chess_analyze_position tool call."""
    inp = AnalyzePositionInput(**args)
    try:
        analysis = await engine.analyze_position(inp.fen)
        return {
            "success": True,
            "best_move": analysis.best_move,
            "candidate_moves": [
                {"move": c.move, "score": c.score, "agent": c.agent_name} for c in analysis.candidate_moves
            ],
            "routing": {
                "primary_agent": analysis.routing_decision.primary_agent,
                "game_phase": analysis.routing_decision.game_phase,
                "confidence": analysis.routing_decision.confidence,
            },
            "consensus_move": analysis.consensus_move,
            "total_time_ms": analysis.total_time_ms,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


async def handle_suggest_moves(args: dict[str, Any], engine: Any) -> dict[str, Any]:
    """Handle chess_suggest_moves tool call."""
    inp = SuggestMovesInput(**args)
    try:
        analysis = await engine.analyze_position(inp.fen)
        moves = [
            {"move": c.move, "score": c.score, "agent": c.agent_name, "reasoning": c.reasoning[:100]}
            for c in analysis.candidate_moves[: inp.num_moves]
        ]
        return {"success": True, "moves": moves, "count": len(moves)}
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


async def handle_evaluate_move(args: dict[str, Any], engine: Any) -> dict[str, Any]:
    """Handle chess_evaluate_move tool call."""
    inp = EvaluateMoveInput(**args)
    try:
        analysis = await engine.analyze_position(inp.fen)
        # Find the requested move among candidates
        for candidate in analysis.candidate_moves:
            if candidate.move == inp.move:
                return {
                    "success": True,
                    "move": inp.move,
                    "score": candidate.score,
                    "reasoning": candidate.reasoning,
                    "is_best": candidate.move == analysis.best_move,
                }
        # Move not among top candidates — return basic evaluation
        return {
            "success": True,
            "move": inp.move,
            "score": 0.3,
            "reasoning": "Move was not among the top candidates",
            "is_best": False,
            "best_move": analysis.best_move,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__}


async def handle_game_status(args: dict[str, Any]) -> dict[str, Any]:
    """Handle chess_game_status tool call (no engine needed)."""
    inp = GameStatusInput(**args)
    from src.games.chess.llm_chess_engine import CHESS_AVAILABLE

    status: dict[str, Any] = {"fen": inp.fen, "success": True}

    if CHESS_AVAILABLE:
        import chess

        try:
            board = chess.Board(inp.fen)
            status.update(
                {
                    "is_check": board.is_check(),
                    "is_checkmate": board.is_checkmate(),
                    "is_stalemate": board.is_stalemate(),
                    "is_game_over": board.is_game_over(),
                    "legal_moves": len(list(board.legal_moves)),
                    "turn": "white" if board.turn else "black",
                }
            )
            if board.is_game_over():
                outcome = board.outcome()
                if outcome:
                    status["winner"] = (
                        "white" if outcome.winner is True else "black" if outcome.winner is False else "draw"
                    )
                    status["termination"] = outcome.termination.name
        except ValueError as e:
            status = {"success": False, "error": str(e)}
    else:
        # Minimal status from FEN parsing
        parts = inp.fen.split()
        status.update(
            {
                "turn": "white" if len(parts) > 1 and parts[1] == "w" else "black",
                "note": "python-chess not available; limited status",
            }
        )

    return status


async def handle_position_features(args: dict[str, Any]) -> dict[str, Any]:
    """Handle chess_position_features tool call (no engine needed)."""
    inp = PositionFeaturesInput(**args)
    from src.games.chess.llm_chess_engine import (
        PIECE_VALUES,
        LLMChessMetaController,
    )

    fen = inp.fen
    parts = fen.split()
    board_part = parts[0] if parts else ""

    move_number = LLMChessMetaController._get_move_number(fen)
    total_material = LLMChessMetaController._get_total_material(fen)

    white_mat = sum(PIECE_VALUES.get(c, 0) for c in board_part if c.isupper())
    black_mat = sum(PIECE_VALUES.get(c.upper(), 0) for c in board_part if c.islower())

    ctrl = LLMChessMetaController()
    phase = ctrl._classify_phase(move_number, total_material)

    return {
        "success": True,
        "fen": fen,
        "move_number": move_number,
        "game_phase": phase,
        "total_material": total_material,
        "white_material": white_mat,
        "black_material": black_mat,
        "material_balance": white_mat - black_mat,
        "side_to_move": "white" if len(parts) > 1 and parts[1] == "w" else "black",
    }


# ---------------------------------------------------------------------------
# Router for tool calls
# ---------------------------------------------------------------------------


CHESS_TOOL_HANDLERS: dict[str, Any] = {
    "chess_analyze_position": handle_analyze_position,
    "chess_suggest_moves": handle_suggest_moves,
    "chess_evaluate_move": handle_evaluate_move,
    "chess_game_status": handle_game_status,
    "chess_position_features": handle_position_features,
}


async def dispatch_chess_tool(
    name: str,
    arguments: dict[str, Any],
    engine: Any = None,
) -> dict[str, Any]:
    """Dispatch a chess tool call.

    Args:
        name: Tool name (must be in CHESS_TOOL_HANDLERS).
        arguments: Tool arguments.
        engine: LLMChessEngine instance (needed for analysis tools).

    Returns:
        Tool result dictionary.
    """
    handler = CHESS_TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown chess tool: {name}")

    # Tools that don't need an engine
    if name in ("chess_game_status", "chess_position_features"):
        return dict(await handler(arguments))
    # Tools that need the engine
    if engine is None:
        return {"success": False, "error": "LLMChessEngine not initialised"}
    return dict(await handler(arguments, engine))
