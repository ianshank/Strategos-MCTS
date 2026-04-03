"""Unit tests for src/games/chess/mcp_chess_tools.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.games.chess.mcp_chess_tools import (
    CHESS_TOOL_HANDLERS,
    AnalyzePositionInput,
    EvaluateMoveInput,
    GameStatusInput,
    PositionFeaturesInput,
    SuggestMovesInput,
    dispatch_chess_tool,
    get_chess_tool_definitions,
    handle_game_status,
    handle_position_features,
)

VALID_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.mark.unit
class TestInputModels:
    def test_analyze_position_default(self):
        inp = AnalyzePositionInput()
        assert inp.fen == VALID_FEN
        assert inp.depth == 8

    def test_analyze_position_custom(self):
        inp = AnalyzePositionInput(fen=VALID_FEN, depth=16)
        assert inp.depth == 16

    def test_analyze_position_invalid_fen(self):
        with pytest.raises((ValueError, ValidationError)):
            AnalyzePositionInput(fen="invalid fen string")

    def test_suggest_moves_default(self):
        inp = SuggestMovesInput()
        assert inp.num_moves == 5

    def test_evaluate_move_valid(self):
        inp = EvaluateMoveInput(fen=VALID_FEN, move="e2e4")
        assert inp.move == "e2e4"

    def test_evaluate_move_invalid(self):
        with pytest.raises((ValueError, ValidationError)):
            EvaluateMoveInput(fen=VALID_FEN, move="zzzz")

    def test_game_status_input(self):
        inp = GameStatusInput(fen=VALID_FEN)
        assert inp.fen == VALID_FEN

    def test_position_features_input(self):
        inp = PositionFeaturesInput(fen=VALID_FEN)
        assert inp.fen == VALID_FEN


@pytest.mark.unit
class TestGetChessToolDefinitions:
    def test_returns_list(self):
        defs = get_chess_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 5

    def test_tool_names(self):
        defs = get_chess_tool_definitions()
        names = {d["name"] for d in defs}
        assert "chess_analyze_position" in names
        assert "chess_suggest_moves" in names
        assert "chess_evaluate_move" in names
        assert "chess_game_status" in names
        assert "chess_position_features" in names

    def test_tool_has_schema(self):
        defs = get_chess_tool_definitions()
        for d in defs:
            assert "inputSchema" in d
            assert "description" in d


@pytest.mark.unit
class TestHandleGameStatus:
    @pytest.mark.asyncio
    async def test_with_chess_available(self):
        result = await handle_game_status({"fen": VALID_FEN})
        assert result["success"] is True
        # python-chess is available in the env
        assert "turn" in result

    @pytest.mark.asyncio
    async def test_invalid_fen(self):
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            await handle_game_status({"fen": "invalid"})


@pytest.mark.unit
class TestHandlePositionFeatures:
    @pytest.mark.asyncio
    async def test_initial_position(self):
        result = await handle_position_features({"fen": VALID_FEN})
        assert result["success"] is True
        assert result["move_number"] == 1
        assert result["material_balance"] == 0
        assert result["side_to_move"] == "white"


@pytest.mark.unit
class TestDispatchChessTool:
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown chess tool"):
            await dispatch_chess_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_game_status_no_engine(self):
        result = await dispatch_chess_tool("chess_game_status", {"fen": VALID_FEN})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_position_features_no_engine(self):
        result = await dispatch_chess_tool("chess_position_features", {"fen": VALID_FEN})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_analyze_position_no_engine(self):
        result = await dispatch_chess_tool("chess_analyze_position", {"fen": VALID_FEN})
        assert result["success"] is False
        assert "not initialised" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_position_with_engine(self):
        mock_engine = AsyncMock()
        mock_analysis = MagicMock()
        mock_analysis.best_move = "e2e4"
        mock_analysis.candidate_moves = []
        mock_analysis.routing_decision.primary_agent = "mcts"
        mock_analysis.routing_decision.game_phase = "opening"
        mock_analysis.routing_decision.confidence = 0.9
        mock_analysis.consensus_move = "e2e4"
        mock_analysis.total_time_ms = 100.0
        mock_engine.analyze_position = AsyncMock(return_value=mock_analysis)

        result = await dispatch_chess_tool(
            "chess_analyze_position", {"fen": VALID_FEN}, engine=mock_engine
        )
        assert result["success"] is True
        assert result["best_move"] == "e2e4"

    @pytest.mark.asyncio
    async def test_suggest_moves_with_engine(self):
        mock_engine = AsyncMock()
        mock_candidate = MagicMock()
        mock_candidate.move = "e2e4"
        mock_candidate.score = 0.9
        mock_candidate.agent_name = "mcts"
        mock_candidate.reasoning = "Best opening move"
        mock_analysis = MagicMock()
        mock_analysis.candidate_moves = [mock_candidate]
        mock_engine.analyze_position = AsyncMock(return_value=mock_analysis)

        result = await dispatch_chess_tool(
            "chess_suggest_moves", {"fen": VALID_FEN, "num_moves": 3}, engine=mock_engine
        )
        assert result["success"] is True
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_move_found(self):
        mock_engine = AsyncMock()
        mock_candidate = MagicMock()
        mock_candidate.move = "e2e4"
        mock_candidate.score = 0.9
        mock_candidate.reasoning = "Strong opening"
        mock_analysis = MagicMock()
        mock_analysis.best_move = "e2e4"
        mock_analysis.candidate_moves = [mock_candidate]
        mock_engine.analyze_position = AsyncMock(return_value=mock_analysis)

        result = await dispatch_chess_tool(
            "chess_evaluate_move", {"fen": VALID_FEN, "move": "e2e4"}, engine=mock_engine
        )
        assert result["success"] is True
        assert result["is_best"] is True

    @pytest.mark.asyncio
    async def test_evaluate_move_not_found(self):
        mock_engine = AsyncMock()
        mock_analysis = MagicMock()
        mock_analysis.best_move = "e2e4"
        mock_analysis.candidate_moves = []
        mock_engine.analyze_position = AsyncMock(return_value=mock_analysis)

        result = await dispatch_chess_tool(
            "chess_evaluate_move", {"fen": VALID_FEN, "move": "a2a3"}, engine=mock_engine
        )
        assert result["success"] is True
        assert result["is_best"] is False

    @pytest.mark.asyncio
    async def test_handler_dict_complete(self):
        assert len(CHESS_TOOL_HANDLERS) == 5
