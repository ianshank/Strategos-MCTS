"""
Tests for the LLM-powered Chess Demo.

Tests cover the LLM chess engine, meta-controller routing, MCP chess tools,
position helpers, prompt formatting, agent integration, and CLI.

Uses a mock LLM adapter to verify behaviour without API calls.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Mock LLM adapter (same pattern as test_llm_agents.py)
# ---------------------------------------------------------------------------


@dataclass
class MockChessLLMResponse:
    text: str
    usage: dict = field(default_factory=lambda: {"total_tokens": 80, "prompt_tokens": 40, "completion_tokens": 40})
    model: str = "chess-mock"
    raw_response: Any = None
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


class MockChessLLMAdapter:
    """Mock LLM client that returns chess-specific responses."""

    def __init__(self, move: str = "e2e4", score: float = 0.75):
        self._move = move
        self._score = score
        self.call_count = 0
        self.last_prompt = None

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> MockChessLLMResponse:
        self.call_count += 1
        self.last_prompt = prompt
        text = (
            f"### Analysis\nPosition analysis for chess.\n\n"
            f"**Move:** {self._move}\n"
            f"**Score:** {self._score}\n"
            f"**Reasoning:** Strong central move controlling key squares.\n"
        )
        return MockChessLLMResponse(text=text)


def run_async(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLEGAME_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
ENDGAME_FEN = "8/5k2/8/8/8/8/4K3/4R3 w - - 0 50"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


# ---------------------------------------------------------------------------
# LLMChessMetaController tests
# ---------------------------------------------------------------------------


class TestLLMChessMetaController:
    def _make_controller(self):
        from src.games.chess.llm_chess_engine import LLMChessMetaController
        return LLMChessMetaController()

    def test_opening_routes_to_hrm(self):
        ctrl = self._make_controller()
        routing = ctrl.route(INITIAL_FEN)
        assert routing.primary_agent == "hrm"
        assert routing.game_phase == "opening"

    def test_middlegame_routes_to_mcts(self):
        ctrl = self._make_controller()
        # Move 15 with most material still on board
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 15"
        routing = ctrl.route(fen)
        assert routing.game_phase == "middlegame"
        assert routing.primary_agent == "mcts"

    def test_endgame_routes_to_trm(self):
        ctrl = self._make_controller()
        routing = ctrl.route(ENDGAME_FEN)
        assert routing.game_phase == "endgame"
        assert routing.primary_agent == "trm"

    def test_routing_returns_valid_decision(self):
        ctrl = self._make_controller()
        routing = ctrl.route(INITIAL_FEN)
        assert routing.primary_agent in ("hrm", "trm", "mcts")
        assert 0.0 <= routing.confidence <= 1.0
        assert routing.game_phase in ("opening", "middlegame", "endgame")
        assert abs(sum(routing.agent_weights.values()) - 1.0) < 0.01

    def test_custom_weights(self):
        from src.games.chess.llm_chess_engine import LLMChessMetaController
        ctrl = LLMChessMetaController(
            opening_weights={"hrm": 0.1, "trm": 0.1, "mcts": 0.8},
        )
        routing = ctrl.route(INITIAL_FEN)
        assert routing.primary_agent == "mcts"

    def test_reasoning_is_nonempty(self):
        ctrl = self._make_controller()
        routing = ctrl.route(INITIAL_FEN)
        assert len(routing.reasoning) > 0


# ---------------------------------------------------------------------------
# ChessMoveResult / ChessAnalysis tests
# ---------------------------------------------------------------------------


class TestChessDataStructures:
    def test_move_result_creation(self):
        from src.games.chess.llm_chess_engine import ChessMoveResult
        mr = ChessMoveResult(
            move="e2e4", score=0.8, reasoning="Central control",
            agent_name="hrm", confidence=0.8,
        )
        assert mr.move == "e2e4"
        assert mr.score == 0.8

    def test_chess_analysis_creation(self):
        from src.games.chess.llm_chess_engine import ChessAnalysis, ChessMoveResult, RoutingDecision
        analysis = ChessAnalysis(
            best_move="e2e4",
            candidate_moves=[ChessMoveResult("e2e4", 0.8, "test", "hrm")],
            routing_decision=RoutingDecision("hrm", {"hrm": 0.5, "trm": 0.3, "mcts": 0.2}, 0.7, "opening", "test"),
            agent_results={},
        )
        assert analysis.best_move == "e2e4"
        assert len(analysis.candidate_moves) == 1

    def test_routing_decision_creation(self):
        from src.games.chess.llm_chess_engine import RoutingDecision
        rd = RoutingDecision("mcts", {"hrm": 0.2, "trm": 0.3, "mcts": 0.5}, 0.7, "middlegame", "complex")
        assert rd.primary_agent == "mcts"


# ---------------------------------------------------------------------------
# LLMChessEngine tests
# ---------------------------------------------------------------------------


class TestLLMChessEngine:
    def _make_engine(self, move: str = "e2e4"):
        from src.games.chess.llm_chess_engine import LLMChessEngine
        adapter = MockChessLLMAdapter(move=move)
        return LLMChessEngine(model_adapter=adapter), adapter

    def test_analyze_returns_analysis(self):
        engine, _ = self._make_engine()
        analysis = run_async(engine.analyze_position(INITIAL_FEN))
        from src.games.chess.llm_chess_engine import ChessAnalysis
        assert isinstance(analysis, ChessAnalysis)

    def test_analyze_has_best_move(self):
        engine, _ = self._make_engine()
        analysis = run_async(engine.analyze_position(INITIAL_FEN))
        assert len(analysis.best_move) >= 4

    def test_analyze_has_routing_decision(self):
        engine, _ = self._make_engine()
        analysis = run_async(engine.analyze_position(INITIAL_FEN))
        assert analysis.routing_decision is not None
        assert analysis.routing_decision.game_phase == "opening"

    def test_analyze_has_agent_results(self):
        engine, _ = self._make_engine()
        analysis = run_async(engine.analyze_position(INITIAL_FEN))
        assert len(analysis.agent_results) > 0

    def test_analyze_uses_all_three_agents(self):
        engine, _ = self._make_engine()
        analysis = run_async(engine.analyze_position(INITIAL_FEN))
        agents_used = analysis.metadata.get("agents_used", [])
        # Should use hrm, trm, mcts
        assert "hrm" in agents_used
        assert "trm" in agents_used
        assert "mcts" in agents_used

    def test_get_best_move_returns_uci(self):
        engine, _ = self._make_engine("d2d4")
        move = run_async(engine.get_best_move(INITIAL_FEN))
        assert isinstance(move, str)
        assert len(move) >= 4

    def test_engine_tracks_stats(self):
        engine, _ = self._make_engine()
        run_async(engine.analyze_position(INITIAL_FEN))
        stats = engine.stats
        assert stats["move_count"] == 1

    def test_engine_configurable_temperature(self):
        from src.games.chess.llm_chess_engine import LLMChessEngine
        adapter = MockChessLLMAdapter()
        engine = LLMChessEngine(model_adapter=adapter, temperature=0.9)
        assert engine._temperature == 0.9

    def test_analyze_timing(self):
        engine, _ = self._make_engine()
        analysis = run_async(engine.analyze_position(INITIAL_FEN))
        assert analysis.total_time_ms > 0

    def test_multiple_analyses_increment_stats(self):
        engine, _ = self._make_engine()
        run_async(engine.analyze_position(INITIAL_FEN))
        run_async(engine.analyze_position(MIDDLEGAME_FEN))
        assert engine.stats["move_count"] == 2


# ---------------------------------------------------------------------------
# Chess-specific agent tests
# ---------------------------------------------------------------------------


class TestChessAgents:
    def test_hrm_agent_returns_result(self):
        from src.games.chess.llm_chess_engine import LLMChessHRMAgent
        adapter = MockChessLLMAdapter()
        agent = LLMChessHRMAgent(adapter)
        result = run_async(agent.process(query=INITIAL_FEN))
        assert "response" in result
        assert "metadata" in result
        assert "move" in result["metadata"]

    def test_trm_agent_returns_result(self):
        from src.games.chess.llm_chess_engine import LLMChessTRMAgent
        adapter = MockChessLLMAdapter()
        agent = LLMChessTRMAgent(adapter)
        result = run_async(agent.process(query=INITIAL_FEN))
        assert "response" in result
        assert "metadata" in result
        assert "move" in result["metadata"]

    def test_mcts_agent_returns_result(self):
        from src.games.chess.llm_chess_engine import LLMChessMCTSAgent
        adapter = MockChessLLMAdapter()
        agent = LLMChessMCTSAgent(adapter)
        result = run_async(agent.process(query=INITIAL_FEN))
        assert "response" in result
        assert "metadata" in result
        assert "move" in result["metadata"]

    def test_mcts_agent_runs_all_strategies(self):
        from src.games.chess.llm_chess_engine import LLMChessMCTSAgent
        adapter = MockChessLLMAdapter()
        agent = LLMChessMCTSAgent(adapter, strategies=["tactical", "positional"])
        result = run_async(agent.process(query=INITIAL_FEN))
        all_strats = result["metadata"].get("all_strategies", [])
        assert len(all_strats) == 2


# ---------------------------------------------------------------------------
# Position helper tests
# ---------------------------------------------------------------------------


class TestPositionHelpers:
    def test_fen_to_board_ascii(self):
        from src.games.chess.llm_chess_engine import fen_to_board_ascii
        board = fen_to_board_ascii(INITIAL_FEN)
        assert isinstance(board, str)
        assert len(board) > 0
        assert "a" in board  # file labels

    def test_describe_position(self):
        from src.games.chess.llm_chess_engine import describe_position
        desc = describe_position(INITIAL_FEN)
        assert "White" in desc
        assert "Opening" in desc
        assert INITIAL_FEN in desc

    def test_extract_uci_move_explicit(self):
        from src.games.chess.llm_chess_engine import extract_uci_move
        assert extract_uci_move("**Move:** e2e4") == "e2e4"
        assert extract_uci_move("**Recommended move:** d2d4") == "d2d4"
        assert extract_uci_move("**Best Move:** g1f3") == "g1f3"

    def test_extract_uci_move_fallback(self):
        from src.games.chess.llm_chess_engine import extract_uci_move
        assert extract_uci_move("I think e2e4 is best") == "e2e4"

    def test_extract_uci_move_none(self):
        from src.games.chess.llm_chess_engine import extract_uci_move
        assert extract_uci_move("No move here") is None

    def test_extract_score(self):
        from src.games.chess.llm_chess_engine import extract_score
        assert extract_score("**Score:** 0.85") == 0.85
        assert extract_score("**Confidence:** 0.7") == 0.7

    def test_extract_score_default(self):
        from src.games.chess.llm_chess_engine import extract_score
        assert extract_score("No score here") == 0.5

    def test_extract_score_clamped(self):
        from src.games.chess.llm_chess_engine import extract_score
        assert extract_score("**Score:** 1.5") == 1.0
        assert extract_score("**Score:** -0.3") == 0.5  # negative doesn't match pattern

    def test_get_legal_moves_without_chess(self):
        from src.games.chess.llm_chess_engine import CHESS_AVAILABLE, get_legal_moves_list
        if not CHESS_AVAILABLE:
            assert get_legal_moves_list(INITIAL_FEN) is None


# ---------------------------------------------------------------------------
# MCP Chess Tools tests
# ---------------------------------------------------------------------------


try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False


class TestMCPChessTools:
    def test_tool_definitions_count(self):
        from src.games.chess.mcp_chess_tools import get_chess_tool_definitions
        tools = get_chess_tool_definitions()
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert "chess_analyze_position" in names
        assert "chess_suggest_moves" in names
        assert "chess_evaluate_move" in names
        assert "chess_game_status" in names
        assert "chess_position_features" in names

    @pytest.mark.skipif(not _HAS_PYDANTIC, reason="pydantic not installed")
    def test_analyze_position_input_validation(self):
        from src.games.chess.mcp_chess_tools import AnalyzePositionInput
        inp = AnalyzePositionInput(fen=INITIAL_FEN, depth=10)
        assert inp.depth == 10

    @pytest.mark.skipif(not _HAS_PYDANTIC, reason="pydantic not installed")
    def test_analyze_position_invalid_fen(self):
        from src.games.chess.mcp_chess_tools import AnalyzePositionInput
        with pytest.raises((ValueError, TypeError)):
            AnalyzePositionInput(fen="not a fen")

    @pytest.mark.skipif(not _HAS_PYDANTIC, reason="pydantic not installed")
    def test_evaluate_move_validation(self):
        from src.games.chess.mcp_chess_tools import EvaluateMoveInput
        inp = EvaluateMoveInput(fen=INITIAL_FEN, move="e2e4")
        assert inp.move == "e2e4"

    @pytest.mark.skipif(not _HAS_PYDANTIC, reason="pydantic not installed")
    def test_evaluate_move_invalid_move(self):
        from src.games.chess.mcp_chess_tools import EvaluateMoveInput
        with pytest.raises((ValueError, TypeError)):
            EvaluateMoveInput(fen=INITIAL_FEN, move="xyz")

    def test_position_features_handler(self):
        from src.games.chess.mcp_chess_tools import handle_position_features
        result = run_async(handle_position_features({"fen": INITIAL_FEN}))
        assert result["success"] is True
        assert result["game_phase"] == "opening"
        assert result["side_to_move"] == "white"
        assert result["total_material"] > 0

    def test_game_status_handler(self):
        from src.games.chess.mcp_chess_tools import handle_game_status
        result = run_async(handle_game_status({"fen": INITIAL_FEN}))
        assert result["success"] is True
        assert "turn" in result or "fen" in result

    def test_dispatch_unknown_tool(self):
        from src.games.chess.mcp_chess_tools import dispatch_chess_tool
        with pytest.raises(ValueError, match="Unknown chess tool"):
            run_async(dispatch_chess_tool("nonexistent", {}))

    def test_analyze_without_engine(self):
        from src.games.chess.mcp_chess_tools import dispatch_chess_tool
        result = run_async(dispatch_chess_tool(
            "chess_analyze_position", {"fen": INITIAL_FEN}, engine=None,
        ))
        assert result["success"] is False
        assert "not initialised" in result["error"]


# ---------------------------------------------------------------------------
# Agent integration tests (ParallelAgent, SequentialAgent)
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    def test_parallel_agent_runs_hrm_and_trm(self):
        from src.framework.agents.base import ParallelAgent
        from src.games.chess.llm_chess_engine import LLMChessHRMAgent, LLMChessTRMAgent
        adapter = MockChessLLMAdapter()
        hrm = LLMChessHRMAgent(adapter)
        trm = LLMChessTRMAgent(adapter)
        parallel = ParallelAgent(adapter, name="TestParallel", sub_agents=[hrm, trm])
        result = run_async(parallel.process(query=INITIAL_FEN))
        assert "response" in result
        assert result["metadata"].get("success", True) is True

    def test_sequential_agent_chains_agents(self):
        from src.framework.agents.base import SequentialAgent
        from src.games.chess.llm_chess_engine import LLMChessHRMAgent, LLMChessTRMAgent
        adapter = MockChessLLMAdapter()
        hrm = LLMChessHRMAgent(adapter)
        trm = LLMChessTRMAgent(adapter)
        seq = SequentialAgent(adapter, name="TestSequential", sub_agents=[hrm, trm])
        result = run_async(seq.process(query=INITIAL_FEN))
        assert "response" in result

    def test_all_agents_produce_compatible_output(self):
        from src.games.chess.llm_chess_engine import (
            LLMChessHRMAgent,
            LLMChessMCTSAgent,
            LLMChessTRMAgent,
        )
        adapter = MockChessLLMAdapter()
        agents = [
            LLMChessHRMAgent(adapter),
            LLMChessTRMAgent(adapter),
            LLMChessMCTSAgent(adapter),
        ]
        for agent in agents:
            result = run_async(agent.process(query=INITIAL_FEN))
            assert "response" in result
            assert "metadata" in result
            assert "move" in result["metadata"]

    def test_engine_backward_compatible_with_base(self):
        """Engine agents inherit from AsyncAgentBase and support stats."""
        from src.games.chess.llm_chess_engine import LLMChessHRMAgent
        adapter = MockChessLLMAdapter()
        agent = LLMChessHRMAgent(adapter)
        run_async(agent.process(query=INITIAL_FEN))
        stats = agent.stats
        assert stats["request_count"] == 1
        assert stats["total_processing_time_ms"] > 0


# ---------------------------------------------------------------------------
# Constants and configuration tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_strategy_prompts_exist(self):
        from src.games.chess.llm_chess_engine import MCTS_CHESS_STRATEGIES
        assert len(MCTS_CHESS_STRATEGIES) >= 4
        for name in ("tactical", "positional", "prophylactic", "endgame"):
            assert name in MCTS_CHESS_STRATEGIES

    def test_configurable_defaults(self):
        from src.games.chess.llm_chess_engine import (
            DEFAULT_CHESS_MAX_TOKENS,
            DEFAULT_CHESS_TEMPERATURE,
            DEFAULT_MCTS_DEPTH,
            DEFAULT_TOP_MOVES,
        )
        assert DEFAULT_CHESS_TEMPERATURE > 0
        assert DEFAULT_CHESS_MAX_TOKENS > 0
        assert DEFAULT_MCTS_DEPTH > 0
        assert DEFAULT_TOP_MOVES > 0

    def test_piece_values(self):
        from src.games.chess.llm_chess_engine import PIECE_VALUES
        assert PIECE_VALUES["Q"] == 9
        assert PIECE_VALUES["P"] == 1
        assert PIECE_VALUES["K"] == 0

    def test_uci_pattern(self):
        from src.games.chess.llm_chess_engine import UCI_MOVE_PATTERN
        assert UCI_MOVE_PATTERN.match("e2e4")
        assert UCI_MOVE_PATTERN.match("a7a8q")
        assert not UCI_MOVE_PATTERN.match("ee4")
        assert not UCI_MOVE_PATTERN.match("e2e9")


# ---------------------------------------------------------------------------
# CLI integration tests (subprocess)
# ---------------------------------------------------------------------------


class TestChessDemoCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "chess_demo.py", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Chess Demo" in result.stdout

    def test_mock_analyze(self):
        result = subprocess.run(
            [sys.executable, "chess_demo.py", "--analyze"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Best Move" in result.stdout or "best_move" in result.stdout

    def test_json_output(self):
        import json

        result = subprocess.run(
            [sys.executable, "chess_demo.py", "--json", "--analyze"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "best_move" in data
        assert "candidates" in data

    def test_mcp_tools_flag(self):
        result = subprocess.run(
            [sys.executable, "chess_demo.py", "--mcp-tools"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "chess_analyze_position" in result.stdout
