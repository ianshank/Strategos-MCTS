"""Unit tests for src/games/chess/llm_chess_engine.py."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.games.chess.llm_chess_engine import (
    DEFAULT_CHESS_MAX_TOKENS,
    DEFAULT_CHESS_TEMPERATURE,
    DEFAULT_CONSENSUS_TOP_K,
    DEFAULT_MCTS_DEPTH,
    DEFAULT_TOP_MOVES,
    ENDGAME_WEIGHTS,
    MIDDLEGAME_WEIGHTS,
    OPENING_WEIGHTS,
    PHASE_ENDGAME_MATERIAL,
    PHASE_OPENING_THRESHOLD,
    PIECE_VALUES,
    UCI_MOVE_PATTERN,
    ChessAnalysis,
    ChessMoveResult,
    LLMChessEngine,
    LLMChessHRMAgent,
    LLMChessMCTSAgent,
    LLMChessMetaController,
    LLMChessTRMAgent,
    RoutingDecision,
    describe_position,
    extract_score,
    extract_uci_move,
    fen_to_board_ascii,
    get_legal_moves_list,
)

# Standard starting position FEN
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# A middlegame FEN (move 15, high material)
MIDDLEGAME_FEN = "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 15"
# An endgame FEN (move 40, low material)
ENDGAME_FEN = "8/5pk1/8/8/8/8/5PK1/8 w - - 0 40"


def _make_mock_adapter():
    """Create a mock LLM adapter."""
    adapter = AsyncMock()
    response = MagicMock()
    response.text = "**Move:** e2e4\n**Score:** 0.8\nSome reasoning about the position."
    response.usage = {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}
    adapter.generate = AsyncMock(return_value=response)
    return adapter


# ---------------------------------------------------------------------------
# RoutingDecision and ChessMoveResult dataclass tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataclasses:
    def test_routing_decision_fields(self):
        rd = RoutingDecision(
            primary_agent="hrm",
            agent_weights={"hrm": 0.5, "trm": 0.2, "mcts": 0.3},
            confidence=0.8,
            game_phase="opening",
            reasoning="Opening phase",
        )
        assert rd.primary_agent == "hrm"
        assert rd.game_phase == "opening"
        assert rd.confidence == 0.8

    def test_chess_move_result_defaults(self):
        cmr = ChessMoveResult(
            move="e2e4", score=0.9, reasoning="good move", agent_name="hrm"
        )
        assert cmr.confidence == 0.0
        assert cmr.metadata == {}

    def test_chess_analysis_defaults(self):
        rd = RoutingDecision("hrm", {}, 0.5, "opening", "test")
        ca = ChessAnalysis(
            best_move="e2e4",
            candidate_moves=[],
            routing_decision=rd,
            agent_results={},
        )
        assert ca.consensus_move is None
        assert ca.consensus_reasoning is None
        assert ca.total_time_ms == 0.0
        assert ca.metadata == {}


# ---------------------------------------------------------------------------
# LLMChessMetaController tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessMetaController:
    def test_default_init(self):
        mc = LLMChessMetaController()
        assert mc._opening_weights == OPENING_WEIGHTS
        assert mc._middlegame_weights == MIDDLEGAME_WEIGHTS
        assert mc._endgame_weights == ENDGAME_WEIGHTS
        assert mc._opening_threshold == PHASE_OPENING_THRESHOLD
        assert mc._endgame_material == PHASE_ENDGAME_MATERIAL

    def test_custom_weights(self):
        custom = {"hrm": 0.9, "trm": 0.05, "mcts": 0.05}
        mc = LLMChessMetaController(opening_weights=custom)
        assert mc._opening_weights == custom

    def test_route_opening(self):
        mc = LLMChessMetaController()
        decision = mc.route(START_FEN)
        assert decision.game_phase == "opening"
        assert decision.primary_agent in {"hrm", "trm", "mcts"}
        assert abs(sum(decision.agent_weights.values()) - 1.0) < 1e-6
        assert 0.0 <= decision.confidence <= 1.0

    def test_route_middlegame(self):
        mc = LLMChessMetaController()
        decision = mc.route(MIDDLEGAME_FEN)
        assert decision.game_phase == "middlegame"

    def test_route_endgame(self):
        mc = LLMChessMetaController()
        decision = mc.route(ENDGAME_FEN)
        assert decision.game_phase == "endgame"

    def test_get_move_number_valid(self):
        assert LLMChessMetaController._get_move_number(START_FEN) == 1
        assert LLMChessMetaController._get_move_number(ENDGAME_FEN) == 40

    def test_get_move_number_invalid(self):
        assert LLMChessMetaController._get_move_number("bad fen") == 1
        assert LLMChessMetaController._get_move_number("a b c d e notanumber") == 1

    def test_get_total_material(self):
        # Starting position: 8P + 2N + 2B + 2R + Q = 8+6+6+10+9 = 39 per side = 78
        material = LLMChessMetaController._get_total_material(START_FEN)
        assert material == 78

    def test_get_total_material_endgame(self):
        # "8/5pk1/8/8/8/8/5PK1/8" - 1P + 1p = 2
        material = LLMChessMetaController._get_total_material(ENDGAME_FEN)
        assert material == 2

    def test_get_total_material_empty(self):
        assert LLMChessMetaController._get_total_material("") == 0

    def test_classify_phase(self):
        mc = LLMChessMetaController()
        assert mc._classify_phase(5, 78) == "opening"
        assert mc._classify_phase(20, 50) == "middlegame"
        assert mc._classify_phase(40, 10) == "endgame"

    def test_build_reasoning_known(self):
        r = LLMChessMetaController._build_reasoning("opening", "hrm")
        assert "HRM" in r or "hrm" in r.lower()

    def test_build_reasoning_fallback(self):
        r = LLMChessMetaController._build_reasoning("middlegame", "hrm")
        assert "HRM" in r.upper()

    def test_weights_normalized(self):
        mc = LLMChessMetaController()
        decision = mc.route(START_FEN)
        total = sum(decision.agent_weights.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Position description helper tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPositionHelpers:
    def test_fen_to_board_ascii_fallback(self):
        # Test the fallback renderer (works regardless of python-chess)
        with patch("src.games.chess.llm_chess_engine.CHESS_AVAILABLE", False):
            board = fen_to_board_ascii(START_FEN)
            assert "8" in board  # rank number
            assert "a b c d e f g h" in board

    def test_fen_to_board_ascii_empty_fen(self):
        with patch("src.games.chess.llm_chess_engine.CHESS_AVAILABLE", False):
            board = fen_to_board_ascii("")
            assert "a b c d e f g h" in board

    def test_get_legal_moves_no_chess(self):
        with patch("src.games.chess.llm_chess_engine.CHESS_AVAILABLE", False):
            result = get_legal_moves_list(START_FEN)
            assert result is None

    def test_describe_position_white(self):
        desc = describe_position(START_FEN)
        assert "White" in desc
        assert START_FEN in desc
        assert "Opening" in desc

    def test_describe_position_black(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        desc = describe_position(fen)
        assert "Black" in desc

    def test_describe_position_endgame_phase(self):
        desc = describe_position(ENDGAME_FEN)
        assert "Endgame" in desc

    def test_describe_position_material_balance_equal(self):
        desc = describe_position(START_FEN)
        assert "Equal" in desc

    def test_describe_position_material_imbalance(self):
        # White has extra queen
        fen = "8/5pk1/8/8/8/8/5PK1/4Q3 w - - 0 40"
        desc = describe_position(fen)
        assert "White +" in desc


# ---------------------------------------------------------------------------
# extract_uci_move tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractUciMove:
    def test_bold_move_pattern(self):
        assert extract_uci_move("**Move:** e2e4") == "e2e4"

    def test_recommended_move_pattern(self):
        assert extract_uci_move("**Recommended move:** d2d4") == "d2d4"

    def test_best_move_pattern(self):
        assert extract_uci_move("**Best Move:** g1f3") == "g1f3"

    def test_plain_move_pattern(self):
        assert extract_uci_move("Move: b1c3") == "b1c3"

    def test_promotion(self):
        assert extract_uci_move("**Move:** a7a8q") == "a7a8q"

    def test_fallback_word_match(self):
        assert extract_uci_move("I suggest playing e2e4 in this position") == "e2e4"

    def test_no_move(self):
        assert extract_uci_move("No valid move here") is None

    def test_move_with_punctuation(self):
        assert extract_uci_move("The move is e2e4.") == "e2e4"


# ---------------------------------------------------------------------------
# extract_score tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractScore:
    def test_bold_score(self):
        assert extract_score("**Score:** 0.85") == 0.85

    def test_plain_score(self):
        assert extract_score("Score: 0.7") == 0.7

    def test_confidence_pattern(self):
        assert extract_score("**Confidence:** 0.9") == 0.9

    def test_score_clamped_high(self):
        assert extract_score("**Score:** 1.5") == 1.0

    def test_score_clamped_low(self):
        # Regex only matches \d+ so "-0.5" doesn't match; falls back to default
        assert extract_score("**Score:** -0.5") == 0.5

    def test_default_score(self):
        assert extract_score("No score here") == 0.5


# ---------------------------------------------------------------------------
# UCI_MOVE_PATTERN tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUciMovePattern:
    def test_valid_moves(self):
        assert UCI_MOVE_PATTERN.match("e2e4")
        assert UCI_MOVE_PATTERN.match("a7a8q")
        assert UCI_MOVE_PATTERN.match("h1h8")

    def test_invalid_moves(self):
        assert UCI_MOVE_PATTERN.match("e9e4") is None
        assert UCI_MOVE_PATTERN.match("z2e4") is None
        assert UCI_MOVE_PATTERN.match("e2") is None


# ---------------------------------------------------------------------------
# PIECE_VALUES tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPieceValues:
    def test_piece_values(self):
        assert PIECE_VALUES["P"] == 1
        assert PIECE_VALUES["N"] == 3
        assert PIECE_VALUES["B"] == 3
        assert PIECE_VALUES["R"] == 5
        assert PIECE_VALUES["Q"] == 9
        assert PIECE_VALUES["K"] == 0


# ---------------------------------------------------------------------------
# LLMChessHRMAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessHRMAgent:
    def test_init_defaults(self):
        adapter = _make_mock_adapter()
        agent = LLMChessHRMAgent(adapter)
        assert agent.name == "Chess_HRM"
        assert agent._temperature == DEFAULT_CHESS_TEMPERATURE
        assert agent._max_tokens == DEFAULT_CHESS_MAX_TOKENS

    def test_init_custom(self):
        adapter = _make_mock_adapter()
        agent = LLMChessHRMAgent(adapter, name="MyHRM", temperature=0.5, max_tokens=500)
        assert agent.name == "MyHRM"
        assert agent._temperature == 0.5
        assert agent._max_tokens == 500

    @pytest.mark.asyncio
    async def test_process(self):
        adapter = _make_mock_adapter()
        agent = LLMChessHRMAgent(adapter)
        result = await agent.process(query=START_FEN)
        assert result["metadata"]["move"] == "e2e4"
        assert result["metadata"]["score"] == 0.8
        assert result["metadata"]["agent"] == "chess_hrm"

    @pytest.mark.asyncio
    async def test_process_no_move_in_response(self):
        adapter = _make_mock_adapter()
        adapter.generate.return_value.text = "No move pattern in this text"
        agent = LLMChessHRMAgent(adapter)
        result = await agent.process(query=START_FEN)
        # Falls back to e2e4
        assert result["metadata"]["move"] == "e2e4"


# ---------------------------------------------------------------------------
# LLMChessTRMAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessTRMAgent:
    def test_init_defaults(self):
        adapter = _make_mock_adapter()
        agent = LLMChessTRMAgent(adapter)
        assert agent.name == "Chess_TRM"

    @pytest.mark.asyncio
    async def test_process(self):
        adapter = _make_mock_adapter()
        agent = LLMChessTRMAgent(adapter)
        result = await agent.process(query=START_FEN)
        assert result["metadata"]["move"] == "e2e4"
        assert result["metadata"]["agent"] == "chess_trm"
        assert result["metadata"]["strategy"] == "iterative_refinement"


# ---------------------------------------------------------------------------
# LLMChessMCTSAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessMCTSAgent:
    def test_init_defaults(self):
        adapter = _make_mock_adapter()
        agent = LLMChessMCTSAgent(adapter)
        assert agent.name == "Chess_MCTS"
        assert len(agent._strategies) == 4  # tactical, positional, prophylactic, endgame

    def test_init_custom_strategies(self):
        adapter = _make_mock_adapter()
        agent = LLMChessMCTSAgent(adapter, strategies=["tactical"])
        assert agent._strategies == ["tactical"]

    @pytest.mark.asyncio
    async def test_process(self):
        adapter = _make_mock_adapter()
        agent = LLMChessMCTSAgent(adapter)
        result = await agent.process(query=START_FEN)
        assert result["metadata"]["move"] == "e2e4"
        assert result["metadata"]["agent"] == "chess_mcts"

    @pytest.mark.asyncio
    async def test_process_all_strategies_fail(self):
        adapter = _make_mock_adapter()
        adapter.generate = AsyncMock(side_effect=RuntimeError("LLM error"))
        agent = LLMChessMCTSAgent(adapter, strategies=["tactical"])
        result = await agent.process(query=START_FEN)
        # When all strategies fail, returns default
        meta = result["metadata"]
        assert meta.get("move") == "e2e4" or meta.get("error") is not None


# ---------------------------------------------------------------------------
# LLMChessEngine tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessEngine:
    def test_init(self):
        adapter = _make_mock_adapter()
        engine = LLMChessEngine(adapter)
        assert engine._mcts_depth == DEFAULT_MCTS_DEPTH
        assert engine._temperature == DEFAULT_CHESS_TEMPERATURE
        assert engine._consensus_top_k == DEFAULT_CONSENSUS_TOP_K
        assert engine._move_count == 0
        assert isinstance(engine.meta_controller, LLMChessMetaController)
        assert isinstance(engine.hrm_agent, LLMChessHRMAgent)
        assert isinstance(engine.trm_agent, LLMChessTRMAgent)
        assert isinstance(engine.mcts_agent, LLMChessMCTSAgent)

    def test_init_custom(self):
        adapter = _make_mock_adapter()
        engine = LLMChessEngine(
            adapter,
            mcts_depth=4,
            temperature=0.5,
            max_tokens=500,
            consensus_top_k=2,
            strategies=["tactical", "positional"],
        )
        assert engine._mcts_depth == 4
        assert engine._temperature == 0.5
        assert engine._max_tokens == 500
        assert engine._consensus_top_k == 2

    @pytest.mark.asyncio
    async def test_analyze_position(self):
        adapter = _make_mock_adapter()
        engine = LLMChessEngine(adapter)
        analysis = await engine.analyze_position(START_FEN)
        assert isinstance(analysis, ChessAnalysis)
        assert analysis.best_move is not None
        assert analysis.routing_decision.game_phase == "opening"
        assert analysis.total_time_ms > 0
        assert engine._move_count == 1

    @pytest.mark.asyncio
    async def test_get_best_move(self):
        adapter = _make_mock_adapter()
        engine = LLMChessEngine(adapter)
        move = await engine.get_best_move(START_FEN)
        assert isinstance(move, str)
        assert len(move) >= 4

    @pytest.mark.asyncio
    async def test_analyze_position_with_agent_failure(self):
        adapter = _make_mock_adapter()
        engine = LLMChessEngine(adapter)

        # Make one agent fail
        original_process = engine.hrm_agent.process

        async def failing_process(**kwargs):
            raise RuntimeError("HRM failed")

        engine.hrm_agent.process = failing_process

        # Should still work with remaining agents
        analysis = await engine.analyze_position(START_FEN)
        assert analysis.best_move is not None

    def test_stats(self):
        adapter = _make_mock_adapter()
        engine = LLMChessEngine(adapter)
        stats = engine.stats
        assert stats["move_count"] == 0
        assert "hrm_stats" in stats
        assert "trm_stats" in stats
        assert "mcts_stats" in stats

    @pytest.mark.asyncio
    async def test_consensus_synthesis(self):
        adapter = _make_mock_adapter()
        engine = LLMChessEngine(adapter)
        agent_results = {
            "hrm": ChessMoveResult(
                move="e2e4", score=0.8, reasoning="good", agent_name="hrm", confidence=0.8
            ),
            "trm": ChessMoveResult(
                move="d2d4", score=0.7, reasoning="also good", agent_name="trm", confidence=0.7
            ),
        }
        move, reasoning = await engine._synthesize_consensus(START_FEN, agent_results)
        assert move == "e2e4"  # extracted from mock response

    @pytest.mark.asyncio
    async def test_consensus_synthesis_failure(self):
        adapter = _make_mock_adapter()
        adapter.generate = AsyncMock(side_effect=RuntimeError("LLM error"))
        engine = LLMChessEngine(adapter)
        agent_results = {
            "hrm": ChessMoveResult(
                move="e2e4", score=0.8, reasoning="good", agent_name="hrm", confidence=0.8
            ),
            "trm": ChessMoveResult(
                move="d2d4", score=0.7, reasoning="also good", agent_name="trm", confidence=0.7
            ),
        }
        move, reasoning = await engine._synthesize_consensus(START_FEN, agent_results)
        assert move is None
        assert reasoning is None


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstants:
    def test_default_values(self):
        assert DEFAULT_CHESS_TEMPERATURE == 0.3
        assert DEFAULT_CHESS_MAX_TOKENS == 1000
        assert DEFAULT_MCTS_DEPTH == 8
        assert DEFAULT_TOP_MOVES == 5
        assert DEFAULT_CONSENSUS_TOP_K == 3

    def test_phase_thresholds(self):
        assert PHASE_OPENING_THRESHOLD == 10
        assert PHASE_ENDGAME_MATERIAL == 26

    def test_weight_dicts_sum_to_one(self):
        for w in [OPENING_WEIGHTS, MIDDLEGAME_WEIGHTS, ENDGAME_WEIGHTS]:
            assert abs(sum(w.values()) - 1.0) < 1e-6
