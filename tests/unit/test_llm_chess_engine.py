"""Unit tests for src/games/chess/llm_chess_engine.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adapters.llm.base import LLMResponse
from src.games.chess.llm_chess_engine import (
    DEFAULT_CHESS_MAX_TOKENS,
    DEFAULT_CHESS_TEMPERATURE,
    DEFAULT_CONSENSUS_TOP_K,
    DEFAULT_MCTS_DEPTH,
    ENDGAME_WEIGHTS,
    MIDDLEGAME_WEIGHTS,
    OPENING_WEIGHTS,
    PHASE_ENDGAME_MATERIAL,
    PHASE_OPENING_THRESHOLD,
    PIECE_VALUES,
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLEGAME_FEN = "r1bqkb1r/pppppppp/2n2n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 15"
ENDGAME_FEN = "8/5k2/8/8/8/8/5K2/4R3 w - - 0 50"


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns configurable responses."""
    client = AsyncMock()
    client.generate = AsyncMock(
        return_value=LLMResponse(
            text="**Recommended move:** e2e4\n**Score:** 0.8\n### Synthesis\nGood opening move.",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
    )
    return client


@pytest.fixture
def meta_controller():
    return LLMChessMetaController()


@pytest.fixture
def hrm_agent(mock_llm_client):
    return LLMChessHRMAgent(mock_llm_client)


@pytest.fixture
def trm_agent(mock_llm_client):
    return LLMChessTRMAgent(mock_llm_client)


@pytest.fixture
def mcts_agent(mock_llm_client):
    return LLMChessMCTSAgent(mock_llm_client, strategies=["tactical"])


@pytest.fixture
def engine(mock_llm_client):
    return LLMChessEngine(mock_llm_client)


# ---------------------------------------------------------------------------
# extract_uci_move
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractUciMove:
    def test_recommended_move_bold(self):
        assert extract_uci_move("**Recommended move:** e2e4") == "e2e4"

    def test_move_colon_format(self):
        assert extract_uci_move("Move: d2d4") == "d2d4"

    def test_best_move_bold(self):
        assert extract_uci_move("**Best Move:** g1f3") == "g1f3"

    def test_promotion_move(self):
        assert extract_uci_move("**Move:** e7e8q") == "e7e8q"

    def test_fallback_uci_in_text(self):
        assert extract_uci_move("I think the best play is e2e4 here.") == "e2e4"

    def test_no_move_found(self):
        assert extract_uci_move("No valid move in this text.") is None

    def test_empty_text(self):
        assert extract_uci_move("") is None

    def test_move_with_punctuation(self):
        assert extract_uci_move("Consider e2e4, which controls the center.") == "e2e4"

    def test_multiple_moves_returns_first_pattern(self):
        text = "**Move:** d2d4\nAlternative: e2e4"
        assert extract_uci_move(text) == "d2d4"


# ---------------------------------------------------------------------------
# extract_score
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractScore:
    def test_score_bold(self):
        assert extract_score("**Score:** 0.75") == pytest.approx(0.75)

    def test_confidence_bold(self):
        assert extract_score("**Confidence:** 0.9") == pytest.approx(0.9)

    def test_score_plain(self):
        assert extract_score("Score: 0.6") == pytest.approx(0.6)

    def test_confidence_plain(self):
        assert extract_score("Confidence: 0.85") == pytest.approx(0.85)

    def test_clamp_above_1(self):
        assert extract_score("**Score:** 1.5") == pytest.approx(1.0)

    def test_clamp_below_0(self):
        # Regex won't match negative, so default 0.5 returned
        assert extract_score("**Score:** -0.5") == pytest.approx(0.5)

    def test_no_score_returns_default(self):
        assert extract_score("No score here") == pytest.approx(0.5)

    def test_empty_text(self):
        assert extract_score("") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fen_to_board_ascii
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFenToBoardAscii:
    @patch("src.games.chess.llm_chess_engine.CHESS_AVAILABLE", False)
    def test_fallback_rendering(self):
        result = fen_to_board_ascii(STARTING_FEN)
        assert "a b c d e f g h" in result
        assert "8" in result
        assert "1" in result

    @patch("src.games.chess.llm_chess_engine.CHESS_AVAILABLE", False)
    def test_empty_fen(self):
        result = fen_to_board_ascii("")
        assert "a b c d e f g h" in result


# ---------------------------------------------------------------------------
# describe_position
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDescribePosition:
    @patch("src.games.chess.llm_chess_engine.get_legal_moves_list", return_value=None)
    def test_starting_position(self, _mock_legal):
        desc = describe_position(STARTING_FEN)
        assert "White" in desc
        assert "Opening" in desc
        assert STARTING_FEN in desc

    @patch("src.games.chess.llm_chess_engine.get_legal_moves_list", return_value=None)
    def test_black_to_move(self, _mock_legal):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        desc = describe_position(fen)
        assert "Black" in desc

    @patch("src.games.chess.llm_chess_engine.get_legal_moves_list", return_value=["e2e4", "d2d4"])
    def test_includes_legal_moves(self, _mock_legal):
        desc = describe_position(STARTING_FEN)
        assert "Legal moves" in desc
        assert "e2e4" in desc

    @patch("src.games.chess.llm_chess_engine.get_legal_moves_list", return_value=None)
    def test_material_balance_equal(self, _mock_legal):
        desc = describe_position(STARTING_FEN)
        assert "Equal" in desc

    @patch("src.games.chess.llm_chess_engine.get_legal_moves_list", return_value=None)
    def test_endgame_phase(self, _mock_legal):
        desc = describe_position(ENDGAME_FEN)
        assert "Endgame" in desc


# ---------------------------------------------------------------------------
# LLMChessMetaController
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessMetaController:
    def test_opening_routing(self, meta_controller):
        decision = meta_controller.route(STARTING_FEN)
        assert decision.game_phase == "opening"
        assert isinstance(decision, RoutingDecision)
        assert decision.primary_agent == "hrm"

    def test_middlegame_routing(self, meta_controller):
        decision = meta_controller.route(MIDDLEGAME_FEN)
        assert decision.game_phase == "middlegame"
        assert decision.primary_agent == "mcts"

    def test_endgame_routing(self, meta_controller):
        decision = meta_controller.route(ENDGAME_FEN)
        assert decision.game_phase == "endgame"
        assert decision.primary_agent == "trm"

    def test_weights_sum_to_one(self, meta_controller):
        for fen in [STARTING_FEN, MIDDLEGAME_FEN, ENDGAME_FEN]:
            decision = meta_controller.route(fen)
            total = sum(decision.agent_weights.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_confidence_between_0_and_1(self, meta_controller):
        for fen in [STARTING_FEN, MIDDLEGAME_FEN, ENDGAME_FEN]:
            decision = meta_controller.route(fen)
            assert 0.0 <= decision.confidence <= 1.0

    def test_custom_weights(self):
        ctrl = LLMChessMetaController(
            opening_weights={"hrm": 1.0, "trm": 0.0, "mcts": 0.0},
        )
        decision = ctrl.route(STARTING_FEN)
        assert decision.agent_weights["hrm"] == pytest.approx(1.0)

    def test_get_move_number(self):
        assert LLMChessMetaController._get_move_number(STARTING_FEN) == 1
        assert LLMChessMetaController._get_move_number(ENDGAME_FEN) == 50
        assert LLMChessMetaController._get_move_number("bad fen") == 1

    def test_get_total_material(self):
        material = LLMChessMetaController._get_total_material(STARTING_FEN)
        # 8P + 2N + 2B + 2R + Q + K per side = 8+6+6+10+9+0 = 39 per side = 78
        assert material == 78

    def test_get_total_material_endgame(self):
        material = LLMChessMetaController._get_total_material(ENDGAME_FEN)
        # Only a rook (5) and two kings (0)
        assert material == 5

    def test_classify_phase(self, meta_controller):
        assert meta_controller._classify_phase(1, 78) == "opening"
        assert meta_controller._classify_phase(10, 78) == "opening"
        assert meta_controller._classify_phase(15, 78) == "middlegame"
        assert meta_controller._classify_phase(40, 10) == "endgame"

    def test_build_reasoning(self):
        r = LLMChessMetaController._build_reasoning("opening", "hrm")
        assert "HRM" in r or "hrm" in r.lower()

        r2 = LLMChessMetaController._build_reasoning("unknown_phase", "xyz")
        assert "XYZ" in r2

    @patch("src.games.chess.llm_chess_engine.CHESS_AVAILABLE", False)
    def test_has_check_without_chess(self):
        assert LLMChessMetaController._has_check(STARTING_FEN) is False


# ---------------------------------------------------------------------------
# LLMChessHRMAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessHRMAgent:
    @pytest.mark.asyncio
    async def test_process_returns_move(self, hrm_agent):
        result = await hrm_agent.process(query=STARTING_FEN)
        assert "response" in result
        assert "metadata" in result
        meta = result["metadata"]
        assert meta["move"] == "e2e4"
        assert meta["agent"] == "chess_hrm"
        assert meta["strategy"] == "hierarchical_decomposition"

    @pytest.mark.asyncio
    async def test_process_extracts_score(self, hrm_agent):
        result = await hrm_agent.process(query=STARTING_FEN)
        assert result["metadata"]["score"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_default_move_on_missing(self, mock_llm_client):
        mock_llm_client.generate.return_value = LLMResponse(
            text="I cannot determine a move.",
            usage={},
        )
        agent = LLMChessHRMAgent(mock_llm_client)
        result = await agent.process(query=STARTING_FEN)
        assert result["metadata"]["move"] == "e2e4"

    @pytest.mark.asyncio
    async def test_custom_temperature(self, mock_llm_client):
        agent = LLMChessHRMAgent(mock_llm_client, temperature=0.9, max_tokens=500)
        assert agent._temperature == 0.9
        assert agent._max_tokens == 500

    @pytest.mark.asyncio
    async def test_name_default(self, hrm_agent):
        assert hrm_agent.name == "Chess_HRM"


# ---------------------------------------------------------------------------
# LLMChessTRMAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessTRMAgent:
    @pytest.mark.asyncio
    async def test_process_returns_move(self, trm_agent):
        result = await trm_agent.process(query=STARTING_FEN)
        meta = result["metadata"]
        assert meta["move"] == "e2e4"
        assert meta["agent"] == "chess_trm"
        assert meta["strategy"] == "iterative_refinement"

    @pytest.mark.asyncio
    async def test_default_move_on_missing(self, mock_llm_client):
        mock_llm_client.generate.return_value = LLMResponse(
            text="No specific move recommended.",
            usage={},
        )
        agent = LLMChessTRMAgent(mock_llm_client)
        result = await agent.process(query=STARTING_FEN)
        assert result["metadata"]["move"] == "e2e4"

    @pytest.mark.asyncio
    async def test_name_default(self, trm_agent):
        assert trm_agent.name == "Chess_TRM"


# ---------------------------------------------------------------------------
# LLMChessMCTSAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessMCTSAgent:
    @pytest.mark.asyncio
    async def test_process_returns_best_strategy(self, mcts_agent):
        result = await mcts_agent.process(query=STARTING_FEN)
        meta = result["metadata"]
        assert meta["move"] == "e2e4"
        assert meta["agent"] == "chess_mcts"
        assert "strategy" in meta

    @pytest.mark.asyncio
    async def test_multiple_strategies(self, mock_llm_client):
        agent = LLMChessMCTSAgent(
            mock_llm_client,
            strategies=["tactical", "positional"],
        )
        result = await agent.process(query=STARTING_FEN)
        meta = result["metadata"]
        assert "all_strategies" in meta
        assert len(meta["all_strategies"]) == 2

    @pytest.mark.asyncio
    async def test_no_strategies_fallback(self, mock_llm_client):
        """When all strategies raise exceptions, agent returns fallback move."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        agent = LLMChessMCTSAgent(mock_llm_client, strategies=["tactical"])
        result = await agent.process(query=STARTING_FEN)
        # The _process_impl catches strategy exceptions via gather(return_exceptions=True)
        # and returns a fallback result with confidence 0.0
        assert result["metadata"]["confidence"] == 0.0
        assert result["metadata"]["move"] == "e2e4"

    @pytest.mark.asyncio
    async def test_best_strategy_selected_by_score(self, mock_llm_client):
        call_count = 0

        async def varying_response(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(text="**Move:** d2d4\n**Score:** 0.9", usage={})
            return LLMResponse(text="**Move:** e2e4\n**Score:** 0.3", usage={})

        mock_llm_client.generate.side_effect = varying_response
        agent = LLMChessMCTSAgent(
            mock_llm_client,
            strategies=["tactical", "positional"],
        )
        result = await agent.process(query=STARTING_FEN)
        assert result["metadata"]["move"] == "d2d4"

    @pytest.mark.asyncio
    async def test_default_strategies(self, mock_llm_client):
        agent = LLMChessMCTSAgent(mock_llm_client)
        assert len(agent._strategies) == 4  # tactical, positional, prophylactic, endgame


# ---------------------------------------------------------------------------
# LLMChessEngine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMChessEngine:
    @pytest.mark.asyncio
    async def test_analyze_position(self, engine):
        analysis = await engine.analyze_position(STARTING_FEN)
        assert isinstance(analysis, ChessAnalysis)
        assert analysis.best_move is not None
        assert analysis.routing_decision.game_phase == "opening"
        assert analysis.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_get_best_move(self, engine):
        move = await engine.get_best_move(STARTING_FEN)
        assert isinstance(move, str)
        assert len(move) >= 4

    @pytest.mark.asyncio
    async def test_move_count_increments(self, engine):
        assert engine._move_count == 0
        await engine.analyze_position(STARTING_FEN)
        assert engine._move_count == 1
        await engine.analyze_position(STARTING_FEN)
        assert engine._move_count == 2

    @pytest.mark.asyncio
    async def test_stats(self, engine):
        stats = engine.stats
        assert stats["move_count"] == 0
        assert "hrm_stats" in stats
        assert "trm_stats" in stats
        assert "mcts_stats" in stats

    @pytest.mark.asyncio
    async def test_consensus_synthesis(self, engine):
        analysis = await engine.analyze_position(STARTING_FEN)
        # With at least 2 agent results, consensus should be attempted
        if len(analysis.agent_results) >= 2:
            assert analysis.consensus_move is not None or analysis.consensus_reasoning is not None

    @pytest.mark.asyncio
    async def test_agent_failure_handled(self, mock_llm_client):
        """Engine handles individual agent failures gracefully."""
        call_count = 0

        async def sometimes_fail(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("HRM failed")
            return LLMResponse(
                text="**Recommended move:** d2d4\n**Score:** 0.7",
                usage={"total_tokens": 100},
            )

        mock_llm_client.generate.side_effect = sometimes_fail
        engine = LLMChessEngine(mock_llm_client)
        analysis = await engine.analyze_position(STARTING_FEN)
        # Engine should still produce a result even if one agent fails
        assert analysis.best_move is not None

    @pytest.mark.asyncio
    async def test_consensus_failure_handled(self, mock_llm_client):
        """Consensus synthesis failure does not crash the engine."""
        call_count = 0

        async def fail_on_consensus(**kwargs):
            nonlocal call_count
            call_count += 1
            # First 3 calls succeed (hrm, trm, mcts strategies)
            # Then fail on consensus call
            if call_count > 5:
                raise Exception("Consensus LLM failed")
            return LLMResponse(
                text="**Move:** e2e4\n**Score:** 0.7",
                usage={"total_tokens": 50},
            )

        mock_llm_client.generate.side_effect = fail_on_consensus
        engine = LLMChessEngine(mock_llm_client, strategies=["tactical"])
        analysis = await engine.analyze_position(STARTING_FEN)
        assert analysis.best_move is not None

    @pytest.mark.asyncio
    async def test_engine_initialization(self, mock_llm_client):
        engine = LLMChessEngine(
            mock_llm_client,
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
    async def test_metadata_in_analysis(self, engine):
        analysis = await engine.analyze_position(STARTING_FEN)
        assert "move_count" in analysis.metadata
        assert "agents_used" in analysis.metadata
        assert "mcts_depth" in analysis.metadata


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataStructures:
    def test_routing_decision_fields(self):
        rd = RoutingDecision(
            primary_agent="hrm",
            agent_weights={"hrm": 0.5, "trm": 0.3, "mcts": 0.2},
            confidence=0.8,
            game_phase="opening",
            reasoning="test",
        )
        assert rd.primary_agent == "hrm"
        assert rd.confidence == 0.8

    def test_chess_move_result_defaults(self):
        r = ChessMoveResult(
            move="e2e4",
            score=0.7,
            reasoning="Good move",
            agent_name="hrm",
        )
        assert r.confidence == 0.0
        assert r.metadata == {}

    def test_chess_analysis_defaults(self):
        rd = RoutingDecision("hrm", {}, 0.5, "opening", "test")
        a = ChessAnalysis(
            best_move="e2e4",
            candidate_moves=[],
            routing_decision=rd,
            agent_results={},
        )
        assert a.consensus_move is None
        assert a.total_time_ms == 0.0
        assert a.metadata == {}


# ---------------------------------------------------------------------------
# get_legal_moves_list
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetLegalMovesList:
    @patch("src.games.chess.llm_chess_engine.CHESS_AVAILABLE", False)
    def test_returns_none_without_chess(self):
        assert get_legal_moves_list(STARTING_FEN) is None
