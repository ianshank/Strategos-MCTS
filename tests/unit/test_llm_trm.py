"""Unit tests for src/framework/agents/llm_trm.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.adapters.llm.base import LLMResponse
from src.framework.agents.llm_trm import (
    DEFAULT_TRM_MAX_TOKENS,
    DEFAULT_TRM_TEMPERATURE,
    QUALITY_BASELINE,
    QUALITY_INITIAL_BONUS,
    QUALITY_LENGTH_BONUS,
    QUALITY_LENGTH_THRESHOLD,
    QUALITY_REFINED_BONUS,
    QUALITY_REVIEW_BONUS,
    LLMTRMAgent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(
        return_value=LLMResponse(
            text=(
                "### Initial Answer\n"
                "The initial approach uses gradient descent.\n\n"
                "### Critical Review\n"
                "- Weakness: Does not account for local minima.\n"
                "- Weakness: Learning rate sensitivity.\n\n"
                "### Refined Answer\n"
                "An improved approach uses Adam optimizer with learning rate scheduling."
            ),
            usage={"prompt_tokens": 150, "completion_tokens": 80, "total_tokens": 230},
        )
    )
    return client


@pytest.fixture
def trm_agent(mock_llm_client):
    return LLMTRMAgent(mock_llm_client)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMTRMAgentInit:
    def test_default_name(self, trm_agent):
        assert trm_agent.name == "LLM_TRM"

    def test_custom_name(self, mock_llm_client):
        agent = LLMTRMAgent(mock_llm_client, name="Custom_TRM")
        assert agent.name == "Custom_TRM"

    def test_default_temperature(self, trm_agent):
        assert trm_agent._temperature == DEFAULT_TRM_TEMPERATURE

    def test_custom_temperature(self, mock_llm_client):
        agent = LLMTRMAgent(mock_llm_client, temperature=0.2)
        assert agent._temperature == 0.2

    def test_default_max_tokens(self, trm_agent):
        assert trm_agent._max_tokens == DEFAULT_TRM_MAX_TOKENS

    def test_custom_max_tokens(self, mock_llm_client):
        agent = LLMTRMAgent(mock_llm_client, max_tokens=3000)
        assert agent._max_tokens == 3000


# ---------------------------------------------------------------------------
# Quality score computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeQualityScore:
    def test_baseline_only(self):
        score = LLMTRMAgent._compute_quality_score("plain text no structure")
        assert score == pytest.approx(QUALITY_BASELINE)

    def test_initial_bonus(self):
        score = LLMTRMAgent._compute_quality_score("The initial answer is X.")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_INITIAL_BONUS)

    def test_review_bonus_with_review_keyword(self):
        score = LLMTRMAgent._compute_quality_score("Critical review of the approach.")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_REVIEW_BONUS)

    def test_review_bonus_with_weakness_keyword(self):
        score = LLMTRMAgent._compute_quality_score("The main weakness is lack of data.")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_REVIEW_BONUS)

    def test_refined_bonus_with_refined_keyword(self):
        score = LLMTRMAgent._compute_quality_score("The refined answer is better.")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_REFINED_BONUS)

    def test_refined_bonus_with_improved_keyword(self):
        score = LLMTRMAgent._compute_quality_score("An improved version of the solution.")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_REFINED_BONUS)

    def test_length_bonus(self):
        long_text = "y" * (QUALITY_LENGTH_THRESHOLD + 1)
        score = LLMTRMAgent._compute_quality_score(long_text)
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_LENGTH_BONUS)

    def test_all_bonuses(self):
        text = (
            "Initial approach.\n"
            "Review shows weakness.\n"
            "Refined and improved solution.\n" + "z" * (QUALITY_LENGTH_THRESHOLD + 1)
        )
        expected = min(
            QUALITY_BASELINE
            + QUALITY_INITIAL_BONUS
            + QUALITY_REVIEW_BONUS
            + QUALITY_REFINED_BONUS
            + QUALITY_LENGTH_BONUS,
            1.0,
        )
        score = LLMTRMAgent._compute_quality_score(text)
        assert score == pytest.approx(expected)

    def test_capped_at_one(self):
        score = LLMTRMAgent._compute_quality_score("initial review weakness refined improved\n" + "z" * 1000)
        assert score <= 1.0

    def test_empty_string(self):
        score = LLMTRMAgent._compute_quality_score("")
        assert score == pytest.approx(QUALITY_BASELINE)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMTRMAgentProcess:
    @pytest.mark.asyncio
    async def test_process_returns_dict(self, trm_agent):
        result = await trm_agent.process(query="How to optimize neural networks?")
        assert isinstance(result, dict)
        assert "response" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_process_response_text(self, trm_agent):
        result = await trm_agent.process(query="test query")
        assert "Initial" in result["response"] or "Refined" in result["response"]

    @pytest.mark.asyncio
    async def test_metadata_has_initial_flag(self, trm_agent):
        result = await trm_agent.process(query="test")
        meta = result["metadata"]
        assert "has_initial_answer" in meta
        assert meta["has_initial_answer"] is True

    @pytest.mark.asyncio
    async def test_metadata_has_review_flag(self, trm_agent):
        result = await trm_agent.process(query="test")
        meta = result["metadata"]
        assert "has_critical_review" in meta
        assert meta["has_critical_review"] is True

    @pytest.mark.asyncio
    async def test_metadata_has_refined_flag(self, trm_agent):
        result = await trm_agent.process(query="test")
        meta = result["metadata"]
        assert "has_refined_answer" in meta
        assert meta["has_refined_answer"] is True

    @pytest.mark.asyncio
    async def test_metadata_strategy(self, trm_agent):
        result = await trm_agent.process(query="test")
        assert result["metadata"]["strategy"] == "iterative_refinement"

    @pytest.mark.asyncio
    async def test_metadata_quality_score(self, trm_agent):
        result = await trm_agent.process(query="test")
        score = result["metadata"]["final_quality_score"]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_token_usage_forwarded(self, trm_agent):
        result = await trm_agent.process(query="test")
        usage = result["metadata"]["token_usage"]
        assert usage["total_tokens"] == 230

    @pytest.mark.asyncio
    async def test_confidence_set(self, trm_agent):
        result = await trm_agent.process(query="test")
        assert result["metadata"]["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_agent_name_in_metadata(self, trm_agent):
        result = await trm_agent.process(query="test")
        assert result["metadata"]["agent_name"] == "LLM_TRM"

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, trm_agent):
        result = await trm_agent.process(query="test")
        assert result["metadata"]["processing_time_ms"] > 0.0


# ---------------------------------------------------------------------------
# RAG context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMTRMAgentRAG:
    @pytest.mark.asyncio
    async def test_rag_context_included(self, mock_llm_client):
        agent = LLMTRMAgent(mock_llm_client)
        await agent.process(
            query="How to train models?",
            rag_context="Use transfer learning for small datasets.",
        )
        assert mock_llm_client.generate.called

    @pytest.mark.asyncio
    async def test_no_rag_context(self, mock_llm_client):
        agent = LLMTRMAgent(mock_llm_client)
        await agent.process(query="How to train models?")
        assert mock_llm_client.generate.called


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMTRMAgentErrors:
    @pytest.mark.asyncio
    async def test_llm_error_handled(self, mock_llm_client):
        mock_llm_client.generate.side_effect = Exception("Service unavailable")
        agent = LLMTRMAgent(mock_llm_client)
        result = await agent.process(query="test")
        assert result["metadata"]["success"] is False
        assert result["metadata"]["error"] is not None

    @pytest.mark.asyncio
    async def test_empty_response(self, mock_llm_client):
        mock_llm_client.generate.return_value = LLMResponse(text="", usage={})
        agent = LLMTRMAgent(mock_llm_client)
        result = await agent.process(query="test")
        assert result["metadata"]["success"] is True
        assert result["metadata"]["final_quality_score"] == pytest.approx(QUALITY_BASELINE)

    @pytest.mark.asyncio
    async def test_no_structure_in_response(self, mock_llm_client):
        mock_llm_client.generate.return_value = LLMResponse(
            text="Just a plain answer without any structure.",
            usage={"total_tokens": 10},
        )
        agent = LLMTRMAgent(mock_llm_client)
        result = await agent.process(query="test")
        meta = result["metadata"]
        assert meta["has_initial_answer"] is False
        assert meta["has_critical_review"] is False
        assert meta["has_refined_answer"] is False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMTRMAgentStats:
    @pytest.mark.asyncio
    async def test_stats_after_processing(self, trm_agent):
        await trm_agent.process(query="query1")
        await trm_agent.process(query="query2")
        stats = trm_agent.stats
        assert stats["request_count"] == 2
        assert stats["total_processing_time_ms"] > 0
        assert stats["error_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_error(self, mock_llm_client):
        mock_llm_client.generate.side_effect = RuntimeError("boom")
        agent = LLMTRMAgent(mock_llm_client)
        await agent.process(query="test")
        stats = agent.stats
        assert stats["error_count"] == 1

    @pytest.mark.asyncio
    async def test_average_processing_time(self, trm_agent):
        await trm_agent.process(query="q1")
        await trm_agent.process(query="q2")
        stats = trm_agent.stats
        assert stats["average_processing_time_ms"] > 0
        assert stats["average_processing_time_ms"] == pytest.approx(
            stats["total_processing_time_ms"] / stats["request_count"]
        )
