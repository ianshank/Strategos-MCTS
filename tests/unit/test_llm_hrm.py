"""Unit tests for src/framework/agents/llm_hrm.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.adapters.llm.base import LLMResponse
from src.framework.agents.llm_hrm import (
    DEFAULT_HRM_MAX_TOKENS,
    DEFAULT_HRM_TEMPERATURE,
    QUALITY_BASELINE,
    QUALITY_LENGTH_BONUS,
    QUALITY_LENGTH_THRESHOLD,
    QUALITY_SUBPROBLEM_BONUS,
    QUALITY_SYNTHESIS_BONUS,
    LLMHRMAgent,
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
                "### Sub-problems\n"
                "1. Analyze the market trends\n"
                "   **Answer:** Markets are volatile.\n"
                "2. Evaluate the competition\n"
                "   **Answer:** Competition is strong.\n\n"
                "### Synthesis\n"
                "The final integrated answer combining all sub-problems "
                "indicates a cautious but optimistic approach."
            ),
            usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        )
    )
    return client


@pytest.fixture
def hrm_agent(mock_llm_client):
    return LLMHRMAgent(mock_llm_client)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMHRMAgentInit:
    def test_default_name(self, hrm_agent):
        assert hrm_agent.name == "LLM_HRM"

    def test_custom_name(self, mock_llm_client):
        agent = LLMHRMAgent(mock_llm_client, name="Custom_HRM")
        assert agent.name == "Custom_HRM"

    def test_default_temperature(self, hrm_agent):
        assert hrm_agent._temperature == DEFAULT_HRM_TEMPERATURE

    def test_custom_temperature(self, mock_llm_client):
        agent = LLMHRMAgent(mock_llm_client, temperature=0.9)
        assert agent._temperature == 0.9

    def test_default_max_tokens(self, hrm_agent):
        assert hrm_agent._max_tokens == DEFAULT_HRM_MAX_TOKENS

    def test_custom_max_tokens(self, mock_llm_client):
        agent = LLMHRMAgent(mock_llm_client, max_tokens=2000)
        assert agent._max_tokens == 2000


# ---------------------------------------------------------------------------
# Quality score computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeQualityScore:
    def test_baseline_only(self):
        score = LLMHRMAgent._compute_quality_score("plain text without structure")
        assert score == pytest.approx(QUALITY_BASELINE)

    def test_subproblem_bonus_with_keyword(self):
        score = LLMHRMAgent._compute_quality_score("Sub-problem 1: something")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_SUBPROBLEM_BONUS)

    def test_subproblem_bonus_with_heading(self):
        score = LLMHRMAgent._compute_quality_score("### Heading\nSome content")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_SUBPROBLEM_BONUS)

    def test_synthesis_bonus(self):
        score = LLMHRMAgent._compute_quality_score("The synthesis of ideas leads to...")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_SYNTHESIS_BONUS)

    def test_synthesis_bonus_with_final_keyword(self):
        score = LLMHRMAgent._compute_quality_score("The final answer is clear.")
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_SYNTHESIS_BONUS)

    def test_length_bonus(self):
        long_text = "x" * (QUALITY_LENGTH_THRESHOLD + 1)
        score = LLMHRMAgent._compute_quality_score(long_text)
        assert score == pytest.approx(QUALITY_BASELINE + QUALITY_LENGTH_BONUS)

    def test_all_bonuses(self):
        text = "### Sub-problem analysis\n" "The synthesis shows that\n" + "x" * (QUALITY_LENGTH_THRESHOLD + 1)
        expected = QUALITY_BASELINE + QUALITY_SUBPROBLEM_BONUS + QUALITY_SYNTHESIS_BONUS + QUALITY_LENGTH_BONUS
        score = LLMHRMAgent._compute_quality_score(text)
        assert score == pytest.approx(min(expected, 1.0))

    def test_capped_at_one(self):
        # Even with all bonuses the score should not exceed 1.0
        score = LLMHRMAgent._compute_quality_score("### sub-problem\nsynthesis final\n" + "x" * 1000)
        assert score <= 1.0

    def test_empty_string(self):
        score = LLMHRMAgent._compute_quality_score("")
        assert score == pytest.approx(QUALITY_BASELINE)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMHRMAgentProcess:
    @pytest.mark.asyncio
    async def test_process_returns_dict(self, hrm_agent):
        result = await hrm_agent.process(query="What is machine learning?")
        assert isinstance(result, dict)
        assert "response" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_process_response_text(self, hrm_agent):
        result = await hrm_agent.process(query="What is ML?")
        assert "Sub-problems" in result["response"] or "Synthesis" in result["response"]

    @pytest.mark.asyncio
    async def test_metadata_has_subproblems_flag(self, hrm_agent):
        result = await hrm_agent.process(query="Explain quantum computing")
        meta = result["metadata"]
        assert "has_subproblems" in meta
        assert meta["has_subproblems"] is True

    @pytest.mark.asyncio
    async def test_metadata_has_synthesis_flag(self, hrm_agent):
        result = await hrm_agent.process(query="Explain quantum computing")
        meta = result["metadata"]
        assert "has_synthesis" in meta
        assert meta["has_synthesis"] is True

    @pytest.mark.asyncio
    async def test_metadata_strategy(self, hrm_agent):
        result = await hrm_agent.process(query="test")
        assert result["metadata"]["strategy"] == "hierarchical_decomposition"

    @pytest.mark.asyncio
    async def test_metadata_quality_score(self, hrm_agent):
        result = await hrm_agent.process(query="test")
        score = result["metadata"]["decomposition_quality_score"]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_token_usage_forwarded(self, hrm_agent):
        result = await hrm_agent.process(query="test")
        usage = result["metadata"]["token_usage"]
        assert usage["total_tokens"] == 300

    @pytest.mark.asyncio
    async def test_confidence_set(self, hrm_agent):
        result = await hrm_agent.process(query="test")
        assert result["metadata"]["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_agent_name_in_metadata(self, hrm_agent):
        result = await hrm_agent.process(query="test")
        assert result["metadata"]["agent_name"] == "LLM_HRM"

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, hrm_agent):
        result = await hrm_agent.process(query="test")
        assert result["metadata"]["processing_time_ms"] > 0.0


# ---------------------------------------------------------------------------
# RAG context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMHRMAgentRAG:
    @pytest.mark.asyncio
    async def test_rag_context_included_in_prompt(self, mock_llm_client):
        agent = LLMHRMAgent(mock_llm_client)
        await agent.process(query="What is AI?", rag_context="AI stands for Artificial Intelligence.")

        # Verify the LLM was called with the rag context
        assert mock_llm_client.generate.called

    @pytest.mark.asyncio
    async def test_no_rag_context(self, mock_llm_client):
        agent = LLMHRMAgent(mock_llm_client)
        await agent.process(query="What is AI?")
        assert mock_llm_client.generate.called


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMHRMAgentErrors:
    @pytest.mark.asyncio
    async def test_llm_error_handled(self, mock_llm_client):
        mock_llm_client.generate.side_effect = Exception("LLM service down")
        agent = LLMHRMAgent(mock_llm_client)
        result = await agent.process(query="test")
        assert result["metadata"]["success"] is False
        assert result["metadata"]["error"] is not None

    @pytest.mark.asyncio
    async def test_empty_response(self, mock_llm_client):
        mock_llm_client.generate.return_value = LLMResponse(text="", usage={})
        agent = LLMHRMAgent(mock_llm_client)
        result = await agent.process(query="test")
        assert result["metadata"]["success"] is True
        # Empty text gets baseline quality score
        assert result["metadata"]["decomposition_quality_score"] == pytest.approx(QUALITY_BASELINE)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMHRMAgentStats:
    @pytest.mark.asyncio
    async def test_stats_after_processing(self, hrm_agent):
        await hrm_agent.process(query="test1")
        await hrm_agent.process(query="test2")
        stats = hrm_agent.stats
        assert stats["request_count"] == 2
        assert stats["total_processing_time_ms"] > 0
        assert stats["error_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_error(self, mock_llm_client):
        mock_llm_client.generate.side_effect = Exception("fail")
        agent = LLMHRMAgent(mock_llm_client)
        await agent.process(query="test")
        stats = agent.stats
        assert stats["error_count"] == 1
