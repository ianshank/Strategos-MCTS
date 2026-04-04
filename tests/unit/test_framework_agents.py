"""
Unit tests for src/framework/agents/base.py, llm_hrm.py, and llm_trm.py.

Covers AgentContext, AgentResult, NoOpMetricsCollector, AsyncAgentBase lifecycle,
LLMHRMAgent, LLMTRMAgent, ParallelAgent, and SequentialAgent.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.adapters.llm.base import LLMResponse
from src.framework.agents.base import (
    AgentContext,
    AgentResult,
    AsyncAgentBase,
    NoOpMetricsCollector,
    ParallelAgent,
    SequentialAgent,
)
from src.framework.agents.llm_hrm import LLMHRMAgent
from src.framework.agents.llm_trm import LLMTRMAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_adapter(text: str = "mock response", usage: dict | None = None) -> AsyncMock:
    """Create a mock LLMClient whose generate() returns an LLMResponse."""
    adapter = AsyncMock()
    adapter.generate.return_value = LLMResponse(
        text=text,
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    return adapter


class _ConcreteAgent(AsyncAgentBase):
    """Minimal concrete subclass for testing the abstract base."""

    def __init__(self, response_text: str = "concrete response", **kwargs):
        super().__init__(**kwargs)
        self._response_text = response_text

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            response=self._response_text,
            confidence=0.8,
            metadata={"test": True},
        )


class _FailingAgent(AsyncAgentBase):
    """Agent that always raises an exception in _process_impl."""

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        raise RuntimeError("intentional failure")


# ===========================================================================
# AgentContext
# ===========================================================================


@pytest.mark.unit
class TestAgentContext:
    def test_creation_defaults(self):
        ctx = AgentContext(query="hello")
        assert ctx.query == "hello"
        assert ctx.rag_context is None
        assert ctx.max_iterations == 5
        assert ctx.temperature == 0.7
        assert isinstance(ctx.session_id, str)
        # session_id should look like a UUID
        uuid.UUID(ctx.session_id)

    def test_creation_custom(self):
        ctx = AgentContext(
            query="q",
            session_id="sid-1",
            rag_context="some context",
            metadata={"k": "v"},
            max_iterations=10,
            temperature=0.3,
        )
        assert ctx.session_id == "sid-1"
        assert ctx.rag_context == "some context"
        assert ctx.metadata == {"k": "v"}
        assert ctx.max_iterations == 10
        assert ctx.temperature == 0.3

    def test_to_dict(self):
        ctx = AgentContext(query="q", session_id="s1")
        d = ctx.to_dict()
        assert d["query"] == "q"
        assert d["session_id"] == "s1"
        assert "rag_context" in d
        assert "additional_context" in d


# ===========================================================================
# AgentResult
# ===========================================================================


@pytest.mark.unit
class TestAgentResult:
    def test_defaults(self):
        r = AgentResult(response="ok")
        assert r.response == "ok"
        assert r.confidence == 0.0
        assert r.success is True
        assert r.error is None
        assert r.agent_name == ""

    def test_to_dict(self):
        r = AgentResult(response="ok", confidence=0.9, agent_name="A")
        d = r.to_dict()
        assert d["response"] == "ok"
        assert d["confidence"] == 0.9
        assert d["agent_name"] == "A"
        assert "created_at" in d
        assert isinstance(d["created_at"], str)  # isoformat

    def test_error_result(self):
        r = AgentResult(response="", success=False, error="boom")
        assert r.success is False
        assert r.error == "boom"


# ===========================================================================
# NoOpMetricsCollector
# ===========================================================================


@pytest.mark.unit
class TestNoOpMetricsCollector:
    def test_all_methods_are_noop(self):
        m = NoOpMetricsCollector()
        # Should not raise
        m.record_latency("agent", 100.0)
        m.record_tokens("agent", 50)
        m.record_error("agent", "ValueError")
        m.record_success("agent")


# ===========================================================================
# AsyncAgentBase lifecycle & process
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestAsyncAgentBase:
    async def test_initialize_and_shutdown(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        assert agent._initialized is False
        await agent.initialize()
        assert agent._initialized is True
        await agent.shutdown()
        assert agent._initialized is False

    async def test_context_manager(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        async with agent as a:
            assert a._initialized is True
            assert a is agent
        assert agent._initialized is False

    async def test_process_with_query_string(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        result = await agent.process(query="test query")
        assert result["response"] == "concrete response"
        assert result["metadata"]["success"] is True
        assert result["metadata"]["confidence"] == 0.8
        assert result["metadata"]["agent_name"] == "_ConcreteAgent"
        assert result["metadata"]["processing_time_ms"] > 0

    async def test_process_with_context(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        ctx = AgentContext(query="ctx query", rag_context="some rag")
        result = await agent.process(context=ctx)
        assert result["response"] == "concrete response"

    async def test_process_no_query_no_context_raises(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        with pytest.raises(ValueError, match="Either 'query' or 'context' must be provided"):
            await agent.process()

    async def test_process_auto_initializes(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        assert agent._initialized is False
        await agent.process(query="q")
        assert agent._initialized is True

    async def test_process_updates_stats(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        await agent.process(query="q1")
        await agent.process(query="q2")
        stats = agent.stats
        assert stats["request_count"] == 2
        assert stats["total_processing_time_ms"] > 0
        assert stats["error_count"] == 0
        assert stats["average_processing_time_ms"] > 0

    async def test_process_error_handling(self):
        agent = _FailingAgent(model_adapter=_make_mock_adapter())
        result = await agent.process(query="will fail")
        assert result["metadata"]["success"] is False
        assert "intentional failure" in result["metadata"]["error"]
        assert result["response"] == ""
        assert agent._error_count == 1

    async def test_custom_name(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter(), name="MyAgent")
        assert agent.name == "MyAgent"
        result = await agent.process(query="q")
        assert result["metadata"]["agent_name"] == "MyAgent"

    async def test_metrics_collector_called(self):
        metrics = MagicMock()
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter(), metrics_collector=metrics)
        await agent.process(query="q")
        metrics.record_latency.assert_called_once()
        metrics.record_success.assert_called_once()

    async def test_metrics_collector_records_tokens(self):
        metrics = MagicMock()
        # _ConcreteAgent doesn't set token_usage, so record_tokens won't be called.
        # Use LLMHRMAgent which does set token_usage.
        adapter = _make_mock_adapter(text="### Sub-problems\n1. sub-problem\n### Synthesis\nfinal answer")
        agent = LLMHRMAgent(model_adapter=adapter, metrics_collector=metrics)
        await agent.process(query="q")
        metrics.record_tokens.assert_called_once()

    async def test_generate_llm_response(self):
        adapter = _make_mock_adapter(text="llm text")
        agent = _ConcreteAgent(model_adapter=adapter)
        resp = await agent.generate_llm_response(prompt="hello")
        assert resp.text == "llm text"
        adapter.generate.assert_awaited_once()


# ===========================================================================
# LLMHRMAgent
# ===========================================================================


@pytest.mark.unit
class TestLLMHRMAgentQuality:
    def test_baseline_only(self):
        # No keywords, short text
        score = LLMHRMAgent._compute_quality_score("short text")
        assert score == pytest.approx(0.5)

    def test_subproblem_bonus(self):
        score = LLMHRMAgent._compute_quality_score("### Sub-problem analysis")
        assert score == pytest.approx(0.7)

    def test_synthesis_bonus(self):
        score = LLMHRMAgent._compute_quality_score("The final conclusion is ...")
        assert score == pytest.approx(0.65)

    def test_length_bonus(self):
        long_text = "a" * 301
        score = LLMHRMAgent._compute_quality_score(long_text)
        assert score == pytest.approx(0.6)

    def test_all_bonuses(self):
        text = "### Sub-problem one\nAnswer\n### Synthesis\nFinal answer\n" + "x" * 300
        score = LLMHRMAgent._compute_quality_score(text)
        assert score == pytest.approx(0.95)

    def test_capped_at_one(self):
        # Even if somehow all bonuses stack, should not exceed 1.0
        text = "sub-problem synthesis final ###\n" + "x" * 500
        score = LLMHRMAgent._compute_quality_score(text)
        assert score <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMHRMAgentProcess:
    async def test_basic_process(self):
        adapter = _make_mock_adapter(
            text="### Sub-problems\n1. sub-problem A\n### Synthesis\nFinal answer here",
            usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
        )
        agent = LLMHRMAgent(model_adapter=adapter)
        result = await agent.process(query="Explain quantum computing")

        assert result["response"].startswith("### Sub-problems")
        assert result["metadata"]["success"] is True
        meta = result["metadata"]
        assert "decomposition_quality_score" in meta
        assert meta["has_subproblems"] is True
        assert meta["has_synthesis"] is True
        assert meta["token_usage"]["total_tokens"] == 150

    async def test_with_rag_context(self):
        adapter = _make_mock_adapter(text="response with rag")
        agent = LLMHRMAgent(model_adapter=adapter)
        ctx = AgentContext(query="q", rag_context="background info")
        await agent.process(context=ctx)

        # Verify the prompt included RAG context
        call_kwargs = adapter.generate.call_args
        prompt_sent = call_kwargs.kwargs.get("prompt", "")
        assert "Additional Context" in prompt_sent
        assert "background info" in prompt_sent

    async def test_without_rag_context(self):
        adapter = _make_mock_adapter(text="response no rag")
        agent = LLMHRMAgent(model_adapter=adapter)
        await agent.process(query="q")

        call_kwargs = adapter.generate.call_args
        prompt_sent = call_kwargs.kwargs.get("prompt", "")
        assert "Additional Context" not in prompt_sent

    async def test_custom_temperature_and_tokens(self):
        adapter = _make_mock_adapter(text="resp")
        agent = LLMHRMAgent(model_adapter=adapter, temperature=0.9, max_tokens=500)
        await agent.process(query="q")

        call_kwargs = adapter.generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0.9
        assert call_kwargs.kwargs["max_tokens"] == 500

    async def test_default_name(self):
        agent = LLMHRMAgent(model_adapter=_make_mock_adapter())
        assert agent.name == "LLM_HRM"

    async def test_llm_error_propagates(self):
        adapter = AsyncMock()
        adapter.generate.side_effect = RuntimeError("LLM unavailable")
        agent = LLMHRMAgent(model_adapter=adapter)
        result = await agent.process(query="q")
        assert result["metadata"]["success"] is False
        assert "LLM unavailable" in result["metadata"]["error"]


# ===========================================================================
# LLMTRMAgent
# ===========================================================================


@pytest.mark.unit
class TestLLMTRMAgentQuality:
    def test_baseline_only(self):
        score = LLMTRMAgent._compute_quality_score("short text")
        assert score == pytest.approx(0.5)

    def test_initial_bonus(self):
        score = LLMTRMAgent._compute_quality_score("Initial answer here")
        assert score == pytest.approx(0.6)

    def test_review_bonus_via_review(self):
        score = LLMTRMAgent._compute_quality_score("Critical review of the answer")
        assert score == pytest.approx(0.7)

    def test_review_bonus_via_weakness(self):
        score = LLMTRMAgent._compute_quality_score("Weakness identified in logic")
        assert score == pytest.approx(0.7)

    def test_refined_bonus_via_refined(self):
        score = LLMTRMAgent._compute_quality_score("Refined answer below")
        assert score == pytest.approx(0.65)

    def test_refined_bonus_via_improved(self):
        score = LLMTRMAgent._compute_quality_score("Improved version of the answer")
        assert score == pytest.approx(0.65)

    def test_length_bonus(self):
        score = LLMTRMAgent._compute_quality_score("x" * 301)
        assert score == pytest.approx(0.55)

    def test_all_bonuses(self):
        text = "Initial answer\nReview: weakness found\nRefined improved version\n" + "x" * 300
        score = LLMTRMAgent._compute_quality_score(text)
        assert score == pytest.approx(1.0)

    def test_capped_at_one(self):
        text = "initial review weakness refined improved\n" + "x" * 500
        score = LLMTRMAgent._compute_quality_score(text)
        assert score <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMTRMAgentProcess:
    async def test_basic_process(self):
        adapter = _make_mock_adapter(
            text="### Initial Answer\nFirst pass\n### Critical Review\n- weakness\n### Refined Answer\nImproved",
            usage={"prompt_tokens": 40, "completion_tokens": 80, "total_tokens": 120},
        )
        agent = LLMTRMAgent(model_adapter=adapter)
        result = await agent.process(query="Summarize relativity")

        assert "Initial Answer" in result["response"]
        meta = result["metadata"]
        assert meta["success"] is True
        assert "final_quality_score" in meta
        assert meta["has_initial_answer"] is True
        assert meta["has_critical_review"] is True
        assert meta["has_refined_answer"] is True
        assert meta["strategy"] == "iterative_refinement"
        assert meta["token_usage"]["total_tokens"] == 120

    async def test_with_rag_context(self):
        adapter = _make_mock_adapter(text="refined response")
        agent = LLMTRMAgent(model_adapter=adapter)
        ctx = AgentContext(query="q", rag_context="reference material")
        await agent.process(context=ctx)

        call_kwargs = adapter.generate.call_args
        prompt_sent = call_kwargs.kwargs.get("prompt", "")
        assert "Additional Context" in prompt_sent
        assert "reference material" in prompt_sent

    async def test_without_rag_context(self):
        adapter = _make_mock_adapter(text="no rag")
        agent = LLMTRMAgent(model_adapter=adapter)
        await agent.process(query="q")

        call_kwargs = adapter.generate.call_args
        prompt_sent = call_kwargs.kwargs.get("prompt", "")
        assert "Additional Context" not in prompt_sent

    async def test_custom_temperature_and_tokens(self):
        adapter = _make_mock_adapter(text="resp")
        agent = LLMTRMAgent(model_adapter=adapter, temperature=0.2, max_tokens=800)
        await agent.process(query="q")

        call_kwargs = adapter.generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0.2
        assert call_kwargs.kwargs["max_tokens"] == 800

    async def test_default_name(self):
        agent = LLMTRMAgent(model_adapter=_make_mock_adapter())
        assert agent.name == "LLM_TRM"

    async def test_llm_error_propagates(self):
        adapter = AsyncMock()
        adapter.generate.side_effect = ConnectionError("network down")
        agent = LLMTRMAgent(model_adapter=adapter)
        result = await agent.process(query="q")
        assert result["metadata"]["success"] is False
        assert "network down" in result["metadata"]["error"]


# ===========================================================================
# ParallelAgent
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestParallelAgent:
    async def test_no_sub_agents(self):
        adapter = _make_mock_adapter()
        agent = ParallelAgent(model_adapter=adapter, sub_agents=[])
        result = await agent.process(query="q")
        assert "No sub-agents" in result["response"]

    async def test_single_sub_agent(self):
        adapter = _make_mock_adapter()
        sub = _ConcreteAgent(model_adapter=adapter, response_text="sub output")
        agent = ParallelAgent(model_adapter=adapter, sub_agents=[sub])
        result = await agent.process(query="q")
        assert result["response"] == "sub output"
        assert result["metadata"]["success"] is True

    async def test_multiple_sub_agents_picks_highest_confidence(self):
        adapter = _make_mock_adapter()
        sub1 = _ConcreteAgent(model_adapter=adapter, response_text="low", name="Low")
        sub2 = _ConcreteAgent(model_adapter=adapter, response_text="high", name="High")

        # sub1 returns confidence 0.8 by default; override sub2 to return higher
        original_process = sub2._process_impl

        async def high_confidence(context):
            r = await original_process(context)
            r.confidence = 0.95
            r.response = "high confidence result"
            return r

        sub2._process_impl = high_confidence

        agent = ParallelAgent(model_adapter=adapter, sub_agents=[sub1, sub2])
        result = await agent.process(query="q")
        assert result["response"] == "high confidence result"

    async def test_all_sub_agents_fail(self):
        adapter = _make_mock_adapter()
        sub1 = _FailingAgent(model_adapter=adapter, name="Fail1")
        sub2 = _FailingAgent(model_adapter=adapter, name="Fail2")
        agent = ParallelAgent(model_adapter=adapter, sub_agents=[sub1, sub2])
        result = await agent.process(query="q")
        # All failed, but the sub-agents return error results (success=False)
        assert result["metadata"]["success"] is False or "failed" in result["response"].lower()

    async def test_add_agent(self):
        adapter = _make_mock_adapter()
        agent = ParallelAgent(model_adapter=adapter)
        assert len(agent.sub_agents) == 0
        sub = _ConcreteAgent(model_adapter=adapter)
        agent.add_agent(sub)
        assert len(agent.sub_agents) == 1

    async def test_initialize_propagates(self):
        adapter = _make_mock_adapter()
        sub = _ConcreteAgent(model_adapter=adapter)
        agent = ParallelAgent(model_adapter=adapter, sub_agents=[sub])
        await agent.initialize()
        assert sub._initialized is True
        await agent.shutdown()
        assert sub._initialized is False


# ===========================================================================
# SequentialAgent
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSequentialAgent:
    async def test_no_sub_agents(self):
        adapter = _make_mock_adapter()
        agent = SequentialAgent(model_adapter=adapter, sub_agents=[])
        result = await agent.process(query="q")
        assert "No sub-agents" in result["response"]

    async def test_single_sub_agent(self):
        adapter = _make_mock_adapter()
        sub = _ConcreteAgent(model_adapter=adapter, response_text="step1 output")
        agent = SequentialAgent(model_adapter=adapter, sub_agents=[sub])
        result = await agent.process(query="q")
        assert result["response"] == "step1 output"
        assert result["metadata"]["success"] is True

    async def test_chain_passes_context(self):
        """Verify that output of first agent becomes rag_context for second."""
        adapter = _make_mock_adapter()

        received_contexts = []

        class _CapturingAgent(AsyncAgentBase):
            async def _process_impl(self, context: AgentContext) -> AgentResult:
                received_contexts.append(context)
                return AgentResult(
                    response=f"output from {self.name}",
                    confidence=0.7,
                )

        sub1 = _CapturingAgent(model_adapter=adapter, name="Step1")
        sub2 = _CapturingAgent(model_adapter=adapter, name="Step2")
        agent = SequentialAgent(model_adapter=adapter, sub_agents=[sub1, sub2])
        result = await agent.process(query="original query")

        # Step2 should have received Step1's output as rag_context
        assert len(received_contexts) == 2
        assert received_contexts[1].rag_context == "output from Step1"
        assert received_contexts[1].query == "original query"
        # Pipeline metadata
        assert result["metadata"]["success"] is True

    async def test_failure_stops_pipeline(self):
        adapter = _make_mock_adapter()
        sub1 = _FailingAgent(model_adapter=adapter, name="FailStep")
        sub2 = _ConcreteAgent(model_adapter=adapter, response_text="should not reach")
        agent = SequentialAgent(model_adapter=adapter, sub_agents=[sub1, sub2])
        result = await agent.process(query="q")
        # The first agent fails; result should indicate failure at FailStep
        assert result["metadata"]["success"] is False

    async def test_pipeline_metadata(self):
        adapter = _make_mock_adapter()
        sub1 = _ConcreteAgent(model_adapter=adapter, response_text="r1", name="A")
        sub2 = _ConcreteAgent(model_adapter=adapter, response_text="r2", name="B")
        agent = SequentialAgent(model_adapter=adapter, sub_agents=[sub1, sub2])
        result = await agent.process(query="q")
        # Final response is from last agent
        assert result["response"] == "r2"


# ===========================================================================
# Edge cases & error paths
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestEdgeCases:
    async def test_process_with_kwargs(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        result = await agent.process(query="q", rag_context="rag stuff", extra_param="val")
        assert result["metadata"]["success"] is True

    async def test_stats_zero_requests(self):
        agent = _ConcreteAgent(model_adapter=_make_mock_adapter())
        stats = agent.stats
        assert stats["request_count"] == 0
        assert stats["average_processing_time_ms"] == 0.0

    async def test_on_error_records_metrics(self):
        metrics = MagicMock()
        agent = _FailingAgent(model_adapter=_make_mock_adapter(), metrics_collector=metrics)
        await agent.process(query="q")
        metrics.record_error.assert_called_once_with("_FailingAgent", "RuntimeError")

    async def test_pre_and_post_hooks(self):
        """Verify pre_process and post_process hooks are called."""
        adapter = _make_mock_adapter()

        class _HookedAgent(AsyncAgentBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.pre_called = False
                self.post_called = False

            async def pre_process(self, context: AgentContext) -> AgentContext:
                self.pre_called = True
                return context

            async def post_process(self, context: AgentContext, result: AgentResult) -> AgentResult:
                self.post_called = True
                result.confidence = 0.99
                return result

            async def _process_impl(self, context: AgentContext) -> AgentResult:
                return AgentResult(response="hooked")

        agent = _HookedAgent(model_adapter=adapter)
        result = await agent.process(query="q")
        assert agent.pre_called is True
        assert agent.post_called is True
        assert result["metadata"]["confidence"] == 0.99

    async def test_hrm_quality_with_hash_markers(self):
        """### triggers subproblem bonus even without 'sub-problem' text."""
        score = LLMHRMAgent._compute_quality_score("###\nSome section")
        assert score >= 0.7

    async def test_trm_no_keywords(self):
        """Plain text with no keywords gets baseline only."""
        score = LLMTRMAgent._compute_quality_score("Hello world")
        assert score == pytest.approx(0.5)
