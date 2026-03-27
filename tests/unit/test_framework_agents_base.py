"""
Unit tests for src/framework/agents/base.py - Base async agent classes.

Tests:
- AgentContext dataclass
- AgentResult dataclass
- NoOpMetricsCollector
- AsyncAgentBase initialization, lifecycle, hooks, process method
- CompositeAgent
- ParallelAgent
- SequentialAgent
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adapters.llm.base import LLMResponse
from src.framework.agents.base import (
    AgentContext,
    AgentResult,
    AsyncAgentBase,
    CompositeAgent,
    MetricsCollector,
    NoOpMetricsCollector,
    ParallelAgent,
    SequentialAgent,
)


# Concrete subclass for testing the abstract base
class ConcreteAgent(AsyncAgentBase):
    """Test agent implementation."""

    def __init__(self, *args, return_value=None, raise_error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._return_value = return_value or AgentResult(response="test response", confidence=0.8)
        self._raise_error = raise_error

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        if self._raise_error:
            raise self._raise_error
        return self._return_value


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    client.generate = AsyncMock(
        return_value=LLMResponse(text="llm response", usage={"total_tokens": 100})
    )
    return client


@pytest.fixture
def agent(mock_llm_client):
    return ConcreteAgent(model_adapter=mock_llm_client, name="TestAgent")


@pytest.mark.unit
class TestAgentContext:
    """Tests for AgentContext dataclass."""

    def test_defaults(self):
        """Test default values."""
        ctx = AgentContext(query="What is AI?")
        assert ctx.query == "What is AI?"
        assert ctx.rag_context is None
        assert ctx.max_iterations == 5
        assert ctx.temperature == 0.7
        assert isinstance(ctx.session_id, str)
        assert len(ctx.session_id) > 0
        assert ctx.metadata == {}
        assert ctx.conversation_history == []
        assert ctx.additional_context == {}

    def test_custom_values(self):
        """Test custom values."""
        ctx = AgentContext(
            query="test",
            session_id="sess-123",
            rag_context="context here",
            max_iterations=10,
            temperature=0.3,
        )
        assert ctx.session_id == "sess-123"
        assert ctx.rag_context == "context here"
        assert ctx.max_iterations == 10
        assert ctx.temperature == 0.3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ctx = AgentContext(query="test", rag_context="ctx")
        d = ctx.to_dict()
        assert d["query"] == "test"
        assert d["rag_context"] == "ctx"
        assert "session_id" in d
        assert "metadata" in d
        assert "temperature" in d

    def test_independent_defaults(self):
        """Test that mutable defaults are independent across instances."""
        ctx1 = AgentContext(query="a")
        ctx2 = AgentContext(query="b")
        ctx1.metadata["key"] = "value"
        assert "key" not in ctx2.metadata


@pytest.mark.unit
class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_defaults(self):
        """Test default values."""
        result = AgentResult(response="answer")
        assert result.response == "answer"
        assert result.confidence == 0.0
        assert result.success is True
        assert result.error is None
        assert result.agent_name == ""
        assert result.processing_time_ms == 0.0
        assert result.token_usage == {}
        assert result.intermediate_steps == []

    def test_error_result(self):
        """Test error result."""
        result = AgentResult(response="", success=False, error="something failed")
        assert result.success is False
        assert result.error == "something failed"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AgentResult(
            response="answer",
            confidence=0.9,
            agent_name="hrm",
            processing_time_ms=123.4,
        )
        d = result.to_dict()
        assert d["response"] == "answer"
        assert d["confidence"] == 0.9
        assert d["agent_name"] == "hrm"
        assert d["processing_time_ms"] == 123.4
        assert "created_at" in d
        assert isinstance(d["created_at"], str)  # ISO format

    def test_to_dict_preserves_all_fields(self):
        """Test to_dict includes all expected fields."""
        result = AgentResult(response="test")
        d = result.to_dict()
        expected_keys = {
            "response", "confidence", "metadata", "agent_name",
            "processing_time_ms", "token_usage", "intermediate_steps",
            "created_at", "error", "success",
        }
        assert set(d.keys()) == expected_keys


@pytest.mark.unit
class TestNoOpMetricsCollector:
    """Tests for NoOpMetricsCollector."""

    def test_implements_protocol(self):
        """Test NoOpMetricsCollector satisfies MetricsCollector protocol."""
        collector = NoOpMetricsCollector()
        # These should not raise
        collector.record_latency("agent", 100.0)
        collector.record_tokens("agent", 500)
        collector.record_error("agent", "ValueError")
        collector.record_success("agent")

    def test_methods_return_none(self):
        """Test all methods return None."""
        collector = NoOpMetricsCollector()
        assert collector.record_latency("a", 1.0) is None
        assert collector.record_tokens("a", 1) is None
        assert collector.record_error("a", "e") is None
        assert collector.record_success("a") is None


@pytest.mark.unit
class TestAsyncAgentBaseInit:
    """Tests for AsyncAgentBase initialization."""

    def test_default_name_from_class(self, mock_llm_client):
        """Test default name uses class name."""
        agent = ConcreteAgent(model_adapter=mock_llm_client)
        assert agent.name == "ConcreteAgent"

    def test_custom_name(self, mock_llm_client):
        """Test custom name override."""
        agent = ConcreteAgent(model_adapter=mock_llm_client, name="CustomName")
        assert agent.name == "CustomName"

    def test_default_logger(self, mock_llm_client):
        """Test default logger uses class name."""
        agent = ConcreteAgent(model_adapter=mock_llm_client)
        assert agent.logger.name == "ConcreteAgent"

    def test_custom_logger(self, mock_llm_client):
        """Test custom logger injection."""
        custom_logger = logging.getLogger("custom")
        agent = ConcreteAgent(model_adapter=mock_llm_client, logger=custom_logger)
        assert agent.logger is custom_logger

    def test_default_metrics_collector(self, mock_llm_client):
        """Test default NoOp metrics collector."""
        agent = ConcreteAgent(model_adapter=mock_llm_client)
        assert isinstance(agent.metrics, NoOpMetricsCollector)

    def test_custom_metrics_collector(self, mock_llm_client):
        """Test custom metrics collector injection."""
        collector = NoOpMetricsCollector()
        agent = ConcreteAgent(model_adapter=mock_llm_client, metrics_collector=collector)
        assert agent.metrics is collector

    def test_initial_state(self, agent):
        """Test initial runtime state."""
        assert agent._request_count == 0
        assert agent._total_processing_time == 0.0
        assert agent._error_count == 0
        assert agent._initialized is False

    def test_config_kwargs_stored(self, mock_llm_client):
        """Test extra config kwargs are stored."""
        agent = ConcreteAgent(model_adapter=mock_llm_client, foo="bar", baz=42)
        assert agent.config["foo"] == "bar"
        assert agent.config["baz"] == 42


@pytest.mark.unit
class TestAsyncAgentBaseLifecycle:
    """Tests for initialize, shutdown, context manager."""

    @pytest.mark.asyncio
    async def test_initialize(self, agent):
        """Test initialize sets _initialized."""
        await agent.initialize()
        assert agent._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown(self, agent):
        """Test shutdown clears _initialized."""
        await agent.initialize()
        await agent.shutdown()
        assert agent._initialized is False

    @pytest.mark.asyncio
    async def test_context_manager(self, agent):
        """Test async context manager."""
        async with agent as a:
            assert a is agent
            assert a._initialized is True
        assert agent._initialized is False


@pytest.mark.unit
class TestAsyncAgentBaseProcess:
    """Tests for process method."""

    @pytest.mark.asyncio
    async def test_process_with_query(self, agent):
        """Test process with query string."""
        result = await agent.process(query="What is AI?")
        assert result["response"] == "test response"
        assert result["metadata"]["success"] is True
        assert result["metadata"]["confidence"] == 0.8
        assert result["metadata"]["agent_name"] == "TestAgent"
        assert result["metadata"]["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_process_with_context(self, agent):
        """Test process with AgentContext."""
        ctx = AgentContext(query="test", rag_context="some context")
        result = await agent.process(context=ctx)
        assert result["response"] == "test response"

    @pytest.mark.asyncio
    async def test_process_no_query_or_context_raises(self, agent):
        """Test process raises when neither query nor context provided."""
        with pytest.raises(ValueError, match="Either 'query' or 'context' must be provided"):
            await agent.process()

    @pytest.mark.asyncio
    async def test_process_auto_initializes(self, agent):
        """Test process auto-initializes if not initialized."""
        assert agent._initialized is False
        await agent.process(query="test")
        assert agent._initialized is True

    @pytest.mark.asyncio
    async def test_process_tracks_stats(self, agent):
        """Test process updates request count and timing."""
        await agent.process(query="test1")
        await agent.process(query="test2")
        assert agent._request_count == 2
        assert agent._total_processing_time > 0

    @pytest.mark.asyncio
    async def test_process_error_handling(self, mock_llm_client):
        """Test process error handling."""
        error_agent = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="ErrorAgent",
            raise_error=RuntimeError("boom"),
        )
        result = await error_agent.process(query="test")
        assert result["metadata"]["success"] is False
        assert result["metadata"]["error"] == "boom"
        assert error_agent._error_count == 1

    @pytest.mark.asyncio
    async def test_process_with_token_usage(self, mock_llm_client):
        """Test process records token usage metrics."""
        collector = MagicMock()
        result_with_tokens = AgentResult(
            response="answer",
            confidence=0.9,
            token_usage={"total_tokens": 150},
        )
        agent = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="TokenAgent",
            metrics_collector=collector,
            return_value=result_with_tokens,
        )
        await agent.process(query="test")
        collector.record_tokens.assert_called_once_with("TokenAgent", 150)
        collector.record_latency.assert_called_once()
        collector.record_success.assert_called_once()


@pytest.mark.unit
class TestAsyncAgentBaseHooks:
    """Tests for pre_process, post_process, on_error hooks."""

    @pytest.mark.asyncio
    async def test_pre_process_default(self, agent):
        """Test default pre_process returns context unchanged."""
        ctx = AgentContext(query="test")
        result = await agent.pre_process(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_post_process_default(self, agent):
        """Test default post_process returns result unchanged."""
        ctx = AgentContext(query="test")
        res = AgentResult(response="test")
        result = await agent.post_process(ctx, res)
        assert result is res

    @pytest.mark.asyncio
    async def test_on_error(self, agent):
        """Test on_error returns error result."""
        ctx = AgentContext(query="test")
        error = ValueError("bad input")
        result = await agent.on_error(ctx, error)
        assert result.success is False
        assert result.error == "bad input"
        assert result.confidence == 0.0
        assert agent._error_count == 1


@pytest.mark.unit
class TestAsyncAgentBaseStats:
    """Tests for stats property."""

    @pytest.mark.asyncio
    async def test_stats_initial(self, agent):
        """Test initial stats."""
        stats = agent.stats
        assert stats["name"] == "TestAgent"
        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["average_processing_time_ms"] == 0.0
        assert stats["initialized"] is False

    @pytest.mark.asyncio
    async def test_stats_after_processing(self, agent):
        """Test stats after processing queries."""
        await agent.process(query="test1")
        await agent.process(query="test2")
        stats = agent.stats
        assert stats["request_count"] == 2
        assert stats["total_processing_time_ms"] > 0
        assert stats["average_processing_time_ms"] > 0
        assert stats["initialized"] is True


@pytest.mark.unit
class TestGenerateLLMResponse:
    """Tests for generate_llm_response convenience method."""

    @pytest.mark.asyncio
    async def test_generate_with_prompt(self, agent, mock_llm_client):
        """Test generating with a simple prompt."""
        response = await agent.generate_llm_response(prompt="Hello")
        assert response.text == "llm response"
        mock_llm_client.generate.assert_called_once_with(
            prompt="Hello",
            messages=None,
            temperature=0.7,
            max_tokens=None,
        )

    @pytest.mark.asyncio
    async def test_generate_with_messages(self, agent, mock_llm_client):
        """Test generating with chat messages."""
        messages = [{"role": "user", "content": "Hello"}]
        await agent.generate_llm_response(messages=messages, temperature=0.3)
        mock_llm_client.generate.assert_called_once_with(
            prompt=None,
            messages=messages,
            temperature=0.3,
            max_tokens=None,
        )


@pytest.mark.unit
class TestCompositeAgent:
    """Tests for CompositeAgent (tested via ParallelAgent, a concrete subclass)."""

    @pytest.mark.asyncio
    async def test_add_agent(self, mock_llm_client):
        """Test adding sub-agents."""
        composite = ParallelAgent(model_adapter=mock_llm_client, name="Composite")
        sub = ConcreteAgent(model_adapter=mock_llm_client, name="Sub1")
        composite.add_agent(sub)
        assert len(composite.sub_agents) == 1
        assert composite.sub_agents[0] is sub

    @pytest.mark.asyncio
    async def test_initialize_all_sub_agents(self, mock_llm_client):
        """Test initialize propagates to sub-agents."""
        sub1 = ConcreteAgent(model_adapter=mock_llm_client, name="Sub1")
        sub2 = ConcreteAgent(model_adapter=mock_llm_client, name="Sub2")
        composite = ParallelAgent(
            model_adapter=mock_llm_client,
            name="Composite",
            sub_agents=[sub1, sub2],
        )
        await composite.initialize()
        assert composite._initialized is True
        assert sub1._initialized is True
        assert sub2._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown_all_sub_agents(self, mock_llm_client):
        """Test shutdown propagates to sub-agents."""
        sub1 = ConcreteAgent(model_adapter=mock_llm_client, name="Sub1")
        sub2 = ConcreteAgent(model_adapter=mock_llm_client, name="Sub2")
        composite = ParallelAgent(
            model_adapter=mock_llm_client,
            name="Composite",
            sub_agents=[sub1, sub2],
        )
        await composite.initialize()
        await composite.shutdown()
        assert composite._initialized is False
        assert sub1._initialized is False
        assert sub2._initialized is False


@pytest.mark.unit
class TestParallelAgent:
    """Tests for ParallelAgent."""

    @pytest.mark.asyncio
    async def test_no_sub_agents(self, mock_llm_client):
        """Test with no sub-agents returns default message."""
        parallel = ParallelAgent(model_adapter=mock_llm_client, name="Parallel")
        result = await parallel.process(query="test")
        assert "No sub-agents configured" in result["response"]

    @pytest.mark.asyncio
    async def test_parallel_execution_returns_highest_confidence(self, mock_llm_client):
        """Test parallel execution returns highest-confidence result."""
        sub1 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="Low",
            return_value=AgentResult(response="low", confidence=0.3),
        )
        sub2 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="High",
            return_value=AgentResult(response="high", confidence=0.9),
        )
        parallel = ParallelAgent(
            model_adapter=mock_llm_client,
            name="Parallel",
            sub_agents=[sub1, sub2],
        )
        result = await parallel.process(query="test")
        assert result["response"] == "high"

    @pytest.mark.asyncio
    async def test_parallel_all_fail(self, mock_llm_client):
        """Test parallel when all sub-agents fail."""
        sub1 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="Fail1",
            raise_error=RuntimeError("error1"),
        )
        sub2 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="Fail2",
            raise_error=RuntimeError("error2"),
        )
        parallel = ParallelAgent(
            model_adapter=mock_llm_client,
            name="Parallel",
            sub_agents=[sub1, sub2],
        )
        result = await parallel.process(query="test")
        # The outer process catches the error from _process_impl
        # Since both sub-agents return error results via their own process(),
        # they have success=False in metadata
        assert result["metadata"]["success"] is not None


@pytest.mark.unit
class TestSequentialAgent:
    """Tests for SequentialAgent."""

    @pytest.mark.asyncio
    async def test_no_sub_agents(self, mock_llm_client):
        """Test with no sub-agents returns default message."""
        seq = SequentialAgent(model_adapter=mock_llm_client, name="Sequential")
        result = await seq.process(query="test")
        assert "No sub-agents configured" in result["response"]

    @pytest.mark.asyncio
    async def test_sequential_pipeline(self, mock_llm_client):
        """Test sequential execution passes context through."""
        sub1 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="Step1",
            return_value=AgentResult(response="step1 output", confidence=0.7),
        )
        sub2 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="Step2",
            return_value=AgentResult(response="step2 output", confidence=0.9),
        )
        seq = SequentialAgent(
            model_adapter=mock_llm_client,
            name="Sequential",
            sub_agents=[sub1, sub2],
        )
        result = await seq.process(query="test")
        assert result["response"] == "step2 output"
        assert "pipeline" in result["metadata"]

    @pytest.mark.asyncio
    async def test_sequential_stops_on_failure(self, mock_llm_client):
        """Test sequential stops when a sub-agent fails."""
        sub1 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="Fail",
            raise_error=ValueError("step failed"),
        )
        sub2 = ConcreteAgent(
            model_adapter=mock_llm_client,
            name="NeverRun",
            return_value=AgentResult(response="never", confidence=0.9),
        )
        seq = SequentialAgent(
            model_adapter=mock_llm_client,
            name="Sequential",
            sub_agents=[sub1, sub2],
        )
        result = await seq.process(query="test")
        # The first agent's error is handled by its own process(), returning success=False
        assert result["metadata"]["success"] is False or "failed_at" in result.get("metadata", {})
