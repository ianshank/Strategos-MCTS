"""
Unit tests for LangGraph streaming support.

Tests node-level streaming (astream) and event-level streaming (astream_events).

Based on: NEXT_STEPS_PLAN.md Phase 3.1
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_langgraph_app():
    """Create a mock LangGraph compiled app."""
    app = AsyncMock()

    # Mock astream to yield node updates
    async def mock_astream(state, config=None):
        yield {"entry": {"iteration": 0, "agent_outputs": []}}
        yield {"retrieve_context": {"rag_context": "Test context"}}
        yield {"route_decision": {"next_agent": "hrm"}}
        yield {"synthesize": {"final_response": "Test response"}}

    app.astream = mock_astream

    # Mock astream_events to yield detailed events
    async def mock_astream_events(state, config=None, version="v2"):
        yield {
            "event": "on_chain_start",
            "name": "entry",
            "run_id": "run-123",
            "data": {},
            "metadata": {},
            "tags": [],
        }
        yield {
            "event": "on_llm_start",
            "name": "synthesis_llm",
            "run_id": "run-456",
            "data": {"input": "Synthesize response"},
            "metadata": {"model": "gpt-4"},
            "tags": ["synthesis"],
        }
        yield {
            "event": "on_llm_stream",
            "name": "synthesis_llm",
            "run_id": "run-456",
            "data": {"chunk": MagicMock(content="Test ")},
            "metadata": {},
            "tags": [],
        }
        yield {
            "event": "on_llm_stream",
            "name": "synthesis_llm",
            "run_id": "run-456",
            "data": {"chunk": MagicMock(content="response")},
            "metadata": {},
            "tags": [],
        }
        yield {
            "event": "on_llm_end",
            "name": "synthesis_llm",
            "run_id": "run-456",
            "data": {"output": "Test response"},
            "metadata": {},
            "tags": [],
        }
        yield {
            "event": "on_chain_end",
            "name": "synthesize",
            "run_id": "run-123",
            "data": {"output": {"final_response": "Test response"}},
            "metadata": {},
            "tags": [],
        }

    app.astream_events = mock_astream_events

    return app


@pytest.fixture
def mock_integrated_framework(mock_langgraph_app):
    """Create a mock IntegratedFramework for testing streaming."""
    framework = MagicMock()
    framework.app = mock_langgraph_app
    framework.logger = MagicMock(spec=logging.Logger)
    framework.graph_builder = MagicMock()
    framework.graph_builder.max_iterations = 3

    # Bind actual streaming methods
    import src.framework.graph as graph_module

    framework.astream = lambda *args, **kwargs: graph_module.IntegratedFramework.astream(framework, *args, **kwargs)
    framework.astream_events = lambda *args, **kwargs: graph_module.IntegratedFramework.astream_events(
        framework, *args, **kwargs
    )

    return framework


# =============================================================================
# Node-Level Streaming Tests (astream)
# =============================================================================


class TestNodeLevelStreaming:
    """Tests for astream method."""

    @pytest.mark.asyncio
    async def test_astream_yields_node_updates(self, mock_integrated_framework):
        """Test astream yields node updates as they complete."""
        nodes_received = []

        async for node_name, _state_update in mock_integrated_framework.astream(query="Test query"):
            nodes_received.append(node_name)

        assert "entry" in nodes_received
        assert "synthesize" in nodes_received
        assert len(nodes_received) == 4

    @pytest.mark.asyncio
    async def test_astream_includes_state_updates(self, mock_integrated_framework):
        """Test astream includes state updates for each node."""
        updates = {}

        async for node_name, state_update in mock_integrated_framework.astream(query="Test query"):
            updates[node_name] = state_update

        assert updates["entry"]["iteration"] == 0
        assert updates["retrieve_context"]["rag_context"] == "Test context"
        assert updates["synthesize"]["final_response"] == "Test response"

    @pytest.mark.asyncio
    async def test_astream_with_custom_config(self, mock_integrated_framework):
        """Test astream accepts custom config."""
        custom_config = {"configurable": {"thread_id": "custom-thread"}}

        nodes = []
        async for node_name, _ in mock_integrated_framework.astream(
            query="Test query",
            config=custom_config,
        ):
            nodes.append(node_name)

        assert len(nodes) > 0

    @pytest.mark.asyncio
    async def test_astream_with_rag_option(self, mock_integrated_framework):
        """Test astream accepts use_rag option."""
        nodes = []
        async for node_name, _ in mock_integrated_framework.astream(
            query="Test query",
            use_rag=True,
        ):
            nodes.append(node_name)

        assert "retrieve_context" in nodes

    @pytest.mark.asyncio
    async def test_astream_raises_error_without_app(self):
        """Test astream raises error when app is None."""
        framework = MagicMock()
        framework.app = None
        framework.logger = MagicMock()

        import src.framework.graph as graph_module

        framework.astream = lambda *args, **kwargs: graph_module.IntegratedFramework.astream(framework, *args, **kwargs)

        with pytest.raises(RuntimeError, match="LangGraph not available"):
            async for _ in framework.astream(query="Test"):
                pass


# =============================================================================
# Event-Level Streaming Tests (astream_events)
# =============================================================================


class TestEventLevelStreaming:
    """Tests for astream_events method."""

    @pytest.mark.asyncio
    async def test_astream_events_yields_all_events(self, mock_integrated_framework):
        """Test astream_events yields all event types."""
        events = []

        async for event in mock_integrated_framework.astream_events(query="Test query"):
            events.append(event)

        event_types = [e["event_type"] for e in events]
        assert "on_chain_start" in event_types
        assert "on_llm_start" in event_types
        assert "on_llm_stream" in event_types
        assert "on_llm_end" in event_types
        assert "on_chain_end" in event_types

    @pytest.mark.asyncio
    async def test_astream_events_extracts_tokens(self, mock_integrated_framework):
        """Test astream_events extracts token content from LLM streams."""
        tokens = []

        async for event in mock_integrated_framework.astream_events(query="Test query"):
            if event["event_type"] == "on_llm_stream" and "token" in event:
                tokens.append(event["token"])

        assert "Test " in tokens
        assert "response" in tokens

    @pytest.mark.asyncio
    async def test_astream_events_filters_by_type(self, mock_integrated_framework):
        """Test astream_events filters by event type."""
        events = []

        async for event in mock_integrated_framework.astream_events(
            query="Test query",
            include_types=["on_llm_stream"],
        ):
            events.append(event)

        # Should only have LLM stream events
        assert all(e["event_type"] == "on_llm_stream" for e in events)
        assert len(events) == 2  # Two stream chunks

    @pytest.mark.asyncio
    async def test_astream_events_includes_metadata(self, mock_integrated_framework):
        """Test astream_events includes event metadata."""
        llm_start_events = []

        async for event in mock_integrated_framework.astream_events(
            query="Test query",
            include_types=["on_llm_start"],
        ):
            llm_start_events.append(event)

        assert len(llm_start_events) == 1
        assert llm_start_events[0]["metadata"].get("model") == "gpt-4"

    @pytest.mark.asyncio
    async def test_astream_events_includes_run_id(self, mock_integrated_framework):
        """Test astream_events includes run_id for tracing."""
        async for event in mock_integrated_framework.astream_events(query="Test query"):
            assert "run_id" in event

    @pytest.mark.asyncio
    async def test_astream_events_standardized_format(self, mock_integrated_framework):
        """Test astream_events returns standardized event format."""
        required_fields = ["event_type", "name", "run_id", "data", "metadata", "tags"]

        async for event in mock_integrated_framework.astream_events(query="Test query"):
            for field in required_fields:
                assert field in event, f"Missing required field: {field}"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestStreamingErrorHandling:
    """Tests for streaming error handling."""

    @pytest.mark.asyncio
    async def test_astream_events_handles_errors_gracefully(self):
        """Test astream_events handles errors and yields error event."""
        app = AsyncMock()

        async def error_stream(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "entry", "run_id": "1", "data": {}}
            raise Exception("Streaming error")

        app.astream_events = error_stream

        framework = MagicMock()
        framework.app = app
        framework.logger = MagicMock()
        framework.graph_builder = MagicMock()
        framework.graph_builder.max_iterations = 3

        import src.framework.graph as graph_module

        framework.astream_events = lambda *args, **kwargs: graph_module.IntegratedFramework.astream_events(
            framework, *args, **kwargs
        )

        events = []
        async for event in framework.astream_events(query="Test"):
            events.append(event)

        # Should have error event
        error_events = [e for e in events if e["event_type"] == "on_error"]
        assert len(error_events) == 1
        assert "Streaming error" in error_events[0]["data"]["error"]


# =============================================================================
# Streaming Configuration Tests
# =============================================================================


class TestStreamingConfiguration:
    """Tests for streaming configuration options."""

    @pytest.mark.asyncio
    async def test_astream_uses_default_config(self, mock_integrated_framework):
        """Test astream uses default config when none provided."""
        async for _ in mock_integrated_framework.astream(query="Test"):
            pass  # Just ensure it runs without error

    @pytest.mark.asyncio
    async def test_astream_events_default_include_types(self, mock_integrated_framework):
        """Test astream_events includes common event types by default."""
        event_types_seen = set()

        async for event in mock_integrated_framework.astream_events(query="Test"):
            event_types_seen.add(event["event_type"])

        # Default should include these types
        expected = {"on_chain_start", "on_chain_end", "on_llm_start", "on_llm_stream", "on_llm_end"}
        assert event_types_seen.intersection(expected) == expected


# =============================================================================
# Logging Tests
# =============================================================================


class TestStreamingLogging:
    """Tests for streaming logging."""

    @pytest.mark.asyncio
    async def test_astream_logs_debug_messages(self, mock_integrated_framework):
        """Test astream logs debug messages for node completions."""
        async for _ in mock_integrated_framework.astream(query="Test query"):
            pass

        mock_integrated_framework.logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_astream_events_logs_event_types(self, mock_integrated_framework):
        """Test astream_events logs event information."""
        async for _ in mock_integrated_framework.astream_events(query="Test query"):
            pass

        mock_integrated_framework.logger.debug.assert_called()
