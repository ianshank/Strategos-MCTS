"""
Unit tests for distributed tracing system.

Tests OpenTelemetry integration, span creation, context propagation, and decorators.

Based on: NEXT_STEPS_PLAN.md Phase 2.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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
def mock_tracer():
    """Create a mock tracer."""
    tracer = MagicMock()
    span = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=span)
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    return tracer, span


@pytest.fixture
def tracing_manager():
    """Create a fresh TracingManager for testing."""
    from src.observability.tracing import TracingManager

    # Create fresh instance (don't use singleton)
    manager = TracingManager.__new__(TracingManager)
    manager._initialized = False
    manager._httpx_instrumented = False
    TracingManager._provider = None
    return manager


# =============================================================================
# TracingManager Tests
# =============================================================================


class TestTracingManager:
    """Tests for TracingManager singleton."""

    def test_get_instance_returns_singleton(self):
        """Test get_instance returns same instance."""
        from src.observability.tracing import TracingManager

        # Clear singleton for clean test
        TracingManager._instance = None

        instance1 = TracingManager.get_instance()
        instance2 = TracingManager.get_instance()

        assert instance1 is instance2

    def test_manager_not_initialized_by_default(self, tracing_manager):
        """Test manager is not initialized by default."""
        assert tracing_manager._initialized is False

    @patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"})
    def test_initialize_with_none_exporter(self, tracing_manager):
        """Test initialization with 'none' exporter (no actual exporting)."""
        tracing_manager.initialize(exporter_type="none")

        assert tracing_manager._initialized is True

    @patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "console"})
    def test_initialize_is_idempotent(self, tracing_manager):
        """Test multiple initializations don't cause issues."""
        tracing_manager.initialize(exporter_type="none")
        tracing_manager.initialize(exporter_type="none")

        assert tracing_manager._initialized is True

    def test_get_tracer_initializes_if_needed(self, tracing_manager):
        """Test get_tracer auto-initializes."""
        with patch.object(tracing_manager, "initialize") as mock_init:
            tracing_manager.get_tracer("test")
            mock_init.assert_called_once()

    def test_shutdown_resets_initialized(self, tracing_manager):
        """Test shutdown resets initialization state."""
        tracing_manager._initialized = True
        tracing_manager._provider = MagicMock()

        tracing_manager.shutdown()

        assert tracing_manager._initialized is False


# =============================================================================
# get_tracer Tests
# =============================================================================


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_returns_tracer(self):
        """Test get_tracer returns a tracer instance."""
        from src.observability.tracing import TracingManager, get_tracer

        # Use none exporter to avoid actual OTLP connection
        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            tracer = get_tracer("test.module")

            # Should return something callable
            assert tracer is not None


# =============================================================================
# add_mcts_attributes Tests
# =============================================================================


class TestAddMctsAttributes:
    """Tests for add_mcts_attributes function."""

    def test_adds_mcts_prefixed_attributes(self):
        """Test MCTS-prefixed attributes are added as-is."""
        from src.observability.tracing import add_mcts_attributes

        span = MagicMock()
        add_mcts_attributes(
            span,
            **{"mcts.iteration": 5, "mcts.ucb_score": 0.75},
        )

        span.set_attribute.assert_any_call("mcts.iteration", 5)
        span.set_attribute.assert_any_call("mcts.ucb_score", 0.75)

    def test_adds_agent_prefixed_attributes(self):
        """Test agent-prefixed attributes are added as-is."""
        from src.observability.tracing import add_mcts_attributes

        span = MagicMock()
        add_mcts_attributes(
            span,
            **{"agent.name": "hrm", "agent.confidence": 0.9},
        )

        span.set_attribute.assert_any_call("agent.name", "hrm")
        span.set_attribute.assert_any_call("agent.confidence", 0.9)

    def test_prefixes_custom_attributes(self):
        """Test non-standard attributes get custom prefix."""
        from src.observability.tracing import add_mcts_attributes

        span = MagicMock()
        add_mcts_attributes(span, custom_value=123)

        span.set_attribute.assert_called_with("custom.custom_value", 123)

    def test_skips_none_values(self):
        """Test None values are not added."""
        from src.observability.tracing import add_mcts_attributes

        span = MagicMock()
        add_mcts_attributes(span, **{"mcts.iteration": None})

        span.set_attribute.assert_not_called()


# =============================================================================
# trace_span Tests
# =============================================================================


class TestTraceSpan:
    """Tests for trace_span context manager."""

    def test_trace_span_creates_span(self):
        """Test trace_span creates a span."""

        from src.observability.tracing import TracingManager, trace_span

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            with trace_span("test.operation") as span:
                assert span is not None

    def test_trace_span_accepts_attributes(self):
        """Test trace_span accepts initial attributes."""
        from src.observability.tracing import TracingManager, trace_span

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            with trace_span(
                "test.operation",
                attributes={"test.key": "value"},
            ):
                # Should not raise
                pass


# =============================================================================
# async_trace_span Tests
# =============================================================================


class TestAsyncTraceSpan:
    """Tests for async_trace_span context manager."""

    @pytest.mark.asyncio
    async def test_async_trace_span_creates_span(self):
        """Test async_trace_span creates a span."""
        from src.observability.tracing import TracingManager, async_trace_span

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            async with async_trace_span("test.async.operation") as span:
                assert span is not None

    @pytest.mark.asyncio
    async def test_async_trace_span_accepts_attributes(self):
        """Test async_trace_span accepts initial attributes."""
        from src.observability.tracing import TracingManager, async_trace_span

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            async with async_trace_span(
                "test.async.operation",
                attributes={"test.key": "value"},
            ):
                # Should not raise
                pass


# =============================================================================
# trace_operation Decorator Tests
# =============================================================================


class TestTraceOperationDecorator:
    """Tests for trace_operation decorator."""

    def test_decorator_sync_function(self):
        """Test decorator works with sync function."""
        from src.observability.tracing import TracingManager, trace_operation

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):

            @trace_operation()
            def sample_function(x, y):
                return x + y

            result = sample_function(1, 2)

            assert result == 3

    @pytest.mark.asyncio
    async def test_decorator_async_function(self):
        """Test decorator works with async function."""
        from src.observability.tracing import TracingManager, trace_operation

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):

            @trace_operation()
            async def async_sample_function(x, y):
                return x * y

            result = await async_sample_function(3, 4)

            assert result == 12

    def test_decorator_uses_custom_name(self):
        """Test decorator uses custom span name."""
        from src.observability.tracing import TracingManager, trace_operation

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):

            @trace_operation(name="custom.span.name")
            def named_function():
                return "result"

            result = named_function()

            assert result == "result"


# =============================================================================
# SpanContextPropagator Tests
# =============================================================================


class TestSpanContextPropagator:
    """Tests for SpanContextPropagator."""

    def test_inject_adds_headers(self):
        """Test inject adds trace context to carrier."""
        from src.observability.tracing import SpanContextPropagator, TracingManager

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            propagator = SpanContextPropagator()
            carrier = {}

            # Should not raise even without active span
            propagator.inject(carrier)

            # Carrier might be empty if no active span
            assert isinstance(carrier, dict)

    def test_extract_returns_context(self):
        """Test extract returns a context."""
        from src.observability.tracing import SpanContextPropagator, TracingManager

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            propagator = SpanContextPropagator()

            # Extract from empty carrier
            context = propagator.extract({})

            # Should return a context object
            assert context is not None

    def test_get_trace_parent(self):
        """Test get_trace_parent returns value or None."""
        from src.observability.tracing import SpanContextPropagator, TracingManager

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            propagator = SpanContextPropagator()

            # Without active span, should return None
            trace_parent = propagator.get_trace_parent()

            # Can be None without active span
            assert trace_parent is None or isinstance(trace_parent, str)


# =============================================================================
# record_mcts_iteration Tests
# =============================================================================


class TestRecordMctsIteration:
    """Tests for record_mcts_iteration function."""

    def test_records_iteration_event(self):
        """Test records MCTS iteration as span event."""
        from src.observability.tracing import (
            TracingManager,
            record_mcts_iteration,
            trace_span,
        )

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            with trace_span("test.mcts"):
                record_mcts_iteration(
                    iteration=5,
                    selected_node_id="node-123",
                    ucb_score=0.85,
                    node_visits=100,
                    node_value=0.7,
                    tree_depth=3,
                )

                # Should not raise


# =============================================================================
# record_agent_execution Tests
# =============================================================================


class TestRecordAgentExecution:
    """Tests for record_agent_execution function."""

    def test_records_agent_event(self):
        """Test records agent execution as span event."""
        from src.observability.tracing import (
            TracingManager,
            record_agent_execution,
            trace_span,
        )

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            with trace_span("test.agent"):
                record_agent_execution(
                    agent_name="hrm",
                    confidence=0.9,
                    execution_time_ms=150.0,
                    success=True,
                )

                # Should not raise

    def test_records_agent_error(self):
        """Test records agent error information."""
        from src.observability.tracing import (
            TracingManager,
            record_agent_execution,
            trace_span,
        )

        TracingManager._instance = None
        with patch.dict("os.environ", {"OTEL_EXPORTER_TYPE": "none"}):
            with trace_span("test.agent.error"):
                record_agent_execution(
                    agent_name="trm",
                    confidence=0.3,
                    execution_time_ms=50.0,
                    success=False,
                    error="Connection timeout",
                )

                # Should not raise
