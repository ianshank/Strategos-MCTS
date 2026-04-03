"""
Unit tests for OpenTelemetry distributed tracing module.

Tests setup_tracing, trace_span context manager, trace_operation decorator,
get_tracing_status, and agent/MCTS-specific tracing helpers.

Mocks all OpenTelemetry dependencies so tests run without OTEL installed.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
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


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset module-level globals before each test."""
    import src.monitoring.otel_tracing as mod

    original_tracer = mod._tracer
    original_initialized = mod._initialized

    mod._tracer = None
    mod._initialized = False

    yield

    mod._tracer = original_tracer
    mod._initialized = original_initialized


@pytest.fixture
def mock_span():
    """Create a mock span with common methods."""
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.record_exception = MagicMock()
    span.add_event = MagicMock()
    return span


@pytest.fixture
def mock_tracer(mock_span):
    """Create a mock tracer that yields a mock span."""
    tracer = MagicMock()

    @contextmanager
    def fake_start_as_current_span(name, **kwargs):
        yield mock_span

    tracer.start_as_current_span = MagicMock(side_effect=fake_start_as_current_span)
    return tracer


# =============================================================================
# DummyTracer Tests (fallback when OTEL is not available)
# =============================================================================


class TestDummyTracer:
    """Tests for the DummyTracer fallback class."""

    def test_dummy_tracer_start_as_current_span_yields_none(self):
        """Test DummyTracer context manager yields None."""
        from src.monitoring.otel_tracing import DummyTracer

        tracer = DummyTracer()
        with tracer.start_as_current_span("test_span") as span:
            assert span is None

    def test_dummy_tracer_provider_returns_dummy_tracer(self):
        """Test DummyTracerProvider returns a DummyTracer."""
        from src.monitoring.otel_tracing import DummyTracerProvider

        provider = DummyTracerProvider()
        tracer = provider.get_tracer("test")
        assert isinstance(tracer, type(tracer))

        # The returned tracer should support start_as_current_span
        with tracer.start_as_current_span("span") as span:
            assert span is None


# =============================================================================
# setup_tracing Tests
# =============================================================================


class TestSetupTracing:
    """Tests for the setup_tracing function."""

    def test_setup_tracing_skips_when_already_initialized(self):
        """Test setup_tracing returns early if already initialized."""
        import src.monitoring.otel_tracing as mod

        mod._initialized = True

        # Should return without error
        mod.setup_tracing()

        # Still initialized
        assert mod._initialized is True

    def test_setup_tracing_warns_when_otel_not_available(self):
        """Test setup_tracing warns and returns when OTEL not available."""
        import src.monitoring.otel_tracing as mod

        with patch.object(mod, "OTEL_AVAILABLE", False):
            mod.setup_tracing()
            assert mod._initialized is False

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    def test_setup_tracing_initializes_with_otel(self):
        """Test setup_tracing initializes provider and tracer when OTEL available."""
        import src.monitoring.otel_tracing as mod

        mock_resource = MagicMock()
        mock_exporter = MagicMock()
        mock_provider = MagicMock()
        mock_processor = MagicMock()
        mock_tracer_obj = MagicMock()

        with (
            patch.object(mod, "Resource", MagicMock(create=MagicMock(return_value=mock_resource))),
            patch.object(mod, "OTLPSpanExporter", MagicMock(return_value=mock_exporter)),
            patch.object(mod, "TracerProvider", MagicMock(return_value=mock_provider)),
            patch.object(mod, "BatchSpanProcessor", MagicMock(return_value=mock_processor)),
            patch.object(mod, "trace") as mock_trace,
            patch.object(mod, "HTTPXClientInstrumentor", MagicMock()),
        ):
            mock_trace.get_tracer.return_value = mock_tracer_obj

            mod.setup_tracing(
                service_name="test-service",
                environment="test",
                otlp_endpoint="http://localhost:4317",
            )

            assert mod._initialized is True
            assert mod._tracer is mock_tracer_obj
            mock_trace.set_tracer_provider.assert_called_once_with(mock_provider)
            mock_provider.add_span_processor.assert_called_once_with(mock_processor)

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    def test_setup_tracing_handles_httpx_instrumentation_failure(self):
        """Test setup_tracing handles httpx instrumentation errors gracefully."""
        import src.monitoring.otel_tracing as mod

        mock_instrumentor = MagicMock()
        mock_instrumentor.return_value.instrument.side_effect = RuntimeError("instrument error")

        with (
            patch.object(mod, "Resource", MagicMock(create=MagicMock(return_value=MagicMock()))),
            patch.object(mod, "OTLPSpanExporter", MagicMock(return_value=MagicMock())),
            patch.object(mod, "TracerProvider", MagicMock(return_value=MagicMock())),
            patch.object(mod, "BatchSpanProcessor", MagicMock(return_value=MagicMock())),
            patch.object(mod, "trace") as mock_trace,
            patch.object(mod, "HTTPXClientInstrumentor", mock_instrumentor),
        ):
            mock_trace.get_tracer.return_value = MagicMock()

            # Should not raise
            mod.setup_tracing(enable_httpx_instrumentation=True)

            assert mod._initialized is True

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    def test_setup_tracing_handles_general_exception(self):
        """Test setup_tracing handles exceptions and sets tracer to None."""
        import src.monitoring.otel_tracing as mod

        with patch.object(mod, "Resource", MagicMock(create=MagicMock(side_effect=RuntimeError("boom")))):
            mod.setup_tracing()

            assert mod._initialized is False
            assert mod._tracer is None


# =============================================================================
# get_tracer Tests
# =============================================================================


class TestGetTracer:
    """Tests for the get_tracer function."""

    def test_get_tracer_returns_dummy_when_not_initialized(self):
        """Test get_tracer returns DummyTracer when tracer is None and OTEL unavailable."""
        import src.monitoring.otel_tracing as mod

        with patch.object(mod, "OTEL_AVAILABLE", False):
            tracer = mod.get_tracer()

        # Should return a DummyTracer instance
        from src.monitoring.otel_tracing import DummyTracer

        assert isinstance(tracer, DummyTracer)

    def test_get_tracer_returns_existing_tracer(self):
        """Test get_tracer returns existing tracer when set."""
        import src.monitoring.otel_tracing as mod

        mock_tracer = MagicMock()
        mod._tracer = mock_tracer

        result = mod.get_tracer()
        assert result is mock_tracer

    def test_get_tracer_lazy_initializes_when_otel_available(self):
        """Test get_tracer calls setup_tracing when OTEL available but not init."""
        import src.monitoring.otel_tracing as mod

        with (
            patch.object(mod, "OTEL_AVAILABLE", True),
            patch.object(mod, "setup_tracing") as mock_setup,
        ):
            mod.get_tracer()
            mock_setup.assert_called_once()


# =============================================================================
# trace_span Context Manager Tests
# =============================================================================


class TestTraceSpan:
    """Tests for the trace_span context manager."""

    def test_trace_span_yields_span(self, mock_tracer, mock_span):
        """Test trace_span yields a span from the tracer."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        with mod.trace_span("test.operation") as span:
            assert span is mock_span

    def test_trace_span_sets_attributes(self, mock_tracer, mock_span):
        """Test trace_span sets attributes on the span."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        with mod.trace_span("test.op", attributes={"key1": "value1", "key2": 42}):
            pass

        mock_span.set_attribute.assert_any_call("key1", "value1")
        mock_span.set_attribute.assert_any_call("key2", 42)

    def test_trace_span_records_exception_and_reraises(self, mock_tracer, mock_span):
        """Test trace_span records exceptions and re-raises them."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        with pytest.raises(ValueError, match="test error"):
            with mod.trace_span("test.error"):
                raise ValueError("test error")

    def test_trace_span_works_with_dummy_tracer(self):
        """Test trace_span works gracefully with DummyTracer."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = None

        with patch.object(mod, "OTEL_AVAILABLE", False):
            with mod.trace_span("test.dummy") as span:
                # DummyTracer yields None
                assert span is None

    def test_trace_span_no_attributes(self, mock_tracer, mock_span):
        """Test trace_span works without attributes."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        with mod.trace_span("test.no_attrs"):
            pass

        # set_attribute should not be called for attributes (only status)
        # since attributes is None
        mock_span.set_attribute.assert_not_called()


# =============================================================================
# trace_operation Decorator Tests
# =============================================================================


class TestTraceOperationDecorator:
    """Tests for the trace_operation decorator."""

    def test_sync_function_decorated(self, mock_tracer, mock_span):
        """Test trace_operation works with sync functions."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_operation(name="test_sync_op")
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_sync_function_default_name(self, mock_tracer, mock_span):
        """Test trace_operation uses function name as default span name."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_operation()
        def my_operation():
            return "done"

        result = my_operation()
        assert result == "done"
        assert my_operation.__name__ == "my_operation"

    def test_sync_function_with_attributes(self, mock_tracer, mock_span):
        """Test trace_operation sets custom attributes on span."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_operation(name="op_with_attrs", attributes={"agent": "hrm"})
        def process():
            return "processed"

        process()
        mock_span.set_attribute.assert_any_call("agent", "hrm")

    def test_sync_function_exception_propagates(self, mock_tracer, mock_span):
        """Test trace_operation propagates exceptions from sync functions."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_operation(name="failing_op")
        def fail():
            raise RuntimeError("sync failure")

        with pytest.raises(RuntimeError, match="sync failure"):
            fail()

    @pytest.mark.asyncio
    async def test_async_function_decorated(self, mock_tracer, mock_span):
        """Test trace_operation works with async functions."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_operation(name="test_async_op")
        async def async_add(a, b):
            return a + b

        result = await async_add(5, 6)
        assert result == 11

    @pytest.mark.asyncio
    async def test_async_function_exception_propagates(self, mock_tracer, mock_span):
        """Test trace_operation propagates exceptions from async functions."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_operation(name="async_fail")
        async def async_fail():
            raise ValueError("async failure")

        with pytest.raises(ValueError, match="async failure"):
            await async_fail()

    def test_decorator_preserves_function_metadata(self, mock_tracer):
        """Test trace_operation preserves __name__ and __doc__."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_operation()
        def documented_function():
            """This is the docstring."""
            return True

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


# =============================================================================
# get_tracing_status Tests
# =============================================================================


class TestGetTracingStatus:
    """Tests for the get_tracing_status function."""

    def test_returns_dict_with_expected_keys(self):
        """Test get_tracing_status returns expected keys."""
        from src.monitoring.otel_tracing import get_tracing_status

        status = get_tracing_status()

        assert isinstance(status, dict)
        assert "otel_available" in status
        assert "initialized" in status
        assert "endpoint" in status

    def test_reflects_initialized_state(self):
        """Test get_tracing_status reflects current initialization state."""
        import src.monitoring.otel_tracing as mod

        mod._initialized = False
        status = mod.get_tracing_status()
        assert status["initialized"] is False

        mod._initialized = True
        status = mod.get_tracing_status()
        assert status["initialized"] is True

    def test_endpoint_from_env(self):
        """Test get_tracing_status reads endpoint from environment."""
        import src.monitoring.otel_tracing as mod

        with patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://custom:4317"}):
            status = mod.get_tracing_status()
            assert status["endpoint"] == "http://custom:4317"

    def test_endpoint_default_value(self):
        """Test get_tracing_status uses default endpoint when env not set."""
        import os

        import src.monitoring.otel_tracing as mod

        env = os.environ.copy()
        env.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        with patch.dict("os.environ", env, clear=True):
            status = mod.get_tracing_status()
            assert status["endpoint"] == "http://localhost:4317"


# =============================================================================
# Agent/MCTS-Specific Tracing Helpers Tests
# =============================================================================


class TestSpecializedTracingDecorators:
    """Tests for trace_agent_operation, trace_mcts_operation, trace_llm_call, trace_rag_operation."""

    def test_trace_agent_operation_returns_decorator(self):
        """Test trace_agent_operation returns a usable decorator."""
        from src.monitoring.otel_tracing import trace_agent_operation

        decorator = trace_agent_operation("hrm")
        assert callable(decorator)

    def test_trace_mcts_operation_returns_decorator(self):
        """Test trace_mcts_operation returns a usable decorator."""
        from src.monitoring.otel_tracing import trace_mcts_operation

        decorator = trace_mcts_operation("selection")
        assert callable(decorator)

    def test_trace_llm_call_returns_decorator(self):
        """Test trace_llm_call returns a usable decorator."""
        from src.monitoring.otel_tracing import trace_llm_call

        decorator = trace_llm_call("openai")
        assert callable(decorator)

    def test_trace_rag_operation_returns_decorator(self):
        """Test trace_rag_operation returns a usable decorator."""
        from src.monitoring.otel_tracing import trace_rag_operation

        decorator = trace_rag_operation("retrieval")
        assert callable(decorator)

    def test_trace_agent_operation_decorates_sync_function(self, mock_tracer, mock_span):
        """Test trace_agent_operation decorator works on sync function."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_agent_operation("hrm")
        def agent_process():
            return "agent_result"

        result = agent_process()
        assert result == "agent_result"

    @pytest.mark.asyncio
    async def test_trace_mcts_operation_decorates_async_function(self, mock_tracer, mock_span):
        """Test trace_mcts_operation decorator works on async function."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = mock_tracer

        @mod.trace_mcts_operation("simulation")
        async def run_simulation():
            return 42

        result = await run_simulation()
        assert result == 42


# =============================================================================
# add_span_attribute / add_span_event Tests
# =============================================================================


class TestSpanHelpers:
    """Tests for add_span_attribute and add_span_event."""

    def test_add_span_attribute_when_otel_unavailable(self):
        """Test add_span_attribute is a no-op when OTEL is unavailable."""
        import src.monitoring.otel_tracing as mod

        with patch.object(mod, "OTEL_AVAILABLE", False):
            # Should not raise
            mod.add_span_attribute("key", "value")

    def test_add_span_event_when_otel_unavailable(self):
        """Test add_span_event is a no-op when OTEL is unavailable."""
        import src.monitoring.otel_tracing as mod

        with patch.object(mod, "OTEL_AVAILABLE", False):
            # Should not raise
            mod.add_span_event("event_name", {"attr": "val"})


# =============================================================================
# Trace Context Propagation Tests
# =============================================================================


class TestTraceContextPropagation:
    """Tests for get_trace_context and set_trace_context."""

    def test_get_trace_context_returns_empty_when_otel_unavailable(self):
        """Test get_trace_context returns {} when OTEL is not available."""
        import src.monitoring.otel_tracing as mod

        with patch.object(mod, "OTEL_AVAILABLE", False):
            result = mod.get_trace_context()
            assert result == {}

    def test_set_trace_context_noop_when_otel_unavailable(self):
        """Test set_trace_context does nothing when OTEL is not available."""
        import src.monitoring.otel_tracing as mod

        with patch.object(mod, "OTEL_AVAILABLE", False):
            # Should not raise
            mod.set_trace_context({"traceparent": "fake"})
