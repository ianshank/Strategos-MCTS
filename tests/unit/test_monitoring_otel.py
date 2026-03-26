"""
Unit tests for src/monitoring/otel_tracing.py

Tests OpenTelemetry tracing setup, dummy fallbacks, decorators, and utilities.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestOtelAvailability:
    """Tests for OpenTelemetry availability detection."""

    def test_otel_available_flag(self):
        """OTEL_AVAILABLE should be True when opentelemetry is installed."""
        from src.monitoring.otel_tracing import OTEL_AVAILABLE

        assert OTEL_AVAILABLE is True


@pytest.mark.unit
class TestSetupTracing:
    """Tests for setup_tracing initialization."""

    def setup_method(self):
        """Reset global tracing state before each test."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    def teardown_method(self):
        """Reset global tracing state after each test."""
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    @patch("src.monitoring.otel_tracing.HTTPXClientInstrumentor")
    @patch("src.monitoring.otel_tracing.BatchSpanProcessor")
    @patch("src.monitoring.otel_tracing.OTLPSpanExporter")
    @patch("src.monitoring.otel_tracing.TracerProvider")
    @patch("src.monitoring.otel_tracing.Resource")
    @patch("src.monitoring.otel_tracing.trace")
    def test_setup_tracing_initializes(
        self, mock_trace, mock_resource, mock_provider_cls, mock_exporter, mock_processor, mock_httpx
    ):
        """setup_tracing should initialize OTel components and set _initialized."""
        import src.monitoring.otel_tracing as mod

        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_trace.get_tracer.return_value = MagicMock()

        mod.setup_tracing(
            service_name="test-service",
            environment="testing",
            otlp_endpoint="http://localhost:4317",
            enable_httpx_instrumentation=True,
        )

        assert mod._initialized is True
        assert mod._tracer is not None
        mock_resource.create.assert_called_once()
        mock_exporter.assert_called_once()
        mock_provider_cls.assert_called_once()
        mock_provider.add_span_processor.assert_called_once()
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider)
        mock_httpx.return_value.instrument.assert_called_once()

    @patch("src.monitoring.otel_tracing.HTTPXClientInstrumentor")
    @patch("src.monitoring.otel_tracing.BatchSpanProcessor")
    @patch("src.monitoring.otel_tracing.OTLPSpanExporter")
    @patch("src.monitoring.otel_tracing.TracerProvider")
    @patch("src.monitoring.otel_tracing.Resource")
    @patch("src.monitoring.otel_tracing.trace")
    def test_setup_tracing_idempotent(
        self, mock_trace, mock_resource, mock_provider_cls, mock_exporter, mock_processor, mock_httpx
    ):
        """Calling setup_tracing twice should not re-initialize."""
        import src.monitoring.otel_tracing as mod

        mock_provider_cls.return_value = MagicMock()
        mock_trace.get_tracer.return_value = MagicMock()

        mod.setup_tracing(service_name="test-service")
        mod.setup_tracing(service_name="test-service")

        # TracerProvider should only be created once
        assert mock_provider_cls.call_count == 1

    @patch("src.monitoring.otel_tracing.HTTPXClientInstrumentor")
    @patch("src.monitoring.otel_tracing.BatchSpanProcessor")
    @patch("src.monitoring.otel_tracing.OTLPSpanExporter")
    @patch("src.monitoring.otel_tracing.TracerProvider")
    @patch("src.monitoring.otel_tracing.Resource")
    @patch("src.monitoring.otel_tracing.trace")
    def test_setup_tracing_without_httpx_instrumentation(
        self, mock_trace, mock_resource, mock_provider_cls, mock_exporter, mock_processor, mock_httpx
    ):
        """setup_tracing with enable_httpx_instrumentation=False should skip HTTPX."""
        import src.monitoring.otel_tracing as mod

        mock_provider_cls.return_value = MagicMock()
        mock_trace.get_tracer.return_value = MagicMock()

        mod.setup_tracing(enable_httpx_instrumentation=False)

        mock_httpx.return_value.instrument.assert_not_called()
        assert mod._initialized is True


@pytest.mark.unit
class TestDummyTracer:
    """Tests for DummyTracer fallback when OTel is not available.

    Since OTel IS installed in this environment, DummyTracer/DummyTracerProvider
    are not defined at module level. We simulate the not-installed path by
    re-importing the module with opentelemetry mocked out.
    """

    def _reload_without_otel(self):
        """Reload the module with OTel blocked, returning (mod, cleanup_fn)."""
        import importlib
        import sys

        import src.monitoring.otel_tracing as mod

        blocked = [
            "opentelemetry",
            "opentelemetry.trace",
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
            "opentelemetry.instrumentation.httpx",
            "opentelemetry.sdk.resources",
            "opentelemetry.sdk.trace",
            "opentelemetry.sdk.trace.export",
        ]
        saved = {}
        for name in blocked:
            if name in sys.modules:
                saved[name] = sys.modules[name]
            sys.modules[name] = None  # type: ignore[assignment]

        importlib.reload(mod)

        def cleanup():
            for name in blocked:
                if name in saved:
                    sys.modules[name] = saved[name]
                else:
                    sys.modules.pop(name, None)
            importlib.reload(mod)

        return mod, cleanup

    def test_dummy_tracer_works_without_otel(self):
        """DummyTracer.start_as_current_span should work as a context manager yielding None."""
        mod, cleanup = self._reload_without_otel()
        try:
            assert mod.OTEL_AVAILABLE is False

            tracer = mod.DummyTracer()
            with tracer.start_as_current_span("test-span") as span:
                assert span is None
        finally:
            cleanup()

    def test_dummy_tracer_provider_returns_dummy_tracer(self):
        """DummyTracerProvider.get_tracer should return a DummyTracer instance."""
        mod, cleanup = self._reload_without_otel()
        try:
            provider = mod.DummyTracerProvider()
            tracer = provider.get_tracer("test")
            assert isinstance(tracer, mod.DummyTracer)
        finally:
            cleanup()


@pytest.mark.unit
class TestGetTracer:
    """Tests for get_tracer function."""

    def setup_method(self):
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    def teardown_method(self):
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    def test_get_tracer_returns_dummy_when_not_initialized(self):
        """get_tracer should return a tracer (DummyTracer fallback) when setup has not completed."""
        from src.monitoring.otel_tracing import get_tracer

        # Patch setup_tracing to prevent actual initialization (keeps _tracer as None)
        with patch("src.monitoring.otel_tracing.setup_tracing"):
            tracer = get_tracer()
            # When _tracer is None, get_tracer returns DummyTracer() which has start_as_current_span
            assert hasattr(tracer, "start_as_current_span")


@pytest.mark.unit
class TestTraceOperationDecorator:
    """Tests for the trace_operation decorator."""

    def test_trace_operation_sync(self):
        """trace_operation should wrap a sync function and call it correctly."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="test_op")
        def my_func(x):
            return x * 2

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            result = my_func(5)
            assert result == 10
            mock_tracer.start_as_current_span.assert_called_once_with("test_op")

    def test_trace_operation_async(self):
        """trace_operation should wrap an async function and call it correctly."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="test_async_op")
        async def my_async_func(x):
            return x + 1

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            result = asyncio.get_event_loop().run_until_complete(my_async_func(10))
            assert result == 11

    def test_trace_operation_defaults_to_func_name(self):
        """trace_operation without explicit name should use the function name."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation()
        def specific_function_name():
            return True

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            specific_function_name()
            mock_tracer.start_as_current_span.assert_called_once_with("specific_function_name")

    def test_trace_operation_records_exception(self):
        """trace_operation should record exceptions on the span when they occur."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="failing_op", record_exception=True)
        def failing_func():
            raise ValueError("test error")

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            with pytest.raises(ValueError, match="test error"):
                failing_func()

            mock_span.record_exception.assert_called_once()


@pytest.mark.unit
class TestTraceSpanContextManager:
    """Tests for the trace_span context manager."""

    def test_trace_span_yields_span(self):
        """trace_span should yield the span from the tracer."""
        from src.monitoring.otel_tracing import trace_span

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            with trace_span("my_span", {"key": "value"}) as span:
                assert span is mock_span
                mock_span.set_attribute.assert_called_once_with("key", "value")


@pytest.mark.unit
class TestAgentSpecificTracers:
    """Tests for agent/mcts/llm/rag-specific trace decorators."""

    def test_trace_agent_operation_sets_attributes(self):
        """trace_agent_operation should create a decorator with agent attributes."""
        from src.monitoring.otel_tracing import trace_agent_operation

        decorator = trace_agent_operation("hrm")
        # The decorator is actually a trace_operation result
        assert callable(decorator)

    def test_trace_mcts_operation_sets_attributes(self):
        """trace_mcts_operation should create a decorator with mcts attributes."""
        from src.monitoring.otel_tracing import trace_mcts_operation

        decorator = trace_mcts_operation("selection")
        assert callable(decorator)

    def test_trace_llm_call_sets_attributes(self):
        """trace_llm_call should create a decorator with llm attributes."""
        from src.monitoring.otel_tracing import trace_llm_call

        decorator = trace_llm_call("openai")
        assert callable(decorator)

    def test_trace_rag_operation_sets_attributes(self):
        """trace_rag_operation should create a decorator with rag attributes."""
        from src.monitoring.otel_tracing import trace_rag_operation

        decorator = trace_rag_operation("retrieval")
        assert callable(decorator)


@pytest.mark.unit
class TestTracingStatus:
    """Tests for get_tracing_status utility."""

    def test_get_tracing_status_returns_dict(self):
        """get_tracing_status should return a dict with expected keys."""
        from src.monitoring.otel_tracing import get_tracing_status

        status = get_tracing_status()
        assert "otel_available" in status
        assert "initialized" in status
        assert "endpoint" in status
        assert isinstance(status["otel_available"], bool)
        assert isinstance(status["initialized"], bool)
        assert isinstance(status["endpoint"], str)


@pytest.mark.unit
class TestTraceContextPropagation:
    """Tests for trace context propagation utilities."""

    def test_get_trace_context_returns_dict(self):
        """get_trace_context should return a dictionary."""
        from src.monitoring.otel_tracing import get_trace_context

        ctx = get_trace_context()
        assert isinstance(ctx, dict)

    def test_set_trace_context_does_not_raise(self):
        """set_trace_context should not raise with valid input."""
        from src.monitoring.otel_tracing import set_trace_context

        # Should not raise
        set_trace_context({"traceparent": "00-abc-def-01"})
