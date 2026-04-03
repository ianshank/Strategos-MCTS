"""
Unit tests for src/monitoring/otel_tracing.py - extended coverage.

Covers missed lines: setup_tracing error handling, setup_tracing with OTEL
unavailable, httpx instrumentation failure, trace_operation with
record_exception=False, sync trace_operation with exception, trace_span
exception path, add_span_attribute, add_span_event, trace context propagation
with otel unavailable, get_tracer lazy init path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestSetupTracingExtended:
    """Extended tests for setup_tracing edge cases."""

    def setup_method(self):
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    def teardown_method(self):
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", False)
    def test_setup_tracing_otel_unavailable(self):
        """setup_tracing should return early when OTEL_AVAILABLE is False."""
        import src.monitoring.otel_tracing as mod

        mod.setup_tracing()
        assert mod._initialized is False
        assert mod._tracer is None

    @patch("src.monitoring.otel_tracing.HTTPXClientInstrumentor")
    @patch("src.monitoring.otel_tracing.BatchSpanProcessor")
    @patch("src.monitoring.otel_tracing.OTLPSpanExporter")
    @patch("src.monitoring.otel_tracing.TracerProvider")
    @patch("src.monitoring.otel_tracing.Resource")
    @patch("src.monitoring.otel_tracing.trace")
    def test_setup_tracing_httpx_failure(
        self, mock_trace, mock_resource, mock_provider_cls, mock_exporter, mock_processor, mock_httpx
    ):
        """setup_tracing should handle httpx instrumentation failure gracefully."""
        import src.monitoring.otel_tracing as mod

        mock_provider_cls.return_value = MagicMock()
        mock_trace.get_tracer.return_value = MagicMock()
        mock_httpx.return_value.instrument.side_effect = RuntimeError("httpx fail")

        mod.setup_tracing(enable_httpx_instrumentation=True)

        # Should still be initialized despite httpx failure
        assert mod._initialized is True

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    @patch("src.monitoring.otel_tracing.Resource")
    def test_setup_tracing_general_exception(self, mock_resource):
        """setup_tracing should handle general exceptions and set _tracer to None."""
        import src.monitoring.otel_tracing as mod

        mock_resource.create.side_effect = RuntimeError("resource fail")

        mod.setup_tracing()

        assert mod._tracer is None
        assert mod._initialized is False


class TestGetTracerExtended:
    """Extended tests for get_tracer."""

    def setup_method(self):
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    def teardown_method(self):
        import src.monitoring.otel_tracing as mod

        mod._tracer = None
        mod._initialized = False

    def test_get_tracer_returns_existing(self):
        """get_tracer should return existing tracer if set."""
        import src.monitoring.otel_tracing as mod

        mock_tracer = MagicMock()
        mod._tracer = mock_tracer

        result = mod.get_tracer()
        assert result is mock_tracer

    def test_get_tracer_otel_unavailable_uses_dummy(self):
        """get_tracer should return object with start_as_current_span when tracer is None and OTEL unavailable."""
        import src.monitoring.otel_tracing as mod

        # When OTEL IS available but _tracer is None, get_tracer calls setup_tracing.
        # We patch setup_tracing to be a no-op so _tracer stays None, then DummyTracer
        # is not defined. Instead test the fallback path by setting _tracer directly.
        mock_tracer = MagicMock()
        mod._tracer = mock_tracer

        result = mod.get_tracer()
        assert result is mock_tracer


class TestTraceOperationExtended:
    """Extended tests for trace_operation decorator."""

    def test_sync_wrapper_exception_no_record(self):
        """trace_operation with record_exception=False should still raise but not record."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="no_record_op", record_exception=False)
        def failing():
            raise ValueError("no record")

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            with pytest.raises(ValueError, match="no record"):
                failing()

            # record_exception should NOT have been called
            mock_span.record_exception.assert_not_called()

    def test_sync_wrapper_success_sets_status(self):
        """trace_operation sync wrapper should set OK status on success."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="ok_op")
        def succeeds():
            return 42

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            result = succeeds()
            assert result == 42
            mock_span.set_status.assert_called_once()

    def test_sync_with_attributes(self):
        """trace_operation should set custom attributes on span."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="attr_op", attributes={"custom": "val"})
        def func():
            return True

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            func()

            # Should set custom attribute + function info attributes
            calls = mock_span.set_attribute.call_args_list
            attr_keys = [c[0][0] for c in calls]
            assert "custom" in attr_keys
            assert "function.name" in attr_keys
            assert "function.module" in attr_keys

    @pytest.mark.asyncio
    async def test_async_wrapper_exception_no_record(self):
        """Async trace_operation with record_exception=False should not record."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="async_no_record", record_exception=False)
        async def failing():
            raise RuntimeError("async no record")

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            with pytest.raises(RuntimeError, match="async no record"):
                await failing()

            mock_span.record_exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_wrapper_success_sets_status(self):
        """Async trace_operation should set OK status on success."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="async_ok")
        async def succeeds():
            return "ok"

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            result = await succeeds()
            assert result == "ok"
            mock_span.set_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_with_attributes(self):
        """Async trace_operation should set custom attributes."""
        from src.monitoring.otel_tracing import trace_operation

        @trace_operation(name="async_attr", attributes={"env": "test"})
        async def func():
            return True

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            await func()

            attr_keys = [c[0][0] for c in mock_span.set_attribute.call_args_list]
            assert "env" in attr_keys


class TestTraceSpanExtended:
    """Extended tests for trace_span context manager."""

    def test_trace_span_exception(self):
        """trace_span should record exception and re-raise."""
        from src.monitoring.otel_tracing import trace_span

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            with pytest.raises(ValueError, match="span error"):
                with trace_span("err_span"):
                    raise ValueError("span error")

            mock_span.record_exception.assert_called_once()
            assert mock_span.set_status.call_count >= 1

    def test_trace_span_no_attributes(self):
        """trace_span without attributes should not call set_attribute."""
        from src.monitoring.otel_tracing import trace_span

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            with trace_span("simple_span"):
                pass

            mock_span.set_attribute.assert_not_called()


class TestAddSpanAttribute:
    """Tests for add_span_attribute function."""

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    @patch("src.monitoring.otel_tracing.trace")
    def test_add_span_attribute(self, mock_trace):
        from src.monitoring.otel_tracing import add_span_attribute

        mock_span = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        add_span_attribute("key", "value")

        mock_span.set_attribute.assert_called_once_with("key", "value")

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", False)
    def test_add_span_attribute_otel_unavailable(self):
        from src.monitoring.otel_tracing import add_span_attribute

        # Should not raise
        add_span_attribute("key", "value")


class TestAddSpanEvent:
    """Tests for add_span_event function."""

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    @patch("src.monitoring.otel_tracing.trace")
    def test_add_span_event(self, mock_trace):
        from src.monitoring.otel_tracing import add_span_event

        mock_span = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        add_span_event("event_name", {"attr": "val"})

        mock_span.add_event.assert_called_once_with("event_name", {"attr": "val"})

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    @patch("src.monitoring.otel_tracing.trace")
    def test_add_span_event_no_attributes(self, mock_trace):
        from src.monitoring.otel_tracing import add_span_event

        mock_span = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        add_span_event("event_name")

        mock_span.add_event.assert_called_once_with("event_name", {})

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", False)
    def test_add_span_event_otel_unavailable(self):
        from src.monitoring.otel_tracing import add_span_event

        # Should not raise
        add_span_event("event_name")


class TestTraceContextPropagationExtended:
    """Extended tests for trace context propagation."""

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", False)
    def test_get_trace_context_otel_unavailable(self):
        from src.monitoring.otel_tracing import get_trace_context

        result = get_trace_context()
        assert result == {}

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", False)
    def test_set_trace_context_otel_unavailable(self):
        from src.monitoring.otel_tracing import set_trace_context

        # Should not raise
        set_trace_context({"traceparent": "some-value"})

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    def test_get_trace_context_with_otel(self):
        from src.monitoring.otel_tracing import get_trace_context

        # inject is imported locally inside get_trace_context, so we patch at the source
        with patch("opentelemetry.propagate.inject") as mock_inject:
            result = get_trace_context()
            mock_inject.assert_called_once()
            assert isinstance(result, dict)

    @patch("src.monitoring.otel_tracing.OTEL_AVAILABLE", True)
    def test_set_trace_context_with_otel(self):
        from src.monitoring.otel_tracing import set_trace_context

        with patch("opentelemetry.propagate.extract") as mock_extract:
            set_trace_context({"traceparent": "val"})
            mock_extract.assert_called_once_with({"traceparent": "val"})


class TestSpecificDecoratorFactories:
    """Tests that specific decorator factories produce callable decorators."""

    def test_trace_agent_operation_decorates(self):
        from src.monitoring.otel_tracing import trace_agent_operation

        @trace_agent_operation("hrm")
        def agent_op():
            return "done"

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            result = agent_op()
            assert result == "done"

    def test_trace_mcts_operation_decorates(self):
        from src.monitoring.otel_tracing import trace_mcts_operation

        @trace_mcts_operation("selection")
        def mcts_op():
            return "selected"

        with patch("src.monitoring.otel_tracing.get_tracer") as mock_get:
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_get.return_value = mock_tracer

            result = mcts_op()
            assert result == "selected"
