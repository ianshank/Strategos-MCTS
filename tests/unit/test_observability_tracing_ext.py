"""
Extended unit tests for src/observability/tracing.py targeting uncovered lines.

Covers:
- TracingManager.initialize with OTLP, console, none exporters and env vars
- TracingManager.shutdown with active provider
- TracingManager.get_tracer auto-initialization
- _instrument_httpx idempotency and exception handling
- trace_span and async_trace_span with custom SpanKind and attributes
- trace_operation decorator for sync/async with default names
- SpanContextPropagator inject/extract/get_trace_parent
- record_mcts_iteration with all attributes
- record_agent_execution with and without error
- add_mcts_attributes with framework prefix

Uses @patch on the actual tracing module instead of sys.modules mocking.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_tracing_singleton():
    """Reset TracingManager singleton between tests."""
    from src.observability.tracing import TracingManager

    TracingManager._instance = None
    TracingManager._provider = None
    yield
    TracingManager._instance = None
    TracingManager._provider = None


@pytest.mark.unit
class TestTracingManagerExtended:
    """Extended tests for TracingManager."""

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    @patch("src.observability.tracing.trace")
    def test_initialize_otlp_with_custom_service_name(self, mock_trace, mock_batch, mock_otlp):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr.initialize(
            service_name="my-service",
            otlp_endpoint="collector:4317",
            exporter_type="otlp",
        )
        assert mgr._initialized is True
        mock_otlp.assert_called_once()
        # Verify endpoint was passed
        call_kwargs = mock_otlp.call_args
        assert "collector:4317" in str(call_kwargs)

    @patch("src.observability.tracing.ConsoleSpanExporter")
    @patch("src.observability.tracing.SimpleSpanProcessor")
    @patch("src.observability.tracing.trace")
    def test_initialize_console_sets_provider(self, mock_trace, mock_simple, mock_console):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr.initialize(exporter_type="console")
        assert mgr._provider is not None
        mock_trace.set_tracer_provider.assert_called_once_with(mgr._provider)

    @patch("src.observability.tracing.trace")
    def test_initialize_none_exporter_no_processor(self, mock_trace):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr.initialize(exporter_type="none")
        assert mgr._initialized is True
        assert mgr._provider is not None

    def test_initialize_unknown_exporter_raises_valueerror(self):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        with pytest.raises(ValueError, match="Unknown exporter type"):
            mgr.initialize(exporter_type="invalid_type")

    @patch("src.observability.tracing.trace")
    def test_initialize_with_additional_resources(self, mock_trace):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr.initialize(
            exporter_type="none",
            additional_resources={"team": "platform", "region": "us-east-1"},
        )
        assert mgr._initialized is True

    def test_shutdown_calls_provider_shutdown(self):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mock_provider = MagicMock()
        mgr._provider = mock_provider
        mgr._initialized = True
        mgr.shutdown()
        assert mgr._initialized is False
        mock_provider.shutdown.assert_called_once()

    def test_shutdown_without_provider(self):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr.shutdown()  # Should not raise
        assert mgr._initialized is False

    @patch("src.observability.tracing.HTTPXClientInstrumentor")
    @patch("src.observability.tracing.trace")
    def test_instrument_httpx_exception_handled(self, mock_trace, mock_httpx_inst):
        from src.observability.tracing import TracingManager

        mock_httpx_inst.return_value.instrument.side_effect = RuntimeError("instrument fail")
        mgr = TracingManager()
        mgr._instrument_httpx()
        # Should not raise, but httpx_instrumented stays False due to exception
        assert mgr._httpx_instrumented is False

    @patch("src.observability.tracing.HTTPXClientInstrumentor")
    @patch("src.observability.tracing.trace")
    def test_instrument_httpx_success(self, mock_trace, mock_httpx_inst):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr._instrument_httpx()
        assert mgr._httpx_instrumented is True
        # Second call is no-op
        mgr._instrument_httpx()
        assert mock_httpx_inst.return_value.instrument.call_count == 1

    @patch.dict(
        "os.environ",
        {
            "OTEL_SERVICE_NAME": "env-service",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "env-endpoint:4317",
            "OTEL_EXPORTER_TYPE": "none",
            "SERVICE_VERSION": "2.0.0",
            "ENVIRONMENT": "staging",
        },
    )
    @patch("src.observability.tracing.trace")
    def test_initialize_reads_all_env_vars(self, mock_trace):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr.initialize()
        assert mgr._initialized is True

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    @patch("src.observability.tracing.trace")
    def test_get_tracer_auto_initializes(self, mock_trace, mock_batch, mock_otlp):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        assert mgr._initialized is False
        mock_trace.get_tracer.return_value = MagicMock()
        tracer = mgr.get_tracer("test-tracer")
        assert mgr._initialized is True
        assert tracer is not None

    @patch("src.observability.tracing.trace")
    def test_get_tracer_when_already_initialized(self, mock_trace):
        from src.observability.tracing import TracingManager

        mgr = TracingManager()
        mgr.initialize(exporter_type="none")
        mock_trace.get_tracer.return_value = MagicMock()
        tracer = mgr.get_tracer("my-tracer")
        mock_trace.get_tracer.assert_called_with("my-tracer")
        assert tracer is not None


@pytest.mark.unit
class TestModuleLevelGetTracerExtended:
    """Extended tests for module-level get_tracer."""

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    @patch("src.observability.tracing.trace")
    def test_get_tracer_default_name(self, mock_trace, mock_batch, mock_otlp):
        from src.observability.tracing import get_tracer

        mock_trace.get_tracer.return_value = MagicMock()
        tracer = get_tracer()
        mock_trace.get_tracer.assert_called_with("mcts-framework")
        assert tracer is not None

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    @patch("src.observability.tracing.trace")
    def test_get_tracer_custom_name(self, mock_trace, mock_batch, mock_otlp):
        from src.observability.tracing import get_tracer

        mock_trace.get_tracer.return_value = MagicMock()
        tracer = get_tracer("custom-tracer")
        mock_trace.get_tracer.assert_called_with("custom-tracer")
        assert tracer is not None


@pytest.mark.unit
class TestAddMctsAttributesExtended:
    """Extended tests for add_mcts_attributes."""

    def test_multiple_mixed_prefixes(self):
        from src.observability.tracing import add_mcts_attributes

        span = MagicMock()
        add_mcts_attributes(
            span,
            **{
                "mcts.iteration": 10,
                "agent.name": "hrm",
                "framework.version": "1.0",
                "custom_key": "value",
            },
        )
        span.set_attribute.assert_any_call("mcts.iteration", 10)
        span.set_attribute.assert_any_call("agent.name", "hrm")
        span.set_attribute.assert_any_call("framework.version", "1.0")
        span.set_attribute.assert_any_call("custom.custom_key", "value")
        assert span.set_attribute.call_count == 4

    def test_all_none_values_skipped(self):
        from src.observability.tracing import add_mcts_attributes

        span = MagicMock()
        add_mcts_attributes(span, **{"mcts.a": None, "agent.b": None, "c": None})
        span.set_attribute.assert_not_called()

    def test_empty_attributes(self):
        from src.observability.tracing import add_mcts_attributes

        span = MagicMock()
        add_mcts_attributes(span)
        span.set_attribute.assert_not_called()


@pytest.mark.unit
class TestTraceSpanExtended:
    """Extended tests for trace_span context manager."""

    def test_trace_span_sets_correlation_id(self):
        from src.observability.tracing import trace_span

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with patch("src.observability.tracing.get_correlation_id", return_value="corr-123"):
                with trace_span("test.op") as span:
                    pass
        span.set_attribute.assert_any_call("correlation_id", "corr-123")

    def test_trace_span_passes_kind_and_attributes(self):
        from src.observability.tracing import SpanKind, trace_span

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with trace_span(
                "test.op",
                kind=SpanKind.CLIENT,
                attributes={"key": "val"},
                record_exception=False,
                set_status_on_exception=False,
            ):
                pass
        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["kind"] == SpanKind.CLIENT
        assert call_kwargs[1]["attributes"] == {"key": "val"}


@pytest.mark.unit
class TestAsyncTraceSpanExtended:
    """Extended tests for async_trace_span."""

    @pytest.mark.asyncio
    async def test_async_trace_span_sets_correlation_id(self):
        from src.observability.tracing import async_trace_span

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with patch("src.observability.tracing.get_correlation_id", return_value="async-corr"):
                async with async_trace_span("async.op") as span:
                    pass
        span.set_attribute.assert_any_call("correlation_id", "async-corr")

    @pytest.mark.asyncio
    async def test_async_trace_span_with_attributes(self):
        from src.observability.tracing import async_trace_span

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            async with async_trace_span("op", attributes={"a": 1}) as span:
                span.set_attribute("b", 2)

        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["attributes"] == {"a": 1}


@pytest.mark.unit
class TestTraceOperationExtended:
    """Extended tests for trace_operation decorator."""

    def test_sync_decorator_preserves_function_name(self):
        from src.observability.tracing import trace_operation

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation()
            def my_function():
                """My docstring."""
                return 42

            assert my_function.__name__ == "my_function"
            assert "My docstring" in (my_function.__doc__ or "")
            result = my_function()
            assert result == 42

    def test_sync_decorator_with_kwargs(self):
        from src.observability.tracing import trace_operation

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="custom_name")
            def func(a, b=10):
                return a + b

            result = func(5, b=20)
            assert result == 25
            mock_span.set_attribute.assert_any_call("function.args_count", 1)
            mock_span.set_attribute.assert_any_call("function.kwargs_count", 1)

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_function_name(self):
        from src.observability.tracing import trace_operation

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="async_custom")
            async def my_async_fn(x):
                """Async docstring."""
                return x * 3

            assert my_async_fn.__name__ == "my_async_fn"
            result = await my_async_fn(7)
            assert result == 21

    @pytest.mark.asyncio
    async def test_async_decorator_with_kwargs(self):
        from src.observability.tracing import trace_operation

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation()
            async def compute(a, b, c=0):
                return a + b + c

            result = await compute(1, 2, c=3)
            assert result == 6
            mock_span.set_attribute.assert_any_call("function.args_count", 2)
            mock_span.set_attribute.assert_any_call("function.kwargs_count", 1)

    def test_sync_decorator_sets_ok_status(self):
        from src.observability.tracing import trace_operation

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation()
            def ok_func():
                return "ok"

            ok_func()
            mock_span.set_status.assert_called_once()


@pytest.mark.unit
class TestSpanContextPropagatorExtended:
    """Extended tests for SpanContextPropagator."""

    def test_inject_with_context(self):
        from src.observability.tracing import SpanContextPropagator

        with patch("src.observability.tracing.inject") as mock_inject:
            prop = SpanContextPropagator()
            ctx = MagicMock()
            carrier = {}
            prop.inject(carrier, context=ctx)
            mock_inject.assert_called_once_with(carrier, context=ctx)

    def test_extract_returns_context(self):
        from src.observability.tracing import SpanContextPropagator

        mock_ctx = MagicMock()
        with patch("src.observability.tracing.extract", return_value=mock_ctx):
            prop = SpanContextPropagator()
            result = prop.extract({"traceparent": "00-trace-span-01"})
            assert result is mock_ctx


@pytest.mark.unit
class TestRecordMctsIterationExtended:
    """Extended tests for record_mcts_iteration."""

    def test_all_attributes_recorded(self):
        from src.observability.tracing import record_mcts_iteration

        mock_span = MagicMock()
        with patch("src.observability.tracing.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            record_mcts_iteration(
                iteration=7,
                selected_node_id="node-42",
                ucb_score=2.5,
                node_visits=100,
                node_value=0.95,
                tree_depth=6,
            )
            mock_span.add_event.assert_called_once()
            attrs = mock_span.add_event.call_args[1]["attributes"]
            assert attrs["mcts.iteration"] == 7
            assert attrs["mcts.selected_node_id"] == "node-42"
            assert attrs["mcts.ucb_score"] == 2.5
            assert attrs["mcts.node_visits"] == 100
            assert attrs["mcts.node_value"] == 0.95
            assert attrs["mcts.tree_depth"] == 6


@pytest.mark.unit
class TestRecordAgentExecutionExtended:
    """Extended tests for record_agent_execution."""

    def test_successful_execution_without_error_field(self):
        from src.observability.tracing import record_agent_execution

        mock_span = MagicMock()
        with patch("src.observability.tracing.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            record_agent_execution(
                agent_name="meta_controller",
                confidence=0.99,
                execution_time_ms=200.0,
                success=True,
            )
            attrs = mock_span.add_event.call_args[1]["attributes"]
            assert "agent.error" not in attrs
            assert attrs["agent.name"] == "meta_controller"
            assert attrs["agent.confidence"] == 0.99
            assert attrs["agent.execution_time_ms"] == 200.0
            assert attrs["agent.success"] is True

    def test_failed_execution_with_error(self):
        from src.observability.tracing import record_agent_execution

        mock_span = MagicMock()
        with patch("src.observability.tracing.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            record_agent_execution(
                agent_name="hrm",
                confidence=0.1,
                execution_time_ms=5000.0,
                success=False,
                error="LLM timeout after 5s",
            )
            attrs = mock_span.add_event.call_args[1]["attributes"]
            assert attrs["agent.error"] == "LLM timeout after 5s"
            assert attrs["agent.success"] is False

    def test_no_span_does_not_raise(self):
        from src.observability.tracing import record_agent_execution

        with patch("src.observability.tracing.trace") as mock_trace:
            mock_trace.get_current_span.return_value = None
            # Should not raise
            record_agent_execution(
                agent_name="trm",
                confidence=0.5,
                execution_time_ms=100.0,
                success=True,
            )
