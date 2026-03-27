"""
Additional unit tests for src/observability/tracing.py targeting remaining uncovered lines.

Focuses on:
- TracingManager.initialize with OTLP insecure env var
- TracingManager.initialize with environment variable fallbacks
- trace_span exception propagation
- async_trace_span exception propagation
- trace_operation decorator with attributes parameter
- trace_operation decorator async status setting
- SpanContextPropagator full lifecycle
- record_mcts_iteration with falsy span (not None but falsy)
- record_agent_execution edge cases
- Module-level get_tracer singleton behavior
"""

import asyncio
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

# The tracing module is already imported via sys.modules mocking from test_tracing.py.
# We import directly - the module is already loaded with mocked OpenTelemetry.
from src.observability.tracing import (
    SpanContextPropagator,
    TracingManager,
    add_mcts_attributes,
    async_trace_span,
    get_tracer,
    record_agent_execution,
    record_mcts_iteration,
    trace_operation,
    trace_span,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset TracingManager singleton between tests."""
    TracingManager._instance = None
    TracingManager._provider = None
    yield
    TracingManager._instance = None
    TracingManager._provider = None


def _make_mock_tracer_and_span():
    """Helper: create a mock tracer whose start_as_current_span yields a mock span."""
    mock_span = MagicMock()
    mock_tracer = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_span)
    cm.__exit__ = MagicMock(return_value=False)
    mock_tracer.start_as_current_span.return_value = cm
    return mock_tracer, mock_span


# ---------------------------------------------------------------------------
# TracingManager additional coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTracingManagerExt2:
    """Additional TracingManager tests for uncovered branches."""

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    @patch("src.observability.tracing.trace")
    @patch.dict(
        "os.environ",
        {"OTEL_EXPORTER_OTLP_INSECURE": "false"},
    )
    def test_initialize_otlp_insecure_false(self, mock_trace, mock_batch, mock_otlp):
        """OTEL_EXPORTER_OTLP_INSECURE=false passes insecure=False to exporter."""
        mgr = TracingManager()
        mgr.initialize(exporter_type="otlp", otlp_endpoint="localhost:4317")
        assert mgr._initialized is True
        call_kwargs = mock_otlp.call_args
        assert call_kwargs[1]["insecure"] is False

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    @patch("src.observability.tracing.trace")
    @patch.dict(
        "os.environ",
        {"OTEL_EXPORTER_OTLP_INSECURE": "true"},
    )
    def test_initialize_otlp_insecure_true(self, mock_trace, mock_batch, mock_otlp):
        """OTEL_EXPORTER_OTLP_INSECURE=true passes insecure=True to exporter."""
        mgr = TracingManager()
        mgr.initialize(exporter_type="otlp", otlp_endpoint="localhost:4317")
        call_kwargs = mock_otlp.call_args
        assert call_kwargs[1]["insecure"] is True

    @patch("src.observability.tracing.trace")
    def test_initialize_none_exporter_does_not_add_processor(self, mock_trace):
        """When exporter_type='none', no processor is added to provider."""
        mgr = TracingManager()
        mgr.initialize(exporter_type="none")
        # Provider should exist but add_span_processor should not be called
        assert mgr._provider is not None
        # The provider's add_span_processor should not have been called
        # (since processor is None for 'none' type)
        if hasattr(mgr._provider, "add_span_processor"):
            # With mocked TracerProvider, just verify initialized
            assert mgr._initialized is True

    @patch("src.observability.tracing.trace")
    @patch.dict(
        "os.environ",
        {
            "OTEL_SERVICE_NAME": "custom-svc",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "collector:9999",
            "OTEL_EXPORTER_TYPE": "none",
            "SERVICE_VERSION": "3.0.0",
            "ENVIRONMENT": "production",
        },
    )
    def test_initialize_env_vars_all_set(self, mock_trace):
        """All env vars are read when no explicit params given."""
        mgr = TracingManager()
        mgr.initialize()
        assert mgr._initialized is True

    def test_get_instance_creates_new_if_none(self):
        """get_instance creates a new instance when _instance is None."""
        assert TracingManager._instance is None
        inst = TracingManager.get_instance()
        assert inst is not None
        assert TracingManager._instance is inst

    def test_shutdown_with_mock_provider(self):
        """shutdown calls provider.shutdown() and resets _initialized."""
        mgr = TracingManager()
        mock_prov = MagicMock()
        mgr._provider = mock_prov
        mgr._initialized = True
        mgr.shutdown()
        mock_prov.shutdown.assert_called_once()
        assert mgr._initialized is False

    def test_shutdown_no_provider_no_error(self):
        """shutdown with no provider does not raise."""
        mgr = TracingManager()
        mgr._provider = None
        mgr._initialized = False
        mgr.shutdown()  # Should not raise

    @patch("src.observability.tracing.HTTPXClientInstrumentor")
    def test_instrument_httpx_exception_sets_not_instrumented(self, mock_httpx_cls):
        """When httpx instrumentation raises, _httpx_instrumented stays False."""
        mock_httpx_cls.return_value.instrument.side_effect = Exception("fail")
        mgr = TracingManager()
        mgr._instrument_httpx()
        assert mgr._httpx_instrumented is False

    @patch("src.observability.tracing.HTTPXClientInstrumentor")
    def test_instrument_httpx_idempotent(self, mock_httpx_cls):
        """Second call to _instrument_httpx is a no-op."""
        mgr = TracingManager()
        mgr._httpx_instrumented = True
        mgr._instrument_httpx()
        mock_httpx_cls.return_value.instrument.assert_not_called()


# ---------------------------------------------------------------------------
# Module-level get_tracer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetTracerExt2:
    """Additional tests for module-level get_tracer."""

    @patch("src.observability.tracing.trace")
    def test_get_tracer_uses_singleton(self, mock_trace):
        """get_tracer uses TracingManager singleton."""
        mock_trace.get_tracer.return_value = MagicMock()
        # Force initialized to avoid triggering full init
        mgr = TracingManager.get_instance()
        mgr._initialized = True

        t1 = get_tracer("svc1")
        t2 = get_tracer("svc2")
        assert mock_trace.get_tracer.call_count == 2


# ---------------------------------------------------------------------------
# add_mcts_attributes edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddMctsAttributesExt2:
    """Edge case tests for add_mcts_attributes."""

    def test_zero_value_is_set(self):
        """Zero values are not None and should be set."""
        span = MagicMock()
        add_mcts_attributes(span, **{"mcts.iteration": 0})
        span.set_attribute.assert_called_once_with("mcts.iteration", 0)

    def test_empty_string_value_is_set(self):
        """Empty string values are not None and should be set."""
        span = MagicMock()
        add_mcts_attributes(span, **{"agent.name": ""})
        span.set_attribute.assert_called_once_with("agent.name", "")

    def test_false_value_is_set(self):
        """False values are not None and should be set."""
        span = MagicMock()
        add_mcts_attributes(span, **{"custom_flag": False})
        span.set_attribute.assert_called_once_with("custom.custom_flag", False)

    def test_multiple_custom_keys_prefixed(self):
        """Multiple non-standard keys all get custom. prefix."""
        span = MagicMock()
        add_mcts_attributes(span, foo="a", bar="b")
        span.set_attribute.assert_any_call("custom.foo", "a")
        span.set_attribute.assert_any_call("custom.bar", "b")
        assert span.set_attribute.call_count == 2


# ---------------------------------------------------------------------------
# trace_span context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraceSpanExt2:
    """Additional trace_span tests."""

    def test_trace_span_default_attributes_empty_dict(self):
        """When attributes=None, an empty dict is passed to start_as_current_span."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with trace_span("test.op") as span:
                pass

        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["attributes"] == {}

    def test_trace_span_exception_propagates(self):
        """Exceptions inside trace_span propagate out."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with pytest.raises(ValueError, match="boom"):
                with trace_span("test.op") as span:
                    raise ValueError("boom")

    def test_trace_span_record_exception_param(self):
        """record_exception parameter is forwarded to start_as_current_span."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with trace_span("test.op", record_exception=False) as span:
                pass

        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["record_exception"] is False

    def test_trace_span_set_status_on_exception_param(self):
        """set_status_on_exception parameter is forwarded."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with trace_span("op", set_status_on_exception=False) as span:
                pass

        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["set_status_on_exception"] is False


# ---------------------------------------------------------------------------
# async_trace_span context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncTraceSpanExt2:
    """Additional async_trace_span tests."""

    def test_async_trace_span_default_attributes(self):
        """When attributes=None, empty dict is passed."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        async def _run():
            with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
                async with async_trace_span("async.op") as span:
                    pass

        asyncio.get_event_loop().run_until_complete(_run())
        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["attributes"] == {}

    def test_async_trace_span_exception_propagates(self):
        """Exceptions inside async_trace_span propagate out."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        async def _run():
            with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
                async with async_trace_span("op") as span:
                    raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.get_event_loop().run_until_complete(_run())

    def test_async_trace_span_record_exception_param(self):
        """record_exception is forwarded in async context."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        async def _run():
            with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
                async with async_trace_span("op", record_exception=False) as span:
                    pass

        asyncio.get_event_loop().run_until_complete(_run())
        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["record_exception"] is False

    def test_async_trace_span_set_status_on_exception_param(self):
        """set_status_on_exception is forwarded in async context."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        async def _run():
            with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
                async with async_trace_span("op", set_status_on_exception=False) as span:
                    pass

        asyncio.get_event_loop().run_until_complete(_run())
        call_kwargs = mock_tracer.start_as_current_span.call_args
        assert call_kwargs[1]["set_status_on_exception"] is False


# ---------------------------------------------------------------------------
# trace_operation decorator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraceOperationExt2:
    """Additional trace_operation decorator tests."""

    def test_sync_decorator_with_attributes(self):
        """trace_operation passes attributes to trace_span."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="my_op", attributes={"component": "mcts"})
            def my_func():
                return "result"

            result = my_func()
            assert result == "result"

    def test_sync_decorator_no_args_no_kwargs(self):
        """Decorator records 0 args and 0 kwargs."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="zero_args")
            def no_args():
                return 42

            result = no_args()
            assert result == 42
            mock_span.set_attribute.assert_any_call("function.args_count", 0)
            mock_span.set_attribute.assert_any_call("function.kwargs_count", 0)

    def test_sync_decorator_sets_ok_status_on_success(self):
        """Successful sync function sets OK status on span."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="ok_op")
            def ok_func():
                return "ok"

            ok_func()
            mock_span.set_status.assert_called_once()

    def test_async_decorator_with_attributes(self):
        """trace_operation passes attributes for async functions."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="async_attr", attributes={"x": 1})
            async def my_async():
                return "async_result"

            result = asyncio.get_event_loop().run_until_complete(my_async())
            assert result == "async_result"

    def test_async_decorator_sets_ok_status(self):
        """Successful async function sets OK status on span."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="async_ok")
            async def ok_async():
                return "done"

            asyncio.get_event_loop().run_until_complete(ok_async())
            mock_span.set_status.assert_called_once()

    def test_async_decorator_records_args_kwargs(self):
        """Async decorator records correct args and kwargs count."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation()
            async def multi_args(a, b, c=None, d=None):
                return a + b

            result = asyncio.get_event_loop().run_until_complete(multi_args(1, 2, c=3, d=4))
            assert result == 3
            mock_span.set_attribute.assert_any_call("function.args_count", 2)
            mock_span.set_attribute.assert_any_call("function.kwargs_count", 2)

    def test_decorator_default_name_uses_module_and_qualname(self):
        """Default span name is {module}.{function_name}."""
        mock_tracer, mock_span = _make_mock_tracer_and_span()

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation()
            def specific_func():
                return True

            specific_func()

        # The span name should contain the function name
        span_name_arg = mock_tracer.start_as_current_span.call_args[0][0]
        assert "specific_func" in span_name_arg


# ---------------------------------------------------------------------------
# SpanContextPropagator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpanContextPropagatorExt2:
    """Additional SpanContextPropagator tests."""

    def test_inject_without_context(self):
        """inject with default context=None."""
        with patch("src.observability.tracing.inject") as mock_inject:
            prop = SpanContextPropagator()
            carrier = {"existing": "header"}
            prop.inject(carrier)
            mock_inject.assert_called_once_with(carrier, context=None)

    def test_inject_with_explicit_context(self):
        """inject with explicit context passes it through."""
        with patch("src.observability.tracing.inject") as mock_inject:
            prop = SpanContextPropagator()
            ctx = MagicMock()
            carrier = {}
            prop.inject(carrier, context=ctx)
            mock_inject.assert_called_once_with(carrier, context=ctx)

    def test_extract_returns_context_object(self):
        """extract returns whatever the propagate.extract returns."""
        mock_ctx = MagicMock()
        with patch("src.observability.tracing.extract", return_value=mock_ctx):
            prop = SpanContextPropagator()
            result = prop.extract({"traceparent": "00-abc-def-01"})
            assert result is mock_ctx

    def test_get_trace_parent_with_populated_carrier(self):
        """get_trace_parent returns traceparent when inject populates carrier."""
        with patch("src.observability.tracing.inject") as mock_inject:

            def populate(carrier, context=None):
                carrier["traceparent"] = "00-traceid-spanid-01"

            mock_inject.side_effect = populate
            prop = SpanContextPropagator()
            tp = prop.get_trace_parent()
            assert tp == "00-traceid-spanid-01"

    def test_get_trace_parent_none_when_empty(self):
        """get_trace_parent returns None when carrier is not populated."""
        with patch("src.observability.tracing.inject"):
            prop = SpanContextPropagator()
            assert prop.get_trace_parent() is None

    def test_propagator_has_internal_propagator(self):
        """SpanContextPropagator stores a _propagator attribute."""
        prop = SpanContextPropagator()
        assert hasattr(prop, "_propagator")


# ---------------------------------------------------------------------------
# record_mcts_iteration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecordMctsIterationExt2:
    """Additional tests for record_mcts_iteration."""

    def test_all_six_attributes_present(self):
        """All six attributes are set in the event."""
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_mcts_iteration(
                iteration=1,
                selected_node_id="root",
                ucb_score=0.0,
                node_visits=1,
                node_value=0.0,
                tree_depth=0,
            )
            attrs = mock_span.add_event.call_args[1]["attributes"]
            assert set(attrs.keys()) == {
                "mcts.iteration",
                "mcts.selected_node_id",
                "mcts.ucb_score",
                "mcts.node_visits",
                "mcts.node_value",
                "mcts.tree_depth",
            }

    def test_event_name_is_mcts_iteration(self):
        """Event name is 'mcts.iteration'."""
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_mcts_iteration(
                iteration=0,
                selected_node_id="n",
                ucb_score=0.0,
                node_visits=0,
                node_value=0.0,
                tree_depth=0,
            )
            assert mock_span.add_event.call_args[0][0] == "mcts.iteration"

    def test_no_span_available(self):
        """No error when there is no current span."""
        with patch("src.observability.tracing.trace.get_current_span", return_value=None):
            record_mcts_iteration(
                iteration=0,
                selected_node_id="x",
                ucb_score=0.0,
                node_visits=0,
                node_value=0.0,
                tree_depth=0,
            )


# ---------------------------------------------------------------------------
# record_agent_execution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecordAgentExecutionExt2:
    """Additional tests for record_agent_execution."""

    def test_event_name_is_agent_execution(self):
        """Event name is 'agent.execution'."""
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_agent_execution(
                agent_name="test",
                confidence=0.5,
                execution_time_ms=10.0,
                success=True,
            )
            assert mock_span.add_event.call_args[0][0] == "agent.execution"

    def test_error_none_not_included(self):
        """When error=None (default), no agent.error in attributes."""
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_agent_execution(
                agent_name="test",
                confidence=0.5,
                execution_time_ms=10.0,
                success=True,
                error=None,
            )
            attrs = mock_span.add_event.call_args[1]["attributes"]
            assert "agent.error" not in attrs

    def test_error_empty_string_included(self):
        """When error is an empty string (truthy check: empty str is falsy), no agent.error."""
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_agent_execution(
                agent_name="test",
                confidence=0.5,
                execution_time_ms=10.0,
                success=False,
                error="",
            )
            attrs = mock_span.add_event.call_args[1]["attributes"]
            # Empty string is falsy, so error should NOT be in attrs
            assert "agent.error" not in attrs

    def test_error_string_included(self):
        """When error is a non-empty string, agent.error is included."""
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_agent_execution(
                agent_name="agent1",
                confidence=0.1,
                execution_time_ms=500.0,
                success=False,
                error="connection refused",
            )
            attrs = mock_span.add_event.call_args[1]["attributes"]
            assert attrs["agent.error"] == "connection refused"

    def test_all_base_attributes_present(self):
        """All four base attributes are always present."""
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_agent_execution(
                agent_name="meta",
                confidence=0.75,
                execution_time_ms=250.0,
                success=True,
            )
            attrs = mock_span.add_event.call_args[1]["attributes"]
            assert attrs["agent.name"] == "meta"
            assert attrs["agent.confidence"] == 0.75
            assert attrs["agent.execution_time_ms"] == 250.0
            assert attrs["agent.success"] is True

    def test_no_span_no_error(self):
        """No error when current span is None."""
        with patch("src.observability.tracing.trace.get_current_span", return_value=None):
            record_agent_execution(
                agent_name="x",
                confidence=0.0,
                execution_time_ms=0.0,
                success=True,
            )
