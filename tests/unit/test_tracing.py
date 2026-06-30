"""
Unit tests for src/observability/tracing.py

Tests the OpenTelemetry tracing infrastructure including:
- TracingManager singleton and initialization
- Span creation context managers (sync and async)
- trace_operation decorator
- SpanContextPropagator
- MCTS attribute helpers
- Record functions for MCTS iterations and agent execution
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure opentelemetry stubs are available before importing the module under test.
# We mock the entire opentelemetry ecosystem so the tests don't require the SDK.

_otel_modules = {
    "opentelemetry": MagicMock(),
    "opentelemetry.trace": MagicMock(),
    "opentelemetry.context": MagicMock(),
    "opentelemetry.exporter": MagicMock(),
    "opentelemetry.exporter.otlp": MagicMock(),
    "opentelemetry.exporter.otlp.proto": MagicMock(),
    "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
    "opentelemetry.instrumentation": MagicMock(),
    "opentelemetry.instrumentation.httpx": MagicMock(),
    "opentelemetry.propagate": MagicMock(),
    "opentelemetry.sdk": MagicMock(),
    "opentelemetry.sdk.resources": MagicMock(),
    "opentelemetry.sdk.trace": MagicMock(),
    "opentelemetry.sdk.trace.export": MagicMock(),
    "opentelemetry.trace.propagation": MagicMock(),
    "opentelemetry.trace.propagation.tracecontext": MagicMock(),
    "psutil": MagicMock(),
}

# Snapshot original sys.modules entries so teardown can remove any stubs we add
# without clobbering a real OpenTelemetry SDK that may already be installed.
_original_sys_modules = {name: sys.modules.get(name) for name in _otel_modules}

for mod_name, mod_mock in _otel_modules.items():
    sys.modules.setdefault(mod_name, mod_mock)

# Record every in-place attribute override so ``teardown_module`` can undo them.
# When the real SDK is installed these assignments mutate the *real* module
# objects; without restoration the MagicMocks leak into other test modules that
# share the process / xdist worker and create real spans, surfacing as spurious
# AttributeErrors (e.g. ``BatchSpanProcessor`` becoming a MagicMock).
_patched_attrs: list[tuple[object, str, bool, object]] = []


def _patch_attr(module: object, name: str, value: object) -> None:
    _patched_attrs.append((module, name, hasattr(module, name), getattr(module, name, None)))
    setattr(module, name, value)


# Now set up specific attribute mocks that the source module references at import time.
_trace_mod = sys.modules["opentelemetry.trace"]
_span_kind = MagicMock()
_span_kind.INTERNAL = "INTERNAL"
_span_kind.CLIENT = "CLIENT"
_patch_attr(_trace_mod, "SpanKind", _span_kind)
_patch_attr(_trace_mod, "Status", MagicMock())
_status_code = MagicMock()
_status_code.OK = "OK"
_patch_attr(_trace_mod, "StatusCode", _status_code)

_sdk_resources = sys.modules["opentelemetry.sdk.resources"]
_patch_attr(_sdk_resources, "SERVICE_NAME", "service.name")
_patch_attr(_sdk_resources, "Resource", MagicMock())

_sdk_trace = sys.modules["opentelemetry.sdk.trace"]
_patch_attr(_sdk_trace, "TracerProvider", MagicMock)

_export_mod = sys.modules["opentelemetry.sdk.trace.export"]
_patch_attr(_export_mod, "BatchSpanProcessor", MagicMock)
_patch_attr(_export_mod, "ConsoleSpanExporter", MagicMock)
_patch_attr(_export_mod, "SimpleSpanProcessor", MagicMock)

_otlp_exporter = sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"]
_patch_attr(_otlp_exporter, "OTLPSpanExporter", MagicMock)

_httpx_instr = sys.modules["opentelemetry.instrumentation.httpx"]
_patch_attr(_httpx_instr, "HTTPXClientInstrumentor", MagicMock)

_propagation = sys.modules["opentelemetry.trace.propagation.tracecontext"]
_patch_attr(_propagation, "TraceContextTextMapPropagator", MagicMock)

# Make trace module's functions behave properly
_trace_top = sys.modules["opentelemetry"]
_patch_attr(_trace_top, "trace", _trace_mod)

# Import the module under test (after stubs are in place)
from src.observability.tracing import (
    SpanContextPropagator,
    TracingManager,
    add_mcts_attributes,
    get_tracer,
    record_agent_execution,
    record_mcts_iteration,
    trace_operation,
    trace_span,
)


def teardown_module(module: object) -> None:
    """Undo the in-place OpenTelemetry SDK stubs applied at import time.

    Restores every mutated attribute, drops only the stub modules this file
    inserted, and reloads the module under test so its globals rebind to the
    real SDK. Without this, the import-time MagicMocks leak into other test
    modules that share the process / xdist worker and create real spans,
    surfacing as spurious AttributeErrors.
    """
    import contextlib
    import importlib

    for mod, name, had, original in reversed(_patched_attrs):
        if had:
            setattr(mod, name, original)
        else:
            with contextlib.suppress(AttributeError):
                delattr(mod, name)

    for mod_name, original in _original_sys_modules.items():
        if original is None:
            sys.modules.pop(mod_name, None)

    # Best-effort: rebind the module-under-test's globals to the restored SDK.
    with contextlib.suppress(Exception):
        importlib.reload(sys.modules["src.observability.tracing"])


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset TracingManager singleton between tests."""
    TracingManager._instance = None
    TracingManager._provider = None
    yield
    TracingManager._instance = None
    TracingManager._provider = None


@pytest.mark.unit
class TestTracingManager:
    """Tests for TracingManager class."""

    def test_singleton_instance(self):
        """get_instance returns the same object on repeated calls."""
        inst1 = TracingManager.get_instance()
        inst2 = TracingManager.get_instance()
        assert inst1 is inst2

    def test_initialize_sets_initialized_flag(self):
        mgr = TracingManager()
        assert mgr._initialized is False
        mgr.initialize(exporter_type="none")
        assert mgr._initialized is True

    def test_initialize_idempotent(self):
        """Calling initialize twice should be a no-op the second time."""
        mgr = TracingManager()
        mgr.initialize(exporter_type="none")
        # Capture provider reference
        provider1 = mgr._provider
        mgr.initialize(exporter_type="none")
        # Provider should be unchanged
        assert mgr._provider is provider1

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    def test_initialize_otlp_exporter(self, mock_batch, mock_otlp):
        mgr = TracingManager()
        mgr.initialize(exporter_type="otlp", otlp_endpoint="localhost:4317")
        assert mgr._initialized is True
        assert mgr._provider is not None
        mock_otlp.assert_called_once()
        mock_batch.assert_called_once()

    @patch("src.observability.tracing.ConsoleSpanExporter")
    @patch("src.observability.tracing.SimpleSpanProcessor")
    def test_initialize_console_exporter(self, mock_simple, mock_console):
        mgr = TracingManager()
        mgr.initialize(exporter_type="console")
        assert mgr._initialized is True
        mock_console.assert_called_once()
        mock_simple.assert_called_once()

    def test_initialize_unknown_exporter_raises(self):
        mgr = TracingManager()
        with pytest.raises(ValueError, match="Unknown exporter type"):
            mgr.initialize(exporter_type="foobar")

    def test_initialize_with_additional_resources(self):
        mgr = TracingManager()
        mgr.initialize(exporter_type="none", additional_resources={"team": "mcts"})
        assert mgr._initialized is True

    def test_shutdown_resets_initialized(self):
        mgr = TracingManager()
        mgr.initialize(exporter_type="none")
        assert mgr._initialized is True
        mgr.shutdown()
        assert mgr._initialized is False

    def test_shutdown_without_initialize(self):
        """Shutdown on uninitialized manager should not raise."""
        mgr = TracingManager()
        mgr.shutdown()  # no error

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    def test_get_tracer_auto_initializes(self, _mock_batch, _mock_otlp):
        mgr = TracingManager()
        assert mgr._initialized is False
        _tracer = mgr.get_tracer("test")
        assert mgr._initialized is True

    @patch.dict("os.environ", {"OTEL_SERVICE_NAME": "test-svc", "OTEL_EXPORTER_TYPE": "none"})
    def test_initialize_reads_env_vars(self):
        mgr = TracingManager()
        mgr.initialize()
        assert mgr._initialized is True

    def test_instrument_httpx_called_once(self):
        mgr = TracingManager()
        mgr._instrument_httpx()
        assert mgr._httpx_instrumented is True
        # Second call should be a no-op
        mgr._instrument_httpx()
        assert mgr._httpx_instrumented is True


@pytest.mark.unit
class TestModuleLevelGetTracer:
    """Tests for the module-level get_tracer function."""

    @patch("src.observability.tracing.OTLPSpanExporter")
    @patch("src.observability.tracing.BatchSpanProcessor")
    def test_get_tracer_returns_tracer(self, _mock_batch, _mock_otlp):
        tracer = get_tracer("my-tracer")
        # Should not raise; just verify something is returned
        assert tracer is not None


@pytest.mark.unit
class TestAddMctsAttributes:
    """Tests for add_mcts_attributes helper."""

    def test_adds_prefixed_attributes(self):
        span = MagicMock()
        add_mcts_attributes(span, **{"mcts.iteration": 5, "agent.name": "hrm"})
        span.set_attribute.assert_any_call("mcts.iteration", 5)
        span.set_attribute.assert_any_call("agent.name", "hrm")

    def test_auto_prefixes_custom_keys(self):
        span = MagicMock()
        add_mcts_attributes(span, foo="bar")
        span.set_attribute.assert_called_once_with("custom.foo", "bar")

    def test_skips_none_values(self):
        span = MagicMock()
        add_mcts_attributes(span, **{"mcts.iteration": None})
        span.set_attribute.assert_not_called()

    def test_framework_prefix_not_overridden(self):
        span = MagicMock()
        add_mcts_attributes(span, **{"framework.version": "1.0"})
        span.set_attribute.assert_called_once_with("framework.version", "1.0")


@pytest.mark.unit
class TestTraceSpan:
    """Tests for the trace_span context manager."""

    def test_trace_span_yields_span(self):
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with trace_span("test.span") as span:
                assert span is mock_span
                span.set_attribute.assert_called()  # correlation_id set

    def test_trace_span_with_attributes(self):
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
            with trace_span("test.span", attributes={"key": "val"}) as span:
                assert span is mock_span


@pytest.mark.unit
class TestAsyncTraceSpan:
    """Tests for the async_trace_span context manager."""

    def test_async_trace_span_yields_span(self):
        from src.observability.tracing import async_trace_span

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def _run():
            with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):
                async with async_trace_span("async.span") as span:
                    assert span is mock_span

        asyncio.run(_run())


@pytest.mark.unit
class TestTraceOperationDecorator:
    """Tests for the trace_operation decorator."""

    def test_sync_function_decorated(self):
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="my_op")
            def my_func(x, y):
                return x + y

            result = my_func(1, 2)
            assert result == 3
            mock_span.set_attribute.assert_any_call("function.args_count", 2)
            mock_span.set_attribute.assert_any_call("function.kwargs_count", 0)

    def test_async_function_decorated(self):
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation(name="async_op")
            async def my_async_func(a):
                return a * 2

            result = asyncio.run(my_async_func(5))
            assert result == 10

    def test_decorator_default_name_uses_qualname(self):
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.observability.tracing.get_tracer", return_value=mock_tracer):

            @trace_operation()
            def another_func():
                return 42

            result = another_func()
            assert result == 42


@pytest.mark.unit
class TestSpanContextPropagator:
    """Tests for SpanContextPropagator."""

    def test_inject_calls_propagate_inject(self):
        with patch("src.observability.tracing.inject") as mock_inject:
            prop = SpanContextPropagator()
            carrier = {}
            prop.inject(carrier)
            mock_inject.assert_called_once_with(carrier, context=None)

    def test_extract_calls_propagate_extract(self):
        with patch("src.observability.tracing.extract") as mock_extract:
            mock_extract.return_value = MagicMock()
            prop = SpanContextPropagator()
            ctx = prop.extract({"traceparent": "abc"})
            mock_extract.assert_called_once()
            assert ctx is not None

    def test_get_trace_parent_returns_string_or_none(self):
        with patch("src.observability.tracing.inject") as mock_inject:
            # Simulate inject populating carrier
            def _populate(carrier, context=None):
                carrier["traceparent"] = "00-abc-def-01"

            mock_inject.side_effect = _populate
            prop = SpanContextPropagator()
            tp = prop.get_trace_parent()
            assert tp == "00-abc-def-01"

    def test_get_trace_parent_returns_none_when_no_traceparent(self):
        with patch("src.observability.tracing.inject"):
            prop = SpanContextPropagator()
            tp = prop.get_trace_parent()
            assert tp is None


@pytest.mark.unit
class TestRecordMctsIteration:
    """Tests for record_mcts_iteration."""

    def test_records_event_on_active_span(self):
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_mcts_iteration(
                iteration=3,
                selected_node_id="node-1",
                ucb_score=1.5,
                node_visits=10,
                node_value=0.8,
                tree_depth=4,
            )
            mock_span.add_event.assert_called_once()
            call_args = mock_span.add_event.call_args
            assert call_args[0][0] == "mcts.iteration"
            attrs = call_args[1]["attributes"]
            assert attrs["mcts.iteration"] == 3
            assert attrs["mcts.selected_node_id"] == "node-1"

    def test_no_error_when_no_current_span(self):
        with patch("src.observability.tracing.trace.get_current_span", return_value=None):
            # Should not raise
            record_mcts_iteration(
                iteration=0,
                selected_node_id="x",
                ucb_score=0.0,
                node_visits=0,
                node_value=0.0,
                tree_depth=0,
            )


@pytest.mark.unit
class TestRecordAgentExecution:
    """Tests for record_agent_execution."""

    def test_records_event_with_error(self):
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_agent_execution(
                agent_name="hrm",
                confidence=0.9,
                execution_time_ms=120.5,
                success=False,
                error="timeout",
            )
            call_args = mock_span.add_event.call_args
            attrs = call_args[1]["attributes"]
            assert attrs["agent.error"] == "timeout"
            assert attrs["agent.success"] is False

    def test_records_event_without_error(self):
        mock_span = MagicMock()
        with patch("src.observability.tracing.trace.get_current_span", return_value=mock_span):
            record_agent_execution(
                agent_name="trm",
                confidence=0.95,
                execution_time_ms=50.0,
                success=True,
            )
            call_args = mock_span.add_event.call_args
            attrs = call_args[1]["attributes"]
            assert "agent.error" not in attrs
            assert attrs["agent.name"] == "trm"

    def test_no_error_when_no_current_span(self):
        with patch("src.observability.tracing.trace.get_current_span", return_value=None):
            record_agent_execution(
                agent_name="x",
                confidence=0.0,
                execution_time_ms=0.0,
                success=True,
            )
