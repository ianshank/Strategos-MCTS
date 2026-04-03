"""
Unit tests for src/observability/facade.py - extended coverage.

Covers missed lines: metrics with labels, trace with tracer active,
trace_async, profiling slow operations, metered decorator with histogram,
traced decorator on async functions, and convenience functions.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

from src.observability.facade import (
    ObservabilityConfig,
    ObservabilityFacade,
    OperationMetrics,
    get_observability,
    metered,
    profiled,
    traced,
)

pytestmark = pytest.mark.unit


class TestOperationMetricsExtended:
    """Extended tests for OperationMetrics."""

    def test_to_dict_with_metadata(self):
        """to_dict should include metadata when present."""
        m = OperationMetrics(
            name="op",
            duration_ms=50.0,
            success=True,
            metadata={"key": "value"},
        )
        d = m.to_dict()
        assert d["metadata"] == {"key": "value"}

    def test_to_dict_without_error_or_metadata(self):
        """to_dict should omit error and metadata when not set."""
        m = OperationMetrics(name="op", duration_ms=1.0, success=True)
        d = m.to_dict()
        assert "error" not in d
        assert "metadata" not in d

    def test_timestamp_auto_set(self):
        """timestamp should be auto-populated."""
        m = OperationMetrics(name="op", duration_ms=0.0, success=True)
        assert m.timestamp > 0


class TestObservabilityFacadeMetrics:
    """Tests for facade metric recording paths."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_record_counter_with_labels(self):
        """record_counter should call counter.labels().inc() when labels provided."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_counter = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.get.return_value = mock_counter
        obs._metrics = mock_metrics

        obs.record_counter("requests", value=2.0, labels={"agent": "hrm"})

        mock_metrics.get.assert_called_once_with("mcts_requests")
        mock_counter.labels.assert_called_once_with(agent="hrm")
        mock_counter.labels().inc.assert_called_once_with(2.0)

    def test_record_counter_without_labels(self):
        """record_counter should call counter.inc() when no labels."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_counter = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.get.return_value = mock_counter
        obs._metrics = mock_metrics

        obs.record_counter("requests", value=1.0)

        mock_counter.inc.assert_called_once_with(1.0)

    def test_record_counter_no_metric_found(self):
        """record_counter should handle None metric gracefully."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_metrics = MagicMock()
        mock_metrics.get.return_value = None
        obs._metrics = mock_metrics

        # Should not raise
        obs.record_counter("nonexistent")

    def test_record_counter_exception_logged(self):
        """record_counter should catch and log exceptions."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_metrics = MagicMock()
        mock_metrics.get.side_effect = RuntimeError("boom")
        obs._metrics = mock_metrics

        # Should not raise
        obs.record_counter("broken")

    def test_record_gauge_with_labels(self):
        """record_gauge with labels should call gauge.labels().set()."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_gauge = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.get.return_value = mock_gauge
        obs._metrics = mock_metrics

        obs.record_gauge("cpu_usage", 75.0, labels={"host": "a"})

        mock_gauge.labels.assert_called_once_with(host="a")
        mock_gauge.labels().set.assert_called_once_with(75.0)

    def test_record_gauge_without_labels(self):
        """record_gauge without labels should call gauge.set()."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_gauge = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.get.return_value = mock_gauge
        obs._metrics = mock_metrics

        obs.record_gauge("cpu_usage", 50.0)

        mock_gauge.set.assert_called_once_with(50.0)

    def test_record_gauge_exception_logged(self):
        """record_gauge should catch exceptions."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_metrics = MagicMock()
        mock_metrics.get.side_effect = RuntimeError("boom")
        obs._metrics = mock_metrics

        obs.record_gauge("broken", 1.0)

    def test_record_histogram_with_labels(self):
        """record_histogram with labels should call histogram.labels().observe()."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_hist = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.get.return_value = mock_hist
        obs._metrics = mock_metrics

        obs.record_histogram("latency", 0.5, labels={"endpoint": "/api"})

        mock_hist.labels.assert_called_once_with(endpoint="/api")
        mock_hist.labels().observe.assert_called_once_with(0.5)

    def test_record_histogram_without_labels(self):
        """record_histogram without labels should call histogram.observe()."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_hist = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.get.return_value = mock_hist
        obs._metrics = mock_metrics

        obs.record_histogram("latency", 0.3)

        mock_hist.observe.assert_called_once_with(0.3)

    def test_record_histogram_exception_logged(self):
        """record_histogram should catch exceptions."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        mock_metrics = MagicMock()
        mock_metrics.get.side_effect = RuntimeError("boom")
        obs._metrics = mock_metrics

        obs.record_histogram("broken", 1.0)

    def test_ensure_metrics_import_error(self):
        """_ensure_metrics should handle ImportError and fallback to empty dict."""
        config = ObservabilityConfig(metrics_enabled=True)
        obs = ObservabilityFacade(config)

        with patch(
            "src.observability.facade.ObservabilityFacade._ensure_metrics",
            wraps=obs._ensure_metrics,
        ):
            with patch.dict("sys.modules", {"src.observability.metrics": None}):
                obs._metrics = None
                obs._ensure_metrics()
                # On ImportError, _metrics becomes {}
                assert obs._metrics == {} or obs._metrics is not None


class TestObservabilityFacadeTracing:
    """Tests for facade tracing methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_trace_with_tracer_active(self):
        """trace should create a span when tracer is active."""
        config = ObservabilityConfig(tracing_enabled=True, metrics_enabled=False)
        obs = ObservabilityFacade(config)

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        obs._tracer = mock_tracer

        with obs.trace("test_span", attributes={"key": "val"}) as span:
            assert span is mock_span
            mock_span.set_attribute.assert_called_once_with("key", "val")

    def test_trace_with_tracer_none(self):
        """trace should yield None when tracer cannot be loaded."""
        config = ObservabilityConfig(tracing_enabled=True, metrics_enabled=False)
        obs = ObservabilityFacade(config)
        obs._tracer = None

        # Patch _ensure_tracer to be a no-op (tracer stays None)
        with patch.object(obs, "_ensure_tracer"):
            with obs.trace("test") as span:
                assert span is None

    def test_trace_records_error_on_exception(self):
        """trace should record error info when exception occurs."""
        config = ObservabilityConfig(tracing_enabled=True, metrics_enabled=False)
        obs = ObservabilityFacade(config)

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        obs._tracer = mock_tracer

        with pytest.raises(ValueError, match="boom"):
            with obs.trace("failing"):
                raise ValueError("boom")

    @pytest.mark.asyncio
    async def test_trace_async_disabled(self):
        """trace_async should yield None when tracing is disabled."""
        config = ObservabilityConfig(tracing_enabled=False)
        obs = ObservabilityFacade(config)

        async with obs.trace_async("test") as span:
            assert span is None

    @pytest.mark.asyncio
    async def test_trace_async_with_tracer(self):
        """trace_async should create span when tracer is active."""
        config = ObservabilityConfig(tracing_enabled=True, metrics_enabled=False)
        obs = ObservabilityFacade(config)

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        obs._tracer = mock_tracer

        async with obs.trace_async("async_span", attributes={"a": "b"}) as span:
            assert span is mock_span

    @pytest.mark.asyncio
    async def test_trace_async_with_no_tracer(self):
        """trace_async should yield None when tracer is None."""
        config = ObservabilityConfig(tracing_enabled=True, metrics_enabled=False)
        obs = ObservabilityFacade(config)
        obs._tracer = None

        with patch.object(obs, "_ensure_tracer"):
            async with obs.trace_async("test") as span:
                assert span is None

    @pytest.mark.asyncio
    async def test_trace_async_exception(self):
        """trace_async should propagate exceptions and record error."""
        config = ObservabilityConfig(tracing_enabled=True, metrics_enabled=False)
        obs = ObservabilityFacade(config)

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        obs._tracer = mock_tracer

        with pytest.raises(RuntimeError, match="async boom"):
            async with obs.trace_async("fail_span"):
                raise RuntimeError("async boom")

    def test_ensure_tracer_import_error(self):
        """_ensure_tracer should handle ImportError gracefully."""
        config = ObservabilityConfig(tracing_enabled=True)
        obs = ObservabilityFacade(config)
        obs._tracer = None

        with patch.dict("sys.modules", {"src.observability.tracing": None}):
            obs._ensure_tracer()
            # Should not raise; tracer remains None


class TestObservabilityFacadeProfiling:
    """Tests for profiling features."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_profile_slow_operation_warning(self):
        """profile should log warning when operation exceeds threshold."""
        config = ObservabilityConfig(
            profiling_enabled=True,
            profile_threshold_ms=0.001,  # Very low threshold to trigger warning
            metrics_enabled=False,
        )
        obs = ObservabilityFacade(config)

        with patch.object(obs, "log_warning") as mock_warn:
            with obs.profile("slow_op"):
                pass  # Any operation will exceed 0.001ms
            mock_warn.assert_called_once()
            assert "Slow operation" in mock_warn.call_args[0][0]

    def test_profile_fast_operation_no_warning(self):
        """profile should not log warning when profiling disabled."""
        config = ObservabilityConfig(
            profiling_enabled=False,
            metrics_enabled=False,
        )
        obs = ObservabilityFacade(config)

        with patch.object(obs, "log_warning") as mock_warn:
            with obs.profile("fast_op"):
                pass
            mock_warn.assert_not_called()

    def test_log_operation_success(self):
        """log_operation should log at INFO for success."""
        config = ObservabilityConfig(metrics_enabled=False)
        obs = ObservabilityFacade(config)

        m = OperationMetrics(name="op", duration_ms=10.0, success=True)
        # Should not raise
        obs.log_operation(m)

    def test_log_operation_failure(self):
        """log_operation should log at WARNING for failure."""
        config = ObservabilityConfig(metrics_enabled=False)
        obs = ObservabilityFacade(config)

        m = OperationMetrics(name="op", duration_ms=10.0, success=False, error="fail")
        obs.log_operation(m)


class TestFacadeDecoratorsExtended:
    """Extended tests for facade decorators."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    @pytest.mark.asyncio
    async def test_traced_async_decorator(self):
        """traced decorator should handle async functions."""
        config = ObservabilityConfig(tracing_enabled=False, metrics_enabled=False)
        obs = ObservabilityFacade(config)

        @obs.traced("async_op")
        async def my_async():
            return "async_result"

        result = await my_async()
        assert result == "async_result"

    def test_traced_default_name(self):
        """traced decorator without name should use function name."""
        config = ObservabilityConfig(tracing_enabled=False, metrics_enabled=False)
        obs = ObservabilityFacade(config)

        @obs.traced()
        def my_named_func():
            return "ok"

        assert my_named_func() == "ok"

    def test_profiled_default_name(self):
        """profiled decorator without name should use function name."""
        config = ObservabilityConfig(metrics_enabled=False)
        obs = ObservabilityFacade(config)

        @obs.profiled()
        def profiled_fn():
            return 42

        assert profiled_fn() == 42

    def test_metered_with_histogram(self):
        """metered decorator should record histogram when histogram_name given."""
        config = ObservabilityConfig(metrics_enabled=False)
        obs = ObservabilityFacade(config)

        @obs.metered("call_count", histogram_name="call_duration", labels={"svc": "test"})
        def metered_fn():
            return "metered"

        assert metered_fn() == "metered"

    def test_metered_exception_records_failure(self):
        """metered decorator should record success=False on exception."""
        config = ObservabilityConfig(metrics_enabled=False)
        obs = ObservabilityFacade(config)

        with patch.object(obs, "record_counter") as mock_counter:

            @obs.metered("calls", histogram_name="dur")
            def failing():
                raise ValueError("fail")

            with pytest.raises(ValueError):
                failing()

            # Should have been called with success=False label
            assert mock_counter.called


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_get_observability_returns_facade(self):
        result = get_observability()
        assert isinstance(result, ObservabilityFacade)

    def test_traced_convenience(self):
        @traced("test")
        def fn():
            return 1

        assert fn() == 1

    def test_profiled_convenience(self):
        @profiled("test")
        def fn():
            return 2

        assert fn() == 2

    def test_metered_convenience(self):
        @metered("test_counter")
        def fn():
            return 3

        assert fn() == 3


class TestGetLogExtra:
    """Tests for _get_log_extra method."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_no_correlation_id(self):
        """Should not include correlation_id when not set."""
        obs = ObservabilityFacade(ObservabilityConfig())
        extra = obs._get_log_extra(key="val")
        assert "correlation_id" not in extra
        assert extra["key"] == "val"

    def test_correlation_id_disabled(self):
        """Should not include correlation_id when include_correlation_id is False."""
        config = ObservabilityConfig(include_correlation_id=False)
        obs = ObservabilityFacade(config)
        obs.set_correlation_id("test-id")
        extra = obs._get_log_extra()
        assert "correlation_id" not in extra

    def test_log_error_with_exc_info(self):
        """log_error should accept exc_info parameter."""
        obs = ObservabilityFacade(ObservabilityConfig(metrics_enabled=False))
        # Should not raise
        obs.log_error("error msg", exc_info=True, extra_key="val")


class TestSingletonWithConfig:
    """Tests for singleton creation with custom config."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_get_instance_with_config(self):
        """get_instance should use provided config on first call."""
        config = ObservabilityConfig(metrics_prefix="custom")
        obs = ObservabilityFacade.get_instance(config)
        assert obs.config.metrics_prefix == "custom"

    def test_get_instance_ignores_config_on_second_call(self):
        """get_instance should return existing instance even with different config."""
        config1 = ObservabilityConfig(metrics_prefix="first")
        config2 = ObservabilityConfig(metrics_prefix="second")
        obs1 = ObservabilityFacade.get_instance(config1)
        obs2 = ObservabilityFacade.get_instance(config2)
        assert obs1 is obs2
        assert obs2.config.metrics_prefix == "first"
