"""
Unit tests for observability modules.

Tests the facade and decorators for logging, metrics, tracing, and profiling.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 10
"""

from __future__ import annotations

import os
import time

import pytest

# Set environment variables before importing modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

# Import with graceful fallback
try:
    from src.observability.facade import (
        ObservabilityConfig,
        ObservabilityFacade,
        OperationMetrics,
        get_observability,
        metered,
        profiled,
        traced,
    )

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False

try:
    from src.observability.decorators import (
        cached,
        debug_on_error,
        logged,
        retry,
        timed,
        validate_args,
    )

    DECORATORS_AVAILABLE = True
except ImportError:
    DECORATORS_AVAILABLE = False

pytestmark = pytest.mark.unit


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Facade not available")
class TestObservabilityConfig:
    """Test ObservabilityConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ObservabilityConfig()
        assert config.log_level == "INFO"
        assert config.json_logging is True
        assert config.metrics_enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ObservabilityConfig(
            log_level="DEBUG",
            metrics_enabled=False,
            profiling_enabled=True,
        )
        assert config.log_level == "DEBUG"
        assert config.metrics_enabled is False

    def test_from_settings(self):
        """Test creating config from settings."""
        config = ObservabilityConfig.from_settings()
        assert isinstance(config.log_level, str)


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Facade not available")
class TestOperationMetrics:
    """Test OperationMetrics class."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = OperationMetrics(
            name="test_op",
            duration_ms=100.5,
            success=True,
        )
        assert metrics.name == "test_op"
        assert metrics.success is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = OperationMetrics(
            name="test",
            duration_ms=123.456789,
            success=True,
        )
        d = metrics.to_dict()
        assert d["operation_name"] == "test"
        assert d["duration_ms"] == 123.457

    def test_to_dict_with_error(self):
        """Test dictionary with error."""
        metrics = OperationMetrics(
            name="error_op",
            duration_ms=10.0,
            success=False,
            error="Test error",
        )
        d = metrics.to_dict()
        assert "error" in d


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Facade not available")
class TestObservabilityFacade:
    """Test ObservabilityFacade class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton between tests."""
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_singleton(self):
        """Test singleton pattern."""
        obs1 = get_observability()
        obs2 = get_observability()
        assert obs1 is obs2

    def test_reset(self):
        """Test reset functionality."""
        obs1 = get_observability()
        ObservabilityFacade.reset()
        obs2 = get_observability()
        assert obs1 is not obs2

    def test_correlation_id(self):
        """Test correlation ID tracking."""
        obs = get_observability()
        obs.set_correlation_id("test-123")
        extra = obs._get_log_extra(custom="value")
        assert extra["correlation_id"] == "test-123"

    def test_log_methods(self):
        """Test logging methods don't raise."""
        obs = get_observability()
        obs.log_debug("Debug message")
        obs.log_info("Info message")
        obs.log_warning("Warning message")
        obs.log_error("Error message")

    def test_trace_disabled(self):
        """Test trace when disabled."""
        obs = ObservabilityFacade(ObservabilityConfig(tracing_enabled=False))
        with obs.trace("test") as span:
            assert span is None

    def test_profile_success(self):
        """Test profile context manager."""
        obs = get_observability()
        with obs.profile("test") as m:
            time.sleep(0.01)
        assert m.success is True
        assert m.duration_ms >= 10

    def test_profile_error(self):
        """Test profile captures errors."""
        obs = get_observability()
        with pytest.raises(ValueError):
            with obs.profile("fail") as m:
                raise ValueError("Test")
        assert m.success is False

    def test_metrics_disabled(self):
        """Test metrics when disabled."""
        obs = ObservabilityFacade(ObservabilityConfig(metrics_enabled=False))
        obs.record_counter("test")
        obs.record_gauge("test", 1.0)
        obs.record_histogram("test", 0.5)


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Facade not available")
class TestFacadeDecorators:
    """Test facade decorators."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ObservabilityFacade.reset()
        yield
        ObservabilityFacade.reset()

    def test_traced_decorator(self):
        """Test traced decorator."""

        @traced("test")
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_profiled_decorator(self):
        """Test profiled decorator."""

        @profiled("test")
        def my_func():
            return "profiled"

        assert my_func() == "profiled"

    def test_metered_decorator(self):
        """Test metered decorator."""

        @metered("test")
        def my_func():
            return "metered"

        assert my_func() == "metered"


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
class TestLoggedDecorator:
    """Test logged decorator."""

    def test_sync_function(self):
        """Test on sync function."""

        @logged()
        def my_func(x, y):
            return x + y

        assert my_func(1, 2) == 3

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test on async function."""

        @logged()
        async def async_func(x):
            return x * 2

        assert await async_func(5) == 10

    def test_captures_exception(self):
        """Test captures exceptions."""

        @logged()
        def fails():
            raise ValueError("Test")

        with pytest.raises(ValueError):
            fails()


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
class TestTimedDecorator:
    """Test timed decorator."""

    def test_basic(self):
        """Test basic timing."""

        @timed()
        def fast():
            return "fast"

        assert fast() == "fast"

    @pytest.mark.asyncio
    async def test_async(self):
        """Test async timing."""

        @timed()
        async def async_timed():
            return "async"

        assert await async_timed() == "async"


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
class TestRetryDecorator:
    """Test retry decorator."""

    def test_success_first_try(self):
        """Test success on first try."""

        @retry(max_attempts=3)
        def succeeds():
            return "ok"

        assert succeeds() == "ok"

    def test_success_after_retry(self):
        """Test success after retry."""
        count = [0]

        @retry(max_attempts=3, initial_delay=0.01)
        def eventually():
            count[0] += 1
            if count[0] < 2:
                raise ConnectionError()
            return "ok"

        assert eventually() == "ok"
        assert count[0] == 2

    def test_all_fail(self):
        """Test all attempts fail."""

        @retry(max_attempts=2, initial_delay=0.01)
        def fails():
            raise ValueError("Fail")

        with pytest.raises(ValueError):
            fails()

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry."""
        count = [0]

        @retry(max_attempts=3, initial_delay=0.01)
        async def async_eventually():
            count[0] += 1
            if count[0] < 2:
                raise ConnectionError()
            return "async_ok"

        assert await async_eventually() == "async_ok"


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
class TestCachedDecorator:
    """Test cached decorator."""

    def test_basic_caching(self):
        """Test basic caching."""
        count = [0]

        @cached(ttl_seconds=60)
        def expensive(x):
            count[0] += 1
            return x * 2

        expensive(5)
        expensive(5)
        assert count[0] == 1

    def test_different_args(self):
        """Test different args cached separately."""
        count = [0]

        @cached(ttl_seconds=60)
        def compute(x):
            count[0] += 1
            return x

        compute(1)
        compute(2)
        assert count[0] == 2

    def test_cache_clear(self):
        """Test cache clearing."""
        count = [0]

        @cached(ttl_seconds=60)
        def clearable():
            count[0] += 1
            return "result"

        clearable()
        clearable.cache_clear()
        clearable()
        assert count[0] == 2


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
class TestValidateArgsDecorator:
    """Test validate_args decorator."""

    def test_valid(self):
        """Test valid args."""

        @validate_args(x=lambda v: v > 0)
        def positive(x):
            return x

        assert positive(5) == 5

    def test_invalid(self):
        """Test invalid args."""

        @validate_args(x=lambda v: v > 0)
        def positive(x):
            return x

        with pytest.raises(ValueError):
            positive(-1)


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="Decorators not available")
class TestDebugOnErrorDecorator:
    """Test debug_on_error decorator."""

    def test_no_error(self):
        """Test no error."""

        @debug_on_error()
        def ok():
            return "ok"

        assert ok() == "ok"

    def test_with_error(self):
        """Test with error."""

        @debug_on_error(reraise=True)
        def fails():
            raise RuntimeError("Test")

        with pytest.raises(RuntimeError):
            fails()

    def test_no_reraise(self):
        """Test no reraise."""

        @debug_on_error(reraise=False)
        def silenced():
            raise RuntimeError("Silenced")

        assert silenced() is None
