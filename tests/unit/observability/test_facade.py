"""
Comprehensive tests for observability facade.

Tests ObservabilityConfig, OperationMetrics, ObservabilityFacade,
and convenience functions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("psutil", reason="psutil required for observability tests")

from src.observability.facade import (
    ObservabilityConfig,
    ObservabilityFacade,
    OperationMetrics,
    get_observability,
    metered,
    profiled,
    traced,
)


@pytest.fixture(autouse=True)
def reset_facade() -> None:
    """Reset facade singleton before each test."""
    ObservabilityFacade.reset()


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings."""
    settings = MagicMock()
    settings.LOG_LEVEL = "INFO"
    settings.JSON_LOGGING = True
    settings.METRICS_ENABLED = True
    settings.METRICS_PREFIX = "mcts"
    settings.METRICS_PORT = 9090
    settings.TRACING_ENABLED = False  # Disable to avoid loading OpenTelemetry
    settings.TRACE_SAMPLE_RATE = 1.0
    settings.SERVICE_NAME = "test-service"
    settings.PROFILING_ENABLED = False
    settings.PROFILE_THRESHOLD_MS = 100.0
    return settings


# ============================================================================
# ObservabilityConfig Tests
# ============================================================================


class TestObservabilityConfig:
    """Tests for ObservabilityConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ObservabilityConfig()

        assert config.log_level == "INFO"
        assert config.json_logging is True
        assert config.include_correlation_id is True
        assert config.metrics_enabled is True
        assert config.metrics_prefix == "mcts"
        assert config.metrics_port == 9090
        assert config.tracing_enabled is True
        assert config.trace_sample_rate == 1.0
        assert config.service_name == "langgraph-mcts"
        assert config.profiling_enabled is False
        assert config.profile_threshold_ms == 100.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ObservabilityConfig(
            log_level="DEBUG",
            metrics_enabled=False,
            tracing_enabled=False,
            profiling_enabled=True,
        )

        assert config.log_level == "DEBUG"
        assert config.metrics_enabled is False
        assert config.tracing_enabled is False
        assert config.profiling_enabled is True

    def test_from_settings(self, mock_settings: MagicMock) -> None:
        """Test creating config from settings."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):
            config = ObservabilityConfig.from_settings()

            assert config.log_level == "INFO"
            assert config.service_name == "test-service"
            assert config.metrics_prefix == "mcts"


# ============================================================================
# OperationMetrics Tests
# ============================================================================


class TestOperationMetrics:
    """Tests for OperationMetrics dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating operation metrics."""
        metrics = OperationMetrics(
            name="test_operation",
            duration_ms=50.5,
            success=True,
        )

        assert metrics.name == "test_operation"
        assert metrics.duration_ms == 50.5
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.metadata == {}

    def test_with_error(self) -> None:
        """Test metrics with error."""
        metrics = OperationMetrics(
            name="failed_op",
            duration_ms=100.0,
            success=False,
            error="Connection timeout",
        )

        assert metrics.success is False
        assert metrics.error == "Connection timeout"

    def test_with_metadata(self) -> None:
        """Test metrics with metadata."""
        metrics = OperationMetrics(
            name="op_with_meta",
            duration_ms=25.0,
            success=True,
            metadata={"user_id": "123", "action": "query"},
        )

        assert metrics.metadata["user_id"] == "123"
        assert metrics.metadata["action"] == "query"

    def test_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = OperationMetrics(
            name="test_op",
            duration_ms=50.123456,
            success=True,
        )

        result = metrics.to_dict()

        assert result["operation_name"] == "test_op"
        assert result["duration_ms"] == 50.123
        assert result["success"] is True
        assert "timestamp" in result

    def test_to_dict_with_error(self) -> None:
        """Test to_dict includes error when present."""
        metrics = OperationMetrics(
            name="test_op",
            duration_ms=100.0,
            success=False,
            error="Test error",
        )

        result = metrics.to_dict()

        assert result["error"] == "Test error"

    def test_to_dict_with_metadata(self) -> None:
        """Test to_dict includes metadata when present."""
        metrics = OperationMetrics(
            name="test_op",
            duration_ms=50.0,
            success=True,
            metadata={"key": "value"},
        )

        result = metrics.to_dict()

        assert result["metadata"] == {"key": "value"}

    def test_timestamp_auto_set(self) -> None:
        """Test timestamp is automatically set."""
        before = time.time()
        metrics = OperationMetrics(
            name="test",
            duration_ms=0,
            success=True,
        )
        after = time.time()

        assert before <= metrics.timestamp <= after


# ============================================================================
# ObservabilityFacade Tests - Singleton
# ============================================================================


class TestObservabilityFacadeSingleton:
    """Tests for ObservabilityFacade singleton pattern."""

    def test_get_instance_returns_same_object(self) -> None:
        """Test get_instance returns singleton."""
        config = ObservabilityConfig()
        instance1 = ObservabilityFacade.get_instance(config)
        instance2 = ObservabilityFacade.get_instance()

        assert instance1 is instance2

    def test_reset_clears_instance(self) -> None:
        """Test reset clears singleton."""
        config = ObservabilityConfig()
        instance1 = ObservabilityFacade.get_instance(config)
        ObservabilityFacade.reset()
        instance2 = ObservabilityFacade.get_instance(config)

        assert instance1 is not instance2

    def test_init_with_config(self) -> None:
        """Test initialization with custom config."""
        config = ObservabilityConfig(log_level="DEBUG")
        facade = ObservabilityFacade(config)

        assert facade.config.log_level == "DEBUG"


# ============================================================================
# ObservabilityFacade Tests - Logging
# ============================================================================


class TestObservabilityFacadeLogging:
    """Tests for ObservabilityFacade logging methods."""

    @pytest.fixture
    def facade(self) -> ObservabilityFacade:
        """Create facade with metrics/tracing disabled."""
        config = ObservabilityConfig(
            metrics_enabled=False,
            tracing_enabled=False,
        )
        return ObservabilityFacade(config)

    def test_set_correlation_id(self, facade: ObservabilityFacade) -> None:
        """Test setting correlation ID."""
        facade.set_correlation_id("test-123")
        assert facade._correlation_id == "test-123"

    def test_log_debug(self, facade: ObservabilityFacade) -> None:
        """Test debug logging."""
        with patch("src.observability.facade.logger") as mock_logger:
            facade.log_debug("Debug message", key="value")
            mock_logger.debug.assert_called_once()

    def test_log_info(self, facade: ObservabilityFacade) -> None:
        """Test info logging."""
        with patch("src.observability.facade.logger") as mock_logger:
            facade.log_info("Info message", key="value")
            mock_logger.info.assert_called_once()

    def test_log_warning(self, facade: ObservabilityFacade) -> None:
        """Test warning logging."""
        with patch("src.observability.facade.logger") as mock_logger:
            facade.log_warning("Warning message")
            mock_logger.warning.assert_called_once()

    def test_log_error(self, facade: ObservabilityFacade) -> None:
        """Test error logging."""
        with patch("src.observability.facade.logger") as mock_logger:
            facade.log_error("Error message", exc_info=True)
            mock_logger.error.assert_called_once()

    def test_log_includes_correlation_id(self, facade: ObservabilityFacade) -> None:
        """Test logs include correlation ID when set."""
        facade.set_correlation_id("corr-123")

        with patch("src.observability.facade.logger") as mock_logger:
            facade.log_info("Test message")

            call_kwargs = mock_logger.info.call_args
            extra = call_kwargs.kwargs.get("extra", {})
            assert extra.get("correlation_id") == "corr-123"

    def test_log_operation(self, facade: ObservabilityFacade) -> None:
        """Test logging operation metrics."""
        metrics = OperationMetrics(
            name="test_op",
            duration_ms=50.0,
            success=True,
        )

        with patch("src.observability.facade.logger") as mock_logger:
            facade.log_operation(metrics)
            mock_logger.log.assert_called()

    def test_log_operation_warning_on_failure(self, facade: ObservabilityFacade) -> None:
        """Test operation failure logs at warning level."""
        metrics = OperationMetrics(
            name="test_op",
            duration_ms=50.0,
            success=False,
            error="Test error",
        )

        with patch("src.observability.facade.logger") as mock_logger:
            facade.log_operation(metrics)

            call_args = mock_logger.log.call_args
            assert call_args.args[0] == logging.WARNING


# ============================================================================
# ObservabilityFacade Tests - Metrics
# ============================================================================


class TestObservabilityFacadeMetrics:
    """Tests for ObservabilityFacade metrics methods."""

    @pytest.fixture
    def facade(self) -> ObservabilityFacade:
        """Create facade with metrics enabled."""
        config = ObservabilityConfig(
            metrics_enabled=True,
            tracing_enabled=False,
        )
        return ObservabilityFacade(config)

    def test_record_counter_disabled(self) -> None:
        """Test counter not recorded when metrics disabled."""
        config = ObservabilityConfig(metrics_enabled=False)
        facade = ObservabilityFacade(config)

        # Should not raise
        facade.record_counter("test_counter")

    def test_record_counter_with_labels(self, facade: ObservabilityFacade) -> None:
        """Test recording counter with labels."""
        mock_counter = MagicMock()
        mock_registry = {"mcts_test_counter": mock_counter}

        with patch.object(facade, "_metrics", mock_registry):
            facade.record_counter("test_counter", value=5, labels={"env": "test"})

            mock_counter.labels.assert_called_with(env="test")

    def test_record_gauge_disabled(self) -> None:
        """Test gauge not recorded when metrics disabled."""
        config = ObservabilityConfig(metrics_enabled=False)
        facade = ObservabilityFacade(config)

        # Should not raise
        facade.record_gauge("test_gauge", 100.0)

    def test_record_histogram_disabled(self) -> None:
        """Test histogram not recorded when metrics disabled."""
        config = ObservabilityConfig(metrics_enabled=False)
        facade = ObservabilityFacade(config)

        # Should not raise
        facade.record_histogram("test_histogram", 50.0)


# ============================================================================
# ObservabilityFacade Tests - Tracing
# ============================================================================


class TestObservabilityFacadeTracing:
    """Tests for ObservabilityFacade tracing methods."""

    @pytest.fixture
    def facade(self) -> ObservabilityFacade:
        """Create facade with tracing disabled."""
        config = ObservabilityConfig(
            metrics_enabled=False,
            tracing_enabled=False,
        )
        return ObservabilityFacade(config)

    def test_trace_disabled(self, facade: ObservabilityFacade) -> None:
        """Test trace context manager when disabled."""
        with facade.trace("test_span") as span:
            assert span is None

    def test_trace_executes_body(self, facade: ObservabilityFacade) -> None:
        """Test trace context manager executes body."""
        result = None

        with facade.trace("test_span"):
            result = "executed"

        assert result == "executed"

    def test_trace_propagates_exception(self, facade: ObservabilityFacade) -> None:
        """Test trace propagates exceptions."""
        with pytest.raises(ValueError, match="test error"):
            with facade.trace("test_span"):
                raise ValueError("test error")

    @pytest.mark.asyncio
    async def test_trace_async_disabled(self, facade: ObservabilityFacade) -> None:
        """Test async trace context manager when disabled."""
        async with facade.trace_async("test_span") as span:
            assert span is None

    @pytest.mark.asyncio
    async def test_trace_async_executes_body(self, facade: ObservabilityFacade) -> None:
        """Test async trace context manager executes body."""
        result = None

        async with facade.trace_async("test_span"):
            await asyncio.sleep(0.01)
            result = "executed"

        assert result == "executed"


# ============================================================================
# ObservabilityFacade Tests - Profiling
# ============================================================================


class TestObservabilityFacadeProfiling:
    """Tests for ObservabilityFacade profiling methods."""

    @pytest.fixture
    def facade(self) -> ObservabilityFacade:
        """Create facade with profiling enabled."""
        config = ObservabilityConfig(
            metrics_enabled=False,
            tracing_enabled=False,
            profiling_enabled=True,
            profile_threshold_ms=10.0,
        )
        return ObservabilityFacade(config)

    def test_profile_returns_metrics(self, facade: ObservabilityFacade) -> None:
        """Test profile context manager returns metrics."""
        with facade.profile("test_op") as metrics:
            time.sleep(0.01)

        assert metrics.name == "test_op"
        assert metrics.duration_ms > 0
        assert metrics.success is True

    def test_profile_captures_error(self, facade: ObservabilityFacade) -> None:
        """Test profile captures errors."""
        with pytest.raises(ValueError):
            with facade.profile("failing_op") as metrics:
                raise ValueError("test error")

        assert metrics.success is False
        assert metrics.error == "test error"

    def test_profile_slow_operation_warning(self, facade: ObservabilityFacade) -> None:
        """Test slow operations log warning."""
        with patch("src.observability.facade.logger") as mock_logger:
            with facade.profile("slow_op"):
                time.sleep(0.02)  # 20ms > 10ms threshold

            # Should have logged warning for slow operation
            warning_calls = [c for c in mock_logger.method_calls if "warning" in str(c)]
            assert len(warning_calls) > 0


# ============================================================================
# ObservabilityFacade Tests - Decorators
# ============================================================================


class TestObservabilityFacadeDecorators:
    """Tests for ObservabilityFacade decorators."""

    @pytest.fixture
    def facade(self) -> ObservabilityFacade:
        """Create facade with disabled tracing."""
        config = ObservabilityConfig(
            metrics_enabled=False,
            tracing_enabled=False,
        )
        return ObservabilityFacade(config)

    def test_traced_decorator_sync(self, facade: ObservabilityFacade) -> None:
        """Test @traced decorator with sync function."""

        @facade.traced("test_span")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_traced_decorator_async(self, facade: ObservabilityFacade) -> None:
        """Test @traced decorator with async function."""

        @facade.traced("test_span")
        async def my_async_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await my_async_func(5)
        assert result == 10

    def test_traced_decorator_default_name(self, facade: ObservabilityFacade) -> None:
        """Test @traced uses function name as default."""

        @facade.traced()
        def my_named_func() -> str:
            return "done"

        result = my_named_func()
        assert result == "done"

    def test_profiled_decorator(self, facade: ObservabilityFacade) -> None:
        """Test @profiled decorator."""

        @facade.profiled("my_profile")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_profiled_decorator_default_name(self, facade: ObservabilityFacade) -> None:
        """Test @profiled uses function name as default."""

        @facade.profiled()
        def my_named_func() -> str:
            return "done"

        result = my_named_func()
        assert result == "done"

    def test_metered_decorator(self, facade: ObservabilityFacade) -> None:
        """Test @metered decorator."""

        @facade.metered("call_counter")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_metered_decorator_with_histogram(self, facade: ObservabilityFacade) -> None:
        """Test @metered decorator with histogram."""

        @facade.metered("call_counter", histogram_name="call_duration")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_metered_decorator_on_exception(self, facade: ObservabilityFacade) -> None:
        """Test @metered records failure on exception."""

        @facade.metered("call_counter")
        def failing_func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_func()


# ============================================================================
# Convenience Functions Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_observability(self, mock_settings: MagicMock) -> None:
        """Test get_observability returns facade."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):
            facade = get_observability()
            assert isinstance(facade, ObservabilityFacade)

    def test_get_observability_returns_singleton(self, mock_settings: MagicMock) -> None:
        """Test get_observability returns same instance."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):
            facade1 = get_observability()
            facade2 = get_observability()
            assert facade1 is facade2

    def test_traced_function(self, mock_settings: MagicMock) -> None:
        """Test traced convenience function."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):

            @traced("test_span")
            def my_func(x: int) -> int:
                return x * 2

            result = my_func(5)
            assert result == 10

    def test_profiled_function(self, mock_settings: MagicMock) -> None:
        """Test profiled convenience function."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):

            @profiled("test_profile")
            def my_func(x: int) -> int:
                return x * 2

            result = my_func(5)
            assert result == 10

    def test_metered_function(self, mock_settings: MagicMock) -> None:
        """Test metered convenience function."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):

            @metered("test_counter")
            def my_func(x: int) -> int:
                return x * 2

            result = my_func(5)
            assert result == 10


# ============================================================================
# Integration Tests
# ============================================================================


class TestObservabilityIntegration:
    """Integration tests for observability facade."""

    def test_full_workflow(self, mock_settings: MagicMock) -> None:
        """Test complete observability workflow."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):
            facade = get_observability()
            facade.set_correlation_id("test-123")

            with patch("src.observability.facade.logger") as mock_logger:
                # Simulate a traced operation
                with facade.profile("test_operation") as metrics:
                    facade.log_info("Processing started")
                    time.sleep(0.01)
                    facade.log_info("Processing completed")

                assert metrics.success is True
                assert metrics.duration_ms > 0
                assert mock_logger.info.call_count >= 2

    def test_error_handling_workflow(self, mock_settings: MagicMock) -> None:
        """Test error handling in observability workflow."""
        with patch("src.observability.facade.get_settings", return_value=mock_settings):
            facade = get_observability()

            with patch("src.observability.facade.logger"):
                with pytest.raises(ValueError):
                    with facade.profile("failing_operation") as metrics:
                        facade.log_info("Starting operation")
                        raise ValueError("Operation failed")

                assert metrics.success is False
                assert metrics.error == "Operation failed"


class TestGetLogExtra:
    """Tests for _get_log_extra method."""

    def test_adds_kwargs(self) -> None:
        """Test kwargs are added to extra dict."""
        config = ObservabilityConfig(include_correlation_id=False)
        facade = ObservabilityFacade(config)

        extra = facade._get_log_extra(key1="value1", key2="value2")

        assert extra["key1"] == "value1"
        assert extra["key2"] == "value2"

    def test_adds_correlation_id(self) -> None:
        """Test correlation ID is added when set."""
        config = ObservabilityConfig(include_correlation_id=True)
        facade = ObservabilityFacade(config)
        facade.set_correlation_id("test-corr-id")

        extra = facade._get_log_extra()

        assert extra["correlation_id"] == "test-corr-id"

    def test_no_correlation_id_when_disabled(self) -> None:
        """Test correlation ID not added when disabled."""
        config = ObservabilityConfig(include_correlation_id=False)
        facade = ObservabilityFacade(config)
        facade.set_correlation_id("test-corr-id")

        extra = facade._get_log_extra()

        assert "correlation_id" not in extra

    def test_no_correlation_id_when_not_set(self) -> None:
        """Test correlation ID not added when not set."""
        config = ObservabilityConfig(include_correlation_id=True)
        facade = ObservabilityFacade(config)

        extra = facade._get_log_extra()

        assert "correlation_id" not in extra
