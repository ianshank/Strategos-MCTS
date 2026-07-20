"""
Unified Observability Facade.

Provides a single entry point for all observability features:
- Logging (structured JSON logging)
- Metrics (Prometheus-compatible)
- Tracing (OpenTelemetry)
- Profiling (performance tracking)

This facade consolidates functionality from:
- src/observability/* modules
- src/monitoring/* modules

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 11
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from src.config.settings import get_settings
from src.observability.logging import sanitize_dict

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ObservabilityConfig:
    """Configuration for observability features."""

    # Logging
    log_level: str = "INFO"
    json_logging: bool = True
    include_correlation_id: bool = True

    # Metrics
    metrics_enabled: bool = True
    metrics_prefix: str = "mcts"
    metrics_port: int = 9090

    # Tracing
    tracing_enabled: bool = True
    trace_sample_rate: float = 1.0
    service_name: str = "langgraph-mcts"

    # Profiling
    profiling_enabled: bool = False
    profile_threshold_ms: float = 100.0

    @classmethod
    def from_settings(cls) -> ObservabilityConfig:
        """Create config from settings."""
        settings = get_settings()
        return cls(
            log_level=getattr(settings, "LOG_LEVEL", "INFO"),
            json_logging=getattr(settings, "JSON_LOGGING", True),
            include_correlation_id=True,
            metrics_enabled=getattr(settings, "METRICS_ENABLED", True),
            metrics_prefix=getattr(settings, "METRICS_PREFIX", "mcts"),
            metrics_port=getattr(settings, "METRICS_PORT", 9090),
            tracing_enabled=getattr(settings, "TRACING_ENABLED", True),
            trace_sample_rate=getattr(settings, "TRACE_SAMPLE_RATE", 1.0),
            service_name=getattr(settings, "SERVICE_NAME", "langgraph-mcts"),
            profiling_enabled=getattr(settings, "PROFILING_ENABLED", False),
            profile_threshold_ms=getattr(settings, "PROFILE_THRESHOLD_MS", 100.0),
        )


@dataclass
class OperationMetrics:
    """Metrics collected for an operation."""

    name: str
    duration_ms: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "operation_name": self.name,  # Use operation_name to avoid LogRecord conflict
            "duration_ms": round(self.duration_ms, 3),
            "success": self.success,
            "timestamp": self.timestamp,
        }
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class ObservabilityFacade:
    """
    Unified facade for observability features.

    Provides methods for:
    - Structured logging with correlation IDs
    - Metric recording (counters, gauges, histograms)
    - Distributed tracing with spans
    - Performance profiling

    Example:
        >>> obs = ObservabilityFacade.get_instance()
        >>> with obs.trace("process_request"):
        ...     obs.log_info("Processing", request_id="123")
        ...     obs.record_counter("requests_total")
    """

    _instance: ObservabilityFacade | None = None
    _initialized: bool = False

    def __init__(self, config: ObservabilityConfig | None = None):
        """Initialize observability facade."""
        self.config = config or ObservabilityConfig.from_settings()
        self._correlation_id: str | None = None

        # Lazy-load modules
        self._tracer = None
        # Latch: once tracer initialization fails (broken/misconfigured backend),
        # stop retrying on every trace() call so a hot path can't flood logs.
        # Cleared only by constructing a fresh instance (e.g. via reset()).
        self._tracer_init_failed = False
        self._metrics: Any = None
        self._profiler = None

    @classmethod
    def get_instance(cls, config: ObservabilityConfig | None = None) -> ObservabilityFacade:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
        cls._initialized = False

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for request tracking."""
        self._correlation_id = correlation_id

    def _get_log_extra(self, **kwargs: Any) -> dict[str, Any]:
        """Build extra dict for logging."""
        extra = dict(kwargs)
        if self.config.include_correlation_id and self._correlation_id:
            extra["correlation_id"] = self._correlation_id
        return extra

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        logger.debug(message, extra=self._get_log_extra(**kwargs))

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        logger.info(message, extra=self._get_log_extra(**kwargs))

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        logger.warning(message, extra=self._get_log_extra(**kwargs))

    def log_error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message with context."""
        logger.error(message, exc_info=exc_info, extra=self._get_log_extra(**kwargs))

    def log_operation(self, metrics: OperationMetrics) -> None:
        """Log operation metrics."""
        log_level = logging.INFO if metrics.success else logging.WARNING
        logger.log(
            log_level,
            f"Operation {metrics.name} completed",
            extra=self._get_log_extra(**metrics.to_dict()),
        )

    # =========================================================================
    # Metrics Methods
    # =========================================================================

    def _ensure_metrics(self) -> None:
        """Lazy-load metrics module."""
        if self._metrics is None and self.config.metrics_enabled:
            try:
                from src.observability.metrics import MetricsCollector

                self._metrics = MetricsCollector.get_instance()
            except ImportError:
                logger.debug("Metrics module not available")
                self._metrics = {}

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record counter metric.

        Args:
            name: Metric name
            value: Value to add (default 1)
            labels: Optional metric labels
        """
        if not self.config.metrics_enabled:
            return

        self._ensure_metrics()
        metric_name = f"{self.config.metrics_prefix}_{name}"

        try:
            if hasattr(self._metrics, "get"):
                counter = self._metrics.get(metric_name)
                if counter:
                    if labels:
                        counter.labels(**labels).inc(value)
                    else:
                        counter.inc(value)
        except Exception as e:
            logger.debug(f"Failed to record counter {metric_name}: {e}")

    def record_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional metric labels
        """
        if not self.config.metrics_enabled:
            return

        self._ensure_metrics()
        metric_name = f"{self.config.metrics_prefix}_{name}"

        try:
            if hasattr(self._metrics, "get"):
                gauge = self._metrics.get(metric_name)
                if gauge:
                    if labels:
                        gauge.labels(**labels).set(value)
                    else:
                        gauge.set(value)
        except Exception as e:
            logger.debug(f"Failed to record gauge {metric_name}: {e}")

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record histogram metric.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional metric labels
        """
        if not self.config.metrics_enabled:
            return

        self._ensure_metrics()
        metric_name = f"{self.config.metrics_prefix}_{name}"

        try:
            if hasattr(self._metrics, "get"):
                histogram = self._metrics.get(metric_name)
                if histogram:
                    if labels:
                        histogram.labels(**labels).observe(value)
                    else:
                        histogram.observe(value)
        except Exception as e:
            logger.debug(f"Failed to record histogram {metric_name}: {e}")

    # =========================================================================
    # Tracing Methods
    # =========================================================================

    def _ensure_tracer(self) -> None:
        """Lazy-load the tracing module, degrading gracefully on failure.

        The tracer is optional: a missing module, or a misconfigured/unavailable
        backend that raises during initialization, must never break the operation
        being traced. On any failure ``self._tracer`` stays ``None``, the failure
        is latched (so subsequent calls skip re-initialization and don't flood
        logs), and callers run without a span.
        """
        if self._tracer is not None or self._tracer_init_failed or not self.config.tracing_enabled:
            return
        try:
            from src.observability.tracing import get_tracer

            self._tracer = get_tracer(self.config.service_name)
        except ImportError:
            self._tracer_init_failed = True
            logger.debug("Tracing module not available; tracing disabled for this instance")
        except Exception as exc:
            self._tracer_init_failed = True
            self._tracer = None
            self.log_warning(
                "tracer initialization failed; tracing disabled for this instance",
                **sanitize_dict({"error": str(exc)}),
            )

    def _start_span_safely(
        self,
        name: str,
        attributes: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        """Begin a tracing span, degrading to a no-op on backend failure.

        Returns a ``(span_context_manager, span)`` tuple. When tracing is
        disabled, or the tracing backend raises (e.g. a misconfigured or
        unavailable exporter), both elements are ``None`` and the caller runs
        the wrapped operation without a span rather than failing.

        Args:
            name: Span name.
            attributes: Optional span attributes.

        Returns:
            Tuple of the entered span context manager (or ``None``) and the
            active span object (or ``None``).
        """
        if self._tracer is None:
            return None, None

        span_cm = None
        try:
            span_cm = self._tracer.start_as_current_span(name)
            span = span_cm.__enter__()
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            return span_cm, span
        except Exception as exc:
            self.log_warning(
                "tracing backend error starting span; continuing without tracing",
                **sanitize_dict({"span": name, "error": str(exc)}),
            )
            if span_cm is not None:
                self._safe_exit_span(span_cm, exc)
            return None, None

    def _safe_exit_span(self, span_cm: Any, exc: BaseException | None) -> None:
        """Exit a span context manager, swallowing tracing-layer errors.

        Args:
            span_cm: The span context manager previously entered.
            exc: The exception propagating through the body, or ``None`` on the
                success path.
        """
        try:
            if exc is None:
                span_cm.__exit__(None, None, None)
            else:
                span_cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception as exit_exc:
            self.log_warning(
                "failed to close trace span cleanly",
                **sanitize_dict({"error": str(exit_exc)}),
            )

    @contextmanager
    def trace(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ):
        """
        Context manager for tracing a code block.

        Args:
            name: Span name
            attributes: Optional span attributes

        Yields:
            Span context (or None if tracing disabled)
        """
        if not self.config.tracing_enabled:
            yield None
            return

        self._ensure_tracer()

        start_time = time.perf_counter()
        error_occurred = False
        error_message = None

        # Start a span, but degrade gracefully when the tracing backend is
        # unavailable or misconfigured: observability must never break the
        # operation it wraps. Only exceptions raised by the wrapped body
        # (the ``yield``) propagate to the caller.
        span_cm, span = self._start_span_safely(name, attributes)

        try:
            yield span
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            if span_cm is not None:
                self._safe_exit_span(span_cm, e)
                span_cm = None
            raise
        finally:
            if span_cm is not None:
                self._safe_exit_span(span_cm, None)
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_debug(
                f"Span {name} completed",
                duration_ms=duration_ms,
                error=error_message,
            )

            # Record histogram for span duration
            self.record_histogram(
                "span_duration_seconds",
                duration_ms / 1000,
                labels={"span": name, "error": str(error_occurred)},
            )

    @asynccontextmanager
    async def trace_async(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ):
        """
        Async context manager for tracing.

        Args:
            name: Span name
            attributes: Optional span attributes

        Yields:
            Span context (or None if tracing disabled)
        """
        if not self.config.tracing_enabled:
            yield None
            return

        self._ensure_tracer()

        start_time = time.perf_counter()
        error_message = None

        # Same graceful-degradation contract as the sync ``trace``: a broken
        # tracing backend must not break the awaited body.
        span_cm, span = self._start_span_safely(name, attributes)

        try:
            yield span
        except Exception as e:
            error_message = str(e)
            if span_cm is not None:
                self._safe_exit_span(span_cm, e)
                span_cm = None
            raise
        finally:
            if span_cm is not None:
                self._safe_exit_span(span_cm, None)
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_debug(
                f"Async span {name} completed",
                duration_ms=duration_ms,
                error=error_message,
            )

    # =========================================================================
    # Profiling Methods
    # =========================================================================

    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling a code block.

        Args:
            name: Profile block name

        Yields:
            OperationMetrics for the profiled block
        """
        start_time = time.perf_counter()
        metrics = OperationMetrics(name=name, duration_ms=0, success=True)

        try:
            yield metrics
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
        finally:
            metrics.duration_ms = (time.perf_counter() - start_time) * 1000

            if self.config.profiling_enabled and metrics.duration_ms > self.config.profile_threshold_ms:
                self.log_warning(
                    f"Slow operation: {name}",
                    duration_ms=metrics.duration_ms,
                    threshold_ms=self.config.profile_threshold_ms,
                )

            self.log_operation(metrics)

    # =========================================================================
    # Decorators
    # =========================================================================

    def traced(self, name: str | None = None) -> Callable[[F], F]:
        """
        Decorator for tracing function calls.

        Args:
            name: Optional span name (defaults to function name)

        Returns:
            Decorated function
        """

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.trace(span_name):
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.trace_async(span_name):
                    return await func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator

    def profiled(self, name: str | None = None) -> Callable[[F], F]:
        """
        Decorator for profiling function calls.

        Args:
            name: Optional profile name (defaults to function name)

        Returns:
            Decorated function
        """

        def decorator(func: F) -> F:
            profile_name = name or func.__name__

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.profile(profile_name):
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # ``profile`` only measures time around the body, so the sync
                # context manager is safe here: entering/exiting it inside the
                # coroutine brackets the awaited execution, not the coroutine
                # object's creation.
                with self.profile(profile_name) as metrics:
                    try:
                        return await func(*args, **kwargs)
                    except BaseException as e:
                        # ``profile`` marks failure only for ``Exception``, but a
                        # cancelled coroutine raises ``asyncio.CancelledError``
                        # (a ``BaseException``) and must not be logged as success.
                        # For plain ``Exception``s, ``profile``'s own handler then
                        # overwrites ``error`` with ``str(e)``, keeping its contract.
                        metrics.success = False
                        metrics.error = type(e).__name__
                        raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator

    def metered(
        self,
        counter_name: str,
        histogram_name: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> Callable[[F], F]:
        """
        Decorator for metering function calls.

        Records:
        - Counter for number of calls
        - Histogram for call duration (if histogram_name provided)

        Args:
            counter_name: Counter metric name
            histogram_name: Optional histogram metric name
            labels: Optional metric labels

        Returns:
            Decorated function
        """

        def decorator(func: F) -> F:
            def _record(success: bool, start_time: float) -> None:
                duration = time.perf_counter() - start_time

                # Record counter
                call_labels = dict(labels or {})
                call_labels["success"] = str(success)
                self.record_counter(counter_name, labels=call_labels)

                # Record histogram
                if histogram_name:
                    self.record_histogram(histogram_name, duration, labels=call_labels)

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                success = True

                try:
                    return func(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    _record(success, start_time)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                success = True

                try:
                    return await func(*args, **kwargs)
                except BaseException:
                    # BaseException, not Exception: a cancelled coroutine raises
                    # asyncio.CancelledError (a BaseException) and must not be
                    # counted as a success.
                    success = False
                    raise
                finally:
                    _record(success, start_time)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator


# Convenience functions for direct access
def get_observability() -> ObservabilityFacade:
    """Get the observability facade singleton."""
    return ObservabilityFacade.get_instance()


def traced(name: str | None = None) -> Callable[[F], F]:
    """Decorator for tracing function calls."""
    return get_observability().traced(name)


def profiled(name: str | None = None) -> Callable[[F], F]:
    """Decorator for profiling function calls."""
    return get_observability().profiled(name)


def metered(
    counter_name: str,
    histogram_name: str | None = None,
    labels: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """Decorator for metering function calls."""
    return get_observability().metered(counter_name, histogram_name, labels)


__all__ = [
    "ObservabilityConfig",
    "ObservabilityFacade",
    "OperationMetrics",
    "get_observability",
    "traced",
    "profiled",
    "metered",
]
