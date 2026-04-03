"""
OpenTelemetry tracing infrastructure for multi-agent MCTS framework.

Provides:
- OpenTelemetry SDK integration
- Automatic span creation for key operations
- Trace context propagation
- OTLP exporter configuration from environment
- Custom attributes for MCTS metrics
- httpx instrumentation for LLM calls
"""

import functools
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .logging import get_correlation_id


class TracingManager:
    """
    Manages OpenTelemetry tracing configuration and lifecycle.

    Environment Variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: localhost:4317)
        OTEL_SERVICE_NAME: Service name for traces (default: mcts-framework)
        OTEL_EXPORTER_TYPE: Exporter type (otlp, console, none) (default: otlp)
        OTEL_TRACE_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
    """

    _instance: Optional["TracingManager"] = None
    _provider: TracerProvider | None = None

    def __init__(self):
        self._initialized = False
        self._httpx_instrumented = False

    @classmethod
    def get_instance(cls) -> "TracingManager":
        """Get singleton instance of TracingManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(
        self,
        service_name: str | None = None,
        otlp_endpoint: str | None = None,
        exporter_type: str | None = None,
        additional_resources: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize OpenTelemetry tracing.

        Args:
            service_name: Service name for traces
            otlp_endpoint: OTLP collector endpoint
            exporter_type: Type of exporter (otlp, console, none)
            additional_resources: Additional resource attributes
        """
        if self._initialized:
            return

        # Get configuration from environment or parameters
        service_name = service_name or os.environ.get("OTEL_SERVICE_NAME", "mcts-framework")
        otlp_endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
        exporter_type = exporter_type or os.environ.get("OTEL_EXPORTER_TYPE", "otlp")

        # Build resource attributes
        resource_attrs = {
            SERVICE_NAME: service_name,
            "service.version": os.environ.get("SERVICE_VERSION", "0.1.0"),
            "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
        }

        if additional_resources:
            resource_attrs.update(additional_resources)

        resource = Resource.create(resource_attrs)

        # Create tracer provider
        self._provider = TracerProvider(resource=resource)

        # Configure exporter based on type
        processor: BatchSpanProcessor | SimpleSpanProcessor | None
        if exporter_type.lower() == "otlp":
            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=os.environ.get("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
            )
            processor = BatchSpanProcessor(exporter)
        elif exporter_type.lower() == "console":
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
        elif exporter_type.lower() == "none":
            processor = None
        else:
            raise ValueError(f"Unknown exporter type: {exporter_type}")

        if processor:
            self._provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(self._provider)

        # Instrument httpx for LLM calls
        self._instrument_httpx()

        self._initialized = True

    def _instrument_httpx(self) -> None:
        """Instrument httpx client for automatic tracing of HTTP requests."""
        if self._httpx_instrumented:
            return

        try:
            HTTPXClientInstrumentor().instrument()
            self._httpx_instrumented = True
        except Exception:
            # httpx instrumentation is optional
            pass

    def shutdown(self) -> None:
        """Shutdown tracing provider."""
        if self._provider:
            self._provider.shutdown()
            self._initialized = False

    def get_tracer(self, name: str = "mcts-framework") -> trace.Tracer:
        """Get a tracer instance."""
        if not self._initialized:
            self.initialize()
        return trace.get_tracer(name)


def get_tracer(name: str = "mcts-framework") -> trace.Tracer:
    """Get a tracer instance from the global TracingManager."""
    return TracingManager.get_instance().get_tracer(name)


def add_mcts_attributes(span: Span, **attributes: Any) -> None:
    """
    Add MCTS-specific attributes to a span.

    Common attributes:
        - mcts.iteration: Current MCTS iteration number
        - mcts.node_visits: Number of visits to current node
        - mcts.node_value: Value of current node
        - mcts.ucb_score: UCB score for selection
        - mcts.exploration_weight: Exploration weight parameter
        - mcts.tree_depth: Current depth in tree
        - agent.name: Name of the agent
        - agent.confidence: Agent confidence score
    """
    for key, value in attributes.items():
        if value is not None:
            # Prefix non-standard attributes
            if not key.startswith(("mcts.", "agent.", "framework.")):
                key = f"custom.{key}"
            span.set_attribute(key, value)


@contextmanager
def trace_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
):
    """
    Context manager for creating a traced span.

    Args:
        name: Name of the span
        kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)
        attributes: Initial attributes for the span
        record_exception: Record exceptions as span events
        set_status_on_exception: Set span status to ERROR on exception

    Example:
        with trace_span("mcts.selection", attributes={"mcts.iteration": 5}) as span:
            # Perform selection
            span.set_attribute("mcts.selected_node", node_id)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes or {},
        record_exception=record_exception,
        set_status_on_exception=set_status_on_exception,
    ) as span:
        # Add correlation ID as attribute
        span.set_attribute("correlation_id", get_correlation_id())
        yield span


@asynccontextmanager
async def async_trace_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
):
    """
    Async context manager for creating a traced span.

    Same as trace_span but for async contexts.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes or {},
        record_exception=record_exception,
        set_status_on_exception=set_status_on_exception,
    ) as span:
        # Add correlation ID as attribute
        span.set_attribute("correlation_id", get_correlation_id())
        yield span


def trace_operation(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """
    Decorator for tracing function execution.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Additional attributes

    Example:
        @trace_operation(attributes={"component": "mcts"})
        async def select_best_child(node):
            ...
    """

    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, kind=kind, attributes=attributes) as span:
                # Add function arguments as attributes (limited)
                span.set_attribute("function.args_count", len(args))
                span.set_attribute("function.kwargs_count", len(kwargs))

                result = func(*args, **kwargs)

                # Mark as successful
                span.set_status(Status(StatusCode.OK))
                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with async_trace_span(span_name, kind=kind, attributes=attributes) as span:
                # Add function arguments as attributes (limited)
                span.set_attribute("function.args_count", len(args))
                span.set_attribute("function.kwargs_count", len(kwargs))

                result = await func(*args, **kwargs)

                # Mark as successful
                span.set_status(Status(StatusCode.OK))
                return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class SpanContextPropagator:
    """
    Utility for propagating trace context across service boundaries.

    Example:
        # Inject context into headers
        headers = {}
        propagator = SpanContextPropagator()
        propagator.inject(headers)

        # Extract context from headers
        context = propagator.extract(headers)
        with trace_span("operation", context=context):
            ...
    """

    def __init__(self):
        self._propagator = TraceContextTextMapPropagator()

    def inject(self, carrier: dict[str, str], context: Context | None = None) -> None:
        """Inject trace context into a carrier (e.g., HTTP headers)."""
        inject(carrier, context=context)

    def extract(self, carrier: dict[str, str]) -> Context:
        """Extract trace context from a carrier."""
        return extract(carrier)

    def get_trace_parent(self) -> str | None:
        """Get the traceparent header value for the current span."""
        carrier: dict[str, str] = {}
        self.inject(carrier)
        return carrier.get("traceparent")


def record_mcts_iteration(
    iteration: int,
    selected_node_id: str,
    ucb_score: float,
    node_visits: int,
    node_value: float,
    tree_depth: int,
) -> None:
    """
    Record MCTS iteration as a span event.

    Call this within an active span to add iteration details.
    """
    current_span = trace.get_current_span()
    if current_span:
        current_span.add_event(
            "mcts.iteration",
            attributes={
                "mcts.iteration": iteration,
                "mcts.selected_node_id": selected_node_id,
                "mcts.ucb_score": ucb_score,
                "mcts.node_visits": node_visits,
                "mcts.node_value": node_value,
                "mcts.tree_depth": tree_depth,
            },
        )


def record_agent_execution(
    agent_name: str,
    confidence: float,
    execution_time_ms: float,
    success: bool,
    error: str | None = None,
) -> None:
    """
    Record agent execution as a span event.

    Call this within an active span to add agent execution details.
    """
    current_span = trace.get_current_span()
    if current_span:
        attrs: dict[str, str | bool | int | float] = {
            "agent.name": agent_name,
            "agent.confidence": confidence,
            "agent.execution_time_ms": execution_time_ms,
            "agent.success": success,
        }
        if error:
            attrs["agent.error"] = error

        current_span.add_event("agent.execution", attributes=attrs)


# Import asyncio for decorator
import asyncio  # noqa: E402
