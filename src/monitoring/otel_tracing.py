"""
OpenTelemetry distributed tracing for LangGraph Multi-Agent MCTS Framework.

This module configures OpenTelemetry for distributed tracing across:
- Multi-agent workflows
- MCTS simulations
- LLM API calls
- External service integrations

Traces are exported to Jaeger via OTLP (OpenTelemetry Protocol).
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

from src.config.constants import DEFAULT_OTLP_HTTP_ENDPOINT
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not installed. Tracing will not be available.")
    OTEL_AVAILABLE = False

    # Define dummy classes
    class DummyTracer:
        def start_as_current_span(self, *_args, **_kwargs):
            @contextmanager
            def dummy_context():
                yield None

            return dummy_context()

    class DummyTracerProvider:
        def get_tracer(self, *_args, **_kwargs):
            return DummyTracer()

    trace = type("trace", (), {"get_tracer_provider": lambda: DummyTracerProvider()})()
    Status = type("Status", (), {})()
    StatusCode = type("StatusCode", (), {"OK": "OK", "ERROR": "ERROR"})()


# Global tracer instance
_tracer: Any | None = None
_initialized: bool = False


def setup_tracing(
    service_name: str = "mcts-framework",
    environment: str = "production",
    otlp_endpoint: str | None = None,
    enable_httpx_instrumentation: bool = True,
) -> None:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for trace identification
        environment: Deployment environment
        otlp_endpoint: OTLP collector endpoint (default: env var or http://localhost:4317)
        enable_httpx_instrumentation: Auto-instrument httpx client
    """
    global _tracer, _initialized

    if _initialized:
        logger.info("Tracing already initialized")
        return

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - tracing disabled")
        return

    try:
        # Get OTLP endpoint from environment or parameter
        endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_HTTP_ENDPOINT)

        # Create resource with service information
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": os.getenv("APP_VERSION", "1.0.0"),
                "deployment.environment": environment,
                "telemetry.sdk.language": "python",
                "telemetry.sdk.name": "opentelemetry",
            }
        )

        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            insecure=True,  # Use secure=False for HTTP (not HTTPS)
        )

        # Create tracer provider with resource
        provider = TracerProvider(resource=resource)

        # Add batch span processor with OTLP exporter
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer instance
        _tracer = trace.get_tracer(__name__)

        # Auto-instrument httpx if enabled
        if enable_httpx_instrumentation:
            try:
                HTTPXClientInstrumentor().instrument()
                logger.info("HTTPX instrumentation enabled")
            except Exception as e:
                logger.warning(f"Could not instrument httpx: {e}")

        _initialized = True
        logger.info(f"OpenTelemetry tracing initialized (service={service_name}, endpoint={endpoint})")

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        _tracer = None


def get_tracer():
    """Get the global tracer instance."""
    global _tracer

    if _tracer is None and OTEL_AVAILABLE:
        # Lazy initialization
        setup_tracing()

    return _tracer or DummyTracer()


# ============================================================================
# Tracing Decorators
# ============================================================================


def trace_operation(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
):
    """
    Decorator to trace a function or method as a span.

    Args:
        name: Span name (defaults to function name)
        attributes: Additional span attributes
        record_exception: Whether to record exceptions in the span

    Usage:
        @trace_operation(name="process_query", attributes={"agent": "hrm"})
        async def process_query(query: str):
            return result
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        span_attributes = attributes or {}

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                for key, value in span_attributes.items():
                    if span:
                        span.set_attribute(key, value)

                # Add function info
                if span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                try:
                    result = await func(*args, **kwargs)

                    # Mark span as successful
                    if span and OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    # Record exception
                    if span and record_exception and OTEL_AVAILABLE:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                for key, value in span_attributes.items():
                    if span:
                        span.set_attribute(key, value)

                # Add function info
                if span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)

                    # Mark span as successful
                    if span and OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    # Record exception
                    if span and record_exception and OTEL_AVAILABLE:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Return appropriate wrapper
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """
    Context manager to create a traced span.

    Args:
        name: Span name
        attributes: Span attributes

    Usage:
        with trace_span("mcts_simulation", {"iterations": 100}):
            # Your operation here
            pass
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        # Add attributes
        if span and attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span

            # Mark as successful
            if span and OTEL_AVAILABLE:
                span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Record exception
            if span and OTEL_AVAILABLE:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def add_span_attribute(key: str, value: Any) -> None:
    """
    Add an attribute to the current active span.

    Args:
        key: Attribute key
        value: Attribute value
    """
    if OTEL_AVAILABLE:
        span = trace.get_current_span()
        if span:
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """
    Add an event to the current active span.

    Args:
        name: Event name
        attributes: Event attributes
    """
    if OTEL_AVAILABLE:
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes or {})


# ============================================================================
# Agent-Specific Tracing
# ============================================================================


def trace_agent_operation(agent_type: str):
    """
    Decorator specifically for agent operations.

    Args:
        agent_type: Type of agent (hrm, trm, etc.)
    """
    return trace_operation(
        name=f"agent.{agent_type}",
        attributes={
            "agent.type": agent_type,
            "component": "agent",
        },
    )


def trace_mcts_operation(operation_name: str):
    """
    Decorator specifically for MCTS operations.

    Args:
        operation_name: MCTS operation name (selection, expansion, simulation, backpropagation)
    """
    return trace_operation(
        name=f"mcts.{operation_name}",
        attributes={
            "mcts.operation": operation_name,
            "component": "mcts",
        },
    )


def trace_llm_call(provider: str):
    """
    Decorator specifically for LLM API calls.

    Args:
        provider: LLM provider name (openai, anthropic, lmstudio)
    """
    return trace_operation(
        name=f"llm.{provider}",
        attributes={
            "llm.provider": provider,
            "component": "llm",
        },
    )


def trace_rag_operation(operation_name: str):
    """
    Decorator specifically for RAG operations.

    Args:
        operation_name: RAG operation (query, retrieval, rerank)
    """
    return trace_operation(
        name=f"rag.{operation_name}",
        attributes={
            "rag.operation": operation_name,
            "component": "rag",
        },
    )


# ============================================================================
# Trace Context Propagation
# ============================================================================


def get_trace_context() -> dict[str, str]:
    """
    Get the current trace context for propagation.

    Returns:
        Dictionary with trace context headers
    """
    if not OTEL_AVAILABLE:
        return {}

    from opentelemetry.propagate import inject

    headers: dict[str, str] = {}
    inject(headers)
    return headers


def set_trace_context(context: dict[str, str]) -> None:
    """
    Set trace context from propagated headers.

    Args:
        context: Trace context headers
    """
    if not OTEL_AVAILABLE:
        return

    from opentelemetry.propagate import extract

    extract(context)


# ============================================================================
# Health Check Utilities
# ============================================================================


def get_tracing_status() -> dict[str, Any]:
    """
    Get tracing system status.

    Returns:
        Dictionary with tracing status information
    """
    return {
        "otel_available": OTEL_AVAILABLE,
        "initialized": _initialized,
        "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_HTTP_ENDPOINT),
    }


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    # Initialize tracing
    setup_tracing(service_name="mcts-framework-demo", environment="development")

    # Example traced function
    @trace_operation(name="example_operation", attributes={"example": "true"})
    def example_function(x: int) -> int:
        add_span_event("processing_start", {"input": x})
        result = x * 2
        add_span_attribute("result", result)
        add_span_event("processing_complete", {"output": result})
        return result

    # Call traced function
    result = example_function(42)
    print(f"Result: {result}")
    print(f"Tracing status: {get_tracing_status()}")
