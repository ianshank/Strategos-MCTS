"""
Prometheus metrics collection for LangGraph Multi-Agent MCTS Framework.

This module defines and manages all Prometheus metrics for observability.
Metrics are exposed via the /metrics endpoint in the REST API.

Metrics Categories:
- Agent Performance: Request counts, latencies, confidence scores
- MCTS Operations: Iteration counts, simulation latencies
- System Health: Active operations, error rates
- LLM Integration: Request success/failure rates
"""

from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Any

from src.observability.logging import get_logger

logger = get_logger(__name__)

# Try to import Prometheus client
try:
    from prometheus_client import Counter, Gauge, Histogram, Info

    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus-client not installed. Metrics will not be collected.")
    PROMETHEUS_AVAILABLE = False

    # Define dummy classes for when Prometheus is not available
    class DummyMetric:
        def __init__(self, *_args, **_kwargs):
            pass

        def labels(self, *_args, **_kwargs):
            return self

        def inc(self, *_args, **_kwargs):
            pass

        def dec(self, *_args, **_kwargs):
            pass

        def set(self, *_args, **_kwargs):
            pass

        def observe(self, *_args, **_kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

    Counter = Gauge = Histogram = Info = DummyMetric


# ============================================================================
# Agent Metrics
# ============================================================================

AGENT_REQUESTS_TOTAL = Counter(
    "mcts_agent_requests_total",
    "Total number of requests processed by each agent type",
    ["agent_type", "status"],
)

AGENT_REQUEST_LATENCY = Histogram(
    "mcts_agent_request_latency_seconds",
    "Latency of agent processing in seconds",
    ["agent_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")),
)

AGENT_CONFIDENCE_SCORES = Histogram(
    "mcts_agent_confidence_score",
    "Distribution of confidence scores by agent",
    ["agent_type"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ============================================================================
# MCTS Metrics
# ============================================================================

MCTS_ITERATIONS_TOTAL = Counter(
    "mcts_iterations_total",
    "Total number of MCTS iterations completed",
    ["outcome"],
)

MCTS_ITERATION_LATENCY = Histogram(
    "mcts_iteration_latency_seconds",
    "Latency per MCTS iteration",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")),
)

MCTS_SIMULATION_DEPTH = Histogram(
    "mcts_simulation_depth",
    "Depth of MCTS tree simulations",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, float("inf")),
)

MCTS_NODE_COUNT = Gauge(
    "mcts_active_nodes",
    "Number of active nodes in MCTS tree",
)

MCTS_BEST_ACTION_CONFIDENCE = Histogram(
    "mcts_best_action_confidence",
    "Confidence score of MCTS best action selection",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ============================================================================
# System Health Metrics
# ============================================================================

ACTIVE_OPERATIONS = Gauge(
    "mcts_active_operations",
    "Number of currently active operations",
    ["operation_type"],
)

LLM_REQUEST_ERRORS = Counter(
    "mcts_llm_request_errors_total",
    "Total number of LLM request errors",
    ["provider", "error_type"],
)

LLM_REQUEST_LATENCY = Histogram(
    "mcts_llm_request_latency_seconds",
    "Latency of LLM API requests",
    ["provider"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

LLM_TOKEN_USAGE = Counter(
    "mcts_llm_tokens_total",
    "Total number of tokens consumed",
    ["provider", "token_type"],
)

# ============================================================================
# RAG Metrics
# ============================================================================

RAG_QUERIES_TOTAL = Counter(
    "mcts_rag_queries_total",
    "Total number of RAG queries",
    ["status"],
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "mcts_rag_retrieval_latency_seconds",
    "Latency of RAG context retrieval",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf")),
)

RAG_DOCUMENTS_RETRIEVED = Histogram(
    "mcts_rag_documents_retrieved",
    "Number of documents retrieved per RAG query",
    buckets=(0, 1, 5, 10, 20, 50, 100, float("inf")),
)

RAG_RELEVANCE_SCORES = Histogram(
    "mcts_rag_relevance_score",
    "Relevance scores of retrieved documents",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ============================================================================
# Request Metrics (API Level)
# ============================================================================

REQUEST_COUNT = Counter(
    "mcts_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "mcts_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")),
)

ACTIVE_REQUESTS = Gauge(
    "mcts_active_requests",
    "Number of active API requests",
)

ERROR_COUNT = Counter(
    "mcts_errors_total",
    "Total number of errors",
    ["error_type"],
)

# ============================================================================
# Rate Limiting Metrics
# ============================================================================

RATE_LIMIT_EXCEEDED = Counter(
    "mcts_rate_limit_exceeded_total",
    "Number of rate limit violations",
    ["client_id"],
)

REQUEST_QUEUE_DEPTH = Gauge(
    "mcts_request_queue_depth",
    "Current depth of request queue",
)

# ============================================================================
# System Resource Metrics
# ============================================================================

SYSTEM_INFO = Info(
    "mcts_system_info",
    "System information",
)


# ============================================================================
# Utility Functions and Decorators
# ============================================================================


def setup_metrics(app_version: str = "1.0.0", environment: str = "production") -> None:
    """
    Initialize metrics with system information.

    Args:
        app_version: Application version
        environment: Deployment environment (production, staging, development)
    """
    if PROMETHEUS_AVAILABLE:
        SYSTEM_INFO.info(
            {
                "version": app_version,
                "environment": environment,
                "framework": "langgraph-mcts",
            }
        )
        logger.info(f"Prometheus metrics initialized (version={app_version}, env={environment})")
    else:
        logger.warning("Prometheus not available - metrics disabled")


@contextmanager
def track_operation(operation_type: str):
    """
    Context manager to track active operations.

    Usage:
        with track_operation("mcts_simulation"):
            # Your operation here
            pass
    """
    ACTIVE_OPERATIONS.labels(operation_type=operation_type).inc()
    try:
        yield
    finally:
        ACTIVE_OPERATIONS.labels(operation_type=operation_type).dec()


@contextmanager
def measure_latency(metric: Histogram, **labels):
    """
    Context manager to measure operation latency.

    Args:
        metric: Prometheus Histogram metric
        **labels: Metric labels

    Usage:
        with measure_latency(AGENT_REQUEST_LATENCY, agent_type="hrm"):
            # Your operation here
            pass
    """
    start_time = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start_time
        if labels:
            metric.labels(**labels).observe(elapsed)
        else:
            metric.observe(elapsed)


def track_agent_request(agent_type: str):
    """
    Decorator to track agent request metrics.

    Usage:
        @track_agent_request("hrm")
        def process_query(query: str):
            return result
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = perf_counter()
            status = "success"

            with track_operation(f"agent_{agent_type}"):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                    raise
                finally:
                    elapsed = perf_counter() - start_time
                    AGENT_REQUESTS_TOTAL.labels(agent_type=agent_type, status=status).inc()
                    AGENT_REQUEST_LATENCY.labels(agent_type=agent_type).observe(elapsed)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = perf_counter()
            status = "success"

            with track_operation(f"agent_{agent_type}"):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                    raise
                finally:
                    elapsed = perf_counter() - start_time
                    AGENT_REQUESTS_TOTAL.labels(agent_type=agent_type, status=status).inc()
                    AGENT_REQUEST_LATENCY.labels(agent_type=agent_type).observe(elapsed)

        # Return appropriate wrapper based on function type
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_mcts_iteration(func: Callable) -> Callable:
    """
    Decorator to track MCTS iteration metrics.

    Usage:
        @track_mcts_iteration
        def run_iteration():
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = perf_counter()
        outcome = "completed"

        try:
            result = func(*args, **kwargs)
            return result
        except TimeoutError:
            outcome = "timeout"
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            elapsed = perf_counter() - start_time
            MCTS_ITERATIONS_TOTAL.labels(outcome=outcome).inc()
            MCTS_ITERATION_LATENCY.observe(elapsed)

    return wrapper


def record_confidence_score(agent_type: str, score: float) -> None:
    """
    Record a confidence score for an agent.

    Args:
        agent_type: Type of agent (hrm, trm, etc.)
        score: Confidence score (0.0 to 1.0)
    """
    if 0.0 <= score <= 1.0:
        AGENT_CONFIDENCE_SCORES.labels(agent_type=agent_type).observe(score)
    else:
        logger.warning(f"Invalid confidence score: {score} (must be 0.0-1.0)")


def record_llm_usage(provider: str, prompt_tokens: int, completion_tokens: int) -> None:
    """
    Record LLM token usage.

    Args:
        provider: LLM provider name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
    """
    LLM_TOKEN_USAGE.labels(provider=provider, token_type="prompt").inc(prompt_tokens)
    LLM_TOKEN_USAGE.labels(provider=provider, token_type="completion").inc(completion_tokens)


def record_rag_retrieval(num_docs: int, relevance_scores: list[float], latency: float) -> None:
    """
    Record RAG retrieval metrics.

    Args:
        num_docs: Number of documents retrieved
        relevance_scores: List of relevance scores
        latency: Retrieval latency in seconds
    """
    RAG_QUERIES_TOTAL.labels(status="success").inc()
    RAG_DOCUMENTS_RETRIEVED.observe(num_docs)
    RAG_RETRIEVAL_LATENCY.observe(latency)

    for score in relevance_scores:
        if 0.0 <= score <= 1.0:
            RAG_RELEVANCE_SCORES.observe(score)


# ============================================================================
# Health Check Utilities
# ============================================================================


def get_metrics_summary() -> dict[str, Any]:
    """
    Get a summary of current metrics for health checks.

    Returns:
        Dictionary with metric summaries
    """
    return {
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_initialized": True,
    }


if __name__ == "__main__":
    # Demo usage
    setup_metrics(app_version="1.0.0", environment="development")

    # Simulate some metrics
    AGENT_REQUESTS_TOTAL.labels(agent_type="hrm", status="success").inc()
    AGENT_REQUEST_LATENCY.labels(agent_type="hrm").observe(1.5)
    record_confidence_score("hrm", 0.85)

    MCTS_ITERATIONS_TOTAL.labels(outcome="completed").inc(100)
    MCTS_ITERATION_LATENCY.observe(0.05)

    print("Metrics recorded successfully")
    print(f"Prometheus available: {PROMETHEUS_AVAILABLE}")
