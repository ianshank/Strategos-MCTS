"""Configurable retry-with-backoff for LangGraph worker-node I/O boundaries.

Retry is applied to the *transient I/O call inside* a node, not to the node as a whole:
several nodes intentionally catch-and-degrade (RAG retrieval returns empty context,
synthesis falls back to the best agent output, the ADK handler returns a 0-confidence
result), so a whole-node wrapper would never observe the exception. Wrapping the I/O call
keeps each node's existing degrade/propagate behaviour, which now runs only *after* retries
are exhausted.

Reuses the existing exponential-backoff ``retry`` decorator from
``src.observability.decorators`` rather than introducing a parallel mechanism. The current
attempt count is published on a ``ContextVar`` so the execution-trace recorder can report it.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING, Any, TypeVar

from src.observability.decorators import retry as _retry_decorator

from .schema import GraphConstructionError

if TYPE_CHECKING:
    from src.config.graph_settings import GraphHardeningSettings

T = TypeVar("T")

# Nodes whose transient I/O boundary is wrapped with retry. Deterministic in-memory nodes
# (``entry``, ``route_decision``, ``aggregate_results``, ``evaluate_consensus``) are excluded.
# ``mcts_simulator`` is listed by explicit decision, but it runs the deterministic MCTS engine
# with heuristic rollouts and performs no network/LLM I/O, so no call site wraps it — its
# membership is inert and documented rather than functional.
RETRYABLE_NODES: frozenset[str] = frozenset(
    {
        "retrieve_context",
        "parallel_agents",
        "hrm_agent",
        "trm_agent",
        "symbolic_agent",
        "synthesize",
        "mcts_simulator",
    }
)


def is_retryable_node(node_name: str) -> bool:
    """Return True if a node's I/O boundary should be wrapped with retry.

    Dynamic ADK handler nodes are named ``adk_<name>`` and are all retryable.
    """
    return node_name in RETRYABLE_NODES or node_name.startswith("adk_")


@dataclass(frozen=True)
class NodeRetryPolicy:
    """Immutable retry policy for graph worker nodes."""

    enabled: bool = False
    max_attempts: int = 1
    initial_delay_seconds: float = 0.0
    backoff_factor: float = 1.0
    retryable_exceptions: tuple[type[Exception], ...] = ()


def _resolve_one(path: str) -> Any:
    """Resolve a single exception spec (bare builtin name or dotted import path)."""
    path = path.strip()
    if not path:
        raise GraphConstructionError("Empty retry exception path")
    if "." not in path:
        obj = getattr(builtins, path, None)
        if obj is None:
            raise GraphConstructionError(f"Unknown builtin exception {path!r} in retry allowlist")
        return obj
    module_path, _, attr = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise GraphConstructionError(f"Cannot import module for retry exception {path!r}: {exc}") from exc
    obj = getattr(module, attr, None)
    if obj is None:
        raise GraphConstructionError(f"Module {module_path!r} has no attribute {attr!r} for retry allowlist")
    return obj


def resolve_exception_types(paths: list[str] | tuple[str, ...]) -> tuple[type[Exception], ...]:
    """Resolve dotted/bare exception paths to a tuple of exception classes.

    Raises:
        GraphConstructionError: if a path is unresolvable or does not name an exception type.
            Resolution happens at graph construction, so a misconfigured allowlist fails fast.
    """
    resolved: list[type[Exception]] = []
    for path in paths:
        obj = _resolve_one(path)
        if not (isinstance(obj, type) and issubclass(obj, Exception)):
            raise GraphConstructionError(f"Retry allowlist entry {path!r} is not an Exception subclass")
        resolved.append(obj)
    return tuple(resolved)


def policy_from_settings(settings: GraphHardeningSettings) -> NodeRetryPolicy:
    """Build a :class:`NodeRetryPolicy` from application settings.

    Raises:
        GraphConstructionError: if the configured exception allowlist is unresolvable.
    """
    if not settings.GRAPH_NODE_RETRY_ENABLED:
        return NodeRetryPolicy(enabled=False)
    return NodeRetryPolicy(
        enabled=True,
        max_attempts=settings.GRAPH_NODE_RETRY_MAX_ATTEMPTS,
        initial_delay_seconds=settings.GRAPH_NODE_RETRY_INITIAL_DELAY_SECONDS,
        backoff_factor=settings.GRAPH_NODE_RETRY_BACKOFF_FACTOR,
        retryable_exceptions=resolve_exception_types(settings.GRAPH_NODE_RETRY_EXCEPTIONS),
    )


# Attempt count for the current node's I/O, published for the execution-trace recorder.
# 1 means "single attempt, no retry". Reset by the trace wrapper before each node; bumped
# on every retry. Note: under asyncio.gather (e.g. parallel_agents) each child runs in its
# own copied context, so retries inside a child are not visible to the parent's count.
_node_attempts: ContextVar[int] = ContextVar("graph_node_attempts", default=1)


def reset_node_attempts() -> None:
    """Reset the current-node attempt counter to 1 (single attempt)."""
    _node_attempts.set(1)


def get_node_attempts() -> int:
    """Return the number of attempts made for the current node's I/O."""
    return _node_attempts.get()


def set_node_attempts(attempts: int) -> None:
    """Publish an explicit attempt count for the current node (used to aggregate concurrent I/O)."""
    _node_attempts.set(attempts)


def _record_attempt(_exc: Exception, attempt: int) -> None:
    # on_retry(exc, attempt) fires after attempt `attempt` fails, before attempt `attempt + 1`.
    _node_attempts.set(attempt + 1)


def with_node_retry(
    policy: NodeRetryPolicy,
    node_name: str,
    fn: Callable[[], T],
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[], T]:
    """Wrap a zero-arg I/O callable with the retry policy, if applicable.

    Returns ``fn`` unchanged when retries are disabled, the allowlist is empty, or the node is
    not retryable — so non-retryable and disabled paths carry zero overhead and unchanged
    semantics. The wrapped callable is sync or async matching ``fn`` (the underlying decorator
    detects coroutine functions). ``on_retry`` overrides the default attempt recorder; concurrent
    callers (e.g. ``parallel_agents``) pass a shared aggregator because each ``asyncio`` task runs
    in a copied context where the module ContextVar would not propagate back to the parent.
    """
    if not policy.enabled or not policy.retryable_exceptions or not is_retryable_node(node_name):
        return fn
    return _retry_decorator(
        max_attempts=policy.max_attempts,
        initial_delay=policy.initial_delay_seconds,
        backoff_factor=policy.backoff_factor,
        exceptions=policy.retryable_exceptions,
        on_retry=on_retry if on_retry is not None else _record_attempt,
    )(fn)
