"""Structured execution-trace logging for LangGraph node transitions.

Every node transition emits a :class:`NodeTraceEvent` (run id, thread id, node, monotonic
sequence, timestamp, input/output digests, duration, status, attempt count). Events always
flow to the structured logger and per-node timings to ``MetricsCollector.record_node_timing``;
when a trace directory is configured they are additionally appended to a per-run JSONL file
from which :func:`load_trace` / :func:`reconstruct_path` recover the ordered execution path.

The recorder is deliberately synchronous so a single wrapping seam can instrument both sync
and async node handlers; JSONL appends use the same atomic ``O_APPEND`` idiom as the harness
memory log, so a torn trailing line (e.g. from SIGKILL) is tolerated on read.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import itertools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.constants import DEFAULT_TRACE_DIGEST_HEX_CHARS, GRAPH_TRACE_LOGGER_NAME
from src.observability.logging import get_structured_logger
from src.utils.jsonl import append_jsonl, iter_jsonl

from .retry import get_node_attempts, reset_node_attempts

TRACE_SCHEMA_VERSION = 1

# Run/thread identity for the currently executing graph, set by IntegratedFramework before
# invocation so the per-node wrappers can attribute each transition without threading args.
_current_run_id: ContextVar[str] = ContextVar("graph_trace_run_id", default="")
_current_thread_id: ContextVar[str] = ContextVar("graph_trace_thread_id", default="")


def set_trace_context(run_id: str, thread_id: str) -> None:
    """Bind the current run id and thread id for node-transition attribution."""
    _current_run_id.set(run_id)
    _current_thread_id.set(thread_id)


def snapshot_trace_context() -> tuple[str, str]:
    """Capture the current ``(run_id, thread_id)`` so it can be restored later."""
    return _current_run_id.get(), _current_thread_id.get()


def current_run_id() -> str:
    return _current_run_id.get()


@dataclass(frozen=True)
class NodeTraceEvent:
    """One graph node transition."""

    run_id: str
    thread_id: str
    node: str
    seq: int
    ts: str
    duration_ms: float
    status: str  # "ok" | "error"
    attempts: int
    input_digest: str
    output_digest: str | None
    error: str | None
    schema_version: int = field(default=TRACE_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "node": self.node,
            "seq": self.seq,
            "ts": self.ts,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attempts": self.attempts,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "error": self.error,
        }


def event_from_dict(data: Mapping[str, Any]) -> NodeTraceEvent:
    """Reconstruct a :class:`NodeTraceEvent` from a persisted record."""
    return NodeTraceEvent(
        run_id=data["run_id"],
        thread_id=data["thread_id"],
        node=data["node"],
        seq=int(data["seq"]),
        ts=data["ts"],
        duration_ms=float(data["duration_ms"]),
        status=data["status"],
        attempts=int(data["attempts"]),
        input_digest=data["input_digest"],
        output_digest=data.get("output_digest"),
        error=data.get("error"),
        schema_version=int(data.get("schema_version", TRACE_SCHEMA_VERSION)),
    )


def state_digest(state: Any) -> str:
    """Return a short, order-invariant sha256 digest of a state mapping.

    ``default=str`` makes arbitrary (non-JSON) values safe; ``sort_keys`` makes the digest
    invariant to key ordering so equal states digest equally.
    """
    payload = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:DEFAULT_TRACE_DIGEST_HEX_CHARS]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class JsonlTraceSink:
    """Append-only per-run JSONL sink (one ``<run_id>.jsonl`` file per run)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def append(self, run_id: str, record: Mapping[str, Any]) -> None:
        append_jsonl(self.root / f"{run_id}.jsonl", record)


class GraphTraceRecorder:
    """Records node transitions to the structured logger, metrics, and an optional sink."""

    def __init__(
        self,
        sink: JsonlTraceSink | None = None,
        clock: Callable[[], datetime] = _utc_now,
        metrics: Any | None = None,
        slog: Any | None = None,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._metrics = metrics
        self._slog = slog or get_structured_logger(GRAPH_TRACE_LOGGER_NAME)
        self._counters: dict[str, itertools.count[int]] = {}

    def begin_run(self, run_id: str) -> None:
        """Start a fresh monotonic sequence counter for a run."""
        self._counters[run_id] = itertools.count()

    def end_run(self, run_id: str) -> None:
        """Drop a run's sequence counter."""
        self._counters.pop(run_id, None)

    def _next_seq(self, run_id: str) -> int:
        counter = self._counters.get(run_id)
        if counter is None:
            counter = itertools.count()
            self._counters[run_id] = counter
        return next(counter)

    def record(
        self,
        *,
        run_id: str,
        thread_id: str,
        node: str,
        duration_ms: float,
        status: str,
        attempts: int,
        input_digest: str,
        output_digest: str | None,
        error: str | None,
    ) -> NodeTraceEvent:
        event = NodeTraceEvent(
            run_id=run_id,
            thread_id=thread_id,
            node=node,
            seq=self._next_seq(run_id),
            ts=self._clock().isoformat(),
            duration_ms=duration_ms,
            status=status,
            attempts=attempts,
            input_digest=input_digest,
            output_digest=output_digest,
            error=error,
        )
        self._slog.info("graph.node.transition", **event.to_dict())
        if self._metrics is not None:
            self._metrics.record_node_timing(node, duration_ms)
        if self._sink is not None:
            self._sink.append(run_id, event.to_dict())
        return event


def _finish(
    recorder: GraphTraceRecorder, node: str, start: float, state: Any, result: Any, error: BaseException | None
) -> None:
    duration_ms = (time.perf_counter() - start) * 1000.0
    recorder.record(
        run_id=_current_run_id.get(),
        thread_id=_current_thread_id.get(),
        node=node,
        duration_ms=duration_ms,
        status="error" if error is not None else "ok",
        attempts=get_node_attempts(),
        input_digest=state_digest(state),
        output_digest=None if error is not None else state_digest(result),
        error=f"{type(error).__name__}: {error}" if error is not None else None,
    )


def make_traced_node(recorder: GraphTraceRecorder, handler: Callable[..., Any], node: str) -> Callable[..., Any]:
    """Wrap a node handler so every transition is recorded (sync or async preserved)."""
    if asyncio.iscoroutinefunction(handler):

        @functools.wraps(handler)
        async def async_traced(state: Any, *args: Any, **kwargs: Any) -> Any:
            reset_node_attempts()
            start = time.perf_counter()
            try:
                result = await handler(state, *args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - record then re-raise
                _finish(recorder, node, start, state, None, exc)
                raise
            _finish(recorder, node, start, state, result, None)
            return result

        return async_traced

    @functools.wraps(handler)
    def sync_traced(state: Any, *args: Any, **kwargs: Any) -> Any:
        reset_node_attempts()
        start = time.perf_counter()
        try:
            result = handler(state, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - record then re-raise
            _finish(recorder, node, start, state, None, exc)
            raise
        _finish(recorder, node, start, state, result, None)
        return result

    return sync_traced


def load_trace(root: str | Path, run_id: str) -> list[NodeTraceEvent]:
    """Load and order the trace events for a run, tolerating a torn trailing line."""
    events: list[NodeTraceEvent] = []
    for record in iter_jsonl(Path(root) / f"{run_id}.jsonl"):
        try:
            events.append(event_from_dict(record))
        except (KeyError, ValueError, TypeError):
            # A record whose fields don't map to a NodeTraceEvent is skipped, not fatal.
            continue
    events.sort(key=lambda e: e.seq)
    return events


def reconstruct_path(events: Sequence[NodeTraceEvent]) -> list[str]:
    """Return the ordered list of node names executed in a run."""
    return [event.node for event in sorted(events, key=lambda e: e.seq)]
