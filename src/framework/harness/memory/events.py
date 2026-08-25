"""Append-only memory event log.

Events are JSON dicts written one-per-line using ``O_APPEND`` so concurrent
writers don't interleave (atomic up to ``PIPE_BUF`` for typical small payloads;
the harness caps event size via ``HARNESS_MEMORY_MAX_EVENT_BYTES``).

The event log is the *only* mutating surface. The compactor reads it and
materialises ``MEMORY.md`` as a derived view; humans should treat
``MEMORY.md`` as read-only and edit specs instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from src.utils.time_utils import utc_now


@dataclass(frozen=True)
class MemoryEvent:
    """A single immutable memory event."""

    correlation_id: str
    phase: str
    payload: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=utc_now)

    def to_jsonl(self, *, max_bytes: int) -> str:
        """Serialise to a single newline-terminated JSON line. Oversized
        payloads are truncated with a marker so the log never exceeds the
        atomic-append limit."""
        record = {
            "correlation_id": self.correlation_id,
            "phase": self.phase,
            "at": self.at.isoformat(),
            "payload": self.payload,
        }
        line = json.dumps(record, sort_keys=True, default=str)
        if len(line.encode("utf-8")) > max_bytes:
            truncated_payload = {"_truncated": True, "_original_keys": sorted(self.payload.keys())}
            record["payload"] = truncated_payload
            line = json.dumps(record, sort_keys=True, default=str)
        return line + "\n"


def parse_event_line(line: str) -> MemoryEvent:
    """Inverse of :meth:`MemoryEvent.to_jsonl`."""
    record = json.loads(line)
    return MemoryEvent(
        correlation_id=record["correlation_id"],
        phase=record["phase"],
        payload=record.get("payload") or {},
        at=datetime.fromisoformat(record["at"]),
    )


def iter_events(path: Path) -> Iterator[MemoryEvent]:
    """Yield events from a single event-log file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield parse_event_line(stripped)


@dataclass
class MemoryEventLog:
    """Async, append-only event log spread across daily files."""

    root: Path
    max_event_bytes: int = 4096
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, when: datetime) -> Path:
        return self.root / f"{when.strftime('%Y-%m-%d')}.jsonl"

    async def append(self, event: MemoryEvent) -> None:
        """Atomically append a single event."""
        line = event.to_jsonl(max_bytes=self.max_event_bytes)
        path = self._path_for(event.at)
        # ``O_APPEND`` guarantees atomic appends up to PIPE_BUF.
        async with self._lock:
            await asyncio.to_thread(self._append_blocking, path, line)

    @staticmethod
    def _append_blocking(path: Path, line: str) -> None:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        fd = os.open(path, flags, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    async def iter_all(self) -> AsyncIterator[MemoryEvent]:
        """Yield every event from every file in chronological order."""
        files = sorted(self.root.glob("*.jsonl"))
        for path in files:
            for event in iter_events(path):
                yield event

    def collect_all(self) -> list[MemoryEvent]:
        """Synchronous variant — used by the compactor and tests."""
        events: list[MemoryEvent] = []
        for path in sorted(self.root.glob("*.jsonl")):
            events.extend(iter_events(path))
        return events


__all__ = ["MemoryEvent", "MemoryEventLog", "iter_events", "parse_event_line"]
