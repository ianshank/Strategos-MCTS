"""Façade :class:`MemoryStore` over the event log + compactor."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from src.framework.harness.memory.compactor import MemoryCompactor
from src.framework.harness.memory.events import MemoryEvent, MemoryEventLog
from src.framework.harness.settings import HarnessSettings
from src.observability.logging import get_logger
from src.utils.time_utils import utc_now


@dataclass
class MarkdownMemoryStore:
    """Implements the :class:`MemoryStore` protocol over filesystem state."""

    settings: HarnessSettings
    log: MemoryEventLog = field(init=False)
    compactor: MemoryCompactor = field(init=False)
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))

    def __post_init__(self) -> None:
        episodic_dir = self.settings.episodic_dir()
        self.log = MemoryEventLog(root=episodic_dir, max_event_bytes=self.settings.MEMORY_MAX_EVENT_BYTES)
        self.compactor = MemoryCompactor(log=self.log, index_path=self.settings.index_path())

    async def append_event(self, event: dict[str, Any]) -> None:
        """Persist a single event."""
        memory_event = MemoryEvent(
            correlation_id=str(event.get("correlation_id") or "unknown"),
            phase=str(event.get("phase") or "unknown"),
            payload={k: v for k, v in event.items() if k not in ("correlation_id", "phase", "at")},
            at=utc_now(),
        )
        await self.log.append(memory_event)
        self.logger.debug(
            "memory event appended cid=%s phase=%s",
            memory_event.correlation_id,
            memory_event.phase,
        )

    async def read_index(self) -> str:
        """Return the most recently materialised ``MEMORY.md`` (compacts on demand)."""
        index = self.settings.index_path()
        if not index.exists():
            await self.compactor.compact_once()
        return index.read_text(encoding="utf-8") if index.exists() else ""

    async def view(self, relative_path: str) -> str:
        """Read a file relative to the memory root (read-only access)."""
        target = (self.settings.MEMORY_ROOT / relative_path).resolve()
        memory_root = self.settings.MEMORY_ROOT.resolve()
        try:
            target.relative_to(memory_root)
        except ValueError as exc:
            raise PermissionError(f"refusing to read outside memory root: {target}") from exc
        if not target.exists():
            return ""
        return target.read_text(encoding="utf-8")

    async def query_episodic(self, *, since_iso: str | None = None) -> list[dict[str, Any]]:
        """Return raw event dicts (optionally filtered by an ISO timestamp)."""
        cutoff = None
        if since_iso:
            from datetime import datetime

            cutoff = datetime.fromisoformat(since_iso)
        out: list[dict[str, Any]] = []
        for event in self.log.collect_all():
            if cutoff is not None and event.at < cutoff:
                continue
            out.append(
                {
                    "correlation_id": event.correlation_id,
                    "phase": event.phase,
                    "at": event.at.isoformat(),
                    "payload": event.payload,
                }
            )
        return out

    async def compact(self) -> str:
        """Force a compaction; returns the new index content."""
        return await self.compactor.compact_once()


__all__ = ["MarkdownMemoryStore"]
