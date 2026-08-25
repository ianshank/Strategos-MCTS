"""Async background heartbeat that compacts memory on a configurable cadence.

The heartbeat is intentionally tiny: it sleeps, runs the compactor, sleeps
again, and exits cleanly when ``stop()`` is called. We don't do LLM-driven
curation here yet — that's an enhancement that can plug into the same lifecycle.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging

from src.framework.harness.memory.markdown import MarkdownMemoryStore
from src.observability.logging import get_logger


@dataclass
class HeartbeatRunner:
    """Run :meth:`MarkdownMemoryStore.compact` on a schedule."""

    store: MarkdownMemoryStore
    interval_seconds: float
    enabled: bool = True
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _task: asyncio.Task[None] | None = field(default=None, init=False)

    async def start(self) -> None:
        """Launch the background loop. No-op if disabled or already running."""
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="harness-heartbeat")

    async def stop(self) -> None:
        """Signal the loop to exit and await its termination."""
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await self.store.compact()
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("heartbeat compaction failed err=%s", type(exc).__name__)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                except TimeoutError:
                    continue
        finally:
            self.logger.debug("heartbeat exited")


__all__ = ["HeartbeatRunner"]
