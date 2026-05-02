"""Single-writer compactor that materialises ``MEMORY.md`` from events.

Compaction is deterministic and idempotent: given the same set of events,
the resulting ``MEMORY.md`` is byte-identical regardless of input ordering
within commutative phases (intent / plan / verify groupings).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.framework.harness.memory.events import MemoryEvent, MemoryEventLog


class CompactorLockTimeout(RuntimeError):
    """The cross-process compactor lock could not be acquired in time."""


@contextlib.contextmanager
def _file_lock(
    path: Path,
    *,
    timeout_seconds: float,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    """Acquire a cross-process lock by exclusively creating ``path``.

    Uses ``O_CREAT | O_EXCL`` so two processes cannot both create the file;
    the loser polls until the file is gone or the timeout elapses. The lock
    file is removed on exit even if the protected block raises. Stale locks
    older than ``timeout_seconds * 4`` are reaped to avoid permanent
    deadlock if a previous holder crashed.
    """
    deadline = time.monotonic() + timeout_seconds
    stale_after = max(timeout_seconds * 4.0, 5.0)
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()}\n".encode())
            break
        except FileExistsError:
            # Reap stale locks left behind by crashed holders.
            try:
                age = time.time() - path.stat().st_mtime
                if age > stale_after:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise CompactorLockTimeout(f"could not acquire {path} within {timeout_seconds}s") from None
            time.sleep(poll_interval)
        except OSError as exc:  # pragma: no cover - filesystem-specific
            if exc.errno == errno.EEXIST:
                if time.monotonic() >= deadline:
                    raise CompactorLockTimeout(f"could not acquire {path} within {timeout_seconds}s") from exc
                time.sleep(poll_interval)
                continue
            raise
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


@dataclass
class MemoryCompactor:
    """Materialise the ``MEMORY.md`` index from the event log.

    Two layers of mutual exclusion:

    * An ``asyncio.Lock`` serialises in-process callers (cheap).
    * A filesystem lock at :attr:`lock_path` serialises across processes,
      using ``O_CREAT | O_EXCL`` semantics with stale-lock reaping so a
      crashed compactor can't permanently block fresh ones.
    """

    log: MemoryEventLog
    index_path: Path
    lock_timeout_seconds: float = 10.0
    lock_path: Path = field(init=False)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _async_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self.lock_path = self.index_path.parent / ".compactor.lock"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    async def compact_once(self) -> str:
        """Rebuild ``MEMORY.md`` from the current event log; returns its content."""
        async with self._async_lock:
            content = await asyncio.to_thread(self._compact_blocking)
            return content

    def _compact_blocking(self) -> str:
        events = self.log.collect_all()
        rendered = render_index(events)
        with _file_lock(self.lock_path, timeout_seconds=self.lock_timeout_seconds):
            # Atomic write via temp file + rename inside the cross-process lock.
            tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
            tmp.write_text(rendered, encoding="utf-8")
            tmp.replace(self.index_path)
        return rendered


def render_index(events: Iterable[MemoryEvent]) -> str:
    """Pure function: turn events into the canonical ``MEMORY.md`` content.

    The output is grouped by ``correlation_id`` (newest first, by max-event
    timestamp), then by phase within each correlation. This grouping is
    commutative within a phase, which is what makes the operation idempotent
    over reorderings inside a single phase block.
    """
    by_corr: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        by_corr[event.correlation_id].append(event)

    sections: list[str] = ["# Harness Memory Index", ""]
    if not by_corr:
        sections.append("_No events yet._")
        return "\n".join(sections) + "\n"

    # Sort correlations by their newest event timestamp (descending).
    ordered = sorted(
        by_corr.items(),
        key=lambda kv: max(e.at for e in kv[1]),
        reverse=True,
    )
    for corr_id, group in ordered:
        sections.append(f"## Correlation `{corr_id}`")
        # Group by phase, stable-sorted by phase name for deterministic output.
        phases: dict[str, list[MemoryEvent]] = defaultdict(list)
        for event in group:
            phases[event.phase].append(event)
        for phase in sorted(phases.keys()):
            sections.append(f"### Phase `{phase}`")
            for event in sorted(phases[phase], key=lambda e: e.at):
                rendered = ", ".join(f"{k}={v}" for k, v in sorted(event.payload.items()))
                sections.append(f"- {event.at.isoformat()} — {rendered}" if rendered else f"- {event.at.isoformat()}")
            sections.append("")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


__all__ = ["MemoryCompactor", "render_index"]
