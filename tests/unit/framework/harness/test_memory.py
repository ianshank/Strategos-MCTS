"""Unit tests for the markdown memory subsystem."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.framework.harness import HarnessSettings
from src.framework.harness.memory import (
    MarkdownMemoryStore,
    MemoryCompactor,
    MemoryEvent,
    MemoryEventLog,
    parse_event_line,
)
from src.framework.harness.memory.compactor import render_index
from src.framework.harness.memory.heartbeat import HeartbeatRunner
from src.framework.harness.memory.tools import register_memory_tools
from src.framework.harness.tools import ToolRegistry

pytestmark = pytest.mark.unit


def _settings(tmp_path: Path) -> HarnessSettings:
    return HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "runs",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
    )


@pytest.mark.asyncio
async def test_event_log_roundtrip(tmp_path: Path) -> None:
    """Appending then iterating returns the same events."""
    log = MemoryEventLog(root=tmp_path / "events")
    e = MemoryEvent(correlation_id="c1", phase="intent", payload={"task": "t1"})
    await log.append(e)
    events = log.collect_all()
    assert len(events) == 1
    assert events[0].phase == "intent"
    assert events[0].payload == {"task": "t1"}


@pytest.mark.asyncio
async def test_event_log_truncates_oversize_payload(tmp_path: Path) -> None:
    """Oversized events are written with a truncation marker, not as raw JSON."""
    log = MemoryEventLog(root=tmp_path / "events", max_event_bytes=200)
    huge = MemoryEvent(correlation_id="c", phase="reason", payload={"text": "x" * 5000})
    await log.append(huge)
    persisted = log.collect_all()[0]
    assert persisted.payload.get("_truncated") is True


def test_parse_event_line_inverse() -> None:
    """``parse_event_line`` is the inverse of ``to_jsonl``."""
    e = MemoryEvent(correlation_id="cid", phase="plan", payload={"a": 1})
    line = e.to_jsonl(max_bytes=4096).strip()
    parsed = parse_event_line(line)
    assert parsed.correlation_id == "cid"
    assert parsed.phase == "plan"
    assert parsed.payload == {"a": 1}


def test_render_index_handles_empty() -> None:
    """No events yields a stub message, not an exception."""
    out = render_index([])
    assert "No events yet" in out


def test_render_index_groups_by_correlation_id() -> None:
    """Events are grouped by correlation, then phase, then timestamp."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    events = [
        MemoryEvent(correlation_id="A", phase="intent", payload={"k": 1}, at=base),
        MemoryEvent(correlation_id="A", phase="plan", payload={"k": 2}, at=base),
        MemoryEvent(correlation_id="B", phase="intent", payload={"k": 3}, at=base.replace(year=2027)),
    ]
    out = render_index(events)
    # Newer correlation appears first.
    pos_b = out.index("Correlation `B`")
    pos_a = out.index("Correlation `A`")
    assert pos_b < pos_a
    assert "Phase `intent`" in out
    assert "Phase `plan`" in out


@pytest.mark.asyncio
async def test_compactor_uses_filesystem_lock(tmp_path: Path) -> None:
    """The compactor's file lock blocks a second holder until the first releases."""
    from src.framework.harness.memory.compactor import CompactorLockTimeout, _file_lock

    lock_path = tmp_path / ".compactor.lock"
    with _file_lock(lock_path, timeout_seconds=0.5):
        # Holding the lock — a second acquisition must time out fast.
        with pytest.raises(CompactorLockTimeout):
            with _file_lock(lock_path, timeout_seconds=0.05):
                pass
    # After the outer block exits, the lock file is gone.
    assert not lock_path.exists()


@pytest.mark.asyncio
async def test_compactor_reaps_stale_lock(tmp_path: Path) -> None:
    """An ancient orphaned lock file is reaped instead of blocking forever."""
    import os
    import time

    from src.framework.harness.memory.compactor import _file_lock

    lock_path = tmp_path / ".compactor.lock"
    lock_path.write_text("12345")
    # Backdate the mtime so the reaping branch trips.
    very_old = time.time() - 3600
    os.utime(lock_path, (very_old, very_old))
    # Should acquire promptly because the stale lock is reaped.
    with _file_lock(lock_path, timeout_seconds=1.0):
        pass


@pytest.mark.asyncio
async def test_compactor_idempotent(tmp_path: Path) -> None:
    """Running the compactor twice produces byte-identical output."""
    log = MemoryEventLog(root=tmp_path / "events")
    await log.append(MemoryEvent(correlation_id="A", phase="intent", payload={"k": 1}))
    await log.append(MemoryEvent(correlation_id="A", phase="plan", payload={"k": 2}))
    compactor = MemoryCompactor(log=log, index_path=tmp_path / "MEMORY.md")
    first = await compactor.compact_once()
    second = await compactor.compact_once()
    assert first == second
    assert (tmp_path / "MEMORY.md").read_text() == first


@pytest.mark.asyncio
async def test_markdown_store_appends_and_reads(tmp_path: Path) -> None:
    """The façade store wires log + compactor end-to-end."""
    store = MarkdownMemoryStore(settings=_settings(tmp_path))
    await store.append_event({"correlation_id": "c", "phase": "intent", "task_id": "t"})
    index = await store.read_index()
    assert "Correlation `c`" in index
    assert "Phase `intent`" in index


@pytest.mark.asyncio
async def test_markdown_store_view_blocks_traversal(tmp_path: Path) -> None:
    """``view`` refuses to escape the memory root."""
    store = MarkdownMemoryStore(settings=_settings(tmp_path))
    with pytest.raises(PermissionError):
        await store.view("../../etc/passwd")


@pytest.mark.asyncio
async def test_heartbeat_compacts_periodically(tmp_path: Path) -> None:
    """The heartbeat runs at least one compaction within its cadence."""
    store = MarkdownMemoryStore(settings=_settings(tmp_path))
    await store.append_event({"correlation_id": "c", "phase": "intent"})
    runner = HeartbeatRunner(store=store, interval_seconds=0.05, enabled=True)
    await runner.start()
    try:
        await asyncio.sleep(0.15)
    finally:
        await runner.stop()
    assert (tmp_path / "mem" / "MEMORY.md").exists()


@pytest.mark.asyncio
async def test_memory_tools_register_and_call(tmp_path: Path) -> None:
    """Memory tools end up on the registry and can be invoked."""
    store = MarkdownMemoryStore(settings=_settings(tmp_path))
    registry = ToolRegistry()
    register_memory_tools(registry, store)
    assert {"memory_view", "memory_query", "memory_compact"}.issubset(set(registry.list_names()))

    await store.append_event({"correlation_id": "c", "phase": "intent"})
    handler = registry.get_handler("memory_view")
    out = await handler({"path": "MEMORY.md"})
    assert "Correlation" in out
