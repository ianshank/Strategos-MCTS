"""Reusable atomic append-only JSONL helpers.

A single home for the ``O_APPEND`` write / torn-line-tolerant read idiom shared by the
execution-trace sink and the benchmark run store (and any future append-only log). ``O_APPEND``
guarantees atomic appends up to ``PIPE_BUF``, so a hard kill mid-write leaves at most a partial
trailing line, which :func:`iter_jsonl` skips.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import json
import os
from pathlib import Path
from typing import Any

# Open flags/mode for atomic append. Centralized so every append-only log is consistent.
_APPEND_FLAGS = os.O_WRONLY | os.O_APPEND | os.O_CREAT
_APPEND_FILE_MODE = 0o644


def append_jsonl(path: str | Path, record: Mapping[str, Any], *, default: Callable[[Any], Any] = str) -> None:
    """Atomically append ``record`` as one JSON line, creating parent dirs as needed.

    Args:
        path: target ``.jsonl`` file.
        record: JSON-serializable mapping.
        default: fallback serializer for non-JSON values (defaults to ``str``).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=default) + "\n"
    fd = os.open(target, _APPEND_FLAGS, _APPEND_FILE_MODE)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def iter_jsonl(path: str | Path, *, tolerate_partial: bool = True) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file (empty when the file is absent).

    Args:
        path: source ``.jsonl`` file.
        tolerate_partial: when True (default) a torn/partial line — e.g. from a SIGKILL
            mid-write — is skipped; when False a ``json.JSONDecodeError`` propagates.
    """
    source = Path(path)
    if not source.exists():
        return
    # Stream line-by-line rather than read_text().splitlines() so a large trace/result log is
    # never fully materialized in memory; both callers consume the generator to exhaustion, so the
    # context manager closes the handle deterministically.
    with source.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if tolerate_partial:
                    continue
                raise
