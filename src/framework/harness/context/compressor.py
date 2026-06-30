"""Trim and summarise long episodic logs so they fit a context budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _CompressorSettings(Protocol):
    """Minimal settings surface the compressor reads (duck-typed)."""

    CONTEXT_COMPRESS_MAX_CHARS: int
    CONTEXT_COMPRESS_HEAD_CHARS: int
    CONTEXT_COMPRESS_TAIL_CHARS: int


@dataclass
class EpisodicCompressor:
    """Truncate and tag an episodic log slice for prompt inclusion.

    No LLM call here — this is the cheap path. The :class:`heartbeat
    <src.framework.harness.memory.heartbeat>` handles LLM-driven curation.

    Defaults are retained for backward-compatible direct construction; prefer
    :meth:`from_settings` so budgets come from ``HarnessSettings`` rather than
    being hardcoded at call sites.
    """

    max_chars: int = 4000
    head_chars: int = 1500
    tail_chars: int = 1500
    truncation_marker: str = "\n…[older entries elided]…\n"

    @classmethod
    def from_settings(cls, settings: _CompressorSettings) -> EpisodicCompressor:
        """Build a compressor from harness settings (no hardcoded budgets)."""
        return cls(
            max_chars=settings.CONTEXT_COMPRESS_MAX_CHARS,
            head_chars=settings.CONTEXT_COMPRESS_HEAD_CHARS,
            tail_chars=settings.CONTEXT_COMPRESS_TAIL_CHARS,
        )

    def compress(self, text: str) -> str:
        """Return ``text`` if short enough, else head + marker + tail.

        ``max_chars`` is a hard upper bound: the head/tail slices are clamped to
        fit around the marker so the result never exceeds ``max_chars`` even when
        ``head_chars``/``tail_chars`` were configured larger than the cap. (The
        sole exception is a marker longer than ``max_chars`` itself, which is
        pathological — markers are short fixed strings.)
        """
        if len(text) <= self.max_chars:
            return text
        budget = max(0, self.max_chars - len(self.truncation_marker))
        head_len = min(self.head_chars, budget)
        tail_len = min(self.tail_chars, budget - head_len)
        head = text[:head_len]
        tail = text[-tail_len:] if tail_len > 0 else ""
        return f"{head}{self.truncation_marker}{tail}"


__all__ = ["EpisodicCompressor"]
