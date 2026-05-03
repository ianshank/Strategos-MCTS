"""Output truncation with spillover-to-disk for full logs."""

from __future__ import annotations

import re
from pathlib import Path

# Allow only safe filename characters; everything else is replaced with ``_``.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_component(value: str, *, fallback: str) -> str:
    """Sanitize an arbitrary string for use as a single filename component.

    Removes path separators, traversal markers, and other unsafe characters.
    Preserves alphanumerics, dot, hyphen, and underscore so the result still
    looks like a normal identifier. Empty / dotfile-only inputs fall back to
    ``fallback`` so a spillover file can always be created.
    """
    cleaned = _FILENAME_SAFE.sub("_", value).strip("._")
    return cleaned or fallback


def truncate_with_spillover(
    payload: str,
    *,
    head_chars: int,
    tail_chars: int,
    spillover_dir: Path,
    correlation_id: str,
    step_id: str,
    marker_template: str,
) -> tuple[str, str | None]:
    """Truncate ``payload`` and persist the full text on disk.

    Returns ``(truncated_payload, spillover_path_or_none)``. If the payload is
    short enough, no spillover file is written and the second tuple element
    is ``None``.

    The marker template may contain ``{path}`` which is substituted with the
    spillover path so the model sees a deterministic pointer to the full log.

    Both ``correlation_id`` and ``step_id`` are sanitized to a safe filename
    component before being joined with ``spillover_dir``, preventing path
    traversal even when callers pass tool-supplied identifiers. The final
    resolved path is verified to live under the resolved spillover root; any
    escape attempt raises :class:`ValueError`.
    """
    if head_chars < 0 or tail_chars < 0:
        raise ValueError("head_chars and tail_chars must be non-negative")
    if len(payload) <= head_chars + tail_chars:
        return payload, None

    safe_corr = _safe_component(correlation_id, fallback="cid")
    safe_step = _safe_component(step_id, fallback="step")
    spillover_root = spillover_dir.resolve()
    target_dir = (spillover_root / safe_corr).resolve()
    # Defence in depth: even after sanitisation, refuse to write outside root.
    try:
        target_dir.relative_to(spillover_root)
    except ValueError as exc:
        raise ValueError(f"refusing to write spillover outside root: {target_dir}") from exc
    target_dir.mkdir(parents=True, exist_ok=True)
    path = (target_dir / f"{safe_step}.log").resolve()
    try:
        path.relative_to(spillover_root)
    except ValueError as exc:
        raise ValueError(f"refusing to write spillover outside root: {path}") from exc
    path.write_text(payload, encoding="utf-8")

    head = payload[:head_chars]
    tail = payload[-tail_chars:] if tail_chars > 0 else ""
    marker = marker_template.format(path=str(path))
    return f"{head}{marker}{tail}", str(path)


__all__ = ["truncate_with_spillover"]
