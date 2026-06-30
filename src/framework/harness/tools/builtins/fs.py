"""File-system tools: read and hash-anchored edit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.framework.harness.settings import HarnessPermissions
from src.framework.harness.tools.hashed_edit import (
    HashAnchorMismatch,
    HashedEdit,
    apply_edit,
    file_sha256,
    window_hash,
)
from src.framework.harness.tools.registry import ToolHandler, ToolSchema
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Backward-compatible literal defaults; production callers inject values from
# HarnessSettings (FILE_READ_MAX_ANCHORS / HASHED_EDIT_WINDOW) via registration.
_DEFAULT_MAX_ANCHORS = 200
_DEFAULT_WINDOW = 1


def _resolve(path: str, root: Path) -> Path:
    """Resolve ``path`` under ``root``, refusing escapes."""
    target = (root / path).resolve()
    root = root.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"refusing to access path outside root: {target}") from exc
    return target


def file_read_tool(
    *,
    root: Path,
    perms: HarnessPermissions,
    max_anchors: int = _DEFAULT_MAX_ANCHORS,
    window: int = _DEFAULT_WINDOW,
) -> tuple[ToolSchema, ToolHandler]:
    """Return a (schema, handler) pair for the ``file_read`` tool.

    ``max_anchors`` caps the number of line-window hashes surfaced and
    ``window`` controls the neighbourhood hashed per anchor; both default to
    the legacy literals and are injected from settings by registration.
    """

    async def handler(args: dict[str, Any]) -> str:
        if not perms.READ:
            logger.debug("file_read denied: READ permission disabled")
            return "permission denied: file reads are disabled"
        path = str(args.get("path") or "")
        if not path:
            return "error: missing required 'path' argument"
        try:
            target = _resolve(path, root)
        except PermissionError as exc:
            logger.warning("file_read refused path outside root: %s", exc)
            return f"permission error: {exc}"
        if not target.exists():
            logger.debug("file_read miss: %s", target)
            return f"file not found: {target}"
        text = target.read_text(encoding="utf-8")
        digest = file_sha256(target)
        lines = text.splitlines()
        # Surface line-window hashes so the model can build hashed edits.
        windows = "\n".join(f"L{i}: {window_hash(lines, i, window)[:12]}" for i in range(min(len(lines), max_anchors)))
        logger.debug("file_read ok: %s (%d lines, %d anchors)", path, len(lines), min(len(lines), max_anchors))
        return f"# file: {path}\n# sha256: {digest}\n{text}\n# anchors:\n{windows}"

    schema = ToolSchema(
        name="file_read",
        description="Read a UTF-8 text file under the harness root. Returns content with hash anchors.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    return schema, handler


def file_edit_hashed_tool(
    *,
    root: Path,
    perms: HarnessPermissions,
    window: int = _DEFAULT_WINDOW,
) -> tuple[ToolSchema, ToolHandler]:
    """Return a (schema, handler) pair for the ``file_edit_hashed`` tool.

    ``window`` is the default line-window size used when the caller does not
    supply an explicit ``window`` argument; it defaults to the legacy literal
    and is injected from ``HASHED_EDIT_WINDOW`` by registration.
    """

    async def handler(args: dict[str, Any]) -> str:
        if not perms.WRITE:
            logger.debug("file_edit denied: WRITE permission disabled")
            return "permission denied: file writes are disabled"
        path = str(args.get("path") or "")
        if not path:
            return "error: missing required 'path' argument"
        try:
            target = _resolve(path, root)
        except PermissionError as exc:
            logger.warning("file_edit refused path outside root: %s", exc)
            return f"permission error: {exc}"
        try:
            edit = HashedEdit(
                path=target,
                expected_file_hash=str(args.get("expected_file_hash") or ""),
                anchor_line=int(args.get("anchor_line", 0)),
                expected_window_hash=str(args.get("expected_window_hash") or ""),
                new_content=str(args.get("new_content") or ""),
                window=int(args.get("window", window)),
            )
        except (TypeError, ValueError) as exc:
            return f"error: invalid arguments: {exc}"
        try:
            apply_edit(edit)
        except HashAnchorMismatch as exc:
            logger.warning("file_edit hash mismatch for %s: %s", path, exc)
            return f"hash_mismatch: {exc}"
        except OSError as exc:
            logger.warning("file_edit io error for %s: %s", path, exc)
            return f"io_error: {exc}"
        logger.debug("file_edit ok: wrote %d chars to %s", len(edit.new_content), path)
        return f"ok: wrote {len(edit.new_content)} chars to {path}"

    schema = ToolSchema(
        name="file_edit_hashed",
        description=(
            "Replace a file's content atomically iff both the whole-file SHA-256 and the line-window hash "
            "still match what the caller observed. Aborts on drift."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "expected_file_hash": {"type": "string"},
                "anchor_line": {"type": "integer"},
                "expected_window_hash": {"type": "string"},
                "new_content": {"type": "string"},
                "window": {"type": "integer"},
            },
            "required": ["path", "expected_file_hash", "anchor_line", "expected_window_hash", "new_content"],
        },
    )
    return schema, handler


__all__ = ["file_edit_hashed_tool", "file_read_tool"]
