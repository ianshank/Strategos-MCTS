"""Hash-anchored file edits.

Goals:

* Reject edits when the file changed concurrently (whole-file SHA-256 CAS).
* Reject edits when the surrounding context shifted (window-hash anchor).
* Atomic write via ``os.replace`` so partial failures never leave half-written
  files on disk.

The semantics: callers supply ``(file_hash, anchor_line, anchor_window_hash,
new_content)``. If both the file hash and the anchor hash still match, the
new content is written. Otherwise we raise :class:`HashAnchorMismatch`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


class HashAnchorMismatch(RuntimeError):
    """The on-disk content no longer matches the expected hash."""


def sha256_hex(payload: bytes | str) -> str:
    """Convenience helper — hex SHA-256 over bytes or UTF-8 strings."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of the file contents (empty string if absent)."""
    if not path.exists():
        return sha256_hex(b"")
    return sha256_hex(path.read_bytes())


def window_hash(lines: list[str], anchor: int, window: int) -> str:
    """Hash the lines around ``anchor`` (zero-indexed) with ``window`` neighbours."""
    if anchor < 0 or anchor >= len(lines):
        raise IndexError(f"anchor {anchor} out of range for {len(lines)} lines")
    start = max(0, anchor - window)
    end = min(len(lines), anchor + window + 1)
    block = "\n".join(lines[start:end])
    return sha256_hex(block)


@dataclass(frozen=True)
class HashedEdit:
    """A pending edit with the hashes the caller observed."""

    path: Path
    expected_file_hash: str
    anchor_line: int
    expected_window_hash: str
    new_content: str
    window: int = 1


def apply_edit(edit: HashedEdit) -> None:
    """Apply ``edit`` if both hashes still match. Raise on mismatch.

    The check-then-write sequence is racy in absolute terms (between the read
    and the write another writer could mutate the file). We accept that
    bounded race because every harness writer goes through this same path
    and would fail symmetrically; the canonical use case is *catching*
    drift, not preventing it deterministically.

    Argument validation: ``anchor_line`` must point at an existing line and
    ``window`` must be non-negative. Out-of-range anchors are surfaced as
    :class:`HashAnchorMismatch` (not ``IndexError``) so the file-edit tool
    can return a stable ``hash_mismatch:`` reply rather than letting an
    untyped exception leak to the caller.
    """
    if edit.window < 0:
        raise HashAnchorMismatch(f"invalid window {edit.window}: must be >= 0")

    actual_file = file_sha256(edit.path)
    if actual_file != edit.expected_file_hash:
        raise HashAnchorMismatch(
            f"file hash mismatch for {edit.path}: expected {edit.expected_file_hash[:12]} got {actual_file[:12]}"
        )
    if edit.path.exists():
        existing_lines = edit.path.read_text(encoding="utf-8").splitlines()
        if existing_lines:
            if not (0 <= edit.anchor_line < len(existing_lines)):
                raise HashAnchorMismatch(
                    f"anchor_line {edit.anchor_line} out of range for {edit.path} ({len(existing_lines)} lines)"
                )
            actual_window = window_hash(existing_lines, edit.anchor_line, edit.window)
            if actual_window != edit.expected_window_hash:
                raise HashAnchorMismatch(f"window hash mismatch for {edit.path} at line {edit.anchor_line}")
    tmp = edit.path.with_suffix(edit.path.suffix + ".tmp")
    edit.path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(edit.new_content, encoding="utf-8")
    os.replace(tmp, edit.path)


__all__ = ["HashAnchorMismatch", "HashedEdit", "apply_edit", "file_sha256", "sha256_hex", "window_hash"]
