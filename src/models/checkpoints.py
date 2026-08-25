"""
Checkpoint integrity inspection and tolerant loading.

Motivation
----------
``Path.exists()`` is not a readiness signal for model weights. A clone made
without Git-LFS leaves ~130-byte pointer stubs where the real tensors should be,
and every existence check in the codebase passes on them. The failure then
surfaces deep inside a deserializer as an opaque ``UnpicklingError`` ("invalid
load key, 'v'" — ``'v'`` being the first byte of ``version https://...``), far
from the missing-data root cause.

This module classifies a checkpoint *before* a deserializer sees it, so callers
can degrade into an explicitly-labelled reduced mode with an actionable message
rather than crashing or, worse, silently pretending to be loaded.

Design notes
------------
- **No torch import at module scope.** Inspection is pure stdlib, so this module
  is importable (and unit-testable) without the ``[neural]`` extra installed.
  ``torch`` is imported lazily inside :func:`load_checkpoint` only.
- **No hardcoded paths or magic literals.** Signatures, suffixes and remediation
  text live in :mod:`src.config.constants`.
- **Directories are first-class.** A PEFT/LoRA adapter is a directory, not a
  file; a directory is OK only when it contains at least one readable weight
  file and no pointer stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any
import zipfile

from src.config.constants import (
    CHECKPOINT_WEIGHT_SUFFIXES,
    GIT_LFS_POINTER_MAGIC,
    GIT_LFS_POINTER_MAX_BYTES,
    GIT_LFS_REMEDIATION,
    PICKLE_MAX_PROTOCOL,
    PICKLE_MIN_PROTOCOL,
    PICKLE_PROTO_OPCODE,
)
from src.observability.logging import StructuredLogger, ensure_structured_logger

__all__ = [
    "CheckpointReport",
    "CheckpointStatus",
    "inspect_checkpoint",
    "load_checkpoint",
]

# Little-endian u64 header length prefix; a safetensors file opens with it and
# the JSON header immediately after.
_SAFETENSORS_HEADER_BYTES = 8
_SAFETENSORS_JSON_START = b"{"


class CheckpointStatus(str, Enum):
    """Outcome of inspecting a checkpoint path.

    ``str`` mixin so the value renders directly in structured log fields and
    JSON payloads without a custom encoder.
    """

    OK = "ok"
    MISSING = "missing"
    LFS_POINTER = "lfs_pointer"
    UNREADABLE = "unreadable"

    @property
    def is_ok(self) -> bool:
        return self is CheckpointStatus.OK


@dataclass(frozen=True)
class CheckpointReport:
    """Structured verdict for a single checkpoint path."""

    path: Path
    status: CheckpointStatus
    detail: str
    size_bytes: int = 0

    @property
    def is_ok(self) -> bool:
        return self.status.is_ok

    @property
    def remediation(self) -> str:
        """Actionable next step for the operator, or empty when none applies."""
        if self.status is CheckpointStatus.LFS_POINTER:
            return GIT_LFS_REMEDIATION
        return ""

    def as_log_fields(self) -> dict[str, Any]:
        """Flatten into structured-logging kwargs."""
        fields: dict[str, Any] = {
            "checkpoint_path": str(self.path),
            "checkpoint_status": self.status.value,
            "checkpoint_detail": self.detail,
            "checkpoint_size_bytes": self.size_bytes,
        }
        if self.remediation:
            fields["remediation"] = self.remediation
        return fields


def _is_lfs_pointer(path: Path, size: int) -> bool:
    """True when ``path`` holds a Git-LFS pointer stub rather than real content."""
    if size > GIT_LFS_POINTER_MAX_BYTES:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(len(GIT_LFS_POINTER_MAGIC)) == GIT_LFS_POINTER_MAGIC
    except OSError:
        return False


def _looks_like_pickle(head: bytes) -> bool:
    """
    True when ``head`` opens with a genuine pickle PROTO opcode.

    The opcode alone (``0x80``) is not a signature — it is an ordinary byte, so
    testing only for it classifies any binary file starting with ``0x80`` as a
    valid legacy checkpoint. The protocol number that must follow it is what makes
    the pair discriminating.
    """
    if len(head) < 2 or head[:1] != PICKLE_PROTO_OPCODE:
        return False
    return PICKLE_MIN_PROTOCOL <= head[1] <= PICKLE_MAX_PROTOCOL


def _looks_like_safetensors(head: bytes, file_size: int) -> bool:
    """
    True when ``head`` opens like a safetensors container.

    Checks the declared header length as well as the opening brace: a u64 length
    that is zero, or larger than the file itself, cannot describe a real header,
    and testing only for ``{`` at offset 8 would accept arbitrary binary content.
    """
    if len(head) <= _SAFETENSORS_HEADER_BYTES:
        return False
    if head[_SAFETENSORS_HEADER_BYTES : _SAFETENSORS_HEADER_BYTES + 1] != _SAFETENSORS_JSON_START:
        return False
    header_len = int.from_bytes(head[:_SAFETENSORS_HEADER_BYTES], "little")
    return 0 < header_len <= file_size - _SAFETENSORS_HEADER_BYTES


def _inspect_file(path: Path) -> CheckpointReport:
    """Classify a single checkpoint file."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return CheckpointReport(path, CheckpointStatus.UNREADABLE, f"stat failed: {exc}")

    if size == 0:
        return CheckpointReport(path, CheckpointStatus.UNREADABLE, "file is empty", size)

    if _is_lfs_pointer(path, size):
        return CheckpointReport(
            path,
            CheckpointStatus.LFS_POINTER,
            "Git-LFS pointer stub, not real weights; content was never fetched",
            size,
        )

    # torch>=1.6 writes a zip container; safetensors and legacy pickle are the
    # other two shapes we expect to see.
    if zipfile.is_zipfile(path):
        return CheckpointReport(path, CheckpointStatus.OK, "zip-format checkpoint", size)

    try:
        with path.open("rb") as handle:
            head = handle.read(_SAFETENSORS_HEADER_BYTES + 1)
    except OSError as exc:
        return CheckpointReport(path, CheckpointStatus.UNREADABLE, f"read failed: {exc}", size)

    if _looks_like_pickle(head):
        return CheckpointReport(path, CheckpointStatus.OK, "legacy pickle checkpoint", size)
    if _looks_like_safetensors(head, size):
        return CheckpointReport(path, CheckpointStatus.OK, "safetensors checkpoint", size)

    return CheckpointReport(
        path,
        CheckpointStatus.UNREADABLE,
        "not a recognized checkpoint container (zip, pickle or safetensors)",
        size,
    )


def _inspect_directory(path: Path) -> CheckpointReport:
    """Classify a checkpoint directory (e.g. a PEFT/LoRA adapter)."""
    weights = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in CHECKPOINT_WEIGHT_SUFFIXES)
    if not weights:
        return CheckpointReport(
            path,
            CheckpointStatus.MISSING,
            f"directory contains no weight files matching {CHECKPOINT_WEIGHT_SUFFIXES}",
        )

    reports = [_inspect_file(weight) for weight in weights]
    total = sum(report.size_bytes for report in reports)

    # A single pointer stub invalidates the directory: the adapter would load
    # partially and behave unpredictably, which is worse than refusing it.
    pointers = [r for r in reports if r.status is CheckpointStatus.LFS_POINTER]
    if pointers:
        names = ", ".join(p.path.name for p in pointers)
        return CheckpointReport(path, CheckpointStatus.LFS_POINTER, f"Git-LFS pointer stubs present: {names}", total)

    broken = [r for r in reports if not r.is_ok]
    if broken:
        names = ", ".join(f"{b.path.name} ({b.detail})" for b in broken)
        return CheckpointReport(path, CheckpointStatus.UNREADABLE, f"unreadable weight files: {names}", total)

    return CheckpointReport(path, CheckpointStatus.OK, f"{len(reports)} weight file(s) readable", total)


def inspect_checkpoint(path: str | Path) -> CheckpointReport:
    """
    Classify a checkpoint path without deserializing it.

    Accepts a file or a directory. Never raises for an absent or malformed
    checkpoint — the verdict is carried in the returned report.

    Args:
        path: Filesystem path to a checkpoint file or directory.

    Returns:
        A :class:`CheckpointReport` describing what is actually on disk.
    """
    resolved = Path(path)
    if not resolved.exists():
        return CheckpointReport(resolved, CheckpointStatus.MISSING, "path does not exist")
    if resolved.is_dir():
        return _inspect_directory(resolved)
    return _inspect_file(resolved)


def load_checkpoint(
    path: str | Path,
    *,
    logger: StructuredLogger | logging.Logger | None = None,
    map_location: Any = "cpu",
    **torch_kwargs: Any,
) -> Any | None:
    """
    Load a torch checkpoint, returning ``None`` instead of raising when it is
    absent, an LFS pointer stub, or otherwise unreadable.

    This is the tolerant counterpart to a bare ``torch.load``: callers get an
    explicit ``None`` to branch on, plus a structured warning explaining what was
    wrong and how to fix it, rather than an opaque deserializer traceback.

    Args:
        path: Checkpoint file to load.
        logger: Logger for the failure path. A stdlib ``Logger`` is normalized to
            a ``StructuredLogger`` — passing one directly would otherwise raise
            ``TypeError`` on the structured fields emitted below.
        map_location: Forwarded to ``torch.load``; defaults to CPU so loading
            never depends on CUDA availability.
        **torch_kwargs: Additional keyword arguments forwarded to ``torch.load``.

    Returns:
        The deserialized object, or ``None`` when the checkpoint is unusable.
    """
    log = ensure_structured_logger(logger, __name__)
    report = inspect_checkpoint(path)

    if not report.is_ok:
        log.warning("Checkpoint unavailable; continuing in degraded mode", **report.as_log_fields())
        return None

    try:
        import torch
    except ImportError as exc:
        log.warning(
            "Checkpoint present but torch is not installed; continuing in degraded mode",
            error=str(exc),
            remediation="pip install -e '.[neural]'",
            **report.as_log_fields(),
        )
        return None

    try:
        return torch.load(report.path, map_location=map_location, **torch_kwargs)
    except Exception as exc:  # noqa: BLE001 - deserializers raise a wide, version-dependent set
        log.warning(
            "Checkpoint failed to deserialize; continuing in degraded mode",
            error=f"{type(exc).__name__}: {exc}",
            **report.as_log_fields(),
        )
        return None
