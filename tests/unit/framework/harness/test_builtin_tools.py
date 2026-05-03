"""Unit tests for builtin tools (file read/edit, shell, test/lint/type runners)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.framework.harness.settings import HarnessPermissions
from src.framework.harness.tools.builtins import register_builtin_tools
from src.framework.harness.tools.builtins.fs import (
    file_edit_hashed_tool,
    file_read_tool,
)
from src.framework.harness.tools.builtins.shell import shell_tool
from src.framework.harness.tools.hashed_edit import (
    HashAnchorMismatch,
    HashedEdit,
    apply_edit,
    file_sha256,
    sha256_hex,
    window_hash,
)
from src.framework.harness.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------
# Hashed-edit primitives
# ---------------------------------------------------------------------


def test_sha256_hex_handles_strings_and_bytes() -> None:
    """Both ``str`` and ``bytes`` inputs hash equally for the same payload."""
    assert sha256_hex("abc") == sha256_hex(b"abc")


def test_window_hash_rejects_invalid_anchor() -> None:
    """An out-of-range anchor raises ``IndexError``."""
    with pytest.raises(IndexError):
        window_hash(["a", "b"], anchor=5, window=1)


def test_apply_edit_round_trip(tmp_path: Path) -> None:
    """A successful edit writes the new content atomically."""
    target = tmp_path / "f.txt"
    target.write_text("line0\nline1\nline2\n")
    lines = target.read_text().splitlines()
    edit = HashedEdit(
        path=target,
        expected_file_hash=file_sha256(target),
        anchor_line=1,
        expected_window_hash=window_hash(lines, 1, 1),
        new_content="newcontent",
        window=1,
    )
    apply_edit(edit)
    assert target.read_text() == "newcontent"


def test_apply_edit_rejects_file_mutation(tmp_path: Path) -> None:
    """If the file changed since the caller read it, the edit aborts."""
    target = tmp_path / "f.txt"
    target.write_text("a\nb\nc\n")
    edit = HashedEdit(
        path=target,
        expected_file_hash=sha256_hex("stale"),  # wrong hash
        anchor_line=1,
        expected_window_hash="",
        new_content="x",
    )
    with pytest.raises(HashAnchorMismatch):
        apply_edit(edit)


def test_apply_edit_rejects_window_drift(tmp_path: Path) -> None:
    """If the surrounding lines drifted, the edit aborts even if the file hash matches."""
    target = tmp_path / "f.txt"
    target.write_text("a\nb\nc\n")
    edit = HashedEdit(
        path=target,
        expected_file_hash=file_sha256(target),
        anchor_line=1,
        expected_window_hash=sha256_hex("wrong-window"),
        new_content="x",
    )
    with pytest.raises(HashAnchorMismatch):
        apply_edit(edit)


# ---------------------------------------------------------------------
# file_read / file_edit_hashed handlers
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_read_returns_hash_and_anchors(tmp_path: Path) -> None:
    """``file_read`` surfaces both the whole-file hash and per-line anchors."""
    target = tmp_path / "doc.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    perms = HarnessPermissions(READ=True)
    _, handler = file_read_tool(root=tmp_path, perms=perms)
    out = await handler({"path": "doc.txt"})
    assert "sha256:" in out
    assert "L0:" in out
    assert "alpha" in out


@pytest.mark.asyncio
async def test_file_read_blocks_traversal(tmp_path: Path) -> None:
    """Path traversal outside ``root`` is denied."""
    perms = HarnessPermissions(READ=True)
    _, handler = file_read_tool(root=tmp_path, perms=perms)
    out = await handler({"path": "../../etc/passwd"})
    assert "permission error" in out


@pytest.mark.asyncio
async def test_file_read_respects_permissions(tmp_path: Path) -> None:
    """Read permission can be revoked by setting ``READ=False``."""
    perms = HarnessPermissions(READ=False)
    _, handler = file_read_tool(root=tmp_path, perms=perms)
    out = await handler({"path": "x"})
    assert "permission denied" in out


@pytest.mark.asyncio
async def test_file_edit_hashed_writes_then_reports(tmp_path: Path) -> None:
    """Successful hashed edits return ``ok:`` and persist the new content."""
    target = tmp_path / "f.txt"
    target.write_text("hello\nworld\n")
    perms = HarnessPermissions(WRITE=True)
    _, handler = file_edit_hashed_tool(root=tmp_path, perms=perms)
    file_hash = file_sha256(target)
    win = window_hash(target.read_text().splitlines(), 1, 1)
    out = await handler(
        {
            "path": "f.txt",
            "expected_file_hash": file_hash,
            "anchor_line": 1,
            "expected_window_hash": win,
            "new_content": "replaced",
        }
    )
    assert out.startswith("ok:")
    assert target.read_text() == "replaced"


@pytest.mark.asyncio
async def test_file_edit_hashed_reports_drift(tmp_path: Path) -> None:
    """A stale hash returns a typed ``hash_mismatch:`` reply (not an exception)."""
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    perms = HarnessPermissions(WRITE=True)
    _, handler = file_edit_hashed_tool(root=tmp_path, perms=perms)
    out = await handler(
        {
            "path": "f.txt",
            "expected_file_hash": sha256_hex("stale"),
            "anchor_line": 0,
            "expected_window_hash": "",
            "new_content": "x",
        }
    )
    assert out.startswith("hash_mismatch:")


# ---------------------------------------------------------------------
# Shell tool
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shell_runs_python_echo(tmp_path: Path) -> None:
    """The shell tool runs a real subprocess and returns combined output."""
    perms = HarnessPermissions(SHELL=True)
    _, handler = shell_tool(cwd=tmp_path, perms=perms, allowlist=[sys.executable])
    out = await handler({"argv": [sys.executable, "-c", "print('hi from harness')"]})
    assert "hi from harness" in out
    assert out.startswith("exit=0")


@pytest.mark.asyncio
async def test_shell_enforces_allowlist(tmp_path: Path) -> None:
    """Commands outside the allowlist are denied without execution."""
    perms = HarnessPermissions(SHELL=True)
    _, handler = shell_tool(cwd=tmp_path, perms=perms, allowlist=[sys.executable])
    out = await handler({"argv": ["/bin/ls"]})
    assert "permission denied" in out


@pytest.mark.asyncio
async def test_shell_enforces_disabled_permission(tmp_path: Path) -> None:
    """``SHELL=False`` blocks all invocations regardless of allowlist."""
    perms = HarnessPermissions(SHELL=False)
    _, handler = shell_tool(cwd=tmp_path, perms=perms)
    out = await handler({"argv": [sys.executable, "-c", "print(1)"]})
    assert "permission denied" in out


@pytest.mark.asyncio
async def test_shell_propagates_correlation_id(tmp_path: Path) -> None:
    """The current correlation ID (read at invocation time) reaches the subprocess."""
    from src.observability.logging import set_correlation_id

    perms = HarnessPermissions(SHELL=True)
    set_correlation_id("abc-123")
    _, handler = shell_tool(
        cwd=tmp_path,
        perms=perms,
        allowlist=[sys.executable],
    )
    out = await handler(
        {"argv": [sys.executable, "-c", "import os; print(os.environ.get('STRATEGOS_CORRELATION_ID'))"]}
    )
    assert "abc-123" in out


@pytest.mark.asyncio
async def test_shell_falls_back_to_construction_correlation_id(tmp_path: Path) -> None:
    """When no contextvar is set, the construction-time override is used."""
    from src.observability.logging import set_correlation_id

    perms = HarnessPermissions(SHELL=True)
    # Explicitly clear so the contextvar doesn't leak from a prior test.
    set_correlation_id("")
    _, handler = shell_tool(
        cwd=tmp_path,
        perms=perms,
        correlation_id="static-fallback",
        allowlist=[sys.executable],
    )
    out = await handler(
        {"argv": [sys.executable, "-c", "import os; print(os.environ.get('STRATEGOS_CORRELATION_ID'))"]}
    )
    assert "static-fallback" in out


# ---------------------------------------------------------------------
# Bulk registration
# ---------------------------------------------------------------------


def test_register_builtin_tools_adds_all(tmp_path: Path) -> None:
    """``register_builtin_tools`` lays down every standard tool."""
    perms = HarnessPermissions(READ=True, WRITE=True, SHELL=True)
    registry = ToolRegistry()
    register_builtin_tools(registry, root=tmp_path, perms=perms)
    assert {
        "file_read",
        "file_edit_hashed",
        "shell",
        "test_run",
        "lint_run",
        "type_check_run",
    }.issubset(set(registry.list_names()))
