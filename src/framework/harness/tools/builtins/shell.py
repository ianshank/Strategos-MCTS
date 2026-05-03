"""Shell-style tools: ``shell``, ``test_run``, ``lint_run``, ``type_check_run``.

All implementations use ``asyncio.create_subprocess_exec`` (no
``signal.SIGALRM``) and propagate the *current* harness correlation ID via the
``STRATEGOS_CORRELATION_ID`` environment variable so subprocess-emitted JSON
logs can be joined with the parent trace. The correlation ID is read at
*invocation time* from :func:`src.observability.logging.get_correlation_id`
(or, as a fallback, an explicit override passed at tool construction).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from src.framework.harness.settings import HarnessPermissions
from src.framework.harness.tools.registry import ToolHandler, ToolSchema
from src.observability.logging import get_correlation_id

_DEFAULT_TIMEOUT = 60.0
_CORRELATION_ENV = "STRATEGOS_CORRELATION_ID"


def _resolve_correlation_id(override: str | None) -> str | None:
    """Pick the correlation ID to propagate to a subprocess.

    Order of precedence:

    1. The harness's *current* correlation ID (read from the contextvar so
       the per-run UUID always wins, even when the registry was built before
       the run started).
    2. An explicit ``override`` supplied at tool-construction time, useful
       for tests and for callers that wire a fixed correlation outside the
       runner.
    3. ``None`` — the env var is not exported.
    """
    cid = get_correlation_id()
    if cid:
        return cid
    return override


async def _run_command(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
    correlation_id: str | None,
) -> tuple[int, str]:
    """Run a command and return ``(returncode, combined_output)``."""
    env = os.environ.copy()
    if correlation_id:
        env[_CORRELATION_ENV] = correlation_id
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, stdout.decode("utf-8", errors="replace")


def shell_tool(
    *,
    cwd: Path,
    perms: HarnessPermissions,
    correlation_id: str | None = None,
    default_timeout: float = _DEFAULT_TIMEOUT,
    allowlist: list[str] | None = None,
) -> tuple[ToolSchema, ToolHandler]:
    """Return a (schema, handler) pair for the ``shell`` tool.

    If ``allowlist`` is supplied, only commands whose first ``argv`` element
    matches an entry are allowed; everything else is rejected. This is the
    primary mechanism for sandboxing the shell tool.
    """

    async def handler(args: dict[str, Any]) -> str:
        if not perms.SHELL:
            return "permission denied: shell is disabled"
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            return "error: 'argv' must be a non-empty list of strings"
        if allowlist is not None and argv[0] not in allowlist:
            return f"permission denied: command '{argv[0]}' not in allowlist"
        timeout = float(args.get("timeout") or default_timeout)
        cid = _resolve_correlation_id(correlation_id)
        try:
            rc, output = await _run_command(argv, cwd=cwd, timeout=timeout, correlation_id=cid)
        except TimeoutError:
            return f"timeout: command exceeded {timeout}s"
        except FileNotFoundError as exc:
            return f"not_found: {exc}"
        return f"exit={rc}\n{output}"

    schema = ToolSchema(
        name="shell",
        description="Execute a non-interactive command via argv (no shell expansion).",
        parameters={
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "timeout": {"type": "number"},
            },
            "required": ["argv"],
        },
    )
    return schema, handler


def _wrap_check(
    *,
    name: str,
    description: str,
    argv: list[str],
    cwd: Path,
    perms: HarnessPermissions,
    correlation_id: str | None,
    timeout: float,
) -> tuple[ToolSchema, ToolHandler]:
    """Create a fixed-argv shell-style check tool (test/lint/type)."""

    async def handler(args: dict[str, Any]) -> str:
        if not perms.SHELL:
            return "permission denied: shell is disabled"
        extra = args.get("extra_args") or []
        if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
            return "error: 'extra_args' must be a list of strings if provided"
        full_argv = argv + extra
        cid = _resolve_correlation_id(correlation_id)
        try:
            rc, output = await _run_command(
                full_argv,
                cwd=cwd,
                timeout=timeout,
                correlation_id=cid,
            )
        except TimeoutError:
            return f"timeout: command exceeded {timeout}s"
        except FileNotFoundError as exc:
            return f"not_found: {exc}"
        verdict = "pass" if rc == 0 else "fail"
        return f"verdict={verdict} exit={rc}\n{output}"

    schema = ToolSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"extra_args": {"type": "array", "items": {"type": "string"}}},
        },
    )
    return schema, handler


def test_run_tool(
    *, cwd: Path, perms: HarnessPermissions, correlation_id: str | None = None, timeout: float = 600.0
) -> tuple[ToolSchema, ToolHandler]:
    return _wrap_check(
        name="test_run",
        description="Run pytest in the harness working tree. Returns verdict + output.",
        argv=["pytest", "-q", "--no-header"],
        cwd=cwd,
        perms=perms,
        correlation_id=correlation_id,
        timeout=timeout,
    )


def lint_run_tool(
    *, cwd: Path, perms: HarnessPermissions, correlation_id: str | None = None, timeout: float = 120.0
) -> tuple[ToolSchema, ToolHandler]:
    return _wrap_check(
        name="lint_run",
        description="Run ruff check. Returns verdict + output.",
        argv=["ruff", "check", "."],
        cwd=cwd,
        perms=perms,
        correlation_id=correlation_id,
        timeout=timeout,
    )


def type_check_run_tool(
    *, cwd: Path, perms: HarnessPermissions, correlation_id: str | None = None, timeout: float = 300.0
) -> tuple[ToolSchema, ToolHandler]:
    return _wrap_check(
        name="type_check_run",
        description="Run mypy. Returns verdict + output.",
        argv=["mypy", "src"],
        cwd=cwd,
        perms=perms,
        correlation_id=correlation_id,
        timeout=timeout,
    )


__all__ = ["lint_run_tool", "shell_tool", "test_run_tool", "type_check_run_tool"]
