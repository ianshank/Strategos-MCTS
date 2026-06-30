"""Bulk registration helper for builtin tools."""

from __future__ import annotations

from pathlib import Path

from src.framework.harness.settings import HarnessPermissions, HarnessSettings, get_harness_settings
from src.framework.harness.tools.builtins.fs import file_edit_hashed_tool, file_read_tool
from src.framework.harness.tools.builtins.shell import (
    lint_run_tool,
    shell_tool,
    test_run_tool,
    type_check_run_tool,
)
from src.framework.harness.tools.registry import ToolRegistry


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    root: Path,
    perms: HarnessPermissions,
    correlation_id: str | None = None,
    shell_allowlist: list[str] | None = None,
    settings: HarnessSettings | None = None,
) -> None:
    """Register every builtin tool against ``registry``.

    Permissions on individual tools still apply: a permissionless registry
    will surface 'permission denied' messages instead of executing.

    Per-tool timeouts, the hashed-edit window, and the read-anchor cap are
    sourced from ``HarnessSettings`` (defaulting to the cached instance when
    not supplied) so no limits are hardcoded at the registration site. The
    setting defaults equal the tools' historical literals, so behaviour is
    unchanged for existing callers.
    """
    hs = settings or get_harness_settings()

    schema, handler = file_read_tool(
        root=root, perms=perms, max_anchors=hs.FILE_READ_MAX_ANCHORS, window=hs.HASHED_EDIT_WINDOW
    )
    registry.register(schema, handler)

    schema, handler = file_edit_hashed_tool(root=root, perms=perms, window=hs.HASHED_EDIT_WINDOW)
    registry.register(schema, handler)

    schema, handler = shell_tool(
        cwd=root,
        perms=perms,
        correlation_id=correlation_id,
        default_timeout=hs.TOOL_DEFAULT_TIMEOUT_SECONDS,
        allowlist=shell_allowlist,
    )
    registry.register(schema, handler)

    schema, handler = test_run_tool(
        cwd=root, perms=perms, correlation_id=correlation_id, timeout=hs.TOOL_TEST_TIMEOUT_SECONDS
    )
    registry.register(schema, handler)

    schema, handler = lint_run_tool(
        cwd=root, perms=perms, correlation_id=correlation_id, timeout=hs.TOOL_LINT_TIMEOUT_SECONDS
    )
    registry.register(schema, handler)

    schema, handler = type_check_run_tool(
        cwd=root, perms=perms, correlation_id=correlation_id, timeout=hs.TOOL_TYPECHECK_TIMEOUT_SECONDS
    )
    registry.register(schema, handler)


__all__ = ["register_builtin_tools"]
