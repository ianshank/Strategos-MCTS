#!/usr/bin/env python3
"""PreToolUse spec gate (SDD Phase 1) — stateless, self-contained, fail-open.

Fires on Edit/Write/MultiEdit/NotebookEdit (matcher in ``.claude/settings.json``).
A write under ``src/`` passes silently iff the target's git branch matches
``spec/<id>`` AND ``specs/<id>.SPEC.md`` has frontmatter status ``approved`` or
``implemented``; otherwise the gate emits a warning (warn mode) or denies the
tool call (block mode — the post-pilot flip is the ``_DEFAULT_MODE`` constant
below, one reviewed line).

Design constraints (docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md §3):
- No import of the repo's ``src/`` — the Phase 3 plugin cannot rely on it. The
  frontmatter reader below mirrors ``SpecLoader._split_frontmatter`` semantics
  and a parity test in tests/unit/tooling/ pins them together.
- Fail-open: any internal error (malformed stdin, git absent, cwd outside a
  repo) exits 1 (non-blocking) or passes silently — the gate must never wedge
  a session. Malformed/missing spec on a ``spec/<id>`` branch is a RULE
  failure (warn/deny), not an internal error.
- Stdout carries ONLY the JSON object when one is emitted.
- Bypass: ``SPEC_GATE_BYPASS=1`` (documented hotfix channel — project hooks
  merge with local settings and cannot be disabled from there).
- Known v0 hole: Bash-based writes (sed -i, tee, heredocs) are not gated.
- Native Windows without a ``python3`` launcher degrades to a per-edit
  non-blocking error (documented limitation; repo is Linux/Docker-oriented).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Post-pilot flip: change to "block" in a reviewed PR (Phase 2 exit action).
_DEFAULT_MODE = "warn"

# Mirrors SPEC_ID_PATTERN in src/framework/harness/intent/spec_scaffold.py
# (pinned by a parity test — keep the two literals identical).
_SPEC_BRANCH_RE = re.compile(r"^spec/([a-z0-9_]+)$")

_GATED_STATUSES = {"approved", "implemented"}

_MESSAGE = (
    "Spec gate ({mode}): {target} is under src/ but branch '{branch}' does not carry an "
    "approved spec (status: {status}). Use /spec-implement <id> for an approved spec, or set "
    "SPEC_GATE_BYPASS=1 for a hotfix and add a 'No-Spec: <reason>' commit trailer for CI."
)


def read_frontmatter_status(text: str) -> str:
    """Extract ``status:`` from ``---``-delimited frontmatter.

    Mirrors ``SpecLoader._split_frontmatter``: literal leading ``---\\n``,
    closing ``\\n---\\n``, line-based ``key: value``, ``#`` comment lines
    skipped, values kept verbatim (no dequoting), last occurrence wins.
    """
    if not text.startswith("---\n"):
        return ""
    end = text.find("\n---\n", 4)
    if end == -1:
        return ""
    status = ""
    for line in text[4:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            if key.strip() == "status":
                status = value.strip()
    return status


def _target_path(payload: dict) -> Path | None:
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(payload.get("cwd") or os.getcwd()) / path
    return Path(os.path.realpath(path))


def _nearest_existing_ancestor(path: Path) -> Path | None:
    for candidate in [path.parent, *path.parent.parents]:
        if candidate.is_dir():
            return candidate
    return None


def _git_branch_and_toplevel(anchor: Path) -> tuple[str, Path] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "--abbrev-ref", "HEAD", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        return None
    return lines[0].strip(), Path(os.path.realpath(lines[1].strip()))


def _emit(mode: str, message: str) -> None:
    if mode == "block":
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": message,
            }
        }
    else:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": message,
            }
        }
    sys.stdout.write(json.dumps(output))


def main() -> int:
    if os.environ.get("SPEC_GATE_BYPASS") == "1":
        return 0
    payload = json.load(sys.stdin)
    target = _target_path(payload)
    if target is None:
        return 0

    anchor = _nearest_existing_ancestor(target)
    if anchor is None:
        anchor = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    located = _git_branch_and_toplevel(anchor)
    if located is None and "CLAUDE_PROJECT_DIR" in os.environ:
        located = _git_branch_and_toplevel(Path(os.environ["CLAUDE_PROJECT_DIR"]))
    if located is None:
        return 0  # cannot scope the write to a repo -> fail open, silently
    branch, toplevel = located

    if not target.is_relative_to(toplevel / "src"):
        return 0

    status = ""
    match = _SPEC_BRANCH_RE.match(branch)
    if match:
        spec_file = toplevel / "specs" / f"{match.group(1)}.SPEC.md"
        try:
            status = read_frontmatter_status(spec_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            status = ""
        if status in _GATED_STATUSES:
            return 0

    mode = os.environ.get("SPEC_GATE_MODE", _DEFAULT_MODE)
    _emit(mode, _MESSAGE.format(mode=mode, target=target, branch=branch, status=status or "<none>"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - fail-open by contract
        sys.stderr.write(f"spec_gate internal error (fail-open): {exc}\n")
        sys.exit(1)
