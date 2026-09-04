#!/usr/bin/env python3
"""PreToolUse gate: no hard-coded torch device in the end-to-end suite.

``tests/README.md`` states the rule plainly — *never write a device literal in a test, take
the fixture* — because the whole point of ``tests/e2e/`` is that a device the host lacks is
reported as a **skipped case with a reason** rather than silently running somewhere else. A
literal ``device="cpu"`` defeats that: the test passes on every machine and quietly proves
nothing about the accelerator path, which is the "green but not checked" failure the suite
exists to remove. The rule was already violated once inside the very change that wrote it
(``tests/e2e/test_user_journeys.py`` hard-coded ``device="cpu"``), which is why a comment in
a README is not sufficient enforcement.

**Deliberately narrow.** It fires only on writes under ``tests/e2e/``. ``src/`` legitimately
contains ~40 device literals — availability probes such as ``torch.cuda.is_available()``
ladders and Pydantic field defaults — so a repo-wide matcher would be almost entirely false
positives and would be learned-around within a day. A gate people ignore is worse than no
gate, because it still costs attention.

**Exemptions**, each because the string is not a device *selection*:

* the shared matrix helpers under ``tests/utils/`` — they are where availability is probed;
* lines naming a device inside a string that is clearly prose (a skip reason, a message);
* any line carrying ``# device-literal: <reason>``, the documented written exception.

Warn by default, matching ``spec_gate.py``'s posture; ``DEVICE_LITERAL_GATE_MODE=block``
denies the write, and ``DEVICE_LITERAL_GATE_BYPASS=1`` disables it entirely. Fail-open: any
unexpected condition returns success, because a broken hook must never wedge an edit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Final

#: Only writes under this directory are gated (see the module docstring).
GATED_PREFIX: Final[str] = "tests/e2e"

#: Helpers that legitimately name devices: the matrix itself, and the conftest that wires it.
EXEMPT_SUFFIXES: Final[tuple[str, ...]] = ("tests/utils/device_matrix.py", "tests/utils/e2e_process.py")

#: The written exception, mirroring the `No-Spec:` trailer convention used elsewhere.
ESCAPE_COMMENT: Final[str] = "# device-literal:"

MODE_ENV: Final[str] = "DEVICE_LITERAL_GATE_MODE"
BYPASS_ENV: Final[str] = "DEVICE_LITERAL_GATE_BYPASS"
DEFAULT_MODE: Final[str] = "warn"

#: A device *assignment*, e.g. `device="cpu"` or `device = 'cuda'`. Deliberately not a bare
#: search for the word: `"cuda is not available"` in a skip reason must not trip this.
_DEVICE_ASSIGNMENT: Final[re.Pattern[str]] = re.compile(
    r"""device\s*=\s*(?P<q>["'])(?P<device>cpu|cuda(?::\d+)?|mps)(?P=q)""",
    re.IGNORECASE,
)

_REMEDIATION: Final[str] = (
    "Take the `device` fixture (or `accelerator_case`) from tests/e2e/conftest.py instead of "
    "naming a device. A literal pins the test to one device on every host, so the accelerator "
    "case silently proves nothing — see tests/README.md and "
    "docs/plans/2026-09-04-e2e-device-agnostic.md. If the literal is genuinely required, append "
    f"`{ESCAPE_COMMENT} <reason>` to the line."
)


def _written_text(payload: dict[str, Any]) -> str:
    """The text this tool call would write, across the Edit/Write/MultiEdit shapes."""
    tool_input = payload.get("tool_input") or {}
    parts: list[str] = []
    for key in ("content", "new_string"):
        value = tool_input.get(key)
        if isinstance(value, str):
            parts.append(value)
    for edit in tool_input.get("edits") or []:
        if isinstance(edit, dict) and isinstance(edit.get("new_string"), str):
            parts.append(edit["new_string"])
    return "\n".join(parts)


def offending_lines(text: str) -> list[str]:
    """Lines that select a device by literal, excluding the documented escape hatch."""
    found: list[str] = []
    for line in text.splitlines():
        if ESCAPE_COMMENT in line:
            continue
        if _DEVICE_ASSIGNMENT.search(line):
            found.append(line.strip())
    return found


def is_gated(file_path: str) -> bool:
    """Whether a path is inside the gated tree and is not one of the exempt helpers."""
    normalized = Path(file_path).as_posix()
    if any(normalized.endswith(suffix) for suffix in EXEMPT_SUFFIXES):
        return False
    return f"/{GATED_PREFIX}/" in f"/{normalized}" or normalized.startswith(f"{GATED_PREFIX}/")


def _emit(mode: str, message: str) -> None:
    if mode == "block":
        payload = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": message,
            }
        }
    else:
        payload = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": message}}
    sys.stdout.write(json.dumps(payload))


def main() -> int:
    if os.environ.get(BYPASS_ENV) == "1":
        return 0
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # fail open: a malformed payload must never wedge an edit

    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return 0
    if not is_gated(file_path):
        return 0

    offenders = offending_lines(_written_text(payload))
    if not offenders:
        return 0

    mode = os.environ.get(MODE_ENV, DEFAULT_MODE).strip().lower()
    shown = "\n".join(f"    {line}" for line in offenders[:5])
    _emit(
        mode,
        f"Device-literal gate ({mode}): {file_path} names a torch device directly:\n{shown}\n{_REMEDIATION}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
