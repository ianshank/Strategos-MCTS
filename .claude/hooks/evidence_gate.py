#!/usr/bin/env python3
"""PostToolUse evidence gate — stateless, self-contained, fail-open.

Companion to ``spec_gate.py``. Where the spec gate asks "is this code change
specified?", this asks the other half of the question the project keeps getting
wrong: "is this *sentence* substantiated?".

Fires after Edit/Write/MultiEdit on a **live claim surface** — ``README.md``,
``CHARTER.md``, ``docs/STATUS.md``. It scans the written file for promotion
vocabulary (phrasings that assert a capability has been demonstrated) and warns
when an occurrence is not tied to a row in ``docs/CLAIM_LEDGER.md`` that supports
it. Two rules, both mechanical:

1. A promotion word on a line that cites no ``CL-<n>`` row is unsubstantiated.
2. A promotion word on a line citing ``CL-<n>`` whose ledger grade is weaker
   than the word requires is an over-claim, and the message names the grade.

Design constraints (mirroring docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md §3):
- No import of the repo's ``src/`` and no subprocess call into it. The ledger is
  parsed as text, so the hook stays portable to the Phase 3 plugin.
- Fail-open. PostToolUse cannot deny a call that already happened, so the only
  output is advisory context. Any internal error exits 1 and blocks nothing.
- Stdout carries ONLY the JSON object when one is emitted.
- Bypass: ``EVIDENCE_GATE_BYPASS=1``.
- Deliberately narrow scope and vocabulary. A noisy advisory hook gets ignored,
  and an ignored hook is worse than no hook: it launders the appearance of a
  check. Two consequences, both measured against the live tree rather than
  guessed at:
  * Only the three surfaces above are scanned. Historical records (``CHANGELOG``,
    ``docs/archive/``, ``docs/training/``, superseded plans) assert things that
    were true when written; rewriting them is falsification, not hygiene.
  * Bare "validated" / "verified" / "guaranteed" are NOT flagged. In this tree
    they overwhelmingly describe mechanisms ("a JWT validated against
    ``JWT_SECRET``", "CI-verified on 3.11"), not capabilities. Only phrasings
    that can *only* be evaluative are matched.
- Fenced code blocks are skipped, so a shell transcript containing "PROVEN" is
  not mistaken for prose, and a double-quoted term is skipped so a document may
  define the word "proven" without tripping over its own definition.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

# The documents whose claims the ledger covers, exactly. Kept as a tuple of
# repo-relative paths so adding a surface is a one-line, reviewable change.
_CLAIM_SURFACES: tuple[str, ...] = ("README.md", "CHARTER.md", "docs/STATUS.md")

# The ledger grades claims; scanning it would fire on every row it contains.
_EXEMPT: frozenset[str] = frozenset({"docs/CLAIM_LEDGER.md"})

# Promotion vocabulary -> the weakest ledger grade that can carry it. Ordered
# most-specific-first so "fully validated" is not reported merely as "proven".
# Every entry must be a phrasing with no mechanical reading; see the module
# docstring for why bare "validated" and "verified" are deliberately absent.
_PROMOTION_WORDS: tuple[tuple[str, str], ...] = (
    (r"fully validated", "PROVEN"),
    (r"empirically validated", "PROVEN"),
    (r"production[- ]ready", "PROVEN"),
    (r"battle[- ]tested", "PROVEN"),
    (r"state[- ]of[- ]the[- ]art", "PROVEN"),
    (r"outperforms", "PROVEN"),
    (r"guaranteed to\b", "PROVEN"),
    (r"\bproven\b", "PROVEN"),
)

_CLAIM_REF_RE = re.compile(r"\bCL-(\d+)\b")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Inline code and double-quoted terms are the document talking ABOUT a word.
_QUOTED_RE = re.compile(r"`[^`]*`|\"[^\"]*\"|“[^”]*”")
# A ledger row: | CL-n | claim | source | GRADE | ...
_LEDGER_ROW_RE = re.compile(r"^\|\s*CL-(\d+)\s*\|[^|]*\|[^|]*\|\s*([A-Z]+)\s*\|")

_GRADE_RANK: tuple[str, ...] = ("FALSE", "UNPROVEN", "PARTIAL", "PROVEN")

_HEADER = (
    "Evidence gate: {target} contains {count} promotion claim(s) the claim ledger does not "
    "support. Narrow the wording or produce the evidence — see .claude/skills/validate-claims/."
)


def ledger_grades(text: str) -> dict[str, str]:
    """Parse ``docs/CLAIM_LEDGER.md`` into ``{"CL-1": "FALSE", ...}``.

    Text-only by contract. Rows that do not match the schema are skipped rather
    than raising: a malformed ledger is the validator's problem, not the hook's.
    """
    grades: dict[str, str] = {}
    for line in text.splitlines():
        match = _LEDGER_ROW_RE.match(line)
        if match and match.group(2) in _GRADE_RANK:
            grades[f"CL-{match.group(1)}"] = match.group(2)
    return grades


def _prose_lines(text: str) -> list[tuple[int, str]]:
    """Numbered lines outside fenced code blocks."""
    out: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append((number, line))
    return out


def findings(document: str, grades: dict[str, str]) -> list[str]:
    """Return one human-readable finding per unsupported promotion claim."""
    results: list[str] = []
    for number, line in _prose_lines(document):
        lowered = _QUOTED_RE.sub(" ", line).lower()
        for pattern, required in _PROMOTION_WORDS:
            hit = re.search(pattern, lowered)
            if not hit:
                continue
            cited = _CLAIM_REF_RE.findall(line)
            if not cited:
                results.append(
                    f"line {number}: '{hit.group(0)}' cites no CL-<n> row " f"(needs a row graded {required})"
                )
                break
            weakest = min(
                (grades.get(f"CL-{ref}", "UNPROVEN") for ref in cited),
                key=_GRADE_RANK.index,
            )
            if _GRADE_RANK.index(weakest) < _GRADE_RANK.index(required):
                rows = ", ".join(f"CL-{ref}" for ref in cited)
                results.append(f"line {number}: '{hit.group(0)}' relies on {rows}, graded {weakest}")
            break
    return results


def _target_path(payload: dict) -> Path | None:
    """The absolute path the tool call wrote, if the payload names one."""
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(raw, str) or not raw:
        return None
    return Path(raw)


def _repo_relative(target: Path, toplevel: Path) -> str | None:
    try:
        return target.resolve().relative_to(toplevel.resolve()).as_posix()
    except ValueError:
        return None


def _project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def _emit(message: str) -> None:
    sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": message}}))


def main() -> int:
    if os.environ.get("EVIDENCE_GATE_BYPASS") == "1":
        return 0
    payload = json.load(sys.stdin)
    target = _target_path(payload)
    if target is None:
        return 0

    toplevel = _project_dir()
    relative = _repo_relative(target, toplevel)
    if relative is None or relative in _EXEMPT:
        return 0
    if not any(relative == surface or relative.startswith(surface) for surface in _CLAIM_SURFACES):
        return 0

    try:
        document = target.read_text(encoding="utf-8")
        ledger = (toplevel / "docs" / "CLAIM_LEDGER.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0  # cannot read one side -> fail open, silently

    problems = findings(document, ledger_grades(ledger))
    if not problems:
        return 0
    _emit(
        _HEADER.format(target=relative, count=len(problems)) + "\n- " + "\n- ".join(problems),
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - fail-open by contract
        sys.stderr.write(f"evidence_gate internal error (fail-open): {exc}\n")
        sys.exit(1)
