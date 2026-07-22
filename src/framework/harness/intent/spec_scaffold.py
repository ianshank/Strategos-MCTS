"""Deterministic spec scaffolding for ``/spec-new`` (SDD Phase 1).

The refusal rules live here — not in slash-command prose — so they are
testable, consistent, and extractable: a new spec is refused when its id is
malformed, when ``specs/<id>.SPEC.md`` already exists, or when an *open*
(``draft``/``approved``) spec already covers an overlapping module prefix.

Overlap is segment-wise: both module values are normalized to a trailing
slash before prefix comparison, so ``src`` does not overlap ``src2/`` while
``src/`` does overlap ``src/api/`` (either direction).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from src.framework.harness.intent.spec_loader import SpecLoader, SpecParseError
from src.framework.harness.intent.spec_validator import SpecValidator

# One id grammar, three consumers: /spec-new refusal (fullmatch, here), the
# PreToolUse gate's branch regex (anchored literal mirrored in
# .claude/hooks/spec_gate.py — pinned by a parity test), and spec-trace's
# branch parsing. Constrained so every id is also a valid git branch suffix.
SPEC_ID_PATTERN: Final[str] = r"[a-z0-9_]+"
_SPEC_ID_RE: Final[re.Pattern[str]] = re.compile(SPEC_ID_PATTERN)

# ``missing-goal`` is a validator *error*, so the scaffold must never write an
# empty goal — this placeholder passes the non-emptiness check by design.
DEFAULT_GOAL: Final[str] = "TODO: state the contract this spec enforces."

# Statuses that make a spec "open": it claims its module against new specs.
OPEN_STATUSES: Final[frozenset[str]] = frozenset({"draft", "approved"})

_TEMPLATE: Final[
    str
] = """---
id: {spec_id}
goal: {goal}
module: {module}
status: draft
---

# Goal

{goal}

# Acceptance Criteria

- AC-1: TODO — make this falsifiable and name an intended test path (e.g. tests/unit/...)

# Constraints

- TODO

# Invariants

- TODO

# Out of Scope

- TODO
"""


class SpecScaffoldError(ValueError):
    """Raised when a new spec must be refused (reason in the message)."""


def _refuse_frontmatter_injection(field: str, value: str) -> None:
    """Refuse values that could rewrite the rendered frontmatter block.

    ``goal``/``module`` are interpolated verbatim into the template; a newline
    lets a value smuggle extra ``key: value`` lines (e.g. ``status: approved``,
    skipping the human draft->approved flip), and ``---`` can close the block
    early with the same effect (first-``\\n---\\n``-wins in the loader).
    """
    if "\n" in value or "\r" in value:
        raise SpecScaffoldError(f"{field} must be a single line (newline would inject frontmatter)")
    if "---" in value:
        raise SpecScaffoldError(f"{field} must not contain '---' (frontmatter delimiter)")


def normalize_module(module: str) -> str:
    """Normalize a module path prefix for overlap comparison (trailing slash)."""
    cleaned = module.strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.rstrip("/") + "/"


def modules_overlap(first: str, second: str) -> bool:
    """Segment-wise prefix overlap in either direction (``src`` ≠ ``src2/``)."""
    a, b = normalize_module(first), normalize_module(second)
    return a.startswith(b) or b.startswith(a)


def scaffold_spec(specs_dir: Path, spec_id: str, module: str, goal: str = "") -> Path:
    """Create ``specs_dir/<spec_id>.SPEC.md`` from the schema-v2 template.

    Raises :class:`SpecScaffoldError` instead of writing anything when a
    refusal rule fires; the rendered text is validated *before* the write so
    nothing half-valid is ever left on disk.
    """
    if not _SPEC_ID_RE.fullmatch(spec_id):
        raise SpecScaffoldError(f"invalid spec id '{spec_id}': must match {SPEC_ID_PATTERN} (lowercase, digits, _)")
    if not module.strip():
        raise SpecScaffoldError("module must be a non-empty repo-relative path prefix (e.g. src/api/)")
    _refuse_frontmatter_injection("module", module)
    _refuse_frontmatter_injection("goal", goal)
    cleaned_module = module.strip()
    if cleaned_module.startswith("/") or ".." in cleaned_module.split("/"):
        raise SpecScaffoldError(f"module '{module}' must be repo-relative without '..' segments (e.g. src/api/)")
    path = specs_dir / f"{spec_id}.SPEC.md"
    if path.exists():
        raise SpecScaffoldError(f"spec already exists: {path}")

    loader = SpecLoader()
    for existing in sorted(specs_dir.glob("*.SPEC.md")):
        try:
            spec = loader.load(existing)
        except (SpecParseError, OSError, UnicodeDecodeError) as exc:
            # Fail closed: an unreadable candidate could be an open spec whose
            # module genuinely collides — refuse rather than silently skip.
            raise SpecScaffoldError(
                f"cannot check module overlap: {existing} is unreadable ({exc}); fix or remove it"
            ) from exc
        if spec.status in OPEN_STATUSES and spec.module and modules_overlap(spec.module, module):
            raise SpecScaffoldError(
                f"module '{module}' overlaps open spec '{spec.id or existing.name}' "
                f"(status={spec.status}, module={spec.module}); close it or pick a narrower module"
            )

    text = _TEMPLATE.format(spec_id=spec_id, goal=goal.strip() or DEFAULT_GOAL, module=normalize_module(module))
    errors = [i for i in SpecValidator().validate_spec(loader.parse(text), path) if i.severity == "error"]
    if errors:  # pragma: no cover - template regression guard, pinned by tests
        rendered = "; ".join(issue.render() for issue in errors)
        raise SpecScaffoldError(f"scaffold template failed self-validation: {rendered}")

    specs_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_GOAL",
    "OPEN_STATUSES",
    "SPEC_ID_PATTERN",
    "SpecScaffoldError",
    "modules_overlap",
    "normalize_module",
    "scaffold_spec",
]
