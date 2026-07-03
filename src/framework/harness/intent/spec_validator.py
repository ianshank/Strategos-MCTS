"""Schema v2 validator for SPEC.md files.

:class:`SpecValidator` is the enforcement counterpart to the deliberately
forgiving :class:`.spec_loader.SpecLoader`: the loader accepts anything it can
read; this module decides what is *valid* under the spec contract defined in
``docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md`` §2. It is CLI-free on purpose —
``harness validate-spec`` (and CI) only render the issues returned here.

Validation is fail-loud but total: every path passed to
:meth:`SpecValidator.validate_paths` is checked and reported, and unreadable
files become error issues rather than exceptions.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from src.framework.harness.intent.spec_loader import Spec, SpecLoader, SpecParseError

# The complete status lifecycle. Schema vocabulary, not tunable configuration —
# kept harness-local (like ``RalphStatus``) so the harness stays extractable.
SPEC_STATUSES: Final[frozenset[str]] = frozenset({"draft", "approved", "implemented", "verified", "superseded"})

SPEC_SUFFIX: Final[str] = ".SPEC.md"

# Signature shared by the error/warning reporters threaded through the checks.
_Report = Callable[[str, str], None]

# No-changelog rule: specs state future contracts, never completed work.
# Deliberately narrow — only the bold-parenthesized done-marker family
# ("**(8a — done)**", "**(Done)**"); broader prose judgment is out of scope
# here (it belongs to spec review, not mechanical validation). ``\bdone``
# keeps "**(overdone)**"-class text out, and the Ralph completion marker
# ("<!-- HARNESS:DONE -->") cannot match the bold-paren shape at all.
DONE_MARKER_RE: Final[re.Pattern[str]] = re.compile(r"\*\*\([^)\n]*?\bdone\)\*\*", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationIssue:
    """One validator finding, renderable as ``<severity>: <path>: <code>: <message>``."""

    severity: Literal["error", "warning"]
    code: str
    message: str
    path: str = ""

    def render(self) -> str:
        prefix = f"{self.path}: " if self.path else ""
        return f"{self.severity}: {prefix}{self.code}: {self.message}"


class SpecValidator:
    """Validate spec files against schema v2; returns issues, never raises."""

    _ALIAS_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
        SpecLoader._GOAL_HEADERS,
        SpecLoader._ACCEPTANCE_HEADERS,
        SpecLoader._CONSTRAINTS_HEADERS,
        SpecLoader._INVARIANTS_HEADERS,
        SpecLoader._OUT_OF_SCOPE_HEADERS,
    )

    def validate_paths(self, paths: Sequence[Path]) -> list[ValidationIssue]:
        """Validate every path, then cross-file rules over the whole set."""
        issues: list[ValidationIssue] = []
        ids_seen: dict[str, Path] = {}
        for path in paths:
            spec = self._load(path, issues)
            if spec is None:
                continue
            issues.extend(self.validate_spec(spec, path))
            if spec.id:
                # Resolve so the same file passed under two spellings
                # (specs/x.SPEC.md vs ./specs/x.SPEC.md) is not a duplicate.
                resolved = path.resolve()
                first = ids_seen.setdefault(spec.id, resolved)
                if first != resolved:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="duplicate-id",
                            message=f"spec id '{spec.id}' already declared by {first}",
                            path=str(path),
                        )
                    )
        return issues

    def validate_file(self, path: Path) -> list[ValidationIssue]:
        """Validate a single file (no cross-file rules)."""
        issues: list[ValidationIssue] = []
        spec = self._load(path, issues)
        if spec is not None:
            issues.extend(self.validate_spec(spec, path))
        return issues

    def validate_spec(self, spec: Spec, path: Path) -> list[ValidationIssue]:
        """Per-file schema checks on an already-parsed spec."""
        issues: list[ValidationIssue] = []

        def error(code: str, message: str) -> None:
            issues.append(ValidationIssue(severity="error", code=code, message=message, path=str(path)))

        def warning(code: str, message: str) -> None:
            issues.append(ValidationIssue(severity="warning", code=code, message=message, path=str(path)))

        if not spec.id:
            error("missing-id", "frontmatter must declare a unique 'id'")
        if not spec.status:
            error("missing-status", "frontmatter must declare 'status'")
        elif spec.status not in SPEC_STATUSES:
            allowed = ", ".join(sorted(SPEC_STATUSES))
            error("unknown-status", f"status '{spec.status}' is not one of: {allowed} (case-sensitive)")
        if not spec.goal:
            error("missing-goal", "spec has no goal (frontmatter 'goal:' or a '# Goal' section)")
        if not spec.criteria:
            error("no-criteria", "spec has no acceptance criteria bullets")
        if spec.id and path.name != f"{spec.id}{SPEC_SUFFIX}":
            error("filename-id-mismatch", f"file must be named '{spec.id}{SPEC_SUFFIX}' to match its id")
        if not spec.module:
            warning("missing-module", "frontmatter has no 'module' (repo-relative path prefix the spec governs)")

        self._check_sections(spec, error)
        self._check_done_markers(spec, error)
        self._check_criterion_ids(spec, error, warning)
        return issues

    def _load(self, path: Path, issues: list[ValidationIssue]) -> Spec | None:
        """Load a spec, converting parse/IO failures into error issues."""
        try:
            return SpecLoader().load(path)
        except SpecParseError as exc:
            issues.append(ValidationIssue(severity="error", code="parse-error", message=str(exc), path=str(path)))
        except (OSError, UnicodeDecodeError) as exc:
            # A directory from a shell glob, an unreadable file, or a binary /
            # non-UTF-8 file (UnicodeDecodeError is a ValueError, not an OSError).
            issues.append(ValidationIssue(severity="error", code="unreadable", message=str(exc), path=str(path)))
        return None

    def _check_sections(self, spec: Spec, error: _Report) -> None:
        """Reject duplicate headers and alias collisions the parser would silently pick between.

        Walks ``spec.body`` (never ``spec.raw``: frontmatter supports ``#``
        comment lines that are not headers), skipping fenced code blocks so a
        ``#`` shell/python comment inside ``` fences is not a false positive.
        Beyond the fence-skip the walk mirrors ``_split_sections`` semantics
        (any ``#``-prefixed line, no space required) so every header the parser
        would act on stays visible to the collision checks. Alias hits are
        counted per alias *group* — e.g. one ``# Constraints (…)`` header
        prefix-matches both ``constraints`` and ``constraint`` and must not
        self-collide.
        """
        titles: list[str] = []
        in_fence = False
        for line in spec.body.splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence and line.startswith("#"):
                titles.append(line.lstrip("#").strip().lower())
        seen: set[str] = set()
        for title in titles:
            if title in seen:
                error("duplicate-section", f"section header '{title}' appears more than once")
            seen.add(title)
        for group in self._ALIAS_GROUPS:
            matching = [t for t in set(titles) if t in group or any(t.startswith(alias) for alias in group)]
            if len(matching) > 1:
                error(
                    "duplicate-section",
                    f"headers {sorted(matching)} all match the '{group[0]}' section; keep exactly one",
                )

    @staticmethod
    def _check_done_markers(spec: Spec, error: _Report) -> None:
        for line_no, line in enumerate(spec.raw.splitlines(), start=1):
            if DONE_MARKER_RE.search(line):
                error(
                    "done-marker",
                    f"line {line_no}: inline done-marker found — status lives only in frontmatter, "
                    "specs state future contracts",
                )

    @staticmethod
    def _check_criterion_ids(spec: Spec, error: _Report, warning: _Report) -> None:
        authored = [c.id for c in spec.criteria if not _is_positional(c.id)]
        if authored:
            if len(authored) != len(spec.criteria):
                error("mixed-criterion-ids", "some acceptance bullets have 'AC-n:' IDs and some do not")
            seen: set[str] = set()
            for cid in authored:
                if cid in seen:
                    error("duplicate-criterion-id", f"criterion id '{cid}' appears more than once")
                seen.add(cid)
        elif spec.criteria:
            warning(
                "positional-criterion-ids",
                "no acceptance bullet carries an 'AC-n:' id; positional ids (c0, c1, ...) are fragile",
            )


def _is_positional(criterion_id: str) -> bool:
    return criterion_id.startswith("c") and criterion_id[1:].isdigit()


__all__ = ["SPEC_STATUSES", "DONE_MARKER_RE", "SpecValidator", "ValidationIssue"]
