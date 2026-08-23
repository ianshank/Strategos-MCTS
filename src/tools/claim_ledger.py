"""Deterministic validation of the repository's capability claim ledger.

``docs/CLAIM_LEDGER.md`` grades every capability claim this project makes in public prose against
executable evidence. Prose alone cannot be trusted to stay true: ``CHARTER.md`` section 2 asserted
that the MCTS engines agree on negamax sign handling while three of the four engines disagreed with
each other in the tree. This module makes the ledger's own rules mechanical, so a grade is a
derived fact rather than an author's opinion.

The central rule (``docs/plans/EVIDENCE_FIRST_PROGRAM.md`` section 4, R1) is that
:data:`~src.config.constants.CLAIM_GRADE_PROVEN` requires a resolvable evidence artefact **and** a
verification command. There is deliberately no flag, environment variable, or allowlist that
relaxes it: an escape hatch on an evidence gate is the same thing as not having the gate.

Design notes:

- **Deterministic and dependency-light.** Pure filesystem plus parsing; no network, no LLM, and no
  imports from the framework, training, or API packages, so it runs on a default install and in CI
  before heavy dependencies exist. It uses the stdlib :mod:`logging` module rather than
  ``src.observability`` for the same reason ``src/tools/context_docs.py`` does — there are no
  secrets or outbound requests here, and keeping the module importable in a minimal environment is
  the point. Do not "fix" this to import the observability stack.
- **No hardcoded values.** The grade vocabulary, the promotion rule, the column contract, and the
  sentinel are all data in ``src/config/constants.py``, shared with
  :mod:`src.tools.status_artifact`. Adding a grade is a constant edit, not a control-flow edit.
- **Reusable.** The engine is a class parameterised by ``repo_root`` and ``ledger_path`` (both
  injected for tests), and it returns structured :class:`LedgerFailure` objects rather than
  pre-formatted strings, so callers can aggregate or re-render them.

Run standalone (exit 1 on any failure)::

    python -m src.tools.claim_ledger              # or the `claim-ledger` console script
    python -m src.tools.claim_ledger --json       # machine-readable ledger + verdict
    python -m src.tools.claim_ledger --debug      # per-row decision trace

Wrapped by ``tests/unit/tools/test_claim_ledger.py`` so drift fails CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.config.constants import (
    CLAIM_CELL_EMPTY_SENTINEL,
    CLAIM_GRADE_FALSE,
    CLAIM_GRADE_PROVEN,
    CLAIM_GRADES,
    CLAIM_GRADES_REQUIRING_EVIDENCE,
    CLAIM_GRADES_REQUIRING_NOTES,
    CLAIM_GRADES_REQUIRING_VERIFY,
    CLAIM_ID_PREFIX,
    CLAIM_LEDGER_COLUMNS,
    CLAIM_LEDGER_RELATIVE_PATH,
    CLAIM_SURFACE_BASELINE_RELATIVE_PATH,
    CLAIM_SURFACE_BASELINE_SCHEMA_VERSION,
    CLAIM_SURFACE_BULLET_PATTERN,
    CLAIM_SURFACE_KEY_BULLETS,
    CLAIM_SURFACE_KEY_PATH,
    CLAIM_SURFACE_KEY_ROWS,
    CLAIM_SURFACE_KEY_SECTION,
    CLAIM_SURFACE_KEY_SURFACES,
    CLAIM_SURFACE_KEY_SURPLUS,
)

__all__ = [
    "ClaimRow",
    "LedgerFailure",
    "LedgerReport",
    "SurfaceCoverage",
    "ClaimLedgerValidator",
    "validate",
    "main",
]

logger = logging.getLogger(__name__)

# This module lives at <repo>/src/tools/claim_ledger.py.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``CL-<digits>``, anchored so ``CL-1a`` or ``XCL-1`` are rejected rather than silently accepted.
_CLAIM_ID = re.compile(rf"^{re.escape(CLAIM_ID_PREFIX)}(\d+)$")

#: A backticked token, used to unwrap ``Verify`` / ``Evidence`` cells written as inline code.
_BACKTICKED = re.compile(r"^`(.*)`$", re.S)

#: A markdown table separator row (``|---|---|``), tolerant of alignment colons and padding.
_SEPARATOR_ROW = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+$")

#: A claim-shaped bullet on a reader-facing surface (``- **Capability**: …``).
_CLAIM_BULLET = re.compile(CLAIM_SURFACE_BULLET_PATTERN)

#: An ATX heading, capturing its level so a section ends at the next heading of equal or higher rank.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

#: Tokens in a ``Notes`` cell that count as citing a location: a rooted repo path, optionally with a
#: ``:line`` or ``:start-end`` anchor. Deliberately stricter than "mentions a filename" so a FALSE
#: row cannot be satisfied by hand-waving at a module.
_LOCATION_CITATION = re.compile(
    r"`[\w./-]+/[\w.-]+\.(?:py|md|toml|ya?ml|json|cfg|sh)(?::\d+(?:-\d+)?)?`|"
    r"`[\w.-]+\.(?:py|md|toml|ya?ml|json|cfg|sh):\d+(?:-\d+)?`"
)


@dataclass(frozen=True)
class ClaimRow:
    """One parsed ledger row.

    ``line`` is the 1-based line number in the ledger, carried so failures point a human at the
    exact row instead of making them grep for a claim id.
    """

    claim_id: str
    claim: str
    source: str
    grade: str
    verify: str
    evidence: str
    notes: str
    section: str
    line: int

    @property
    def ordinal(self) -> int:
        """Numeric part of the claim id, for stable sorting independent of string collation."""
        match = _CLAIM_ID.match(self.claim_id)
        return int(match.group(1)) if match else -1

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view, key order fixed so ``--json`` output is byte-stable."""
        return {
            "id": self.claim_id,
            "claim": self.claim,
            "source": self.source,
            "grade": self.grade,
            "verify": self.verify,
            "evidence": self.evidence,
            "notes": self.notes,
            "section": self.section,
            "line": self.line,
        }


@dataclass(frozen=True)
class LedgerFailure:
    """One validation problem.

    ``category`` is one of ``structure`` | ``schema`` | ``path`` | ``promotion`` | ``notes``.
    """

    claim_id: str
    category: str
    message: str
    line: int = 0

    def __str__(self) -> str:
        where = f"{CLAIM_LEDGER_RELATIVE_PATH}:{self.line}" if self.line else CLAIM_LEDGER_RELATIVE_PATH
        subject = self.claim_id or "<ledger>"
        return f"[{self.category}] {where} {subject}: {self.message}"

    def as_dict(self) -> dict[str, object]:
        return {"id": self.claim_id, "category": self.category, "message": self.message, "line": self.line}


@dataclass(frozen=True)
class LedgerReport:
    """Outcome of a validation run: the parsed rows, the failures, and the derived grade counts."""

    rows: tuple[ClaimRow, ...] = ()
    failures: tuple[LedgerFailure, ...] = ()
    grade_counts: dict[str, int] = field(default_factory=dict)
    surfaces: tuple[SurfaceCoverage, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view with a fixed key order, so two runs at one commit are identical."""
        return {
            "ledger": CLAIM_LEDGER_RELATIVE_PATH,
            "ok": self.ok,
            "claim_count": len(self.rows),
            "grade_counts": {grade: self.grade_counts.get(grade, 0) for grade in CLAIM_GRADES},
            "claims": [row.as_dict() for row in self.rows],
            "surfaces": [coverage.as_dict() for coverage in self.surfaces],
            "failures": [failure.as_dict() for failure in self.failures],
        }


@dataclass(frozen=True)
class SurfaceCoverage:
    """Measured claim coverage for one reader-facing surface.

    ``surplus`` is the ratcheted quantity: claim-shaped bullets on the surface that no ledger row
    accounts for. ``bullets`` and ``rows`` are recorded for a human reading the baseline; only
    ``surplus`` gates.
    """

    path: str
    section: str
    bullets: int
    rows: int

    @property
    def surplus(self) -> int:
        return max(self.bullets - self.rows, 0)

    def as_dict(self) -> dict[str, object]:
        return {
            CLAIM_SURFACE_KEY_PATH: self.path,
            CLAIM_SURFACE_KEY_SECTION: self.section,
            CLAIM_SURFACE_KEY_BULLETS: self.bullets,
            CLAIM_SURFACE_KEY_ROWS: self.rows,
            CLAIM_SURFACE_KEY_SURPLUS: self.surplus,
        }


def _section_lines(text: str, section: str) -> list[str] | None:
    """Return the lines of the named section, or ``None`` when no heading starts with it.

    Matching is by heading prefix so ``"2. Mission"`` finds
    ``## 2. Mission (what we are building)`` without pinning the parenthetical. The section ends at
    the next heading of the same or higher level, which is what stops a nested ``###`` subsection
    from truncating the region.
    """
    collected: list[str] = []
    level = 0
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading is None:
            if level:
                collected.append(line)
            continue
        depth, title = len(heading.group(1)), heading.group(2)
        if level and depth <= level:
            break
        if not level and title.startswith(section):
            level = depth
    return collected if level else None


def _unwrap_cell(cell: str) -> str:
    """Normalise a table cell: trim, then strip a single enclosing backtick pair.

    Ledger authors write commands and paths as inline code. Unwrapping here means the rest of the
    validator never has to care about the presentation.
    """
    text = cell.strip()
    match = _BACKTICKED.match(text)
    return match.group(1).strip() if match else text


def _is_empty(cell: str) -> bool:
    """True when a cell asserts nothing — blank or the explicit sentinel."""
    return cell == "" or cell == CLAIM_CELL_EMPTY_SENTINEL


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into unwrapped cells, dropping the leading/trailing delimiters."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_unwrap_cell(cell) for cell in stripped.split("|")]


class ClaimLedgerValidator:
    """Validate a claim ledger under a given ``repo_root``.

    Both ``repo_root`` and ``ledger_path`` are injectable so tests can build a synthetic ledger in a
    ``tmp_path`` tree rather than mutating the real one — the same seam
    :class:`src.tools.context_docs.ContextDocValidator` uses.
    """

    def __init__(self, repo_root: Path | str | None = None, ledger_path: Path | str | None = None) -> None:
        self.repo = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT
        self.ledger = Path(ledger_path) if ledger_path is not None else self.repo / CLAIM_LEDGER_RELATIVE_PATH
        self.surface_baseline = self.repo / CLAIM_SURFACE_BASELINE_RELATIVE_PATH

    # -- parsing ---------------------------------------------------------------------------------

    def parse(self) -> tuple[list[ClaimRow], list[LedgerFailure]]:
        """Parse ledger tables into rows.

        Only tables whose header matches :data:`~src.config.constants.CLAIM_LEDGER_COLUMNS` exactly
        are treated as ledger tables, so the document can carry explanatory tables (the grade
        legend) without the parser mistaking them for claims. The most recent ``##`` heading is
        recorded as each row's ``section``, which is what makes the artifact's per-area rollup
        possible without a second source of truth.
        """
        rows: list[ClaimRow] = []
        failures: list[LedgerFailure] = []

        if not self.ledger.is_file():
            failures.append(
                LedgerFailure(
                    "", "structure", f"ledger not found at {self.ledger} (expected {CLAIM_LEDGER_RELATIVE_PATH})"
                )
            )
            return rows, failures

        section = ""
        in_table = False
        expected_width = len(CLAIM_LEDGER_COLUMNS)

        for number, raw in enumerate(self.ledger.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.rstrip()

            if line.startswith("## "):
                section = line[3:].strip()
                in_table = False
                continue

            if not line.startswith("|"):
                in_table = False
                continue

            if _SEPARATOR_ROW.match(line):
                continue

            cells = _split_row(line)

            if tuple(cells) == CLAIM_LEDGER_COLUMNS:
                logger.debug("ledger table header at line %d in section %r", number, section)
                in_table = True
                continue

            if not in_table:
                continue

            if len(cells) != expected_width:
                failures.append(
                    LedgerFailure(
                        cells[0] if cells else "",
                        "structure",
                        f"row has {len(cells)} cell(s), expected {expected_width} ({', '.join(CLAIM_LEDGER_COLUMNS)})",
                        number,
                    )
                )
                continue

            row = ClaimRow(
                claim_id=cells[0],
                claim=cells[1],
                source=cells[2],
                grade=cells[3],
                verify=cells[4],
                evidence=cells[5],
                notes=cells[6],
                section=section,
                line=number,
            )
            logger.debug("parsed %s grade=%s section=%r", row.claim_id, row.grade, row.section)
            rows.append(row)

        if not rows:
            failures.append(
                LedgerFailure("", "structure", "no claim rows found; a ledger with no claims proves nothing")
            )

        return rows, failures

    # -- checks ----------------------------------------------------------------------------------

    def check_ids(self, rows: Sequence[ClaimRow]) -> list[LedgerFailure]:
        """Every id matches ``CL-<n>`` and is unique."""
        failures: list[LedgerFailure] = []
        seen: dict[str, int] = {}
        for row in rows:
            if not _CLAIM_ID.match(row.claim_id):
                failures.append(
                    LedgerFailure(
                        row.claim_id, "schema", f"malformed claim id (expected {CLAIM_ID_PREFIX}<n>)", row.line
                    )
                )
                continue
            if row.claim_id in seen:
                failures.append(
                    LedgerFailure(
                        row.claim_id,
                        "schema",
                        f"duplicate claim id (first seen at line {seen[row.claim_id]})",
                        row.line,
                    )
                )
                continue
            seen[row.claim_id] = row.line
        return failures

    def check_grades(self, rows: Sequence[ClaimRow]) -> list[LedgerFailure]:
        """Every grade is in the declared vocabulary, and the claim text is non-empty."""
        failures: list[LedgerFailure] = []
        for row in rows:
            if row.grade not in CLAIM_GRADES:
                failures.append(
                    LedgerFailure(
                        row.claim_id,
                        "schema",
                        f"unknown grade {row.grade!r} (expected one of {', '.join(CLAIM_GRADES)})",
                        row.line,
                    )
                )
            if not row.claim.strip():
                failures.append(LedgerFailure(row.claim_id, "schema", "empty claim text", row.line))
        return failures

    def check_paths(self, rows: Sequence[ClaimRow]) -> list[LedgerFailure]:
        """Cited ``Source`` and ``Evidence`` paths resolve on disk.

        The sentinel is accepted for both: a process claim may have no single prose home, and only
        the grades in :data:`~src.config.constants.CLAIM_GRADES_REQUIRING_EVIDENCE` are required to
        carry an artefact. Unresolvable *asserted* paths always fail, whatever the grade — a broken
        citation is a broken citation.
        """
        failures: list[LedgerFailure] = []
        for row in rows:
            for column, value in (("Source", row.source), ("Evidence", row.evidence)):
                if _is_empty(value):
                    continue
                if not (self.repo / value).exists():
                    failures.append(
                        LedgerFailure(row.claim_id, "path", f"{column} path does not resolve: {value}", row.line)
                    )
        return failures

    def check_promotion_rule(self, rows: Sequence[ClaimRow]) -> list[LedgerFailure]:
        """Enforce R1 structurally.

        A grade in :data:`~src.config.constants.CLAIM_GRADES_REQUIRING_EVIDENCE` needs a resolvable
        evidence artefact *and* a non-empty verification command; grades in
        :data:`~src.config.constants.CLAIM_GRADES_REQUIRING_VERIFY` need the command. There is no
        parameter that turns this off, which is the whole point: the grade is derived from the tree.
        """
        failures: list[LedgerFailure] = []
        for row in rows:
            if row.grade in CLAIM_GRADES_REQUIRING_EVIDENCE:
                if _is_empty(row.evidence):
                    failures.append(
                        LedgerFailure(
                            row.claim_id,
                            "promotion",
                            f"grade {row.grade} requires an Evidence artefact; the grade is derived from evidence, "
                            "not asserted (see docs/plans/EVIDENCE_FIRST_PROGRAM.md R1)",
                            row.line,
                        )
                    )
                if _is_empty(row.source):
                    failures.append(
                        LedgerFailure(
                            row.claim_id, "promotion", f"grade {row.grade} requires a Source document", row.line
                        )
                    )
            if row.grade in CLAIM_GRADES_REQUIRING_VERIFY and _is_empty(row.verify):
                failures.append(
                    LedgerFailure(row.claim_id, "promotion", f"grade {row.grade} requires a Verify command", row.line)
                )
        return failures

    def check_notes(self, rows: Sequence[ClaimRow]) -> list[LedgerFailure]:
        """Grades that assert an absence must say what is absent; FALSE must cite a location."""
        failures: list[LedgerFailure] = []
        for row in rows:
            if row.grade in CLAIM_GRADES_REQUIRING_NOTES and _is_empty(row.notes):
                failures.append(
                    LedgerFailure(
                        row.claim_id, "notes", f"grade {row.grade} requires Notes stating the missing link", row.line
                    )
                )
                continue
            if row.grade == CLAIM_GRADE_FALSE and not _LOCATION_CITATION.search(row.notes):
                failures.append(
                    LedgerFailure(
                        row.claim_id,
                        "notes",
                        f"grade {CLAIM_GRADE_FALSE} requires Notes citing the contradicting location "
                        "(a backticked repo path, optionally with a :line anchor)",
                        row.line,
                    )
                )
        return failures

    # -- orchestration ---------------------------------------------------------------------------

    # -- surface coverage ------------------------------------------------------------------------

    def load_surface_baseline(self) -> tuple[list[dict[str, object]], list[LedgerFailure]]:
        """Read the committed surface baseline. A missing or malformed file is a failure, not a zero.

        Defaulting to an empty baseline would silently disable the ratchet, which is the one outcome
        a coverage gate must never produce.
        """
        failures: list[LedgerFailure] = []
        if not self.surface_baseline.is_file():
            failures.append(
                LedgerFailure(
                    "",
                    "surface",
                    f"surface baseline {CLAIM_SURFACE_BASELINE_RELATIVE_PATH} is missing; "
                    f"regenerate with: make claims-baseline",
                )
            )
            return [], failures
        try:
            payload = json.loads(self.surface_baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(LedgerFailure("", "surface", f"surface baseline is unreadable: {exc}"))
            return [], failures

        version = payload.get("schema_version") if isinstance(payload, dict) else None
        if version != CLAIM_SURFACE_BASELINE_SCHEMA_VERSION:
            failures.append(
                LedgerFailure(
                    "",
                    "surface",
                    f"surface baseline schema_version {version!r} is not the supported "
                    f"{CLAIM_SURFACE_BASELINE_SCHEMA_VERSION}",
                )
            )
            return [], failures

        surfaces = payload.get(CLAIM_SURFACE_KEY_SURFACES)
        if not isinstance(surfaces, list) or not surfaces:
            failures.append(LedgerFailure("", "surface", "surface baseline declares no surfaces"))
            return [], failures
        return [entry for entry in surfaces if isinstance(entry, dict)], failures

    def measure_surfaces(self, rows: Sequence[ClaimRow]) -> tuple[list[SurfaceCoverage], list[LedgerFailure]]:
        """Count claim-shaped bullets and ledger rows for every surface the baseline declares."""
        entries, failures = self.load_surface_baseline()
        measured: list[SurfaceCoverage] = []

        for entry in entries:
            relative = str(entry.get(CLAIM_SURFACE_KEY_PATH, ""))
            section = str(entry.get(CLAIM_SURFACE_KEY_SECTION, ""))
            document = self.repo / relative
            if not relative or not document.is_file():
                failures.append(LedgerFailure("", "surface", f"declared surface {relative!r} does not resolve on disk"))
                continue

            lines = _section_lines(document.read_text(encoding="utf-8"), section) if section else None
            if section and lines is None:
                failures.append(LedgerFailure("", "surface", f"{relative}: no heading starts with {section!r}"))
                continue
            if lines is None:
                lines = document.read_text(encoding="utf-8").splitlines()

            bullets = sum(1 for line in lines if _CLAIM_BULLET.match(line))
            graded = sum(1 for row in rows if row.source == relative)
            coverage = SurfaceCoverage(path=relative, section=section, bullets=bullets, rows=graded)
            logger.debug(
                "surface %s [%s]: %d claim bullet(s), %d ledger row(s), surplus %d",
                relative,
                section or "<whole file>",
                bullets,
                graded,
                coverage.surplus,
            )
            measured.append(coverage)

        return measured, failures

    def check_surface_coverage(self, rows: Sequence[ClaimRow]) -> tuple[list[SurfaceCoverage], list[LedgerFailure]]:
        """Ratchet the ungraded surplus per surface: it may fall, never rise, and slack is staleness.

        Two directions both fail, for the same reason the action-pin ratchet treats slack as a
        violation: a baseline that no longer describes the tree cannot be relied on to catch the next
        regression.
        """
        measured, failures = self.measure_surfaces(rows)
        entries, _ = self.load_surface_baseline()
        recorded = {
            str(entry.get(CLAIM_SURFACE_KEY_PATH, "")): entry.get(CLAIM_SURFACE_KEY_SURPLUS) for entry in entries
        }

        for coverage in measured:
            expected = recorded.get(coverage.path)
            if not isinstance(expected, int):
                failures.append(
                    LedgerFailure(
                        "",
                        "surface",
                        f"{coverage.path}: baseline records no integer {CLAIM_SURFACE_KEY_SURPLUS}",
                    )
                )
                continue
            if coverage.surplus > expected:
                failures.append(
                    LedgerFailure(
                        "",
                        "surface",
                        f"{coverage.path} section {coverage.section!r}: {coverage.surplus} claim bullet(s) "
                        f"have no ledger row, above the committed {expected}. Add a ledger row for the new "
                        f"claim, or drop the claim.",
                    )
                )
            elif coverage.surplus < expected:
                failures.append(
                    LedgerFailure(
                        "",
                        "surface",
                        f"{coverage.path}: surplus fell to {coverage.surplus} but the baseline still says "
                        f"{expected}; the baseline is stale. Regenerate with: make claims-baseline",
                    )
                )
        return measured, failures

    def write_surface_baseline(self) -> list[SurfaceCoverage]:
        """Regenerate the baseline from the current tree and return what was written."""
        rows, _ = self.parse()
        measured, _ = self.measure_surfaces(rows)
        payload = {
            "schema_version": CLAIM_SURFACE_BASELINE_SCHEMA_VERSION,
            "_README": (
                "Ungraded claim-surface surplus per reader-facing surface. A claim-shaped bullet "
                "is a list item whose first token is bold. 'ungraded_surplus' is the gated number: "
                "it may only decrease, and slack means the baseline is stale. The surface list is "
                "curated by hand; regeneration only refreshes the counts of surfaces already "
                "declared here. Regenerate with: make claims-baseline"
            ),
            CLAIM_SURFACE_KEY_SURFACES: [coverage.as_dict() for coverage in measured],
        }
        self.surface_baseline.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s with %d surface(s)", CLAIM_SURFACE_BASELINE_RELATIVE_PATH, len(measured))
        return measured

    def validate(self) -> LedgerReport:
        """Run every check and return a structured report. Never raises on ledger content."""
        rows, failures = self.parse()
        if rows:
            for check in (
                self.check_ids,
                self.check_grades,
                self.check_paths,
                self.check_promotion_rule,
                self.check_notes,
            ):
                found = check(rows)
                logger.debug("%s produced %d failure(s)", check.__name__, len(found))
                failures.extend(found)

        # Coverage runs even with zero rows: an empty ledger against a populated surface is exactly
        # the regression the ratchet exists to catch.
        surfaces, surface_failures = self.check_surface_coverage(rows)
        logger.debug("check_surface_coverage produced %d failure(s)", len(surface_failures))
        failures.extend(surface_failures)

        ordered_rows = tuple(sorted(rows, key=lambda row: (row.ordinal, row.claim_id)))
        ordered_failures = tuple(sorted(failures, key=lambda item: (item.line, item.category, item.claim_id)))
        counts = {grade: sum(1 for row in ordered_rows if row.grade == grade) for grade in CLAIM_GRADES}

        if ordered_failures:
            logger.info(
                "claim ledger INVALID: %d claim(s), %d failure(s)",
                len(ordered_rows),
                len(ordered_failures),
            )
        else:
            logger.info(
                "claim ledger valid: %d claim(s), %d %s",
                len(ordered_rows),
                counts.get(CLAIM_GRADE_PROVEN, 0),
                CLAIM_GRADE_PROVEN,
            )
        return LedgerReport(
            rows=ordered_rows,
            failures=ordered_failures,
            grade_counts=counts,
            surfaces=tuple(sorted(surfaces, key=lambda item: item.path)),
        )


def validate(repo_root: Path | str | None = None, ledger_path: Path | str | None = None) -> LedgerReport:
    """Convenience wrapper so callers (tests, the status artifact) need not build the class."""
    return ClaimLedgerValidator(repo_root, ledger_path).validate()


def _render(failures: Iterable[LedgerFailure]) -> None:
    print("Claim-ledger validation FAILED:\n", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claim-ledger",
        description="Deterministically validate docs/CLAIM_LEDGER.md against the live tree.",
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="repo root to validate (default: this repo)")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="ledger path (default: <repo-root>/" + CLAIM_LEDGER_RELATIVE_PATH + ")",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO-level trace to stderr")
    parser.add_argument("--debug", action="store_true", help="DEBUG-level trace (per-row decisions)")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="emit the parsed ledger and verdict as JSON"
    )
    parser.add_argument(
        "--write-surface-baseline",
        action="store_true",
        dest="write_surface_baseline",
        help="regenerate " + CLAIM_SURFACE_BASELINE_RELATIVE_PATH + " from the current tree, then exit",
    )
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    validator = ClaimLedgerValidator(args.repo_root, args.ledger)

    if args.write_surface_baseline:
        written = validator.write_surface_baseline()
        for coverage in written:
            print(
                f"{coverage.path} [{coverage.section or '<whole file>'}]: "
                f"{coverage.bullets} claim bullet(s), {coverage.rows} ledger row(s), "
                f"surplus {coverage.surplus}"
            )
        print(f"Wrote {CLAIM_SURFACE_BASELINE_RELATIVE_PATH} ({len(written)} surface(s)).")
        return 0

    report = validator.validate()

    if args.as_json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=False))
    elif not report.ok:
        _render(report.failures)
    else:
        summary = ", ".join(f"{report.grade_counts.get(grade, 0)} {grade}" for grade in reversed(CLAIM_GRADES))
        print(f"Claim-ledger validation OK — {len(report.rows)} claim(s): {summary}.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
