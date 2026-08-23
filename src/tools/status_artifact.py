"""Generate ``artifacts/status.json`` — a provenance-stamped snapshot of what this tree can prove.

``docs/STATUS.md`` records measured test and coverage figures for humans. It cannot answer the
question that matters when a result is quoted six months later: *how was this produced?* A number
from a mock LLM, a randomly-initialised network, and a trained checkpoint look identical once
they reach a slide. This module emits a machine-readable artifact in which
:class:`ResultEntry` provenance is a required field, so the distinction cannot be dropped.

What the artifact carries:

- **Environment** — commit sha, working-tree dirty flag, Python version, platform, and which
  optional-dependency extras are actually importable. A result produced without the ``neural``
  extra is a different claim from one produced with it.
- **Coverage** — the ``fail_under`` gate *read from* ``pyproject.toml`` (never duplicated here) and
  the measured total when a ``coverage json`` report is supplied.
- **Claims** — the per-grade counts derived from ``docs/CLAIM_LEDGER.md`` via
  :mod:`src.tools.claim_ledger`, plus that validator's verdict.
- **Capability maturity** — the declarative map in ``docs/capability_maturity.json``, checked
  against the ledger so a capability cannot advertise a stage its own claims contradict.
- **Results** — zero or more :class:`ResultEntry` items, each with a provenance drawn from a closed
  vocabulary.

Determinism (``docs/plans/EVIDENCE_FIRST_PROGRAM.md`` section 4, R1): the clock is injected, key
order is fixed, and no wall-clock or environment value is read except through an explicit argument.
Two invocations at one commit with one clock produce byte-identical output.

Dependency-light by design: stdlib plus :mod:`src.config.constants` and
:mod:`src.tools.claim_ledger` only, so it runs in CI before heavy dependencies are installed. It
uses stdlib :mod:`logging` for the same reason ``src/tools/context_docs.py`` does.

Run standalone::

    python -m src.tools.status_artifact                     # or the `status-artifact` script
    python -m src.tools.status_artifact --coverage-json coverage.json
    python -m src.tools.status_artifact --stdout --now 1970-01-01T00:00:00+00:00

Wrapped by ``tests/unit/tools/test_status_artifact.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform as platform_module
import subprocess  # nosec B404 - local `git` metadata only; no shell, fixed argv, no user input.
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from src.config.constants import (
    CAPABILITY_MATURITY_RELATIVE_PATH,
    CAPABILITY_MATURITY_STAGES,
    CLAIM_GRADE_MATURITY_CEILING,
    CLAIM_GRADES,
    COVERAGE_GATE_PYPROJECT_PATH,
    EVIDENCE_PROVENANCES,
    OPTIONAL_EXTRA_PROBE_MODULES,
    STATUS_ARTIFACT_RELATIVE_PATH,
    STATUS_ARTIFACT_SCHEMA_VERSION,
)
from src.tools.claim_ledger import ClaimLedgerValidator, ClaimRow, LedgerReport

__all__ = [
    "ProvenanceError",
    "ResultEntry",
    "StatusArtifactBuilder",
    "build",
    "main",
]

logger = logging.getLogger(__name__)

# This module lives at <repo>/src/tools/status_artifact.py.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Timeout for the local ``git`` metadata calls. A hung subprocess must not hang CI.
_GIT_TIMEOUT_SECONDS = 15

#: Sentinel recorded when a value could not be determined. Explicit, so a consumer can tell
#: "unknown" from "zero" — silently emitting 0 for an unmeasured coverage figure would be exactly
#: the kind of unfalsifiable number this artifact exists to prevent.
_UNKNOWN = "unknown"

_STAGE_INDEX = {stage: index for index, stage in enumerate(CAPABILITY_MATURITY_STAGES)}
_GRADE_CEILING = dict(CLAIM_GRADE_MATURITY_CEILING)

# CLAIM_GRADES is ordered worst-to-best, so its index is the strength rank of a grade.
_GRADE_RANK = {grade: rank for rank, grade in enumerate(CLAIM_GRADES)}


class ProvenanceError(ValueError):
    """Raised when a result entry omits or misstates how it was produced."""


@dataclass(frozen=True)
class ResultEntry:
    """One reported measurement, with a mandatory provenance.

    ``provenance`` must be a member of :data:`~src.config.constants.EVIDENCE_PROVENANCES`. The
    check is in ``__post_init__`` rather than at serialisation time so an invalid entry cannot be
    constructed and passed around, and so the failure names the offending entry.
    """

    name: str
    value: float | int | str
    provenance: str
    command: str = ""
    artifact: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ProvenanceError("result entry requires a name")
        if not self.provenance:
            raise ProvenanceError(
                f"result {self.name!r} omits provenance; one of {', '.join(EVIDENCE_PROVENANCES)} is required"
            )
        if self.provenance not in EVIDENCE_PROVENANCES:
            raise ProvenanceError(
                f"result {self.name!r} has unknown provenance {self.provenance!r}; "
                f"expected one of {', '.join(EVIDENCE_PROVENANCES)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "provenance": self.provenance,
            "command": self.command,
            "artifact": self.artifact,
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ResultEntry:
        """Build an entry from an untrusted mapping (e.g. a JSON file passed on the CLI)."""
        if "provenance" not in payload:
            raise ProvenanceError(
                f"result {payload.get('name', '<unnamed>')!r} omits provenance; "
                f"one of {', '.join(EVIDENCE_PROVENANCES)} is required"
            )
        return cls(
            name=str(payload.get("name", "")),
            value=payload.get("value", ""),
            provenance=str(payload["provenance"]),
            command=str(payload.get("command", "")),
            artifact=str(payload.get("artifact", "")),
            notes=str(payload.get("notes", "")),
        )


def _git(repo: Path, *args: str) -> str | None:
    """Run a read-only local ``git`` command, returning stripped stdout or ``None`` on any failure.

    No shell, fixed argv, and a timeout: this is metadata collection, and it must degrade to
    ``unknown`` rather than break a build when run from an exported tarball with no ``.git``.
    """
    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, no shell, no user input
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed: %s", " ".join(args), exc)
        return None
    if completed.returncode != 0:
        logger.debug("git %s exited %d: %s", " ".join(args), completed.returncode, completed.stderr.strip())
        return None
    return completed.stdout.strip()


class StatusArtifactBuilder:
    """Assemble the status artifact.

    Every input is injectable: ``repo_root`` locates the tree, ``now`` fixes the clock (which is
    what makes the output byte-stable), ``coverage_json`` supplies a measured figure, and
    ``ledger_path`` / ``maturity_path`` relocate the two data sources for tests.
    """

    def __init__(
        self,
        repo_root: Path | str | None = None,
        *,
        now: datetime | None = None,
        coverage_json: Path | str | None = None,
        ledger_path: Path | str | None = None,
        maturity_path: Path | str | None = None,
    ) -> None:
        self.repo = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT
        self.now = now
        self.coverage_json = Path(coverage_json) if coverage_json is not None else None
        self.ledger_path = ledger_path
        self.maturity_path = (
            Path(maturity_path) if maturity_path is not None else self.repo / CAPABILITY_MATURITY_RELATIVE_PATH
        )

    # -- inputs ----------------------------------------------------------------------------------

    def coverage_gate(self) -> float | str:
        """Read the enforced coverage threshold from ``pyproject.toml``.

        Read, never duplicated: a literal here could drift from the gate CI actually enforces, and a
        status artifact that misreports its own gate is worse than one that omits it.
        """
        pyproject = self.repo / "pyproject.toml"
        if not pyproject.is_file():
            logger.debug("no pyproject.toml at %s", pyproject)
            return _UNKNOWN
        try:
            data: Any = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("could not parse %s: %s", pyproject, exc)
            return _UNKNOWN
        for key in COVERAGE_GATE_PYPROJECT_PATH:
            if not isinstance(data, Mapping) or key not in data:
                logger.warning(
                    "coverage gate key %s missing from pyproject.toml", ".".join(COVERAGE_GATE_PYPROJECT_PATH)
                )
                return _UNKNOWN
            data = data[key]
        return float(data) if isinstance(data, (int, float)) else _UNKNOWN

    def measured_coverage(self) -> float | str:
        """Total percent covered from a ``coverage json`` report, or ``unknown`` when absent."""
        if self.coverage_json is None:
            return _UNKNOWN
        if not self.coverage_json.is_file():
            logger.warning("coverage report not found: %s", self.coverage_json)
            return _UNKNOWN
        try:
            payload = json.loads(self.coverage_json.read_text(encoding="utf-8"))
            return float(payload["totals"]["percent_covered"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("could not read coverage total from %s: %s", self.coverage_json, exc)
            return _UNKNOWN

    def detected_extras(self) -> list[str]:
        """Which optional extras are importable here, probed from the declarative constant."""
        found: list[str] = []
        for extra, module in OPTIONAL_EXTRA_PROBE_MODULES:
            try:
                present = find_spec(module) is not None
            except (ImportError, ValueError):  # pragma: no cover - defensive; malformed sys.path
                present = False
            logger.debug("extra %s (probe %s): %s", extra, module, "present" if present else "absent")
            if present:
                found.append(extra)
        return sorted(found)

    def environment(self) -> dict[str, Any]:
        """Commit, dirtiness, interpreter, platform, and extras."""
        sha = _git(self.repo, "rev-parse", "HEAD")
        status = _git(self.repo, "status", "--porcelain")
        return {
            "commit": sha or _UNKNOWN,
            "dirty": bool(status) if status is not None else _UNKNOWN,
            "branch": _git(self.repo, "rev-parse", "--abbrev-ref", "HEAD") or _UNKNOWN,
            "python": platform_module.python_version(),
            "platform": platform_module.platform(terse=True),
            "extras": self.detected_extras(),
        }

    # -- maturity --------------------------------------------------------------------------------

    def capability_maturity(self, rows: Sequence[ClaimRow]) -> tuple[list[dict[str, Any]], list[str]]:
        """Load the declarative maturity map and check it against the ledger.

        Two structural checks, both of which make the map falsifiable rather than decorative:

        1. every cited claim id exists in the ledger; and
        2. the declared stage does not exceed the ceiling implied by the weakest supporting grade
           (:data:`~src.config.constants.CLAIM_GRADE_MATURITY_CEILING`).

        Returns the rendered rows and a list of problems. Problems are returned rather than raised
        so one run reports every disagreement instead of the first.
        """
        problems: list[str] = []
        if not self.maturity_path.is_file():
            return [], [f"capability maturity map not found: {self.maturity_path}"]
        try:
            payload = json.loads(self.maturity_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [], [f"could not parse {self.maturity_path}: {exc}"]

        grades = {row.claim_id: row.grade for row in rows}
        out: list[dict[str, Any]] = []

        for entry in payload.get("capabilities", []):
            name = str(entry.get("name", "")) or "<unnamed>"
            stage = str(entry.get("stage", ""))
            claim_ids = [str(item) for item in entry.get("claims", [])]

            if stage not in _STAGE_INDEX:
                problems.append(
                    f"{name}: unknown stage {stage!r} (expected one of {', '.join(CAPABILITY_MATURITY_STAGES)})"
                )
                continue

            missing = [claim for claim in claim_ids if claim not in grades]
            if missing:
                problems.append(f"{name}: cites claim(s) absent from the ledger: {', '.join(missing)}")
                continue
            if not claim_ids:
                problems.append(f"{name}: declares a stage with no supporting claims")
                continue

            # An unrecognised grade cannot be ranked, so it cannot be used to justify a stage.
            # The ledger validator reports the bad grade itself; here we fail closed rather than
            # ranking an unknown value as if it were the strongest one.
            ungradable = [claim for claim in claim_ids if grades[claim] not in _GRADE_RANK]
            if ungradable:
                problems.append(
                    f"{name}: cites claim(s) whose grade is not recognised and therefore cannot "
                    f"support any stage: {', '.join(ungradable)}"
                )
                continue

            ceiling_stage = min(
                (_GRADE_CEILING.get(grades[claim], CAPABILITY_MATURITY_STAGES[0]) for claim in claim_ids),
                key=lambda candidate: _STAGE_INDEX.get(candidate, 0),
            )
            if _STAGE_INDEX[stage] > _STAGE_INDEX[ceiling_stage]:
                weakest = min(claim_ids, key=lambda claim: _GRADE_RANK[grades[claim]])
                problems.append(
                    f"{name}: declares stage {stage!r} but its weakest supporting claim {weakest} is "
                    f"graded {grades[weakest]}, which caps the stage at {ceiling_stage!r}"
                )
                continue

            out.append(
                {
                    "name": name,
                    "stage": stage,
                    "stage_index": _STAGE_INDEX[stage],
                    "ceiling": ceiling_stage,
                    "claims": claim_ids,
                    "grades": {claim: grades[claim] for claim in claim_ids},
                    "notes": str(entry.get("notes", "")),
                }
            )

        out.sort(key=lambda item: (item["stage_index"], item["name"]))
        return out, problems

    # -- assembly --------------------------------------------------------------------------------

    def build(self, results: Sequence[ResultEntry] | None = None) -> dict[str, Any]:
        """Assemble the artifact payload. Key order is fixed, so serialisation is byte-stable."""
        ledger: LedgerReport = ClaimLedgerValidator(self.repo, self.ledger_path).validate()
        maturity, maturity_problems = self.capability_maturity(ledger.rows)
        entries = list(results or ())

        generated = (self.now or datetime.now(UTC)).astimezone(UTC)

        payload: dict[str, Any] = {
            "schema_version": STATUS_ARTIFACT_SCHEMA_VERSION,
            "generated_at": generated.isoformat(),
            "environment": self.environment(),
            "coverage": {
                "gate": self.coverage_gate(),
                "measured": self.measured_coverage(),
                "source": str(self.coverage_json) if self.coverage_json else _UNKNOWN,
            },
            "claims": {
                "ledger": ledger.as_dict()["ledger"],
                "valid": ledger.ok,
                "total": len(ledger.rows),
                "by_grade": {grade: ledger.grade_counts.get(grade, 0) for grade in CLAIM_GRADES},
                "failures": [failure.as_dict() for failure in ledger.failures],
            },
            "capability_maturity": maturity,
            "capability_maturity_problems": maturity_problems,
            "results": [entry.as_dict() for entry in entries],
        }
        payload["ok"] = ledger.ok and not maturity_problems
        logger.info(
            "status artifact assembled: ok=%s claims=%d maturity_rows=%d results=%d",
            payload["ok"],
            len(ledger.rows),
            len(maturity),
            len(entries),
        )
        return payload

    def serialise(self, payload: Mapping[str, Any]) -> str:
        """Render the payload deterministically, with a trailing newline for POSIX-friendly diffs."""
        return json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"

    def write(self, payload: Mapping[str, Any], destination: Path | None = None) -> Path:
        """Write the artifact, creating the (git-ignored) parent directory if needed."""
        target = destination or self.repo / STATUS_ARTIFACT_RELATIVE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.serialise(payload), encoding="utf-8")
        logger.info("wrote %s", target)
        return target


def build(
    repo_root: Path | str | None = None,
    *,
    now: datetime | None = None,
    coverage_json: Path | str | None = None,
    results: Sequence[ResultEntry] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for callers that do not need to hold the builder."""
    return StatusArtifactBuilder(repo_root, now=now, coverage_json=coverage_json).build(results)


def _load_results(path: Path | None) -> list[ResultEntry]:
    """Load result entries from a JSON list, rejecting any entry without a provenance."""
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("results", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        raise ProvenanceError(f"{path}: expected a JSON list of results or an object with a 'results' key")
    return [ResultEntry.from_mapping(item) for item in items]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="status-artifact",
        description="Generate a provenance-stamped status artifact describing what this tree can prove.",
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="repo root (default: this repo)")
    parser.add_argument(
        "--out", type=Path, default=None, help=f"output path (default: {STATUS_ARTIFACT_RELATIVE_PATH})"
    )
    parser.add_argument(
        "--coverage-json", type=Path, default=None, help="a `coverage json` report to read the measured total from"
    )
    parser.add_argument(
        "--results", type=Path, default=None, help="JSON file of result entries; each requires a provenance"
    )
    parser.add_argument(
        "--now", default=None, help="ISO-8601 timestamp to stamp instead of the wall clock (for reproducible output)"
    )
    parser.add_argument("--stdout", action="store_true", help="write to stdout instead of a file")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO-level trace to stderr")
    parser.add_argument("--debug", action="store_true", help="DEBUG-level trace")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the ledger is invalid or the maturity map disagrees with it",
    )
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    try:
        stamp = datetime.fromisoformat(args.now) if args.now else None
    except ValueError as exc:
        print(f"status-artifact: --now is not an ISO-8601 timestamp: {exc}", file=sys.stderr)
        return 2

    builder = StatusArtifactBuilder(args.repo_root, now=stamp, coverage_json=args.coverage_json)
    try:
        payload = builder.build(_load_results(args.results))
    except (ProvenanceError, OSError, ValueError) as exc:
        print(f"status-artifact: {exc}", file=sys.stderr)
        return 2

    if args.stdout:
        sys.stdout.write(builder.serialise(payload))
    else:
        target = builder.write(payload, args.out)
        print(f"Wrote {target} (schema {STATUS_ARTIFACT_SCHEMA_VERSION}).")

    for problem in payload["capability_maturity_problems"]:
        print(f"  - capability-maturity: {problem}", file=sys.stderr)

    if args.strict and not payload["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
