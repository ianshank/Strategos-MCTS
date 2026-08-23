"""Audit GitHub Actions references for commit-SHA pinning, and maintain a ratchet baseline.

Spec: ``specs/evidence_claim_ledger.SPEC.md`` AC-8.

**Why this exists.** ``uses: actions/checkout@v7`` is a *mutable* reference. Whoever controls that
repository can move the tag, and the next CI run executes whatever it now points at — with the
workflow's tokens and secrets in scope. Pinning to a 40-character commit SHA makes the reference
immutable, so upgrading becomes a reviewed change rather than something that happens to you.

**Why a ratchet rather than a gate.** The tree currently has no SHA-pinned action at all. A gate
demanding full pinning would have to be introduced disabled, and a disabled gate is decoration. So
this module reports counts, and a committed baseline records where the tree stands. The counts may
only go down. New actions must be pinned on first use. That converts an all-or-nothing cleanup into
a property that holds continuously while the number shrinks.

The scanner is textual on purpose. ``uses:`` inside a job's ``with:`` block, in a comment, or in a
composite action all matter equally to the runner, and a YAML-shaped walk would need to know about
every one of those shapes to stay correct.

CLI::

    python -m src.tools.action_pins                  # report; exit 1 if the ratchet is violated
    python -m src.tools.action_pins --json           # machine-readable report
    python -m src.tools.action_pins --write-baseline # re-tighten the baseline after pinning
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
from typing import Any

from src.config.constants import (
    ACTION_PIN_BASELINE_RELATIVE_PATH,
    ACTION_PIN_BASELINE_SCHEMA_VERSION,
    ACTION_PIN_SHA_LENGTH,
    ACTION_USES_LOCAL_PREFIXES,
    WORKFLOW_DIR_RELATIVE_PATH,
    WORKFLOW_FILE_GLOBS,
)

logger = logging.getLogger(__name__)

# `uses: owner/repo@ref`, `uses: owner/repo/sub@ref`, `uses: ./local`, `uses: docker://image`.
_USES_RE = re.compile(r"uses:\s*['\"]?(?P<ref>[^'\"\s#]+)")

_SHA_RE = re.compile(rf"^[0-9a-f]{{{ACTION_PIN_SHA_LENGTH}}}$")


@dataclass(frozen=True)
class ActionUse:
    """One ``uses:`` reference, with enough location detail to fix it without searching."""

    workflow: str
    line: int
    action: str
    ref: str

    @property
    def pinned(self) -> bool:
        """True when the reference is an immutable 40-character commit SHA."""
        return bool(_SHA_RE.match(self.ref))

    def __str__(self) -> str:
        return f"{WORKFLOW_DIR_RELATIVE_PATH}/{self.workflow}:{self.line}: {self.action}@{self.ref}"


@dataclass
class PinReport:
    """The audit result: every use, the unpinned tally, and any ratchet violations."""

    uses: tuple[ActionUse, ...] = ()
    baseline: dict[str, int] = field(default_factory=dict)
    violations: tuple[str, ...] = ()

    @property
    def unpinned(self) -> tuple[ActionUse, ...]:
        return tuple(use for use in self.uses if not use.pinned)

    @property
    def counts(self) -> dict[str, int]:
        """Unpinned uses per action name, sorted for stable output."""
        tally: dict[str, int] = {}
        for use in self.unpinned:
            tally[use.action] = tally.get(use.action, 0) + 1
        return dict(sorted(tally.items()))

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total_uses": len(self.uses),
            "total_unpinned_uses": len(self.unpinned),
            "unpinned_uses_by_action": self.counts,
            "baseline": dict(sorted(self.baseline.items())),
            "violations": list(self.violations),
        }

    def baseline_document(self, readme: str) -> dict[str, Any]:
        """The baseline file contents describing the *current* tree."""
        return {
            "_README": readme,
            "schema_version": ACTION_PIN_BASELINE_SCHEMA_VERSION,
            "total_unpinned_uses": len(self.unpinned),
            "unpinned_uses_by_action": self.counts,
        }


class ActionPinAuditor:
    """Scans workflows for ``uses:`` references and compares them against the ratchet baseline.

    Both roots are injected rather than discovered so a caller — and the test suite — can audit a
    synthetic tree without touching the repository.
    """

    def __init__(
        self,
        repo_root: Path,
        workflow_dir: Path | None = None,
        baseline_path: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.workflow_dir = Path(workflow_dir) if workflow_dir else self.repo_root / WORKFLOW_DIR_RELATIVE_PATH
        self.baseline_path = (
            Path(baseline_path) if baseline_path else self.repo_root / ACTION_PIN_BASELINE_RELATIVE_PATH
        )

    # ---------------------------------------------------------------- scanning

    def workflow_files(self) -> list[Path]:
        """Every workflow file, discovered so a new workflow is covered without an edit here."""
        found: list[Path] = []
        for pattern in WORKFLOW_FILE_GLOBS:
            found.extend(sorted(self.workflow_dir.glob(pattern)))
        return sorted(set(found))

    def scan(self) -> tuple[ActionUse, ...]:
        """Collect every third-party ``uses:`` reference across all workflows."""
        uses: list[ActionUse] = []

        for path in self.workflow_files():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                match = _USES_RE.search(line)
                if not match:
                    continue

                ref = match.group("ref")

                # Local composite actions and container images are not tag-mutable third-party
                # code fetched from a registry we do not control, so the pin rule does not apply.
                if any(ref.startswith(prefix) for prefix in ACTION_USES_LOCAL_PREFIXES):
                    logger.debug("%s:%d: skipping local/container reference %s", path.name, number, ref)
                    continue

                action, separator, version = ref.partition("@")
                if not separator:
                    logger.debug("%s:%d: skipping reference without a version: %s", path.name, number, ref)
                    continue

                uses.append(ActionUse(path.name, number, action, version))

        logger.debug("scanned %d workflow file(s), found %d action reference(s)", len(self.workflow_files()), len(uses))
        return tuple(uses)

    # ---------------------------------------------------------------- baseline

    def load_baseline(self) -> tuple[dict[str, int], tuple[str, ...]]:
        """Read the committed baseline, reporting rather than raising when it is unusable."""
        if not self.baseline_path.is_file():
            return {}, (
                f"pin baseline not found at {self.baseline_path}. Create it with: "
                "python -m src.tools.action_pins --write-baseline",
            )

        try:
            payload = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, (f"could not parse {self.baseline_path}: {exc}",)

        version = payload.get("schema_version")
        if version != ACTION_PIN_BASELINE_SCHEMA_VERSION:
            return {}, (
                f"{self.baseline_path} declares schema_version {version!r}, "
                f"expected {ACTION_PIN_BASELINE_SCHEMA_VERSION}",
            )

        raw = payload.get("unpinned_uses_by_action") or {}
        if not isinstance(raw, dict):
            return {}, (f"{self.baseline_path}: unpinned_uses_by_action must be an object",)

        baseline: dict[str, int] = {}
        problems: list[str] = []
        for action, count in raw.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                problems.append(f"{self.baseline_path}: {action} has a non-integer count {count!r}")
                continue
            baseline[str(action)] = count

        return baseline, tuple(problems)

    # ---------------------------------------------------------------- ratchet

    def check(self, uses: Iterable[ActionUse], baseline: dict[str, int]) -> tuple[str, ...]:
        """Compare the tree against the baseline in both directions.

        Three rules, each chosen so the ratchet cannot be satisfied by editing the baseline upward:

        1. An action's unpinned count may not exceed its baseline.
        2. An action with no baseline entry must be SHA-pinned on first use.
        3. If a count has *improved*, the baseline must be re-tightened in the same change —
           otherwise the recovered slack silently becomes budget for the next unpinned step.
        """
        report = PinReport(tuple(uses))
        counts = report.counts
        violations: list[str] = []

        for action, count in counts.items():
            allowed = baseline.get(action)

            if allowed is None:
                locations = [str(use) for use in report.unpinned if use.action == action]
                violations.append(
                    f"{action} is used without a commit-SHA pin and has no baseline entry. New actions "
                    f"must be pinned to a {ACTION_PIN_SHA_LENGTH}-character commit SHA on first use "
                    f"(seen at: {', '.join(locations)})"
                )
                continue

            if count > allowed:
                violations.append(
                    f"{action}: {count} unpinned use(s), baseline allows {allowed}. The pin ratchet only "
                    "moves down; pin the new reference to a commit SHA instead of raising the baseline."
                )

        for action, allowed in sorted(baseline.items()):
            current = counts.get(action, 0)
            if current < allowed:
                violations.append(
                    f"{action}: baseline is stale — it allows {allowed} unpinned use(s) but the tree now "
                    f"has {current}. Re-tighten it in this change: python -m src.tools.action_pins "
                    "--write-baseline"
                )

        return tuple(violations)

    def audit(self) -> PinReport:
        """Full audit: scan, load the baseline, and apply the ratchet."""
        uses = self.scan()
        baseline, baseline_problems = self.load_baseline()
        violations = baseline_problems + self.check(uses, baseline)
        report = PinReport(uses, baseline, violations)
        logger.info(
            "action pin audit: %d use(s), %d unpinned, %s",
            len(report.uses),
            len(report.unpinned),
            "OK" if report.ok else f"{len(report.violations)} violation(s)",
        )
        return report


def audit(repo_root: Path | str | None = None) -> PinReport:
    """Audit ``repo_root`` (default: the repository containing this file)."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return ActionPinAuditor(root).audit()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 when the ratchet holds, 1 when it does not."""
    parser = argparse.ArgumentParser(description="Audit GitHub Actions commit-SHA pinning.")
    parser.add_argument("--repo-root", default=None, help="Repository root to audit.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Rewrite the baseline to describe the current tree (use after pinning an action).",
    )
    parser.add_argument("--debug", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    auditor = ActionPinAuditor(root)
    report = auditor.audit()

    if args.write_baseline:
        existing_readme = ""
        if auditor.baseline_path.is_file():
            try:
                existing_readme = str(json.loads(auditor.baseline_path.read_text(encoding="utf-8")).get("_README", ""))
            except (OSError, json.JSONDecodeError):
                existing_readme = ""

        document = report.baseline_document(existing_readme)
        auditor.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        auditor.baseline_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {auditor.baseline_path}: {len(report.unpinned)} unpinned use(s)")
        return 0

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        for violation in report.violations:
            print(f"  - {violation}", file=sys.stderr)
        if report.ok:
            print(
                f"OK — {len(report.uses)} action reference(s), {len(report.unpinned)} unpinned "
                f"(within the committed ratchet)"
            )
        else:
            print(f"FAILED — {len(report.violations)} pin ratchet violation(s)", file=sys.stderr)

    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
