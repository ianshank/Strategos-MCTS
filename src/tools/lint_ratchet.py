"""Enforce a shrinking baseline for ruff rules that cannot yet be turned on as a gate.

**Why this exists.** ``ruff``'s NumPy ruleset (``NPY``) was never selected in
``pyproject.toml``, so none of it was enforced. Turning it on reports **NPY002**
(``numpy-legacy-random``) at 108 call sites — every ``np.random.seed``, ``np.random.choice``,
``np.random.dirichlet`` that uses NumPy's process-global legacy RNG instead of an explicit
``np.random.Generator``.

That is not a style preference here. One of those 108 is
``src/framework/mcts/neural_mcts.py:322``, the root Dirichlet noise that makes ``NeuralMCTS``
irreproducible under a torch-only seed — the defect recorded in
``docs/plans/EVIDENCE_FIRST_PROGRAM.md`` §2.5 and specified for repair in
``specs/hygiene_determinism.SPEC.md`` AC-3. The rule that would have caught it was available
the whole time and switched off. Three more sit on ``np.random.seed`` calls in the training
drivers, which is the same defect wearing a different hat: seeding the legacy global RNG does
not seed a ``Generator`` anyone later constructs.

**Why a ratchet rather than a gate.** Converting 108 call sites touches ``src/training/`` and
``src/framework/mcts/``, both claimed by open approved specs, so a blanket refactor here would
violate CHARTER.md NG-4. And a gate that must be introduced switched off is decoration. So
``NPY`` is selected in ``pyproject.toml`` with ``NPY002`` ignored — every *other* NumPy rule is
enforced from now on at zero refactor cost — and this module holds ``NPY002`` to a committed
baseline that may only shrink. That converts an all-or-nothing cleanup into a property that
holds continuously, and it makes the determinism debt a number that visibly goes down when
``hygiene_determinism`` AC-3 lands.

This is deliberately the *same* mechanism as ``src/tools/action_pins.py`` — a declarative
registry, a committed baseline grouped by area, counts that may only decrease, and a
``--write-baseline`` re-tightening step — rather than a second ratchet system with its own
conventions (CHARTER.md NG-7 in spirit). Adding a rule is data: append to
:data:`RATCHETED_RULES`.

Design notes:

- **Injected runner.** :class:`LintRatchet` takes the callable that shells out to ruff, so the
  tests drive it with recorded output and never depend on the host's ruff version or on the
  tree's current count.
- **No hardcoded values.** Paths, schema version and grouping depth come from
  ``src/config/constants.py``; the rules and their rationales are a registry, not literals
  scattered through control flow.
- **Structured failures.** Returns :class:`Violation` objects, so the CLI, the unit tests and
  any future reporter format them differently from one source.

CLI::

    python -m src.tools.lint_ratchet                  # report; exit 1 if the ratchet is violated
    python -m src.tools.lint_ratchet --json           # machine-readable report
    python -m src.tools.lint_ratchet --write-baseline # re-tighten after fixing call sites
    python -m src.tools.lint_ratchet --debug          # DEBUG trace of every ruff invocation
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

from src.config.constants import (
    LINT_RATCHET_BASELINE_RELATIVE_PATH,
    LINT_RATCHET_BASELINE_SCHEMA_VERSION,
    LINT_RATCHET_GROUP_DEPTH,
    RUFF_JSON_OUTPUT_FORMAT,
)

__all__ = [
    "RatchetedRule",
    "RATCHETED_RULES",
    "Violation",
    "RuleCount",
    "LintRatchet",
    "group_key",
    "main",
]

logger = logging.getLogger(__name__)

# This module lives at <repo>/src/tools/lint_ratchet.py.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RatchetedRule:
    """One ruff rule held to a shrinking baseline, with the reason it is not simply fixed."""

    code: str
    #: Paths passed to ruff. Relative to the repo root; ruff's own excludes still apply.
    paths: tuple[str, ...]
    #: Why this rule is ratcheted rather than gated. Printed by the CLI so the next reader
    #: does not have to reconstruct the argument from a commit message.
    rationale: str


#: The registry. Adding a rule is data, not a control-flow edit.
RATCHETED_RULES: tuple[RatchetedRule, ...] = (
    RatchetedRule(
        code="NPY002",
        paths=(".",),
        rationale=(
            "Legacy global-RNG calls. One of them (src/framework/mcts/neural_mcts.py, root "
            "Dirichlet noise) is the known determinism defect specified in "
            "specs/hygiene_determinism.SPEC.md AC-3. src/training/ and src/framework/mcts/ are "
            "claimed by open approved specs, so a blanket conversion here would violate NG-4."
        ),
    ),
)


@dataclass(frozen=True)
class RuleCount:
    """Current findings for one rule, in total and grouped by area."""

    code: str
    total: int
    by_group: dict[str, int]


@dataclass(frozen=True)
class Violation:
    """A ratchet breach, structured so callers choose their own wording."""

    code: str
    group: str
    baseline: int
    current: int

    @property
    def is_new_area(self) -> bool:
        """True when the group is absent from the baseline entirely."""
        return self.baseline == 0

    def describe(self) -> str:
        if self.is_new_area:
            return (
                f"{self.code}: {self.group!r} is not in the baseline but has {self.current} "
                f"finding(s). A new area must not introduce this pattern."
            )
        return f"{self.code}: {self.group!r} rose from {self.baseline} to {self.current}."


def group_key(relative_path: str, depth: int = LINT_RATCHET_GROUP_DEPTH) -> str:
    """The baseline bucket for a repo-relative path.

    The first ``depth`` path components, or the whole path when it is shallower — so a
    top-level script groups under its own name rather than colliding with the repo root.
    """
    parts = Path(relative_path).parts
    if len(parts) <= 1:
        return relative_path
    return "/".join(parts[:depth])


def _default_runner(argv: Sequence[str], *, cwd: Path) -> str:
    """Shell out to ruff via the running interpreter, so the pinned dev version is used."""
    logger.debug("ruff: %s", " ".join(argv))
    completed = subprocess.run(  # noqa: S603 - argv is built here, never from user input
        [sys.executable, "-m", "ruff", *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    # ruff exits 1 when it finds violations, which is the normal case here. Anything else
    # (2 = usage/internal error) means the count would be wrong, and a wrong count that
    # silently reads as "zero findings" would turn the ratchet into decoration.
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed ({completed.returncode}): {completed.stderr.strip()[:2000]}")
    return completed.stdout


class LintRatchet:
    """Counts ratcheted rule findings and compares them against the committed baseline."""

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        rules: Sequence[RatchetedRule] = RATCHETED_RULES,
        runner: Callable[..., str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT
        self.rules = tuple(rules)
        self._runner = runner or _default_runner

    @property
    def baseline_path(self) -> Path:
        return self.repo_root / LINT_RATCHET_BASELINE_RELATIVE_PATH

    def count(self, rule: RatchetedRule) -> RuleCount:
        """Current findings for one rule, grouped by area."""
        raw = self._runner(
            ["check", *rule.paths, "--select", rule.code, "--output-format", RUFF_JSON_OUTPUT_FORMAT],
            cwd=self.repo_root,
        )
        findings = json.loads(raw) if raw.strip() else []
        by_group: dict[str, int] = {}
        for finding in findings:
            relative = self._relative(finding.get("filename", ""))
            key = group_key(relative)
            by_group[key] = by_group.get(key, 0) + 1
        logger.debug("%s: %d finding(s) across %d group(s)", rule.code, len(findings), len(by_group))
        return RuleCount(code=rule.code, total=len(findings), by_group=dict(sorted(by_group.items())))

    def _relative(self, filename: str) -> str:
        """Ruff reports absolute paths; the baseline must be machine-independent."""
        try:
            return Path(filename).resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return Path(filename).as_posix()

    def load_baseline(self) -> dict[str, Any]:
        if not self.baseline_path.exists():
            return {}
        loaded = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        # A baseline that is not a mapping (an empty file, a list, a bare number) would make
        # every `.get` below raise deep inside `check`. Treat it as absent, which fails closed:
        # with no recorded allowance, every current finding is a violation.
        if not isinstance(loaded, dict):
            logger.warning("Ignoring malformed baseline at %s (expected an object)", self.baseline_path)
            return {}
        return loaded

    def counts(self) -> dict[str, RuleCount]:
        return {rule.code: self.count(rule) for rule in self.rules}

    def check(self, counts: dict[str, RuleCount] | None = None) -> list[Violation]:
        """Every way the current tree exceeds the baseline. Empty means the ratchet holds."""
        measured = self.counts() if counts is None else counts
        recorded = self.load_baseline().get("rules", {})
        violations: list[Violation] = []
        for code, count in measured.items():
            baseline_groups = (recorded.get(code) or {}).get("by_group", {})
            for group in sorted(set(count.by_group) | set(baseline_groups)):
                current = count.by_group.get(group, 0)
                allowed = int(baseline_groups.get(group, 0))
                if current > allowed:
                    violations.append(Violation(code=code, group=group, baseline=allowed, current=current))
        return violations

    def slack(self, counts: dict[str, RuleCount] | None = None) -> dict[str, int]:
        """How far each rule is *below* its baseline — what re-tightening would reclaim."""
        measured = self.counts() if counts is None else counts
        recorded = self.load_baseline().get("rules", {})
        return {
            code: int((recorded.get(code) or {}).get("total", count.total)) - count.total
            for code, count in measured.items()
        }

    def render_baseline(self, counts: dict[str, RuleCount] | None = None) -> dict[str, Any]:
        measured = self.counts() if counts is None else counts
        return {
            "_README": (
                "Ratchet baseline for ruff rules that cannot yet be enforced as a gate. Counts are "
                "grouped by the first two path components and may only DECREASE; a group absent "
                "from this file must have zero findings. Re-tighten in the same change that fixes "
                "call sites: make lint-ratchet-baseline. Rationale per rule is in "
                "src/tools/lint_ratchet.py RATCHETED_RULES."
            ),
            "schema_version": LINT_RATCHET_BASELINE_SCHEMA_VERSION,
            "rules": {
                code: {"total": count.total, "by_group": count.by_group} for code, count in sorted(measured.items())
            },
        }

    def write_baseline(self, counts: dict[str, RuleCount] | None = None) -> Path:
        self.baseline_path.write_text(json.dumps(self.render_baseline(counts), indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", self.baseline_path)
        return self.baseline_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 when the ratchet holds, 1 when it does not."""
    parser = argparse.ArgumentParser(description="Enforce shrinking baselines for ratcheted ruff rules.")
    parser.add_argument("--repo-root", default=None, help="Repository root to audit.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Rewrite the baseline to describe the current tree (use after fixing call sites).",
    )
    parser.add_argument("--debug", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    ratchet = LintRatchet(Path(args.repo_root) if args.repo_root else None)
    counts = ratchet.counts()

    if args.write_baseline:
        path = ratchet.write_baseline(counts)
        print(f"Wrote {path}")
        return 0

    violations = ratchet.check(counts)
    slack = ratchet.slack(counts)

    if args.json:
        print(
            json.dumps(
                {
                    "rules": {c: {"total": k.total, "by_group": k.by_group} for c, k in counts.items()},
                    "violations": [
                        {"code": v.code, "group": v.group, "baseline": v.baseline, "current": v.current}
                        for v in violations
                    ],
                    "slack": slack,
                    "ok": not violations,
                },
                indent=2,
            )
        )
        return 1 if violations else 0

    for rule in ratchet.rules:
        count = counts[rule.code]
        logger.info("lint ratchet: %s at %d finding(s) — %s", rule.code, count.total, rule.rationale)

    if violations:
        print("Lint ratchet VIOLATED:")
        for violation in violations:
            print(f"  - {violation.describe()}")
        print("\nFix the new call sites, or justify and re-tighten with: make lint-ratchet-baseline")
        return 1

    reclaimable = {code: value for code, value in slack.items() if value > 0}
    if reclaimable:
        # Not a failure on its own; the unit test is what forces the file to be re-tightened,
        # so a decrease is reported here rather than silently tolerated.
        for code, value in reclaimable.items():
            print(f"{code}: {value} finding(s) below the baseline — re-tighten with: make lint-ratchet-baseline")

    totals = ", ".join(f"{code}={count.total}" for code, count in sorted(counts.items()))
    print(f"OK — lint ratchet holds ({totals})")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
