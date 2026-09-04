"""Invariants for the ruff rule ratchet.

A ratchet is only worth having if it fails on the thing it exists to prevent and stays quiet
otherwise, so both halves are asserted. The failure mode that would make it decoration is a
counting bug that silently reports zero — the tree would look clean, the baseline would be
re-tightened to zero on the next `--write-baseline`, and the debt would vanish from the record
without a single call site being fixed. Several tests below exist only to make that impossible.

Almost every test drives :class:`LintRatchet` through an **injected runner** returning recorded
ruff JSON. That keeps them independent of the host's ruff version and of the tree's current
count, which is what lets them assert exact numbers. Two tests deliberately do use the real
ruff and the real tree — the ones whose whole point is that the committed baseline describes
*this* repository — and they skip cleanly where ruff is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.config.constants import (
    LINT_RATCHET_BASELINE_RELATIVE_PATH,
    LINT_RATCHET_BASELINE_SCHEMA_VERSION,
    LINT_RATCHET_GROUP_DEPTH,
)
from src.tools.lint_ratchet import (
    RATCHETED_RULES,
    LintRatchet,
    RatchetedRule,
    RuleCount,
    Violation,
    group_key,
    main,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]

RULE = RatchetedRule(code="NPY002", paths=(".",), rationale="test double")


def _runner_for(findings: list[str]):
    """A runner returning ruff-shaped JSON for the given repo-relative filenames."""

    def _run(argv, *, cwd):  # noqa: ARG001 - signature must match the real runner
        return json.dumps([{"filename": str(Path(cwd) / name), "code": "NPY002"} for name in findings])

    return _run


def _ratchet(tmp_path: Path, findings: list[str], baseline: dict | None = None) -> LintRatchet:
    if baseline is not None:
        (tmp_path / LINT_RATCHET_BASELINE_RELATIVE_PATH).write_text(json.dumps(baseline), encoding="utf-8")
    return LintRatchet(tmp_path, rules=(RULE,), runner=_runner_for(findings))


def _baseline(by_group: dict[str, int]) -> dict:
    return {
        "schema_version": LINT_RATCHET_BASELINE_SCHEMA_VERSION,
        "rules": {"NPY002": {"total": sum(by_group.values()), "by_group": by_group}},
    }


# ------------------------------------------------------------------------ grouping


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/training/self_play_trainer.py", "src/training"),
        ("tests/unit/tooling/test_x.py", "tests/unit"),
        # A top-level file has no second component; it must group under itself rather than
        # collapsing every root-level script into one bucket named after the repo root.
        ("app.py", "app.py"),
        ("training/meta_controller.py", "training/meta_controller.py"),
    ],
)
def test_findings_group_by_area(path: str, expected: str) -> None:
    assert group_key(path) == expected


def test_the_grouping_depth_is_configurable_not_hardcoded() -> None:
    assert group_key("a/b/c/d.py", depth=3) == "a/b/c"
    assert group_key("a/b/c/d.py") == "/".join(("a", "b", "c", "d.py")[:LINT_RATCHET_GROUP_DEPTH])


# -------------------------------------------------------------------------- counting


def test_counting_groups_and_totals(tmp_path: Path) -> None:
    ratchet = _ratchet(tmp_path, ["src/training/a.py", "src/training/b.py", "tests/unit/c.py"])
    count = ratchet.count(RULE)
    assert count.total == 3
    assert count.by_group == {"src/training": 2, "tests/unit": 1}


def test_an_empty_ruff_run_counts_zero_not_an_error(tmp_path: Path) -> None:
    """ruff prints nothing at all when a rule has no findings; that must not raise."""
    ratchet = LintRatchet(tmp_path, rules=(RULE,), runner=lambda argv, *, cwd: "")  # noqa: ARG005
    assert ratchet.count(RULE) == RuleCount(code="NPY002", total=0, by_group={})


def test_paths_in_the_baseline_are_repo_relative(tmp_path: Path) -> None:
    """An absolute path would make the baseline machine-specific and unmergeable."""
    ratchet = _ratchet(tmp_path, ["src/training/a.py"])
    rendered = ratchet.render_baseline()
    assert list(rendered["rules"]["NPY002"]["by_group"]) == ["src/training"]
    assert str(tmp_path) not in json.dumps(rendered)


def test_a_ruff_crash_raises_instead_of_counting_zero() -> None:
    """The failure that would silently erase the debt, so it is made loud.

    A usage or internal error from ruff yields no findings on stdout. Treating that as "zero
    violations" would report a clean ratchet and, on the next `--write-baseline`, rewrite the
    file to zero without a single call site fixed.
    """
    from src.tools.lint_ratchet import _default_runner

    with pytest.raises(RuntimeError, match="ruff failed"):
        _default_runner(["check", "--not-a-real-flag"], cwd=REPO_ROOT)


# ------------------------------------------------------------------------- the ratchet


def test_holding_steady_is_not_a_violation(tmp_path: Path) -> None:
    ratchet = _ratchet(tmp_path, ["src/training/a.py"], _baseline({"src/training": 1}))
    assert ratchet.check() == []


def test_a_decrease_is_not_a_violation(tmp_path: Path) -> None:
    """Fixing call sites must never fail the build."""
    ratchet = _ratchet(tmp_path, [], _baseline({"src/training": 3}))
    assert ratchet.check() == []


def test_an_increase_in_a_known_area_fails(tmp_path: Path) -> None:
    ratchet = _ratchet(tmp_path, ["src/training/a.py", "src/training/b.py"], _baseline({"src/training": 1}))
    violations = ratchet.check()
    assert violations == [Violation(code="NPY002", group="src/training", baseline=1, current=2)]
    assert "rose from 1 to 2" in violations[0].describe()


def test_a_new_area_fails_even_though_the_total_is_unchanged(tmp_path: Path) -> None:
    """The reason grouping exists: a repo-wide total would call this a clean swap.

    One call site fixed in `src/training` and one introduced in `src/api` leaves the total at
    108. Grouped, it is correctly a regression in a package that had none.
    """
    ratchet = _ratchet(tmp_path, ["src/api/new.py"], _baseline({"src/training": 1}))
    violations = ratchet.check()
    assert [v.group for v in violations] == ["src/api"]
    assert violations[0].is_new_area
    assert "not in the baseline" in violations[0].describe()


def test_a_missing_baseline_file_makes_every_finding_a_violation(tmp_path: Path) -> None:
    """Deleting the baseline must not read as "nothing to enforce"."""
    ratchet = _ratchet(tmp_path, ["src/training/a.py"])
    assert [v.group for v in ratchet.check()] == ["src/training"]


def test_slack_reports_what_retightening_would_reclaim(tmp_path: Path) -> None:
    ratchet = _ratchet(tmp_path, ["src/training/a.py"], _baseline({"src/training": 4}))
    assert ratchet.slack() == {"NPY002": 3}


def test_write_baseline_round_trips(tmp_path: Path) -> None:
    ratchet = _ratchet(tmp_path, ["src/training/a.py", "tests/unit/b.py"])
    ratchet.write_baseline()
    assert ratchet.check() == [], "a freshly written baseline must describe the tree it was written from"
    written = json.loads(ratchet.baseline_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == LINT_RATCHET_BASELINE_SCHEMA_VERSION
    assert written["_README"], "the baseline must explain itself to the next reader"


# ------------------------------------------------------------------------------ CLI


def test_the_cli_exits_nonzero_on_a_violation(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / LINT_RATCHET_BASELINE_RELATIVE_PATH).write_text(json.dumps(_baseline({})), encoding="utf-8")
    # Uses the real runner against an empty tmp tree, so a stray ruff finding is impossible.
    (tmp_path / "probe.py").write_text("import numpy as np\nx = np.random.choice([1])\n", encoding="utf-8")
    assert main(["--repo-root", str(tmp_path)]) == 1
    assert "VIOLATED" in capsys.readouterr().out


def test_the_cli_exits_zero_when_the_ratchet_holds(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / LINT_RATCHET_BASELINE_RELATIVE_PATH).write_text(json.dumps(_baseline({})), encoding="utf-8")
    assert main(["--repo-root", str(tmp_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_the_json_report_is_machine_readable(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / LINT_RATCHET_BASELINE_RELATIVE_PATH).write_text(json.dumps(_baseline({})), encoding="utf-8")
    main(["--repo-root", str(tmp_path), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["rules"]["NPY002"]["total"] == 0


# ------------------------------------------------- the committed baseline and the wiring


def _ruff_available() -> bool:
    return subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, check=False).returncode == 0


needs_ruff = pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed in this environment")


@needs_ruff
def test_the_committed_baseline_holds_for_this_repository() -> None:
    """The gate itself, run against the real tree."""
    violations = LintRatchet(REPO_ROOT).check()
    assert not violations, "lint ratchet violated:\n" + "\n".join(v.describe() for v in violations)


@needs_ruff
def test_the_committed_baseline_is_tight() -> None:
    """A decrease must be banked in the same change that earns it.

    Without this, fixing call sites leaves slack that a later regression can spend silently —
    the ratchet would stop at the high-water mark instead of following the tree down.
    """
    slack = LintRatchet(REPO_ROOT).slack()
    stale = {code: value for code, value in slack.items() if value > 0}
    assert not stale, f"the baseline is looser than the tree ({stale}); re-tighten: make lint-ratchet-baseline"


def test_every_ratcheted_rule_explains_itself() -> None:
    """A ratchet with no stated rationale becomes permanent by default."""
    for rule in RATCHETED_RULES:
        assert len(rule.rationale) > 40, f"{rule.code} has no usable rationale"
        assert rule.paths, f"{rule.code} scans nothing"


def test_a_ratcheted_rule_is_ignored_by_ruff_itself() -> None:
    """Otherwise the rule fails `ruff check .` and the ratchet is unreachable.

    The two must agree: `select` turns the family on, `ignore` hands the specific rule to this
    ratchet. If someone removes the ignore, `make lint` goes red at 108 findings and the next
    person deletes the select — losing every other NumPy rule with it.
    """
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
        import tomli as toml_reader  # type: ignore[import-not-found]

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        lint = toml_reader.load(handle)["tool"]["ruff"]["lint"]

    for rule in RATCHETED_RULES:
        family = rule.code.rstrip("0123456789")
        assert family in lint["select"], f"ruff does not select {family}, so {rule.code} is unenforceable"
        assert rule.code in lint["ignore"], f"{rule.code} is ratcheted but not ignored by ruff; `make lint` would fail"


def test_the_ratchet_is_wired_into_ci_and_the_makefile() -> None:
    """An unwired validator is a file, not a gate — the same rule the hook registry applies."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "src.tools.lint_ratchet" in workflow, "the lint ratchet is not run by any CI job"

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "lint-ratchet:" in makefile, "no `make lint-ratchet` target"
    gate_line = next(line for line in makefile.splitlines() if line.startswith("gate:"))
    assert "lint-ratchet" in gate_line, "`make gate` does not run the lint ratchet"
