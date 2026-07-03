"""Unit tests for the CI spec-traceability engine (``spec-trace``)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.framework.harness.intent.spec_trace import TraceResult, VerifiedFlip, evaluate_trace, run_trace

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure rule evaluation (no git)
# ---------------------------------------------------------------------------


def _trace(**overrides) -> TraceResult:
    kwargs = {
        "branch": "spec/my_spec",
        "changed_files": ["src/x.py"],
        "base_status": "approved",
        "head_status": "approved",
        "trailers": [],
        "verified_flips": [],
    }
    kwargs.update(overrides)
    return evaluate_trace(**kwargs)


def test_no_src_changes_passes() -> None:
    result = _trace(branch="feature/docs", changed_files=["docs/x.md"], base_status=None, head_status=None)
    assert result.ok


def test_trailer_exempts_src_changes() -> None:
    result = _trace(branch="feature/hotfix", base_status=None, head_status=None, trailers=["urgent hotfix"])
    assert result.ok
    assert any("urgent hotfix" in m for m in result.messages)


def test_non_spec_branch_without_trailer_fails_with_remediation() -> None:
    result = _trace(branch="feature/foo", base_status=None, head_status=None)
    assert not result.ok
    assert any("No-Spec" in m and "spec/<id>" in m for m in result.messages)


def test_approved_base_passes_for_in_flight_and_completing_prs() -> None:
    assert _trace(head_status="approved").ok
    assert _trace(head_status="implemented").ok  # the completing PR's own flip


def test_implemented_base_fails_with_followup_hint() -> None:
    result = _trace(base_status="implemented")
    assert not result.ok
    assert any("already completed" in m for m in result.messages)


def test_absent_base_spec_fails() -> None:
    result = _trace(base_status=None)
    assert not result.ok
    assert any("absent" in m for m in result.messages)


def test_in_pr_flip_to_verified_on_implementing_branch_fails() -> None:
    result = _trace(head_status="verified")
    assert not result.ok


def test_verified_flip_rule_fires_despite_trailer() -> None:
    """Rule (d) is an unconditional conjunct: an exemption trailer must not bypass it."""
    result = _trace(
        branch="feature/flip",
        changed_files=["specs/other_spec.SPEC.md", "tests/test_other.py"],
        base_status=None,
        head_status=None,
        trailers=["some reason"],
        verified_flips=[VerifiedFlip(spec_id="other_spec", unmapped_criteria=("AC-1",))],
    )
    assert not result.ok
    assert any("no test mapping" in m for m in result.messages)


def test_verified_flip_with_full_mapping_passes() -> None:
    result = _trace(
        branch="feature/flip",
        changed_files=["specs/other_spec.SPEC.md", "tests/test_other.py"],
        base_status=None,
        head_status=None,
        verified_flips=[VerifiedFlip(spec_id="other_spec", unmapped_criteria=())],
    )
    assert result.ok


def test_verified_flip_softening_flag_downgrades_to_warning() -> None:
    result = _trace(
        branch="feature/flip",
        changed_files=["specs/other_spec.SPEC.md"],
        base_status=None,
        head_status=None,
        verified_flips=[VerifiedFlip(spec_id="other_spec", unmapped_criteria=("AC-1",))],
        allow_unmapped_verified=True,
    )
    assert result.ok
    assert any("warning (softened)" in m for m in result.messages)


# ---------------------------------------------------------------------------
# End-to-end against hermetic tmp git repos
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _spec_text(spec_id: str, status: str) -> str:
    return (
        f"---\nid: {spec_id}\ngoal: g\nmodule: src/\nstatus: {status}\n---\n\n"
        "# Goal\ng\n\n# Acceptance Criteria\n- AC-1: works\n"
    )


def _make_repo(tmp_path: Path, spec_status: str = "approved") -> Path:
    repo = tmp_path / "repo"
    (repo / "specs").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "specs" / "demo_trace_spec.SPEC.md").write_text(_spec_text("demo_trace_spec", spec_status))
    (repo / "src" / "x.py").write_text("X = 1\n")
    (repo / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_run_trace_passes_on_approved_spec_branch(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _git(repo, "switch", "-q", "-c", "spec/demo_trace_spec")
    (repo / "src" / "x.py").write_text("X = 2\n")
    _git(repo, "commit", "-qam", "implement")
    result = run_trace(repo, "main", "HEAD", "spec/demo_trace_spec")
    assert result.ok, result.messages


def test_run_trace_fails_on_feature_branch_then_passes_with_trailer(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _git(repo, "switch", "-q", "-c", "feature/foo")
    (repo / "src" / "x.py").write_text("X = 3\n")
    _git(repo, "commit", "-qam", "change src")
    assert not run_trace(repo, "main", "HEAD", "feature/foo").ok
    _git(repo, "commit", "-q", "--allow-empty", "-m", "hotfix\n\nNo-Spec: emergency fix")
    assert run_trace(repo, "main", "HEAD", "feature/foo").ok


def test_run_trace_catches_renames_out_of_src(tmp_path: Path) -> None:
    """--no-renames makes the old src/ path visible; a rename can't dodge the rule."""
    repo = _make_repo(tmp_path)
    _git(repo, "switch", "-q", "-c", "feature/mv")
    _git(repo, "mv", "src/x.py", "tests/x.py")
    _git(repo, "commit", "-qm", "move out of src")
    assert not run_trace(repo, "main", "HEAD", "feature/mv").ok


def test_run_trace_verified_flip_requires_same_line_mapping(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, spec_status="implemented")
    _git(repo, "switch", "-q", "-c", "feature/verify")
    (repo / "specs" / "demo_trace_spec.SPEC.md").write_text(_spec_text("demo_trace_spec", "verified"))
    _git(repo, "commit", "-qam", "flip to verified\n\nNo-Spec: verification flip")
    result = run_trace(repo, "main", "HEAD", "feature/verify")
    assert not result.ok  # trailer must not bypass rule (d)

    (repo / "tests" / "test_x.py").write_text(
        '"""Covers demo_trace_spec AC-1."""\n\n\ndef test_x():\n    assert True\n'
    )
    _git(repo, "commit", "-qam", "map AC-1")
    assert run_trace(repo, "main", "HEAD", "feature/verify").ok


def test_run_trace_word_bounds_ac_tokens(tmp_path: Path) -> None:
    """A mapping line for AC-10 must not satisfy AC-1."""
    repo = _make_repo(tmp_path, spec_status="implemented")
    _git(repo, "switch", "-q", "-c", "feature/verify")
    (repo / "specs" / "demo_trace_spec.SPEC.md").write_text(_spec_text("demo_trace_spec", "verified"))
    (repo / "tests" / "test_x.py").write_text(
        '"""Covers demo_trace_spec AC-10 only."""\n\n\ndef test_x():\n    assert True\n'
    )
    _git(repo, "commit", "-qam", "flip\n\nNo-Spec: flip")
    assert not run_trace(repo, "main", "HEAD", "feature/verify").ok
