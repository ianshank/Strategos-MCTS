"""CLI smoke tests — exercise ``dry-run`` and ``validate-spec`` paths.

We avoid invoking ``harness run`` here because that path needs a real LLM
client; the runner integration test already covers the run path with a
scripted LLM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.framework.harness.cli import (
    _apply_settings_overrides,
    _resolve_intent,
    main,
)
from src.framework.harness.loop.runner import RunResult
from src.framework.harness.outcomes import Terminal

pytestmark = [pytest.mark.unit, pytest.mark.harness]


def _spec(tmp_path: Path) -> Path:
    """Legacy/ad-hoc spec: run/dry-run stay permissive on these (no v2 frontmatter)."""
    spec = tmp_path / "spec.md"
    spec.write_text(
        "# Goal\nDo a thing.\n\n# Acceptance Criteria\n- one\n- two\n\n# Constraints\n- safe\n",
        encoding="utf-8",
    )
    return spec


def _valid_spec(tmp_path: Path, spec_id: str = "demo_spec") -> Path:
    """Schema-v2 spec: full frontmatter, AC-n: criterion IDs, filename matches id."""
    spec = tmp_path / f"{spec_id}.SPEC.md"
    spec.write_text(
        f"---\nid: {spec_id}\ngoal: Do a thing\nmodule: src/\nstatus: approved\n---\n\n"
        "# Goal\nDo a thing.\n\n# Acceptance Criteria\n- AC-1: one\n- AC-2: two\n\n# Constraints\n- safe\n",
        encoding="utf-8",
    )
    return spec


def test_validate_spec_ok(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``harness validate-spec`` accepts a schema-v2 spec and exits 0."""
    spec = _valid_spec(tmp_path)
    rc = main(["validate-spec", str(spec)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok:" in captured.out
    assert "id='demo_spec'" in captured.out
    assert "status=approved" in captured.out
    assert "criteria=2" in captured.out


def test_validate_spec_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A missing spec file is a hard error with exit code 1."""
    rc = main(["validate-spec", str(tmp_path / "absent.md")])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error" in captured.err


def test_dry_run_prints_plan(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``harness dry-run`` parses the spec and prints the heuristic plan."""
    spec = _spec(tmp_path)
    rc = main(["dry-run", "--spec", str(spec)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "plan_steps" in captured.out
    assert "Do a thing" in captured.out


def test_dry_run_strips_authored_id_prefixes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """dry-run criteria descriptions carry no ``AC-n:`` prefix (verifier matches on them)."""
    spec = _valid_spec(tmp_path)
    rc = main(["dry-run", "--spec", str(spec)])
    captured = capsys.readouterr()
    assert rc == 0
    assert '"one"' in captured.out
    assert "AC-1:" not in captured.out


def test_validate_spec_ok_line_carries_warning_count(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """'ok with warnings' is visible on stdout, not only stderr; exit stays 0."""
    spec = tmp_path / "warned.SPEC.md"
    spec.write_text(
        "---\nid: warned\ngoal: g\nstatus: draft\n---\n\n# Goal\ng\n\n# Acceptance Criteria\n- AC-1: x\n",
        encoding="utf-8",
    )
    rc = main(["validate-spec", str(spec)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "warnings=1" in captured.out  # missing-module
    assert "missing-module" in captured.err


def test_validate_spec_errors_on_missing_goal(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A spec with no goal is now an error (exit 1), reported with its code."""
    spec = tmp_path / "nogoal.SPEC.md"
    spec.write_text(
        "---\nid: nogoal\nmodule: src/\nstatus: draft\n---\n\n# Acceptance Criteria\n- AC-1: safe\n",
        encoding="utf-8",
    )
    rc = main(["validate-spec", str(spec)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing-goal" in captured.err


def test_validate_spec_rejects_legacy_format(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A pre-v2 spec (no frontmatter) fails schema validation with clear codes."""
    spec = _spec(tmp_path)
    rc = main(["validate-spec", str(spec)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing-id" in captured.err
    assert "missing-status" in captured.err


def test_validate_spec_multiple_paths_duplicate_id(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The multi-path form detects the same spec id declared by two files."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    first = _valid_spec(a_dir)
    second = _valid_spec(b_dir)
    rc = main(["validate-spec", str(first), str(second)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "duplicate-id" in captured.err


# ---------------------------------------------------------------------------
# spec-new / spec-status / spec-trace subcommands (SDD Phase 1)
# ---------------------------------------------------------------------------


def test_spec_new_creates_draft_and_refuses_duplicates(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    specs = tmp_path / "specs"
    rc = main(["spec-new", "--id", "demo_new", "--module", "src/api/", "--specs-dir", str(specs)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "created:" in captured.out
    assert (specs / "demo_new.SPEC.md").exists()

    rc = main(["spec-new", "--id", "demo_new", "--module", "docs/", "--specs-dir", str(specs)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "already exists" in captured.err

    # The fresh draft is open: an overlapping module is refused deterministically.
    rc = main(["spec-new", "--id", "demo_other", "--module", "src/", "--specs-dir", str(specs)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "overlaps open spec" in captured.err


def test_spec_status_reports_and_requires(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    specs = tmp_path / "specs"
    main(["spec-new", "--id", "demo_status", "--module", "src/api/", "--specs-dir", str(specs)])
    capsys.readouterr()

    rc = main(["spec-status", "demo_status", "--specs-dir", str(specs)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "status=draft" in captured.out

    rc = main(["spec-status", "demo_status", "--require", "approved", "--specs-dir", str(specs)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "required 'approved'" in captured.err

    rc = main(["spec-status", "absent_spec", "--specs-dir", str(specs)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error" in captured.err


def test_spec_trace_cli_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import subprocess

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "x.py").write_text("X = 1\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "initial"], check=True)
    subprocess.run([*git, "switch", "-qc", "feature/foo"], check=True)
    (repo / "src" / "x.py").write_text("X = 2\n")
    subprocess.run([*git, "commit", "-qam", "change"], check=True)
    monkeypatch.chdir(repo)

    rc = main(["spec-trace", "--base-ref", "main", "--branch", "feature/foo"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAILED" in captured.out
    assert "No-Spec" in captured.err

    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x\n\nNo-Spec: test exemption"], check=True)
    rc = main(["spec-trace", "--base-ref", "main", "--branch", "feature/foo"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK" in captured.out

    # Operational failure (bogus base ref): error line + exit 1, no traceback.
    rc = main(["spec-trace", "--base-ref", "no_such_ref", "--branch", "feature/foo"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error: spec-trace:" in captured.err


def test_spec_trace_cli_outside_a_repo_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    monkeypatch.chdir(outside)
    rc = main(["spec-trace", "--base-ref", "main", "--branch", "feature/foo"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error: spec-trace:" in captured.err


# ---------------------------------------------------------------------------
# _resolve_intent / _apply_settings_overrides units
# ---------------------------------------------------------------------------


def test_resolve_intent_from_spec(tmp_path: Path) -> None:
    """A --spec path yields a normalized intent dict with criteria/constraints."""
    spec = _spec(tmp_path)
    args = argparse.Namespace(spec=spec, goal=None)
    intent = _resolve_intent(args)
    assert isinstance(intent, dict)
    assert intent["goal"] == "Do a thing."
    assert [c["description"] for c in intent["acceptance_criteria"]] == ["one", "two"]
    assert intent["constraints"] == ["safe"]
    assert intent["metadata"]["spec_path"] == str(spec)


def test_resolve_intent_uses_authored_ids(tmp_path: Path) -> None:
    """Authored AC-n criterion IDs flow into the intent; descriptions are prefix-free."""
    spec = _valid_spec(tmp_path)
    args = argparse.Namespace(spec=spec, goal=None)
    intent = _resolve_intent(args)
    assert isinstance(intent, dict)
    assert [c["id"] for c in intent["acceptance_criteria"]] == ["AC-1", "AC-2"]
    assert [c["description"] for c in intent["acceptance_criteria"]] == ["one", "two"]


def test_resolve_intent_from_goal() -> None:
    """An inline --goal (no spec) is returned verbatim as a string."""
    args = argparse.Namespace(spec=None, goal="just do it")
    assert _resolve_intent(args) == "just do it"


def test_resolve_intent_requires_spec_or_goal() -> None:
    """Neither --spec nor --goal is a hard error."""
    args = argparse.Namespace(spec=None, goal=None)
    with pytest.raises(SystemExit):
        _resolve_intent(args)


def test_apply_settings_overrides_promotes_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI flags are promoted into HARNESS_* env vars before settings load."""
    # _apply_settings_overrides writes directly to os.environ; pre-clear every key
    # it sets via monkeypatch so they are restored after the test (no env leak).
    monkeypatch.delenv("HARNESS_MAX_ITERATIONS", raising=False)
    monkeypatch.delenv("HARNESS_MEMORY_ROOT", raising=False)
    monkeypatch.delenv("HARNESS_OUTPUT_DIR", raising=False)
    args = argparse.Namespace(
        max_iterations=7,
        memory_root=tmp_path / "mem",
        output_dir=tmp_path / "out",
    )
    settings = _apply_settings_overrides(args)
    import os

    assert os.environ["HARNESS_MAX_ITERATIONS"] == "7"
    assert os.environ["HARNESS_MEMORY_ROOT"] == str(tmp_path / "mem")
    assert os.environ["HARNESS_OUTPUT_DIR"] == str(tmp_path / "out")
    assert settings is not None


# ---------------------------------------------------------------------------
# run / replay paths (factory mocked — no real LLM)
# ---------------------------------------------------------------------------


def _patch_factory(monkeypatch: pytest.MonkeyPatch, runner: object) -> MagicMock:
    """Patch cli.HarnessFactory so create_runner returns ``runner``."""
    factory = MagicMock()
    factory.create_runner.return_value = runner
    factory_cls = MagicMock(return_value=factory)
    monkeypatch.setattr("src.framework.harness.cli.HarnessFactory", factory_cls)
    return factory


def test_run_accepted_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """An accepted terminal outcome exits 0 and emits JSON when --json is set."""
    runner = MagicMock()
    runner.run = AsyncMock(
        return_value=RunResult(
            outcome=Terminal(accepted=True),
            state=MagicMock(),
            iterations=2,
            duration_ms=12.345,
            confidence=0.9,
            metadata={"k": "v"},
        )
    )
    _patch_factory(monkeypatch, runner)
    rc = main(["run", "--goal", "do a thing", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"accepted": true' in out
    assert '"outcome": "terminal"' in out


def test_run_rejected_plain_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """A non-accepted outcome exits 2 and prints key=value lines."""
    runner = MagicMock()
    runner.run = AsyncMock(
        return_value=RunResult(outcome=Terminal(accepted=False), state=MagicMock(), iterations=1, confidence=0.1)
    )
    _patch_factory(monkeypatch, runner)
    rc = main(["run", "--goal", "do a thing"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "outcome=terminal" in out
    assert "accepted=False" in out


def test_run_ralph_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """The --ralph flag drives the Ralph loop and reports its status."""
    runner = MagicMock()
    last_run = SimpleNamespace(outcome=SimpleNamespace(kind="terminal"), confidence=0.8)
    ralph_result = SimpleNamespace(status="accepted", rounds=3, stuck_kind=None, last_run=last_run)
    ralph_loop = MagicMock()
    ralph_loop.run = AsyncMock(return_value=ralph_result)
    factory = _patch_factory(monkeypatch, runner)
    factory.create_ralph.return_value = ralph_loop

    rc = main(["run", "--goal", "do a thing", "--ralph", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"status": "accepted"' in out
    assert '"rounds": 3' in out


def test_replay_sets_env_and_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    """``harness replay`` promotes the cassette dir into the env and runs."""
    runner = MagicMock()
    runner.run = AsyncMock(return_value=RunResult(outcome=Terminal(accepted=True), state=MagicMock(), iterations=1))
    _patch_factory(monkeypatch, runner)
    cassette = tmp_path / "cass"
    cassette.mkdir()
    rc = main(["replay", "--cassette-dir", str(cassette), "--goal", "do a thing"])
    import os

    assert rc == 0
    assert os.environ["HARNESS_REPLAY_DIR"] == str(cassette)
