"""CLI smoke tests — exercise ``dry-run`` and ``validate-spec`` paths.

We avoid invoking ``harness run`` here because that path needs a real LLM
client; the runner integration test already covers the run path with a
scripted LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.harness.cli import main

pytestmark = pytest.mark.unit


def _spec(tmp_path: Path) -> Path:
    spec = tmp_path / "spec.md"
    spec.write_text(
        "# Goal\nDo a thing.\n\n# Acceptance Criteria\n- one\n- two\n\n# Constraints\n- safe\n",
        encoding="utf-8",
    )
    return spec


def test_validate_spec_ok(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``harness validate-spec`` accepts a well-formed spec and exits 0."""
    spec = _spec(tmp_path)
    rc = main(["validate-spec", str(spec)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok:" in captured.out
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
