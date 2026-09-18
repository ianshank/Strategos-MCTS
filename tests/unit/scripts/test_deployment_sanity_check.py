"""Regression tests for the pre-deployment smoke-test runner."""

from pathlib import Path
import subprocess

import pytest

import scripts.deployment_sanity_check as sanity

pytestmark = pytest.mark.unit


def _write_test(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_discover_smoke_test_files_excludes_unmarked_modules(tmp_path: Path) -> None:
    smoke_test = tmp_path / "test_smoke.py"
    _write_test(smoke_test, "@pytest.mark.smoke\ndef test_ready(): pass\n")
    _write_test(tmp_path / "test_other.py", "def test_other(): pass\n")

    assert sanity._discover_smoke_test_files(tmp_path) == [smoke_test]


def test_checker_rejects_nonpositive_smoke_timeout() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        sanity.DeploymentSanityChecker(smoke_test_timeout_seconds=0)


def test_check_test_suite_collects_only_smoke_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_test = tmp_path / "tests" / "test_smoke.py"
    _write_test(smoke_test, "pytestmark = pytest.mark.smoke\ndef test_ready(): pass\n")
    _write_test(tmp_path / "tests" / "test_other.py", "def test_other(): pass\n")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sanity, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sanity.subprocess, "run", fake_run)

    checker = sanity.DeploymentSanityChecker(smoke_test_timeout_seconds=17)

    assert checker.check_test_suite() is True
    assert captured["command"] == [
        sanity.sys.executable,
        "-m",
        "pytest",
        "tests/test_smoke.py",
        "-m",
        "smoke",
        "-v",
        "--tb=short",
    ]
    assert captured["timeout"] == 17
