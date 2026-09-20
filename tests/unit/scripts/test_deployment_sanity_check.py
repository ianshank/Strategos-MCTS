"""Regression tests for the pre-deployment smoke-test runner."""

import subprocess

import pytest

import scripts.deployment_sanity_check as sanity

pytestmark = pytest.mark.unit


def test_checker_rejects_nonpositive_smoke_timeout() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        sanity.DeploymentSanityChecker(smoke_test_timeout_seconds=0)


def test_check_test_suite_runs_explicit_smoke_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sanity.subprocess, "run", fake_run)

    checker = sanity.DeploymentSanityChecker(smoke_test_timeout_seconds=17)

    assert checker.check_test_suite() is True
    assert captured["command"] == [sanity.sys.executable, "-m", "pytest", *sanity.SANITY_SMOKE_ARGS]
    marker_index = captured["command"].index("-m", 3)
    assert captured["command"][marker_index + 1] == "smoke"
    assert captured["timeout"] == 17
