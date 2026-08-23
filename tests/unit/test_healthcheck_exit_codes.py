"""
Tests for the healthcheck exit-code contract.

The training image's ``Dockerfile.train`` wires ``python /app/healthcheck.py``
as the container ``HEALTHCHECK``. Docker treats only exit 0 as healthy; any
non-zero exit (including the previously-used code 2, which Docker reserves)
is a check failure. The smoke-test environment runs the image with no GPU, no
LLM provider key, no Pinecone host and no OTEL endpoint, so every check
reports DEGRADED (non-critical). The overall status is therefore DEGRADED.

The regression: DEGRADED previously mapped to exit 2, so the container could
never become healthy on a CPU-only host or in CI, and
``docker-deployment.yml``'s Container Smoke Tests stayed red for a reason
unrelated to the image under test. DEGRADED now maps to exit 0 — the container
is operational — while the structured JSON report still distinguishes HEALTHY
from DEGRADED so the degraded signal is not lost.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import healthcheck  # noqa: E402

pytestmark = pytest.mark.unit


class TestExitCodeMapping:
    """The Docker HEALTHCHECK exit-code contract for each overall status."""

    @pytest.mark.parametrize(
        "status, expected",
        [
            (healthcheck.HealthStatus.HEALTHY, 0),
            (healthcheck.HealthStatus.DEGRADED, 0),
            (healthcheck.HealthStatus.UNHEALTHY, 1),
        ],
    )
    def test_status_maps_to_exit_code(self, status, expected) -> None:
        """HEALTHY and DEGRADED are operational (exit 0); UNHEALTHY is exit 1."""
        assert healthcheck.exit_code_for_status(status) == expected

    def test_degraded_is_not_exit_two(self) -> None:
        """The core regression: exit 2 is reserved by Docker and must not be used."""
        assert healthcheck.exit_code_for_status(healthcheck.HealthStatus.DEGRADED) != 2


def _degraded_report() -> healthcheck.HealthCheckReport:
    """A report where only non-critical optional services are down."""
    return healthcheck.HealthCheckReport(
        status=healthcheck.HealthStatus.DEGRADED,
        checks=[
            healthcheck.CheckResult(
                name="cuda",
                status=healthcheck.HealthStatus.DEGRADED,
                message="CUDA not available",
                duration_ms=0,
                critical=False,
            ),
        ],
        timestamp="2026-08-23T00:00:00Z",
        duration_ms=1.0,
    )


class TestMainDegradedIsHealthy:
    """The smoke-test regression: a DEGRADED environment must exit 0."""

    def test_main_exits_zero_when_degraded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A container whose only failures are optional services is operational.

        This is the exact CI smoke environment: no GPU, no LLM key, no Pinecone
        host, no OTEL endpoint. ``run_all_checks`` is stubbed to return that
        DEGRADED report; ``main()`` must exit 0 so Docker reports the container
        as healthy.
        """

        async def _stub_run_all_checks(self):
            return _degraded_report()

        monkeypatch.setattr(
            healthcheck.HealthChecker,
            "run_all_checks",
            _stub_run_all_checks,
        )
        # healthcheck.main prints a JSON report to stdout; keep it quiet.
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(healthcheck.main())

        assert exc_info.value.code == 0

    def test_main_exits_one_when_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A critical failure must still fail the healthcheck (exit 1)."""
        report = healthcheck.HealthCheckReport(
            status=healthcheck.HealthStatus.UNHEALTHY,
            checks=[
                healthcheck.CheckResult(
                    name="llm_openai",
                    status=healthcheck.HealthStatus.UNHEALTHY,
                    message="Client error",
                    duration_ms=0,
                    critical=True,
                ),
            ],
            timestamp="2026-08-23T00:00:00Z",
            duration_ms=1.0,
        )

        async def _stub_run_all_checks(self):
            return report

        monkeypatch.setattr(
            healthcheck.HealthChecker,
            "run_all_checks",
            _stub_run_all_checks,
        )
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)

        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(healthcheck.main())

        assert exc_info.value.code == 1
