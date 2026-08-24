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

from datetime import UTC, datetime

import pytest

import healthcheck

pytestmark = pytest.mark.unit

# Deterministic timestamp for test fixtures — avoids hard-coded date strings
# that silently rot and makes test output reproducible.
_TEST_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


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

    def test_all_enum_members_produce_valid_exit_code(self) -> None:
        """Guard against enum extensibility: every HealthStatus member must map to 0 or 1."""
        for status in healthcheck.HealthStatus:
            code = healthcheck.exit_code_for_status(status)
            assert code in {0, 1}, f"Unexpected exit code {code} for {status}"


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
        timestamp=_TEST_TIMESTAMP,
        duration_ms=1.0,
    )


class TestMainDegradedIsHealthy:
    """The smoke-test regression: a DEGRADED environment must exit 0."""

    @pytest.mark.asyncio
    async def test_main_exits_zero_when_degraded(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            await healthcheck.main()

        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_main_exits_one_when_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            timestamp=_TEST_TIMESTAMP,
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
            await healthcheck.main()

        assert exc_info.value.code == 1


class TestDegradedReportPreservesSignal:
    """The structured JSON report must still carry the DEGRADED status.

    The exit-code change (DEGRADED → 0) must not lose the signal that optional
    integrations are down. Operators rely on the report to distinguish HEALTHY
    from DEGRADED even though Docker sees both as exit 0.
    """

    def test_degraded_report_dict_carries_status(self) -> None:
        """The report's to_dict output must include 'degraded' status."""
        report = _degraded_report()
        report_dict = report.to_dict()
        assert report_dict["status"] == "degraded"

    def test_healthy_report_dict_carries_status(self) -> None:
        """Sanity: a HEALTHY report must carry 'healthy' status."""
        report = healthcheck.HealthCheckReport(
            status=healthcheck.HealthStatus.HEALTHY,
            checks=[],
            timestamp=_TEST_TIMESTAMP,
            duration_ms=0.5,
        )
        report_dict = report.to_dict()
        assert report_dict["status"] == "healthy"
