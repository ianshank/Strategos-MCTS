"""
Tests for the CUDA probe in the root ``healthcheck.py``.

The training image is used on GPU hosts and on CPU-only hosts — CI smoke tests run
it on ``ubuntu-latest``. The probe previously reported ``UNHEALTHY`` with
``critical=True`` whenever CUDA was absent, and the check registration hard-coded
the same flag, so the container could never report healthy off-GPU. That kept
``docker-deployment.yml``'s Container Smoke Tests red for a reason unrelated to the
image under test.

An absent GPU is now DEGRADED and non-critical by default, and fatal only where an
operator declares a GPU necessary via ``REQUIRE_GPU``.

``torch`` is stubbed rather than probed so both branches are exercised on any host,
including the GPU-present path this machine cannot reach.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import healthcheck  # noqa: E402

pytestmark = pytest.mark.unit


def _run(coro):
    """Drive a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch):
    """Stub torch as installed but reporting no CUDA device."""
    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", stub)


@pytest.fixture
def with_cuda(monkeypatch: pytest.MonkeyPatch):
    """Stub torch reporting one visible GPU."""
    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda i: "STUB-GPU",
    )
    monkeypatch.setitem(sys.modules, "torch", stub)


class TestGpuRequiredFlag:
    """Parsing of the REQUIRE_GPU environment flag."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " 1 "])
    def test_truthy_values_require_a_gpu(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Case and surrounding whitespace must not change the verdict."""
        monkeypatch.setenv("REQUIRE_GPU", value)

        assert healthcheck.HealthChecker._gpu_required() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_falsy_values_do_not(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """An explicit falsy value is as permissive as an unset one."""
        monkeypatch.setenv("REQUIRE_GPU", value)

        assert healthcheck.HealthChecker._gpu_required() is False

    def test_unset_defaults_to_not_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default must be permissive — CI runs this image without a GPU."""
        monkeypatch.delenv("REQUIRE_GPU", raising=False)

        assert healthcheck.HealthChecker._gpu_required() is False


class TestCudaAbsentByDefault:
    """The regression: a CPU-only host must still be able to report healthy."""

    def test_absent_cuda_is_degraded_not_unhealthy(self, monkeypatch: pytest.MonkeyPatch, no_cuda) -> None:
        """The core regression: a CPU host must not be reported as unhealthy."""
        monkeypatch.delenv("REQUIRE_GPU", raising=False)

        result = _run(healthcheck.HealthChecker().check_cuda())

        assert result.status is healthcheck.HealthStatus.DEGRADED
        assert result.critical is False

    def test_message_names_the_override(self, monkeypatch: pytest.MonkeyPatch, no_cuda) -> None:
        """An operator reading the degraded message learns how to make it fatal."""
        monkeypatch.delenv("REQUIRE_GPU", raising=False)

        assert "REQUIRE_GPU=1" in _run(healthcheck.HealthChecker().check_cuda()).message

    def test_zero_devices_is_also_degraded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """torch.cuda.is_available() can be True with device_count() == 0."""
        monkeypatch.delenv("REQUIRE_GPU", raising=False)
        stub = types.ModuleType("torch")
        stub.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 0)
        monkeypatch.setitem(sys.modules, "torch", stub)

        result = _run(healthcheck.HealthChecker().check_cuda())

        assert result.status is healthcheck.HealthStatus.DEGRADED
        assert result.critical is False

    def test_missing_torch_is_degraded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No torch is a missing capability, not a failed container."""
        monkeypatch.delenv("REQUIRE_GPU", raising=False)
        monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`

        result = _run(healthcheck.HealthChecker().check_cuda())

        assert result.status is healthcheck.HealthStatus.DEGRADED
        assert result.critical is False


class TestCudaRequired:
    """Where a GPU is declared necessary, the check must still gate."""

    def test_absent_cuda_is_unhealthy_and_critical(self, monkeypatch: pytest.MonkeyPatch, no_cuda) -> None:
        """Opting in must restore the original gating behaviour exactly."""
        monkeypatch.setenv("REQUIRE_GPU", "1")

        result = _run(healthcheck.HealthChecker().check_cuda())

        assert result.status is healthcheck.HealthStatus.UNHEALTHY
        assert result.critical is True

    def test_no_override_hint_when_already_required(self, monkeypatch: pytest.MonkeyPatch, no_cuda) -> None:
        """Do not advise setting a flag the operator has already set."""
        monkeypatch.setenv("REQUIRE_GPU", "1")

        assert "REQUIRE_GPU=1" not in _run(healthcheck.HealthChecker().check_cuda()).message


class TestCudaPresent:
    """With a visible GPU the probe is healthy regardless of the flag."""

    def test_available_gpu_reports_healthy(self, monkeypatch: pytest.MonkeyPatch, with_cuda) -> None:
        """Device metadata is surfaced so operators can confirm what was found."""
        monkeypatch.delenv("REQUIRE_GPU", raising=False)

        result = _run(healthcheck.HealthChecker().check_cuda())

        assert result.status is healthcheck.HealthStatus.HEALTHY
        assert result.metadata["gpu_count"] == 1

    def test_healthy_regardless_of_require_gpu(self, monkeypatch: pytest.MonkeyPatch, with_cuda) -> None:
        """The flag changes the absent-GPU verdict only, never the present-GPU one."""
        monkeypatch.setenv("REQUIRE_GPU", "1")

        assert _run(healthcheck.HealthChecker().check_cuda()).status is healthcheck.HealthStatus.HEALTHY
