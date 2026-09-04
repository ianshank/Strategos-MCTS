"""Device-matrix helpers shared by the end-to-end and integration suites.

One answer to "which torch devices does this host offer, and which did the operator ask
for?" so no test file carries its own CUDA -> MPS -> CPU ladder. Availability is probed
through the same torch calls ``src/utils/device.py`` uses; the *policy* differs
deliberately, because a test suite and a runtime want opposite things from an absent
accelerator:

* ``src.utils.device.resolve_device("cuda")`` on a CPU host falls back to CPU with a
  warning - the right call for a training job that should still run.
* A test case labelled ``[cuda]`` must never silently run on CPU. Here an unavailable
  device is reported as a **skip with a reason** when the matrix is implicit, and as a
  **failure** when the operator named it explicitly in ``E2E_DEVICES`` - a pinned matrix
  that cannot be honoured is a broken host, not a green run.

The matrix is static (``DEVICE_MATRIX``) rather than "whatever is available" on purpose:
on a CPU-only CI runner the accelerator cases then appear in the report as *skipped*,
which is visible, instead of being absent, which is not. Registered marker: ``gpu``
(``pyproject.toml``), attached to every non-CPU case so ``-m "not gpu"`` deselects them.

Environment contract (test-side only; nothing under ``src/`` reads it):

``E2E_DEVICES``
    Comma-separated subset of ``cpu,cuda,mps``. When set, only those devices are
    parametrized and each is *required*. Unknown names raise at collection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any, Final

import pytest

CPU_DEVICE: Final[str] = "cpu"
CUDA_DEVICE: Final[str] = "cuda"
MPS_DEVICE: Final[str] = "mps"

#: Every device a test may be parametrized on, in report order.
DEVICE_MATRIX: Final[tuple[str, ...]] = (CPU_DEVICE, CUDA_DEVICE, MPS_DEVICE)
#: The non-CPU subset; these carry the ``gpu`` marker.
ACCELERATOR_DEVICES: Final[tuple[str, ...]] = tuple(d for d in DEVICE_MATRIX if d != CPU_DEVICE)
#: Operator override naming the exact matrix to run (see module docstring).
DEVICE_MATRIX_ENV: Final[str] = "E2E_DEVICES"

GPU_MARKER_NAME: Final[str] = "gpu"

#: Reported when an accelerator-only parametrization has no accelerator to run on, so the
#: skip names its cause instead of pytest's bare "got empty parameter set".
NO_ACCELERATOR_ID: Final[str] = "no-accelerator"
NO_ACCELERATOR_SKIP_REASON: Final[str] = (
    f"the device matrix contains no accelerator, so there is nothing to compare against CPU "
    f"(set {DEVICE_MATRIX_ENV} to a set that includes one)"
)


@dataclass(frozen=True)
class DeviceCase:
    """One cell of the device matrix, with everything a fixture needs to decide its fate."""

    name: str
    available: bool
    requested: bool

    @property
    def is_accelerator(self) -> bool:
        return self.name != CPU_DEVICE

    @property
    def skip_reason(self) -> str | None:
        """Why this case should be skipped, or ``None`` when it must run (or fail)."""
        if self.available or self.requested:
            return None
        return f"{self.name} is not available on this host (set {DEVICE_MATRIX_ENV}={self.name} to require it)"

    @property
    def failure_reason(self) -> str | None:
        """Why this case must *fail*: it was explicitly required and the host lacks it."""
        if self.available or not self.requested:
            return None
        return (
            f"{DEVICE_MATRIX_ENV} names {self.name!r} but this host does not provide it; "
            f"a pinned device matrix that cannot be honoured is a broken host, not a skip"
        )


def torch_available() -> bool:
    """True when ``torch`` imports; never raises, so conftest import stays safe without the neural extra."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def device_available(device: str) -> bool:
    """Whether ``device`` can host a tensor on this machine. CPU is always available."""
    if device == CPU_DEVICE:
        return True
    if device not in DEVICE_MATRIX:
        raise ValueError(f"unknown device {device!r}; expected one of {DEVICE_MATRIX}")
    if not torch_available():
        return False
    import torch

    if device == CUDA_DEVICE:
        return bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def requested_devices(environ: Mapping[str, str] | None = None) -> tuple[str, ...] | None:
    """Parse ``E2E_DEVICES``; ``None`` when unset or blank, so the implicit matrix applies."""
    raw = (environ if environ is not None else os.environ).get(DEVICE_MATRIX_ENV, "")
    names = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not names:
        return None
    unknown = [name for name in names if name not in DEVICE_MATRIX]
    if unknown:
        raise ValueError(
            f"{DEVICE_MATRIX_ENV} contains unknown device(s) {unknown}; expected a subset of {DEVICE_MATRIX}"
        )
    # Preserve matrix order and drop duplicates.
    return tuple(device for device in DEVICE_MATRIX if device in names)


def device_cases(environ: Mapping[str, str] | None = None) -> tuple[DeviceCase, ...]:
    """The matrix this host will run: every ``DEVICE_MATRIX`` entry, or exactly the requested ones."""
    requested = requested_devices(environ)
    names = DEVICE_MATRIX if requested is None else requested
    return tuple(
        DeviceCase(name=name, available=device_available(name), requested=requested is not None) for name in names
    )


def device_params(
    cases: tuple[DeviceCase, ...] | None = None,
    *,
    accelerators_only: bool = False,
) -> list[Any]:
    """``pytest.param`` list for a ``params=`` fixture or ``parametrize``.

    Skip and ``gpu`` marks are attached at collection so ``-m "not gpu"`` and the
    junit/summary skip counts see them. A required-but-absent device is *not* marked
    skip here; the consuming fixture fails it with ``DeviceCase.failure_reason`` so the
    failure carries the operator's own configuration in its message.

    The list is never empty. With ``accelerators_only`` and a matrix that contains no
    accelerator at all — which is what ``E2E_DEVICES=cpu`` produces — an explicitly
    reasoned placeholder is emitted instead. pytest renders an empty ``params=`` as
    "got empty parameter set", which says nothing about *why* the case vanished, and in
    this suite a skip that does not name its reason is the failure being designed
    against.
    """
    resolved = device_cases() if cases is None else cases
    params: list[Any] = []
    for case in resolved:
        if accelerators_only and not case.is_accelerator:
            continue
        marks: list[Any] = []
        if case.is_accelerator:
            marks.append(getattr(pytest.mark, GPU_MARKER_NAME))
        if case.skip_reason is not None:
            marks.append(pytest.mark.skip(reason=case.skip_reason))
        params.append(pytest.param(case, id=case.name, marks=marks))

    if accelerators_only and not params:
        # Deliberately carries an *unavailable accelerator* rather than a CPU case: if the
        # skip mark below were ever removed, the consuming test would fail loudly on a
        # missing device instead of silently "passing" a cross-device comparison against
        # CPU on both sides.
        placeholder = DeviceCase(name=ACCELERATOR_DEVICES[0], available=False, requested=False)
        params.append(
            pytest.param(
                placeholder,
                id=NO_ACCELERATOR_ID,
                marks=[
                    getattr(pytest.mark, GPU_MARKER_NAME),
                    pytest.mark.skip(reason=NO_ACCELERATOR_SKIP_REASON),
                ],
            )
        )
    return params


def require_case(case: DeviceCase) -> str:
    """Return the device string for ``case`` or fail loudly when it was required but is absent."""
    if case.failure_reason is not None:
        pytest.fail(case.failure_reason)
    return case.name
