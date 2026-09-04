"""Fixtures for the end-to-end suite.

Two things every module here needs, provided once:

* **A device matrix.** ``device_case`` parametrizes over ``cpu``/``cuda``/``mps``
  statically, so on a CPU-only runner the accelerator cases are reported as *skipped
  with a reason* rather than being absent from the report. That distinction is the whole
  point: "68 passed" on a CPU host must not read as "the CUDA path is tested".
  ``accelerator_case`` is the second matrix for tests that compare an accelerator
  *against* CPU; it is empty (one collected skip) on a CPU-only host.
* **A hermetic subprocess environment.** ``e2e_env`` strips every provider and tracker
  credential the parent shell may hold before a child process starts, so a test that
  spawns a real entry point cannot reach a paid API even when this suite runs in the
  post-merge workflow that *does* export real keys.

This file must stay importable without the ``neural`` extra: it is imported at
collection for the whole directory, including the torch-free modules and the 68 legacy
tests, so every torch probe goes through ``tests.utils.device_matrix``, which returns
"unavailable" instead of raising when torch is absent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
import os
from pathlib import Path
from typing import Any, Final

import pytest

from tests.utils.device_matrix import (
    DeviceCase,
    device_params,
    require_case,
)
from tests.utils.e2e_process import (
    DUMMY_PROVIDER_ENV,
    ProcessResult,
    hermetic_env,
    run_command,
    run_python_module,
)

logger = logging.getLogger("tests.e2e")

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Seed handed to every entry point that takes one. Read from the process environment so
#: an operator can re-run the suite under a different seed without editing tests; this
#: reuses the project's existing ``SEED`` channel rather than adding a variable
#: (``specs/hygiene_determinism.SPEC.md`` AC-2).
DEFAULT_E2E_SEED: Final[int] = 0
SEED_ENV: Final[str] = "SEED"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository checkout under test."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def e2e_seed() -> int:
    """The seed every entry point in this suite is driven with."""
    raw = os.environ.get(SEED_ENV, "").strip()
    if not raw:
        return DEFAULT_E2E_SEED
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r; using %d", SEED_ENV, raw, DEFAULT_E2E_SEED)
        return DEFAULT_E2E_SEED


@pytest.fixture(params=device_params(), ids=lambda case: case.name)
def device_case(request: pytest.FixtureRequest) -> DeviceCase:
    """One cell of the device matrix.

    Unavailable devices arrive already marked skip (attached at collection, so ``-m "not
    gpu"`` and the junit skip counts see them). The one case that reaches this body
    unavailable is a device the operator *required* via ``E2E_DEVICES``; that fails here
    rather than silently running on CPU.
    """
    case: DeviceCase = request.param
    require_case(case)
    logger.debug("device case: %s", case.name)
    return case


@pytest.fixture
def device(device_case: DeviceCase) -> str:
    """The device string to hand to a ``src/`` seam (``NeuralMCTS``, ``--device``, ...)."""
    return device_case.name


@pytest.fixture(params=device_params(accelerators_only=True), ids=lambda case: case.name)
def accelerator_case(request: pytest.FixtureRequest) -> DeviceCase:
    """A non-CPU device, for tests that compare accelerator output against CPU."""
    case: DeviceCase = request.param
    require_case(case)
    return case


@pytest.fixture
def e2e_env(repo_root: Path) -> Callable[..., dict[str, str]]:
    """Factory for a hermetic child environment.

    Called with no arguments it yields the default posture: credentials stripped, offline
    flags pinned, and the placeholder provider key that satisfies ``Settings`` validation
    without ever being sent anywhere. Pass ``overrides`` to add or (with ``None``) remove
    variables — for example the rank variables ``torchrun`` would set.
    """

    def _make(overrides: Mapping[str, str | None] | None = None, *, with_provider_key: bool = True) -> dict[str, str]:
        base_overrides: dict[str, str | None] = dict(DUMMY_PROVIDER_ENV) if with_provider_key else {}
        base_overrides.update(overrides or {})
        return hermetic_env(repo_root=repo_root, overrides=base_overrides)

    return _make


@pytest.fixture
def run_module(repo_root: Path, e2e_env: Callable[..., dict[str, str]]) -> Callable[..., ProcessResult]:
    """Run ``python -m <module>`` hermetically and return its :class:`ProcessResult`."""

    def _run(
        module: str,
        args: list[str] | tuple[str, ...] = (),
        *,
        env_overrides: Mapping[str, str | None] | None = None,
        timeout: float | None = None,
        **env_kwargs: Any,
    ) -> ProcessResult:
        env = e2e_env(env_overrides, **env_kwargs)
        result = run_python_module(module, list(args), env=env, cwd=repo_root, timeout=timeout)
        logger.debug("run_module(%s) -> exit %s", module, result.returncode)
        return result

    return _run


@pytest.fixture
def run_script(repo_root: Path, e2e_env: Callable[..., dict[str, str]]) -> Callable[..., ProcessResult]:
    """Run an installed console script by name, exactly as a user would."""

    def _run(
        argv: list[str] | tuple[str, ...],
        *,
        env_overrides: Mapping[str, str | None] | None = None,
        timeout: float | None = None,
        **env_kwargs: Any,
    ) -> ProcessResult:
        env = e2e_env(env_overrides, **env_kwargs)
        result = run_command(list(argv), env=env, cwd=repo_root, timeout=timeout)
        logger.debug("run_script(%s) -> exit %s", argv[0], result.returncode)
        return result

    return _run
