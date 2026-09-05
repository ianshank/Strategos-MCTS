"""Rank-aware seeding utilities for reproducible runs.

Centralises the hand-rolled ``torch.manual_seed`` / ``np.random.seed`` /
``random.seed`` blocks that previously diverged across training drivers and
engines. Callers pass an explicit seed (typically ``Settings.SEED`` or
``DEFAULT_SEED``) — this module does **not** introduce a new seed env var
(``hygiene_determinism`` AC-2).

Also provides :func:`new_rng` for constructing an injected
``numpy.random.Generator`` (the path NeuralMCTS uses for Dirichlet noise).

NumPy legacy ``np.random.seed`` / ``RandomState`` only accepts integers in
``[NUMPY_LEGACY_SEED_MIN, NUMPY_LEGACY_SEED_MAX]`` (``0 .. 2**32 - 1``).
:func:`validate_numpy_seed` enforces that bound so callers fail early with a
clear ``ValueError`` rather than deep inside NumPy.
"""

from __future__ import annotations

import random
from typing import Final

import numpy as np

from src.config.constants import DEFAULT_SEED
from src.observability.logging import get_logger

logger = get_logger(__name__)

# NumPy legacy RandomState / np.random.seed accepted range (MT19937).
# SeedSequence / Generator accept a wider domain, but we keep one shared
# bound so resolve_seed / set_all_seeds / new_rng fail consistently early.
NUMPY_LEGACY_SEED_MIN: Final[int] = 0
NUMPY_LEGACY_SEED_MAX: Final[int] = 2**32 - 1  # 4_294_967_295

__all__ = [
    "NUMPY_LEGACY_SEED_MIN",
    "NUMPY_LEGACY_SEED_MAX",
    "validate_numpy_seed",
    "set_all_seeds",
    "new_rng",
    "resolve_seed",
]


def validate_numpy_seed(seed: int, *, label: str = "seed") -> int:
    """Validate ``seed`` is in the NumPy legacy-safe integer range.

    Args:
        seed: Candidate seed value (coerced with ``int(...)``).
        label: Name used in the ``ValueError`` message (e.g. ``"effective seed"``).

    Returns:
        The validated integer seed.

    Raises:
        ValueError: If ``seed`` is outside
            ``[NUMPY_LEGACY_SEED_MIN, NUMPY_LEGACY_SEED_MAX]``.
    """
    value = int(seed)
    if value < NUMPY_LEGACY_SEED_MIN or value > NUMPY_LEGACY_SEED_MAX:
        raise ValueError(
            f"{label} must be in [{NUMPY_LEGACY_SEED_MIN}, {NUMPY_LEGACY_SEED_MAX}] "
            f"(NumPy legacy np.random.seed / RandomState range); got {value}"
        )
    return value


def resolve_seed(seed: int | None = None) -> int:
    """Resolve an effective seed from an explicit value, Settings.SEED, or DEFAULT_SEED.

    Preference order:
    1. Explicit ``seed`` argument when not ``None``.
    2. ``Settings.SEED`` when configured (optional reproducibility override).
    3. ``DEFAULT_SEED`` from :mod:`src.config.constants`.

    The resolved value is validated against the NumPy legacy-safe range so
    callers fail early (before seeding or constructing a Generator).

    No new environment variable is read here (AC-2).
    """
    if seed is not None:
        return validate_numpy_seed(seed, label="seed")
    try:
        from src.config.settings import get_settings

        settings_seed = get_settings().SEED
        if settings_seed is not None:
            return validate_numpy_seed(settings_seed, label="Settings.SEED")
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 — settings may be unavailable at import/test time
        pass
    return validate_numpy_seed(DEFAULT_SEED, label="DEFAULT_SEED")


def set_all_seeds(
    seed: int,
    *,
    rank: int = 0,
    deterministic_torch: bool = False,
) -> int:
    """Seed Python ``random``, NumPy's legacy global RNG, and torch (when installed).

    The effective seed is rank-aware (``seed + rank``) so DDP workers diverge
    in a controlled way. Torch seeding is behind an import guard: if torch is
    not installed the Python/NumPy seeds are still applied and torch is skipped.

    The **effective** seed (``seed + rank``) is validated against the NumPy
    legacy-safe range before any RNG is touched.

    Args:
        seed: Base seed (typically ``Settings.SEED`` or ``DEFAULT_SEED``).
        rank: Process rank for distributed runs; added to ``seed``.
        deterministic_torch: When True and torch is available, force cudnn
            deterministic mode (slower; useful for bitwise reproducibility).

    Returns:
        The effective seed that was applied (``seed + rank``).

    Raises:
        ValueError: If the effective seed is outside the NumPy legacy-safe range.
    """
    effective = validate_numpy_seed(int(seed) + int(rank), label="effective seed")
    random.seed(effective)
    np.random.seed(effective)  # noqa: NPY002 — deliberate: this IS the central legacy-RNG seeder

    try:
        import torch
    except ImportError:
        logger.info("Effective seed=%d (torch unavailable; python/numpy only)", effective)
        return effective

    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)

    if deterministic_torch:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Effective seed=%d (deterministic_torch=True)", effective)
    else:
        logger.info("Effective seed=%d", effective)

    return effective


def new_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh ``numpy.random.Generator`` seeded via :func:`resolve_seed`.

    Prefer this over NumPy's process-global legacy RNG when injecting noise into
    search engines (e.g. NeuralMCTS Dirichlet root noise).
    """
    return np.random.default_rng(resolve_seed(seed))
