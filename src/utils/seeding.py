"""Rank-aware seeding utilities for reproducible runs.

Centralises the hand-rolled ``torch.manual_seed`` / ``np.random.seed`` /
``random.seed`` blocks that previously diverged across training drivers and
engines. Callers pass an explicit seed (typically ``Settings.SEED`` or
``DEFAULT_SEED``) — this module does **not** introduce a new seed env var
(``hygiene_determinism`` AC-2).

Also provides :func:`new_rng` for constructing an injected
``numpy.random.Generator`` (the path NeuralMCTS uses for Dirichlet noise).
"""

from __future__ import annotations

import random

import numpy as np

from src.config.constants import DEFAULT_SEED
from src.observability.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "set_all_seeds",
    "new_rng",
    "resolve_seed",
]


def resolve_seed(seed: int | None = None) -> int:
    """Resolve an effective seed from an explicit value, Settings.SEED, or DEFAULT_SEED.

    Preference order:
    1. Explicit ``seed`` argument when not ``None``.
    2. ``Settings.SEED`` when configured (optional reproducibility override).
    3. ``DEFAULT_SEED`` from :mod:`src.config.constants`.

    No new environment variable is read here (AC-2).
    """
    if seed is not None:
        return int(seed)
    try:
        from src.config.settings import get_settings

        settings_seed = get_settings().SEED
        if settings_seed is not None:
            return int(settings_seed)
    except Exception:  # noqa: BLE001 — settings may be unavailable at import/test time
        pass
    return DEFAULT_SEED


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

    Args:
        seed: Base seed (typically ``Settings.SEED`` or ``DEFAULT_SEED``).
        rank: Process rank for distributed runs; added to ``seed``.
        deterministic_torch: When True and torch is available, force cudnn
            deterministic mode (slower; useful for bitwise reproducibility).

    Returns:
        The effective seed that was applied (``seed + rank``).
    """
    effective = int(seed) + int(rank)
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
