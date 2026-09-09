"""Visit-count policy algebra: π_train is always τ=1 (visit/sum)."""

from __future__ import annotations

import numpy as np

_VISIT_SUM_EPS = 0.0


def visits_to_policy(visits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Convert a visit vector into a probability vector.

    ``temperature == 1`` is visit/sum (the distillation training target).
    ``temperature == 0`` is one-hot on the unique argmax (ties split uniformly).
    Other temperatures apply ``N^{1/τ}`` then normalize — play only, never labels.
    """
    counts = np.asarray(visits, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError(f"visit vector must be 1-D, got shape {counts.shape}")
    if counts.size == 0:
        # Empty legal action list → empty π (NeuralMCTS.evaluate_state does the same).
        return np.asarray([], dtype=np.float64)
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("visit counts must be finite and non-negative")

    total = float(counts.sum())
    if total <= _VISIT_SUM_EPS:
        raise ValueError("visit counts must sum to > 0 (refuse zero-simulation searches)")

    if temperature == 0:
        max_visits = counts.max()
        winners = counts == max_visits
        probs = np.zeros_like(counts)
        probs[winners] = 1.0 / float(winners.sum())
        return probs

    scaled = counts if temperature == 1.0 else np.power(counts, 1.0 / temperature)
    return scaled / scaled.sum()


def play_temperature(*, move_count: int, threshold: int, tau_init: float, tau_final: float) -> float:
    """Play-time τ. Training π never reads this value."""
    return tau_init if move_count < threshold else tau_final
