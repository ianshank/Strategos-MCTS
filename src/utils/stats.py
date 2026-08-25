"""Shared, scipy-free statistics helpers for evaluation and benchmarking.

Extracted from :class:`~src.training.evaluation_service.EvaluationService` so that both
the training-side evaluator and the M5 policy-lift benchmark
(:mod:`src.benchmark.policy_comparison`) gate on the same interval math.

All helpers avoid a scipy dependency: z-scores come from a small lookup table and the
mean/difference intervals use the normal approximation (adequate for the game counts
these gates require; n >= 30 for mean-reward, n >= 100 for win-rate).
"""

from __future__ import annotations

from collections.abc import Sequence
import math

# Common two-sided z-scores (avoid scipy dependency).
_Z_SCORES: dict[float, float] = {
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}
_DEFAULT_Z = 1.96


def z_score(confidence: float = 0.95) -> float:
    """Two-sided z-score for a confidence level (0.90/0.95/0.99; defaults to 1.96)."""
    return _Z_SCORES.get(confidence, _DEFAULT_Z)


def wilson_score_interval(
    successes: float,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    More accurate than the normal approximation for small samples and extreme
    proportions. ``successes`` accepts floats so draw-adjusted counts
    (``wins + 0.5 * draws``) can be passed directly.

    Args:
        successes: Number of successes (may be fractional for draw-adjusted counts).
        total: Total trials.
        confidence: Confidence level (default 0.95).

    Returns:
        ``(lower_bound, upper_bound)``, clamped to ``[0, 1]``; ``(0.0, 0.0)`` when
        ``total`` is 0.
    """
    if total == 0:
        return 0.0, 0.0

    z = z_score(confidence)
    p_hat = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


def mean_confidence_interval(
    samples: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Normal-approximation confidence interval for a sample mean.

    Returns ``(mean, lower, upper)`` using ``mean ± z * s / sqrt(n)`` with the
    unbiased sample standard deviation. Degenerate inputs collapse the interval:
    ``(0.0, 0.0, 0.0)`` for empty input, ``(m, m, m)`` for a single sample.
    """
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(samples) / n
    if n < 2:
        return mean, mean, mean
    variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
    margin = z_score(confidence) * math.sqrt(variance / n)
    return mean, mean - margin, mean + margin


def difference_confidence_interval(
    baseline: Sequence[float],
    treatment: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Confidence interval for ``mean(treatment) - mean(baseline)`` (unpaired).

    Uses the Welch-style unpooled standard error
    ``sqrt(s_b²/n_b + s_t²/n_t)`` with a normal-approximation critical value.
    Groups with fewer than two samples contribute zero variance (their spread
    cannot be estimated), so degenerate inputs collapse toward a point interval.

    Returns:
        ``(difference, lower, upper)``; ``(0.0, 0.0, 0.0)`` when either group is empty.
    """
    if not baseline or not treatment:
        return 0.0, 0.0, 0.0

    def _mean_and_sem_sq(samples: Sequence[float]) -> tuple[float, float]:
        n = len(samples)
        mean = sum(samples) / n
        if n < 2:
            return mean, 0.0
        variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
        return mean, variance / n

    baseline_mean, baseline_sem_sq = _mean_and_sem_sq(baseline)
    treatment_mean, treatment_sem_sq = _mean_and_sem_sq(treatment)

    difference = treatment_mean - baseline_mean
    margin = z_score(confidence) * math.sqrt(baseline_sem_sq + treatment_sem_sq)
    return difference, difference - margin, difference + margin


__all__ = [
    "z_score",
    "wilson_score_interval",
    "mean_confidence_interval",
    "difference_confidence_interval",
]
