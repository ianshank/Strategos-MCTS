"""Tests for the shared scipy-free statistics helpers (src.utils.stats).

Golden values, edge cases, and a regression check that
EvaluationService's private helpers still delegate to the same math.
"""

from __future__ import annotations

import math

import pytest

from src.utils.stats import (
    difference_confidence_interval,
    mean_confidence_interval,
    wilson_score_interval,
    z_score,
)

pytestmark = [pytest.mark.unit]


class TestZScore:
    def test_known_confidence_levels(self):
        assert z_score(0.90) == pytest.approx(1.645)
        assert z_score(0.95) == pytest.approx(1.96)
        assert z_score(0.99) == pytest.approx(2.576)

    def test_unknown_confidence_falls_back_to_95(self):
        assert z_score(0.83) == pytest.approx(1.96)
        assert z_score() == pytest.approx(1.96)


class TestWilsonScoreInterval:
    def test_golden_value_65_of_100(self):
        # Wilson 95% interval for 65/100.
        lower, upper = wilson_score_interval(65, 100)
        assert lower == pytest.approx(0.5525, abs=1e-3)
        assert upper == pytest.approx(0.7364, abs=1e-3)

    def test_zero_total_returns_degenerate(self):
        assert wilson_score_interval(0, 0) == (0.0, 0.0)

    def test_zero_successes_lower_clamped(self):
        lower, upper = wilson_score_interval(0, 20)
        assert lower == 0.0
        assert 0.0 < upper < 0.25

    def test_all_successes_upper_clamped(self):
        lower, upper = wilson_score_interval(20, 20)
        assert upper == pytest.approx(1.0)
        assert 0.75 < lower < 1.0

    def test_fractional_successes_for_draw_adjusted_counts(self):
        # wins + 0.5*draws must be accepted directly.
        lower, upper = wilson_score_interval(10.5, 20)
        strict_lower, strict_upper = wilson_score_interval(10, 20)
        assert lower > strict_lower
        assert upper > strict_upper

    def test_higher_confidence_widens_interval(self):
        lo95, hi95 = wilson_score_interval(65, 100, confidence=0.95)
        lo99, hi99 = wilson_score_interval(65, 100, confidence=0.99)
        assert lo99 < lo95
        assert hi99 > hi95


class TestMeanConfidenceInterval:
    def test_empty_input(self):
        assert mean_confidence_interval([]) == (0.0, 0.0, 0.0)

    def test_single_sample_collapses(self):
        assert mean_confidence_interval([0.7]) == (0.7, 0.7, 0.7)

    def test_symmetric_around_mean(self):
        samples = [0.4, 0.5, 0.6, 0.5]
        mean, lower, upper = mean_confidence_interval(samples)
        assert mean == pytest.approx(0.5)
        assert upper - mean == pytest.approx(mean - lower)
        assert lower < mean < upper

    def test_golden_value(self):
        # mean=0.5, sample std ~0.08165, n=4 -> margin = 1.96 * 0.08165/2 ~ 0.08
        samples = [0.4, 0.5, 0.6, 0.5]
        std = math.sqrt(sum((x - 0.5) ** 2 for x in samples) / 3)
        expected_margin = 1.96 * std / 2
        mean, lower, upper = mean_confidence_interval(samples)
        assert upper - mean == pytest.approx(expected_margin)

    def test_zero_variance_collapses(self):
        mean, lower, upper = mean_confidence_interval([0.5, 0.5, 0.5])
        assert (mean, lower, upper) == (0.5, 0.5, 0.5)


class TestDifferenceConfidenceInterval:
    def test_empty_group_returns_degenerate(self):
        assert difference_confidence_interval([], [0.5]) == (0.0, 0.0, 0.0)
        assert difference_confidence_interval([0.5], []) == (0.0, 0.0, 0.0)

    def test_sign_of_difference(self):
        diff, lower, upper = difference_confidence_interval([0.5, 0.5], [0.7, 0.7])
        assert diff == pytest.approx(0.2)
        # Zero-variance groups collapse to a point interval.
        assert lower == pytest.approx(0.2)
        assert upper == pytest.approx(0.2)

    def test_symmetry_under_swap(self):
        a = [0.4, 0.5, 0.6]
        b = [0.6, 0.7, 0.8]
        diff_ab, lo_ab, hi_ab = difference_confidence_interval(a, b)
        diff_ba, lo_ba, hi_ba = difference_confidence_interval(b, a)
        assert diff_ab == pytest.approx(-diff_ba)
        assert lo_ab == pytest.approx(-hi_ba)
        assert hi_ab == pytest.approx(-lo_ba)

    def test_interval_contains_true_difference_for_clean_data(self):
        baseline = [0.5, 0.52, 0.48, 0.51, 0.49]
        treatment = [0.6, 0.62, 0.58, 0.61, 0.59]
        diff, lower, upper = difference_confidence_interval(baseline, treatment)
        assert diff == pytest.approx(0.1, abs=1e-9)
        assert lower < 0.1 < upper

    def test_welch_unpooled_standard_error(self):
        baseline = [0.0, 1.0]  # var 0.5, n 2
        treatment = [0.0, 2.0]  # var 2.0, n 2
        diff, lower, upper = difference_confidence_interval(baseline, treatment)
        expected_margin = 1.96 * math.sqrt(0.5 / 2 + 2.0 / 2)
        assert diff == pytest.approx(0.5)
        assert upper - diff == pytest.approx(expected_margin)


class TestEvaluationServiceDelegation:
    """EvaluationService's private helpers must return identical values post-extraction."""

    @pytest.fixture()
    def service(self):
        pytest.importorskip("torch")
        from src.training.evaluation_service import EvaluationService

        return EvaluationService.__new__(EvaluationService)  # no init needed for pure helpers

    def test_wilson_delegation(self, service):
        assert service._wilson_score_interval(65, 100) == wilson_score_interval(65, 100)
        assert service._wilson_score_interval(0, 0) == wilson_score_interval(0, 0)
        assert service._wilson_score_interval(3, 7, confidence=0.99) == wilson_score_interval(3, 7, confidence=0.99)

    def test_z_score_delegation(self, service):
        for confidence in (0.90, 0.95, 0.99, 0.5):
            assert service._get_z_score(confidence) == z_score(confidence)
