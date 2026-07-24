"""Tests for CoarseTransitionAggregator (strategos_coarse_dynamics_mdn AC-1). Torch-free."""

from __future__ import annotations

import numpy as np
import pytest

from src.config.constants import MIN_COARSE_WINDOW
from src.models.coarse_dynamics import CoarseTransitionAggregator


class TestOutputDim:
    def test_output_dim_is_four_times_state_dim(self):  # AC-1
        assert CoarseTransitionAggregator.output_dim(3) == 12
        assert CoarseTransitionAggregator.output_dim(1) == 4


class TestAggregate:
    def test_shape_is_four_times_state_dim(self):  # AC-1
        agg = CoarseTransitionAggregator(window=3)
        vec = agg.aggregate([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert vec.shape == (8,)
        assert vec.dtype == np.float32

    def test_values_are_first_last_mean_delta(self):  # AC-1
        agg = CoarseTransitionAggregator(window=3)
        vec = agg.aggregate([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        # first=[0,0], last=[3,4], mean=[4/3,2], delta=last-first=[3,4]
        np.testing.assert_allclose(vec, [0.0, 0.0, 3.0, 4.0, 4.0 / 3.0, 2.0, 3.0, 4.0], rtol=1e-5)

    def test_deterministic(self):  # AC-1
        agg = CoarseTransitionAggregator(window=4)
        states = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        assert np.array_equal(agg.aggregate(states), agg.aggregate(states))

    def test_shape_independent_of_window(self):  # AC-1
        states = [[float(i), float(i)] for i in range(10)]
        a2 = CoarseTransitionAggregator(window=2).aggregate(states)
        a8 = CoarseTransitionAggregator(window=8).aggregate(states)
        assert a2.shape == a8.shape == (8,)  # 4 * state_dim(=2), regardless of window
        assert not np.array_equal(a2, a8)  # different windows summarize different states

    def test_window_selects_last_states(self):  # AC-1
        states = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        # window=2 -> first=[10,10]; window=3 -> first=[0,0]
        a2 = CoarseTransitionAggregator(window=2).aggregate(states)
        a3 = CoarseTransitionAggregator(window=3).aggregate(states)
        np.testing.assert_allclose(a2[:2], [10.0, 10.0])
        np.testing.assert_allclose(a3[:2], [0.0, 0.0])

    def test_fewer_states_than_window(self):  # AC-1
        vec = CoarseTransitionAggregator(window=100).aggregate([[1.0, 2.0]])
        # single state: first=last=mean=[1,2], delta=[0,0]
        np.testing.assert_allclose(vec, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0])


class TestValidation:
    def test_window_below_minimum_raises(self):
        with pytest.raises(ValueError, match="window must be"):
            CoarseTransitionAggregator(window=MIN_COARSE_WINDOW - 1)

    def test_empty_states_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            CoarseTransitionAggregator().aggregate([])

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CoarseTransitionAggregator().aggregate([1.0, 2.0])  # 1-D, not [T, state_dim]
