"""Tests for the torch-free numpy dispersion reference (strategos_coarse_dynamics_mdn AC-2).

mixture_variance_trace is the reference that CoarseDynamicsMDN.dispersion mirrors; verifying it
without torch pins the law-of-total-variance math and the non-negativity invariant locally.
"""

from __future__ import annotations

import numpy as np

from src.models.coarse_dynamics import mixture_variance_trace


def test_single_component_equals_variance_sum():  # AC-2
    weights = np.array([[1.0]])
    means = np.array([[[0.0, 0.0]]])
    variances = np.array([[[2.0, 3.0]]])
    # K=1: no between-variance -> trace = 2 + 3
    np.testing.assert_allclose(mixture_variance_trace(weights, means, variances), [5.0])


def test_between_component_variance():  # AC-2
    weights = np.array([[0.5, 0.5]])
    means = np.array([[[0.0], [2.0]]])
    variances = np.array([[[0.0], [0.0]]])
    # means 0 and 2, equal weight -> mean=1, between = 0.5*1 + 0.5*1 = 1.0
    np.testing.assert_allclose(mixture_variance_trace(weights, means, variances), [1.0])


def test_identical_components_have_no_between_variance():  # AC-2
    weights = np.array([[0.3, 0.7]])
    means = np.array([[[1.0, 2.0], [1.0, 2.0]]])
    variances = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    # identical means -> between=0; within = 0.5 + 0.5 = 1.0
    np.testing.assert_allclose(mixture_variance_trace(weights, means, variances), [1.0])


def test_non_negative_and_shape_on_random_batch():  # AC-2
    rng = np.random.default_rng(0)
    batch, components, dims = 4, 3, 5
    logits = rng.normal(size=(batch, components))
    weights = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    means = rng.normal(size=(batch, components, dims))
    variances = np.abs(rng.normal(size=(batch, components, dims)))
    dispersion = mixture_variance_trace(weights, means, variances)
    assert dispersion.shape == (batch,)
    assert np.all(dispersion >= 0.0)
