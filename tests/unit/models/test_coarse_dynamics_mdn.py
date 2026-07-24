"""Torch MDN tests (strategos_coarse_dynamics_mdn AC-2). Skipped when torch is absent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.coarse_dynamics import (  # noqa: E402  (after importorskip)
    CoarseDynamicsMDN,
    create_coarse_dynamics_mdn,
    mixture_variance_trace,
)


def _mdn(input_dim=8, num_components=5, output_dim=None):
    torch.manual_seed(0)
    return CoarseDynamicsMDN(input_dim=input_dim, num_components=num_components, output_dim=output_dim)


def test_forward_shapes():  # AC-2
    mdn = _mdn(input_dim=8, num_components=4, output_dim=3)
    params = mdn.forward(torch.randn(6, 8))
    assert params.logits.shape == (6, 4)
    assert params.means.shape == (6, 4, 3)
    assert params.log_vars.shape == (6, 4, 3)


def test_dispersion_shape_and_non_negative():  # AC-2
    mdn = _mdn()
    dispersion = mdn.dispersion(torch.randn(6, 8))
    assert dispersion.shape == (6, 1)
    assert bool(torch.all(dispersion >= 0))


def test_dispersion_matches_numpy_reference():  # AC-2
    mdn = _mdn(input_dim=8, num_components=4, output_dim=3)
    x = torch.randn(5, 8)
    params = mdn.forward(x)
    weights = torch.softmax(params.logits, dim=-1).detach().numpy()
    means = params.means.detach().numpy()
    variances = torch.exp(params.log_vars).detach().numpy()
    reference = mixture_variance_trace(weights, means, variances)
    got = mdn.dispersion(x).detach().numpy().squeeze(-1)
    np.testing.assert_allclose(got, reference, atol=1e-5)


def test_configurable_component_count():  # AC-2
    assert _mdn(num_components=3).forward(torch.randn(2, 8)).logits.shape == (2, 3)
    assert _mdn(num_components=7).forward(torch.randn(2, 8)).logits.shape == (2, 7)


def test_out_of_range_component_count_raises():  # AC-2
    with pytest.raises(ValueError, match="num_components"):
        CoarseDynamicsMDN(input_dim=8, num_components=0)


def test_factory_builds_module():
    mdn = create_coarse_dynamics_mdn(input_dim=8, num_components=5)
    assert isinstance(mdn, CoarseDynamicsMDN)
    assert isinstance(mdn, torch.nn.Module)
