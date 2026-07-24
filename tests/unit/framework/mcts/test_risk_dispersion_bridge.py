"""Composition test: coarse-dynamics dispersion (#95) flows to the risk scorer (#3) via metadata.

Torch-free: uses the numpy ``mixture_variance_trace`` reference to produce a real dispersion value,
attaches it to candidate metadata, and shows the RiskAverseSubgoalScorer penalizes the
higher-dispersion candidate. This exercises the intended #95 -> #3 bridge without torch.
"""

from __future__ import annotations

import numpy as np

from src.framework.mcts.risk_scoring import MetadataDispersionSource, RiskAverseSubgoalScorer
from src.framework.mcts.scoring import CandidateRecord
from src.models.coarse_dynamics import mixture_variance_trace


def _single_component_dispersion(variances: list[float]) -> float:
    """Dispersion of a 1-component diagonal-Gaussian mixture = trace of its variances."""
    weights = np.array([[1.0]])
    means = np.array([[[0.0] * len(variances)]])
    var = np.array([[variances]])
    return float(mixture_variance_trace(weights, means, var)[0])


def test_mdn_dispersion_penalizes_via_metadata():
    dispersion_low = _single_component_dispersion([0.1, 0.1])
    dispersion_high = _single_component_dispersion([2.0, 3.0])
    assert dispersion_high > dispersion_low >= 0.0

    low = CandidateRecord("low", value=1.0, visits=1, metadata={"dispersion": dispersion_low})
    high = CandidateRecord("high", value=1.0, visits=9, metadata={"dispersion": dispersion_high})
    scorer = RiskAverseSubgoalScorer(lambda_weight=0.5, dispersion_source=MetadataDispersionSource())

    # Equal value -> the risk scorer picks the lower-dispersion candidate.
    assert scorer.select_best([low, high], engine_choice="high") == "low"
