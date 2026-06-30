"""Tests for the meta-controller learning loop (Phase 5.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.agents.meta_controller.base import MetaControllerFeatures
from src.training.meta_controller_data_collector import (
    AGENT_LABELS,
    FEATURE_DIM,
    MetaControllerDataCollector,
    agent_to_label,
    features_to_vector,
    train_and_validate,
)

pytestmark = [pytest.mark.unit]


def _features(last_agent="none", hrm=0.5, technical=False) -> MetaControllerFeatures:
    return MetaControllerFeatures(
        hrm_confidence=hrm,
        trm_confidence=0.4,
        mcts_value=0.3,
        consensus_score=0.6,
        last_agent=last_agent,
        iteration=1,
        query_length=120,
        has_rag_context=True,
        rag_relevance_score=0.7,
        is_technical_query=technical,
    )


def test_features_to_vector_shape_and_determinism():
    vec = features_to_vector(_features(last_agent="hrm"))
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    # last_agent one-hot set for "hrm"
    assert np.array_equal(features_to_vector(_features(last_agent="hrm")), vec)


def test_agent_to_label_roundtrip():
    for i, name in enumerate(AGENT_LABELS):
        assert agent_to_label(name) == i
    with pytest.raises(ValueError):
        agent_to_label("unknown_agent")


def test_collector_records_and_builds_dataset():
    c = MetaControllerDataCollector()
    c.record_features(_features(technical=True), "hrm", outcome=0.9)
    c.record(features_to_vector(_features()), "trm", outcome=0.4)
    assert len(c) == 2
    x, y = c.to_dataset()
    assert x.shape == (2, FEATURE_DIM)
    assert y.tolist() == [agent_to_label("hrm"), agent_to_label("trm")]


def test_record_rejects_wrong_shape():
    c = MetaControllerDataCollector()
    with pytest.raises(ValueError):
        c.record(np.zeros(FEATURE_DIM - 1, dtype=np.float32), "hrm", 0.5)


def test_train_and_validate_learns_separable_routing():
    """On a linearly-separable synthetic routing dataset, the model beats the baseline."""
    c = MetaControllerDataCollector()
    # Build a separable signal: technical+hrm-confident -> hrm; else -> trm.
    for _ in range(30):
        c.record_features(_features(last_agent="hrm", hrm=0.95, technical=True), "hrm", outcome=0.9)
        c.record_features(_features(last_agent="trm", hrm=0.05, technical=False), "trm", outcome=0.8)

    model = nn.Linear(FEATURE_DIM, len(AGENT_LABELS))
    report = train_and_validate(model, c, epochs=50, learning_rate=0.05, seed=1)

    assert report.train_examples + report.val_examples == len(c)
    assert 0.0 <= report.val_accuracy <= 1.0
    # The separable signal should be learnable above the majority baseline.
    assert report.val_accuracy >= report.baseline_accuracy
    assert report.val_accuracy >= 0.75


def test_train_and_validate_is_reproducible():
    c = MetaControllerDataCollector()
    for _ in range(20):
        c.record_features(_features(last_agent="hrm", hrm=0.9, technical=True), "hrm", 0.9)
        c.record_features(_features(last_agent="trm", hrm=0.1, technical=False), "trm", 0.8)

    # Identical model init (seed before construction) + identical training seed -> identical run.
    torch.manual_seed(0)
    m1 = nn.Linear(FEATURE_DIM, len(AGENT_LABELS))
    torch.manual_seed(0)
    m2 = nn.Linear(FEATURE_DIM, len(AGENT_LABELS))

    r1 = train_and_validate(m1, c, epochs=10, seed=7)
    r2 = train_and_validate(m2, c, epochs=10, seed=7)
    assert r1.final_train_loss == pytest.approx(r2.final_train_loss)
    assert r1.val_accuracy == pytest.approx(r2.val_accuracy)
