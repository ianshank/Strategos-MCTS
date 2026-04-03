"""
Tests for RNN meta-controller module.

Tests RNNMetaControllerModel forward pass and RNNMetaController
initialization, prediction, model save/load, and hidden state.
"""

import pytest
import torch

from src.agents.meta_controller.base import MetaControllerFeatures
from src.agents.meta_controller.rnn_controller import (
    RNNMetaController,
    RNNMetaControllerModel,
)


def _make_features(**overrides):
    defaults = {
        "hrm_confidence": 0.5,
        "trm_confidence": 0.3,
        "mcts_value": 0.2,
        "consensus_score": 0.7,
        "last_agent": "none",
        "iteration": 0,
        "query_length": 50,
        "has_rag_context": False,
        "rag_relevance_score": 0.0,
        "is_technical_query": False,
    }
    defaults.update(overrides)
    return MetaControllerFeatures(**defaults)


@pytest.mark.unit
class TestRNNMetaControllerModel:
    """Tests for the GRU-based model."""

    def test_init(self):
        model = RNNMetaControllerModel(input_dim=10, hidden_dim=32, num_agents=3)
        assert model.hidden_dim == 32

    def test_forward_2d(self):
        model = RNNMetaControllerModel(input_dim=10, hidden_dim=32)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_forward_3d(self):
        model = RNNMetaControllerModel(input_dim=10, hidden_dim=32)
        x = torch.randn(4, 5, 10)  # batch=4, seq_len=5
        out = model(x)
        assert out.shape == (4, 3)

    def test_multi_layer(self):
        model = RNNMetaControllerModel(input_dim=10, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_gradient_flow(self):
        model = RNNMetaControllerModel(input_dim=10, hidden_dim=16)
        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


@pytest.mark.unit
class TestRNNMetaController:
    """Tests for RNNMetaController."""

    def test_init_defaults(self):
        ctrl = RNNMetaController(device="cpu")
        assert ctrl.hidden_dim == 64
        assert ctrl.num_layers == 1
        assert ctrl.dropout == 0.1
        assert ctrl.device == torch.device("cpu")

    def test_init_custom(self):
        ctrl = RNNMetaController(hidden_dim=128, num_layers=2, dropout=0.2, device="cpu")
        assert ctrl.hidden_dim == 128
        assert ctrl.num_layers == 2

    def test_predict(self):
        ctrl = RNNMetaController(device="cpu", hidden_dim=16)
        features = _make_features()
        pred = ctrl.predict(features)
        assert pred.agent in ["hrm", "trm", "mcts"]
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.probabilities) == 3
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5

    def test_predict_deterministic_with_seed(self):
        ctrl1 = RNNMetaController(device="cpu", hidden_dim=16, seed=42)
        ctrl2 = RNNMetaController(device="cpu", hidden_dim=16, seed=42)
        features = _make_features()
        pred1 = ctrl1.predict(features)
        pred2 = ctrl2.predict(features)
        assert pred1.agent == pred2.agent
        assert pred1.confidence == pytest.approx(pred2.confidence, abs=1e-5)

    def test_save_and_load_model(self, tmp_path):
        ctrl = RNNMetaController(device="cpu", hidden_dim=16)
        path = str(tmp_path / "model.pt")
        ctrl.save_model(path)

        ctrl2 = RNNMetaController(device="cpu", hidden_dim=16)
        ctrl2.load_model(path)

        features = _make_features()
        pred1 = ctrl.predict(features)
        pred2 = ctrl2.predict(features)
        assert pred1.agent == pred2.agent

    def test_reset_hidden_state(self):
        ctrl = RNNMetaController(device="cpu")
        ctrl.hidden_state = torch.zeros(1, 1, 64)
        ctrl.reset_hidden_state()
        assert ctrl.hidden_state is None

    def test_agent_names(self):
        assert RNNMetaController.AGENT_NAMES == ["hrm", "trm", "mcts"]

    def test_predict_various_features(self):
        ctrl = RNNMetaController(device="cpu", hidden_dim=16)
        # Different feature patterns should produce valid predictions
        for hrm_conf in [0.1, 0.5, 0.9]:
            features = _make_features(hrm_confidence=hrm_conf)
            pred = ctrl.predict(features)
            assert pred.agent in ["hrm", "trm", "mcts"]
