"""Unit tests for src/models/value_network.py - extended coverage."""

from __future__ import annotations

import pytest
import torch

from src.models.value_network import (
    EnsembleValueNetwork,
    TemporalDifferenceLoss,
    ValueLoss,
    ValueNetwork,
    ValueOutput,
    create_value_network,
)


@pytest.mark.unit
class TestValueOutput:
    def test_basic(self):
        out = ValueOutput(value=torch.tensor([[0.5]]))
        assert out.features is None
        assert out.uncertainty is None

    def test_with_features_and_uncertainty(self):
        out = ValueOutput(
            value=torch.tensor([[0.5]]),
            features=torch.randn(1, 128),
            uncertainty=torch.tensor([[0.1]]),
        )
        assert out.features is not None
        assert out.uncertainty is not None


@pytest.mark.unit
class TestValueNetwork:
    def test_forward_tanh(self):
        net = ValueNetwork(state_dim=64, hidden_dims=[32, 16], output_activation="tanh")
        x = torch.randn(4, 64)
        out = net(x)
        assert out.value.shape == (4, 1)
        assert (out.value >= -1).all() and (out.value <= 1).all()

    def test_forward_sigmoid(self):
        net = ValueNetwork(state_dim=32, hidden_dims=[16], output_activation="sigmoid")
        x = torch.randn(2, 32)
        out = net(x)
        assert (out.value >= 0).all() and (out.value <= 1).all()

    def test_forward_no_activation(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8], output_activation=None)
        x = torch.randn(2, 16)
        out = net(x)
        assert out.value.shape == (2, 1)

    def test_forward_unknown_activation(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8], output_activation="unknown")
        x = torch.randn(2, 16)
        with pytest.raises(ValueError, match="Unknown output_activation"):
            net(x)

    def test_forward_return_features(self):
        net = ValueNetwork(state_dim=32, hidden_dims=[16])
        x = torch.randn(2, 32)
        out = net(x, return_features=True)
        assert out.features is not None
        assert out.features.shape == (2, 16)

    def test_forward_no_features_by_default(self):
        net = ValueNetwork(state_dim=32, hidden_dims=[16])
        x = torch.randn(2, 32)
        out = net(x)
        assert out.features is None

    def test_uncertainty_head(self):
        net = ValueNetwork(state_dim=32, hidden_dims=[16], estimate_uncertainty=True)
        x = torch.randn(2, 32)
        out = net(x)
        assert out.uncertainty is not None
        assert out.uncertainty.shape == (2, 1)
        assert (out.uncertainty >= 0).all()  # Softplus ensures positive

    def test_no_uncertainty_by_default(self):
        net = ValueNetwork(state_dim=32, hidden_dims=[16])
        x = torch.randn(2, 32)
        out = net(x)
        assert out.uncertainty is None

    def test_no_batch_norm(self):
        net = ValueNetwork(state_dim=32, hidden_dims=[16], use_batch_norm=False)
        x = torch.randn(2, 32)
        out = net(x)
        assert out.value.shape == (2, 1)

    def test_evaluate_single(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8])
        x = torch.randn(16)
        val = net.evaluate(x)
        assert isinstance(val, float)
        assert -1 <= val <= 1

    def test_evaluate_batch_dim(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8])
        x = torch.randn(1, 16)
        val = net.evaluate(x)
        assert isinstance(val, float)

    def test_evaluate_batch(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8])
        x = torch.randn(4, 16)
        vals = net.evaluate_batch(x)
        assert vals.shape == (4,)

    def test_get_confidence_tanh(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8], output_activation="tanh")
        x = torch.randn(16)
        conf = net.get_confidence(x)
        assert 0 <= conf <= 1

    def test_get_confidence_sigmoid(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8], output_activation="sigmoid")
        x = torch.randn(16)
        conf = net.get_confidence(x)
        assert 0 <= conf <= 1

    def test_get_confidence_no_activation_with_uncertainty(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8], output_activation=None, estimate_uncertainty=True)
        x = torch.randn(16)
        conf = net.get_confidence(x)
        assert 0 <= conf <= 1

    def test_get_confidence_no_activation_no_uncertainty(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8], output_activation=None)
        x = torch.randn(16)
        conf = net.get_confidence(x)
        assert conf == 0.5

    def test_parameter_count(self):
        net = ValueNetwork(state_dim=16, hidden_dims=[8, 4])
        count = net.get_parameter_count()
        assert count > 0

    def test_default_hidden_dims(self):
        net = ValueNetwork(state_dim=64)
        assert net.hidden_dims == [512, 256, 128]


@pytest.mark.unit
class TestValueLoss:
    def test_mse(self):
        loss_fn = ValueLoss(loss_type="mse")
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        loss, d = loss_fn(pred, target)
        assert loss.item() > 0
        assert "mse" in d
        assert "total" in d

    def test_huber(self):
        loss_fn = ValueLoss(loss_type="huber")
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        loss, d = loss_fn(pred, target)
        assert "huber" in d

    def test_quantile(self):
        loss_fn = ValueLoss(loss_type="quantile")
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        loss, d = loss_fn(pred, target)
        assert "quantile" in d

    def test_unknown_loss_type(self):
        loss_fn = ValueLoss(loss_type="unknown")
        with pytest.raises(ValueError, match="Unknown loss_type"):
            loss_fn(torch.randn(4, 1), torch.randn(4, 1))

    def test_target_1d(self):
        loss_fn = ValueLoss()
        pred = torch.randn(4, 1)
        target = torch.randn(4)  # 1D
        loss, d = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_with_uncertainty(self):
        loss_fn = ValueLoss()
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        uncertainty = torch.abs(torch.randn(4, 1)) + 0.1
        loss, d = loss_fn(pred, target, uncertainty=uncertainty)
        assert "uncertainty" in d

    def test_with_l2_reg(self):
        loss_fn = ValueLoss(l2_weight=0.01)
        model = torch.nn.Linear(10, 1)
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        loss, d = loss_fn(pred, target, model=model)
        assert "l2" in d


@pytest.mark.unit
class TestTemporalDifferenceLoss:
    def test_basic(self):
        loss_fn = TemporalDifferenceLoss()
        values = torch.randn(4)
        rewards = torch.randn(4)
        next_values = torch.randn(4)
        dones = torch.zeros(4)
        loss, d = loss_fn(values, rewards, next_values, dones)
        assert "td_loss" in d
        assert "mean_td_error" in d

    def test_with_dones(self):
        loss_fn = TemporalDifferenceLoss(gamma=0.99)
        values = torch.randn(4)
        rewards = torch.ones(4)
        next_values = torch.ones(4)
        dones = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
        loss, d = loss_fn(values, rewards, next_values, dones)
        assert loss.item() >= 0

    def test_huber_loss_type(self):
        loss_fn = TemporalDifferenceLoss(loss_type="huber")
        values = torch.randn(4)
        rewards = torch.randn(4)
        next_values = torch.randn(4)
        dones = torch.zeros(4)
        loss, d = loss_fn(values, rewards, next_values, dones)
        assert loss.item() >= 0

    def test_unknown_loss_type(self):
        loss_fn = TemporalDifferenceLoss(loss_type="unknown")
        with pytest.raises(ValueError):
            loss_fn(torch.randn(4), torch.randn(4), torch.randn(4), torch.zeros(4))


@pytest.mark.unit
class TestCreateValueNetwork:
    def test_default(self):
        net = create_value_network(64)
        assert isinstance(net, ValueNetwork)
        assert net.state_dim == 64

    def test_with_config(self):
        config = {"hidden_dims": [32, 16], "dropout": 0.2, "output_activation": "sigmoid"}
        net = create_value_network(32, config=config)
        assert net.hidden_dims == [32, 16]
        assert net.output_activation == "sigmoid"


@pytest.mark.unit
class TestEnsembleValueNetwork:
    def test_forward(self):
        net = EnsembleValueNetwork(state_dim=16, num_networks=3, network_config={"hidden_dims": [8]})
        x = torch.randn(2, 16)
        out = net(x)
        assert out.value.shape == (2, 1)
        assert out.uncertainty is not None
        assert out.uncertainty.shape == (2, 1)

    def test_evaluate(self):
        net = EnsembleValueNetwork(state_dim=16, num_networks=3, network_config={"hidden_dims": [8]})
        x = torch.randn(16)
        val, unc = net.evaluate(x)
        assert isinstance(val, float)
        assert isinstance(unc, float)
        assert unc >= 0

    def test_num_networks(self):
        net = EnsembleValueNetwork(state_dim=16, num_networks=5, network_config={"hidden_dims": [8]})
        assert len(net.networks) == 5
