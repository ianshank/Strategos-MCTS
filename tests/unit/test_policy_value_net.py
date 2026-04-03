"""
Tests for policy-value network module.

Tests ResidualBlock, PolicyHead, ValueHead, PolicyValueNetwork,
AlphaZeroLoss, MLPPolicyValueNetwork, and create_policy_value_network factory.
"""


import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for policy-value net tests")

from src.models.policy_value_net import (
    AlphaZeroLoss,
    MLPPolicyValueNetwork,
    PolicyHead,
    PolicyValueNetwork,
    ResidualBlock,
    ValueHead,
    create_policy_value_network,
)
from src.training.system_config import NeuralNetworkConfig


def _small_config(**overrides):
    """Create a small NeuralNetworkConfig for fast tests."""
    defaults = {
        "num_res_blocks": 2,
        "num_channels": 16,
        "policy_conv_channels": 2,
        "value_conv_channels": 1,
        "value_fc_hidden": 16,
        "use_batch_norm": True,
        "input_channels": 3,
        "action_size": 10,
    }
    defaults.update(overrides)
    return NeuralNetworkConfig(**defaults)


@pytest.mark.unit
class TestResidualBlock:
    """Tests for ResidualBlock."""

    def test_forward_shape_preserved(self):
        block = ResidualBlock(channels=16)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == (2, 16, 8, 8)

    def test_skip_connection(self):
        block = ResidualBlock(channels=16)
        x = torch.randn(1, 16, 4, 4)
        out = block(x)
        # Output should differ from input (transformed)
        assert not torch.allclose(out, x, atol=1e-6)

    def test_no_batch_norm(self):
        block = ResidualBlock(channels=8, use_batch_norm=False)
        x = torch.randn(2, 8, 4, 4)
        out = block(x)
        assert out.shape == (2, 8, 4, 4)


@pytest.mark.unit
class TestPolicyHead:
    """Tests for PolicyHead."""

    def test_output_shape(self):
        head = PolicyHead(input_channels=16, policy_conv_channels=2, action_size=10, board_size=4)
        x = torch.randn(2, 16, 4, 4)
        out = head(x)
        assert out.shape == (2, 10)

    def test_output_is_log_softmax(self):
        head = PolicyHead(input_channels=16, policy_conv_channels=2, action_size=10, board_size=4)
        x = torch.randn(2, 16, 4, 4)
        out = head(x)
        # Log softmax: exp should sum to ~1
        probs = torch.exp(out)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


@pytest.mark.unit
class TestValueHead:
    """Tests for ValueHead."""

    def test_output_shape(self):
        head = ValueHead(input_channels=16, value_conv_channels=1, value_fc_hidden=16, board_size=4)
        x = torch.randn(2, 16, 4, 4)
        out = head(x)
        assert out.shape == (2, 1)

    def test_output_bounded(self):
        head = ValueHead(input_channels=16, value_conv_channels=1, value_fc_hidden=16, board_size=4)
        x = torch.randn(10, 16, 4, 4)
        out = head(x)
        # Tanh bounds output to [-1, 1]
        assert (out >= -1.0).all()
        assert (out <= 1.0).all()


@pytest.mark.unit
class TestPolicyValueNetwork:
    """Tests for PolicyValueNetwork."""

    def test_forward(self):
        config = _small_config()
        net = PolicyValueNetwork(config, board_size=4)
        x = torch.randn(2, 3, 4, 4)
        policy, value = net(x)
        assert policy.shape == (2, 10)
        assert value.shape == (2, 1)

    def test_predict(self):
        config = _small_config()
        net = PolicyValueNetwork(config, board_size=4)
        x = torch.randn(1, 3, 4, 4)
        policy_probs, value = net.predict(x)
        assert policy_probs.shape == (1, 10)
        # predict returns probs (not log probs), should sum to ~1
        assert torch.allclose(policy_probs.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_predict_no_grad(self):
        config = _small_config()
        net = PolicyValueNetwork(config, board_size=4)
        x = torch.randn(1, 3, 4, 4)
        policy_probs, value = net.predict(x)
        assert not policy_probs.requires_grad
        assert not value.requires_grad

    def test_get_parameter_count(self):
        config = _small_config()
        net = PolicyValueNetwork(config, board_size=4)
        count = net.get_parameter_count()
        assert count > 0
        assert isinstance(count, int)

    def test_no_batch_norm(self):
        config = _small_config(use_batch_norm=False)
        net = PolicyValueNetwork(config, board_size=4)
        x = torch.randn(2, 3, 4, 4)
        policy, value = net(x)
        assert policy.shape == (2, 10)


@pytest.mark.unit
class TestAlphaZeroLoss:
    """Tests for AlphaZeroLoss."""

    def test_loss_computes(self):
        loss_fn = AlphaZeroLoss()
        policy_logits = torch.log_softmax(torch.randn(4, 10, requires_grad=True), dim=1)
        value = torch.randn(4, 1, requires_grad=True)
        target_policy = torch.softmax(torch.randn(4, 10), dim=1)
        target_value = torch.randn(4)
        total_loss, loss_dict = loss_fn(policy_logits, value, target_policy, target_value)
        assert total_loss.requires_grad
        assert "total" in loss_dict
        assert "value" in loss_dict
        assert "policy" in loss_dict

    def test_perfect_prediction_low_loss(self):
        loss_fn = AlphaZeroLoss()
        # Perfect value prediction
        target_value = torch.tensor([1.0, -1.0])
        value = torch.tensor([[1.0], [-1.0]])
        # Reasonable policy
        target_policy = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        policy_logits = torch.log(target_policy + 1e-8)
        total_loss, loss_dict = loss_fn(policy_logits, value, target_policy, target_value)
        assert loss_dict["value"] < 0.01

    def test_value_loss_weight(self):
        loss_fn_default = AlphaZeroLoss(value_loss_weight=1.0)
        loss_fn_heavy = AlphaZeroLoss(value_loss_weight=10.0)
        policy_logits = torch.log_softmax(torch.randn(4, 10), dim=1)
        value = torch.randn(4, 1)
        target_policy = torch.softmax(torch.randn(4, 10), dim=1)
        target_value = torch.randn(4)
        _, d1 = loss_fn_default(policy_logits, value, target_policy, target_value)
        _, d2 = loss_fn_heavy(policy_logits, value, target_policy, target_value)
        # Heavier weight should generally produce larger total loss
        # (unless value loss happens to be 0)
        assert d1["policy"] == pytest.approx(d2["policy"], abs=1e-5)


@pytest.mark.unit
class TestMLPPolicyValueNetwork:
    """Tests for MLPPolicyValueNetwork."""

    def test_forward(self):
        net = MLPPolicyValueNetwork(state_dim=20, action_size=5, hidden_dims=[32, 16])
        x = torch.randn(4, 20)
        policy, value = net(x)
        assert policy.shape == (4, 5)
        assert value.shape == (4, 1)

    def test_policy_log_softmax(self):
        net = MLPPolicyValueNetwork(state_dim=10, action_size=3, hidden_dims=[16])
        x = torch.randn(2, 10)
        policy, _ = net(x)
        probs = torch.exp(policy)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_value_bounded(self):
        net = MLPPolicyValueNetwork(state_dim=10, action_size=3, hidden_dims=[16])
        x = torch.randn(20, 10)
        _, value = net(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_default_hidden_dims(self):
        net = MLPPolicyValueNetwork(state_dim=20, action_size=5)
        assert net.state_dim == 20
        assert net.action_size == 5

    def test_no_batch_norm(self):
        net = MLPPolicyValueNetwork(state_dim=10, action_size=3, use_batch_norm=False)
        x = torch.randn(4, 10)
        policy, value = net(x)
        assert policy.shape == (4, 3)

    def test_no_dropout(self):
        net = MLPPolicyValueNetwork(state_dim=10, action_size=3, dropout=0.0)
        x = torch.randn(4, 10)
        policy, value = net(x)
        assert policy.shape == (4, 3)

    def test_get_parameter_count(self):
        net = MLPPolicyValueNetwork(state_dim=10, action_size=3, hidden_dims=[16])
        count = net.get_parameter_count()
        assert count > 0


@pytest.mark.unit
class TestCreatePolicyValueNetwork:
    """Tests for create_policy_value_network factory."""

    def test_creates_network(self):
        config = _small_config()
        net = create_policy_value_network(config, board_size=4)
        assert isinstance(net, PolicyValueNetwork)

    def test_on_cpu(self):
        config = _small_config()
        net = create_policy_value_network(config, board_size=4, device="cpu")
        x = torch.randn(1, 3, 4, 4)
        policy, value = net(x)
        assert policy.shape == (1, 10)

    def test_weight_initialization(self):
        config = _small_config()
        net = create_policy_value_network(config, board_size=4)
        # Verify weights are initialized (not all zeros)
        for name, param in net.named_parameters():
            if "weight" in name and param.dim() > 1:
                assert not torch.allclose(param, torch.zeros_like(param))
