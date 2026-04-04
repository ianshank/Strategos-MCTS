"""
Unit tests for src/models/policy_network.py.

Covers PolicyNetwork, PolicyOutput, ActionSelection, PolicyLoss,
and the create_policy_network factory function.

Target uncovered lines: 126->exit, 211->214, 232, 238->241, 255-261, 265,
292-296, 322-380, 398-410.
"""

import pytest
import torch
import torch.nn as nn

from src.models.policy_network import (
    ActionSelection,
    PolicyLoss,
    PolicyNetwork,
    PolicyOutput,
    create_policy_network,
)


# ---------------------------------------------------------------------------
# PolicyOutput dataclass
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyOutput:
    """Tests for the PolicyOutput dataclass."""

    def test_creation_minimal(self):
        logits = torch.randn(2, 5)
        value = torch.randn(2, 1)
        out = PolicyOutput(policy_logits=logits, state_value=value)
        assert out.action_probs is None
        assert out.entropy is None

    def test_creation_with_optionals(self):
        logits = torch.randn(2, 5)
        value = torch.randn(2, 1)
        probs = torch.softmax(logits, dim=-1)
        entropy = torch.tensor([1.0, 1.0])
        out = PolicyOutput(
            policy_logits=logits,
            state_value=value,
            action_probs=probs,
            entropy=entropy,
        )
        assert out.action_probs is not None
        assert out.entropy is not None


# ---------------------------------------------------------------------------
# ActionSelection dataclass
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestActionSelection:
    """Tests for the ActionSelection dataclass."""

    def test_creation(self):
        sel = ActionSelection(action=2, log_prob=-0.5, confidence=0.7, entropy=1.2)
        assert sel.action == 2
        assert sel.log_prob == -0.5
        assert sel.confidence == 0.7
        assert sel.entropy == 1.2


# ---------------------------------------------------------------------------
# PolicyNetwork construction and forward pass
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyNetworkInit:
    """Tests for PolicyNetwork initialization."""

    def test_default_hidden_dims(self):
        net = PolicyNetwork(state_dim=10, action_dim=4)
        assert net.hidden_dims == [256, 256, 128]

    def test_custom_hidden_dims(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32, 16])
        assert net.hidden_dims == [32, 16]

    def test_relu_activation(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, activation="relu")
        assert isinstance(net.activation_fn, nn.ReLU)

    def test_gelu_activation(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, activation="gelu")
        assert isinstance(net.activation_fn, nn.GELU)

    def test_tanh_activation(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, activation="tanh")
        assert isinstance(net.activation_fn, nn.Tanh)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            PolicyNetwork(state_dim=10, action_dim=4, activation="swish")

    def test_no_batch_norm(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, use_batch_norm=False, hidden_dims=[32])
        # Should have no BatchNorm1d layers
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
        assert not has_bn

    def test_with_batch_norm(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, use_batch_norm=True, hidden_dims=[32])
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
        assert has_bn

    def test_weight_initialization(self):
        """Linear layers should have Xavier-initialized weights (non-zero)."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        for m in net.modules():
            if isinstance(m, nn.Linear):
                assert m.weight.abs().sum() > 0
                if m.bias is not None:
                    # Bias should be zeros
                    assert m.bias.abs().sum() == 0.0

    def test_init_weights_linear_no_bias(self):
        """_init_weights handles Linear without bias (line 126->exit)."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[16])
        linear_no_bias = nn.Linear(16, 8, bias=False)
        net._init_weights(linear_no_bias)
        # Should not raise; weight should be set
        assert linear_no_bias.weight.abs().sum() > 0

    def test_init_weights_non_linear_module(self):
        """_init_weights is a no-op for non-Linear modules."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[16])
        bn = nn.BatchNorm1d(16)
        # Should not raise
        net._init_weights(bn)


# ---------------------------------------------------------------------------
# PolicyNetwork forward
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyNetworkForward:
    """Tests for PolicyNetwork.forward method."""

    def test_forward_output_shapes(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32, 16])
        net.eval()
        x = torch.randn(8, 10)
        out = net(x)
        assert out.policy_logits.shape == (8, 4)
        assert out.state_value.shape == (8, 1)
        assert out.action_probs is None
        assert out.entropy is None

    def test_forward_with_return_probs(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32, 16])
        net.eval()
        x = torch.randn(8, 10)
        out = net(x, return_probs=True)
        assert out.action_probs is not None
        assert out.entropy is not None
        assert out.action_probs.shape == (8, 4)
        assert out.entropy.shape == (8,)
        # Probabilities should sum to 1
        prob_sums = out.action_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(8), atol=1e-5)

    def test_forward_single_sample(self):
        net = PolicyNetwork(state_dim=10, action_dim=3, hidden_dims=[16])
        net.eval()
        x = torch.randn(1, 10)
        out = net(x)
        assert out.policy_logits.shape == (1, 3)

    def test_gradient_flow(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        x = torch.randn(4, 10, requires_grad=True)
        output = net(x)
        loss = output.policy_logits.sum() + output.state_value.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# PolicyNetwork.select_action
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyNetworkSelectAction:
    """Tests for PolicyNetwork.select_action method."""

    def test_select_action_deterministic(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(10)
        sel = net.select_action(state, deterministic=True)
        assert isinstance(sel, ActionSelection)
        assert 0 <= sel.action < 4
        assert sel.confidence > 0.0
        assert sel.entropy >= 0.0

    def test_select_action_stochastic(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(10)
        sel = net.select_action(state, deterministic=False)
        assert 0 <= sel.action < 4

    def test_select_action_with_batch_dim(self):
        """State already has batch dimension [1, state_dim]."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(1, 10)
        sel = net.select_action(state, deterministic=True)
        assert 0 <= sel.action < 4

    def test_select_action_temperature(self):
        """Low temperature should make selection more greedy."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(10)
        sel_low = net.select_action(state, temperature=0.01, deterministic=True)
        sel_high = net.select_action(state, temperature=10.0, deterministic=True)
        # Both should produce valid actions
        assert 0 <= sel_low.action < 4
        assert 0 <= sel_high.action < 4
        # Low temperature should be more confident
        assert sel_low.confidence >= sel_high.confidence - 0.01

    def test_select_action_top_k(self):
        """Top-k filtering restricts to k actions."""
        net = PolicyNetwork(state_dim=10, action_dim=8, hidden_dims=[32])
        state = torch.randn(10)
        sel = net.select_action(state, top_k=3, deterministic=True)
        assert 0 <= sel.action < 8

    def test_select_action_top_k_equals_action_dim(self):
        """top_k >= action_dim should behave same as no top_k."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(10)
        sel = net.select_action(state, top_k=4, deterministic=True)
        assert 0 <= sel.action < 4

    def test_select_action_restores_training_mode(self):
        """select_action restores training mode after call (line 211->214)."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        net.train()
        assert net.training is True
        state = torch.randn(10)
        net.select_action(state, deterministic=True)
        assert net.training is True

    def test_select_action_stays_eval_if_was_eval(self):
        """If model was in eval mode, stays in eval mode."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        net.eval()
        state = torch.randn(10)
        net.select_action(state, deterministic=True)
        assert net.training is False


# ---------------------------------------------------------------------------
# PolicyNetwork.get_action_probs
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyNetworkGetActionProbs:
    """Tests for PolicyNetwork.get_action_probs method."""

    def test_get_action_probs_batched(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(5, 10)
        probs = net.get_action_probs(state)
        assert probs.shape == (5, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_get_action_probs_1d_input(self):
        """1D input should be unsqueezed to batch (line 232)."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(10)
        probs = net.get_action_probs(state)
        assert probs.shape == (1, 4)

    def test_get_action_probs_temperature(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        state = torch.randn(10)
        probs_default = net.get_action_probs(state, temperature=1.0)
        probs_low = net.get_action_probs(state, temperature=0.01)
        # Low temp -> one action has most of the probability
        assert probs_low.max() >= probs_default.max() - 0.01

    def test_get_action_probs_restores_training_mode(self):
        """get_action_probs restores training mode (line 238->241)."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        net.train()
        state = torch.randn(10)
        net.get_action_probs(state)
        assert net.training is True

    def test_get_action_probs_stays_eval(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        net.eval()
        state = torch.randn(10)
        net.get_action_probs(state)
        assert net.training is False


# ---------------------------------------------------------------------------
# PolicyNetwork.evaluate_actions (lines 255-261)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyNetworkEvaluateActions:
    """Tests for PolicyNetwork.evaluate_actions method."""

    def test_evaluate_actions_shapes(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32, 16])
        net.eval()
        states = torch.randn(8, 10)
        actions = torch.randint(0, 4, (8,))
        log_probs, entropy = net.evaluate_actions(states, actions)
        assert log_probs.shape == (8,)
        assert entropy.shape == (8,)

    def test_evaluate_actions_log_probs_negative(self):
        """Log probabilities should be <= 0."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        net.eval()
        states = torch.randn(16, 10)
        actions = torch.randint(0, 4, (16,))
        log_probs, _ = net.evaluate_actions(states, actions)
        assert (log_probs <= 0.0 + 1e-6).all()

    def test_evaluate_actions_entropy_nonnegative(self):
        """Entropy should be >= 0."""
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        net.eval()
        states = torch.randn(16, 10)
        actions = torch.randint(0, 4, (16,))
        _, entropy = net.evaluate_actions(states, actions)
        assert (entropy >= -1e-6).all()

    def test_evaluate_actions_gradient_flows(self):
        """evaluate_actions should support gradient computation for training."""
        # Use use_batch_norm=False to avoid issues with unused BN params
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[16], use_batch_norm=False)
        states = torch.randn(4, 10)
        actions = torch.randint(0, 4, (4,))
        log_probs, entropy = net.evaluate_actions(states, actions)
        loss = -log_probs.mean() - 0.01 * entropy.mean()
        loss.backward()
        # At least some parameters should have gradients
        grads_found = sum(1 for p in net.parameters() if p.requires_grad and p.grad is not None)
        assert grads_found > 0


# ---------------------------------------------------------------------------
# PolicyNetwork.get_parameter_count (line 265)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyNetworkParameterCount:
    """Tests for get_parameter_count."""

    def test_parameter_count_positive(self):
        net = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[32])
        count = net.get_parameter_count()
        assert count > 0

    def test_parameter_count_increases_with_dims(self):
        small = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[16])
        large = PolicyNetwork(state_dim=10, action_dim=4, hidden_dims=[256, 256])
        assert large.get_parameter_count() > small.get_parameter_count()


# ---------------------------------------------------------------------------
# PolicyLoss (lines 292-296, 322-380)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPolicyLoss:
    """Tests for the PolicyLoss module."""

    def _make_policy_output(self, batch_size=4, action_dim=3, with_entropy=True):
        """Helper to create a PolicyOutput for testing."""
        logits = torch.randn(batch_size, action_dim, requires_grad=True)
        value = torch.randn(batch_size, 1, requires_grad=True)
        if with_entropy:
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
        else:
            entropy = None
        return PolicyOutput(
            policy_logits=logits,
            state_value=value,
            action_probs=torch.softmax(logits, dim=-1) if with_entropy else None,
            entropy=entropy,
        )

    def test_init_default_weights(self):
        loss_fn = PolicyLoss()
        assert loss_fn.policy_weight == 1.0
        assert loss_fn.value_weight == 0.5
        assert loss_fn.entropy_weight == 0.01
        assert loss_fn.l2_weight == 0.0001

    def test_init_custom_weights(self):
        loss_fn = PolicyLoss(policy_weight=2.0, value_weight=1.0, entropy_weight=0.05, l2_weight=0.001)
        assert loss_fn.policy_weight == 2.0
        assert loss_fn.value_weight == 1.0

    def test_supervised_with_target_policy(self):
        """Policy loss with target distribution (KL-divergence-like)."""
        loss_fn = PolicyLoss()
        output = self._make_policy_output(batch_size=4, action_dim=3)
        target_policy = torch.softmax(torch.randn(4, 3), dim=-1)
        target_values = torch.randn(4)

        total_loss, loss_dict = loss_fn(
            output,
            target_policy=target_policy,
            target_values=target_values,
        )
        assert "policy_supervised" in loss_dict
        assert "value_mse" in loss_dict
        assert "total" in loss_dict
        assert total_loss.requires_grad

    def test_supervised_with_target_actions(self):
        """Policy loss with target action labels (cross-entropy)."""
        loss_fn = PolicyLoss()
        output = self._make_policy_output(batch_size=4, action_dim=3)
        target_actions = torch.randint(0, 3, (4,))
        target_values = torch.randn(4)

        total_loss, loss_dict = loss_fn(
            output,
            target_actions=target_actions,
            target_values=target_values,
        )
        assert "policy_ce" in loss_dict
        assert "value_mse" in loss_dict

    def test_policy_gradient_with_advantages(self):
        """Policy gradient loss with advantages."""
        loss_fn = PolicyLoss()
        output = self._make_policy_output(batch_size=4, action_dim=3)
        target_actions = torch.randint(0, 3, (4,))
        advantages = torch.randn(4)

        total_loss, loss_dict = loss_fn(
            output,
            target_actions=target_actions,
            advantages=advantages,
        )
        assert "policy_gradient" in loss_dict

    def test_no_policy_target(self):
        """When no policy target is given, policy loss should be zero."""
        loss_fn = PolicyLoss()
        output = self._make_policy_output(batch_size=4, action_dim=3)

        total_loss, loss_dict = loss_fn(output)
        assert "total" in loss_dict
        # Should not have any policy-specific loss key
        assert "policy_supervised" not in loss_dict
        assert "policy_ce" not in loss_dict
        assert "policy_gradient" not in loss_dict

    def test_value_loss_only(self):
        """Only value targets, no policy targets."""
        loss_fn = PolicyLoss()
        output = self._make_policy_output(batch_size=4, action_dim=3)
        target_values = torch.randn(4)

        total_loss, loss_dict = loss_fn(output, target_values=target_values)
        assert "value_mse" in loss_dict
        assert loss_dict["value_mse"] > 0.0

    def test_no_value_target(self):
        """When no value target, value loss should be zero."""
        loss_fn = PolicyLoss()
        output = self._make_policy_output(batch_size=4, action_dim=3)

        _, loss_dict = loss_fn(output)
        assert "value_mse" not in loss_dict

    def test_entropy_regularization_with_entropy(self):
        """Entropy from PolicyOutput is used when available."""
        loss_fn = PolicyLoss(entropy_weight=0.1)
        output = self._make_policy_output(batch_size=4, action_dim=3, with_entropy=True)

        _, loss_dict = loss_fn(output)
        assert "entropy" in loss_dict
        assert loss_dict["entropy"] >= 0.0

    def test_entropy_regularization_without_entropy(self):
        """Entropy computed manually when not in PolicyOutput."""
        loss_fn = PolicyLoss(entropy_weight=0.1)
        output = self._make_policy_output(batch_size=4, action_dim=3, with_entropy=False)

        _, loss_dict = loss_fn(output)
        assert "entropy" in loss_dict
        assert loss_dict["entropy"] >= 0.0

    def test_l2_regularization(self):
        """L2 regularization using model parameters."""
        loss_fn = PolicyLoss(l2_weight=0.01)
        output = self._make_policy_output(batch_size=4, action_dim=3)
        model = PolicyNetwork(state_dim=10, action_dim=3, hidden_dims=[16])

        total_loss, loss_dict = loss_fn(output, model=model)
        assert "l2" in loss_dict
        assert loss_dict["l2"] > 0.0

    def test_l2_regularization_no_model(self):
        """L2 regularization is zero when no model is provided."""
        loss_fn = PolicyLoss(l2_weight=0.01)
        output = self._make_policy_output(batch_size=4, action_dim=3)

        _, loss_dict = loss_fn(output)
        assert "l2" not in loss_dict

    def test_l2_regularization_zero_weight(self):
        """L2 regularization is skipped when l2_weight=0."""
        loss_fn = PolicyLoss(l2_weight=0.0)
        output = self._make_policy_output(batch_size=4, action_dim=3)
        model = PolicyNetwork(state_dim=10, action_dim=3, hidden_dims=[16])

        _, loss_dict = loss_fn(output, model=model)
        assert "l2" not in loss_dict

    def test_combined_loss_all_components(self):
        """Full combined loss with all components."""
        loss_fn = PolicyLoss(policy_weight=1.0, value_weight=0.5, entropy_weight=0.01, l2_weight=0.001)
        output = self._make_policy_output(batch_size=4, action_dim=3)
        target_actions = torch.randint(0, 3, (4,))
        target_values = torch.randn(4)
        model = PolicyNetwork(state_dim=10, action_dim=3, hidden_dims=[16])

        total_loss, loss_dict = loss_fn(
            output,
            target_actions=target_actions,
            target_values=target_values,
            model=model,
        )
        assert "policy_ce" in loss_dict
        assert "value_mse" in loss_dict
        assert "entropy" in loss_dict
        assert "l2" in loss_dict
        assert "total" in loss_dict
        assert total_loss.requires_grad

    def test_loss_backward(self):
        """Total loss should support backward pass."""
        loss_fn = PolicyLoss()
        output = self._make_policy_output(batch_size=4, action_dim=3)
        target_actions = torch.randint(0, 3, (4,))

        total_loss, _ = loss_fn(output, target_actions=target_actions)
        total_loss.backward()
        assert output.policy_logits.grad is not None


# ---------------------------------------------------------------------------
# create_policy_network factory (lines 398-410)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCreatePolicyNetwork:
    """Tests for the create_policy_network factory function."""

    def test_default_config(self):
        net = create_policy_network(state_dim=10, action_dim=4)
        assert isinstance(net, PolicyNetwork)
        assert net.state_dim == 10
        assert net.action_dim == 4
        assert net.hidden_dims == [256, 256, 128]

    def test_custom_config(self):
        config = {
            "hidden_dims": [64, 32],
            "dropout": 0.2,
            "use_batch_norm": False,
            "activation": "gelu",
        }
        net = create_policy_network(state_dim=20, action_dim=6, config=config)
        assert net.state_dim == 20
        assert net.action_dim == 6
        assert net.hidden_dims == [64, 32]
        assert net.use_batch_norm is False

    def test_none_config(self):
        net = create_policy_network(state_dim=10, action_dim=4, config=None)
        assert isinstance(net, PolicyNetwork)

    def test_device_placement(self):
        net = create_policy_network(state_dim=10, action_dim=4, device="cpu")
        # All parameters should be on CPU
        for p in net.parameters():
            assert p.device == torch.device("cpu")

    def test_forward_pass_after_creation(self):
        """Network created by factory should work for forward pass."""
        net = create_policy_network(state_dim=10, action_dim=4)
        net.eval()
        x = torch.randn(2, 10)
        out = net(x)
        assert out.policy_logits.shape == (2, 4)
        assert out.state_value.shape == (2, 1)

    def test_partial_config(self):
        """Factory handles config with only some keys specified."""
        config = {"hidden_dims": [64]}
        net = create_policy_network(state_dim=10, action_dim=4, config=config)
        assert net.hidden_dims == [64]
        # Other defaults should apply
        assert net.dropout == 0.1
        assert net.use_batch_norm is True
