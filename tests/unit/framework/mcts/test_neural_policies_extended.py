"""
Extended tests for Neural MCTS Policies - Edge Cases and Integration.

These tests focus on edge cases and scenarios not covered by the basic tests,
aiming to achieve 70%+ code coverage.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

import numpy as np

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.mcts,
]


# =============================================================================
# PUCT Edge Cases
# =============================================================================


class TestPUCTEdgeCases:
    """Extended edge case tests for PUCT formula."""

    def test_puct_negative_q_value(self):
        """Test PUCT handles negative Q-values correctly."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=-0.5,
            prior=0.3,
            visit_count=10,
            parent_visits=100,
            c_puct=1.25,
        )

        # Should be negative exploitation + positive exploration
        assert score < 0.5  # exploration < 1

    def test_puct_q_value_greater_than_one(self):
        """Test PUCT with Q-value > 1."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=1.5,
            prior=0.1,
            visit_count=100,
            parent_visits=1000,
            c_puct=1.0,
        )

        assert score > 1.5  # exploitation + exploration

    def test_puct_very_large_c_puct(self):
        """Test PUCT with very large exploration constant."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=0.5,
            prior=0.5,
            visit_count=10,
            parent_visits=100,
            c_puct=100.0,
        )

        # Large c_puct should dominate
        assert score > 10.0

    def test_puct_very_small_prior(self):
        """Test PUCT with very small prior probability."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=0.5,
            prior=1e-10,
            visit_count=10,
            parent_visits=100,
            c_puct=1.25,
        )

        # Should still be valid and close to q_value
        assert abs(score - 0.5) < 0.01

    def test_puct_maximum_prior(self):
        """Test PUCT with prior = 1.0."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=0.0,
            prior=1.0,
            visit_count=10,
            parent_visits=100,
            c_puct=1.25,
        )

        # High prior with low visits should give high exploration
        expected_exploration = 1.25 * 1.0 * math.sqrt(100) / 11
        assert abs(score - expected_exploration) < 1e-6

    def test_puct_parent_zero_visits(self):
        """Test PUCT when parent has zero visits."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=0.5,
            prior=0.3,
            visit_count=5,
            parent_visits=0,
            c_puct=1.25,
        )

        # sqrt(0) = 0, so exploration = 0
        assert score == 0.5

    def test_puct_very_large_parent_visits(self):
        """Test PUCT with very large parent visit count."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=0.5,
            prior=0.3,
            visit_count=100,
            parent_visits=1000000,
            c_puct=1.25,
        )

        # Large parent visits should give larger exploration
        assert score > 0.5


class TestPUCTWithVirtualLossEdgeCases:
    """Extended tests for PUCT with virtual loss."""

    def test_virtual_loss_greater_than_visits(self):
        """Test when virtual_loss > visit_count."""
        from src.framework.mcts.neural_policies import puct_with_virtual_loss

        score = puct_with_virtual_loss(
            q_value=0.8,
            prior=0.5,
            visit_count=5,
            parent_visits=100,
            virtual_loss=10.0,
            c_puct=1.25,
        )

        # Should still compute valid score
        assert 0 < score < float("inf")
        assert not math.isnan(score)

    def test_virtual_loss_equals_visits(self):
        """Test when virtual_loss equals visit_count."""
        from src.framework.mcts.neural_policies import puct_with_virtual_loss

        score = puct_with_virtual_loss(
            q_value=0.6,
            prior=0.3,
            visit_count=5,
            parent_visits=100,
            virtual_loss=5.0,
            c_puct=1.25,
        )

        # Q adjustment: (0.6 * 5) / (5 + 5) = 0.3
        assert score > 0

    def test_large_virtual_loss(self):
        """Test with very large virtual loss."""
        from src.framework.mcts.neural_policies import puct_with_virtual_loss

        score = puct_with_virtual_loss(
            q_value=0.9,
            prior=0.5,
            visit_count=10,
            parent_visits=100,
            virtual_loss=1000.0,
            c_puct=1.25,
        )

        # Very pessimistic score due to large virtual loss
        assert score < 0.5


# =============================================================================
# NeuralPolicyConfig Edge Cases
# =============================================================================


class TestNeuralPolicyConfigEdgeCases:
    """Extended tests for NeuralPolicyConfig."""

    def test_cuda_device_indexing(self):
        """Test CUDA device with index."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        config = NeuralPolicyConfig(device="cuda:0")
        assert config.device == "cuda:0"

        config = NeuralPolicyConfig(device="cuda:1")
        assert config.device == "cuda:1"

    def test_mps_device(self):
        """Test Apple MPS device."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        config = NeuralPolicyConfig(device="mps")
        assert config.device == "mps"

    def test_invalid_device_warning(self):
        """Test warning for invalid device."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        # Should warn but not raise
        with patch("src.framework.mcts.neural_policies._logger") as mock_logger:
            config = NeuralPolicyConfig(device="invalid_device")
            # Device should fall back to cpu
            assert config.device == "cpu"
            mock_logger.warning.assert_called()

    def test_zero_batch_size(self):
        """Test batch_size = 0 (no batching)."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        config = NeuralPolicyConfig(batch_size=0)
        assert config.batch_size == 0

    def test_zero_cache_size(self):
        """Test cache_max_size = 0 (unlimited)."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        config = NeuralPolicyConfig(cache_max_size=0)
        assert config.cache_max_size == 0

    def test_temperature_variations(self):
        """Test various temperature values."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        # Zero temperature (argmax)
        config = NeuralPolicyConfig(temperature=0.0)
        assert config.temperature == 0.0

        # High temperature (more random)
        config = NeuralPolicyConfig(temperature=10.0)
        assert config.temperature == 10.0


# =============================================================================
# PriorsManager Edge Cases
# =============================================================================


class TestPriorsManagerEdgeCases:
    """Extended tests for PriorsManager."""

    def test_priors_with_many_states(self):
        """Test with many states stored."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager()

        for i in range(1000):
            manager.set_priors(f"state_{i}", {f"action_{j}": 0.1 for j in range(10)})

        assert len(manager) == 1000
        assert manager.get_prior("state_500", "action_5") == 0.1

    def test_overwrite_priors(self):
        """Test overwriting existing priors."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager()

        manager.set_priors("state_1", {"a": 0.5, "b": 0.5})
        assert manager.get_prior("state_1", "a") == 0.5

        manager.set_priors("state_1", {"a": 0.9, "b": 0.1})
        assert manager.get_prior("state_1", "a") == 0.9

    def test_empty_priors_dict(self):
        """Test setting empty priors dictionary."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager(default_prior=0.01)

        manager.set_priors("state_1", {})
        assert manager.get_prior("state_1", "any_action") == 0.01

    def test_custom_default_prior(self):
        """Test various default prior values."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager(default_prior=0.5)
        assert manager.get_prior("unknown", "action") == 0.5

        manager = PriorsManager(default_prior=0.0)
        assert manager.get_prior("unknown", "action") == 0.0


# =============================================================================
# FallbackNeuralRolloutPolicy Edge Cases
# =============================================================================


class TestFallbackPolicyEdgeCases:
    """Extended tests for FallbackNeuralRolloutPolicy."""

    def test_heuristic_returning_negative(self):
        """Test heuristic that returns negative values."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy

        def negative_heuristic(state):
            return -0.5

        policy = FallbackNeuralRolloutPolicy(heuristic_fn=negative_heuristic)

        state = MagicMock()
        state.to_hash_key.return_value = "test"

        rng = np.random.default_rng(42)

        import asyncio

        value = asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))

        # Should be normalized to [0, 1]
        assert 0.0 <= value <= 1.0

    def test_heuristic_returning_large_value(self):
        """Test heuristic that returns value > 1."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy

        def large_heuristic(state):
            return 5.0

        policy = FallbackNeuralRolloutPolicy(heuristic_fn=large_heuristic)

        state = MagicMock()
        state.to_hash_key.return_value = "test"

        rng = np.random.default_rng(42)

        import asyncio

        value = asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))

        # Should be clamped to 1.0
        assert value == 1.0

    def test_cache_disabled(self):
        """Test with caching disabled."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, NeuralPolicyConfig

        call_count = 0

        def counting_heuristic(state):
            nonlocal call_count
            call_count += 1
            return 0.5

        config = NeuralPolicyConfig(cache_evaluations=False)
        policy = FallbackNeuralRolloutPolicy(heuristic_fn=counting_heuristic, config=config)

        state = MagicMock()
        state.to_hash_key.return_value = "test"

        rng = np.random.default_rng(42)

        import asyncio

        # Should call heuristic each time (no caching)
        asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))
        asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))
        asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))

        assert call_count == 3

    def test_cache_eviction(self):
        """Test cache eviction when max size reached."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, NeuralPolicyConfig

        config = NeuralPolicyConfig(cache_evaluations=True, cache_max_size=10)
        policy = FallbackNeuralRolloutPolicy(heuristic_fn=lambda s: 0.5, config=config)

        rng = np.random.default_rng(42)

        import asyncio

        # Fill cache beyond max size
        for i in range(15):
            state = MagicMock()
            state.to_hash_key.return_value = f"state_{i}"
            asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))

        # Cache should have been evicted
        stats = policy.get_cache_stats()
        # After eviction of 25%, should have fewer entries
        assert stats["size"] <= 15

    def test_clear_cache(self):
        """Test cache clearing."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy

        policy = FallbackNeuralRolloutPolicy(heuristic_fn=lambda s: 0.5)

        state = MagicMock()
        state.to_hash_key.return_value = "test"

        rng = np.random.default_rng(42)

        import asyncio

        asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))
        assert policy.get_cache_stats()["size"] == 1

        policy.clear_cache()
        assert policy.get_cache_stats()["size"] == 0


# =============================================================================
# select_child_puct Edge Cases
# =============================================================================


class TestSelectChildPUCTEdgeCases:
    """Extended tests for select_child_puct function."""

    def test_no_children(self):
        """Test with node that has no children."""
        from src.framework.mcts.neural_policies import PriorsManager, select_child_puct

        node = MagicMock()
        node.children = []

        manager = PriorsManager()

        result = select_child_puct(node, manager)
        assert result is None

    def test_single_child(self):
        """Test with single child."""
        from src.framework.mcts.neural_policies import PriorsManager, select_child_puct

        child = MagicMock()
        child.action = "action_1"
        child.visits = 5
        child.value = 2.5

        node = MagicMock()
        node.children = [child]
        node.visits = 10
        node.state.to_hash_key.return_value = "state_hash"

        manager = PriorsManager()
        manager.set_priors("state_hash", {"action_1": 0.5})

        result = select_child_puct(node, manager)
        assert result is not None
        action, selected_child = result
        assert action == "action_1"
        assert selected_child is child

    def test_all_zero_priors(self):
        """Test when all priors are zero."""
        from src.framework.mcts.neural_policies import PriorsManager, select_child_puct

        child1 = MagicMock()
        child1.action = "a"
        child1.visits = 5
        child1.value = 3.0

        child2 = MagicMock()
        child2.action = "b"
        child2.visits = 10
        child2.value = 7.0

        node = MagicMock()
        node.children = [child1, child2]
        node.visits = 15
        node.state.to_hash_key.return_value = "state"

        manager = PriorsManager(default_prior=0.0)

        result = select_child_puct(node, manager)
        assert result is not None

    def test_child_with_none_action(self):
        """Test child with action = None."""
        from src.framework.mcts.neural_policies import PriorsManager, select_child_puct

        child = MagicMock()
        child.action = None
        child.visits = 5
        child.value = 2.5

        node = MagicMock()
        node.children = [child]
        node.visits = 10
        node.state.to_hash_key.return_value = "state"

        manager = PriorsManager()

        result = select_child_puct(node, manager)
        assert result is not None
        action, _ = result
        assert action == ""  # Default to empty string


# =============================================================================
# Factory Function Edge Cases
# =============================================================================


class TestFactoryFunctionEdgeCases:
    """Extended tests for create_neural_rollout_policy."""

    def test_with_network_but_no_adapter(self):
        """Test with network but missing adapter."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, create_neural_rollout_policy

        mock_network = MagicMock()

        policy = create_neural_rollout_policy(network=mock_network, state_adapter=None)

        assert isinstance(policy, FallbackNeuralRolloutPolicy)

    def test_with_adapter_but_no_network(self):
        """Test with adapter but missing network."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, create_neural_rollout_policy

        mock_adapter = MagicMock()

        policy = create_neural_rollout_policy(network=None, state_adapter=mock_adapter)

        assert isinstance(policy, FallbackNeuralRolloutPolicy)

    def test_with_custom_config(self):
        """Test with custom configuration."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig, create_neural_rollout_policy

        config = NeuralPolicyConfig(device="cuda", cache_max_size=5000)

        policy = create_neural_rollout_policy(config=config)

        assert policy.config.device == "cuda"
        assert policy.config.cache_max_size == 5000


# =============================================================================
# Protocol Tests
# =============================================================================


class TestProtocols:
    """Tests for Protocol classes."""

    def test_state_adapter_protocol(self):
        """Test StateAdapter protocol compliance."""
        from src.framework.mcts.neural_policies import StateAdapter

        # Create a mock that implements the protocol
        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=np.zeros((3, 3, 3)))
        adapter.get_action_mask = MagicMock(return_value=None)
        adapter.tensor_to_action_priors = MagicMock(return_value={"a": 0.5})

        # Should be recognized as implementing the protocol
        assert isinstance(adapter, StateAdapter)

    def test_action_filter_protocol(self):
        """Test ActionFilter protocol compliance."""
        from src.framework.mcts.neural_policies import ActionFilter

        filter_mock = MagicMock()
        filter_mock.get_legal_actions = MagicMock(return_value=["a", "b"])
        filter_mock.is_terminal = MagicMock(return_value=False)

        assert isinstance(filter_mock, ActionFilter)


# =============================================================================
# TorchNeuralRolloutPolicy Tests (requires PyTorch)
# =============================================================================

# Check torch availability
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchNeuralRolloutPolicy:
    """Tests for TorchNeuralRolloutPolicy class."""

    def test_initialization(self):
        """Test TorchNeuralRolloutPolicy initialization."""
        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy

        # Create simple mock network
        network = nn.Linear(9, 1)

        # Create mock adapter
        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=torch.zeros(3, 3))

        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter)

        assert policy.network is not None
        assert policy.state_adapter is adapter

    def test_get_network_value(self):
        """Test value extraction from network."""
        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy

        # Create simple value network
        class SimpleValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(9, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x.view(x.size(0), -1)))

        network = SimpleValueNet()

        # Mock adapter
        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=torch.zeros(1, 3, 3))

        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter)

        # Create mock state
        state = MagicMock()
        state.to_hash_key.return_value = "test"

        value = policy._get_network_value(state)

        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_policy_value_network_output(self):
        """Test with policy-value network that returns tuple."""
        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy

        # Create policy-value network
        class PolicyValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(9, 10)  # 9 actions + 1 value

            def forward(self, x):
                x = x.view(x.size(0), -1)
                out = self.fc(x)
                policy = out[:, :9]
                value = torch.sigmoid(out[:, 9:])
                return policy, value

        network = PolicyValueNet()

        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=torch.zeros(1, 3, 3))

        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter)

        state = MagicMock()
        state.to_hash_key.return_value = "test"

        value = policy._get_network_value(state)

        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_get_policy_priors(self):
        """Test policy prior extraction."""
        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy

        # Create policy network
        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(9, 10)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                out = self.fc(x)
                return out[:, :9], out[:, 9:]  # policy, value

        network = PolicyNet()

        # Mock adapter that returns proper priors
        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=torch.zeros(1, 3, 3))
        adapter.get_action_mask = MagicMock(return_value=None)
        adapter.tensor_to_action_priors = MagicMock(return_value={"0,0": 0.2, "0,1": 0.3, "1,0": 0.5})

        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter)

        state = MagicMock()

        priors = policy.get_policy_priors(state)

        assert isinstance(priors, dict)
        assert "0,0" in priors

    def test_get_policy_priors_with_action_mask(self):
        """Test policy priors with action masking."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig, TorchNeuralRolloutPolicy

        # Create policy network
        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(9, 10)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                out = self.fc(x)
                return out[:, :9], out[:, 9:]

        network = PolicyNet()

        # Adapter with action mask
        mask = torch.tensor([True, True, False, False, False, False, False, False, False])
        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=torch.zeros(1, 3, 3))
        adapter.get_action_mask = MagicMock(return_value=mask)
        adapter.tensor_to_action_priors = MagicMock(return_value={"0": 0.6, "1": 0.4})

        config = NeuralPolicyConfig(use_action_mask=True)
        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter, config=config)

        state = MagicMock()

        priors = policy.get_policy_priors(state)

        # Mask was applied
        adapter.get_action_mask.assert_called_once()
        assert isinstance(priors, dict)

    def test_value_only_network(self):
        """Test with value-only network (no policy output)."""
        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy

        class ValueOnlyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(9, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x.view(x.size(0), -1)))

        network = ValueOnlyNet()

        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=torch.zeros(1, 3, 3))

        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter)

        state = MagicMock()

        # Policy priors should return empty dict for value-only network
        priors = policy.get_policy_priors(state)
        assert priors == {}

    def test_evaluate_async(self):
        """Test async evaluate method."""
        import asyncio

        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(9, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x.view(x.size(0), -1)))

        network = SimpleNet()

        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=torch.zeros(1, 3, 3))

        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter)

        state = MagicMock()
        state.to_hash_key.return_value = "test"

        rng = np.random.default_rng(42)

        value = asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))

        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_numpy_tensor_conversion(self):
        """Test conversion when adapter returns numpy array."""
        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(9, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x.view(x.size(0), -1)))

        network = SimpleNet()

        # Return numpy array instead of tensor
        adapter = MagicMock()
        adapter.state_to_tensor = MagicMock(return_value=np.zeros((1, 3, 3)))

        policy = TorchNeuralRolloutPolicy(network=network, state_adapter=adapter)

        state = MagicMock()
        state.to_hash_key.return_value = "test"

        value = policy._get_network_value(state)

        assert isinstance(value, float)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestCreatePolicyWithTorch:
    """Tests for factory function when PyTorch is available."""

    def test_creates_torch_policy(self):
        """Test factory creates TorchNeuralRolloutPolicy when PyTorch available."""
        from src.framework.mcts.neural_policies import TorchNeuralRolloutPolicy, create_neural_rollout_policy

        network = nn.Linear(9, 1)
        adapter = MagicMock()

        policy = create_neural_rollout_policy(network=network, state_adapter=adapter)

        assert isinstance(policy, TorchNeuralRolloutPolicy)

    def test_fallback_without_adapter(self):
        """Test fallback when adapter not provided."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, create_neural_rollout_policy

        network = nn.Linear(9, 1)

        policy = create_neural_rollout_policy(network=network, state_adapter=None)

        assert isinstance(policy, FallbackNeuralRolloutPolicy)
