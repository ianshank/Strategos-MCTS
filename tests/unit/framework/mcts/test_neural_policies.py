"""
Unit tests for Neural MCTS Policies.

Tests:
- PUCT selection formula
- NeuralPolicyConfig validation
- PriorsManager functionality
- NeuralRolloutPolicy (fallback mode)
- Factory function
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

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
# PUCT Tests
# =============================================================================


class TestPUCT:
    """Tests for PUCT selection formula."""

    def test_puct_basic_calculation(self):
        """Test basic PUCT calculation."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=0.5,
            prior=0.3,
            visit_count=10,
            parent_visits=100,
            c_puct=1.25,
        )

        # Manual calculation:
        # exploration = 1.25 * 0.3 * sqrt(100) / (1 + 10) = 1.25 * 0.3 * 10 / 11 ≈ 0.341
        # score = 0.5 + 0.341 ≈ 0.841
        expected = 0.5 + 1.25 * 0.3 * math.sqrt(100) / (1 + 10)

        assert abs(score - expected) < 1e-6

    def test_puct_unvisited_node(self):
        """Test PUCT returns infinity for unvisited nodes with prior > 0."""
        from src.framework.mcts.neural_policies import puct

        score = puct(
            q_value=0.0,
            prior=0.5,
            visit_count=0,
            parent_visits=100,
            c_puct=1.25,
        )

        assert score == float("inf")

    def test_puct_zero_prior_unvisited(self):
        """Test PUCT for zero-prior unvisited node returns c_puct."""
        from src.framework.mcts.neural_policies import puct

        c_puct = 1.25
        score = puct(
            q_value=0.0,
            prior=0.0,
            visit_count=0,
            parent_visits=100,
            c_puct=c_puct,
        )

        assert score == c_puct

    def test_puct_exploration_decreases_with_visits(self):
        """Test that exploration bonus decreases as node is visited more."""
        from src.framework.mcts.neural_policies import puct

        scores = []
        for visits in [1, 10, 100, 1000]:
            score = puct(
                q_value=0.5,
                prior=0.3,
                visit_count=visits,
                parent_visits=10000,
                c_puct=1.25,
            )
            scores.append(score)

        # Scores should decrease as visits increase (less exploration bonus)
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_puct_with_virtual_loss(self):
        """Test PUCT with virtual loss for parallel search."""
        from src.framework.mcts.neural_policies import puct_with_virtual_loss

        # Without virtual loss
        score_no_vl = puct_with_virtual_loss(
            q_value=0.5,
            prior=0.3,
            visit_count=10,
            parent_visits=100,
            virtual_loss=0.0,
            c_puct=1.25,
        )

        # With virtual loss - should be lower (pessimistic)
        score_with_vl = puct_with_virtual_loss(
            q_value=0.5,
            prior=0.3,
            visit_count=10,
            parent_visits=100,
            virtual_loss=3.0,
            c_puct=1.25,
        )

        assert score_with_vl < score_no_vl


# =============================================================================
# NeuralPolicyConfig Tests
# =============================================================================


class TestNeuralPolicyConfig:
    """Tests for NeuralPolicyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        config = NeuralPolicyConfig()

        assert config.device == "cpu"
        assert config.batch_size == 0
        assert config.use_action_mask is True
        assert config.temperature == 0.0
        assert config.cache_evaluations is True
        assert config.cache_max_size == 10000
        assert config.normalize_value is True

    def test_custom_config(self):
        """Test custom configuration."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        config = NeuralPolicyConfig(
            device="cuda",
            batch_size=32,
            use_action_mask=False,
            temperature=1.0,
            cache_evaluations=False,
            cache_max_size=5000,
            normalize_value=False,
        )

        assert config.device == "cuda"
        assert config.batch_size == 32
        assert config.use_action_mask is False
        assert config.temperature == 1.0
        assert config.cache_evaluations is False
        assert config.cache_max_size == 5000
        assert config.normalize_value is False

    def test_invalid_batch_size(self):
        """Test that negative batch size raises error."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        with pytest.raises(ValueError, match="batch_size"):
            NeuralPolicyConfig(batch_size=-1)

    def test_invalid_cache_size(self):
        """Test that negative cache size raises error."""
        from src.framework.mcts.neural_policies import NeuralPolicyConfig

        with pytest.raises(ValueError, match="cache_max_size"):
            NeuralPolicyConfig(cache_max_size=-1)


# =============================================================================
# PriorsManager Tests
# =============================================================================


class TestPriorsManager:
    """Tests for PriorsManager class."""

    def test_set_and_get_priors(self):
        """Test setting and retrieving priors."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager()

        priors = {"action1": 0.6, "action2": 0.3, "action3": 0.1}
        manager.set_priors("state_hash_1", priors)

        assert manager.get_prior("state_hash_1", "action1") == 0.6
        assert manager.get_prior("state_hash_1", "action2") == 0.3
        assert manager.get_prior("state_hash_1", "action3") == 0.1

    def test_default_prior_for_unknown_action(self):
        """Test default prior for unknown action."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager(default_prior=0.01)

        priors = {"action1": 0.9}
        manager.set_priors("state_hash_1", priors)

        # Unknown action should return default
        assert manager.get_prior("state_hash_1", "unknown_action") == 0.01

    def test_default_prior_for_unknown_state(self):
        """Test default prior for unknown state."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager(default_prior=0.05)

        # No priors set for this state
        assert manager.get_prior("unknown_state", "any_action") == 0.05

    def test_get_all_priors(self):
        """Test getting all priors for a state."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager()

        priors = {"a": 0.5, "b": 0.5}
        manager.set_priors("state", priors)

        all_priors = manager.get_all_priors("state")
        assert all_priors == priors

        # Unknown state returns empty dict
        assert manager.get_all_priors("unknown") == {}

    def test_clear(self):
        """Test clearing all priors."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager()

        manager.set_priors("state1", {"a": 0.5})
        manager.set_priors("state2", {"b": 0.5})

        assert len(manager) == 2

        manager.clear()

        assert len(manager) == 0

    def test_len(self):
        """Test length (number of states)."""
        from src.framework.mcts.neural_policies import PriorsManager

        manager = PriorsManager()

        assert len(manager) == 0

        manager.set_priors("state1", {"a": 0.5})
        assert len(manager) == 1

        manager.set_priors("state2", {"b": 0.5})
        assert len(manager) == 2


# =============================================================================
# FallbackNeuralRolloutPolicy Tests
# =============================================================================


class TestFallbackNeuralRolloutPolicy:
    """Tests for FallbackNeuralRolloutPolicy."""

    def test_fallback_with_heuristic(self):
        """Test fallback policy with custom heuristic."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy

        def heuristic(state):
            return 0.75

        policy = FallbackNeuralRolloutPolicy(heuristic_fn=heuristic)

        # Create mock state
        state = MagicMock()
        state.to_hash_key.return_value = "test_state"

        rng = np.random.default_rng(42)

        import asyncio

        value = asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))

        # Value should be normalized from heuristic output
        # Assuming normalization: (0.75 + 1) / 2 = 0.875
        assert 0.0 <= value <= 1.0

    def test_fallback_without_heuristic(self):
        """Test fallback policy without heuristic returns neutral value."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy

        policy = FallbackNeuralRolloutPolicy()

        state = MagicMock()
        state.to_hash_key.return_value = "test_state"

        rng = np.random.default_rng(42)

        import asyncio

        value = asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))

        # Should return normalized neutral value
        assert value == 0.5  # (0 + 1) / 2 = 0.5

    def test_caching(self):
        """Test that evaluations are cached."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, NeuralPolicyConfig

        call_count = 0

        def heuristic(state):
            nonlocal call_count
            call_count += 1
            return 0.6

        config = NeuralPolicyConfig(cache_evaluations=True)
        policy = FallbackNeuralRolloutPolicy(heuristic_fn=heuristic, config=config)

        state = MagicMock()
        state.to_hash_key.return_value = "test_state"

        rng = np.random.default_rng(42)

        import asyncio

        # First evaluation
        asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))
        assert call_count == 1

        # Second evaluation - should use cache
        asyncio.get_event_loop().run_until_complete(policy.evaluate(state, rng))
        assert call_count == 1  # No additional call

    def test_cache_stats(self):
        """Test cache statistics."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, NeuralPolicyConfig

        config = NeuralPolicyConfig(cache_evaluations=True, cache_max_size=100)
        policy = FallbackNeuralRolloutPolicy(heuristic_fn=lambda s: 0.5, config=config)

        stats = policy.get_cache_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 100


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateNeuralRolloutPolicy:
    """Tests for create_neural_rollout_policy factory."""

    def test_creates_fallback_without_network(self):
        """Test factory creates fallback when no network provided."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, create_neural_rollout_policy

        policy = create_neural_rollout_policy(heuristic_fn=lambda s: 0.5)

        assert isinstance(policy, FallbackNeuralRolloutPolicy)

    def test_creates_fallback_without_adapter(self):
        """Test factory creates fallback when no adapter provided."""
        from src.framework.mcts.neural_policies import FallbackNeuralRolloutPolicy, create_neural_rollout_policy

        # Even with a "network", if no adapter, should fall back
        mock_network = MagicMock()

        policy = create_neural_rollout_policy(network=mock_network)

        assert isinstance(policy, FallbackNeuralRolloutPolicy)


# =============================================================================
# is_torch_available Tests
# =============================================================================


class TestTorchAvailability:
    """Tests for torch availability checking."""

    def test_is_torch_available_returns_bool(self):
        """Test that is_torch_available returns a boolean."""
        from src.framework.mcts.neural_policies import is_torch_available

        result = is_torch_available()
        assert isinstance(result, bool)
