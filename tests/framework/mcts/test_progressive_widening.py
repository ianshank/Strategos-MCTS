"""
Tests for Progressive Widening and RAVE MCTS implementation.

Tests the advanced MCTS features including:
- Progressive widening for large action spaces
- RAVE (Rapid Action Value Estimation)
- Hybrid UCB + RAVE selection
"""

import math
from unittest.mock import Mock

import pytest

from src.framework.mcts.progressive_widening import (
    ProgressiveWideningConfig,
    RAVEConfig,
    RAVENode,
)


class TestProgressiveWideningConfig:
    """Tests for ProgressiveWideningConfig."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ProgressiveWideningConfig()

        assert config.k == 1.0
        assert config.alpha == 0.5
        assert config.adaptive is False
        assert config.k_min == 0.5
        assert config.k_max == 3.0

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ProgressiveWideningConfig(k=2.0, alpha=0.3, adaptive=True, k_min=1.0, k_max=5.0)

        assert config.k == 2.0
        assert config.alpha == 0.3
        assert config.adaptive is True
        assert config.k_min == 1.0
        assert config.k_max == 5.0

    def test_should_expand_first_child(self):
        """Test that first child is always expanded."""
        config = ProgressiveWideningConfig()

        # With zero children, should always expand regardless of visits
        assert config.should_expand(visits=0, num_children=0)
        assert config.should_expand(visits=1, num_children=0)
        assert config.should_expand(visits=100, num_children=0)

    def test_should_expand_threshold_calculation(self):
        """Test expansion threshold calculation."""
        config = ProgressiveWideningConfig(k=1.0, alpha=0.5)

        # Threshold = k * num_children^alpha = 1.0 * 1^0.5 = 1.0
        # visits > threshold, so should expand
        assert config.should_expand(visits=2, num_children=1)

        # Threshold = 1.0 * 4^0.5 = 2.0
        # visits <= threshold, so should not expand
        assert not config.should_expand(visits=2, num_children=4)

    def test_should_expand_with_different_alpha(self):
        """Test expansion behavior with different alpha values."""
        # Lower alpha = slower expansion (more conservative)
        config_conservative = ProgressiveWideningConfig(k=1.0, alpha=0.25)

        # Higher alpha = faster expansion (more aggressive)
        config_aggressive = ProgressiveWideningConfig(k=1.0, alpha=0.75)

        # With same visits and children, conservative requires more visits
        visits = 10
        num_children = 4

        # Threshold for conservative: 1.0 * 4^0.25 ≈ 1.41
        # Threshold for aggressive: 1.0 * 4^0.75 ≈ 2.83

        conservative_expands = config_conservative.should_expand(visits, num_children)
        aggressive_expands = config_aggressive.should_expand(visits, num_children)

        # Both should expand since visits > threshold in both cases
        assert conservative_expands
        assert aggressive_expands

    def test_min_visits_for_next_child(self):
        """Test calculation of minimum visits for next expansion."""
        config = ProgressiveWideningConfig(k=1.0, alpha=0.5)

        # With 0 children, threshold = 0, min_visits = 1
        min_visits = config.min_visits_for_next_child(num_children=0)
        assert min_visits == 1

        # With 1 child, threshold = 1.0, min_visits = 2
        min_visits = config.min_visits_for_next_child(num_children=1)
        assert min_visits == 2

        # With 4 children, threshold = 2.0, min_visits = 3
        min_visits = config.min_visits_for_next_child(num_children=4)
        assert min_visits == 3

    def test_min_visits_ceiling_behavior(self):
        """Test that min_visits properly rounds up."""
        config = ProgressiveWideningConfig(k=1.5, alpha=0.5)

        # threshold = 1.5 * 2^0.5 ≈ 2.121
        # ceiling(2.121) + 1 = 4
        min_visits = config.min_visits_for_next_child(num_children=2)
        expected = int(math.ceil(1.5 * (2**0.5))) + 1
        assert min_visits == expected


class TestRAVEConfig:
    """Tests for RAVE configuration."""

    def test_default_rave_configuration(self):
        """Test default RAVE configuration."""
        config = RAVEConfig()

        assert config.rave_constant == 300.0
        assert config.enable_rave is True
        assert config.min_visits_for_rave == 5

    def test_custom_rave_configuration(self):
        """Test custom RAVE configuration."""
        config = RAVEConfig(rave_constant=500.0, enable_rave=False, min_visits_for_rave=10)

        assert config.rave_constant == 500.0
        assert config.enable_rave is False
        assert config.min_visits_for_rave == 10

    def test_compute_beta_disabled_rave(self):
        """Test beta computation when RAVE is disabled."""
        config = RAVEConfig(enable_rave=False)

        beta = config.compute_beta(node_visits=10, rave_visits=10)

        # Should return 0.0 when RAVE is disabled
        assert beta == 0.0

    def test_compute_beta_insufficient_rave_visits(self):
        """Test beta computation with insufficient RAVE visits."""
        config = RAVEConfig(min_visits_for_rave=5)

        beta = config.compute_beta(node_visits=10, rave_visits=3)

        # Should return 0.0 when rave_visits < min_visits_for_rave
        assert beta == 0.0

    def test_compute_beta_basic_calculation(self):
        """Test basic beta computation."""
        config = RAVEConfig(rave_constant=300.0, min_visits_for_rave=5)

        beta = config.compute_beta(node_visits=10, rave_visits=10)

        # Beta should be between 0 and 1
        assert 0.0 <= beta <= 1.0

        # Beta formula: rave_visits / (node_visits + rave_visits + 4*k²*n*r/10000)
        expected_denominator = 10 + 10 + 4 * (300.0**2) * 10 * 10 / 10000.0
        expected_beta = 10 / expected_denominator

        assert abs(beta - expected_beta) < 1e-6

    def test_compute_beta_decay_with_visits(self):
        """Test that beta decays as visits increase."""
        config = RAVEConfig(rave_constant=300.0)

        # With few visits, beta should be higher
        beta_early = config.compute_beta(node_visits=10, rave_visits=10)

        # With many visits, beta should be lower (rely more on UCB)
        beta_late = config.compute_beta(node_visits=1000, rave_visits=1000)

        assert beta_early > beta_late

    def test_compute_beta_bounds(self):
        """Test that beta is always bounded in [0, 1]."""
        config = RAVEConfig()

        # Test various scenarios
        test_cases = [
            (5, 5),
            (10, 100),
            (100, 10),
            (1000, 1000),
            (0, 5),  # Edge case: zero node visits
        ]

        for node_visits, rave_visits in test_cases:
            beta = config.compute_beta(node_visits, rave_visits)
            assert 0.0 <= beta <= 1.0, f"Beta out of bounds for n={node_visits}, r={rave_visits}"

    def test_compute_beta_zero_denominator(self):
        """Test beta computation handles zero denominator gracefully."""
        config = RAVEConfig(rave_constant=0.0)  # This could cause zero denominator

        beta = config.compute_beta(node_visits=0, rave_visits=0)

        # Should handle gracefully and return 0.0
        assert beta == 0.0


class TestRAVENode:
    """Tests for RAVENode."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        state = Mock()
        state.state_hash = "test_state"
        state.available_actions = ["action1", "action2", "action3"]
        return state

    @pytest.fixture
    def rave_node(self, mock_state):
        """Create a RAVENode for testing."""
        return RAVENode(state=mock_state, parent=None)

    def test_rave_node_initialization(self, rave_node):
        """Test RAVE node initializes with RAVE statistics."""
        assert hasattr(rave_node, "rave_visits")
        assert hasattr(rave_node, "rave_value_sum")
        assert isinstance(rave_node.rave_visits, dict)
        assert isinstance(rave_node.rave_value_sum, dict)
        assert len(rave_node.rave_visits) == 0
        assert len(rave_node.rave_value_sum) == 0

    def test_rave_node_update_new_action(self, rave_node):
        """Test updating RAVE stats for a new action."""
        rave_node.update_rave(action="action1", value=0.8)

        assert "action1" in rave_node.rave_visits
        assert rave_node.rave_visits["action1"] == 1
        assert rave_node.rave_value_sum["action1"] == 0.8

    def test_rave_node_update_existing_action(self, rave_node):
        """Test updating RAVE stats for an existing action."""
        # First update
        rave_node.update_rave(action="action1", value=0.8)

        # Second update
        rave_node.update_rave(action="action1", value=0.6)

        assert rave_node.rave_visits["action1"] == 2
        assert rave_node.rave_value_sum["action1"] == 1.4  # 0.8 + 0.6

    def test_rave_node_multiple_actions(self, rave_node):
        """Test updating multiple different actions."""
        rave_node.update_rave(action="action1", value=0.8)
        rave_node.update_rave(action="action2", value=0.6)
        rave_node.update_rave(action="action3", value=0.9)

        assert len(rave_node.rave_visits) == 3
        assert rave_node.rave_visits["action1"] == 1
        assert rave_node.rave_visits["action2"] == 1
        assert rave_node.rave_visits["action3"] == 1

    def test_get_rave_value_existing_action(self, rave_node):
        """Test getting RAVE value for tracked action."""
        rave_node.update_rave(action="action1", value=0.8)
        rave_node.update_rave(action="action1", value=0.6)

        rave_value = rave_node.get_rave_value(action="action1")

        # Average = 1.4 / 2 = 0.7
        assert abs(rave_value - 0.7) < 1e-6

    def test_get_rave_value_missing_action(self, rave_node):
        """Test getting RAVE value for untracked action."""
        rave_value = rave_node.get_rave_value(action="unknown_action")

        # Should return 0.5 (neutral value) for unknown actions
        assert rave_value == 0.5

    def test_get_rave_visits_existing_action(self, rave_node):
        """Test getting RAVE visit count."""
        rave_node.update_rave(action="action1", value=0.8)
        rave_node.update_rave(action="action1", value=0.6)

        visits = rave_node.get_rave_visits(action="action1")
        assert visits == 2

    def test_get_rave_visits_missing_action(self, rave_node):
        """Test getting RAVE visits for untracked action."""
        visits = rave_node.get_rave_visits(action="unknown_action")
        assert visits == 0


@pytest.mark.integration
class TestProgressiveWideningIntegration:
    """Integration tests for progressive widening with MCTS."""

    def test_expansion_pattern_over_time(self):
        """Test that expansion follows expected pattern."""
        config = ProgressiveWideningConfig(k=1.0, alpha=0.5)

        # Simulate visits and track when children should be expanded
        expansions = []
        num_children = 0

        for visits in range(1, 50):
            if config.should_expand(visits, num_children):
                expansions.append((visits, num_children))
                num_children += 1

        # Should have expanded multiple times
        assert len(expansions) > 1

        # Expansion rate should slow down (increasing gaps between expansions)
        if len(expansions) >= 3:
            gap1 = expansions[1][0] - expansions[0][0]
            gap2 = expansions[2][0] - expansions[1][0]
            # Later gaps should be larger (due to alpha < 1)
            assert gap2 >= gap1

    def test_rave_statistics_accumulation(self):
        """Test that RAVE statistics accumulate correctly."""
        mock_state = Mock()
        mock_state.state_hash = "test"
        mock_state.available_actions = ["a1", "a2", "a3"]

        node = RAVENode(state=mock_state, parent=None)

        # Simulate a sequence of updates
        updates = [
            ("a1", 0.9),
            ("a2", 0.5),
            ("a1", 0.7),
            ("a3", 0.3),
            ("a1", 0.8),
        ]

        for action, value in updates:
            node.update_rave(action, value)

        # Check accumulated statistics
        assert node.get_rave_visits("a1") == 3
        assert node.get_rave_visits("a2") == 1
        assert node.get_rave_visits("a3") == 1

        # Check average values
        expected_a1_avg = (0.9 + 0.7 + 0.8) / 3
        assert abs(node.get_rave_value("a1") - expected_a1_avg) < 1e-6


@pytest.mark.performance
class TestProgressiveWideningPerformance:
    """Performance tests for progressive widening."""

    def test_expansion_controlled_growth(self):
        """Test that progressive widening controls expansion rate."""
        config = ProgressiveWideningConfig(k=2.0, alpha=0.5)

        # Test specific scenarios where expansion should be controlled
        test_cases = [
            # (visits, num_children, should_expand?)
            (1, 0, True),  # Always expand first child
            (2, 1, False),  # threshold=2.0, 2 is not > 2.0
            (3, 1, True),  # threshold=2.0, 3 > 2.0
            (4, 4, False),  # threshold=2*4^0.5=4.0, 4 is not > 4.0
            (5, 4, True),  # threshold=2*4^0.5=4.0, 5 > 4.0
            (10, 6, True),  # threshold=2*6^0.5≈4.9, 10 > 4.9
        ]

        for visits, num_children, expected in test_cases:
            result = config.should_expand(visits, num_children)
            assert result == expected, f"visits={visits}, children={num_children}: expected {expected}, got {result}"

    def test_min_visits_increases_with_children(self):
        """Test that required visits increase as children grow."""
        config = ProgressiveWideningConfig(k=2.0, alpha=0.5)

        # As we add more children, minimum visits should increase
        min_visits_sequence = []
        for num_children in range(1, 10):
            min_visits = config.min_visits_for_next_child(num_children)
            min_visits_sequence.append(min_visits)

        # Each value should be >= previous (monotonic increase)
        for i in range(1, len(min_visits_sequence)):
            assert min_visits_sequence[i] >= min_visits_sequence[i - 1]
