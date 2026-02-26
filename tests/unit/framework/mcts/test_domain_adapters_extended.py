"""
Extended tests for Domain Adapters - Edge Cases and Integration.

These tests focus on edge cases and scenarios not covered by the basic tests,
aiming to achieve 70%+ code coverage.
"""

from __future__ import annotations

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
# GridStateAdapter Edge Cases
# =============================================================================


class TestGridStateAdapterEdgeCases:
    """Extended edge case tests for GridStateAdapter."""

    def test_large_board_8x8(self):
        """Test with 8x8 board like Othello."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(board_size=8, num_channels=3)
        adapter = GridStateAdapter(config=config)

        assert adapter.action_space_size == 64
        assert adapter.state_dim == (3, 8, 8)

    def test_large_board_19x19(self):
        """Test with 19x19 board like Go."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(board_size=19, num_channels=2)
        adapter = GridStateAdapter(config=config)

        assert adapter.action_space_size == 361
        assert adapter.state_dim == (2, 19, 19)

    def test_board_with_history(self):
        """Test with history enabled."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(
            board_size=3,
            num_channels=3,
            include_history=True,
            history_length=4,
        )
        adapter = GridStateAdapter(config=config)

        # Should have 3 channels * 4 history = 12 channels
        assert adapter.state_dim == (12, 3, 3)

    def test_state_to_tensor_with_history(self):
        """Test tensor conversion with history."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(
            board_size=3,
            num_channels=3,
            include_history=True,
            history_length=2,
        )
        adapter = GridStateAdapter(config=config)

        state = MagicMock()
        state.features = {
            "board": [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
            "player": 1,
            "history": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # Previous state
            ],
        }

        tensor = adapter.state_to_tensor(state)

        # 3 channels for current + 2 channels for (history_length - 1) history entries
        # With history_length=2: 3 + (2-1)*2 = 5 channels
        assert tensor.shape == (5, 3, 3)

    def test_absolute_perspective(self):
        """Test absolute (not player-relative) perspective."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(board_size=3, player_perspective=False)
        adapter = GridStateAdapter(config=config)

        state = MagicMock()
        state.features = {
            "board": [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
            "player": -1,  # Player 2's turn
        }

        tensor = adapter.state_to_tensor(state)

        # Channel 0 should be player 1 pieces (value 1)
        assert tensor[0, 0, 0] == 1.0  # Player 1 at (0,0)
        # Channel 1 should be player 2 pieces (value -1)
        assert tensor[1, 0, 2] == 1.0  # Player 2 at (0,2)

    def test_empty_board(self):
        """Test with completely empty board."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        state = MagicMock()
        state.features = {
            "board": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "player": 1,
        }

        tensor = adapter.state_to_tensor(state)

        # All channels should have correct empty representation
        assert tensor[0].sum() == 0  # No current player pieces
        assert tensor[1].sum() == 0  # No opponent pieces
        assert tensor[2].sum() == 9  # All 9 cells empty

    def test_full_board(self):
        """Test with completely full board (no legal moves)."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        state = MagicMock()
        state.features = {
            "board": [[1, -1, 1], [-1, 1, -1], [1, -1, 1]],  # Full board
            "player": 1,
        }

        legal = adapter._extract_legal_actions(state)
        assert len(legal) == 0  # No legal moves

        mask = adapter.get_action_mask(state)
        # When no legal actions exist, get_action_mask returns None
        assert mask is None

    def test_action_to_index_large_board(self):
        """Test action to index conversion for larger board."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(board_size=8)
        adapter = GridStateAdapter(config=config)

        # Test corner positions
        assert adapter.action_to_index("0,0") == 0
        assert adapter.action_to_index("0,7") == 7
        assert adapter.action_to_index("7,0") == 56
        assert adapter.action_to_index("7,7") == 63

    def test_index_to_action_large_board(self):
        """Test index to action conversion for larger board."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(board_size=8)
        adapter = GridStateAdapter(config=config)

        assert adapter.index_to_action(0) == "0,0"
        assert adapter.index_to_action(7) == "0,7"
        assert adapter.index_to_action(56) == "7,0"
        assert adapter.index_to_action(63) == "7,7"

    def test_tensor_to_priors_with_zeros(self):
        """Test prior conversion when all probabilities are zero."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        policy_output = np.zeros(9)  # All zeros

        state = MagicMock()
        state.features = {
            "legal_moves": ["0,0", "0,1", "0,2"],
        }

        priors = adapter.tensor_to_action_priors(policy_output, state)

        # Should default to uniform distribution
        assert len(priors) == 3
        for p in priors.values():
            assert abs(p - 1 / 3) < 1e-6


# =============================================================================
# FeatureStateAdapter Edge Cases
# =============================================================================


class TestFeatureStateAdapterEdgeCases:
    """Extended tests for FeatureStateAdapter."""

    def test_negative_features(self):
        """Test with negative feature values."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=4, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = [-1.0, -2.0, -3.0, -4.0]

        tensor = adapter.state_to_tensor(state)

        np.testing.assert_array_equal(tensor, [-1.0, -2.0, -3.0, -4.0])

    def test_very_large_features(self):
        """Test with very large feature values."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=3, normalize_features=True)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = [1e6, 2e6, 3e6]

        tensor = adapter.state_to_tensor(state)

        # Should be normalized
        norm = np.linalg.norm(tensor)
        assert abs(norm - 1.0) < 1e-6

    def test_zero_norm_features(self):
        """Test normalization with zero-norm vector."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=3, normalize_features=True)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = [0.0, 0.0, 0.0]

        tensor = adapter.state_to_tensor(state)

        # Should handle zero norm gracefully (no normalization)
        np.testing.assert_array_equal(tensor, [0.0, 0.0, 0.0])

    def test_named_features(self):
        """Test with named feature configuration."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(
            feature_dim=3,
            normalize_features=False,
            feature_names=["width", "height", "depth"],
        )
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = {
            "width": 10.0,
            "height": 20.0,
            "depth": 30.0,
            "extra": 100.0,  # Should be ignored
        }

        tensor = adapter.state_to_tensor(state)

        np.testing.assert_array_equal(tensor, [10.0, 20.0, 30.0])

    def test_missing_named_features(self):
        """Test with missing named features."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(
            feature_dim=3,
            normalize_features=False,
            feature_names=["a", "b", "c"],
        )
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = {"a": 1.0}  # Missing b and c

        tensor = adapter.state_to_tensor(state)

        np.testing.assert_array_equal(tensor, [1.0, 0.0, 0.0])

    def test_empty_features(self):
        """Test with empty feature list."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=3, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = []

        tensor = adapter.state_to_tensor(state)

        np.testing.assert_array_equal(tensor, [0.0, 0.0, 0.0])

    def test_truncation(self):
        """Test feature truncation when too many features."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=3, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = [1.0, 2.0, 3.0, 4.0, 5.0]  # 5 features, dim=3

        tensor = adapter.state_to_tensor(state)

        np.testing.assert_array_equal(tensor, [1.0, 2.0, 3.0])


# =============================================================================
# TextStateAdapter Edge Cases
# =============================================================================


class TestTextStateAdapterEdgeCases:
    """Extended tests for TextStateAdapter."""

    def test_very_long_text(self):
        """Test with text longer than max_sequence_length."""
        from src.framework.mcts.domain_adapters import TextAdapterConfig, TextStateAdapter

        config = TextAdapterConfig(max_sequence_length=10, embedding_dim=10)
        adapter = TextStateAdapter(config=config, action_space_size=100)

        state = MagicMock()
        state.features = {"text": "A" * 1000}  # Very long text

        tensor = adapter.state_to_tensor(state)

        # Should be truncated to embedding_dim
        assert len(tensor) == 10

    def test_empty_text(self):
        """Test with empty text string."""
        from src.framework.mcts.domain_adapters import TextAdapterConfig, TextStateAdapter

        config = TextAdapterConfig(embedding_dim=64)
        adapter = TextStateAdapter(config=config, action_space_size=100)

        state = MagicMock()
        state.features = {"text": ""}

        tensor = adapter.state_to_tensor(state)

        # Should return zeros
        assert len(tensor) == 64
        assert tensor.sum() == 0

    def test_unicode_text(self):
        """Test with Unicode characters."""
        from src.framework.mcts.domain_adapters import TextAdapterConfig, TextStateAdapter

        config = TextAdapterConfig(embedding_dim=32)
        adapter = TextStateAdapter(config=config, action_space_size=100)

        state = MagicMock()
        state.features = {"text": "Hello 世界 🌍"}

        tensor = adapter.state_to_tensor(state)

        # Should handle Unicode without errors
        assert len(tensor) == 32

    def test_precomputed_embedding_wrong_shape(self):
        """Test with pre-computed embedding of different shape."""
        from src.framework.mcts.domain_adapters import TextAdapterConfig, TextStateAdapter

        config = TextAdapterConfig(embedding_dim=64)
        adapter = TextStateAdapter(config=config, action_space_size=100)

        # Embedding with different dimension
        embedding = np.random.randn(128).astype(np.float32)

        state = MagicMock()
        state.features = {"embedding": embedding}

        tensor = adapter.state_to_tensor(state)

        # Should return the embedding as-is (128 elements)
        assert len(tensor) == 128

    def test_no_text_or_embedding(self):
        """Test with features missing both text and embedding."""
        from src.framework.mcts.domain_adapters import TextAdapterConfig, TextStateAdapter

        config = TextAdapterConfig(embedding_dim=32)
        adapter = TextStateAdapter(config=config, action_space_size=100)

        state = MagicMock()
        state.features = {"some_other_key": "value"}

        tensor = adapter.state_to_tensor(state)

        # Should return zeros
        assert len(tensor) == 32
        assert tensor.sum() == 0


# =============================================================================
# BaseDomainAdapter Edge Cases
# =============================================================================


class TestBaseDomainAdapterEdgeCases:
    """Extended tests for BaseDomainAdapter base class behavior."""

    def test_action_to_index_hash_fallback(self):
        """Test hash-based action to index conversion."""
        from src.framework.mcts.domain_adapters import FeatureStateAdapter

        adapter = FeatureStateAdapter(action_space_size=100)

        # Non-numeric action string should use hash
        idx = adapter.action_to_index("move_forward")

        assert 0 <= idx < 100

        # Same action should give same index (caching)
        idx2 = adapter.action_to_index("move_forward")
        assert idx == idx2

    def test_action_to_index_caching(self):
        """Test that action mapping is cached."""
        from src.framework.mcts.domain_adapters import FeatureStateAdapter

        adapter = FeatureStateAdapter(action_space_size=100)

        # First call
        idx1 = adapter.action_to_index("test_action")
        assert "test_action" in adapter._action_to_index

        # Second call should use cache
        idx2 = adapter.action_to_index("test_action")
        assert idx1 == idx2

    def test_index_to_action_unknown(self):
        """Test index to action for unknown index."""
        from src.framework.mcts.domain_adapters import FeatureStateAdapter

        adapter = FeatureStateAdapter(action_space_size=100)

        # Unknown index should return string representation
        action = adapter.index_to_action(42)
        assert action == "42"

    def test_get_action_mask_no_legal_actions(self):
        """Test action mask when no legal actions specified."""
        from src.framework.mcts.domain_adapters import FeatureStateAdapter

        adapter = FeatureStateAdapter(action_space_size=10)

        state = MagicMock()
        state.features = {}  # No legal_actions key

        mask = adapter.get_action_mask(state)

        # Should return None (all actions legal)
        assert mask is None


# =============================================================================
# Factory Function Edge Cases
# =============================================================================


class TestCreateDomainAdapterEdgeCases:
    """Extended tests for create_domain_adapter factory."""

    def test_grid_with_all_options(self):
        """Test grid adapter with all configuration options."""
        from src.framework.mcts.domain_adapters import create_domain_adapter

        adapter = create_domain_adapter(
            "grid",
            board_size=9,
            num_channels=4,
            include_history=True,
            history_length=3,
            player_perspective=False,
        )

        assert adapter.grid_config.board_size == 9
        assert adapter.grid_config.num_channels == 4
        assert adapter.grid_config.include_history is True
        assert adapter.grid_config.history_length == 3
        assert adapter.grid_config.player_perspective is False

    def test_feature_with_all_options(self):
        """Test feature adapter with all configuration options."""
        from src.framework.mcts.domain_adapters import create_domain_adapter

        adapter = create_domain_adapter(
            "feature",
            feature_dim=512,
            normalize_features=False,
            action_space_size=200,
        )

        assert adapter.feature_config.feature_dim == 512
        assert adapter.feature_config.normalize_features is False
        assert adapter.action_space_size == 200

    def test_text_with_all_options(self):
        """Test text adapter with all configuration options."""
        from src.framework.mcts.domain_adapters import create_domain_adapter

        adapter = create_domain_adapter(
            "text",
            max_sequence_length=1024,
            embedding_dim=1536,
            action_space_size=10000,
        )

        assert adapter.text_config.max_sequence_length == 1024
        assert adapter.text_config.embedding_dim == 1536
        assert adapter.action_space_size == 10000

    def test_mixed_params(self):
        """Test with mix of config and non-config parameters."""
        from src.framework.mcts.domain_adapters import create_domain_adapter

        adapter = create_domain_adapter(
            "feature",
            feature_dim=256,  # Goes to config
            action_space_size=50,  # Goes to constructor
            device="cuda",  # Goes to constructor
        )

        assert adapter.feature_config.feature_dim == 256
        assert adapter.action_space_size == 50
        assert adapter.device == "cuda"
