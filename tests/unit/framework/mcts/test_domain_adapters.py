"""
Unit tests for Domain Adapters.

Tests:
- GridStateAdapter for grid-based domains
- FeatureStateAdapter for feature vectors
- TextStateAdapter for text domains
- Factory function
- Action masking and prior conversion
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.mcts,
]


# =============================================================================
# GridStateAdapter Tests
# =============================================================================


class TestGridStateAdapter:
    """Tests for GridStateAdapter."""

    def test_initialization_default(self):
        """Test default initialization."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        assert adapter.action_space_size == 9  # 3x3 board
        assert adapter.state_dim == (3, 3, 3)  # 3 channels, 3x3

    def test_initialization_custom_board(self):
        """Test custom board size."""
        from src.framework.mcts.domain_adapters import GridAdapterConfig, GridStateAdapter

        config = GridAdapterConfig(board_size=8, num_channels=2)
        adapter = GridStateAdapter(config=config)

        assert adapter.action_space_size == 64  # 8x8 board
        assert adapter.state_dim == (2, 8, 8)

    def test_state_to_tensor_basic(self):
        """Test basic state to tensor conversion."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        # Create mock state with a simple board
        state = MagicMock()
        state.features = {
            "board": [
                [1, 0, -1],
                [0, 1, 0],
                [-1, 0, 1],
            ],
            "player": 1,
        }

        tensor = adapter.state_to_tensor(state)

        # Should have shape (3, 3, 3) for 3 channels
        assert tensor.shape == (3, 3, 3)

        # Channel 0: current player pieces (1s)
        assert tensor[0, 0, 0] == 1.0  # Top-left
        assert tensor[0, 1, 1] == 1.0  # Center
        assert tensor[0, 2, 2] == 1.0  # Bottom-right

        # Channel 1: opponent pieces (-1s)
        assert tensor[1, 0, 2] == 1.0
        assert tensor[1, 2, 0] == 1.0

        # Channel 2: empty cells (0s)
        assert tensor[2, 0, 1] == 1.0
        assert tensor[2, 1, 0] == 1.0

    def test_state_to_tensor_flat_board(self):
        """Test conversion with flat board array."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        state = MagicMock()
        state.features = {
            "board": [1, 0, -1, 0, 1, 0, -1, 0, 1],  # Flat
            "player": 1,
        }

        tensor = adapter.state_to_tensor(state)
        assert tensor.shape == (3, 3, 3)

    def test_extract_legal_actions_from_board(self):
        """Test legal action extraction from board state."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        state = MagicMock()
        state.features = {
            "board": [
                [1, 0, -1],
                [0, 1, 0],
                [-1, 0, 0],
            ],
        }

        legal = adapter._extract_legal_actions(state)

        # Empty cells: (0,1), (1,0), (1,2), (2,1), (2,2)
        expected = ["0,1", "1,0", "1,2", "2,1", "2,2"]
        assert set(legal) == set(expected)

    def test_extract_legal_actions_explicit(self):
        """Test legal action extraction from explicit list."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        state = MagicMock()
        state.features = {
            "legal_moves": [(0, 1), (1, 2)],
        }

        legal = adapter._extract_legal_actions(state)
        assert legal == ["(0, 1)", "(1, 2)"]

    def test_action_to_index(self):
        """Test action string to index conversion."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        # Grid action format: "row,col"
        assert adapter.action_to_index("0,0") == 0
        assert adapter.action_to_index("0,1") == 1
        assert adapter.action_to_index("1,0") == 3
        assert adapter.action_to_index("2,2") == 8

    def test_index_to_action(self):
        """Test index to action string conversion."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        assert adapter.index_to_action(0) == "0,0"
        assert adapter.index_to_action(4) == "1,1"
        assert adapter.index_to_action(8) == "2,2"

    def test_get_action_mask(self):
        """Test action mask generation."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        state = MagicMock()
        state.features = {
            "board": [
                [1, 0, -1],
                [0, 0, 0],
                [-1, 0, 1],
            ],
        }

        mask = adapter.get_action_mask(state)

        # Convert to numpy if tensor
        if hasattr(mask, "numpy"):
            mask = mask.numpy()

        assert len(mask) == 9

        # Legal positions: (0,1), (1,0), (1,1), (1,2), (2,1)
        legal_indices = [1, 3, 4, 5, 7]
        for i in range(9):
            if i in legal_indices:
                assert mask[i], f"Index {i} should be legal"
            else:
                assert not mask[i], f"Index {i} should be illegal"

    def test_tensor_to_action_priors(self):
        """Test converting policy output to action priors."""
        from src.framework.mcts.domain_adapters import GridStateAdapter

        adapter = GridStateAdapter()

        # Mock policy output (9 actions, unnormalized)
        policy_output = np.array([0.1, 0.3, 0.1, 0.2, 0.1, 0.1, 0.0, 0.05, 0.05])

        state = MagicMock()
        state.features = {
            "legal_moves": ["0,1", "1,0", "1,1"],  # Only 3 legal
        }

        priors = adapter.tensor_to_action_priors(policy_output, state)

        # Should only contain legal actions
        assert set(priors.keys()) == {"0,1", "1,0", "1,1"}

        # Should be normalized
        total = sum(priors.values())
        assert abs(total - 1.0) < 1e-6


# =============================================================================
# FeatureStateAdapter Tests
# =============================================================================


class TestFeatureStateAdapter:
    """Tests for FeatureStateAdapter."""

    def test_initialization_default(self):
        """Test default initialization."""
        from src.framework.mcts.domain_adapters import FeatureStateAdapter

        adapter = FeatureStateAdapter(action_space_size=50)

        assert adapter.action_space_size == 50
        assert adapter.state_dim == (128,)  # Default feature_dim

    def test_initialization_custom(self):
        """Test custom initialization."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=256, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=100)

        assert adapter.action_space_size == 100
        assert adapter.state_dim == (256,)

    def test_state_to_tensor_dict_features(self):
        """Test conversion from dictionary features."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=5, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = {
            "feature_a": 1.0,
            "feature_b": 2.0,
            "feature_c": 3.0,
        }

        tensor = adapter.state_to_tensor(state)

        assert len(tensor) == 5
        # Values should be 1, 2, 3, 0, 0 (padded)
        assert tensor[0] == 1.0
        assert tensor[1] == 2.0
        assert tensor[2] == 3.0

    def test_state_to_tensor_list_features(self):
        """Test conversion from list features."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=4, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # More than feature_dim

        tensor = adapter.state_to_tensor(state)

        assert len(tensor) == 4
        # Should be truncated
        assert list(tensor) == [1.0, 2.0, 3.0, 4.0]

    def test_normalization(self):
        """Test feature normalization."""
        from src.framework.mcts.domain_adapters import FeatureAdapterConfig, FeatureStateAdapter

        config = FeatureAdapterConfig(feature_dim=4, normalize_features=True)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)

        state = MagicMock()
        state.features = [3.0, 4.0, 0.0, 0.0]  # Norm = 5

        tensor = adapter.state_to_tensor(state)

        # Should be normalized
        norm = np.linalg.norm(tensor)
        assert abs(norm - 1.0) < 1e-6


# =============================================================================
# TextStateAdapter Tests
# =============================================================================


class TestTextStateAdapter:
    """Tests for TextStateAdapter."""

    def test_initialization_default(self):
        """Test default initialization."""
        from src.framework.mcts.domain_adapters import TextStateAdapter

        adapter = TextStateAdapter(action_space_size=1000)

        assert adapter.action_space_size == 1000
        assert adapter.state_dim == (512, 768)  # max_seq_len, embed_dim

    def test_state_to_tensor_with_embedding(self):
        """Test conversion with pre-computed embedding."""
        from src.framework.mcts.domain_adapters import TextAdapterConfig, TextStateAdapter

        config = TextAdapterConfig(embedding_dim=128)
        adapter = TextStateAdapter(config=config, action_space_size=100)

        # Pre-computed embedding
        embedding = np.random.randn(128).astype(np.float32)

        state = MagicMock()
        state.features = {"embedding": embedding}

        tensor = adapter.state_to_tensor(state)

        # Should return the embedding directly
        np.testing.assert_array_almost_equal(tensor, embedding)

    def test_state_to_tensor_with_text(self):
        """Test conversion from raw text (fallback)."""
        from src.framework.mcts.domain_adapters import TextAdapterConfig, TextStateAdapter

        config = TextAdapterConfig(embedding_dim=64)
        adapter = TextStateAdapter(config=config, action_space_size=100)

        state = MagicMock()
        state.features = {"text": "Hello world"}

        tensor = adapter.state_to_tensor(state)

        # Should have embedding dimension
        assert len(tensor) == 64
        # First characters should be encoded
        assert tensor[0] == ord("H") / 255.0
        assert tensor[1] == ord("e") / 255.0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateDomainAdapter:
    """Tests for create_domain_adapter factory."""

    def test_create_grid_adapter(self):
        """Test creating grid adapter."""
        from src.framework.mcts.domain_adapters import GridStateAdapter, create_domain_adapter

        adapter = create_domain_adapter("grid", board_size=8)

        assert isinstance(adapter, GridStateAdapter)
        assert adapter.action_space_size == 64

    def test_create_feature_adapter(self):
        """Test creating feature adapter."""
        from src.framework.mcts.domain_adapters import FeatureStateAdapter, create_domain_adapter

        adapter = create_domain_adapter("feature", feature_dim=256, action_space_size=50)

        assert isinstance(adapter, FeatureStateAdapter)
        assert adapter.action_space_size == 50

    def test_create_text_adapter(self):
        """Test creating text adapter."""
        from src.framework.mcts.domain_adapters import TextStateAdapter, create_domain_adapter

        adapter = create_domain_adapter("text", embedding_dim=512, action_space_size=5000)

        assert isinstance(adapter, TextStateAdapter)
        assert adapter.action_space_size == 5000

    def test_unknown_domain_type(self):
        """Test error on unknown domain type."""
        from src.framework.mcts.domain_adapters import create_domain_adapter

        with pytest.raises(ValueError, match="Unknown domain type"):
            create_domain_adapter("unknown")


# =============================================================================
# Integration with MCTS Module Tests
# =============================================================================


class TestMCTSModuleIntegration:
    """Tests for integration with MCTS module exports."""

    def test_exports_from_mcts_module(self):
        """Test that adapters are exported from MCTS module."""
        from src.framework.mcts import (
            BaseDomainAdapter,
            FeatureStateAdapter,
            GridStateAdapter,
            TextStateAdapter,
            create_domain_adapter,
        )

        # All should be importable
        assert BaseDomainAdapter is not None
        assert GridStateAdapter is not None
        assert FeatureStateAdapter is not None
        assert TextStateAdapter is not None
        assert create_domain_adapter is not None

    def test_neural_policies_exports(self):
        """Test that neural policies are exported from MCTS module."""
        from src.framework.mcts import (
            NeuralPolicyConfig,
            NeuralRolloutPolicy,
            PriorsManager,
            create_neural_rollout_policy,
            is_torch_available,
            puct,
            puct_with_virtual_loss,
            select_child_puct,
        )

        # All should be importable
        assert puct is not None
        assert puct_with_virtual_loss is not None
        assert NeuralPolicyConfig is not None
        assert NeuralRolloutPolicy is not None
        assert PriorsManager is not None
        assert create_neural_rollout_policy is not None
        assert select_child_puct is not None
        assert is_torch_available is not None
