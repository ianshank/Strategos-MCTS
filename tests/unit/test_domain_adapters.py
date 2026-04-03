"""
Unit tests for src/framework/mcts/domain_adapters.py

Tests domain adapter behavior: state conversion, action mapping,
action masks, factory function, and edge cases for Grid, Feature,
and Text adapters.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Set environment variables before importing modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

from src.framework.mcts.core import MCTSState
from src.framework.mcts.domain_adapters import (
    FeatureAdapterConfig,
    FeatureStateAdapter,
    GridAdapterConfig,
    GridStateAdapter,
    TextAdapterConfig,
    TextStateAdapter,
    create_domain_adapter,
)

pytestmark = pytest.mark.unit


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _make_state(state_id: str = "s1", **features) -> MCTSState:
    return MCTSState(state_id=state_id, features=features)


def _make_grid_state(
    board: list | None = None,
    player: int = 1,
    board_size: int = 3,
    **extra,
) -> MCTSState:
    if board is None:
        board = [0] * (board_size * board_size)
    return MCTSState(state_id="grid", features={"board": board, "player": player, **extra})


# ────────────────────────────────────────────────────────────────────
# BaseDomainAdapter (via GridStateAdapter as concrete subclass)
# ────────────────────────────────────────────────────────────────────


class TestBaseDomainAdapter:
    """Tests for shared BaseDomainAdapter functionality."""

    def test_action_to_index_integer_string(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        # "5" can be parsed as int
        assert adapter.action_to_index("5") == 5

    def test_action_to_index_hash_fallback(self):
        """Non-integer, non-grid action falls through to hash-based index."""
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=100,
        )
        idx = adapter.action_to_index("some_action")
        assert 0 <= idx < 100

    def test_action_to_index_caching(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=100,
        )
        idx1 = adapter.action_to_index("some_action")
        idx2 = adapter.action_to_index("some_action")
        assert idx1 == idx2

    def test_index_to_action_known(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=100,
        )
        adapter.action_to_index("42")  # populates reverse map
        assert adapter.index_to_action(42) == "42"

    def test_index_to_action_unknown(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=100,
        )
        # Unknown index returns str(index)
        assert adapter.index_to_action(99) == "99"

    def test_state_dim_tuple_conversion(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=8),
            action_space_size=10,
        )
        assert adapter.state_dim == (8,)

    def test_get_action_mask_no_legal_actions(self):
        """When no legal actions, returns None."""
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=10,
        )
        state = _make_state()  # no legal_actions key
        mask = adapter.get_action_mask(state)
        assert mask is None

    def test_get_action_mask_with_legal_actions(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=10,
        )
        state = _make_state(legal_actions=["0", "3", "7"])
        mask = adapter.get_action_mask(state)
        # Convert to numpy for comparison
        mask_np = np.asarray(mask).astype(bool)
        assert mask_np[0] is np.True_
        assert mask_np[3] is np.True_
        assert mask_np[7] is np.True_
        assert mask_np[1] is np.False_

    def test_tensor_to_action_priors_no_legal_actions(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=4,
        )
        state = _make_state()  # no legal_actions
        policy = np.array([0.1, 0.2, 0.3, 0.4])
        priors = adapter.tensor_to_action_priors(policy, state)
        assert len(priors) == 4
        assert abs(sum(priors.values()) - 1.0) < 1e-5

    def test_tensor_to_action_priors_with_legal_actions(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=4,
        )
        state = _make_state(legal_actions=["0", "2"])
        policy = np.array([0.3, 0.1, 0.5, 0.1])
        priors = adapter.tensor_to_action_priors(policy, state)
        assert set(priors.keys()) == {"0", "2"}
        # Renormalized
        assert abs(sum(priors.values()) - 1.0) < 1e-5

    def test_tensor_to_action_priors_all_zero_uniform(self):
        adapter = FeatureStateAdapter(
            FeatureAdapterConfig(feature_dim=4),
            action_space_size=4,
        )
        state = _make_state(legal_actions=["0", "2"])
        policy = np.array([0.0, 0.0, 0.0, 0.0])
        priors = adapter.tensor_to_action_priors(policy, state)
        # Should be uniform over legal actions
        assert abs(priors["0"] - 0.5) < 1e-5
        assert abs(priors["2"] - 0.5) < 1e-5


# ────────────────────────────────────────────────────────────────────
# GridStateAdapter
# ────────────────────────────────────────────────────────────────────


class TestGridStateAdapter:
    """Tests for GridStateAdapter."""

    def test_default_config(self):
        adapter = GridStateAdapter()
        assert adapter.action_space_size == 9  # 3x3
        assert adapter.state_dim == (3, 3, 3)  # channels, H, W

    def test_custom_board_size(self):
        config = GridAdapterConfig(board_size=8, num_channels=3)
        adapter = GridStateAdapter(config=config)
        assert adapter.action_space_size == 64
        assert adapter.state_dim == (3, 8, 8)

    def test_state_to_tensor_flat_board(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        board = [1, 0, -1, 0, 1, 0, -1, 0, 0]
        state = _make_grid_state(board=board, player=1)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (3, 3, 3)
        # Channel 0: current player (1) pieces
        assert tensor[0, 0, 0] == 1.0  # board[0,0] == 1 == player
        # Channel 1: opponent (-1) pieces
        assert tensor[1, 0, 2] == 1.0  # board[0,2] == -1 == -player
        # Channel 2: empty
        assert tensor[2, 0, 1] == 1.0  # board[0,1] == 0

    def test_state_to_tensor_2d_board(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        board = [[1, 0, -1], [0, 1, 0], [-1, 0, 0]]
        state = _make_grid_state(board=board, player=1)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (3, 3, 3)

    def test_state_to_tensor_absolute_representation(self):
        config = GridAdapterConfig(board_size=3, player_perspective=False)
        adapter = GridStateAdapter(config=config)
        board = [1, 0, -1, 0, 0, 0, 0, 0, 0]
        state = _make_grid_state(board=board, player=1)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (3, 3, 3)
        # Channel 0: player 1 pieces (absolute)
        assert tensor[0, 0, 0] == 1.0
        # Channel 1: player -1 pieces
        assert tensor[1, 0, 2] == 1.0

    def test_state_to_tensor_with_history(self):
        config = GridAdapterConfig(board_size=3, num_channels=3, include_history=True, history_length=3)
        adapter = GridStateAdapter(config=config)
        board = [0] * 9
        history = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]]
        state = _make_grid_state(board=board, player=1, history=history)
        tensor = np.asarray(adapter.state_to_tensor(state))
        # channels = 3 * 3 (history_length) = 9; first 3 for current, then 2*2=4 for history, then 2*0 padded
        assert tensor.shape[1] == 3
        assert tensor.shape[2] == 3

    def test_state_to_tensor_history_padding(self):
        """History shorter than history_length pads with zeros."""
        config = GridAdapterConfig(board_size=3, include_history=True, history_length=4)
        adapter = GridStateAdapter(config=config)
        board = [0] * 9
        state = _make_grid_state(board=board, player=1, history=[])
        tensor = np.asarray(adapter.state_to_tensor(state))
        # Should not crash; padded with zeros
        assert tensor.shape[1] == 3

    def test_extract_legal_actions_from_board(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        board = [1, 0, -1, 0, 1, -1, 1, -1, 0]
        state = _make_grid_state(board=board, player=1)
        legal = adapter._extract_legal_actions(state)
        # Empty cells: (0,1), (1,0), (2,2)
        assert "0,1" in legal
        assert "1,0" in legal
        assert "2,2" in legal
        assert len(legal) == 3

    def test_extract_legal_actions_explicit(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        state = _make_grid_state(board=[0] * 9, player=1, legal_moves=[0, 4, 8])
        legal = adapter._extract_legal_actions(state)
        assert legal == ["0", "4", "8"]

    def test_action_to_index_grid_format(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        assert adapter.action_to_index("0,0") == 0
        assert adapter.action_to_index("1,2") == 5
        assert adapter.action_to_index("2,2") == 8

    def test_action_to_index_integer_fallback(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        # No comma, falls through to parent
        assert adapter.action_to_index("5") == 5

    def test_index_to_action(self):
        adapter = GridStateAdapter(GridAdapterConfig(board_size=3))
        assert adapter.index_to_action(0) == "0,0"
        assert adapter.index_to_action(4) == "1,1"
        assert adapter.index_to_action(8) == "2,2"


# ────────────────────────────────────────────────────────────────────
# FeatureStateAdapter
# ────────────────────────────────────────────────────────────────────


class TestFeatureStateAdapter:
    """Tests for FeatureStateAdapter."""

    def test_default_config(self):
        adapter = FeatureStateAdapter()
        assert adapter.state_dim == (128,)
        assert adapter.action_space_size == 100

    def test_state_to_tensor_from_dict(self):
        config = FeatureAdapterConfig(feature_dim=4, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = _make_state(a=1.0, b=2.0, c=3.0, d=4.0)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (4,)

    def test_state_to_tensor_from_dict_with_feature_names(self):
        config = FeatureAdapterConfig(
            feature_dim=3,
            feature_names=["x", "y", "z"],
            normalize_features=False,
        )
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = _make_state(x=1.0, y=2.0, z=3.0, extra=99.0)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (3,)
        np.testing.assert_array_almost_equal(tensor, [1.0, 2.0, 3.0])

    def test_state_to_tensor_from_list(self):
        config = FeatureAdapterConfig(feature_dim=3, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = MCTSState(state_id="s1", features=[1.0, 2.0, 3.0])
        tensor = np.asarray(adapter.state_to_tensor(state))
        np.testing.assert_array_almost_equal(tensor, [1.0, 2.0, 3.0])

    def test_state_to_tensor_padding(self):
        config = FeatureAdapterConfig(feature_dim=5, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = _make_state(a=1.0, b=2.0)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (5,)
        # Last 3 values padded with 0
        assert tensor[2] == 0.0

    def test_state_to_tensor_truncation(self):
        config = FeatureAdapterConfig(feature_dim=2, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = _make_state(a=1.0, b=2.0, c=3.0, d=4.0)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (2,)

    def test_state_to_tensor_normalization(self):
        config = FeatureAdapterConfig(feature_dim=3, normalize_features=True)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = _make_state(a=3.0, b=4.0, c=0.0)
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert abs(np.linalg.norm(tensor) - 1.0) < 1e-5

    def test_state_to_tensor_zero_vector_normalization(self):
        """Zero vector should not cause division by zero."""
        config = FeatureAdapterConfig(feature_dim=3, normalize_features=True)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = _make_state(a=0.0, b=0.0, c=0.0)
        tensor = np.asarray(adapter.state_to_tensor(state))
        np.testing.assert_array_equal(tensor, [0.0, 0.0, 0.0])

    def test_state_to_tensor_non_dict_non_list_fallback(self):
        """Non-dict, non-list features fall back to zeros."""
        config = FeatureAdapterConfig(feature_dim=3, normalize_features=False)
        adapter = FeatureStateAdapter(config=config, action_space_size=10)
        state = MCTSState(state_id="s1", features="not_a_dict_or_list")
        tensor = np.asarray(adapter.state_to_tensor(state))
        np.testing.assert_array_equal(tensor, [0.0, 0.0, 0.0])

    def test_extract_legal_actions_present(self):
        adapter = FeatureStateAdapter(FeatureAdapterConfig(feature_dim=4), action_space_size=10)
        state = _make_state(legal_actions=["a", "b"])
        assert adapter._extract_legal_actions(state) == ["a", "b"]

    def test_extract_legal_actions_absent(self):
        adapter = FeatureStateAdapter(FeatureAdapterConfig(feature_dim=4), action_space_size=10)
        state = _make_state(x=1.0)
        assert adapter._extract_legal_actions(state) == []


# ────────────────────────────────────────────────────────────────────
# TextStateAdapter
# ────────────────────────────────────────────────────────────────────


class TestTextStateAdapter:
    """Tests for TextStateAdapter."""

    def test_default_config(self):
        adapter = TextStateAdapter()
        assert adapter.action_space_size == 1000
        assert adapter.state_dim == (512, 768)

    def test_state_to_tensor_with_embedding(self):
        adapter = TextStateAdapter(TextAdapterConfig(embedding_dim=4))
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        state = _make_state(embedding=embedding)
        tensor = np.asarray(adapter.state_to_tensor(state))
        np.testing.assert_array_almost_equal(tensor, embedding)

    def test_state_to_tensor_with_list_embedding(self):
        adapter = TextStateAdapter(TextAdapterConfig(embedding_dim=4))
        state = _make_state(embedding=[0.1, 0.2, 0.3, 0.4])
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (4,)

    def test_state_to_tensor_text_fallback(self):
        config = TextAdapterConfig(embedding_dim=10, max_sequence_length=5)
        adapter = TextStateAdapter(config=config)
        state = _make_state(text="hello")
        tensor = np.asarray(adapter.state_to_tensor(state))
        assert tensor.shape == (10,)
        # First chars should be non-zero
        assert tensor[0] > 0

    def test_state_to_tensor_empty_text(self):
        config = TextAdapterConfig(embedding_dim=10)
        adapter = TextStateAdapter(config=config)
        state = _make_state(text="")
        tensor = np.asarray(adapter.state_to_tensor(state))
        np.testing.assert_array_equal(tensor, np.zeros(10))

    def test_state_to_tensor_no_text_no_embedding(self):
        config = TextAdapterConfig(embedding_dim=10)
        adapter = TextStateAdapter(config=config)
        state = _make_state()  # no text, no embedding
        tensor = np.asarray(adapter.state_to_tensor(state))
        np.testing.assert_array_equal(tensor, np.zeros(10))

    def test_extract_legal_actions_present(self):
        adapter = TextStateAdapter(TextAdapterConfig(embedding_dim=4))
        state = _make_state(legal_actions=["action1", "action2"])
        assert adapter._extract_legal_actions(state) == ["action1", "action2"]

    def test_extract_legal_actions_absent(self):
        adapter = TextStateAdapter(TextAdapterConfig(embedding_dim=4))
        state = _make_state(text="hello")
        assert adapter._extract_legal_actions(state) == []


# ────────────────────────────────────────────────────────────────────
# create_domain_adapter factory
# ────────────────────────────────────────────────────────────────────


class TestCreateDomainAdapter:
    """Tests for create_domain_adapter factory."""

    def test_create_grid_adapter(self):
        adapter = create_domain_adapter("grid", board_size=5)
        assert isinstance(adapter, GridStateAdapter)
        assert adapter.grid_config.board_size == 5

    def test_create_feature_adapter(self):
        adapter = create_domain_adapter("feature", feature_dim=64, action_space_size=50)
        assert isinstance(adapter, FeatureStateAdapter)
        assert adapter.feature_config.feature_dim == 64
        assert adapter.action_space_size == 50

    def test_create_text_adapter(self):
        adapter = create_domain_adapter("text", embedding_dim=256)
        assert isinstance(adapter, TextStateAdapter)
        assert adapter.text_config.embedding_dim == 256

    def test_create_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain type"):
            create_domain_adapter("unknown")

    def test_create_with_no_config_params(self):
        adapter = create_domain_adapter("grid")
        assert isinstance(adapter, GridStateAdapter)
        assert adapter.grid_config.board_size == 3  # default

    def test_create_with_device_kwarg(self):
        adapter = create_domain_adapter("grid", device="cpu")
        assert adapter.device == "cpu"

    def test_create_feature_with_mixed_params(self):
        """Config params go to config, others go to adapter constructor."""
        adapter = create_domain_adapter(
            "feature",
            feature_dim=32,
            normalize_features=False,
            action_space_size=20,
        )
        assert isinstance(adapter, FeatureStateAdapter)
        assert adapter.feature_config.feature_dim == 32
        assert adapter.feature_config.normalize_features is False
        assert adapter.action_space_size == 20
