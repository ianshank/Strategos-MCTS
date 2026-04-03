"""
Tests for neural-guided MCTS module.

Tests GameState, NeuralMCTSNode, and NeuralMCTS classes.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for neural MCTS")

from src.framework.mcts.neural_mcts import GameState, NeuralMCTS, NeuralMCTSNode


class SimpleGameState(GameState):
    """Simple concrete GameState for testing."""

    def __init__(self, board=None, terminal=False, reward=0.0):
        self._board = board or [0] * 9
        self._terminal = terminal
        self._reward = reward

    def get_legal_actions(self):
        return [i for i, v in enumerate(self._board) if v == 0]

    def apply_action(self, action):
        new_board = self._board.copy()
        new_board[action] = 1
        return SimpleGameState(new_board)

    def is_terminal(self):
        return self._terminal

    def get_reward(self, player=1):
        return self._reward

    def to_tensor(self):
        return torch.tensor(self._board, dtype=torch.float32)

    def get_hash(self):
        return str(self._board)

    def action_to_index(self, action):
        return int(action)


@pytest.mark.unit
class TestGameState:
    """Tests for GameState base class."""

    def test_get_canonical_form_default(self):
        state = SimpleGameState()
        canonical = state.get_canonical_form(1)
        assert canonical is state  # Default returns self

    def test_action_to_index_grid(self):
        state = GameState()
        # Test default grid-based mapping "row,col"
        assert state.action_to_index("0,0") == 0
        assert state.action_to_index("1,1") == 4
        assert state.action_to_index("2,2") == 8

    def test_action_to_index_integer(self):
        state = GameState()
        assert state.action_to_index(5) == 5


@pytest.mark.unit
class TestNeuralMCTSNode:
    """Tests for NeuralMCTSNode."""

    def test_init(self):
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state, prior=0.5)
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.5
        assert node.is_expanded is False
        assert node.virtual_loss == 0.0

    def test_value_no_visits(self):
        node = NeuralMCTSNode(state=SimpleGameState(), prior=0.5)
        assert node.value == 0.0

    def test_value_with_visits(self):
        node = NeuralMCTSNode(state=SimpleGameState(), prior=0.5)
        node.visit_count = 10
        node.value_sum = 7.0
        assert node.value == pytest.approx(0.7)

    def test_update(self):
        node = NeuralMCTSNode(state=SimpleGameState(), prior=0.5)
        node.update(0.8)
        assert node.visit_count == 1
        assert node.value_sum == 0.8

    def test_expand(self):
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state, prior=0.5)
        actions = [0, 1, 2]
        priors = np.array([0.5, 0.3, 0.2])
        node.expand(priors, actions)

        assert node.is_expanded is True
        assert len(node.children) == 3
        assert node.children[0].prior == 0.5
        assert node.children[1].prior == 0.3

    def test_expand_no_duplicate_children(self):
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state, prior=0.5)
        actions = [0, 1]
        priors = np.array([0.5, 0.5])
        node.expand(priors, actions)
        # Expanding again with same actions should not add duplicates
        node.expand(priors, actions)
        assert len(node.children) == 2

    def test_select_child(self):
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state, prior=0.5)
        node.visit_count = 10
        actions = [0, 1]
        priors = np.array([0.8, 0.2])
        node.expand(priors, actions)

        action, child = node.select_child(c_puct=1.0)
        assert action is not None
        assert child is not None
        assert action in [0, 1]

    def test_select_child_empty(self):
        node = NeuralMCTSNode(state=SimpleGameState(), prior=0.5)
        action, child = node.select_child(c_puct=1.0)
        assert action is None
        assert child is None

    def test_virtual_loss(self):
        node = NeuralMCTSNode(state=SimpleGameState(), prior=0.5)
        node.add_virtual_loss(3.0)
        assert node.virtual_loss == 3.0
        node.revert_virtual_loss(3.0)
        assert node.virtual_loss == 0.0

    def test_get_action_probs_empty(self):
        node = NeuralMCTSNode(state=SimpleGameState(), prior=0.5)
        assert node.get_action_probs() == {}

    def test_get_action_probs_temperature_0(self):
        """Temperature 0 = deterministic (argmax)."""
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state, prior=0.5)
        actions = [0, 1, 2]
        priors = np.array([0.33, 0.33, 0.34])
        node.expand(priors, actions)

        # Give action 1 more visits
        node.children[1].visit_count = 10
        node.children[0].visit_count = 5
        node.children[2].visit_count = 3

        probs = node.get_action_probs(temperature=0)
        assert probs[1] == 1.0  # Most visited
        assert probs[0] == 0.0
        assert probs[2] == 0.0

    def test_get_action_probs_temperature_1(self):
        """Temperature 1 = proportional to visits."""
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state, prior=0.5)
        actions = [0, 1]
        priors = np.array([0.5, 0.5])
        node.expand(priors, actions)

        node.children[0].visit_count = 30
        node.children[1].visit_count = 70

        probs = node.get_action_probs(temperature=1.0)
        assert probs[0] == pytest.approx(0.3)
        assert probs[1] == pytest.approx(0.7)

    def test_get_action_probs_temperature_higher(self):
        """Higher temperature = more uniform."""
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state, prior=0.5)
        actions = [0, 1]
        priors = np.array([0.5, 0.5])
        node.expand(priors, actions)

        node.children[0].visit_count = 10
        node.children[1].visit_count = 90

        probs_t1 = node.get_action_probs(temperature=1.0)
        probs_t5 = node.get_action_probs(temperature=5.0)

        # Higher temp should be more uniform
        assert abs(probs_t5[0] - probs_t5[1]) < abs(probs_t1[0] - probs_t1[1])

    def test_is_terminal(self):
        node = NeuralMCTSNode(state=SimpleGameState(terminal=True), prior=0.5)
        assert node.is_terminal is True


@pytest.mark.unit
class TestNeuralMCTS:
    """Tests for NeuralMCTS class."""

    def _make_network(self):
        """Create a mock policy-value network."""
        net = MagicMock()
        net.eval = MagicMock(return_value=net)
        # Return policy logits and value
        net.return_value = (torch.randn(1, 9), torch.tensor([[0.5]]))
        return net

    def _make_config(self):
        from src.training.system_config import MCTSConfig
        return MCTSConfig()

    def test_init(self):
        net = self._make_network()
        config = self._make_config()
        mcts = NeuralMCTS(policy_value_network=net, config=config)
        assert mcts.cache_hits == 0
        assert mcts.cache_misses == 0

    def test_add_dirichlet_noise(self):
        net = self._make_network()
        config = self._make_config()
        mcts = NeuralMCTS(policy_value_network=net, config=config)

        policy = np.array([0.5, 0.3, 0.2])
        noised = mcts.add_dirichlet_noise(policy)
        assert len(noised) == 3
        assert np.sum(noised) == pytest.approx(1.0, abs=0.01)
        # Noise should change the distribution
        assert not np.allclose(noised, policy)
