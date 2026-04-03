"""
Extended tests for neural-guided MCTS module.

Covers uncovered paths in NeuralMCTSNode, NeuralMCTS, SelfPlayCollector, and MCTSExample.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for neural MCTS")

from src.framework.mcts.neural_mcts import (
    GameState,
    MCTSExample,
    NeuralMCTS,
    NeuralMCTSNode,
    SelfPlayCollector,
)
from src.training.system_config import MCTSConfig

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class SimpleGameState(GameState):
    """Concrete GameState for testing."""

    def __init__(self, board=None, terminal=False, reward=0.0, legal_actions=None):
        self._board = board or [0] * 9
        self._terminal = terminal
        self._reward = reward
        self._legal_actions = legal_actions

    def get_legal_actions(self):
        if self._legal_actions is not None:
            return list(self._legal_actions)
        return [i for i, v in enumerate(self._board) if v == 0]

    def apply_action(self, action):
        new_board = self._board.copy()
        new_board[action] = 1
        remaining = [i for i, v in enumerate(new_board) if v == 0]
        # Become terminal when no actions left
        return SimpleGameState(new_board, terminal=len(remaining) == 0, reward=self._reward)

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


class FailingActionIndexState(SimpleGameState):
    """State where action_to_index raises an error."""

    def action_to_index(self, action):
        raise ValueError("bad action")


def _make_network(action_size=9):
    """Create a mock policy-value network."""
    net = MagicMock()
    net.eval = MagicMock(return_value=net)
    net.return_value = (torch.randn(1, action_size), torch.tensor([[0.5]]))
    return net


def _make_config(**overrides):
    defaults = {"num_simulations": 4, "c_puct": 1.25, "virtual_loss": 3.0}
    defaults.update(overrides)
    return MCTSConfig(**defaults)


# ---------------------------------------------------------------------------
# NeuralMCTSNode extended tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNeuralMCTSNodeExt:
    """Extended coverage for NeuralMCTSNode."""

    def test_select_child_prefers_high_prior_unvisited(self):
        """Among unvisited children, higher prior wins."""
        state = SimpleGameState()
        root = NeuralMCTSNode(state=state)
        root.visit_count = 1
        actions = [0, 1, 2]
        priors = np.array([0.1, 0.7, 0.2])
        root.expand(priors, actions)

        action, child = root.select_child(c_puct=1.0)
        assert action == 1
        assert child.prior == 0.7

    def test_select_child_considers_virtual_loss(self):
        """Virtual loss should reduce the attractiveness of a child."""
        state = SimpleGameState()
        root = NeuralMCTSNode(state=state)
        root.visit_count = 10
        actions = [0, 1]
        priors = np.array([0.5, 0.5])
        root.expand(priors, actions)

        # Give both children equal visits and value
        for c in root.children.values():
            c.visit_count = 3
            c.value_sum = 1.5

        # Add heavy virtual loss to child 0
        root.children[0].add_virtual_loss(100.0)

        action, child = root.select_child(c_puct=1.0)
        assert action == 1, "Should avoid child with high virtual loss"

    def test_get_action_probs_temperature_0_tie(self):
        """Temperature 0 with tied visit counts distributes uniformly among best."""
        state = SimpleGameState()
        node = NeuralMCTSNode(state=state)
        actions = [0, 1, 2]
        priors = np.array([0.33, 0.33, 0.34])
        node.expand(priors, actions)

        # Two children tied for most visits
        node.children[0].visit_count = 10
        node.children[1].visit_count = 10
        node.children[2].visit_count = 3

        probs = node.get_action_probs(temperature=0)
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)
        assert probs[2] == 0.0

    def test_expand_creates_children_with_correct_states(self):
        """Expanded children should have states from apply_action."""
        state = SimpleGameState(board=[0, 0, 0, 1, 1, 1, 1, 1, 1])
        node = NeuralMCTSNode(state=state)
        actions = [0, 1, 2]
        priors = np.array([0.4, 0.3, 0.3])
        node.expand(priors, actions)

        for action in actions:
            child = node.children[action]
            assert child.parent is node
            assert child.action == action
            # child state should have the action applied
            assert child.state._board[action] == 1

    def test_node_with_terminal_state(self):
        """Node created with terminal state should have is_terminal True."""
        state = SimpleGameState(terminal=True)
        node = NeuralMCTSNode(state=state, prior=0.3)
        assert node.is_terminal is True

    def test_multiple_updates(self):
        """Multiple updates accumulate correctly."""
        node = NeuralMCTSNode(state=SimpleGameState())
        values = [0.5, 0.8, -0.3, 1.0]
        for v in values:
            node.update(v)
        assert node.visit_count == 4
        assert node.value_sum == pytest.approx(sum(values))
        assert node.value == pytest.approx(sum(values) / 4)


# ---------------------------------------------------------------------------
# NeuralMCTS extended tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNeuralMCTSExt:
    """Extended coverage for NeuralMCTS."""

    @pytest.mark.asyncio
    async def test_evaluate_state_caches_result(self):
        """Second call with same state should hit cache."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        state = SimpleGameState()
        policy1, value1 = await mcts.evaluate_state(state, add_noise=False)
        policy2, value2 = await mcts.evaluate_state(state, add_noise=False)

        assert mcts.cache_hits == 1
        assert mcts.cache_misses == 1
        np.testing.assert_array_equal(policy1, policy2)
        assert value1 == value2

    @pytest.mark.asyncio
    async def test_evaluate_state_no_cache_with_noise(self):
        """Noised evaluations should not be cached."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        state = SimpleGameState()
        await mcts.evaluate_state(state, add_noise=True)
        await mcts.evaluate_state(state, add_noise=True)

        # Both should be misses since noised results aren't cached
        assert mcts.cache_hits == 0
        assert mcts.cache_misses == 2

    @pytest.mark.asyncio
    async def test_evaluate_state_no_legal_actions(self):
        """State with no legal actions should return empty policy."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        state = SimpleGameState(legal_actions=[])
        policy, value = await mcts.evaluate_state(state)
        assert len(policy) == 0
        assert value == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_state_action_to_index_fallback(self):
        """When action_to_index fails, should fall back to sequential mapping."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        state = FailingActionIndexState(board=[0, 0, 1, 1, 1, 1, 1, 1, 1])
        policy, value = await mcts.evaluate_state(state, add_noise=False)

        # Should still return valid probabilities
        assert len(policy) == 2  # two legal actions (indices 0, 1)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)

    def test_select_action_deterministic(self):
        """Deterministic select_action picks highest probability."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        probs = {"a": 0.1, "b": 0.6, "c": 0.3}
        action = mcts.select_action(probs, deterministic=True)
        assert action == "b"

    def test_select_action_temperature_zero(self):
        """Temperature 0 also triggers deterministic selection."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        probs = {"x": 0.2, "y": 0.8}
        action = mcts.select_action(probs, temperature=0)
        assert action == "y"

    def test_select_action_empty_probs(self):
        """Empty action_probs returns None."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        assert mcts.select_action({}) is None

    def test_select_action_stochastic(self):
        """Stochastic selection returns a valid action."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        probs = {"a": 0.5, "b": 0.5}
        np.random.seed(42)
        action = mcts.select_action(probs, temperature=1.0)
        assert action in ("a", "b")

    def test_clear_cache(self):
        """clear_cache resets all cache state."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        mcts.cache["key"] = (np.array([0.5]), 0.5)
        mcts.cache_hits = 5
        mcts.cache_misses = 3

        mcts.clear_cache()
        assert len(mcts.cache) == 0
        assert mcts.cache_hits == 0
        assert mcts.cache_misses == 0

    def test_get_cache_stats(self):
        """get_cache_stats returns correct statistics."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        mcts.cache["k1"] = (np.array([0.5]), 0.5)
        mcts.cache["k2"] = (np.array([0.5]), 0.5)
        mcts.cache_hits = 8
        mcts.cache_misses = 2

        stats = mcts.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["cache_hits"] == 8
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] == pytest.approx(0.8)

    def test_get_cache_stats_no_lookups(self):
        """hit_rate should be 0 when no lookups."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        stats = mcts.get_cache_stats()
        assert stats["hit_rate"] == 0.0

    def test_add_dirichlet_noise_custom_params(self):
        """Dirichlet noise with custom epsilon/alpha."""
        net = _make_network()
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        policy = np.array([0.5, 0.3, 0.2])
        noised = mcts.add_dirichlet_noise(policy, epsilon=0.5, alpha=1.0)
        assert len(noised) == 3
        assert noised.sum() == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_search_returns_action_probs_and_root(self):
        """search() returns valid action probabilities and root node."""
        net = _make_network()
        config = _make_config(num_simulations=2)
        mcts = NeuralMCTS(net, config)

        state = SimpleGameState()
        action_probs, root = await mcts.search(state, num_simulations=2, temperature=1.0)

        assert isinstance(action_probs, dict)
        assert len(action_probs) > 0
        assert root.is_expanded
        # Probabilities should sum to ~1
        assert sum(action_probs.values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_search_without_root_noise(self):
        """search() without root noise still works."""
        net = _make_network()
        config = _make_config(num_simulations=2)
        mcts = NeuralMCTS(net, config)

        state = SimpleGameState()
        action_probs, root = await mcts.search(state, num_simulations=2, add_root_noise=False)
        assert isinstance(action_probs, dict)

    @pytest.mark.asyncio
    async def test_simulate_terminal_leaf(self):
        """_simulate handles terminal leaf nodes (uses get_reward)."""
        net = _make_network()
        config = _make_config(num_simulations=1)
        mcts = NeuralMCTS(net, config)

        # Create a state with only one legal action that leads to terminal
        board = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        state = SimpleGameState(board=board, reward=1.0)
        action_probs, root = await mcts.search(state, num_simulations=1, temperature=1.0)

        # The only child should have been visited
        assert 8 in root.children
        assert root.children[8].visit_count >= 1

    @pytest.mark.asyncio
    async def test_search_uses_config_num_simulations(self):
        """search() defaults to config.num_simulations when not specified."""
        net = _make_network()
        config = _make_config(num_simulations=3)
        mcts = NeuralMCTS(net, config)

        state = SimpleGameState()
        action_probs, root = await mcts.search(state, temperature=1.0)

        # Root should have been visited num_simulations + 1 times (root expand counts)
        total_child_visits = sum(c.visit_count for c in root.children.values())
        assert total_child_visits == 3

    @pytest.mark.asyncio
    async def test_evaluate_state_uniform_fallback(self):
        """When softmax produces zero probs, falls back to uniform."""
        net = MagicMock()
        # Return -inf logits so exp produces zeros
        net.return_value = (torch.full((1, 9), -1e10), torch.tensor([[0.0]]))
        config = _make_config()
        mcts = NeuralMCTS(net, config)

        state = SimpleGameState(board=[0, 0, 1, 1, 1, 1, 1, 1, 1])
        policy, value = await mcts.evaluate_state(state, add_noise=False)
        # Should still get valid probabilities
        assert len(policy) == 2
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# MCTSExample tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMCTSExample:
    """Tests for MCTSExample dataclass."""

    def test_create_example(self):
        state_tensor = torch.zeros(9)
        policy = np.array([0.5, 0.3, 0.2])
        ex = MCTSExample(state=state_tensor, policy_target=policy, value_target=1.0, player=1)
        assert ex.player == 1
        assert ex.value_target == 1.0
        assert torch.equal(ex.state, state_tensor)
        np.testing.assert_array_equal(ex.policy_target, policy)


# ---------------------------------------------------------------------------
# SelfPlayCollector tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSelfPlayCollector:
    """Tests for SelfPlayCollector."""

    @pytest.mark.asyncio
    async def test_play_game_returns_examples(self):
        """play_game should return training examples with correct value targets."""
        net = _make_network()
        config = _make_config(num_simulations=2, temperature_threshold=2)
        mcts = NeuralMCTS(net, config)
        collector = SelfPlayCollector(mcts, config)

        # A state that becomes terminal after one move
        board = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        initial = SimpleGameState(board=board, reward=0.5)
        examples = await collector.play_game(initial, temperature_threshold=1)

        assert len(examples) >= 1
        # Value target should be set from game outcome
        for ex in examples:
            assert isinstance(ex.value_target, float)  # may be 0.0 for draw

    @pytest.mark.asyncio
    async def test_play_game_temperature_switch(self):
        """After temperature_threshold moves, temperature should switch to final."""
        net = _make_network()
        config = _make_config(
            num_simulations=1,
            temperature_threshold=0,
            temperature_init=1.0,
            temperature_final=0.01,
        )
        mcts = NeuralMCTS(net, config)
        collector = SelfPlayCollector(mcts, config)

        board = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        initial = SimpleGameState(board=board, reward=1.0)
        examples = await collector.play_game(initial)
        # Should complete without error
        assert len(examples) >= 1

    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """generate_batch plays multiple games and concatenates examples."""
        net = _make_network()
        config = _make_config(num_simulations=1, temperature_threshold=1)
        mcts = NeuralMCTS(net, config)
        collector = SelfPlayCollector(mcts, config)

        board = [1, 1, 1, 1, 1, 1, 1, 1, 0]

        def make_state():
            return SimpleGameState(board=board.copy(), reward=1.0)

        examples = await collector.generate_batch(num_games=2, initial_state_fn=make_state)
        assert len(examples) >= 2

    @pytest.mark.asyncio
    async def test_generate_batch_cache_clear(self):
        """Cache should be cleared when it grows too large."""
        net = _make_network()
        config = _make_config(num_simulations=1)
        mcts = NeuralMCTS(net, config)
        collector = SelfPlayCollector(mcts, config)

        # Pre-fill cache beyond threshold
        for i in range(10001):
            mcts.cache[f"state_{i}"] = (np.array([0.5]), 0.5)

        board = [1, 1, 1, 1, 1, 1, 1, 1, 0]

        def make_state():
            return SimpleGameState(board=board.copy(), reward=0.0)

        await collector.generate_batch(num_games=1, initial_state_fn=make_state)
        # Cache should have been cleared
        assert len(mcts.cache) <= 10000


# ---------------------------------------------------------------------------
# GameState base class extended tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGameStateBase:
    """Test abstract GameState base methods raise NotImplementedError."""

    def test_get_legal_actions_raises(self):
        with pytest.raises(NotImplementedError):
            GameState().get_legal_actions()

    def test_apply_action_raises(self):
        with pytest.raises(NotImplementedError):
            GameState().apply_action("x")

    def test_is_terminal_raises(self):
        with pytest.raises(NotImplementedError):
            GameState().is_terminal()

    def test_get_reward_raises(self):
        with pytest.raises(NotImplementedError):
            GameState().get_reward()

    def test_to_tensor_raises(self):
        with pytest.raises(NotImplementedError):
            GameState().to_tensor()

    def test_get_hash_raises(self):
        with pytest.raises(NotImplementedError):
            GameState().get_hash()

    def test_action_to_index_non_grid_string(self):
        """String without comma should be converted via int()."""
        state = GameState()
        assert state.action_to_index("7") == 7
