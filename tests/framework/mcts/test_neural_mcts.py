"""
Comprehensive tests for Neural-Guided MCTS implementation.

Tests cover:
- NeuralMCTSNode functionality
- PUCT selection
- Dirichlet noise
- Virtual loss mechanism
- Network evaluation and caching
- Self-play data collection
- Temperature-based action selection
"""

import numpy as np
import pytest

# Skip entire module if torch not available
torch = pytest.importorskip("torch", reason="PyTorch required for neural MCTS tests")
import torch.nn as nn

from src.framework.mcts.neural_mcts import (
    GameState,
    MCTSExample,
    NeuralMCTS,
    NeuralMCTSNode,
    SelfPlayCollector,
)
from src.training.system_config import MCTSConfig


# Test GameState implementation
class TicTacToeState(GameState):
    """Simple Tic-Tac-Toe for testing."""

    def __init__(self, board=None, player=1):
        self.board = board if board is not None else np.zeros((3, 3), dtype=int)
        self.player = player

    def get_legal_actions(self):
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions

    def apply_action(self, action):
        new_board = self.board.copy()
        new_board[action] = self.player
        return TicTacToeState(new_board, -self.player)

    def is_terminal(self):
        # Check rows, cols, diagonals
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return True
            if abs(self.board[:, i].sum()) == 3:
                return True

        if abs(np.trace(self.board)) == 3:
            return True
        if abs(np.trace(np.fliplr(self.board))) == 3:
            return True

        # Check draw
        return len(self.get_legal_actions()) == 0

    def get_reward(self, player=1):
        for i in range(3):
            if self.board[i, :].sum() == 3 * player:
                return 1.0
            if self.board[:, i].sum() == 3 * player:
                return 1.0

        if np.trace(self.board) == 3 * player:
            return 1.0
        if np.trace(np.fliplr(self.board)) == 3 * player:
            return 1.0

        # Check opponent win
        if self.is_terminal() and len(self.get_legal_actions()) > 0:
            return -1.0

        return 0.0

    def to_tensor(self):
        # Simple encoding: one-hot for X, O, empty
        tensor = torch.zeros(3, 3, 3)
        tensor[:, :, 0] = torch.from_numpy((self.board == 1).astype(float))
        tensor[:, :, 1] = torch.from_numpy((self.board == -1).astype(float))
        tensor[:, :, 2] = torch.from_numpy((self.board == 0).astype(float))
        return tensor

    def action_to_index(self, action: tuple[int, int]) -> int:
        """Map (row, col) action to index 0-8."""
        return action[0] * 3 + action[1]

    def get_hash(self):
        return str(self.board.tobytes())


class TestPolicyValueNetwork(nn.Module):
    """Simple network for testing."""

    def __init__(self, action_size=9):
        super().__init__()
        self.fc1 = nn.Linear(27, 64)
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


class TestNeuralMCTSNode:
    """Tests for NeuralMCTSNode."""

    def test_node_initialization(self):
        """Test node is initialized correctly."""
        state = TicTacToeState()
        node = NeuralMCTSNode(state=state, prior=0.5)

        assert node.state == state
        assert node.prior == 0.5
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.virtual_loss == 0.0
        assert not node.is_expanded
        assert len(node.children) == 0

    def test_node_value_property(self):
        """Test value property calculation."""
        state = TicTacToeState()
        node = NeuralMCTSNode(state=state)

        assert node.value == 0.0  # No visits

        node.visit_count = 10
        node.value_sum = 7.5
        assert node.value == 0.75

    def test_expand(self):
        """Test node expansion."""
        state = TicTacToeState()
        node = NeuralMCTSNode(state=state)

        actions = state.get_legal_actions()
        policy_probs = np.ones(len(actions)) / len(actions)

        node.expand(policy_probs, actions)

        assert node.is_expanded
        assert len(node.children) == len(actions)

        # Check children have correct priors
        for action, child in node.children.items():
            assert child.prior == policy_probs[actions.index(action)]

    def test_virtual_loss(self):
        """Test virtual loss add and revert."""
        state = TicTacToeState()
        node = NeuralMCTSNode(state=state)

        node.visit_count = 10
        node.value_sum = 5.0

        # Add virtual loss
        node.add_virtual_loss(3.0)
        assert node.virtual_loss == 3.0

        # Revert
        node.revert_virtual_loss(3.0)
        assert node.virtual_loss == 0.0

    def test_select_child_puct(self):
        """Test PUCT child selection."""
        state = TicTacToeState()
        root = NeuralMCTSNode(state=state)

        # Create some children with different stats
        actions = [(0, 0), (1, 1), (2, 2)]
        priors = [0.5, 0.3, 0.2]

        root.visit_count = 10

        for action, prior in zip(actions, priors, strict=True):
            child_state = state.apply_action(action)
            child = NeuralMCTSNode(child_state, root, action, prior)
            root.children[action] = child

        # Update some children
        root.children[(0, 0)].visit_count = 5
        root.children[(0, 0)].value_sum = 3.0

        root.children[(1, 1)].visit_count = 3
        root.children[(1, 1)].value_sum = 2.5

        # Unvisited child should be selected (need high exploration to overcome value)
        action, child = root.select_child(c_puct=5.0)
        assert child.visit_count == 0

    def test_update(self):
        """Test node update."""
        state = TicTacToeState()
        node = NeuralMCTSNode(state=state)

        node.update(0.8)
        assert node.visit_count == 1
        assert node.value_sum == 0.8

        node.update(0.6)
        assert node.visit_count == 2
        assert node.value_sum == 1.4

    def test_get_action_probs_deterministic(self):
        """Test action probability computation with temperature=0."""
        state = TicTacToeState()
        root = NeuralMCTSNode(state=state)

        # Create children
        for action in [(0, 0), (1, 1), (2, 2)]:
            child = NeuralMCTSNode(state.apply_action(action), root, action)
            root.children[action] = child

        # Set different visit counts
        root.children[(0, 0)].visit_count = 10  # Best
        root.children[(1, 1)].visit_count = 5
        root.children[(2, 2)].visit_count = 2

        probs = root.get_action_probs(temperature=0.0)

        # Only best action should have probability
        assert probs[(0, 0)] == 1.0
        assert probs[(1, 1)] == 0.0
        assert probs[(2, 2)] == 0.0

    def test_get_action_probs_stochastic(self):
        """Test action probability computation with temperature=1."""
        state = TicTacToeState()
        root = NeuralMCTSNode(state=state)

        # Create children
        for action in [(0, 0), (1, 1)]:
            child = NeuralMCTSNode(state.apply_action(action), root, action)
            root.children[action] = child

        root.children[(0, 0)].visit_count = 10
        root.children[(1, 1)].visit_count = 10

        probs = root.get_action_probs(temperature=1.0)

        # Equal visits should give equal probability
        assert abs(probs[(0, 0)] - 0.5) < 0.01
        assert abs(probs[(1, 1)] - 0.5) < 0.01


class TestNeuralMCTS:
    """Tests for NeuralMCTS engine."""

    @pytest.fixture
    def network(self):
        """Create test network."""
        return TestPolicyValueNetwork()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MCTSConfig(
            num_simulations=10,
            c_puct=1.0,
            virtual_loss=3.0,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3,
        )

    @pytest.fixture
    def mcts(self, network, config):
        """Create MCTS instance."""
        return NeuralMCTS(network, config, device="cpu")

    def test_initialization(self, mcts, network, config):
        """Test MCTS initialization."""
        assert mcts.network == network
        assert mcts.config == config
        assert mcts.device == "cpu"
        assert len(mcts.cache) == 0

    def test_add_dirichlet_noise(self, mcts):
        """Test Dirichlet noise addition."""
        policy_probs = np.array([0.4, 0.3, 0.2, 0.1])

        noised = mcts.add_dirichlet_noise(policy_probs, epsilon=0.25, alpha=0.3)

        # Check shape
        assert noised.shape == policy_probs.shape

        # Check normalization
        assert abs(noised.sum() - 1.0) < 0.01

        # Check noise was added (should be different)
        assert not np.allclose(noised, policy_probs)

    @pytest.mark.asyncio
    async def test_evaluate_state(self, mcts):
        """Test state evaluation with network."""
        state = TicTacToeState()

        policy_probs, value = await mcts.evaluate_state(state, add_noise=False)

        assert len(policy_probs) == len(state.get_legal_actions())
        assert abs(policy_probs.sum() - 1.0) < 0.01
        assert -1.0 <= value <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_state_caching(self, mcts):
        """Test state evaluation caching."""
        state = TicTacToeState()

        # First evaluation
        result1 = await mcts.evaluate_state(state, add_noise=False)

        assert mcts.cache_misses == 1
        assert mcts.cache_hits == 0

        # Second evaluation should hit cache
        result2 = await mcts.evaluate_state(state, add_noise=False)

        assert mcts.cache_hits == 1
        assert np.array_equal(result1[0], result2[0])
        assert result1[1] == result2[1]

    @pytest.mark.asyncio
    async def test_search_basic(self, mcts):
        """Test basic MCTS search."""
        initial_state = TicTacToeState()

        action_probs, root = await mcts.search(
            root_state=initial_state,
            num_simulations=5,
            temperature=1.0,
            add_root_noise=False,
        )

        assert len(action_probs) > 0
        assert abs(sum(action_probs.values()) - 1.0) < 0.01
        assert root.visit_count > 0

    @pytest.mark.asyncio
    async def test_simulate(self, mcts):
        """Test single simulation."""
        state = TicTacToeState()
        root = NeuralMCTSNode(state=state)

        # Expand root
        policy_probs, _ = await mcts.evaluate_state(state)
        actions = state.get_legal_actions()
        root.expand(policy_probs, actions)

        # Run simulation
        value = await mcts._simulate(root)

        assert -1.0 <= value <= 1.0
        assert root.visit_count > 0

    def test_select_action(self, mcts):
        """Test action selection."""
        action_probs = {(0, 0): 0.7, (1, 1): 0.2, (2, 2): 0.1}

        # Deterministic
        action = mcts.select_action(action_probs, deterministic=True)
        assert action == (0, 0)

        # Stochastic (run multiple times to check sampling)
        actions = [mcts.select_action(action_probs, deterministic=False) for _ in range(100)]
        # (0,0) should be selected most often
        assert actions.count((0, 0)) > 50

    def test_cache_management(self, mcts):
        """Test cache clearing and stats."""
        state = TicTacToeState()

        # Add to cache
        mcts.cache[state.get_hash()] = (np.array([0.5, 0.5]), 0.3)
        mcts.cache_hits = 5
        mcts.cache_misses = 10

        # Get stats
        stats = mcts.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["cache_hits"] == 5
        assert stats["cache_misses"] == 10
        assert stats["hit_rate"] == 5 / 15

        # Clear cache
        mcts.clear_cache()
        assert len(mcts.cache) == 0
        assert mcts.cache_hits == 0
        assert mcts.cache_misses == 0


class TestSelfPlayCollector:
    """Tests for self-play data collection."""

    @pytest.fixture
    def network(self):
        return TestPolicyValueNetwork()

    @pytest.fixture
    def config(self):
        return MCTSConfig(num_simulations=5, temperature_threshold=3)

    @pytest.fixture
    def mcts(self, network, config):
        return NeuralMCTS(network, config, device="cpu")

    @pytest.fixture
    def collector(self, mcts, config):
        return SelfPlayCollector(mcts, config)

    @pytest.mark.asyncio
    async def test_play_game(self, collector):
        """Test single game generation."""
        initial_state = TicTacToeState()

        examples = await collector.play_game(initial_state, temperature_threshold=2)

        # Should have some examples
        assert len(examples) > 0

        # Each example should have correct format
        for example in examples:
            assert isinstance(example, MCTSExample)
            assert example.state.shape == (3, 3, 3)
            assert len(example.policy_target) > 0
            assert -1.0 <= example.value_target <= 1.0
            assert example.player in [-1, 1]

    @pytest.mark.asyncio
    async def test_generate_batch(self, collector):
        """Test batch generation."""

        def initial_state_fn():
            return TicTacToeState()

        examples = await collector.generate_batch(num_games=2, initial_state_fn=initial_state_fn)

        # Should have examples from multiple games
        assert len(examples) > 0

    @pytest.mark.asyncio
    async def test_value_assignment(self, collector):
        """Test that game outcomes are assigned to examples."""
        initial_state = TicTacToeState()

        examples = await collector.play_game(initial_state)

        # All examples should have value targets assigned
        for example in examples:
            assert example.value_target in [-1.0, 0.0, 1.0]


@pytest.mark.integration
class TestNeuralMCTSIntegration:
    """Integration tests for complete neural MCTS system."""

    @pytest.mark.asyncio
    async def test_full_search_workflow(self):
        """Test complete search workflow."""
        network = TestPolicyValueNetwork()
        config = MCTSConfig(num_simulations=20)
        mcts = NeuralMCTS(network, config, device="cpu")

        initial_state = TicTacToeState()

        # Run search
        action_probs, root = await mcts.search(
            root_state=initial_state,
            num_simulations=20,
            temperature=1.0,
        )

        # Verify results
        assert len(action_probs) > 0
        assert root.visit_count >= 20
        assert root.is_expanded

        # Select action
        action = mcts.select_action(action_probs, deterministic=True)
        assert action in initial_state.get_legal_actions()

    @pytest.mark.asyncio
    async def test_deterministic_search(self):
        """Test that search is deterministic with same seed."""
        network = TestPolicyValueNetwork()
        config = MCTSConfig(num_simulations=10)

        # Two MCTS instances with same network
        mcts1 = NeuralMCTS(network, config, device="cpu")
        mcts2 = NeuralMCTS(network, config, device="cpu")

        initial_state = TicTacToeState()

        # Run searches
        result1, _ = await mcts1.search(initial_state, num_simulations=10, add_root_noise=False)
        result2, _ = await mcts2.search(initial_state, num_simulations=10, add_root_noise=False)

        # Should get same results (without noise)
        for action in result1:
            assert abs(result1[action] - result2[action]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
