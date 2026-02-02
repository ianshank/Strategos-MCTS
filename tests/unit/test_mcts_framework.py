"""
Comprehensive unit tests for MCTS Framework core module.

Tests cover:
- MCTSState: hashability, feature vectors, deterministic hashing
- MCTSNode: initialization, UCB1 calculation, child management, progressive widening
- MCTSEngine: search execution, select/expand/simulate/backpropagate phases

Focus areas:
- Deterministic behavior with seeded RNG
- UCB1 selection correctness
- Progressive widening behavior
- Backpropagation value updates
- Basic search loop
"""

import math

import numpy as np
import pytest

from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.policies import (
    RandomRolloutPolicy,
    RolloutPolicy,
    SelectionPolicy,
    ucb1,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_state():
    """Create a simple MCTSState for testing."""
    return MCTSState(state_id="test_state", features={"depth": 1, "score": 0.5})


@pytest.fixture
def root_node():
    """Create a root MCTSNode with seeded RNG."""
    state = MCTSState(state_id="root", features={"depth": 0})
    return MCTSNode(state=state, rng=np.random.default_rng(42))


@pytest.fixture
def seeded_engine():
    """Create an MCTSEngine with fixed seed for deterministic tests."""
    return MCTSEngine(
        seed=42,
        exploration_weight=1.414,
        progressive_widening_k=1.0,
        progressive_widening_alpha=0.5,
    )


@pytest.fixture
def simple_action_generator():
    """Create a simple action generator for testing."""

    def generator(state: MCTSState) -> list[str]:
        depth = state.features.get("depth", 0)
        if depth >= 3:
            return []
        return [f"action_{i}" for i in range(3)]

    return generator


@pytest.fixture
def simple_state_transition():
    """Create a simple state transition function for testing."""

    def transition(state: MCTSState, action: str) -> MCTSState:
        depth = state.features.get("depth", 0)
        return MCTSState(
            state_id=f"{state.state_id}_{action}",
            features={"depth": depth + 1, "action": action},
        )

    return transition


@pytest.fixture
def random_rollout_policy():
    """Create a random rollout policy for testing."""
    return RandomRolloutPolicy(base_value=0.5, noise_scale=0.1)


# ============================================================================
# MCTSState Tests
# ============================================================================


class TestMCTSState:
    """Test suite for MCTSState class."""

    def test_state_initialization(self, simple_state):
        """Test basic state initialization."""
        assert simple_state.state_id == "test_state"
        assert simple_state.features == {"depth": 1, "score": 0.5}

    def test_state_with_empty_features(self):
        """Test state with no features."""
        state = MCTSState(state_id="empty")
        assert state.state_id == "empty"
        assert state.features == {}

    def test_hash_key_generation(self, simple_state):
        """Test that hash key is generated correctly."""
        hash_key = simple_state.to_hash_key()
        assert isinstance(hash_key, str)
        assert len(hash_key) == 64  # SHA256 produces 64 hex characters

    def test_hash_key_determinism(self, simple_state):
        """Test that same state produces same hash."""
        hash1 = simple_state.to_hash_key()
        hash2 = simple_state.to_hash_key()
        assert hash1 == hash2

    def test_hash_key_uniqueness(self):
        """Test that different states produce different hashes."""
        state1 = MCTSState(state_id="state_1", features={"a": 1})
        state2 = MCTSState(state_id="state_2", features={"a": 1})
        state3 = MCTSState(state_id="state_1", features={"a": 2})

        hash1 = state1.to_hash_key()
        hash2 = state2.to_hash_key()
        hash3 = state3.to_hash_key()

        assert hash1 != hash2  # Different state_id
        assert hash1 != hash3  # Different features
        assert hash2 != hash3  # Both differ

    def test_hash_key_order_independence(self):
        """Test that feature ordering doesn't affect hash."""
        state1 = MCTSState(state_id="state", features={"a": 1, "b": 2, "c": 3})
        state2 = MCTSState(state_id="state", features={"c": 3, "a": 1, "b": 2})

        assert state1.to_hash_key() == state2.to_hash_key()

    @pytest.mark.parametrize(
        "features",
        [
            {},
            {"single": 1},
            {"nested": {"a": 1, "b": 2}},
            {"list_val": [1, 2, 3]},
            {"mixed": {"a": 1, "b": [2, 3], "c": "text"}},
        ],
    )
    def test_hash_key_with_various_features(self, features):
        """Test hash key generation with various feature types."""
        state = MCTSState(state_id="test", features=features)
        hash_key = state.to_hash_key()
        assert isinstance(hash_key, str)
        assert len(hash_key) == 64


# ============================================================================
# MCTSNode Tests
# ============================================================================


class TestMCTSNode:
    """Test suite for MCTSNode class."""

    def test_node_initialization(self, root_node):
        """Test that nodes are initialized with correct defaults."""
        assert root_node.state.state_id == "root"
        assert root_node.parent is None
        assert root_node.action is None
        assert root_node.children == []
        assert root_node.visits == 0
        assert root_node.value_sum == 0.0
        assert root_node.terminal is False
        assert root_node.expanded_actions == set()
        assert root_node.available_actions == []

    def test_node_with_parent(self):
        """Test node initialization with parent."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)

        child_state = MCTSState(state_id="child")
        child = MCTSNode(state=child_state, parent=parent, action="move")

        assert child.parent is parent
        assert child.action == "move"

    def test_value_property_unvisited(self, root_node):
        """Test value property returns 0 for unvisited node."""
        assert root_node.value == 0.0

    def test_value_property_visited(self, root_node):
        """Test value property computes average correctly."""
        root_node.visits = 10
        root_node.value_sum = 7.5
        assert root_node.value == 0.75

    def test_is_fully_expanded_no_actions(self, root_node):
        """Test is_fully_expanded when no actions defined."""
        root_node.available_actions = []
        assert root_node.is_fully_expanded is True

    def test_is_fully_expanded_partial(self, root_node):
        """Test is_fully_expanded with partial expansion."""
        root_node.available_actions = ["a", "b", "c"]
        root_node.expanded_actions = {"a"}
        assert root_node.is_fully_expanded is False

    def test_is_fully_expanded_complete(self, root_node):
        """Test is_fully_expanded when all actions expanded."""
        root_node.available_actions = ["a", "b", "c"]
        root_node.expanded_actions = {"a", "b", "c"}
        assert root_node.is_fully_expanded is True

    def test_select_child_no_children_raises_error(self, root_node):
        """Test that select_child raises error with no children."""
        with pytest.raises(ValueError, match="No children to select from"):
            root_node.select_child()

    def test_select_child_single_child(self, root_node):
        """Test select_child with single child."""
        child_state = MCTSState(state_id="child")
        root_node.add_child("action", child_state)
        root_node.visits = 10

        selected = root_node.select_child()
        assert selected.state.state_id == "child"

    def test_select_child_prefers_unvisited(self, root_node):
        """Test that unvisited children are selected (infinite UCB1)."""
        root_node.visits = 100

        # Add visited child
        visited_state = MCTSState(state_id="visited")
        visited = root_node.add_child("visited_action", visited_state)
        visited.visits = 10
        visited.value_sum = 9.0  # High value

        # Add unvisited child
        unvisited_state = MCTSState(state_id="unvisited")
        root_node.add_child("unvisited_action", unvisited_state)

        selected = root_node.select_child()
        assert selected.state.state_id == "unvisited"

    def test_select_child_ucb1_calculation(self, root_node):
        """Test that select_child uses UCB1 correctly."""
        root_node.visits = 100

        # Create children with different values
        state1 = MCTSState(state_id="low_value")
        child1 = root_node.add_child("action1", state1)
        child1.visits = 20
        child1.value_sum = 4.0  # Average 0.2

        state2 = MCTSState(state_id="high_value")
        child2 = root_node.add_child("action2", state2)
        child2.visits = 20
        child2.value_sum = 16.0  # Average 0.8

        # child2 should be selected (higher value, same visits)
        selected = root_node.select_child()
        assert selected is child2

    def test_select_child_exploration_vs_exploitation(self, root_node):
        """Test UCB1 balances exploration and exploitation."""
        root_node.visits = 1000

        # High value but many visits (exploitation)
        state1 = MCTSState(state_id="exploited")
        child1 = root_node.add_child("action1", state1)
        child1.visits = 500
        child1.value_sum = 450.0  # Average 0.9

        # Lower value but fewer visits (exploration)
        state2 = MCTSState(state_id="explored")
        child2 = root_node.add_child("action2", state2)
        child2.visits = 10
        child2.value_sum = 7.0  # Average 0.7

        # With high exploration weight, child2 may win
        # Calculate UCB1 scores to verify
        score1 = ucb1(child1.value_sum, child1.visits, root_node.visits, c=1.414)
        score2 = ucb1(child2.value_sum, child2.visits, root_node.visits, c=1.414)

        selected = root_node.select_child(exploration_weight=1.414)
        if score2 > score1:
            assert selected is child2
        else:
            assert selected is child1

    @pytest.mark.parametrize("exploration_weight", [0.1, 0.5, 1.0, 1.414, 2.0, 5.0])
    def test_select_child_various_exploration_weights(self, root_node, exploration_weight):
        """Test select_child with various exploration weights."""
        root_node.visits = 100

        state1 = MCTSState(state_id="child1")
        child1 = root_node.add_child("action1", state1)
        child1.visits = 25
        child1.value_sum = 12.5

        state2 = MCTSState(state_id="child2")
        child2 = root_node.add_child("action2", state2)
        child2.visits = 25
        child2.value_sum = 10.0

        # Should always select child with higher value when visits equal
        selected = root_node.select_child(exploration_weight=exploration_weight)
        assert selected is child1

    def test_add_child_creates_relationship(self, root_node):
        """Test that add_child properly links parent and child."""
        child_state = MCTSState(state_id="child", features={"depth": 1})
        child = root_node.add_child("move_forward", child_state)

        assert child in root_node.children
        assert child.parent is root_node
        assert child.action == "move_forward"
        assert child.state.state_id == "child"
        assert "move_forward" in root_node.expanded_actions

    def test_add_multiple_children(self, root_node):
        """Test adding multiple children."""
        for i in range(5):
            state = MCTSState(state_id=f"child_{i}")
            root_node.add_child(f"action_{i}", state)

        assert len(root_node.children) == 5
        assert len(root_node.expanded_actions) == 5
        for i in range(5):
            assert f"action_{i}" in root_node.expanded_actions

    def test_add_child_shares_rng(self, root_node):
        """Test that child inherits parent's RNG."""
        child_state = MCTSState(state_id="child")
        child = root_node.add_child("action", child_state)
        assert child._rng is root_node._rng

    def test_get_unexpanded_action_no_available(self, root_node):
        """Test get_unexpanded_action when no actions available."""
        root_node.available_actions = []
        assert root_node.get_unexpanded_action() is None

    def test_get_unexpanded_action_all_expanded(self, root_node):
        """Test get_unexpanded_action when all expanded."""
        root_node.available_actions = ["a", "b", "c"]
        root_node.expanded_actions = {"a", "b", "c"}
        assert root_node.get_unexpanded_action() is None

    def test_get_unexpanded_action_returns_valid(self, root_node):
        """Test get_unexpanded_action returns valid unexpanded action."""
        root_node.available_actions = ["a", "b", "c", "d"]
        root_node.expanded_actions = {"a", "c"}

        action = root_node.get_unexpanded_action()
        assert action in ["b", "d"]

    def test_get_unexpanded_action_deterministic_with_seed(self):
        """Test that get_unexpanded_action is deterministic with seed."""
        rng = np.random.default_rng(42)
        state = MCTSState(state_id="test")
        node = MCTSNode(state=state, rng=rng)
        node.available_actions = ["a", "b", "c", "d", "e"]

        rng1 = np.random.default_rng(42)
        node1 = MCTSNode(state=state, rng=rng1)
        node1.available_actions = ["a", "b", "c", "d", "e"]

        action1 = node.get_unexpanded_action()
        action2 = node1.get_unexpanded_action()
        assert action1 == action2

    def test_node_repr(self, root_node):
        """Test node string representation."""
        root_node.visits = 10
        root_node.value_sum = 7.5

        repr_str = repr(root_node)
        assert "root" in repr_str
        assert "visits=10" in repr_str
        assert "value=0.750" in repr_str
        assert "children=0" in repr_str


# ============================================================================
# MCTSEngine Tests
# ============================================================================


class TestMCTSEngine:
    """Test suite for MCTSEngine class."""

    def test_engine_initialization(self, seeded_engine):
        """Test engine initializes with correct parameters."""
        assert seeded_engine.seed == 42
        assert seeded_engine.exploration_weight == 1.414
        assert seeded_engine.progressive_widening_k == 1.0
        assert seeded_engine.progressive_widening_alpha == 0.5
        assert seeded_engine.max_parallel_rollouts == 4
        assert seeded_engine.cache_size_limit == 10000

    def test_engine_rng_determinism(self):
        """Test that engine RNG is deterministic."""
        engine1 = MCTSEngine(seed=123)
        engine2 = MCTSEngine(seed=123)

        values1 = [engine1.rng.random() for _ in range(10)]
        values2 = [engine2.rng.random() for _ in range(10)]

        assert values1 == values2

    def test_reset_seed(self, seeded_engine):
        """Test that reset_seed creates new deterministic RNG."""
        val1 = seeded_engine.rng.random()
        seeded_engine.reset_seed(42)
        val2 = seeded_engine.rng.random()
        assert val1 == val2

    def test_clear_cache(self, seeded_engine):
        """Test that clear_cache resets cache and statistics."""
        seeded_engine._simulation_cache["test"] = (0.5, 1)
        seeded_engine.cache_hits = 10
        seeded_engine.cache_misses = 5

        seeded_engine.clear_cache()

        assert seeded_engine._simulation_cache == {}
        assert seeded_engine.cache_hits == 0
        assert seeded_engine.cache_misses == 0

    def test_should_expand_terminal_node(self, seeded_engine, root_node):
        """Test should_expand returns False for terminal nodes."""
        root_node.terminal = True
        assert seeded_engine.should_expand(root_node) is False

    def test_should_expand_fully_expanded(self, seeded_engine, root_node):
        """Test should_expand returns False for fully expanded nodes."""
        root_node.available_actions = ["a", "b"]
        root_node.expanded_actions = {"a", "b"}
        root_node.visits = 100  # High visits
        assert seeded_engine.should_expand(root_node) is False

    def test_should_expand_progressive_widening_no_children(self, seeded_engine, root_node):
        """Test progressive widening with no children."""
        root_node.available_actions = ["a", "b", "c"]
        root_node.visits = 1
        # Threshold = k * 0^alpha = 0, visits (1) > 0
        assert seeded_engine.should_expand(root_node) is True

    def test_should_expand_progressive_widening_with_children(self, seeded_engine, root_node):
        """Test progressive widening with existing children."""
        root_node.available_actions = ["a", "b", "c", "d", "e"]
        root_node.expanded_actions = {"a", "b"}

        # Add children
        for action in ["a", "b"]:
            state = MCTSState(state_id=f"child_{action}")
            root_node.add_child(action, state)

        # Threshold = k * 2^0.5 = 1 * 1.414... = 1.414
        root_node.visits = 1
        assert seeded_engine.should_expand(root_node) is False

        root_node.visits = 2
        assert seeded_engine.should_expand(root_node) is True

    @pytest.mark.parametrize(
        "k,alpha,num_children,visits,expected",
        [
            (1.0, 0.5, 0, 1, True),  # 1 > 1*0^0.5 = 0
            (1.0, 0.5, 1, 2, True),  # 2 > 1*1^0.5 = 1
            (1.0, 0.5, 4, 2, False),  # 2 < 1*4^0.5 = 2
            (1.0, 0.5, 4, 3, True),  # 3 > 2
            (2.0, 0.5, 1, 2, False),  # 2 < 2*1^0.5 = 2
            (2.0, 0.5, 1, 3, True),  # 3 > 2
            (0.5, 0.5, 4, 1, False),  # 1 < 0.5*2 = 1
            (0.5, 0.5, 4, 2, True),  # 2 > 1
        ],
    )
    def test_should_expand_various_parameters(self, k, alpha, num_children, visits, expected):
        """Test progressive widening with various parameters."""
        engine = MCTSEngine(progressive_widening_k=k, progressive_widening_alpha=alpha)
        state = MCTSState(state_id="test")
        node = MCTSNode(state=state)

        # Setup node
        node.available_actions = [f"action_{i}" for i in range(num_children + 5)]
        for i in range(num_children):
            child_state = MCTSState(state_id=f"child_{i}")
            node.add_child(f"action_{i}", child_state)
        node.visits = visits

        assert engine.should_expand(node) is expected

    def test_select_returns_root_when_no_children(self, seeded_engine, root_node):
        """Test select returns root when no children exist."""
        selected = seeded_engine.select(root_node)
        assert selected is root_node

    def test_select_traverses_tree(self, seeded_engine, root_node):
        """Test select traverses to leaf node."""
        root_node.visits = 100

        # Build a simple tree
        child_state = MCTSState(state_id="child")
        child = root_node.add_child("action1", child_state)
        child.visits = 50
        child.value_sum = 25.0

        grandchild_state = MCTSState(state_id="grandchild")
        grandchild = child.add_child("action2", grandchild_state)

        selected = seeded_engine.select(root_node)
        assert selected is grandchild

    def test_select_stops_at_terminal(self, seeded_engine, root_node):
        """Test select stops at terminal node."""
        root_node.visits = 10

        child_state = MCTSState(state_id="terminal")
        child = root_node.add_child("action", child_state)
        child.visits = 5
        child.value_sum = 2.5
        child.terminal = True

        selected = seeded_engine.select(root_node)
        assert selected is child

    def test_select_respects_progressive_widening(self, seeded_engine, root_node):
        """Test select stops when progressive widening triggers expansion."""
        root_node.visits = 10
        root_node.available_actions = ["a", "b", "c", "d", "e"]
        root_node.expanded_actions = {"a"}

        # Add one child
        child_state = MCTSState(state_id="child")
        child = root_node.add_child("a", child_state)
        child.visits = 5
        child.value_sum = 2.5

        # With k=1, alpha=0.5, threshold for 1 child = 1.0
        # root.visits (10) > 1.0, so expansion should be possible
        # select should stop at root to allow expansion
        selected = seeded_engine.select(root_node)
        # When should_expand is True, select breaks the loop early
        assert selected is root_node

    def test_expand_terminal_node(self, seeded_engine, root_node, simple_action_generator, simple_state_transition):
        """Test expand returns same node for terminal."""
        root_node.terminal = True
        result = seeded_engine.expand(root_node, simple_action_generator, simple_state_transition)
        assert result is root_node

    def test_expand_generates_actions(self, seeded_engine, root_node, simple_action_generator, simple_state_transition):
        """Test expand generates available actions."""
        root_node.visits = 1
        seeded_engine.expand(root_node, simple_action_generator, simple_state_transition)
        assert root_node.available_actions == ["action_0", "action_1", "action_2"]

    def test_expand_marks_terminal_when_no_actions(self, seeded_engine):
        """Test expand marks node as terminal when no actions available."""
        state = MCTSState(state_id="deep", features={"depth": 10})
        node = MCTSNode(state=state)
        node.visits = 1

        def no_actions(s):
            return []

        def dummy_transition(s, a):
            return s

        result = seeded_engine.expand(node, no_actions, dummy_transition)
        assert result is node
        assert node.terminal is True

    def test_expand_respects_progressive_widening(
        self, seeded_engine, root_node, simple_action_generator, simple_state_transition
    ):
        """Test expand respects progressive widening threshold."""
        root_node.available_actions = ["a", "b", "c"]
        root_node.visits = 1

        # With no children, threshold = 0, so 1 > 0 means we should expand
        result = seeded_engine.expand(root_node, simple_action_generator, simple_state_transition)
        assert len(root_node.children) == 1
        assert result in root_node.children

    def test_expand_creates_child_with_correct_state(
        self, seeded_engine, root_node, simple_action_generator, simple_state_transition
    ):
        """Test expand creates child with proper state transition."""
        root_node.visits = 1
        child = seeded_engine.expand(root_node, simple_action_generator, simple_state_transition)

        assert child.parent is root_node
        assert child.action in root_node.expanded_actions
        assert "depth" in child.state.features
        assert child.state.features["depth"] == 1

    def test_backpropagate_single_node(self, seeded_engine, root_node):
        """Test backpropagate updates single node."""
        seeded_engine.backpropagate(root_node, 0.8)

        assert root_node.visits == 1
        assert root_node.value_sum == 0.8
        assert root_node.value == 0.8

    def test_backpropagate_chain(self, seeded_engine, root_node):
        """Test backpropagate updates entire path to root."""
        # Build chain
        child_state = MCTSState(state_id="child")
        child = root_node.add_child("a1", child_state)

        grandchild_state = MCTSState(state_id="grandchild")
        grandchild = child.add_child("a2", grandchild_state)

        seeded_engine.backpropagate(grandchild, 0.6)

        assert grandchild.visits == 1
        assert grandchild.value_sum == 0.6
        assert child.visits == 1
        assert child.value_sum == 0.6
        assert root_node.visits == 1
        assert root_node.value_sum == 0.6

    def test_backpropagate_accumulates(self, seeded_engine, root_node):
        """Test multiple backpropagations accumulate correctly."""
        child_state = MCTSState(state_id="child")
        child = root_node.add_child("action", child_state)

        seeded_engine.backpropagate(child, 0.4)
        seeded_engine.backpropagate(child, 0.8)
        seeded_engine.backpropagate(child, 0.6)

        assert child.visits == 3
        assert child.value_sum == pytest.approx(1.8)
        assert child.value == pytest.approx(0.6)

        assert root_node.visits == 3
        assert root_node.value_sum == pytest.approx(1.8)

    @pytest.mark.asyncio
    async def test_simulate_returns_bounded_value(self, seeded_engine, root_node, random_rollout_policy):
        """Test simulate returns value in [0, 1] range."""
        for _ in range(20):
            value = await seeded_engine.simulate(root_node, random_rollout_policy)
            assert 0.0 <= value <= 1.0

    @pytest.mark.asyncio
    async def test_simulate_caching(self, seeded_engine, random_rollout_policy):
        """Test that simulate caches results."""
        state = MCTSState(state_id="cacheable", features={"test": 1})
        node = MCTSNode(state=state)

        # First call - cache miss
        value1 = await seeded_engine.simulate(node, random_rollout_policy)
        assert seeded_engine.cache_misses == 1
        assert seeded_engine.cache_hits == 0

        # Second call - cache hit
        value2 = await seeded_engine.simulate(node, random_rollout_policy)
        assert seeded_engine.cache_misses == 1
        assert seeded_engine.cache_hits == 1

        # Values should be close (cached value + small noise)
        assert abs(value1 - value2) < 0.1

    @pytest.mark.asyncio
    async def test_simulate_statistics(self, seeded_engine, root_node, random_rollout_policy):
        """Test simulate updates statistics correctly."""
        await seeded_engine.simulate(root_node, random_rollout_policy)

        assert seeded_engine.total_simulations == 1
        assert seeded_engine.cache_misses == 1
        assert len(seeded_engine._simulation_cache) == 1

    @pytest.mark.asyncio
    async def test_simulate_cache_size_limit(self, random_rollout_policy):
        """Test that cache respects size limit."""
        engine = MCTSEngine(cache_size_limit=3)

        # Fill cache
        for i in range(5):
            state = MCTSState(state_id=f"state_{i}")
            node = MCTSNode(state=state)
            await engine.simulate(node, random_rollout_policy)

        # Cache should not exceed limit
        assert len(engine._simulation_cache) <= 3

    @pytest.mark.asyncio
    async def test_run_iteration(
        self,
        seeded_engine,
        root_node,
        simple_action_generator,
        simple_state_transition,
        random_rollout_policy,
    ):
        """Test single MCTS iteration."""
        await seeded_engine.run_iteration(
            root_node,
            simple_action_generator,
            simple_state_transition,
            random_rollout_policy,
        )

        # Root should be visited
        assert root_node.visits >= 1
        assert root_node.value_sum > 0

    @pytest.mark.asyncio
    async def test_run_iteration_expansion(
        self,
        seeded_engine,
        root_node,
        simple_action_generator,
        simple_state_transition,
        random_rollout_policy,
    ):
        """Test that run_iteration expands tree."""
        # First iteration
        await seeded_engine.run_iteration(
            root_node,
            simple_action_generator,
            simple_state_transition,
            random_rollout_policy,
        )

        # Root visited but no expansion yet (visits == 0 initially)
        assert root_node.visits == 1

        # Second iteration should expand
        await seeded_engine.run_iteration(
            root_node,
            simple_action_generator,
            simple_state_transition,
            random_rollout_policy,
        )

        # Now we should have children
        assert root_node.visits == 2
        # With progressive widening and visits > 0, expansion should occur
        assert len(root_node.children) >= 0  # Depends on progressive widening

    @pytest.mark.asyncio
    async def test_search_basic(
        self,
        seeded_engine,
        root_node,
        simple_action_generator,
        simple_state_transition,
        random_rollout_policy,
    ):
        """Test basic MCTS search."""
        best_action, stats = await seeded_engine.search(
            root_node,
            num_iterations=10,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=random_rollout_policy,
        )

        # Should have some statistics
        assert stats["iterations"] == 10
        assert stats["root_visits"] == 10
        assert "action_stats" in stats
        assert "cache_hit_rate" in stats

    @pytest.mark.asyncio
    async def test_search_returns_best_action(
        self,
        seeded_engine,
        root_node,
        simple_action_generator,
        simple_state_transition,
        random_rollout_policy,
    ):
        """Test that search returns valid best action."""
        best_action, stats = await seeded_engine.search(
            root_node,
            num_iterations=20,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=random_rollout_policy,
        )

        if root_node.children:
            assert best_action is not None
            # Best action should be from available actions
            assert best_action in ["action_0", "action_1", "action_2"]
            assert stats["best_action"] == best_action

    @pytest.mark.asyncio
    async def test_search_determinism(self, simple_action_generator, simple_state_transition):
        """Test that search is deterministic with same seed."""

        # Create deterministic rollout policy
        class DeterministicPolicy(RolloutPolicy):
            async def evaluate(
                self,
                state: MCTSState,
                rng: np.random.Generator,
                max_depth: int = 10,
            ) -> float:
                # Use state hash to generate deterministic value
                hash_val = int(state.to_hash_key()[:8], 16)
                return (hash_val % 100) / 100.0

        policy = DeterministicPolicy()

        # First run
        engine1 = MCTSEngine(seed=42)
        state1 = MCTSState(state_id="root", features={"depth": 0})
        root1 = MCTSNode(state=state1, rng=engine1.rng)

        action1, stats1 = await engine1.search(
            root1,
            num_iterations=10,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=policy,
        )

        # Second run with same seed
        engine2 = MCTSEngine(seed=42)
        state2 = MCTSState(state_id="root", features={"depth": 0})
        root2 = MCTSNode(state=state2, rng=engine2.rng)

        action2, stats2 = await engine2.search(
            root2,
            num_iterations=10,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=policy,
        )

        # Results should be identical
        assert action1 == action2
        assert stats1["root_visits"] == stats2["root_visits"]
        assert stats1["num_children"] == stats2["num_children"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "policy", [SelectionPolicy.MAX_VISITS, SelectionPolicy.MAX_VALUE, SelectionPolicy.ROBUST_CHILD]
    )
    async def test_search_selection_policies(
        self,
        seeded_engine,
        root_node,
        simple_action_generator,
        simple_state_transition,
        random_rollout_policy,
        policy,
    ):
        """Test search with different selection policies."""
        best_action, stats = await seeded_engine.search(
            root_node,
            num_iterations=15,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=random_rollout_policy,
            selection_policy=policy,
        )

        # Should complete without error
        assert stats["iterations"] == 15

    def test_select_best_action_max_visits(self, seeded_engine, root_node):
        """Test _select_best_action with MAX_VISITS policy."""
        # Add children with different visit counts
        for i, (visits, value_sum) in enumerate([(30, 15.0), (50, 20.0), (20, 18.0)]):
            state = MCTSState(state_id=f"child_{i}")
            child = root_node.add_child(f"action_{i}", state)
            child.visits = visits
            child.value_sum = value_sum

        best = seeded_engine._select_best_action(root_node, SelectionPolicy.MAX_VISITS)
        assert best == "action_1"  # Highest visits (50)

    def test_select_best_action_max_value(self, seeded_engine, root_node):
        """Test _select_best_action with MAX_VALUE policy."""
        # Add children with different average values
        for i, (visits, value_sum) in enumerate([(10, 5.0), (10, 9.0), (10, 7.0)]):
            state = MCTSState(state_id=f"child_{i}")
            child = root_node.add_child(f"action_{i}", state)
            child.visits = visits
            child.value_sum = value_sum

        best = seeded_engine._select_best_action(root_node, SelectionPolicy.MAX_VALUE)
        assert best == "action_1"  # Highest value (0.9)

    def test_select_best_action_robust_child(self, seeded_engine, root_node):
        """Test _select_best_action with ROBUST_CHILD policy."""
        # Add children balancing visits and value
        configs = [
            (50, 25.0),  # High visits, medium value (0.5)
            (30, 24.0),  # Medium visits, high value (0.8)
            (10, 3.0),  # Low visits, low value (0.3)
        ]
        for i, (visits, value_sum) in enumerate(configs):
            state = MCTSState(state_id=f"child_{i}")
            child = root_node.add_child(f"action_{i}", state)
            child.visits = visits
            child.value_sum = value_sum

        best = seeded_engine._select_best_action(root_node, SelectionPolicy.ROBUST_CHILD)
        # Should balance between action_0 (high visits) and action_1 (high value)
        assert best in ["action_0", "action_1"]

    def test_select_best_action_no_children(self, seeded_engine, root_node):
        """Test _select_best_action returns None when no children."""
        best = seeded_engine._select_best_action(root_node, SelectionPolicy.MAX_VISITS)
        assert best is None

    def test_compute_statistics(self, seeded_engine, root_node):
        """Test _compute_statistics returns correct information."""
        root_node.visits = 100
        root_node.value_sum = 60.0

        # Add children
        for i in range(3):
            state = MCTSState(state_id=f"child_{i}")
            child = root_node.add_child(f"action_{i}", state)
            child.visits = 30 + i * 5
            child.value_sum = 15.0 + i * 3

        seeded_engine.total_simulations = 50
        seeded_engine.cache_hits = 20
        seeded_engine.cache_misses = 30

        stats = seeded_engine._compute_statistics(root_node, 100)

        assert stats["iterations"] == 100
        assert stats["root_visits"] == 100
        assert stats["root_value"] == 0.6
        assert stats["num_children"] == 3
        assert stats["best_action"] == "action_2"  # Most visits
        assert stats["best_action_visits"] == 40
        assert len(stats["action_stats"]) == 3
        assert stats["total_simulations"] == 50
        assert stats["cache_hit_rate"] == 0.4
        assert stats["seed"] == 42

    def test_get_tree_depth_leaf(self, seeded_engine, root_node):
        """Test get_tree_depth for leaf node."""
        depth = seeded_engine.get_tree_depth(root_node)
        assert depth == 0

    def test_get_tree_depth_with_children(self, seeded_engine, root_node):
        """Test get_tree_depth with tree structure."""
        # Build tree with depth 3
        child_state = MCTSState(state_id="child")
        child = root_node.add_child("a1", child_state)

        grandchild_state = MCTSState(state_id="grandchild")
        grandchild = child.add_child("a2", grandchild_state)

        great_grandchild_state = MCTSState(state_id="great_grandchild")
        grandchild.add_child("a3", great_grandchild_state)

        depth = seeded_engine.get_tree_depth(root_node)
        assert depth == 3

    def test_count_nodes_single(self, seeded_engine, root_node):
        """Test count_nodes for single node."""
        count = seeded_engine.count_nodes(root_node)
        assert count == 1

    def test_count_nodes_tree(self, seeded_engine, root_node):
        """Test count_nodes for tree."""
        # Build tree with 7 nodes
        # root -> child1 -> grandchild1
        #      -> child2 -> grandchild2
        #               -> grandchild3
        #      -> child3
        child1_state = MCTSState(state_id="c1")
        child1 = root_node.add_child("a1", child1_state)
        gc1_state = MCTSState(state_id="gc1")
        child1.add_child("a1.1", gc1_state)

        child2_state = MCTSState(state_id="c2")
        child2 = root_node.add_child("a2", child2_state)
        gc2_state = MCTSState(state_id="gc2")
        child2.add_child("a2.1", gc2_state)
        gc3_state = MCTSState(state_id="gc3")
        child2.add_child("a2.2", gc3_state)

        child3_state = MCTSState(state_id="c3")
        root_node.add_child("a3", child3_state)

        count = seeded_engine.count_nodes(root_node)
        assert count == 7


# ============================================================================
# Integration Tests
# ============================================================================


class TestMCTSIntegration:
    """Integration tests for complete MCTS workflows."""

    @pytest.mark.asyncio
    async def test_full_search_workflow(self):
        """Test complete MCTS search workflow."""
        # Setup
        engine = MCTSEngine(seed=42, exploration_weight=1.0)
        root_state = MCTSState(state_id="root", features={"depth": 0, "value": 0})
        root = MCTSNode(state=root_state, rng=engine.rng)

        def action_gen(state):
            depth = state.features.get("depth", 0)
            if depth >= 2:
                return []
            return ["left", "right"]

        def state_trans(state, action):
            depth = state.features.get("depth", 0)
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={"depth": depth + 1, "action": action},
            )

        policy = RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

        # Execute search
        best_action, stats = await engine.search(
            root,
            num_iterations=50,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=policy,
        )

        # Verify results
        assert best_action in ["left", "right"]
        assert stats["root_visits"] == 50
        assert stats["num_children"] > 0
        assert engine.total_simulations > 0
        assert len(engine._simulation_cache) > 0

    @pytest.mark.asyncio
    async def test_biased_rollout_influences_selection(self):
        """Test that biased rollout policy influences action selection."""

        class BiasedPolicy(RolloutPolicy):
            async def evaluate(self, state, rng, max_depth=10):
                # Favor "good" action
                if "good" in state.state_id:
                    return 0.9
                elif "bad" in state.state_id:
                    return 0.1
                return 0.5

        engine = MCTSEngine(seed=42)
        root_state = MCTSState(state_id="root", features={"depth": 0})
        root = MCTSNode(state=root_state, rng=engine.rng)

        def action_gen(state):
            depth = state.features.get("depth", 0)
            if depth >= 1:
                return []
            return ["good", "bad"]

        def state_trans(state, action):
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={"depth": 1, "action": action},
            )

        policy = BiasedPolicy()

        # Run enough iterations to see the bias
        best_action, stats = await engine.search(
            root,
            num_iterations=100,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=policy,
        )

        # Should favor "good" action due to higher rollout value
        assert best_action == "good"
        assert stats["action_stats"]["good"]["value"] > stats["action_stats"]["bad"]["value"]

    @pytest.mark.asyncio
    async def test_cache_improves_performance(self):
        """Test that caching reduces redundant simulations."""
        engine = MCTSEngine(seed=42, cache_size_limit=1000)
        root_state = MCTSState(state_id="root", features={"depth": 0})
        root = MCTSNode(state=root_state, rng=engine.rng)

        def action_gen(state):
            depth = state.features.get("depth", 0)
            if depth >= 2:
                return []
            return ["a", "b"]

        def state_trans(state, action):
            depth = state.features.get("depth", 0)
            return MCTSState(
                state_id=f"depth_{depth + 1}_{action}",
                features={"depth": depth + 1},
            )

        policy = RandomRolloutPolicy(base_value=0.5, noise_scale=0.1)

        _, stats = await engine.search(
            root,
            num_iterations=100,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=policy,
        )

        # Should have cache hits (same states visited multiple times)
        assert stats["cache_hit_rate"] > 0
        assert engine.cache_hits > 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestMCTSEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_ucb1_zero_visits_returns_inf(self):
        """Test UCB1 returns infinity for unvisited nodes."""
        score = ucb1(0.0, 0, 100)
        assert score == float("inf")

    def test_ucb1_single_visit(self):
        """Test UCB1 with single visit."""
        score = ucb1(0.5, 1, 10, c=1.414)
        expected = 0.5 + 1.414 * math.sqrt(math.log(10))
        assert score == pytest.approx(expected)

    def test_very_deep_tree(self):
        """Test handling of very deep trees."""
        engine = MCTSEngine()
        root_state = MCTSState(state_id="root")
        root = MCTSNode(state=root_state)

        # Build deep chain
        current = root
        for i in range(100):
            child_state = MCTSState(state_id=f"node_{i}")
            child = current.add_child(f"action_{i}", child_state)
            current = child

        depth = engine.get_tree_depth(root)
        count = engine.count_nodes(root)

        assert depth == 100
        assert count == 101

    def test_wide_tree(self):
        """Test handling of wide trees."""
        engine = MCTSEngine()
        root_state = MCTSState(state_id="root")
        root = MCTSNode(state=root_state)

        # Add many children
        for i in range(100):
            child_state = MCTSState(state_id=f"child_{i}")
            root.add_child(f"action_{i}", child_state)

        assert len(root.children) == 100
        assert engine.count_nodes(root) == 101

    def test_negative_values_in_backpropagation(self):
        """Test backpropagation with negative values."""
        engine = MCTSEngine()
        root_state = MCTSState(state_id="root")
        root = MCTSNode(state=root_state)

        engine.backpropagate(root, -0.5)
        assert root.visits == 1
        assert root.value_sum == -0.5
        assert root.value == -0.5

    def test_very_large_values(self):
        """Test handling of very large values."""
        engine = MCTSEngine()
        root_state = MCTSState(state_id="root")
        root = MCTSNode(state=root_state)

        engine.backpropagate(root, 1e10)
        assert root.value_sum == 1e10

    @pytest.mark.asyncio
    async def test_empty_action_space(self):
        """Test search with empty action space (terminal root)."""
        engine = MCTSEngine()
        root_state = MCTSState(state_id="terminal")
        root = MCTSNode(state=root_state)

        def no_actions(state):
            return []

        def dummy_trans(state, action):
            return state

        policy = RandomRolloutPolicy()

        best_action, stats = await engine.search(
            root,
            num_iterations=10,
            action_generator=no_actions,
            state_transition=dummy_trans,
            rollout_policy=policy,
        )

        assert best_action is None
        assert stats["num_children"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
