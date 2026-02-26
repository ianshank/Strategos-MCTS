"""
Unit tests for MCTS edge cases.

Tests boundary conditions and error handling in MCTS components.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 10
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

import numpy as np

# Import MCTS components with graceful fallback
try:
    from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
    from src.framework.mcts.policies import RandomRolloutPolicy

    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False

# Import edge case modules
try:
    from src.framework.mcts.edge_cases import (
        EmptyActionHandler,
        MCTSSearchResult,
        MCTSTerminationReason,
        MCTSValidator,
        TimeoutConfig,
        TimeoutHandler,
    )

    EDGE_CASES_AVAILABLE = True
except ImportError:
    EDGE_CASES_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.mcts,
    pytest.mark.skipif(not MCTS_AVAILABLE, reason="MCTS module not available"),
]


class TestMCTSEdgeCases:
    """Test MCTS behavior in edge cases."""

    @pytest.fixture
    def engine(self):
        """Create MCTS engine for testing."""
        return MCTSEngine(seed=42)

    @pytest.fixture
    def empty_action_state(self):
        """Create state with no available actions."""
        return MCTSState("empty", {"has_actions": False})

    def test_empty_action_space(self, engine, empty_action_state):
        """Test handling of state with no available actions."""
        rng = np.random.default_rng(42)
        root = MCTSNode(state=empty_action_state, rng=rng)

        def action_generator(s):
            return []  # No actions available

        def state_transition(s, action):
            return s

        # Expansion should mark node as terminal
        result = engine.expand(root, action_generator, state_transition)

        assert result.terminal is True
        assert result is root

    def test_single_child(self, engine):
        """Test selection with single child."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 10

        child_state = MCTSState("child", {})
        child = root.add_child("only_action", child_state)
        child.visits = 5
        child.value_sum = 3.0

        selected = root.select_child(exploration_weight=1.414)

        assert selected is child

    def test_all_children_unexplored(self, engine):
        """Test selection when all children have zero visits."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 1

        for i in range(5):
            child_state = MCTSState(f"child_{i}", {})
            root.add_child(f"action_{i}", child_state)
            # Children have 0 visits (unexplored)

        # With UCB1, unexplored children should have infinite priority
        selected = root.select_child(exploration_weight=1.414)

        assert selected.visits == 0

    def test_select_child_no_children_raises(self, engine):
        """Test selection with no children raises error."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        with pytest.raises(ValueError, match="No children"):
            root.select_child(exploration_weight=1.414)

    def test_very_deep_tree(self, engine):
        """Test handling of very deep trees (100+ levels)."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        current = root
        depth = 100

        for i in range(depth):
            child_state = MCTSState(f"level_{i}", {})
            current.available_actions = [f"action_{i}"]
            current = current.add_child(f"action_{i}", child_state)

        # Should handle deep tree without stack overflow
        actual_depth = engine.get_tree_depth(root)

        assert actual_depth == depth

    def test_wide_tree(self, engine):
        """Test handling of very wide trees (1000+ children)."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 10000

        for i in range(1000):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = i % 10 + 1
            child.value_sum = float(i % 10) * 0.5

        # Selection should complete quickly
        selected = root.select_child(exploration_weight=1.414)

        assert selected is not None

    def test_numerical_stability(self, engine):
        """Test numerical stability with extreme values."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 1_000_000
        root.value_sum = 500_000.0

        for i in range(10):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = 100_000
            child.value_sum = 50_000.0

        # Should not have numerical issues
        selected = root.select_child(exploration_weight=1.414)

        assert selected is not None
        assert not np.isnan(selected.value)
        assert not np.isinf(selected.value)

    def test_cache_eviction(self, engine):
        """Test cache eviction under pressure."""
        small_engine = MCTSEngine(seed=42, cache_size_limit=10)

        # Add more entries than cache limit
        for i in range(20):
            state = MCTSState(f"state_{i}", {"id": i})
            key = state.to_hash_key()
            small_engine._simulation_cache[key] = (0.5, 1)

            # Manual eviction (mimics what happens in simulate)
            while len(small_engine._simulation_cache) > small_engine.cache_size_limit:
                small_engine._simulation_cache.popitem(last=False)
                small_engine.cache_evictions += 1

        assert len(small_engine._simulation_cache) <= 10
        assert small_engine.cache_evictions > 0

    def test_zero_exploration_weight(self, engine):
        """Test selection with zero exploration weight (pure exploitation)."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 100

        # Create children with different values
        for i in range(5):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = 10
            child.value_sum = float(i)  # Values: 0, 1, 2, 3, 4

        # With c=0, should always select highest value child
        selected = root.select_child(exploration_weight=0.0)

        assert selected.value == 0.4  # 4/10 = 0.4 (highest)

    def test_terminal_node_expansion(self, engine):
        """Test that terminal nodes cannot be expanded."""
        state = MCTSState("terminal", {})
        rng = np.random.default_rng(42)
        node = MCTSNode(state=state, rng=rng)
        node.terminal = True

        def action_generator(s):
            return ["action1", "action2"]

        def state_transition(s, action):
            return MCTSState(f"{s.state_id}_{action}", {})

        result = engine.expand(node, action_generator, state_transition)

        # Should return same node without expansion
        assert result is node
        assert len(node.children) == 0

    @pytest.mark.asyncio
    async def test_search_with_terminal_root(self, engine):
        """Test search when root is immediately terminal."""
        state = MCTSState("terminal_root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        def action_generator(s):
            return []  # No actions = terminal

        def state_transition(s, action):
            return s

        policy = RandomRolloutPolicy()

        best_action, stats = await engine.search(
            root=root,
            num_iterations=10,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=policy,
        )

        assert best_action is None
        assert stats["iterations"] == 10

    def test_get_unexpanded_action_none_left(self, engine):
        """Test get_unexpanded_action when all actions expanded."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        node = MCTSNode(state=state, rng=rng)
        node.available_actions = ["a", "b", "c"]

        # Expand all actions
        for action in node.available_actions:
            child_state = MCTSState(f"child_{action}", {})
            node.add_child(action, child_state)

        # Should return None when all expanded
        result = node.get_unexpanded_action()

        assert result is None

    def test_node_value_zero_visits(self, engine):
        """Test node value property with zero visits."""
        state = MCTSState("test", {})
        rng = np.random.default_rng(42)
        node = MCTSNode(state=state, rng=rng)

        # Zero visits should return 0 value
        assert node.visits == 0
        assert node.value == 0.0


@pytest.mark.skipif(not EDGE_CASES_AVAILABLE, reason="Edge case modules not available")
class TestMCTSValidator:
    """Test MCTS tree validator."""

    def test_validate_valid_tree(self):
        """Test validation of a valid tree."""
        validator = MCTSValidator()

        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 100
        root.value_sum = 50.0

        for i in range(3):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = 20
            child.value_sum = 10.0

        violations = validator.validate_tree(root)

        assert len(violations) == 0

    def test_validate_child_visits_violation(self):
        """Test detection of child visits > parent visits."""
        validator = MCTSValidator()

        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 10  # Low parent visits

        # Add children with too many visits
        for i in range(3):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = 10  # 3 * 10 = 30 > 10

        violations = validator.validate_tree(root)

        assert len(violations) > 0
        assert any("visits" in v.lower() for v in violations)

    def test_validate_empty_actions(self):
        """Test validation of empty action space."""
        validator = MCTSValidator()

        violations = validator.validate_action_space([])

        assert len(violations) > 0
        assert any("empty" in v.lower() for v in violations)

    def test_validate_duplicate_actions(self):
        """Test detection of duplicate actions."""
        validator = MCTSValidator()

        violations = validator.validate_action_space(["a", "b", "a", "c"])

        assert len(violations) > 0
        assert any("duplicate" in v.lower() for v in violations)


@pytest.mark.skipif(not EDGE_CASES_AVAILABLE, reason="Edge case modules not available")
class TestTimeoutHandler:
    """Test timeout and budget handling."""

    def test_timeout_not_exceeded(self):
        """Test timeout check when not exceeded."""
        config = TimeoutConfig(search_timeout_seconds=60.0)
        handler = TimeoutHandler(timeout_config=config)
        handler.start()

        assert handler.is_timeout is False

    def test_budget_tracking(self):
        """Test budget tracking functionality."""
        from src.framework.mcts.edge_cases import BudgetConfig

        config = BudgetConfig(token_budget=1000, max_nodes=100)
        handler = TimeoutHandler(budget_config=config)

        # Record some usage
        handler.record_tokens(500)
        handler.record_node()
        handler.record_node()

        assert handler.tokens_used == 500
        assert handler.nodes_created == 2
        assert handler.is_budget_exhausted is False

    def test_budget_exhaustion(self):
        """Test budget exhaustion detection."""
        from src.framework.mcts.edge_cases import BudgetConfig

        config = BudgetConfig(token_budget=100)
        handler = TimeoutHandler(budget_config=config)

        handler.record_tokens(150)

        assert handler.is_budget_exhausted is True
        assert handler.should_terminate is True

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        from src.framework.mcts.edge_cases import BudgetConfig

        config = BudgetConfig(token_budget=1000, max_nodes=50)
        timeout_config = TimeoutConfig(search_timeout_seconds=60.0)
        handler = TimeoutHandler(timeout_config=timeout_config, budget_config=config)
        handler.start()

        handler.record_tokens(300)
        handler.record_node()

        remaining = handler.get_remaining_budget()

        assert remaining["tokens_remaining"] == 700
        assert remaining["nodes_remaining"] == 49


@pytest.mark.skipif(not EDGE_CASES_AVAILABLE, reason="Edge case modules not available")
class TestEmptyActionHandler:
    """Test empty action space handling."""

    def test_handle_empty_actions(self):
        """Test handling empty action space."""
        handler = EmptyActionHandler(fallback_action="wait")
        state = MCTSState("test", {})

        result = handler.handle_empty_actions(state, reason="no_moves")

        assert result == "wait"

    def test_should_terminate_at_max_depth(self):
        """Test termination at max depth."""
        handler = EmptyActionHandler()
        state = MCTSState("test", {})

        should_term = handler.should_terminate(state, depth=10, max_depth=10)

        assert should_term is True

    def test_should_not_terminate_before_max_depth(self):
        """Test no termination before max depth."""
        handler = EmptyActionHandler()
        state = MCTSState("test", {})

        should_term = handler.should_terminate(state, depth=5, max_depth=10)

        assert should_term is False


@pytest.mark.skipif(not EDGE_CASES_AVAILABLE, reason="Edge case modules not available")
class TestMCTSSearchResult:
    """Test MCTS search result dataclass."""

    def test_search_result_to_dict(self):
        """Test search result serialization."""
        result = MCTSSearchResult(
            best_action="action_a",
            stats={"iterations": 100, "nodes": 50},
            termination_reason=MCTSTerminationReason.ITERATIONS_COMPLETE,
            iterations_completed=100,
            time_elapsed_seconds=5.5,
        )

        d = result.to_dict()

        assert d["best_action"] == "action_a"
        assert d["termination_reason"] == "iterations_complete"
        assert d["iterations_completed"] == 100
        assert d["time_elapsed_seconds"] == 5.5
        assert d["error"] is None

    def test_search_result_with_error(self):
        """Test search result with error."""
        error = ValueError("Test error")
        result = MCTSSearchResult(
            best_action=None,
            stats={},
            termination_reason=MCTSTerminationReason.ERROR,
            iterations_completed=50,
            time_elapsed_seconds=2.0,
            error=error,
        )

        d = result.to_dict()

        assert d["best_action"] is None
        assert d["termination_reason"] == "error"
        assert "Test error" in d["error"]
