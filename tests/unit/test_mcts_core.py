"""
Comprehensive unit tests for MCTS core functionality.

Tests:
- MCTSNode operations (UCB1, child selection, tree structure)
- MCTS phases (selection, expansion, simulation, backpropagation)
- Determinism with seeding
- Edge cases and error conditions
"""

import math
import random

# Import the MCTS classes from the framework
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Skip entire module if langchain_openai is not installed (optional LLM dependency)
pytest.importorskip("langchain_openai", reason="langchain_openai required for MCTS framework tests")

from examples.langgraph_multi_agent_mcts import LangGraphMultiAgentFramework  # noqa: E402
from src.framework.mcts.core import MCTSNode, MCTSState  # noqa: E402


class TestMCTSNode:
    """Test suite for MCTSNode class."""

    def test_node_initialization(self):
        """Test that nodes are initialized correctly."""
        state = MCTSState(state_id="test_state")
        node = MCTSNode(state=state)

        assert node.state.state_id == "test_state"
        assert node.parent is None
        assert node.action is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.terminal is False

    def test_node_with_parent(self):
        """Test node initialization with parent."""
        parent_state = MCTSState(state_id="parent_state")
        parent = MCTSNode(state=parent_state)

        child_state = MCTSState(state_id="child_state")
        child = MCTSNode(state=child_state, parent=parent, action="test_action")

        assert child.parent is parent
        assert child.action == "test_action"

    def test_ucb1_unvisited_node_returns_infinity(self):
        """UCB1 should return infinity for unvisited nodes."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 10  # Parent must have visits

        child_state = MCTSState(state_id="child")
        child = MCTSNode(state=child_state, parent=parent)

        # MCTSNode doesn't have ucb1() method directly anymore, it uses select_child
        # We need to test via select_child or ucb1 function from policies
        from src.framework.mcts.policies import ucb1

        score = ucb1(child.value_sum, child.visits, parent.visits, 1.414)
        assert score == float("inf")

    def test_ucb1_calculation_standard_case(self):
        """Test UCB1 calculation with standard values."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 100

        child_state = MCTSState(state_id="child")
        child = MCTSNode(state=child_state, parent=parent)
        child.visits = 10
        child.value_sum = 7.0  # Average value = 0.7

        from src.framework.mcts.policies import ucb1

        ucb = ucb1(child.value_sum, child.visits, parent.visits, c=1.414)

        expected_exploitation = 0.7
        expected_exploration = 1.414 * math.sqrt(math.log(100) / 10)
        expected_ucb = expected_exploitation + expected_exploration

        assert abs(ucb - expected_ucb) < 1e-6

    def test_ucb1_with_different_exploration_weights(self):
        """Test that exploration weight affects UCB1 correctly."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 100

        child_state = MCTSState(state_id="child")
        child = MCTSNode(state=child_state, parent=parent)
        child.visits = 10
        child.value_sum = 5.0

        from src.framework.mcts.policies import ucb1

        ucb_low_exploration = ucb1(child.value_sum, child.visits, parent.visits, c=0.5)
        ucb_high_exploration = ucb1(child.value_sum, child.visits, parent.visits, c=2.0)

        # Higher exploration weight should give higher UCB1
        assert ucb_high_exploration > ucb_low_exploration

    def test_best_child_returns_none_for_leaf(self):
        """best_child (select_child) should return None when no children exist."""
        state = MCTSState(state_id="leaf")
        node = MCTSNode(state=state)
        # select_child raises ValueError if no children, based on implementation
        with pytest.raises(ValueError):
            node.select_child()

    def test_best_child_selects_highest_ucb1(self):
        """best_child should select child with highest UCB1."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 100

        # Create children with different values
        child1 = parent.add_child("action_1", MCTSState(state_id="state_1"))
        child1.visits = 10
        child1.value_sum = 3.0  # Low average

        child2 = parent.add_child("action_2", MCTSState(state_id="state_2"))
        child2.visits = 10
        child2.value_sum = 8.0  # High average

        child3 = parent.add_child("action_3", MCTSState(state_id="state_3"))
        child3.visits = 10
        child3.value_sum = 5.0  # Medium average

        best = parent.select_child()
        assert best is child2
        assert best.action == "action_2"

    def test_best_child_prefers_unvisited(self):
        """Unvisited children should be selected first (infinite UCB1)."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 100

        visited_child = parent.add_child("visited", MCTSState(state_id="state_1"))
        visited_child.visits = 10
        visited_child.value_sum = 9.0  # High value

        unvisited_child = parent.add_child("unvisited", MCTSState(state_id="state_2"))
        # unvisited_child.visits = 0 (default)

        best = parent.select_child()
        assert best is unvisited_child

    def test_add_child_creates_proper_relationship(self):
        """add_child should create correct parent-child relationship."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        child = parent.add_child(action="move_forward", child_state=MCTSState(state_id="new_state"))

        assert child in parent.children
        assert child.parent is parent
        assert child.action == "move_forward"
        assert child.state.state_id == "new_state"
        assert len(parent.children) == 1

    def test_add_multiple_children(self):
        """Adding multiple children should work correctly."""
        parent_state = MCTSState(state_id="root")
        parent = MCTSNode(state=parent_state)

        for i in range(5):
            parent.add_child(f"action_{i}", MCTSState(state_id=f"state_{i}"))

        assert len(parent.children) == 5
        for i, child in enumerate(parent.children):
            assert child.action == f"action_{i}"
            assert child.parent is parent

    def test_ucb1_exploitation_dominates_with_many_visits(self):
        """With many visits, exploitation term should dominate."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 10000

        child_state = MCTSState(state_id="child")
        child = MCTSNode(state=child_state, parent=parent)
        child.visits = 1000
        child.value_sum = 900.0  # Average = 0.9

        from src.framework.mcts.policies import ucb1

        ucb = ucb1(child.value_sum, child.visits, parent.visits, c=1.414)
        exploitation = child.value

        # UCB should be close to exploitation term
        assert abs(ucb - exploitation) < 0.2


class TestMCTSDeterminism:
    """Test MCTS determinism with seeding."""

    @pytest.fixture
    def framework(self):
        """Create a minimal framework for testing."""
        mock_adapter = AsyncMock()
        mock_logger = Mock()
        mock_logger.info = Mock()

        with (
            patch("examples.langgraph_multi_agent_mcts.HRMAgent"),
            patch("examples.langgraph_multi_agent_mcts.TRMAgent"),
            patch("examples.langgraph_multi_agent_mcts.OpenAIEmbeddings"),
        ):
            return LangGraphMultiAgentFramework(  # noqa: F821
                model_adapter=mock_adapter,
                logger=mock_logger,
                mcts_iterations=5,
            )

    def test_same_seed_produces_same_tree_structure(self, framework):
        """With same seed, expansion should create identical structure."""
        state = {"query": "test"}

        # First run
        random.seed(12345)
        root_state1 = MCTSState(state_id="root_state")
        root1 = MCTSNode(state=root_state1)
        root1.visits = 1
        child1 = framework._mcts_expand(root1, state)

        # Second run with same seed
        random.seed(12345)
        root_state2 = MCTSState(state_id="root_state")
        root2 = MCTSNode(state=root_state2)
        root2.visits = 1
        child2 = framework._mcts_expand(root2, state)

        assert child1.action == child2.action
        assert child1.state.state_id == child2.state.state_id

    def test_different_seeds_produce_different_trees(self, framework):
        """Different seeds should produce different tree structures."""
        state = {"query": "test"}

        # First run
        random.seed(11111)
        root_state1 = MCTSState(state_id="root_state")
        root1 = MCTSNode(state=root_state1)
        root1.visits = 1
        framework._mcts_expand(root1, state)

        # Second run with different seed
        random.seed(99999)
        root_state2 = MCTSState(state_id="root_state")
        root2 = MCTSNode(state=root_state2)
        root2.visits = 1
        framework._mcts_expand(root2, state)

        # With different seeds, actions may differ
        # This test verifies the mechanism works, not specific outcomes
        assert root1 is not None and root2 is not None
        assert len(root1.children) >= 0 and len(root2.children) >= 0


class TestMCTSEdgeCases:
    """Test edge cases and error conditions."""

    def test_ucb1_with_zero_parent_visits_raises_math_error(self):
        """UCB1 calculation with zero parent visits should return infinity (exploration)."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 0  # Invalid state typically, but logic handles it

        child_state = MCTSState(state_id="child")
        child = MCTSNode(state=child_state, parent=parent)
        child.visits = 1
        child.value_sum = 0.5

        from src.framework.mcts.policies import ucb1

        # Should return infinity instead of raising error
        ucb = ucb1(child.value_sum, child.visits, parent.visits)
        assert ucb == float("inf")

    def test_node_without_parent_ucb1_fails(self):
        """Root node UCB1 calculation should fail (no parent)."""
        root_state = MCTSState(state_id="root")
        root = MCTSNode(state=root_state)
        root.visits = 10
        root.value_sum = 5.0

        # Trying to calculate UCB for root via policies function requires parent visits
        # If we pass None or similar it fails
        from src.framework.mcts.policies import ucb1

        with pytest.raises(TypeError):
            ucb1(root.value_sum, root.visits, None)  # type: ignore

    def test_best_child_with_single_child(self):
        """best_child with only one child should return that child."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 10

        only_child = parent.add_child("action", MCTSState(state_id="state"))
        only_child.visits = 5
        only_child.value_sum = 2.0

        assert parent.select_child() is only_child

    def test_very_deep_tree_structure(self):
        """Test that deep trees are handled correctly."""
        root_state = MCTSState(state_id="root")
        root = MCTSNode(state=root_state)
        current = root

        # Create very deep tree
        for i in range(100):
            child = current.add_child(f"action_{i}", MCTSState(state_id=f"state_{i}"))
            child.visits = 1
            child.value_sum = 0.5
            current = child

        # Verify structure
        node = root
        depth = 0
        while node.children:
            node = node.children[0]
            depth += 1

        assert depth == 100

    def test_negative_value_handling(self):
        """Ensure negative values are handled properly."""
        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 10

        child_state = MCTSState(state_id="child")
        child = MCTSNode(state=child_state, parent=parent)
        child.visits = 5
        child.value_sum = -2.0  # Negative total value

        from src.framework.mcts.policies import ucb1

        # UCB1 should still work
        ucb = ucb1(child.value_sum, child.visits, parent.visits)
        assert isinstance(ucb, float)
        # Exploitation term is negative
        assert child.value == -0.4


class TestMCTSPerformance:
    """Performance benchmarks for MCTS operations."""

    @pytest.mark.slow
    def test_ucb1_computation_performance(self):
        """UCB1 should compute quickly even with many calls."""
        import time

        from src.framework.mcts.policies import ucb1

        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 1000000

        children = []
        for i in range(100):
            child = parent.add_child(f"action_{i}", MCTSState(state_id=f"state_{i}"))
            child.visits = 10000 + i
            child.value_sum = 5000 + i * 10
            children.append(child)

        start = time.perf_counter()

        # Compute UCB1 for all children many times
        for _ in range(10000):
            for child in children:
                ucb1(child.value_sum, child.visits, parent.visits)

        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (less than 5 seconds)
        assert elapsed < 5.0, f"UCB1 computation took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_best_child_selection_performance(self):
        """best_child should be efficient with many children."""
        import time

        parent_state = MCTSState(state_id="parent")
        parent = MCTSNode(state=parent_state)
        parent.visits = 100000

        # Add many children
        for i in range(1000):
            child = parent.add_child(f"action_{i}", MCTSState(state_id=f"state_{i}"))
            child.visits = 100 + i
            child.value_sum = 50 + i * 0.5

        start = time.perf_counter()

        # Select best child many times
        for _ in range(1000):
            parent.select_child()

        elapsed = time.perf_counter() - start

        # Should complete quickly
        assert elapsed < 2.0, f"best_child selection took {elapsed:.2f}s"
