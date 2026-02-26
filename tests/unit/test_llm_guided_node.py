"""
Unit tests for LLM-Guided MCTS Node.

Tests node state management, UCB1 calculation, tree structure,
and training data serialization.
"""

from __future__ import annotations

import math

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("src.framework.mcts.llm_guided.node")

from src.framework.mcts.llm_guided.node import (
    NodeState,
    NodeStatus,
    create_root_node,
)


class TestNodeState:
    """Tests for NodeState dataclass."""

    def test_create_state(self) -> None:
        """NodeState can be created with required fields."""
        state = NodeState(
            code="def foo(): return 42",
            problem="Write a function that returns 42",
        )
        assert state.code == "def foo(): return 42"
        assert state.problem == "Write a function that returns 42"
        assert state.test_cases == []
        assert state.errors == []

    def test_state_with_test_cases(self) -> None:
        """NodeState can include test cases."""
        state = NodeState(
            code="def add(a, b): return a + b",
            problem="Add two numbers",
            test_cases=["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        )
        assert len(state.test_cases) == 2

    def test_hash_consistency(self) -> None:
        """Same state produces same hash."""
        state = NodeState(code="def foo(): pass", problem="test")
        hash1 = state.to_hash_key()
        hash2 = state.to_hash_key()
        assert hash1 == hash2

    def test_different_states_different_hashes(self) -> None:
        """Different states produce different hashes."""
        state1 = NodeState(code="def foo(): pass", problem="test")
        state2 = NodeState(code="def bar(): pass", problem="test")
        assert state1.to_hash_key() != state2.to_hash_key()

    def test_hash_includes_problem(self) -> None:
        """Hash should include problem text."""
        state1 = NodeState(code="x = 1", problem="problem A")
        state2 = NodeState(code="x = 1", problem="problem B")
        assert state1.to_hash_key() != state2.to_hash_key()

    def test_with_new_code_preserves_problem(self) -> None:
        """with_new_code preserves problem and tests."""
        original = NodeState(
            code="def foo(): pass",
            problem="Write a function",
            test_cases=["assert True"],
            errors=["Error 1"],
        )

        new = original.with_new_code("def bar(): pass")

        assert new.code == "def bar(): pass"
        assert new.problem == "Write a function"
        assert new.test_cases == ["assert True"]
        assert "def foo(): pass" in new.attempt_history

    def test_with_new_code_updates_attempt_history(self) -> None:
        """with_new_code adds old code to attempt history."""
        state = NodeState(code="v1", problem="test")
        state2 = state.with_new_code("v2")
        state3 = state2.with_new_code("v3")

        assert "v1" in state3.attempt_history
        assert "v2" in state3.attempt_history
        assert state3.code == "v3"


class TestLLMGuidedMCTSNode:
    """Tests for MCTS node operations."""

    def test_create_root_node(self) -> None:
        """Root node can be created."""
        root = create_root_node(
            problem="Test problem",
            test_cases=["assert True"],
            seed=42,
        )

        # Root has no parent
        assert root.parent is None
        assert root.depth == 0
        assert root.visits == 0
        assert root.value_sum == 0.0
        assert root.status == NodeStatus.UNEXPANDED

    def test_add_child(self) -> None:
        """Children can be added to nodes."""
        root = create_root_node("test", seed=42)
        child_state = NodeState(code="child code", problem="test")

        child = root.add_child(child_state, action="action_0")

        assert child.parent == root
        assert child.depth == 1
        assert child in root.children
        assert child.action == "action_0"
        # Child has a parent (not root)
        assert child.parent is not None

    def test_add_multiple_children(self) -> None:
        """Multiple children can be added."""
        root = create_root_node("test", seed=42)

        child1 = root.add_child(NodeState(code="a", problem="test"), action="a")
        child2 = root.add_child(NodeState(code="b", problem="test"), action="b")
        child3 = root.add_child(NodeState(code="c", problem="test"), action="c")

        assert len(root.children) == 3
        assert child1 in root.children
        assert child2 in root.children
        assert child3 in root.children

    def test_leaf_detection(self) -> None:
        """Nodes without children are leaves."""
        root = create_root_node("test", seed=42)
        # Initially root has no children (is a leaf)
        assert len(root.children) == 0

        child = root.add_child(NodeState(code="x", problem="test"), action="a")
        # Now root has children (not a leaf)
        assert len(root.children) > 0
        # Child has no children (is a leaf)
        assert len(child.children) == 0


class TestUCB1Calculation:
    """Tests for UCB1 score calculation."""

    def test_ucb1_unvisited_node(self) -> None:
        """Unvisited nodes have infinite UCB score."""
        root = create_root_node("test", seed=42)
        root.visits = 10

        child = root.add_child(NodeState(code="a", problem="test"), action="a")
        # Child has 0 visits

        ucb = child.ucb1(c=1.414)
        assert ucb == float("inf")

    def test_ucb1_calculation(self) -> None:
        """UCB1 combines exploitation and exploration."""
        root = create_root_node("test", seed=42)
        root.visits = 10

        child = root.add_child(NodeState(code="a", problem="test"), action="a")
        child.visits = 5
        child.value_sum = 3.0  # q_value = 0.6

        ucb = child.ucb1(c=1.414)

        # exploitation: 0.6
        # exploration: 1.414 * sqrt(ln(10)/5) ≈ 1.414 * 0.68 ≈ 0.96
        expected_exploitation = 0.6
        expected_exploration = 1.414 * math.sqrt(math.log(10) / 5)
        expected_ucb = expected_exploitation + expected_exploration

        assert ucb == pytest.approx(expected_ucb, abs=0.01)

    def test_ucb1_exploration_constant_effect(self) -> None:
        """Higher exploration constant increases exploration term."""
        root = create_root_node("test", seed=42)
        root.visits = 100

        child = root.add_child(NodeState(code="a", problem="test"), action="a")
        child.visits = 10
        child.value_sum = 5.0

        ucb_low = child.ucb1(c=1.0)
        ucb_high = child.ucb1(c=2.0)

        assert ucb_high > ucb_low

    def test_ucb1_with_zero_parent_visits(self) -> None:
        """UCB1 handles zero parent visits gracefully."""
        root = create_root_node("test", seed=42)
        root.visits = 0

        child = root.add_child(NodeState(code="a", problem="test"), action="a")
        child.visits = 1
        child.value_sum = 0.5

        # Should not raise, returns just exploitation
        ucb = child.ucb1(c=1.414)
        assert ucb == pytest.approx(0.5)


class TestNodeSelection:
    """Tests for child selection."""

    def test_select_child_picks_unvisited_first(self) -> None:
        """select_child prefers unvisited nodes."""
        root = create_root_node("test", seed=42)
        root.visits = 10

        visited = root.add_child(NodeState(code="a", problem="test"), action="a")
        visited.visits = 5
        visited.value_sum = 5.0

        unvisited = root.add_child(NodeState(code="b", problem="test"), action="b")
        # unvisited.visits = 0

        selected = root.select_child(c=1.414)
        assert selected == unvisited

    def test_select_child_picks_best_ucb(self) -> None:
        """select_child returns highest UCB child."""
        root = create_root_node("test", seed=42)
        root.visits = 100

        high_value = root.add_child(NodeState(code="a", problem="test"), action="a")
        high_value.visits = 10
        high_value.value_sum = 9.0  # q = 0.9

        low_value = root.add_child(NodeState(code="b", problem="test"), action="b")
        low_value.visits = 10
        low_value.value_sum = 1.0  # q = 0.1

        selected = root.select_child(c=1.414)
        assert selected == high_value

    def test_select_child_no_children(self) -> None:
        """select_child returns None if no children."""
        root = create_root_node("test", seed=42)
        assert root.select_child() is None


class TestBackpropagation:
    """Tests for value backpropagation."""

    def test_backpropagate_single_node(self) -> None:
        """Backpropagation updates single node."""
        node = create_root_node("test", seed=42)

        node.backpropagate(0.8)

        assert node.visits == 1
        assert node.value_sum == pytest.approx(0.8)

    def test_backpropagate_updates_all_ancestors(self) -> None:
        """Backpropagation updates all nodes to root."""
        root = create_root_node("test", seed=42)
        child = root.add_child(NodeState(code="a", problem="test"), action="a")
        grandchild = child.add_child(NodeState(code="b", problem="test"), action="b")

        grandchild.backpropagate(1.0)

        assert grandchild.visits == 1
        assert grandchild.value_sum == pytest.approx(1.0)
        assert child.visits == 1
        assert child.value_sum == pytest.approx(1.0)
        assert root.visits == 1
        assert root.value_sum == pytest.approx(1.0)

    def test_backpropagate_accumulates(self) -> None:
        """Multiple backpropagations accumulate values."""
        root = create_root_node("test", seed=42)
        child = root.add_child(NodeState(code="a", problem="test"), action="a")

        child.backpropagate(0.5)
        child.backpropagate(0.7)
        child.backpropagate(0.3)

        assert child.visits == 3
        assert child.value_sum == pytest.approx(1.5)
        assert child.q_value == pytest.approx(0.5)

        assert root.visits == 3
        assert root.value_sum == pytest.approx(1.5)


class TestQValue:
    """Tests for Q-value calculation."""

    def test_q_value_unvisited(self) -> None:
        """Unvisited node has q_value of 0."""
        node = create_root_node("test", seed=42)
        assert node.q_value == 0.0

    def test_q_value_calculation(self) -> None:
        """Q-value is average of backpropagated values."""
        node = create_root_node("test", seed=42)
        node.visits = 4
        node.value_sum = 2.0

        assert node.q_value == pytest.approx(0.5)


class TestMCTSPolicy:
    """Tests for MCTS policy computation."""

    def test_compute_mcts_policy_from_visits(self) -> None:
        """MCTS policy is computed from visit counts."""
        root = create_root_node("test", seed=42)

        child_a = root.add_child(NodeState(code="a", problem="test"), action="a")
        child_a.visits = 70

        child_b = root.add_child(NodeState(code="b", problem="test"), action="b")
        child_b.visits = 30

        policy = root.compute_mcts_policy()

        assert policy["a"] == pytest.approx(0.7)
        assert policy["b"] == pytest.approx(0.3)

    def test_compute_mcts_policy_normalizes(self) -> None:
        """MCTS policy sums to 1.0."""
        root = create_root_node("test", seed=42)

        for i in range(5):
            child = root.add_child(NodeState(code=str(i), problem="test"), action=f"action_{i}")
            child.visits = (i + 1) * 10

        policy = root.compute_mcts_policy()

        total = sum(policy.values())
        assert total == pytest.approx(1.0)

    def test_compute_mcts_policy_no_children(self) -> None:
        """MCTS policy is empty for leaf nodes."""
        root = create_root_node("test", seed=42)
        policy = root.compute_mcts_policy()
        assert policy == {}


class TestBestChild:
    """Tests for best child selection."""

    def test_get_best_child_by_visits(self) -> None:
        """get_best_child returns most visited child."""
        root = create_root_node("test", seed=42)

        child_a = root.add_child(NodeState(code="a", problem="test"), action="a")
        child_a.visits = 50

        child_b = root.add_child(NodeState(code="b", problem="test"), action="b")
        child_b.visits = 100

        # get_best_child selects by visits (no parameter)
        best = root.get_best_child()
        assert best == child_b

    def test_get_best_child_distinguishes_visits(self) -> None:
        """get_best_child correctly distinguishes visit counts."""
        root = create_root_node("test", seed=42)

        child_a = root.add_child(NodeState(code="a", problem="test"), action="a")
        child_a.visits = 100
        child_a.value_sum = 50  # q = 0.5

        child_b = root.add_child(NodeState(code="b", problem="test"), action="b")
        child_b.visits = 50
        child_b.value_sum = 45  # q = 0.9 (higher value but fewer visits)

        # Best child is by visits, so child_a should win
        best = root.get_best_child()
        assert best == child_a

    def test_get_best_child_no_children(self) -> None:
        """get_best_child returns None for leaf nodes."""
        root = create_root_node("test", seed=42)
        assert root.get_best_child() is None


class TestPathToRoot:
    """Tests for path reconstruction."""

    def test_get_path_to_root(self) -> None:
        """Path from leaf to root is correct."""
        root = create_root_node("test", seed=42)
        child = root.add_child(NodeState(code="a", problem="test"), action="a")
        grandchild = child.add_child(NodeState(code="b", problem="test"), action="b")

        path = grandchild.get_path_to_root()

        # Path is reversed: from root to this node
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child
        assert path[2] == grandchild

    def test_get_path_to_root_single_node(self) -> None:
        """Path for root node is just root."""
        root = create_root_node("test", seed=42)
        path = root.get_path_to_root()

        assert len(path) == 1
        assert path[0] == root


class TestTerminalStatus:
    """Tests for terminal node detection."""

    def test_is_terminal_unexpanded(self) -> None:
        """Unexpanded nodes are not terminal."""
        node = create_root_node("test", seed=42)
        assert node.status == NodeStatus.UNEXPANDED
        assert node.is_terminal is False

    def test_is_terminal_expanded(self) -> None:
        """Expanded nodes are not terminal."""
        node = create_root_node("test", seed=42)
        node.status = NodeStatus.EXPANDED
        assert node.is_terminal is False

    def test_is_terminal_success(self) -> None:
        """Terminal success nodes are terminal."""
        node = create_root_node("test", seed=42)
        node.status = NodeStatus.TERMINAL_SUCCESS
        assert node.is_terminal is True
        assert node.is_solution is True

    def test_is_terminal_failure(self) -> None:
        """Terminal failure nodes are terminal."""
        node = create_root_node("test", seed=42)
        node.status = NodeStatus.TERMINAL_FAILURE
        assert node.is_terminal is True
        assert node.is_solution is False


class TestTrainingDataSerialization:
    """Tests for training data serialization."""

    def test_to_training_dict(self) -> None:
        """Node can be serialized to training dict."""
        root = create_root_node("test problem", seed=42)
        child = root.add_child(NodeState(code="x = 1", problem="test problem"), action="a")
        child.visits = 10
        child.value_sum = 8.0

        data = child.to_training_dict()

        # Check that required fields exist
        assert "state" in data
        assert "depth" in data
        assert "visits" in data
        assert "q_value" in data
        # State is a nested dict
        assert data["state"]["code"] == "x = 1"
        assert data["state"]["problem"] == "test problem"
        assert data["visits"] == 10
        assert data["q_value"] == pytest.approx(0.8)

    def test_to_training_dict_includes_policies(self) -> None:
        """Training dict includes LLM and MCTS policies."""
        root = create_root_node("test", seed=42)

        child_a = root.add_child(NodeState(code="a", problem="test"), action="a")
        child_a.visits = 70
        # Set LLM action probs on parent
        root.llm_action_probs = {"a": 0.6, "b": 0.4}

        child_b = root.add_child(NodeState(code="b", problem="test"), action="b")
        child_b.visits = 30

        # Compute MCTS policy
        root.compute_mcts_policy()

        data = root.to_training_dict()

        assert "llm_action_probs" in data
        assert "mcts_action_probs" in data
        # LLM probs were set manually
        assert data["llm_action_probs"] == {"a": 0.6, "b": 0.4}
        # MCTS probs computed from visits
        assert data["mcts_action_probs"]["a"] == pytest.approx(0.7)
        assert data["mcts_action_probs"]["b"] == pytest.approx(0.3)

    def test_to_training_dict_includes_all_fields(self) -> None:
        """Training dict includes all expected fields."""
        root = create_root_node("test", episode_id="ep123", seed=42)
        root.llm_value_estimate = 0.75
        root.test_results = {"passed": 3, "failed": 1}

        data = root.to_training_dict()

        expected_keys = {
            "state",
            "action",
            "depth",
            "visits",
            "q_value",
            "llm_action_probs",
            "llm_value_estimate",
            "mcts_action_probs",
            "episode_id",
            "timestamp",
            "test_results",
            "is_terminal",
            "is_solution",
        }

        for key in expected_keys:
            assert key in data, f"Missing expected key: {key}"

        assert data["episode_id"] == "ep123"
        assert data["llm_value_estimate"] == pytest.approx(0.75)
        assert data["test_results"] == {"passed": 3, "failed": 1}


class TestNodeRepr:
    """Tests for node string representation."""

    def test_repr_format(self) -> None:
        """Node repr contains key information."""
        node = create_root_node("test", seed=42)
        node.visits = 10
        node.value_sum = 5.0

        repr_str = repr(node)

        assert "depth=0" in repr_str
        assert "visits=10" in repr_str
        assert "q=0.500" in repr_str
        assert "children=0" in repr_str
        assert "unexpanded" in repr_str


class TestNodeStateRepr:
    """Tests for node state string representation."""

    def test_state_repr_short_code(self) -> None:
        """Short code is shown in full."""
        state = NodeState(code="x = 1", problem="test")
        repr_str = repr(state)
        assert "x = 1" in repr_str

    def test_state_repr_long_code_truncated(self) -> None:
        """Long code is truncated in repr."""
        long_code = "x = " + "1" * 100
        state = NodeState(code=long_code, problem="test")
        repr_str = repr(state)
        assert "..." in repr_str
        assert len(repr_str) < 100
