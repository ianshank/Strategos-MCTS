"""Tests for LLM-Guided MCTS Node."""

import math

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.llm_guided.node import (
    LLMGuidedMCTSNode,
    NodeState,
    NodeStatus,
    create_root_node,
)


class TestNodeState:
    """Tests for NodeState."""

    def test_creation(self):
        """Test basic creation."""
        state = NodeState(
            code="def foo(): pass",
            problem="Write a function foo",
        )
        assert state.code == "def foo(): pass"
        assert state.problem == "Write a function foo"
        assert state.test_cases == []
        assert state.errors == []
        assert state.attempt_history == []

    def test_with_full_parameters(self):
        """Test creation with all parameters."""
        state = NodeState(
            code="def foo(): return 1",
            problem="Return 1",
            test_cases=["assert foo() == 1"],
            errors=["NameError"],
            attempt_history=["def foo(): pass"],
            metadata={"source": "test"},
        )
        assert len(state.test_cases) == 1
        assert len(state.errors) == 1
        assert len(state.attempt_history) == 1
        assert state.metadata["source"] == "test"

    def test_to_hash_key(self):
        """Test hash key generation."""
        state1 = NodeState(code="def foo(): pass", problem="test")
        state2 = NodeState(code="def foo(): pass", problem="test")
        state3 = NodeState(code="def bar(): pass", problem="test")

        # Same code should produce same hash
        assert state1.to_hash_key() == state2.to_hash_key()

        # Different code should produce different hash
        assert state1.to_hash_key() != state3.to_hash_key()

    def test_with_new_code(self):
        """Test creating new state with updated code."""
        state1 = NodeState(
            code="def foo(): pass",
            problem="test",
            test_cases=["assert 1 == 1"],
        )

        state2 = state1.with_new_code("def foo(): return 1", errors=["fixed"])

        # New state should have new code
        assert state2.code == "def foo(): return 1"
        assert state2.errors == ["fixed"]

        # Original code should be in history
        assert "def foo(): pass" in state2.attempt_history

        # Problem and tests should be preserved
        assert state2.problem == "test"
        assert state2.test_cases == ["assert 1 == 1"]

        # Original state unchanged
        assert state1.code == "def foo(): pass"
        assert state1.attempt_history == []

    def test_with_new_code_empty_initial(self):
        """Test that empty initial code isn't added to history."""
        state1 = NodeState(code="", problem="test")
        state2 = state1.with_new_code("def foo(): pass")

        assert len(state2.attempt_history) == 0

    def test_repr(self):
        """Test string representation."""
        state = NodeState(code="def foo(): return 1", problem="test")
        repr_str = repr(state)
        assert "NodeState" in repr_str
        assert "def foo()" in repr_str


class TestLLMGuidedMCTSNode:
    """Tests for LLMGuidedMCTSNode."""

    def test_creation(self):
        """Test basic node creation."""
        state = NodeState(code="def foo(): pass", problem="test")
        node = LLMGuidedMCTSNode(state=state)

        assert node.state == state
        assert node.parent is None
        assert node.action is None
        assert node.children == []
        assert node.visits == 0
        assert node.value_sum == 0.0
        assert node.depth == 0
        assert node.status == NodeStatus.UNEXPANDED

    def test_depth_calculation(self):
        """Test that depth is calculated from parent."""
        state = NodeState(code="", problem="test")
        root = LLMGuidedMCTSNode(state=state)
        child1 = LLMGuidedMCTSNode(state=state, parent=root)
        child2 = LLMGuidedMCTSNode(state=state, parent=child1)

        assert root.depth == 0
        assert child1.depth == 1
        assert child2.depth == 2

    def test_q_value_empty(self):
        """Test q_value with no visits."""
        state = NodeState(code="", problem="test")
        node = LLMGuidedMCTSNode(state=state)

        assert node.q_value == 0.0

    def test_q_value_with_visits(self):
        """Test q_value with visits."""
        state = NodeState(code="", problem="test")
        node = LLMGuidedMCTSNode(state=state)
        node.visits = 10
        node.value_sum = 5.0

        assert node.q_value == 0.5

    def test_is_terminal(self):
        """Test terminal node detection."""
        state = NodeState(code="", problem="test")
        node = LLMGuidedMCTSNode(state=state)

        assert node.is_terminal is False

        node.status = NodeStatus.TERMINAL_SUCCESS
        assert node.is_terminal is True

        node.status = NodeStatus.TERMINAL_FAILURE
        assert node.is_terminal is True

    def test_is_solution(self):
        """Test solution node detection."""
        state = NodeState(code="", problem="test")
        node = LLMGuidedMCTSNode(state=state)

        assert node.is_solution is False

        node.status = NodeStatus.TERMINAL_SUCCESS
        assert node.is_solution is True

        node.status = NodeStatus.TERMINAL_FAILURE
        assert node.is_solution is False

    def test_ucb1_unvisited(self):
        """Test UCB1 for unvisited nodes."""
        state = NodeState(code="", problem="test")
        node = LLMGuidedMCTSNode(state=state)

        assert node.ucb1() == float("inf")

    def test_ucb1_with_parent(self):
        """Test UCB1 calculation with parent."""
        state = NodeState(code="", problem="test")
        parent = LLMGuidedMCTSNode(state=state)
        parent.visits = 100

        child = LLMGuidedMCTSNode(state=state, parent=parent)
        child.visits = 10
        child.value_sum = 5.0

        ucb = child.ucb1(c=1.414)

        # UCB1 = Q + c * sqrt(ln(N) / n)
        expected_exploitation = 0.5
        expected_exploration = 1.414 * math.sqrt(math.log(100) / 10)
        expected = expected_exploitation + expected_exploration

        assert abs(ucb - expected) < 0.001

    def test_select_child(self):
        """Test UCB1-based child selection."""
        state = NodeState(code="", problem="test")
        root = LLMGuidedMCTSNode(state=state)
        root.visits = 100

        # Add children with different values
        child1 = LLMGuidedMCTSNode(state=state, parent=root)
        child1.visits = 50
        child1.value_sum = 25.0  # q = 0.5
        root.children.append(child1)

        child2 = LLMGuidedMCTSNode(state=state, parent=root)
        child2.visits = 10
        child2.value_sum = 8.0  # q = 0.8
        root.children.append(child2)

        # Child2 should have higher UCB1 due to lower visits
        selected = root.select_child()
        assert selected == child2

    def test_select_child_no_children(self):
        """Test select_child with no children."""
        state = NodeState(code="", problem="test")
        node = LLMGuidedMCTSNode(state=state)

        assert node.select_child() is None

    def test_add_child(self):
        """Test adding child nodes."""
        state1 = NodeState(code="", problem="test")
        state2 = NodeState(code="def foo(): pass", problem="test")

        parent = LLMGuidedMCTSNode(state=state1, episode_id="ep1")
        child = parent.add_child(
            state=state2,
            action="variant_0",
            llm_action_probs={"variant_0": 0.8, "variant_1": 0.2},
            episode_id="ep1",
        )

        assert len(parent.children) == 1
        assert child.parent == parent
        assert child.action == "variant_0"
        assert child.llm_action_probs == {"variant_0": 0.8, "variant_1": 0.2}
        assert child.episode_id == "ep1"
        assert child.depth == 1

    def test_backpropagate(self):
        """Test reward backpropagation."""
        state = NodeState(code="", problem="test")
        root = LLMGuidedMCTSNode(state=state)
        child = LLMGuidedMCTSNode(state=state, parent=root)
        grandchild = LLMGuidedMCTSNode(state=state, parent=child)

        grandchild.backpropagate(1.0)

        assert grandchild.visits == 1
        assert grandchild.value_sum == 1.0
        assert child.visits == 1
        assert child.value_sum == 1.0
        assert root.visits == 1
        assert root.value_sum == 1.0

    def test_compute_mcts_policy(self):
        """Test MCTS policy computation from visit counts."""
        state = NodeState(code="", problem="test")
        root = LLMGuidedMCTSNode(state=state)

        # Add children with different visit counts
        child1 = root.add_child(
            state=state,
            action="action_a",
        )
        child1.visits = 30

        child2 = root.add_child(
            state=state,
            action="action_b",
        )
        child2.visits = 70

        policy = root.compute_mcts_policy()

        assert policy["action_a"] == 0.3
        assert policy["action_b"] == 0.7
        assert root.mcts_action_probs == policy

    def test_compute_mcts_policy_no_children(self):
        """Test MCTS policy with no children."""
        state = NodeState(code="", problem="test")
        node = LLMGuidedMCTSNode(state=state)

        policy = node.compute_mcts_policy()
        assert policy == {}

    def test_get_best_child(self):
        """Test getting best child by visit count."""
        state = NodeState(code="", problem="test")
        root = LLMGuidedMCTSNode(state=state)

        child1 = root.add_child(state=state, action="a")
        child1.visits = 10

        child2 = root.add_child(state=state, action="b")
        child2.visits = 50

        best = root.get_best_child()
        assert best == child2

    def test_get_path_to_root(self):
        """Test getting path from node to root."""
        state = NodeState(code="", problem="test")
        root = LLMGuidedMCTSNode(state=state)
        child = LLMGuidedMCTSNode(state=state, parent=root)
        grandchild = LLMGuidedMCTSNode(state=state, parent=child)

        path = grandchild.get_path_to_root()

        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child
        assert path[2] == grandchild

    def test_to_training_dict(self):
        """Test converting node to training dictionary."""
        state = NodeState(
            code="def foo(): return 1",
            problem="Return 1",
            test_cases=["assert foo() == 1"],
        )
        node = LLMGuidedMCTSNode(
            state=state,
            action="variant_0",
            episode_id="test_ep",
        )
        node.visits = 10
        node.value_sum = 5.0
        node.llm_action_probs = {"variant_0": 0.8}
        node.llm_value_estimate = 0.7
        node.mcts_action_probs = {"variant_0": 0.9}
        node.test_results = {"passed": True}
        node.status = NodeStatus.TERMINAL_SUCCESS

        d = node.to_training_dict()

        assert d["state"]["code"] == "def foo(): return 1"
        assert d["state"]["problem"] == "Return 1"
        assert d["action"] == "variant_0"
        assert d["depth"] == 0
        assert d["visits"] == 10
        assert d["q_value"] == 0.5
        assert d["llm_action_probs"] == {"variant_0": 0.8}
        assert d["llm_value_estimate"] == 0.7
        assert d["mcts_action_probs"] == {"variant_0": 0.9}
        assert d["episode_id"] == "test_ep"
        assert d["test_results"] == {"passed": True}
        assert d["is_terminal"] is True
        assert d["is_solution"] is True


class TestCreateRootNode:
    """Tests for create_root_node factory function."""

    def test_basic_creation(self):
        """Test basic root node creation."""
        root = create_root_node(
            problem="Write a function",
            episode_id="ep123",
        )

        assert root.state.problem == "Write a function"
        assert root.state.code == ""
        assert root.episode_id == "ep123"
        assert root.parent is None
        assert root.depth == 0

    def test_with_initial_code(self):
        """Test root node with initial code."""
        root = create_root_node(
            problem="Fix this function",
            initial_code="def foo(): pass",
            episode_id="ep456",
        )

        assert root.state.code == "def foo(): pass"

    def test_with_test_cases(self):
        """Test root node with test cases."""
        root = create_root_node(
            problem="test",
            test_cases=["assert 1 == 1", "assert 2 == 2"],
            episode_id="ep789",
        )

        assert len(root.state.test_cases) == 2

    def test_with_seed(self):
        """Test root node with deterministic seed."""
        root1 = create_root_node(problem="test", seed=42)
        root2 = create_root_node(problem="test", seed=42)

        # Should have same RNG state
        assert root1._rng is not None
        assert root2._rng is not None


class TestNodeStatus:
    """Tests for NodeStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert NodeStatus.UNEXPANDED.value == "unexpanded"
        assert NodeStatus.EXPANDED.value == "expanded"
        assert NodeStatus.TERMINAL_SUCCESS.value == "terminal_success"
        assert NodeStatus.TERMINAL_FAILURE.value == "terminal_failure"
