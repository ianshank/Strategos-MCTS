"""
Unit tests for MCTS Node and Core Logic.
"""

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.core import MCTSNode, MCTSState
from src.framework.mcts.policies import ucb1


class TestMCTSNode:
    def test_node_initialization(self):
        """Test basic node initialization."""
        state = MCTSState(state_id="root", features={"value": 1})
        node = MCTSNode(state=state)

        assert node.state.state_id == "root"
        assert node.visits == 0
        assert node.value == 0.0
        assert node.children == []
        assert node.parent is None
        assert node.depth == 0

    def test_add_child(self):
        """Test adding a child node."""
        root_state = MCTSState(state_id="root")
        root = MCTSNode(state=root_state)

        child_state = MCTSState(state_id="child")
        child = root.add_child(action="action1", child_state=child_state)

        assert len(root.children) == 1
        assert root.children[0] == child
        assert child.parent == root
        assert child.action == "action1"
        assert child.depth == 1
        assert "action1" in root.expanded_actions

    def test_ucb1_score(self):
        """Test UCB1 score calculation."""
        # Case 1: No visits -> infinity
        assert ucb1(value_sum=0, visits=0, parent_visits=10) == float("inf")

        # Case 2: Standard calculation
        # value = 0.5, visits = 10, parent_visits = 100
        # exploration = c * sqrt(ln(100) / 10)
        # c = 1.414
        # ln(100) approx 4.605
        # sqrt(0.4605) approx 0.678
        # expl approx 1.414 * 0.678 approx 0.959
        # total approx 0.5 + 0.959 = 1.459

        score = ucb1(value_sum=5.0, visits=10, parent_visits=100, c=1.414)
        assert 1.4 < score < 1.5

    def test_select_child(self):
        """Test child selection based on UCB1."""
        root = MCTSNode(state=MCTSState(state_id="root"))
        root.visits = 100

        # Child 1: high value, moderate visits
        child1 = root.add_child("a1", MCTSState("c1"))
        child1.visits = 20
        child1.value_sum = 18.0  # mean 0.9

        # Child 2: low value, low visits (high exploration)
        child2 = root.add_child("a2", MCTSState("c2"))
        child2.visits = 2
        child2.value_sum = 0.2  # mean 0.1

        # Select
        selected = root.select_child(exploration_weight=1.414)

        # Calculate scores manually
        # c1: 0.9 + 1.414 * sqrt(ln(100)/20) = 0.9 + 1.414 * sqrt(4.6/20) = 0.9 + 1.414 * 0.48 = 0.9 + 0.68 = 1.58
        # c2: 0.1 + 1.414 * sqrt(ln(100)/2) = 0.1 + 1.414 * sqrt(2.3) = 0.1 + 1.414 * 1.51 = 0.1 + 2.13 = 2.23

        # Child 2 should be selected due to exploration bonus
        assert selected == child2

    def test_state_hashing(self):
        """Test that states can be hashed correctly for caching."""
        s1 = MCTSState(state_id="s1", features={"a": 1, "b": 2})
        s2 = MCTSState(state_id="s1", features={"b": 2, "a": 1})  # Same content, diff order
        s3 = MCTSState(state_id="s1", features={"a": 2})  # Diff content

        assert s1.to_hash_key() == s2.to_hash_key()
        assert s1.to_hash_key() != s3.to_hash_key()
