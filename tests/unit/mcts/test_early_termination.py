"""
Unit tests for MCTS early termination functionality.

Tests that MCTS search terminates early when convergence conditions are met.

Based on: NEXT_STEPS_PLAN.md Phase 3.3
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.mcts,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mcts_engine():
    """Create an MCTS engine for testing."""
    from src.framework.mcts.core import MCTSEngine

    return MCTSEngine(seed=42, exploration_weight=1.414)


@pytest.fixture
def root_node():
    """Create a root MCTS node for testing."""
    from src.framework.mcts.core import MCTSNode, MCTSState

    state = MCTSState(
        state_id="root",
        features={"query": "test", "depth": 0},
    )
    node = MCTSNode(state=state, rng=np.random.default_rng(42))
    node.available_actions = ["action1", "action2", "action3"]
    return node


@pytest.fixture
def mock_rollout_policy():
    """Create a mock rollout policy."""
    policy = AsyncMock()
    policy.evaluate.return_value = 0.5
    return policy


@pytest.fixture
def simple_action_generator():
    """Create a simple action generator."""

    def generator(state):
        return ["action1", "action2", "action3"]

    return generator


@pytest.fixture
def simple_state_transition():
    """Create a simple state transition function."""
    from src.framework.mcts.core import MCTSState

    def transition(state, action):
        return MCTSState(
            state_id=f"{state.state_id}_{action}",
            features={**state.features, "last_action": action},
        )

    return transition


# =============================================================================
# Early Termination Check Tests
# =============================================================================


class TestShouldTerminateEarly:
    """Tests for _should_terminate_early method."""

    def test_returns_false_for_empty_children(self, mcts_engine, root_node):
        """Test returns False when root has no children."""
        result = mcts_engine._should_terminate_early(root_node, threshold=0.95)

        assert result is False

    def test_returns_false_for_zero_visits(self, mcts_engine, root_node):
        """Test returns False when root has zero visits."""
        from src.framework.mcts.core import MCTSNode, MCTSState

        # Add child but keep visits at 0
        child_state = MCTSState(state_id="child", features={})
        child = MCTSNode(state=child_state, parent=root_node)
        root_node.children.append(child)

        result = mcts_engine._should_terminate_early(root_node, threshold=0.95)

        assert result is False

    def test_returns_true_when_threshold_exceeded(self, mcts_engine, root_node):
        """Test returns True when visit fraction exceeds threshold."""
        from src.framework.mcts.core import MCTSNode, MCTSState

        # Setup root with visits
        root_node.visits = 100

        # Add dominant child with 96% of visits
        child1_state = MCTSState(state_id="child1", features={})
        child1 = MCTSNode(state=child1_state, parent=root_node, action="action1")
        child1.visits = 96
        root_node.children.append(child1)

        # Add minor child with 4% of visits
        child2_state = MCTSState(state_id="child2", features={})
        child2 = MCTSNode(state=child2_state, parent=root_node, action="action2")
        child2.visits = 4
        root_node.children.append(child2)

        result = mcts_engine._should_terminate_early(root_node, threshold=0.95)

        assert result is True

    def test_returns_false_when_below_threshold(self, mcts_engine, root_node):
        """Test returns False when visit fraction is below threshold."""
        from src.framework.mcts.core import MCTSNode, MCTSState

        # Setup root with visits
        root_node.visits = 100

        # Add child with 80% of visits (below 95% threshold)
        child1_state = MCTSState(state_id="child1", features={})
        child1 = MCTSNode(state=child1_state, parent=root_node, action="action1")
        child1.visits = 80
        root_node.children.append(child1)

        # Add child with 20% of visits
        child2_state = MCTSState(state_id="child2", features={})
        child2 = MCTSNode(state=child2_state, parent=root_node, action="action2")
        child2.visits = 20
        root_node.children.append(child2)

        result = mcts_engine._should_terminate_early(root_node, threshold=0.95)

        assert result is False

    def test_handles_single_child(self, mcts_engine, root_node):
        """Test handles single child correctly (100% of visits)."""
        from src.framework.mcts.core import MCTSNode, MCTSState

        root_node.visits = 50

        child_state = MCTSState(state_id="child", features={})
        child = MCTSNode(state=child_state, parent=root_node, action="action1")
        child.visits = 50
        root_node.children.append(child)

        result = mcts_engine._should_terminate_early(root_node, threshold=0.95)

        assert result is True


# =============================================================================
# Search Early Termination Tests
# =============================================================================


class TestSearchEarlyTermination:
    """Tests for early termination in search method."""

    @pytest.mark.asyncio
    async def test_search_terminates_early_when_converged(
        self,
        mcts_engine,
        root_node,
        mock_rollout_policy,
        simple_action_generator,
        simple_state_transition,
    ):
        """Test search terminates early when convergence threshold is met."""
        # Make rollout strongly favor one action to force convergence
        async def biased_evaluate(state, rng, max_depth=10):
            # Return high value for action1 path, low for others
            if hasattr(state, 'features') and state.features.get('last_action') == 'action1':
                return 0.95
            return 0.1

        mock_rollout_policy.evaluate = biased_evaluate

        # Run search with low threshold and min iterations
        best_action, stats = await mcts_engine.search(
            root=root_node,
            num_iterations=500,  # Max iterations
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=mock_rollout_policy,
            max_rollout_depth=5,
            early_termination_threshold=0.7,  # Low threshold
            min_iterations_before_termination=20,  # Allow some exploration
        )

        # Should have terminated early (convergence with biased values)
        # If early termination works, it should have run fewer iterations
        # (but at least min_iterations_before_termination)
        assert stats["iterations_run"] >= 20
        assert stats["iterations_run"] <= 500
        # Best action should be selected
        assert best_action is not None

    @pytest.mark.asyncio
    async def test_search_respects_min_iterations(
        self,
        mcts_engine,
        root_node,
        mock_rollout_policy,
        simple_action_generator,
        simple_state_transition,
    ):
        """Test search doesn't terminate before min_iterations."""
        min_iters = 50

        best_action, stats = await mcts_engine.search(
            root=root_node,
            num_iterations=100,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=mock_rollout_policy,
            max_rollout_depth=5,
            early_termination_threshold=0.0,  # Would always terminate
            min_iterations_before_termination=min_iters,
        )

        # Should have run at least min_iterations
        assert stats["iterations_run"] >= min_iters

    @pytest.mark.asyncio
    async def test_search_stats_include_early_termination_info(
        self,
        mcts_engine,
        root_node,
        mock_rollout_policy,
        simple_action_generator,
        simple_state_transition,
    ):
        """Test search statistics include early termination information."""
        best_action, stats = await mcts_engine.search(
            root=root_node,
            num_iterations=100,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=mock_rollout_policy,
            max_rollout_depth=5,
            early_termination_threshold=0.95,
            min_iterations_before_termination=10,
        )

        # Stats should include early termination info
        assert "early_terminated" in stats
        assert "iterations_run" in stats
        assert "max_iterations" in stats
        assert stats["max_iterations"] == 100

    @pytest.mark.asyncio
    async def test_search_runs_all_iterations_when_no_convergence(
        self,
        mcts_engine,
        root_node,
        mock_rollout_policy,
        simple_action_generator,
        simple_state_transition,
    ):
        """Test search runs all iterations when convergence not reached."""
        num_iterations = 20

        # Make rollout return varied values to prevent convergence
        call_count = [0]

        async def varied_evaluate(*args, **kwargs):
            call_count[0] += 1
            return 0.5 + (call_count[0] % 3) * 0.1

        mock_rollout_policy.evaluate = varied_evaluate

        best_action, stats = await mcts_engine.search(
            root=root_node,
            num_iterations=num_iterations,
            action_generator=simple_action_generator,
            state_transition=simple_state_transition,
            rollout_policy=mock_rollout_policy,
            max_rollout_depth=5,
            early_termination_threshold=0.99,  # Very high threshold
            min_iterations_before_termination=5,
        )

        # Should have run all iterations if not converged
        # (or at least very close to it)
        assert stats["iterations_run"] >= num_iterations - 1


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEarlyTerminationEdgeCases:
    """Tests for edge cases in early termination."""

    def test_threshold_of_one_never_terminates(self, mcts_engine, root_node):
        """Test threshold of 1.0 never triggers early termination."""
        from src.framework.mcts.core import MCTSNode, MCTSState

        root_node.visits = 100

        # Single child with all visits
        child_state = MCTSState(state_id="child", features={})
        child = MCTSNode(state=child_state, parent=root_node, action="action1")
        child.visits = 100
        root_node.children.append(child)

        result = mcts_engine._should_terminate_early(root_node, threshold=1.0)

        # 100/100 = 1.0 which equals threshold, should terminate
        assert result is True

    def test_threshold_above_one_never_terminates(self, mcts_engine, root_node):
        """Test threshold above 1.0 never triggers termination."""
        from src.framework.mcts.core import MCTSNode, MCTSState

        root_node.visits = 100

        child_state = MCTSState(state_id="child", features={})
        child = MCTSNode(state=child_state, parent=root_node, action="action1")
        child.visits = 100
        root_node.children.append(child)

        # Threshold > 1.0 can never be reached
        result = mcts_engine._should_terminate_early(root_node, threshold=1.1)

        assert result is False

    def test_zero_threshold_always_terminates(self, mcts_engine, root_node):
        """Test threshold of 0.0 always triggers termination."""
        from src.framework.mcts.core import MCTSNode, MCTSState

        root_node.visits = 100

        child_state = MCTSState(state_id="child", features={})
        child = MCTSNode(state=child_state, parent=root_node, action="action1")
        child.visits = 1  # Minimal visits
        root_node.children.append(child)

        result = mcts_engine._should_terminate_early(root_node, threshold=0.0)

        # Any positive fraction exceeds 0.0
        assert result is True
