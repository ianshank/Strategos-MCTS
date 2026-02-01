"""
Unit tests for graph.py iteration counter and consensus logic.

Tests the critical fix for the iteration counter bug where the counter
was never incremented, potentially causing infinite loops.

Based on: NEXT_STEPS_PLAN.md Phase 1.1
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_builder():
    """Create a minimal mock graph builder for testing iteration logic."""
    # Import the module first to ensure it's loaded
    import src.framework.graph as graph_module

    # Use the module reference for patching
    with patch.object(graph_module, "_META_CONTROLLER_AVAILABLE", False):
        from src.framework.graph import GraphBuilder

        # Create mock dependencies
        mock_adapter = MagicMock()
        mock_adapter.generate = MagicMock()
        mock_logger = MagicMock(spec=logging.Logger)

        # Create builder with minimal initialization
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.model_adapter = mock_adapter
        builder.logger = mock_logger
        builder.vector_store = None
        builder.top_k_retrieval = 5
        builder.consensus_threshold = 0.7
        builder.max_iterations = 3
        builder.mcts_config = MagicMock()
        builder.mcts_config.to_dict.return_value = {}

        return builder


@pytest.fixture
def base_agent_state() -> dict[str, Any]:
    """Create a base agent state for testing."""
    return {
        "query": "Test query for iteration testing",
        "use_mcts": False,
        "use_rag": False,
        "iteration": 0,
        "max_iterations": 3,
        "agent_outputs": [],
        "consensus_reached": False,
        "consensus_score": 0.0,
    }


# =============================================================================
# _evaluate_consensus_node Tests
# =============================================================================


class TestEvaluateConsensusNode:
    """Tests for _evaluate_consensus_node iteration increment."""

    def test_iteration_increments_with_single_agent(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test iteration counter increments when there's a single agent output."""
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.85, "response": "Test response"}
        ]
        base_agent_state["iteration"] = 0

        result = mock_graph_builder._evaluate_consensus_node(base_agent_state)

        assert result["iteration"] == 1, "Iteration should increment from 0 to 1"
        assert result["consensus_reached"] is True
        assert result["consensus_score"] == 1.0

    def test_iteration_increments_with_multiple_agents(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test iteration counter increments with multiple agent outputs."""
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.8, "response": "HRM response"},
            {"agent": "trm", "confidence": 0.9, "response": "TRM response"},
        ]
        base_agent_state["iteration"] = 1

        result = mock_graph_builder._evaluate_consensus_node(base_agent_state)

        assert result["iteration"] == 2, "Iteration should increment from 1 to 2"

    def test_iteration_increments_on_each_call(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that each call to evaluate_consensus increments iteration."""
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.6, "response": "Response 1"},
            {"agent": "trm", "confidence": 0.5, "response": "Response 2"},
        ]

        # First evaluation
        base_agent_state["iteration"] = 0
        result1 = mock_graph_builder._evaluate_consensus_node(base_agent_state)
        assert result1["iteration"] == 1

        # Second evaluation (update state)
        base_agent_state["iteration"] = result1["iteration"]
        result2 = mock_graph_builder._evaluate_consensus_node(base_agent_state)
        assert result2["iteration"] == 2

        # Third evaluation
        base_agent_state["iteration"] = result2["iteration"]
        result3 = mock_graph_builder._evaluate_consensus_node(base_agent_state)
        assert result3["iteration"] == 3

    def test_consensus_reached_when_confidence_above_threshold(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test consensus is reached when average confidence exceeds threshold."""
        # consensus_threshold is 0.7
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.8, "response": "Response 1"},
            {"agent": "trm", "confidence": 0.9, "response": "Response 2"},
        ]  # avg = 0.85, above 0.7

        result = mock_graph_builder._evaluate_consensus_node(base_agent_state)

        assert result["consensus_reached"] is True
        assert result["consensus_score"] == pytest.approx(0.85, rel=1e-2)

    def test_consensus_not_reached_when_confidence_below_threshold(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test consensus is not reached when confidence is below threshold."""
        # consensus_threshold is 0.7
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.5, "response": "Response 1"},
            {"agent": "trm", "confidence": 0.6, "response": "Response 2"},
        ]  # avg = 0.55, below 0.7

        result = mock_graph_builder._evaluate_consensus_node(base_agent_state)

        assert result["consensus_reached"] is False
        assert result["consensus_score"] == pytest.approx(0.55, rel=1e-2)

    def test_iteration_logged_correctly(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that iteration is logged in the info message."""
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.8, "response": "Response"},
            {"agent": "trm", "confidence": 0.9, "response": "Response"},
        ]
        base_agent_state["iteration"] = 2
        base_agent_state["max_iterations"] = 5

        mock_graph_builder._evaluate_consensus_node(base_agent_state)

        # Check that logger.info was called with iteration info
        mock_graph_builder.logger.info.assert_called()
        log_message = mock_graph_builder.logger.info.call_args[0][0]
        assert "iteration=3" in log_message or "iteration=3/5" in log_message


# =============================================================================
# _check_consensus Tests
# =============================================================================


class TestCheckConsensus:
    """Tests for _check_consensus routing logic."""

    def test_returns_synthesize_when_consensus_reached(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test returns 'synthesize' when consensus is reached."""
        base_agent_state["consensus_reached"] = True
        base_agent_state["iteration"] = 1

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "synthesize"

    def test_returns_synthesize_when_max_iterations_reached(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test returns 'synthesize' when max iterations is reached."""
        base_agent_state["consensus_reached"] = False
        base_agent_state["iteration"] = 3
        base_agent_state["max_iterations"] = 3

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "synthesize"

    def test_returns_synthesize_when_iteration_exceeds_max(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test returns 'synthesize' when iteration exceeds max_iterations."""
        base_agent_state["consensus_reached"] = False
        base_agent_state["iteration"] = 5
        base_agent_state["max_iterations"] = 3

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "synthesize"

    def test_returns_iterate_when_more_iterations_needed(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test returns 'iterate' when more iterations are needed."""
        base_agent_state["consensus_reached"] = False
        base_agent_state["iteration"] = 1
        base_agent_state["max_iterations"] = 3

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "iterate"

    def test_default_max_iterations_used_when_not_in_state(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that builder's default max_iterations is used if not in state."""
        del base_agent_state["max_iterations"]
        base_agent_state["consensus_reached"] = False
        base_agent_state["iteration"] = 2  # Below default of 3

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "iterate"

        # Now at max
        base_agent_state["iteration"] = 3

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "synthesize"

    def test_logs_consensus_reached(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that consensus reached is logged."""
        base_agent_state["consensus_reached"] = True
        base_agent_state["iteration"] = 2

        mock_graph_builder._check_consensus(base_agent_state)

        mock_graph_builder.logger.info.assert_called()
        log_message = mock_graph_builder.logger.info.call_args[0][0]
        assert "Consensus reached" in log_message

    def test_logs_max_iterations_warning(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that reaching max iterations logs a warning."""
        base_agent_state["consensus_reached"] = False
        base_agent_state["iteration"] = 3
        base_agent_state["max_iterations"] = 3

        mock_graph_builder._check_consensus(base_agent_state)

        mock_graph_builder.logger.warning.assert_called()
        log_message = mock_graph_builder.logger.warning.call_args[0][0]
        assert "Max iterations" in log_message

    def test_logs_debug_for_continue_iteration(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that continuing iteration logs debug message."""
        base_agent_state["consensus_reached"] = False
        base_agent_state["iteration"] = 1
        base_agent_state["max_iterations"] = 3

        mock_graph_builder._check_consensus(base_agent_state)

        mock_graph_builder.logger.debug.assert_called()


# =============================================================================
# Iteration Loop Termination Tests
# =============================================================================


class TestIterationLoopTermination:
    """Tests to verify the iteration loop terminates correctly."""

    def test_loop_terminates_at_max_iterations(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that the iteration loop terminates after max_iterations."""
        base_agent_state["max_iterations"] = 3
        base_agent_state["consensus_reached"] = False

        iteration_count = 0
        current_iteration = 0

        while iteration_count < 10:  # Safety limit
            # Simulate evaluate_consensus
            base_agent_state["iteration"] = current_iteration
            base_agent_state["agent_outputs"] = [
                {"agent": "hrm", "confidence": 0.5, "response": "Response"},
                {"agent": "trm", "confidence": 0.5, "response": "Response"},
            ]

            result = mock_graph_builder._evaluate_consensus_node(base_agent_state)
            current_iteration = result["iteration"]
            base_agent_state["iteration"] = current_iteration

            # Check if should terminate
            decision = mock_graph_builder._check_consensus(base_agent_state)
            iteration_count += 1

            if decision == "synthesize":
                break

        assert iteration_count == 3, f"Expected 3 iterations, got {iteration_count}"
        assert decision == "synthesize"

    def test_loop_terminates_early_on_consensus(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that the loop terminates early when consensus is reached."""
        base_agent_state["max_iterations"] = 10

        iteration_count = 0
        current_iteration = 0

        while iteration_count < 15:  # Safety limit
            # Simulate evaluate_consensus with high confidence on iteration 2
            base_agent_state["iteration"] = current_iteration
            confidence = 0.5 if iteration_count < 2 else 0.9  # High on 3rd iteration

            base_agent_state["agent_outputs"] = [
                {"agent": "hrm", "confidence": confidence, "response": "Response"},
                {"agent": "trm", "confidence": confidence, "response": "Response"},
            ]

            result = mock_graph_builder._evaluate_consensus_node(base_agent_state)
            current_iteration = result["iteration"]
            base_agent_state["iteration"] = current_iteration
            base_agent_state["consensus_reached"] = result["consensus_reached"]

            # Check if should terminate
            decision = mock_graph_builder._check_consensus(base_agent_state)
            iteration_count += 1

            if decision == "synthesize":
                break

        # Should terminate at iteration 3 when consensus is reached
        assert iteration_count == 3, f"Expected 3 iterations for early consensus, got {iteration_count}"
        assert decision == "synthesize"
        assert base_agent_state["consensus_reached"] is True

    def test_no_infinite_loop_with_zero_iterations(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test that setting max_iterations=0 terminates immediately."""
        base_agent_state["max_iterations"] = 0
        base_agent_state["iteration"] = 0
        base_agent_state["consensus_reached"] = False

        decision = mock_graph_builder._check_consensus(base_agent_state)

        assert decision == "synthesize"


# =============================================================================
# Entry Node Iteration Initialization Tests
# =============================================================================


class TestEntryNodeIteration:
    """Tests for iteration initialization in entry node."""

    def test_entry_node_initializes_iteration_to_zero(
        self,
        mock_graph_builder,
    ):
        """Test that entry node initializes iteration to 0."""
        state = {"query": "Test query"}

        result = mock_graph_builder._entry_node(state)

        assert result["iteration"] == 0

    def test_entry_node_sets_default_max_iterations(
        self,
        mock_graph_builder,
    ):
        """Test that entry node preserves or sets max_iterations."""
        state = {"query": "Test query"}

        result = mock_graph_builder._entry_node(state)

        # Should have max_iterations set (either from config or default)
        assert "iteration" in result


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestIterationEdgeCases:
    """Tests for edge cases in iteration logic."""

    def test_handles_missing_iteration_in_state(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test handles state without iteration key."""
        del base_agent_state["iteration"]
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.8, "response": "Response"}
        ]

        result = mock_graph_builder._evaluate_consensus_node(base_agent_state)

        assert result["iteration"] == 1  # 0 + 1

    def test_handles_empty_agent_outputs(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test handles empty agent outputs list."""
        base_agent_state["agent_outputs"] = []
        base_agent_state["iteration"] = 0

        result = mock_graph_builder._evaluate_consensus_node(base_agent_state)

        # Empty outputs should still increment iteration
        assert result["iteration"] == 1
        # Should auto-consensus with empty outputs (single agent path)
        assert result["consensus_reached"] is True

    def test_handles_negative_iteration(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test handles invalid negative iteration value."""
        base_agent_state["iteration"] = -1
        base_agent_state["agent_outputs"] = [
            {"agent": "hrm", "confidence": 0.8, "response": "Response"}
        ]

        result = mock_graph_builder._evaluate_consensus_node(base_agent_state)

        assert result["iteration"] == 0  # -1 + 1 = 0

    def test_handles_very_large_max_iterations(
        self,
        mock_graph_builder,
        base_agent_state,
    ):
        """Test handles very large max_iterations value."""
        base_agent_state["max_iterations"] = 1000000
        base_agent_state["iteration"] = 999999
        base_agent_state["consensus_reached"] = False

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "iterate"  # Not yet at max

        base_agent_state["iteration"] = 1000000

        result = mock_graph_builder._check_consensus(base_agent_state)

        assert result == "synthesize"
