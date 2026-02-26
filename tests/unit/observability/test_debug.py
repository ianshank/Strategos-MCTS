"""
Comprehensive tests for observability debug utilities.

Tests MCTSDebugger class, tree visualization, DOT export,
and verbose debugging controls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

pytest.importorskip("psutil", reason="psutil required for observability tests")

from src.observability.debug import (
    MCTSDebugger,
    disable_verbose_debugging,
    enable_verbose_debugging,
    export_tree_to_dot,
    log_agent_state_snapshot,
    print_debug_banner,
    visualize_mcts_tree,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockMCTSNode:
    """Mock MCTS node for testing."""

    state_id: str = "node_0"
    action: str = "root"
    visits: int = 0
    value: float = 0.0
    children: list[MockMCTSNode] = field(default_factory=list)
    _ucb_value: float = 0.0

    def ucb1(self) -> float:
        """Return mock UCB value."""
        if self.visits == 0:
            return float("inf")
        return self._ucb_value


@pytest.fixture
def mock_tree() -> MockMCTSNode:
    """Create a mock MCTS tree for testing."""
    root = MockMCTSNode(state_id="root", action="root", visits=100, value=50.0, _ucb_value=0.707)
    child1 = MockMCTSNode(state_id="c1", action="move_a", visits=60, value=36.0, _ucb_value=0.8)
    child2 = MockMCTSNode(state_id="c2", action="move_b", visits=40, value=14.0, _ucb_value=0.6)
    grandchild = MockMCTSNode(state_id="gc1", action="move_c", visits=20, value=12.0, _ucb_value=0.9)

    child1.children = [grandchild]
    root.children = [child1, child2]

    return root


@pytest.fixture
def debugger_enabled() -> MCTSDebugger:
    """Create an enabled MCTS debugger."""
    return MCTSDebugger(session_id="test_session", enabled=True)


@pytest.fixture
def debugger_disabled() -> MCTSDebugger:
    """Create a disabled MCTS debugger."""
    return MCTSDebugger(session_id="test_session", enabled=False)


# ============================================================================
# MCTSDebugger Tests
# ============================================================================


class TestMCTSDebuggerInit:
    """Tests for MCTSDebugger initialization."""

    def test_init_with_enabled_true(self) -> None:
        """Test initialization with enabled=True."""
        debugger = MCTSDebugger(session_id="test", enabled=True)
        assert debugger.session_id == "test"
        assert debugger.enabled is True
        assert debugger._iteration_count == 0

    def test_init_with_enabled_false(self) -> None:
        """Test initialization with enabled=False."""
        debugger = MCTSDebugger(session_id="test", enabled=False)
        assert debugger.enabled is False

    def test_init_auto_enable_debug_level(self) -> None:
        """Test auto-enable when LOG_LEVEL=DEBUG."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            debugger = MCTSDebugger(session_id="test")
            assert debugger.enabled is True

    def test_init_auto_disable_info_level(self) -> None:
        """Test auto-disable when LOG_LEVEL=INFO."""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO"}):
            debugger = MCTSDebugger(session_id="test")
            assert debugger.enabled is False

    def test_init_default_session_id(self) -> None:
        """Test default session ID."""
        debugger = MCTSDebugger(enabled=False)
        assert debugger.session_id == "default"


class TestMCTSDebuggerLogging:
    """Tests for MCTSDebugger logging methods."""

    def test_log_iteration_start_enabled(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging iteration start when enabled."""
        debugger_enabled.log_iteration_start(5)
        assert debugger_enabled._iteration_count == 5

    def test_log_iteration_start_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging iteration start when disabled does nothing."""
        debugger_disabled.log_iteration_start(5)
        assert debugger_disabled._iteration_count == 0

    def test_log_selection_enabled(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging selection when enabled."""
        debugger_enabled.log_selection(
            node_id="node_1",
            ucb_score=1.5,
            visits=10,
            value=5.0,
            depth=2,
            children_count=3,
            is_selected=True,
        )
        assert len(debugger_enabled._selection_history) == 1
        assert debugger_enabled._selection_history[0]["node_id"] == "node_1"
        assert debugger_enabled._selection_history[0]["is_selected"] is True

    def test_log_selection_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging selection when disabled does nothing."""
        debugger_disabled.log_selection(
            node_id="node_1",
            ucb_score=1.5,
            visits=10,
            value=5.0,
            depth=2,
            children_count=3,
        )
        assert len(debugger_disabled._selection_history) == 0

    def test_log_ucb_comparison_enabled(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging UCB comparison when enabled."""
        children_ucb = {"child_a": 1.2, "child_b": 0.8, "child_c": 1.5}
        debugger_enabled.log_ucb_comparison("parent", children_ucb, "child_c")

        assert len(debugger_enabled._ucb_history) == 1
        assert debugger_enabled._ucb_history[0] == children_ucb

    def test_log_ucb_comparison_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging UCB comparison when disabled does nothing."""
        debugger_disabled.log_ucb_comparison("parent", {"a": 1.0}, "a")
        assert len(debugger_disabled._ucb_history) == 0

    def test_log_expansion_enabled(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging expansion when enabled - should not raise."""
        debugger_enabled.log_expansion(
            parent_id="p1",
            action="move",
            new_node_id="n1",
            available_actions=["a", "b", "c"],
        )

    def test_log_expansion_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging expansion when disabled does nothing."""
        # Should not raise
        debugger_disabled.log_expansion(
            parent_id="p1",
            action="move",
            new_node_id="n1",
            available_actions=["a"],
        )

    def test_log_simulation_enabled(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging simulation when enabled."""
        debugger_enabled.log_simulation(
            node_id="n1",
            simulation_result=0.75,
            simulation_details={"steps": 10},
        )

    def test_log_simulation_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging simulation when disabled does nothing."""
        debugger_disabled.log_simulation(node_id="n1", simulation_result=0.5)

    def test_log_backpropagation_enabled(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging backpropagation when enabled."""
        debugger_enabled.log_backpropagation(
            path=["root", "child", "grandchild"],
            value=0.8,
            updates=[{"node": "child", "new_value": 1.2}],
        )

    def test_log_backpropagation_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging backpropagation when disabled does nothing."""
        debugger_disabled.log_backpropagation(path=["a"], value=0.5, updates=[])

    def test_log_iteration_end_enabled(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging iteration end when enabled."""
        debugger_enabled.log_iteration_end(
            iteration=5,
            best_action="move_a",
            best_ucb=1.5,
            tree_size=100,
        )

    def test_log_iteration_end_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging iteration end when disabled does nothing."""
        debugger_disabled.log_iteration_end(
            iteration=5,
            best_action="move_a",
            best_ucb=1.5,
            tree_size=100,
        )


class TestMCTSDebuggerStateDiff:
    """Tests for state diff functionality."""

    def test_log_state_diff_with_changes(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging state diff when there are changes."""
        old_state = {"a": 1, "b": 2}
        new_state = {"a": 1, "b": 3, "c": 4}

        debugger_enabled.log_state_diff(old_state, new_state, "Test change")

        assert len(debugger_enabled._state_history) == 1
        assert debugger_enabled._state_history[0]["description"] == "Test change"

    def test_log_state_diff_no_changes(self, debugger_enabled: MCTSDebugger) -> None:
        """Test logging state diff when there are no changes."""
        state = {"a": 1, "b": 2}
        debugger_enabled.log_state_diff(state, state.copy(), "No change")

        assert len(debugger_enabled._state_history) == 0

    def test_log_state_diff_disabled(self, debugger_disabled: MCTSDebugger) -> None:
        """Test logging state diff when disabled does nothing."""
        debugger_disabled.log_state_diff({"a": 1}, {"a": 2}, "Change")
        assert len(debugger_disabled._state_history) == 0

    def test_compute_state_diff_added_key(self, debugger_enabled: MCTSDebugger) -> None:
        """Test computing diff with added key."""
        diff = debugger_enabled._compute_state_diff({"a": 1}, {"a": 1, "b": 2})
        assert "b" in diff
        assert diff["b"] == {"added": 2}

    def test_compute_state_diff_removed_key(self, debugger_enabled: MCTSDebugger) -> None:
        """Test computing diff with removed key."""
        diff = debugger_enabled._compute_state_diff({"a": 1, "b": 2}, {"a": 1})
        assert "b" in diff
        assert diff["b"] == {"removed": 2}

    def test_compute_state_diff_changed_value(self, debugger_enabled: MCTSDebugger) -> None:
        """Test computing diff with changed value."""
        diff = debugger_enabled._compute_state_diff({"a": 1}, {"a": 2})
        assert "a" in diff
        assert diff["a"] == {"old": 1, "new": 2}

    def test_compute_state_diff_nested(self, debugger_enabled: MCTSDebugger) -> None:
        """Test computing diff with nested dictionaries."""
        old = {"outer": {"inner": 1}}
        new = {"outer": {"inner": 2}}
        diff = debugger_enabled._compute_state_diff(old, new)
        assert "outer.inner" in diff


class TestMCTSDebuggerSummary:
    """Tests for debug summary."""

    def test_get_debug_summary(self, debugger_enabled: MCTSDebugger) -> None:
        """Test getting debug summary."""
        debugger_enabled.log_iteration_start(5)
        debugger_enabled.log_selection("n1", 1.0, 10, 5.0, 1, 2, True)
        debugger_enabled.log_ucb_comparison("p", {"a": 1.0}, "a")

        summary = debugger_enabled.get_debug_summary()

        assert summary["session_id"] == "test_session"
        assert summary["total_iterations"] == 5
        assert summary["selection_history_count"] == 1
        assert summary["ucb_comparisons_count"] == 1


# ============================================================================
# Tree Visualization Tests
# ============================================================================


class TestVisualizeMCTSTree:
    """Tests for MCTS tree visualization."""

    def test_visualize_basic_tree(self, mock_tree: MockMCTSNode) -> None:
        """Test basic tree visualization."""
        result = visualize_mcts_tree(mock_tree)

        assert "MCTS Tree Visualization" in result
        assert "root" in result
        assert "move_a" in result
        assert "visits=100" in result

    def test_visualize_with_ucb(self, mock_tree: MockMCTSNode) -> None:
        """Test visualization with UCB scores."""
        result = visualize_mcts_tree(mock_tree, show_ucb=True)
        assert "UCB=" in result

    def test_visualize_without_ucb(self, mock_tree: MockMCTSNode) -> None:
        """Test visualization without UCB scores."""
        # Create node with zero visits (UCB = inf)
        simple_node = MockMCTSNode(visits=0)
        result = visualize_mcts_tree(simple_node, show_ucb=True)
        # Should not crash with inf UCB
        assert "MCTS Tree Visualization" in result

    def test_visualize_max_depth(self, mock_tree: MockMCTSNode) -> None:
        """Test visualization respects max_depth."""
        result = visualize_mcts_tree(mock_tree, max_depth=1)
        # Should stop at depth 1, grandchild (gc1) should not appear
        assert "gc1" not in result or "max depth reached" in result

    def test_visualize_max_children(self) -> None:
        """Test visualization respects max_children."""
        root = MockMCTSNode(state_id="root", visits=100)
        root.children = [MockMCTSNode(state_id=f"c{i}", visits=i) for i in range(10)]

        result = visualize_mcts_tree(root, max_children=3)
        assert "more children" in result

    def test_visualize_empty_tree(self) -> None:
        """Test visualization of single node tree."""
        node = MockMCTSNode()
        result = visualize_mcts_tree(node)
        assert "MCTS Tree Visualization" in result


# ============================================================================
# DOT Export Tests
# ============================================================================


class TestExportTreeToDot:
    """Tests for DOT format export."""

    def test_export_basic_tree(self, mock_tree: MockMCTSNode, tmp_path) -> None:
        """Test basic DOT export."""
        filename = str(tmp_path / "test_tree.dot")
        result = export_tree_to_dot(mock_tree, filename=filename)

        assert "digraph MCTSTree" in result
        assert "root" in result
        assert "->" in result  # Edges present

        # File should be created
        assert os.path.exists(filename)

    def test_export_without_file(self, mock_tree: MockMCTSNode) -> None:
        """Test DOT export without writing file."""
        result = export_tree_to_dot(mock_tree, filename="")
        assert "digraph MCTSTree" in result

    def test_export_with_ucb(self, mock_tree: MockMCTSNode) -> None:
        """Test DOT export includes UCB."""
        result = export_tree_to_dot(mock_tree, filename="", include_ucb=True)
        assert "UCB:" in result

    def test_export_without_ucb(self, mock_tree: MockMCTSNode) -> None:
        """Test DOT export without UCB."""
        result = export_tree_to_dot(mock_tree, filename="", include_ucb=False)
        # UCB should not be in labels
        lines = [l for l in result.split("\n") if "UCB:" in l]
        assert len(lines) == 0

    def test_export_max_depth(self, mock_tree: MockMCTSNode) -> None:
        """Test DOT export respects max_depth."""
        result = export_tree_to_dot(mock_tree, filename="", max_depth=0)
        # Only root should be exported
        node_lines = [l for l in result.split("\n") if "node_" in l and "[label=" in l]
        assert len(node_lines) == 1

    def test_export_color_coding(self, mock_tree: MockMCTSNode) -> None:
        """Test DOT export uses color coding."""
        result = export_tree_to_dot(mock_tree, filename="")
        assert "fillcolor=" in result


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestPrintDebugBanner:
    """Tests for debug banner printing."""

    def test_print_debug_banner_default(self) -> None:
        """Test printing debug banner with defaults."""
        # Should not raise
        print_debug_banner("Test Message")

    def test_print_debug_banner_custom(self) -> None:
        """Test printing debug banner with custom settings."""
        print_debug_banner("Custom", char="-", width=40)


class TestLogAgentStateSnapshot:
    """Tests for agent state snapshot logging."""

    def test_log_full_state(self) -> None:
        """Test logging full agent state."""
        state = {"key1": "value1", "key2": 42}
        # Should not raise
        log_agent_state_snapshot("TestAgent", state)

    def test_log_filtered_state(self) -> None:
        """Test logging filtered agent state."""
        state = {"key1": "value1", "key2": 42, "key3": "ignored"}
        log_agent_state_snapshot("TestAgent", state, include_keys=["key1", "key2"])

    def test_log_with_missing_keys(self) -> None:
        """Test logging with keys not in state."""
        state = {"key1": "value1"}
        log_agent_state_snapshot("TestAgent", state, include_keys=["key1", "nonexistent"])


class TestVerboseDebugging:
    """Tests for verbose debugging controls."""

    def test_enable_verbose_debugging(self) -> None:
        """Test enabling verbose debugging."""
        original_level = os.environ.get("LOG_LEVEL", "INFO")
        try:
            enable_verbose_debugging()
            assert os.environ.get("LOG_LEVEL") == "DEBUG"
        finally:
            os.environ["LOG_LEVEL"] = original_level

    def test_disable_verbose_debugging(self) -> None:
        """Test disabling verbose debugging."""
        os.environ["LOG_LEVEL"] = "DEBUG"
        try:
            disable_verbose_debugging()
            assert os.environ.get("LOG_LEVEL") == "INFO"
        finally:
            pass

    def test_enable_disable_roundtrip(self) -> None:
        """Test enabling then disabling verbose debugging."""
        original_level = os.environ.get("LOG_LEVEL", "INFO")
        try:
            enable_verbose_debugging()
            assert os.environ.get("LOG_LEVEL") == "DEBUG"

            disable_verbose_debugging()
            assert os.environ.get("LOG_LEVEL") == "INFO"
        finally:
            os.environ["LOG_LEVEL"] = original_level


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_node_without_ucb1_method(self) -> None:
        """Test visualization with node lacking ucb1 method."""

        class SimpleNode:
            state_id = "simple"
            action = "act"
            visits = 5
            value = 2.5
            children: list = []

        result = visualize_mcts_tree(SimpleNode())
        assert "simple" in result

    def test_node_with_ucb1_exception(self) -> None:
        """Test visualization handles ucb1 exceptions gracefully."""

        class BrokenNode:
            state_id = "broken"
            action = "act"
            visits = 5
            value = 2.5
            children: list = []

            def ucb1(self):
                raise ValueError("Broken UCB")

        result = visualize_mcts_tree(BrokenNode())
        assert "broken" in result

    def test_deeply_nested_state_diff(self, debugger_enabled: MCTSDebugger) -> None:
        """Test state diff with deeply nested structures."""
        old = {"l1": {"l2": {"l3": {"value": 1}}}}
        new = {"l1": {"l2": {"l3": {"value": 2}}}}

        diff = debugger_enabled._compute_state_diff(old, new)
        assert "l1.l2.l3.value" in diff
