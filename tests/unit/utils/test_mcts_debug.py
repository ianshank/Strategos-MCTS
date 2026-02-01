"""
Unit tests for MCTS Debug utilities.

Tests the debugging and inspection tools for the MCTS framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from src.utils.mcts_debug import (
    MCTSDebugger,
    NodeStats,
    SearchStats,
    create_debugger,
)


class TestNodeStats:
    """Tests for NodeStats dataclass."""

    def test_node_stats_creation(self) -> None:
        """Test creating node stats."""
        stats = NodeStats(
            node_id="node_123",
            depth=2,
            visits=100,
            value=0.75,
            ucb_score=1.23,
            children_count=5,
            is_terminal=False,
            is_expanded=True,
            action="move_a",
        )

        assert stats.node_id == "node_123"
        assert stats.depth == 2
        assert stats.visits == 100
        assert stats.value == 0.75

    def test_node_stats_to_dict(self) -> None:
        """Test converting node stats to dictionary."""
        stats = NodeStats(
            node_id="test",
            depth=1,
            visits=10,
            value=0.5,
            ucb_score=1.0,
            children_count=3,
            is_terminal=True,
            is_expanded=True,
        )

        d = stats.to_dict()

        assert d["node_id"] == "test"
        assert d["visits"] == 10
        assert d["is_terminal"] is True


class TestSearchStats:
    """Tests for SearchStats dataclass."""

    def test_search_stats_creation(self) -> None:
        """Test creating search stats."""
        stats = SearchStats(
            total_iterations=1000,
            total_nodes=500,
            max_depth=10,
            average_branching_factor=3.5,
            selection_time_ms=100.0,
            expansion_time_ms=50.0,
            simulation_time_ms=200.0,
            backprop_time_ms=30.0,
            total_time_ms=400.0,
            best_path=["a", "b", "c"],
            nodes_by_depth={0: 1, 1: 3, 2: 9},
        )

        assert stats.total_iterations == 1000
        assert stats.max_depth == 10
        assert len(stats.best_path) == 3

    def test_search_stats_to_dict(self) -> None:
        """Test converting search stats to dictionary."""
        stats = SearchStats(
            total_iterations=100,
            total_nodes=50,
            max_depth=5,
            average_branching_factor=2.5,
            selection_time_ms=10.0,
            expansion_time_ms=5.0,
            simulation_time_ms=20.0,
            backprop_time_ms=3.0,
            total_time_ms=40.0,
        )

        d = stats.to_dict()

        assert d["total_iterations"] == 100
        assert "timing_ms" in d
        assert d["timing_ms"]["selection"] == 10.0
        assert d["average_branching_factor"] == 2.5


class TestMCTSDebugger:
    """Tests for MCTSDebugger class."""

    def test_debugger_creation(self) -> None:
        """Test creating a debugger."""
        debugger = MCTSDebugger()
        assert debugger._engine is None

    def test_debugger_attach(self) -> None:
        """Test attaching debugger to engine."""
        debugger = MCTSDebugger()
        mock_engine = MagicMock()

        debugger.attach(mock_engine)

        assert debugger._engine is mock_engine

    def test_search_lifecycle(self) -> None:
        """Test start/end search lifecycle."""
        debugger = MCTSDebugger()

        debugger.start_search()
        assert debugger._search_start_time is not None
        assert debugger._iteration_count == 0

        # Record some activity
        debugger.record_iteration("node_1")
        debugger.record_iteration("node_2")
        debugger.record_phase("selection", 5.0)
        debugger.record_phase("selection", 3.0)

        stats = debugger.end_search()

        assert stats.total_iterations == 2
        assert stats.selection_time_ms == 8.0
        assert stats.total_time_ms > 0

    def test_record_phase_timing(self) -> None:
        """Test recording phase timings."""
        debugger = MCTSDebugger()
        debugger.start_search()

        debugger.record_phase("selection", 10.0)
        debugger.record_phase("expansion", 5.0)
        debugger.record_phase("simulation", 15.0)
        debugger.record_phase("backpropagation", 3.0)

        stats = debugger.end_search()

        assert stats.selection_time_ms == 10.0
        assert stats.expansion_time_ms == 5.0
        assert stats.simulation_time_ms == 15.0
        assert stats.backprop_time_ms == 3.0

    def test_format_tree_no_engine(self) -> None:
        """Test formatting tree without attached engine."""
        debugger = MCTSDebugger()
        result = debugger.format_tree()
        assert result == "No tree available"

    def test_format_tree_no_root(self) -> None:
        """Test formatting tree when engine has no root."""
        debugger = MCTSDebugger()
        mock_engine = MagicMock()
        del mock_engine._root  # Ensure _root doesn't exist
        debugger.attach(mock_engine)

        result = debugger.format_tree()
        assert result == "No tree available"

    def test_format_tree_with_mock_tree(self) -> None:
        """Test formatting tree with mock tree structure."""
        # Create mock tree structure
        @dataclass
        class MockNode:
            visits: int
            value: float
            action: str
            children: list

        root = MockNode(visits=100, value=0.5, action="root", children=[])
        child1 = MockNode(visits=60, value=0.6, action="move_a", children=[])
        child2 = MockNode(visits=40, value=0.4, action="move_b", children=[])
        root.children = [child1, child2]

        mock_engine = MagicMock()
        mock_engine._root = root

        debugger = MCTSDebugger()
        debugger.attach(mock_engine)

        result = debugger.format_tree()

        assert "MCTS Tree" in result
        assert "root" in result
        assert "move_a" in result

    def test_to_json_no_engine(self) -> None:
        """Test JSON export without engine."""
        debugger = MCTSDebugger()
        result = debugger.to_json()
        assert "error" in result

    def test_to_json_with_tree(self) -> None:
        """Test JSON export with mock tree."""
        @dataclass
        class MockNode:
            visits: int
            value: float
            action: str
            is_terminal: bool
            children: list

        root = MockNode(visits=10, value=0.5, action="root", is_terminal=False, children=[])

        mock_engine = MagicMock()
        mock_engine._root = root

        debugger = MCTSDebugger()
        debugger.attach(mock_engine)

        result = debugger.to_json()

        import json

        data = json.loads(result)
        assert data["visits"] == 10
        assert data["value"] == 0.5

    def test_get_node_stats(self) -> None:
        """Test getting statistics for a node."""
        @dataclass
        class MockNode:
            depth: int = 2
            visits: int = 50
            value: float = 0.75
            ucb_score: float = 1.5
            is_terminal: bool = False
            is_expanded: bool = True
            action: str = "test_action"
            children: list = None

            def __post_init__(self):
                self.children = self.children or []

        node = MockNode()
        debugger = MCTSDebugger()

        stats = debugger.get_node_stats(node)

        assert stats.depth == 2
        assert stats.visits == 50
        assert stats.value == 0.75
        assert stats.action == "test_action"

    def test_log_search_summary(self) -> None:
        """Test logging search summary (should not raise)."""
        debugger = MCTSDebugger()
        stats = SearchStats(
            total_iterations=100,
            total_nodes=50,
            max_depth=5,
            average_branching_factor=2.0,
            selection_time_ms=10.0,
            expansion_time_ms=5.0,
            simulation_time_ms=20.0,
            backprop_time_ms=3.0,
            total_time_ms=40.0,
            best_path=["a", "b"],
        )

        # Should not raise
        debugger.log_search_summary(stats)


class TestCreateDebugger:
    """Tests for factory function."""

    def test_create_debugger_no_engine(self) -> None:
        """Test creating debugger without engine."""
        debugger = create_debugger()
        assert isinstance(debugger, MCTSDebugger)
        assert debugger._engine is None

    def test_create_debugger_with_engine(self) -> None:
        """Test creating debugger with engine."""
        mock_engine = MagicMock()
        debugger = create_debugger(mock_engine)

        assert isinstance(debugger, MCTSDebugger)
        assert debugger._engine is mock_engine
