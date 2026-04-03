"""Unit tests for src/utils/mcts_debug.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.utils.mcts_debug import (
    MCTSDebugger,
    NodeStats,
    SearchStats,
    create_debugger,
)


def _make_node(
    visits=0,
    value=0.0,
    action=None,
    children=None,
    depth=0,
    is_terminal=False,
    is_expanded=False,
    ucb_score=None,
):
    """Create a mock MCTS node using SimpleNamespace."""
    return SimpleNamespace(
        visits=visits,
        value=value,
        action=action,
        children=children or [],
        depth=depth,
        is_terminal=is_terminal,
        is_expanded=is_expanded,
        ucb_score=ucb_score,
    )


@pytest.mark.unit
class TestNodeStats:
    """Tests for NodeStats dataclass."""

    def test_creation(self):
        stats = NodeStats(
            node_id="123",
            depth=2,
            visits=10,
            value=0.75,
            ucb_score=1.5,
            children_count=3,
            is_terminal=False,
            is_expanded=True,
            action="move_1",
        )
        assert stats.node_id == "123"
        assert stats.depth == 2
        assert stats.visits == 10
        assert stats.value == 0.75
        assert stats.ucb_score == 1.5
        assert stats.children_count == 3
        assert stats.is_terminal is False
        assert stats.is_expanded is True
        assert stats.action == "move_1"

    def test_to_dict(self):
        stats = NodeStats(
            node_id="abc",
            depth=0,
            visits=5,
            value=0.5,
            ucb_score=None,
            children_count=2,
            is_terminal=False,
            is_expanded=True,
        )
        d = stats.to_dict()
        assert d["node_id"] == "abc"
        assert d["visits"] == 5
        assert d["ucb_score"] is None
        assert d["action"] is None

    def test_default_action_none(self):
        stats = NodeStats(
            node_id="x", depth=0, visits=0, value=0.0,
            ucb_score=None, children_count=0,
            is_terminal=False, is_expanded=False,
        )
        assert stats.action is None


@pytest.mark.unit
class TestSearchStats:
    """Tests for SearchStats dataclass."""

    def test_creation_with_defaults(self):
        stats = SearchStats(
            total_iterations=100,
            total_nodes=50,
            max_depth=5,
            average_branching_factor=3.5,
            selection_time_ms=10.0,
            expansion_time_ms=20.0,
            simulation_time_ms=30.0,
            backprop_time_ms=15.0,
            total_time_ms=75.0,
        )
        assert stats.best_path == []
        assert stats.nodes_by_depth == {}

    def test_to_dict(self):
        stats = SearchStats(
            total_iterations=50,
            total_nodes=25,
            max_depth=3,
            average_branching_factor=2.333,
            selection_time_ms=5.123,
            expansion_time_ms=10.456,
            simulation_time_ms=15.789,
            backprop_time_ms=8.012,
            total_time_ms=39.38,
            best_path=["a", "b", "c"],
            nodes_by_depth={0: 1, 1: 3, 2: 9},
        )
        d = stats.to_dict()
        assert d["total_iterations"] == 50
        assert d["total_nodes"] == 25
        assert d["max_depth"] == 3
        assert d["average_branching_factor"] == 2.33
        assert d["timing_ms"]["selection"] == 5.12
        assert d["timing_ms"]["total"] == 39.38
        assert d["best_path"] == ["a", "b", "c"]
        assert d["nodes_by_depth"] == {0: 1, 1: 3, 2: 9}


@pytest.mark.unit
class TestMCTSDebugger:
    """Tests for MCTSDebugger class."""

    def test_init_no_engine(self):
        debugger = MCTSDebugger()
        assert debugger._engine is None
        assert debugger._iteration_count == 0

    def test_init_with_engine(self):
        engine = MagicMock()
        debugger = MCTSDebugger(engine=engine)
        assert debugger._engine is engine

    def test_attach(self):
        debugger = MCTSDebugger()
        engine = MagicMock()
        debugger.attach(engine)
        assert debugger._engine is engine

    def test_start_search_resets_state(self):
        debugger = MCTSDebugger()
        debugger._iteration_count = 5
        debugger._nodes_visited.add("a")
        debugger._phase_timings["selection"].append(1.0)

        debugger.start_search()
        assert debugger._search_start_time is not None
        assert debugger._iteration_count == 0
        assert len(debugger._nodes_visited) == 0
        assert len(debugger._phase_timings) == 0

    def test_record_phase(self):
        debugger = MCTSDebugger()
        debugger.record_phase("selection", 5.0)
        debugger.record_phase("selection", 3.0)
        debugger.record_phase("expansion", 2.0)
        assert debugger._phase_timings["selection"] == [5.0, 3.0]
        assert debugger._phase_timings["expansion"] == [2.0]

    def test_record_iteration(self):
        debugger = MCTSDebugger()
        debugger.record_iteration("node1")
        debugger.record_iteration("node2")
        debugger.record_iteration("node1")  # duplicate
        assert debugger._iteration_count == 3
        assert debugger._nodes_visited == {"node1", "node2"}

    def test_end_search_without_start(self):
        """end_search without start_search should still return stats."""
        debugger = MCTSDebugger()
        stats = debugger.end_search()
        assert stats.total_time_ms == 0.0
        assert stats.total_iterations == 0

    def test_end_search_with_timing(self):
        debugger = MCTSDebugger()
        debugger.start_search()
        debugger.record_phase("selection", 10.0)
        debugger.record_phase("expansion", 5.0)
        debugger.record_phase("simulation", 8.0)
        debugger.record_phase("backpropagation", 3.0)
        debugger.record_iteration("n1")
        debugger.record_iteration("n2")

        stats = debugger.end_search()
        assert stats.total_iterations == 2
        assert stats.selection_time_ms == 10.0
        assert stats.expansion_time_ms == 5.0
        assert stats.simulation_time_ms == 8.0
        assert stats.backprop_time_ms == 3.0
        assert stats.total_time_ms > 0  # some real elapsed time
        assert stats.total_nodes == 2  # from _nodes_visited since no engine

    @patch("src.utils.mcts_debug.sanitize_dict", side_effect=lambda d: d)
    def test_end_search_with_engine_tree(self, _mock_sanitize):
        """end_search with engine analyzes tree."""
        child1 = _make_node(visits=10, value=0.8, action="a1")
        child2 = _make_node(visits=5, value=0.4, action="a2")
        root = _make_node(visits=15, value=0.6, action="root", children=[child1, child2])

        engine = MagicMock()
        engine._root = root

        debugger = MCTSDebugger(engine=engine)
        debugger.start_search()
        stats = debugger.end_search()

        assert stats.total_nodes == 3
        assert stats.max_depth == 1
        assert stats.average_branching_factor == 2.0
        assert stats.best_path == ["a1"]
        assert stats.nodes_by_depth == {0: 1, 1: 2}

    def test_analyze_tree_no_engine(self):
        debugger = MCTSDebugger()
        result = debugger._analyze_tree()
        assert result == {}

    def test_analyze_tree_no_root(self):
        engine = MagicMock(spec=[])  # no _root attribute
        debugger = MCTSDebugger(engine=engine)
        result = debugger._analyze_tree()
        assert result == {}

    def test_analyze_tree_root_none(self):
        engine = MagicMock()
        engine._root = None
        debugger = MCTSDebugger(engine=engine)
        result = debugger._analyze_tree()
        assert result == {}

    def test_get_node_stats(self):
        node = _make_node(
            visits=10, value=0.75, action="move",
            depth=2, is_terminal=True, is_expanded=True,
            ucb_score=1.5, children=[_make_node(), _make_node()],
        )
        debugger = MCTSDebugger()
        stats = debugger.get_node_stats(node)
        assert stats.visits == 10
        assert stats.value == 0.75
        assert stats.depth == 2
        assert stats.is_terminal is True
        assert stats.is_expanded is True
        assert stats.ucb_score == 1.5
        assert stats.children_count == 2
        assert stats.action == "move"

    def test_get_node_stats_no_attributes(self):
        """get_node_stats handles nodes without expected attributes."""
        node = SimpleNamespace()  # bare node
        debugger = MCTSDebugger()
        stats = debugger.get_node_stats(node)
        assert stats.visits == 0
        assert stats.value == 0.0
        assert stats.depth == 0
        assert stats.children_count == 0

    def test_format_tree_no_engine(self):
        debugger = MCTSDebugger()
        result = debugger.format_tree()
        assert result == "No tree available"

    def test_format_tree_no_root(self):
        engine = MagicMock()
        engine._root = None
        debugger = MCTSDebugger(engine=engine)
        result = debugger.format_tree()
        assert result == "No root node"

    def test_format_tree_basic(self):
        child = _make_node(visits=5, value=0.5, action="c1")
        root = _make_node(visits=10, value=0.7, action="root", children=[child])
        engine = MagicMock()
        engine._root = root
        debugger = MCTSDebugger(engine=engine)

        output = debugger.format_tree()
        assert "MCTS Tree" in output
        assert "root" in output
        assert "c1" in output

    def test_format_tree_max_depth(self):
        """Nodes beyond max_depth should not appear."""
        deep_child = _make_node(visits=1, value=0.1, action="deep")
        mid_child = _make_node(visits=3, value=0.3, action="mid", children=[deep_child])
        root = _make_node(visits=10, value=0.5, action="root", children=[mid_child])
        engine = MagicMock()
        engine._root = root
        debugger = MCTSDebugger(engine=engine)

        output = debugger.format_tree(max_depth=1)
        assert "root" in output
        assert "mid" in output
        assert "deep" not in output

    def test_format_tree_max_children(self):
        """Only max_children are shown, rest indicated by '... more'."""
        children = [_make_node(visits=i, value=0.1 * i, action=f"c{i}") for i in range(10)]
        root = _make_node(visits=100, value=0.9, action="root", children=children)
        engine = MagicMock()
        engine._root = root
        debugger = MCTSDebugger(engine=engine)

        output = debugger.format_tree(max_children=3)
        assert "more children" in output

    def test_to_json_no_engine(self):
        debugger = MCTSDebugger()
        result = json.loads(debugger.to_json())
        assert result == {"error": "No tree available"}

    def test_to_json_no_root(self):
        engine = MagicMock()
        engine._root = None
        debugger = MCTSDebugger(engine=engine)
        result = json.loads(debugger.to_json())
        assert result == {"error": "No root node"}

    def test_to_json_with_tree(self):
        child = _make_node(visits=5, value=0.5, action="c1")
        root = _make_node(visits=10, value=0.7, action="root", children=[child])
        engine = MagicMock()
        engine._root = root
        debugger = MCTSDebugger(engine=engine)

        result = json.loads(debugger.to_json())
        assert result["visits"] == 10
        assert result["action"] == "root"
        assert len(result["children"]) == 1
        assert result["children"][0]["visits"] == 5
        assert result["children"][0]["action"] == "c1"

    def test_log_search_summary(self):
        debugger = MCTSDebugger()
        stats = SearchStats(
            total_iterations=100,
            total_nodes=50,
            max_depth=5,
            average_branching_factor=3.0,
            selection_time_ms=10.0,
            expansion_time_ms=20.0,
            simulation_time_ms=30.0,
            backprop_time_ms=15.0,
            total_time_ms=75.0,
            best_path=["a", "b"],
        )
        # Should not raise
        debugger.log_search_summary(stats)

    def test_log_search_summary_no_path(self):
        debugger = MCTSDebugger()
        stats = SearchStats(
            total_iterations=0,
            total_nodes=0,
            max_depth=0,
            average_branching_factor=0.0,
            selection_time_ms=0.0,
            expansion_time_ms=0.0,
            simulation_time_ms=0.0,
            backprop_time_ms=0.0,
            total_time_ms=0.0,
            best_path=[],
        )
        # Should not raise, path shows "N/A"
        debugger.log_search_summary(stats)

    def test_find_best_path_empty_tree(self):
        root = _make_node(visits=0, action="root")
        debugger = MCTSDebugger()
        path = debugger._find_best_path(root)
        assert path == []

    def test_find_best_path_multi_level(self):
        leaf1 = _make_node(visits=2, action="leaf1")
        leaf2 = _make_node(visits=8, action="leaf2")
        mid = _make_node(visits=10, action="mid", children=[leaf1, leaf2])
        other = _make_node(visits=3, action="other")
        root = _make_node(visits=13, action="root", children=[mid, other])

        debugger = MCTSDebugger()
        path = debugger._find_best_path(root)
        assert path == ["mid", "leaf2"]


@pytest.mark.unit
class TestCreateDebugger:
    """Tests for the create_debugger factory function."""

    def test_create_without_engine(self):
        debugger = create_debugger()
        assert isinstance(debugger, MCTSDebugger)
        assert debugger._engine is None

    def test_create_with_engine(self):
        engine = MagicMock()
        debugger = create_debugger(engine)
        assert isinstance(debugger, MCTSDebugger)
        assert debugger._engine is engine
