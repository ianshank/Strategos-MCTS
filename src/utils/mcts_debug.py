"""
MCTS Debug Utilities.

Provides debugging and inspection tools for the MCTS framework including:
- Tree visualization
- Node statistics
- Search path analysis
- Performance profiling

Based on: MULTI_AGENT_MCTS_TEMPLATE.md debugging patterns
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.observability.logging import get_correlation_id, get_logger, sanitize_dict

if TYPE_CHECKING:
    from src.framework.mcts.core import MCTSEngine, MCTSNode

logger = get_logger(__name__)


@dataclass
class NodeStats:
    """Statistics for a single MCTS node."""

    node_id: str
    depth: int
    visits: int
    value: float
    ucb_score: float | None
    children_count: int
    is_terminal: bool
    is_expanded: bool
    action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "visits": self.visits,
            "value": self.value,
            "ucb_score": self.ucb_score,
            "children_count": self.children_count,
            "is_terminal": self.is_terminal,
            "is_expanded": self.is_expanded,
            "action": self.action,
        }


@dataclass
class SearchStats:
    """Statistics for an MCTS search."""

    total_iterations: int
    total_nodes: int
    max_depth: int
    average_branching_factor: float
    selection_time_ms: float
    expansion_time_ms: float
    simulation_time_ms: float
    backprop_time_ms: float
    total_time_ms: float
    best_path: list[str] = field(default_factory=list)
    nodes_by_depth: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_iterations": self.total_iterations,
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
            "average_branching_factor": round(self.average_branching_factor, 2),
            "timing_ms": {
                "selection": round(self.selection_time_ms, 2),
                "expansion": round(self.expansion_time_ms, 2),
                "simulation": round(self.simulation_time_ms, 2),
                "backprop": round(self.backprop_time_ms, 2),
                "total": round(self.total_time_ms, 2),
            },
            "best_path": self.best_path,
            "nodes_by_depth": self.nodes_by_depth,
        }


class MCTSDebugger:
    """
    Debugger for MCTS search operations.

    Usage:
        debugger = MCTSDebugger(mcts_engine)
        debugger.start_search()
        # ... run search ...
        stats = debugger.end_search()
        print(debugger.format_tree())
    """

    def __init__(self, engine: MCTSEngine | None = None):
        """
        Initialize debugger.

        Args:
            engine: MCTS engine to debug (can be set later)
        """
        self._engine = engine
        self._search_start_time: float | None = None
        self._phase_timings: dict[str, list[float]] = defaultdict(list)
        self._iteration_count = 0
        self._nodes_visited: set[str] = set()

    def attach(self, engine: MCTSEngine) -> None:
        """Attach debugger to an MCTS engine."""
        self._engine = engine
        logger.debug(f"MCTSDebugger attached to engine, correlation_id={get_correlation_id()}")

    def start_search(self) -> None:
        """Mark start of a search operation."""
        self._search_start_time = time.perf_counter()
        self._phase_timings.clear()
        self._iteration_count = 0
        self._nodes_visited.clear()
        logger.debug("MCTS search started")

    def record_phase(self, phase: str, duration_ms: float) -> None:
        """Record timing for a search phase."""
        self._phase_timings[phase].append(duration_ms)

    def record_iteration(self, node_id: str) -> None:
        """Record an iteration visit."""
        self._iteration_count += 1
        self._nodes_visited.add(node_id)

    def end_search(self) -> SearchStats:
        """
        End search and collect statistics.

        Returns:
            Collected search statistics
        """
        if self._search_start_time is None:
            total_time = 0.0
        else:
            total_time = (time.perf_counter() - self._search_start_time) * 1000

        # Aggregate phase timings
        selection_time = sum(self._phase_timings.get("selection", []))
        expansion_time = sum(self._phase_timings.get("expansion", []))
        simulation_time = sum(self._phase_timings.get("simulation", []))
        backprop_time = sum(self._phase_timings.get("backpropagation", []))

        # Calculate tree stats
        tree_stats = self._analyze_tree() if self._engine else {}

        stats = SearchStats(
            total_iterations=self._iteration_count,
            total_nodes=tree_stats.get("total_nodes", len(self._nodes_visited)),
            max_depth=tree_stats.get("max_depth", 0),
            average_branching_factor=tree_stats.get("avg_branching", 0.0),
            selection_time_ms=selection_time,
            expansion_time_ms=expansion_time,
            simulation_time_ms=simulation_time,
            backprop_time_ms=backprop_time,
            total_time_ms=total_time,
            best_path=tree_stats.get("best_path", []),
            nodes_by_depth=tree_stats.get("nodes_by_depth", {}),
        )

        logger.info(
            "MCTS search completed",
            extra={
                "correlation_id": get_correlation_id(),
                "mcts_stats": sanitize_dict(stats.to_dict()),
            },
        )

        return stats

    def _analyze_tree(self) -> dict[str, Any]:
        """Analyze the MCTS tree structure."""
        if not self._engine or not hasattr(self._engine, "_root"):
            return {}

        root = getattr(self._engine, "_root", None)
        if root is None:
            return {}

        total_nodes = 0
        max_depth = 0
        nodes_by_depth: dict[int, int] = defaultdict(int)
        branching_factors: list[int] = []

        def traverse(node: MCTSNode, depth: int) -> None:
            nonlocal total_nodes, max_depth
            total_nodes += 1
            max_depth = max(max_depth, depth)
            nodes_by_depth[depth] += 1

            children = getattr(node, "children", [])
            if children:
                branching_factors.append(len(children))
                for child in children:
                    traverse(child, depth + 1)

        traverse(root, 0)

        # Find best path
        best_path = self._find_best_path(root)

        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "avg_branching": sum(branching_factors) / len(branching_factors) if branching_factors else 0.0,
            "best_path": best_path,
            "nodes_by_depth": dict(nodes_by_depth),
        }

    def _find_best_path(self, root: MCTSNode) -> list[str]:
        """Find the path of best moves from root."""
        path = []
        current = root

        while True:
            children = getattr(current, "children", [])
            if not children:
                break

            # Select child with highest visit count
            best_child = max(children, key=lambda c: getattr(c, "visits", 0))
            action = getattr(best_child, "action", None)
            if action:
                path.append(str(action))
            current = best_child

        return path

    def get_node_stats(self, node: MCTSNode) -> NodeStats:
        """Get statistics for a specific node."""
        return NodeStats(
            node_id=str(id(node)),
            depth=getattr(node, "depth", 0),
            visits=getattr(node, "visits", 0),
            value=getattr(node, "value", 0.0),
            ucb_score=getattr(node, "ucb_score", None),
            children_count=len(getattr(node, "children", [])),
            is_terminal=getattr(node, "is_terminal", False),
            is_expanded=getattr(node, "is_expanded", False),
            action=str(getattr(node, "action", None)) if hasattr(node, "action") else None,
        )

    def format_tree(self, max_depth: int = 3, max_children: int = 5) -> str:
        """
        Format MCTS tree as ASCII art.

        Args:
            max_depth: Maximum depth to display
            max_children: Maximum children per node to display

        Returns:
            ASCII representation of tree
        """
        if not self._engine or not hasattr(self._engine, "_root"):
            return "No tree available"

        root = getattr(self._engine, "_root", None)
        if root is None:
            return "No root node"

        lines = ["MCTS Tree (visits/value):"]

        def format_node(node: MCTSNode, prefix: str, depth: int, is_last: bool) -> None:
            visits = getattr(node, "visits", 0)
            value = getattr(node, "value", 0.0)
            action = getattr(node, "action", "root")

            # Build node representation
            connector = "└── " if is_last else "├── "
            node_str = f"{action}: {visits}/{value:.3f}"
            lines.append(f"{prefix}{connector}{node_str}")

            # Don't recurse if at max depth
            if depth >= max_depth:
                return

            # Get children sorted by visits
            children = getattr(node, "children", [])
            children = sorted(children, key=lambda c: getattr(c, "visits", 0), reverse=True)

            # Limit children displayed
            if len(children) > max_children:
                displayed = children[:max_children]
                remaining = len(children) - max_children
            else:
                displayed = children
                remaining = 0

            # Recurse into children
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(displayed):
                is_child_last = i == len(displayed) - 1 and remaining == 0
                format_node(child, child_prefix, depth + 1, is_child_last)

            # Indicate more children
            if remaining > 0:
                lines.append(f"{child_prefix}└── ... ({remaining} more children)")

        format_node(root, "", 0, True)
        return "\n".join(lines)

    def to_json(self) -> str:
        """
        Export tree structure as JSON.

        Returns:
            JSON string of tree structure
        """
        if not self._engine or not hasattr(self._engine, "_root"):
            return json.dumps({"error": "No tree available"})

        root = getattr(self._engine, "_root", None)
        if root is None:
            return json.dumps({"error": "No root node"})

        def node_to_dict(node: MCTSNode) -> dict[str, Any]:
            children = getattr(node, "children", [])
            return {
                "visits": getattr(node, "visits", 0),
                "value": getattr(node, "value", 0.0),
                "action": str(getattr(node, "action", None)),
                "is_terminal": getattr(node, "is_terminal", False),
                "children": [node_to_dict(c) for c in children],
            }

        return json.dumps(node_to_dict(root), indent=2)

    def log_search_summary(self, stats: SearchStats) -> None:
        """Log a formatted search summary."""
        summary = [
            "=" * 50,
            "MCTS Search Summary",
            "=" * 50,
            f"Iterations: {stats.total_iterations}",
            f"Nodes: {stats.total_nodes}",
            f"Max Depth: {stats.max_depth}",
            f"Avg Branching: {stats.average_branching_factor:.2f}",
            "-" * 50,
            "Timing (ms):",
            f"  Selection: {stats.selection_time_ms:.2f}",
            f"  Expansion: {stats.expansion_time_ms:.2f}",
            f"  Simulation: {stats.simulation_time_ms:.2f}",
            f"  Backprop: {stats.backprop_time_ms:.2f}",
            f"  Total: {stats.total_time_ms:.2f}",
            "-" * 50,
            f"Best Path: {' -> '.join(stats.best_path) if stats.best_path else 'N/A'}",
            "=" * 50,
        ]

        logger.info("\n".join(summary))


def create_debugger(engine: MCTSEngine | None = None) -> MCTSDebugger:
    """
    Factory function to create an MCTS debugger.

    Args:
        engine: Optional MCTS engine to attach

    Returns:
        Configured debugger instance
    """
    return MCTSDebugger(engine)


__all__ = [
    "NodeStats",
    "SearchStats",
    "MCTSDebugger",
    "create_debugger",
]
