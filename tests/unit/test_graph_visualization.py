"""
Unit tests for LangGraph visualization methods.

Tests graph structure extraction, Mermaid generation, and execution tracing.

Based on: NEXT_STEPS_PLAN.md Phase 3.2
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

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
    """Create a mock GraphBuilder."""
    builder = MagicMock()
    builder.use_symbolic_reasoning = False
    builder.adk_agents = {}
    builder.mcts_config = MagicMock()
    builder.mcts_config.seed = 42
    builder.mcts_engine = MagicMock()
    return builder


@pytest.fixture
def mock_integrated_framework(mock_graph_builder):
    """Create a mock IntegratedFramework for testing visualization."""
    import src.framework.graph as graph_module

    framework = MagicMock()
    framework.app = MagicMock()
    framework.logger = MagicMock(spec=logging.Logger)
    framework.graph_builder = mock_graph_builder

    # Bind actual visualization methods
    framework.get_graph = lambda: graph_module.IntegratedFramework.get_graph(framework)
    framework.get_graph_structure = lambda: graph_module.IntegratedFramework.get_graph_structure(framework)
    framework.get_graph_mermaid = lambda **kwargs: graph_module.IntegratedFramework.get_graph_mermaid(
        framework, **kwargs
    )
    framework.draw_mermaid = lambda **kwargs: graph_module.IntegratedFramework.draw_mermaid(framework, **kwargs)
    framework.get_execution_trace_mermaid = lambda *args, **kwargs: (
        graph_module.IntegratedFramework.get_execution_trace_mermaid(framework, *args, **kwargs)
    )

    return framework


# =============================================================================
# get_graph Tests
# =============================================================================


class TestGetGraph:
    """Tests for get_graph method."""

    def test_get_graph_returns_app(self, mock_integrated_framework):
        """Test get_graph returns the LangGraph app."""
        result = mock_integrated_framework.get_graph()

        assert result is mock_integrated_framework.app


# =============================================================================
# get_graph_structure Tests
# =============================================================================


class TestGetGraphStructure:
    """Tests for get_graph_structure method."""

    def test_returns_dict_with_required_keys(self, mock_integrated_framework):
        """Test structure contains required keys."""
        structure = mock_integrated_framework.get_graph_structure()

        assert "nodes" in structure
        assert "edges" in structure
        assert "conditional_edges" in structure
        assert "entry_point" in structure
        assert "terminal_node" in structure

    def test_nodes_have_required_fields(self, mock_integrated_framework):
        """Test each node has required fields."""
        structure = mock_integrated_framework.get_graph_structure()

        for node in structure["nodes"]:
            assert "id" in node
            assert "label" in node
            assert "type" in node

    def test_edges_have_source_and_target(self, mock_integrated_framework):
        """Test each edge has source and target."""
        structure = mock_integrated_framework.get_graph_structure()

        for edge in structure["edges"]:
            assert "source" in edge
            assert "target" in edge

    def test_includes_core_nodes(self, mock_integrated_framework):
        """Test structure includes core graph nodes."""
        structure = mock_integrated_framework.get_graph_structure()

        node_ids = [n["id"] for n in structure["nodes"]]

        assert "entry" in node_ids
        assert "retrieve_context" in node_ids
        assert "route_decision" in node_ids
        assert "hrm_agent" in node_ids
        assert "trm_agent" in node_ids
        assert "synthesize" in node_ids

    def test_includes_conditional_routing(self, mock_integrated_framework):
        """Test structure includes conditional edges."""
        structure = mock_integrated_framework.get_graph_structure()

        assert "route_decision" in structure["conditional_edges"]
        assert "evaluate_consensus" in structure["conditional_edges"]

    def test_includes_symbolic_agent_when_enabled(self, mock_integrated_framework):
        """Test symbolic agent is included when enabled."""
        mock_integrated_framework.graph_builder.use_symbolic_reasoning = True

        structure = mock_integrated_framework.get_graph_structure()

        node_ids = [n["id"] for n in structure["nodes"]]
        assert "symbolic_agent" in node_ids

        # Also check edge exists
        edge_sources = [e["source"] for e in structure["edges"]]
        assert "symbolic_agent" in edge_sources

    def test_includes_adk_agents(self, mock_integrated_framework):
        """Test ADK agents are included in structure."""
        mock_integrated_framework.graph_builder.adk_agents = {
            "research": MagicMock(),
            "code": MagicMock(),
        }

        structure = mock_integrated_framework.get_graph_structure()

        node_ids = [n["id"] for n in structure["nodes"]]
        assert "adk_research" in node_ids
        assert "adk_code" in node_ids


# =============================================================================
# get_graph_mermaid Tests
# =============================================================================


class TestGetGraphMermaid:
    """Tests for get_graph_mermaid method."""

    def test_returns_mermaid_string(self, mock_integrated_framework):
        """Test returns valid Mermaid diagram string."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        assert isinstance(mermaid, str)
        assert "flowchart TD" in mermaid

    def test_includes_node_definitions(self, mock_integrated_framework):
        """Test Mermaid includes node definitions."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        assert "entry" in mermaid
        assert "retrieve_context" in mermaid
        assert "synthesize" in mermaid

    def test_includes_edge_definitions(self, mock_integrated_framework):
        """Test Mermaid includes edge definitions."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Check for edge syntax (-->)
        assert "-->" in mermaid

    def test_includes_conditional_edges(self, mock_integrated_framework):
        """Test Mermaid includes conditional edge labels."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Check for labeled edges (-->|label|)
        assert "-->|" in mermaid

    def test_includes_style_definitions(self, mock_integrated_framework):
        """Test Mermaid includes style definitions."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        assert "classDef" in mermaid
        assert "startNode" in mermaid
        assert "endNode" in mermaid

    def test_include_descriptions_option(self, mock_integrated_framework):
        """Test include_descriptions controls description display."""
        mermaid_with = mock_integrated_framework.get_graph_mermaid(include_descriptions=True)
        mermaid_without = mock_integrated_framework.get_graph_mermaid(include_descriptions=False)

        # With descriptions should have more content
        assert len(mermaid_with) >= len(mermaid_without)

    def test_theme_configuration(self, mock_integrated_framework):
        """Test theme option is included in output."""
        mermaid = mock_integrated_framework.get_graph_mermaid(theme="forest")

        assert "forest" in mermaid

    def test_default_theme_no_init(self, mock_integrated_framework):
        """Test default theme doesn't include init block."""
        mermaid = mock_integrated_framework.get_graph_mermaid(theme="default")

        # Default theme should not have init block
        assert "%%{init" not in mermaid


# =============================================================================
# draw_mermaid Tests
# =============================================================================


class TestDrawMermaid:
    """Tests for draw_mermaid method."""

    def test_returns_mermaid_without_output_file(self, mock_integrated_framework):
        """Test returns Mermaid code when no output file specified."""
        result = mock_integrated_framework.draw_mermaid()

        assert isinstance(result, str)
        assert "flowchart TD" in result

    def test_attempts_render_when_output_file_specified(self, mock_integrated_framework):
        """Test attempts to render when output file is specified."""
        with patch("httpx.Client") as mock_client:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.content = b"PNG data here"
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            with patch("builtins.open", MagicMock()):
                result = mock_integrated_framework.draw_mermaid(output_file="/tmp/graph.png")

                assert "flowchart TD" in result

    def test_handles_missing_httpx(self, mock_integrated_framework):
        """Test handles missing httpx gracefully."""
        with patch.dict("sys.modules", {"httpx": None}):
            # Should not raise, just log warning
            result = mock_integrated_framework.draw_mermaid(output_file="/tmp/graph.png")

            # Should still return the mermaid code
            assert "flowchart" in result


# =============================================================================
# get_execution_trace_mermaid Tests
# =============================================================================


class TestGetExecutionTraceMermaid:
    """Tests for get_execution_trace_mermaid method."""

    def test_returns_sequence_diagram(self, mock_integrated_framework):
        """Test returns a sequence diagram."""
        execution_path = ["entry", "retrieve_context", "route_decision", "hrm_agent"]

        mermaid = mock_integrated_framework.get_execution_trace_mermaid(execution_path)

        assert "sequenceDiagram" in mermaid

    def test_includes_participants(self, mock_integrated_framework):
        """Test includes participant definitions."""
        execution_path = ["entry", "hrm_agent", "synthesize"]

        mermaid = mock_integrated_framework.get_execution_trace_mermaid(execution_path)

        assert "participant" in mermaid
        assert "entry" in mermaid
        assert "hrm_agent" in mermaid
        assert "synthesize" in mermaid

    def test_includes_autonumber(self, mock_integrated_framework):
        """Test includes autonumber directive."""
        execution_path = ["entry", "hrm_agent"]

        mermaid = mock_integrated_framework.get_execution_trace_mermaid(execution_path)

        assert "autonumber" in mermaid

    def test_includes_timing_labels(self, mock_integrated_framework):
        """Test includes timing information when provided."""
        execution_path = ["entry", "hrm_agent", "synthesize"]
        timings = {"entry": 15.5, "hrm_agent": 250.0}

        mermaid = mock_integrated_framework.get_execution_trace_mermaid(execution_path, timings=timings)

        # Timing is shown for the source node in each transition
        assert "15.5ms" in mermaid
        # hrm_agent timing would show in transition to synthesize
        # but synthesize is the end node, so it may not show

    def test_handles_duplicate_nodes(self, mock_integrated_framework):
        """Test handles nodes visited multiple times."""
        # Simulates iteration loop
        execution_path = ["entry", "route_decision", "hrm_agent", "route_decision", "synthesize"]

        mermaid = mock_integrated_framework.get_execution_trace_mermaid(execution_path)

        # Should only define each participant once
        assert mermaid.count("participant route_decision") == 1


# =============================================================================
# Node Type Visualization Tests
# =============================================================================


class TestNodeTypeVisualization:
    """Tests for node type specific visualizations."""

    def test_start_nodes_use_rounded_shape(self, mock_integrated_framework):
        """Test start nodes use rounded rectangle shape."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Entry node should use rounded shape ([...])
        assert "entry([" in mermaid or "entry((" in mermaid

    def test_branch_nodes_use_diamond_shape(self, mock_integrated_framework):
        """Test branch nodes use diamond/rhombus shape."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Route decision should use diamond shape {...}
        assert "route_decision{" in mermaid

    def test_agent_nodes_use_parallelogram_shape(self, mock_integrated_framework):
        """Test agent nodes use parallelogram shape."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Agent nodes should use parallelogram [/.../]
        assert "hrm_agent[/" in mermaid
        assert "trm_agent[/" in mermaid

    def test_process_nodes_use_rectangle_shape(self, mock_integrated_framework):
        """Test process nodes use standard rectangle shape."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Process nodes use [...]
        assert "retrieve_context[" in mermaid
        assert "aggregate_results[" in mermaid


# =============================================================================
# Edge Rendering Tests
# =============================================================================


class TestEdgeRendering:
    """Tests for edge rendering in Mermaid output."""

    def test_sequential_edges_rendered(self, mock_integrated_framework):
        """Test sequential edges are rendered."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Entry to retrieve_context
        assert "entry" in mermaid
        assert "retrieve_context" in mermaid
        assert "-->" in mermaid

    def test_conditional_edge_labels(self, mock_integrated_framework):
        """Test conditional edges have labels."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Should have labeled conditional edges
        assert "-->|parallel|" in mermaid
        assert "-->|hrm|" in mermaid
        assert "-->|trm|" in mermaid

    def test_consensus_conditional_edges(self, mock_integrated_framework):
        """Test consensus conditional edges are rendered."""
        mermaid = mock_integrated_framework.get_graph_mermaid()

        # Consensus can lead to synthesize or iterate
        assert "-->|synthesize|" in mermaid
        assert "-->|iterate|" in mermaid
