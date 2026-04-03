"""
Unit tests for src/framework/assembly/graph.py (AssemblyGraph, AssemblyNode).
"""

import tempfile
from pathlib import Path

import pytest

from src.framework.assembly.graph import AssemblyGraph, AssemblyNode

# ---------------------------------------------------------------------------
# AssemblyNode tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAssemblyNode:
    """Tests for AssemblyNode dataclass."""

    def test_defaults(self):
        node = AssemblyNode(id="n1", label="Node 1")
        assert node.id == "n1"
        assert node.label == "Node 1"
        assert node.assembly_index == 0
        assert node.components == []
        assert node.metadata == {}

    def test_custom_fields(self):
        node = AssemblyNode(
            id="n2",
            label="Complex",
            assembly_index=5,
            components=["a", "b"],
            metadata={"key": "val"},
        )
        assert node.assembly_index == 5
        assert node.components == ["a", "b"]
        assert node.metadata["key"] == "val"

    def test_to_dict(self):
        node = AssemblyNode(id="n1", label="L", assembly_index=3, components=["x"])
        d = node.to_dict()
        assert d["id"] == "n1"
        assert d["label"] == "L"
        assert d["assembly_index"] == 3
        assert d["components"] == ["x"]
        assert d["metadata"] == {}

    def test_from_dict_full(self):
        data = {
            "id": "n1",
            "label": "L",
            "assembly_index": 2,
            "components": ["a"],
            "metadata": {"k": 1},
        }
        node = AssemblyNode.from_dict(data)
        assert node.id == "n1"
        assert node.assembly_index == 2
        assert node.metadata == {"k": 1}

    def test_from_dict_minimal(self):
        data = {"id": "n1", "label": "L"}
        node = AssemblyNode.from_dict(data)
        assert node.assembly_index == 0
        assert node.components == []
        assert node.metadata == {}

    def test_roundtrip(self):
        original = AssemblyNode(id="rt", label="Round", assembly_index=7, components=["c1", "c2"])
        restored = AssemblyNode.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.assembly_index == original.assembly_index
        assert restored.components == original.components


# ---------------------------------------------------------------------------
# AssemblyGraph tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAssemblyGraph:
    """Tests for AssemblyGraph class."""

    def _make_linear_graph(self, n: int = 4) -> AssemblyGraph:
        """Build a linear graph: 0 -> 1 -> ... -> n-1."""
        g = AssemblyGraph()
        for i in range(n):
            g.add_assembly_node(str(i), label=f"Node {i}", assembly_index=i)
        for i in range(n - 1):
            g.add_edge(str(i), str(i + 1))
        return g

    # -- add_assembly_node / get_node_data --

    def test_add_assembly_node(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A", assembly_index=2, components=["x"])
        assert "a" in g.nodes
        assert g.nodes["a"]["label"] == "A"
        assert g.nodes["a"]["assembly_index"] == 2
        assert g.nodes["a"]["components"] == ["x"]

    def test_add_assembly_node_with_metadata(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A", weight=0.5)
        assert g.nodes["a"]["weight"] == 0.5

    def test_get_node_data_exists(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A", assembly_index=3, components=["b"])
        node = g.get_node_data("a")
        assert node is not None
        assert isinstance(node, AssemblyNode)
        assert node.id == "a"
        assert node.assembly_index == 3
        assert node.components == ["b"]

    def test_get_node_data_not_found(self):
        g = AssemblyGraph()
        assert g.get_node_data("missing") is None

    def test_get_node_data_metadata_separation(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A", assembly_index=1, components=["c"], extra="val")
        node = g.get_node_data("a")
        assert node is not None
        assert "extra" in node.metadata
        assert "label" not in node.metadata
        assert "assembly_index" not in node.metadata

    # -- get_min_construction_pathway --

    def test_min_pathway_linear(self):
        g = self._make_linear_graph(4)
        path = g.get_min_construction_pathway()
        assert path == ["0", "1", "2", "3"]

    def test_min_pathway_explicit_source_target(self):
        g = self._make_linear_graph(5)
        path = g.get_min_construction_pathway(source="1", target="3")
        assert path == ["1", "2", "3"]

    def test_min_pathway_empty_graph(self):
        g = AssemblyGraph()
        assert g.get_min_construction_pathway() == []

    def test_min_pathway_single_node(self):
        g = AssemblyGraph()
        g.add_assembly_node("only", label="Only")
        path = g.get_min_construction_pathway()
        assert path == ["only"]

    def test_min_pathway_no_path_between(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A")
        g.add_assembly_node("b", label="B")
        # No edge between them, both are sources and sinks
        path = g.get_min_construction_pathway(source="a", target="b")
        assert path == ["a", "b"]

    def test_min_pathway_missing_nodes(self):
        g = self._make_linear_graph(3)
        assert g.get_min_construction_pathway(source="missing", target="0") == []

    # -- get_all_pathways --

    def test_all_pathways_linear(self):
        g = self._make_linear_graph(3)
        pathways = g.get_all_pathways()
        assert len(pathways) == 1
        assert pathways[0] == ["0", "1", "2"]

    def test_all_pathways_diamond(self):
        g = AssemblyGraph()
        for n in ["s", "a", "b", "t"]:
            g.add_assembly_node(n, label=n)
        g.add_edge("s", "a")
        g.add_edge("s", "b")
        g.add_edge("a", "t")
        g.add_edge("b", "t")
        pathways = g.get_all_pathways(source="s", target="t")
        assert len(pathways) == 2

    def test_all_pathways_max_limit(self):
        g = AssemblyGraph()
        for n in ["s", "a", "b", "c", "t"]:
            g.add_assembly_node(n, label=n)
        g.add_edge("s", "a")
        g.add_edge("s", "b")
        g.add_edge("s", "c")
        g.add_edge("a", "t")
        g.add_edge("b", "t")
        g.add_edge("c", "t")
        pathways = g.get_all_pathways(source="s", target="t", max_pathways=2)
        assert len(pathways) == 2

    def test_all_pathways_missing_node(self):
        g = self._make_linear_graph(3)
        assert g.get_all_pathways(source="missing", target="0") == []

    # -- calculate_pathway_similarity --

    def test_similarity_identical(self):
        g1 = self._make_linear_graph(3)
        g2 = self._make_linear_graph(3)
        assert g1.calculate_pathway_similarity(g2) == 1.0

    def test_similarity_both_empty(self):
        g1 = AssemblyGraph()
        g2 = AssemblyGraph()
        assert g1.calculate_pathway_similarity(g2) == 1.0

    def test_similarity_one_empty(self):
        g1 = self._make_linear_graph(3)
        g2 = AssemblyGraph()
        assert g1.calculate_pathway_similarity(g2) == 0.0
        assert g2.calculate_pathway_similarity(g1) == 0.0

    def test_similarity_partial_overlap(self):
        g1 = AssemblyGraph()
        g1.add_assembly_node("a", label="A")
        g1.add_assembly_node("b", label="B")
        g1.add_edge("a", "b")

        g2 = AssemblyGraph()
        g2.add_assembly_node("b", label="B")
        g2.add_assembly_node("c", label="C")
        g2.add_edge("b", "c")

        sim = g1.calculate_pathway_similarity(g2)
        assert 0.0 < sim < 1.0

    # -- calculate_reuse_factor --

    def test_reuse_factor_empty(self):
        g = AssemblyGraph()
        assert g.calculate_reuse_factor() == 0.0

    def test_reuse_factor_single_node(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A", assembly_index=0)
        assert g.calculate_reuse_factor() == 1.0

    def test_reuse_factor_max_reuse(self):
        g = AssemblyGraph()
        for i in range(5):
            g.add_assembly_node(str(i), label=str(i), assembly_index=0)
        assert g.calculate_reuse_factor() == 1.0

    def test_reuse_factor_no_reuse(self):
        g = AssemblyGraph()
        for i in range(5):
            g.add_assembly_node(str(i), label=str(i), assembly_index=i)
        # avg_index = 2.0, max_possible = 4, reuse = 1 - 2/4 = 0.5
        assert g.calculate_reuse_factor() == 0.5

    # -- get_assembly_layers --

    def test_layers_dag(self):
        g = self._make_linear_graph(3)
        layers = g.get_assembly_layers()
        assert len(layers) == 3
        assert "0" in layers[0]
        assert "1" in layers[1]
        assert "2" in layers[2]

    def test_layers_with_cycle(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A", assembly_index=0)
        g.add_assembly_node("b", label="B", assembly_index=1)
        g.add_edge("a", "b")
        g.add_edge("b", "a")  # cycle
        layers = g.get_assembly_layers()
        assert len(layers) >= 1

    # -- serialization --

    def test_to_dict(self):
        g = self._make_linear_graph(2)
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "pathways" in d
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_from_dict(self):
        g = self._make_linear_graph(3)
        g._pathways = [["0", "1", "2"]]
        d = g.to_dict()
        g2 = AssemblyGraph.from_dict(d)
        assert g2.number_of_nodes() == 3
        assert g2.number_of_edges() == 2
        assert g2._pathways == [["0", "1", "2"]]

    def test_to_json_from_json(self):
        g = self._make_linear_graph(3)
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "graph.json")
            g.to_json(path)
            g2 = AssemblyGraph.from_json(path)
            assert g2.number_of_nodes() == 3
            assert g2.number_of_edges() == 2

    def test_to_json_creates_parents(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A")
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "sub" / "dir" / "graph.json")
            g.to_json(path)
            assert Path(path).exists()

    # -- compute_statistics --

    def test_statistics_empty(self):
        g = AssemblyGraph()
        stats = g.compute_statistics()
        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0
        assert stats["is_dag"] is True
        assert stats["is_connected"] is False

    def test_statistics_linear(self):
        g = self._make_linear_graph(4)
        stats = g.compute_statistics()
        assert stats["num_nodes"] == 4
        assert stats["num_edges"] == 3
        assert stats["is_dag"] is True
        assert stats["is_connected"] is True
        assert stats["avg_assembly_index"] == 1.5
        assert stats["max_assembly_index"] == 3
        assert stats["min_assembly_index"] == 0
        assert "reuse_factor" in stats
        assert "min_pathway_length" in stats
        assert stats["min_pathway_length"] == 4

    # -- _make_dag --

    def test_make_dag_removes_cycle(self):
        g = AssemblyGraph()
        g.add_assembly_node("a", label="A")
        g.add_assembly_node("b", label="B")
        g.add_assembly_node("c", label="C")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")  # cycle
        dag = g._make_dag()
        import networkx as nx
        assert nx.is_directed_acyclic_graph(dag)
        assert dag.number_of_nodes() == 3
