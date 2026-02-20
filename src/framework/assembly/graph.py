"""
Assembly Graph Data Structure (Story 1.3).

Specialized graph structure for representing and manipulating assembly pathways.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass
class AssemblyNode:
    """
    Node in an assembly graph.

    Attributes:
        id: Unique node identifier
        label: Node label/description
        assembly_index: Assembly index at this node
        components: List of component node IDs that were assembled
        metadata: Additional node metadata
    """

    id: str
    label: str
    assembly_index: int = 0
    components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "assembly_index": self.assembly_index,
            "components": self.components,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssemblyNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            assembly_index=data.get("assembly_index", 0),
            components=data.get("components", []),
            metadata=data.get("metadata", {}),
        )


class AssemblyGraph(nx.DiGraph):
    """
    Specialized directed graph for assembly pathways.

    Extends NetworkX DiGraph with assembly-specific functionality:
    - Track assembly indices per node
    - Store construction pathways
    - Calculate reuse factors
    - Graph similarity comparison
    - Serialization
    """

    def __init__(self, *args, **kwargs):
        """Initialize assembly graph."""
        super().__init__(*args, **kwargs)
        self._pathways: list[list[str]] = []  # Construction pathways

    def add_assembly_node(
        self, node_id: str, label: str, assembly_index: int = 0, components: list[str] | None = None, **metadata
    ) -> None:
        """
        Add node with assembly metadata.

        Args:
            node_id: Unique node identifier
            label: Node label
            assembly_index: Assembly index at this node
            components: List of component node IDs
            **metadata: Additional metadata
        """
        AssemblyNode(
            id=node_id,
            label=label,
            assembly_index=assembly_index,
            components=components or [],
            metadata=metadata,
        )

        self.add_node(node_id, label=label, assembly_index=assembly_index, components=components or [], **metadata)

    def get_node_data(self, node_id: str) -> AssemblyNode | None:
        """
        Get node as AssemblyNode object.

        Args:
            node_id: Node identifier

        Returns:
            AssemblyNode or None if not found
        """
        if node_id not in self.nodes:
            return None

        data = self.nodes[node_id]
        return AssemblyNode(
            id=node_id,
            label=data.get("label", node_id),
            assembly_index=data.get("assembly_index", 0),
            components=data.get("components", []),
            metadata={k: v for k, v in data.items() if k not in ["label", "assembly_index", "components"]},
        )

    def get_min_construction_pathway(self, source: str | None = None, target: str | None = None) -> list[str]:
        """
        Return minimal assembly pathway from source to target.

        If source/target not specified, finds global minimum pathway.

        Args:
            source: Source node ID (or None for auto-detect)
            target: Target node ID (or None for auto-detect)

        Returns:
            List of node IDs representing minimal construction pathway
        """
        # Auto-detect source (nodes with no incoming edges)
        if source is None:
            sources = [n for n in self.nodes if self.in_degree(n) == 0]
            if not sources:
                sources = list(self.nodes)[:1]
            source = sources[0] if sources else None

        # Auto-detect target (nodes with no outgoing edges)
        if target is None:
            targets = [n for n in self.nodes if self.out_degree(n) == 0]
            if not targets:
                targets = list(self.nodes)[-1:]
            target = targets[0] if targets else None

        if source is None or target is None or source not in self.nodes or target not in self.nodes:
            return []

        # Find shortest path
        try:
            if nx.has_path(self, source, target):
                return list(nx.shortest_path(self, source, target))
            else:
                return [source, target] if source != target else [source]
        except nx.NetworkXNoPath:
            return [source]

    def get_all_pathways(
        self, source: str | None = None, target: str | None = None, max_pathways: int = 10
    ) -> list[list[str]]:
        """
        Get all construction pathways from source to target.

        Args:
            source: Source node ID
            target: Target node ID
            max_pathways: Maximum number of pathways to return

        Returns:
            List of pathways (each pathway is a list of node IDs)
        """
        # Auto-detect source/target
        if source is None:
            sources = [n for n in self.nodes if self.in_degree(n) == 0]
            source = sources[0] if sources else list(self.nodes)[0]

        if target is None:
            targets = [n for n in self.nodes if self.out_degree(n) == 0]
            target = targets[0] if targets else list(self.nodes)[-1]

        if source not in self.nodes or target not in self.nodes:
            return []

        # Find all simple paths
        try:
            pathways = []
            for path in nx.all_simple_paths(self, source, target):
                pathways.append(path)
                if len(pathways) >= max_pathways:
                    break
            return pathways
        except nx.NetworkXNoPath:
            return []

    def calculate_pathway_similarity(self, other: "AssemblyGraph") -> float:
        """
        Compute similarity between this and another assembly graph.

        Uses graph edit distance normalized by graph size.

        Args:
            other: Another assembly graph

        Returns:
            Similarity score (0.0 = completely different, 1.0 = identical)
        """
        if self.number_of_nodes() == 0 and other.number_of_nodes() == 0:
            return 1.0

        if self.number_of_nodes() == 0 or other.number_of_nodes() == 0:
            return 0.0

        # Simple similarity based on common nodes and edges
        # (Full graph edit distance is computationally expensive)

        # Node similarity
        self_nodes = set(self.nodes())
        other_nodes = set(other.nodes())
        common_nodes = self_nodes.intersection(other_nodes)
        total_nodes = self_nodes.union(other_nodes)

        node_similarity = len(common_nodes) / len(total_nodes) if total_nodes else 0.0

        # Edge similarity
        self_edges = set(self.edges())
        other_edges = set(other.edges())
        common_edges = self_edges.intersection(other_edges)
        total_edges = self_edges.union(other_edges)

        edge_similarity = len(common_edges) / len(total_edges) if total_edges else 0.0

        # Combined similarity (60% edges, 40% nodes)
        similarity = 0.4 * node_similarity + 0.6 * edge_similarity

        return similarity

    def calculate_reuse_factor(self) -> float:
        """
        Calculate reuse factor based on assembly indices.

        High reuse = low average assembly index
        Low reuse = high average assembly index

        Returns:
            Reuse factor (0.0 = no reuse, 1.0 = maximum reuse)
        """
        if self.number_of_nodes() == 0:
            return 0.0

        # Calculate average assembly index
        indices = [data.get("assembly_index", 0) for _, data in self.nodes(data=True)]
        avg_index = sum(indices) / len(indices) if indices else 0.0

        # Calculate max possible index (worst case: no reuse)
        max_possible = self.number_of_nodes() - 1

        # Reuse factor (inverse of normalized index)
        if max_possible > 0:
            reuse = 1.0 - (avg_index / max_possible)
        else:
            reuse = 1.0

        return float(max(0.0, min(1.0, reuse)))

    def get_assembly_layers(self) -> list[set[str]]:
        """
        Get assembly layers (topological sorting by assembly depth).

        Returns:
            List of layers, where each layer is a set of node IDs
        """
        if not nx.is_directed_acyclic_graph(self):
            # Make it a DAG first
            graph = self._make_dag()
        else:
            graph = self

        try:
            # Topological generations
            layers = list(nx.topological_generations(graph))
            return layers
        except nx.NetworkXError:
            # Fallback: group by assembly index
            index_layers = {}
            for node, data in self.nodes(data=True):
                idx = data.get("assembly_index", 0)
                if idx not in index_layers:
                    index_layers[idx] = set()
                index_layers[idx].add(node)

            return [index_layers[i] for i in sorted(index_layers.keys())]

    def _make_dag(self) -> "AssemblyGraph":
        """Create DAG version by removing cycles."""
        dag = AssemblyGraph()

        # Copy nodes
        for node, data in self.nodes(data=True):
            dag.add_node(node, **data)

        # Copy edges, removing cycles
        edges_to_add = list(self.edges(data=True))
        for u, v, data in edges_to_add:
            dag.add_edge(u, v, **data)

            # Check if adding edge created cycle
            try:
                nx.find_cycle(dag)
                # Remove this edge
                dag.remove_edge(u, v)
            except nx.NetworkXNoCycle:
                pass

        return dag

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize graph to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "nodes": [{"id": node, **data} for node, data in self.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in self.edges(data=True)],
            "pathways": self._pathways,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssemblyGraph":
        """
        Deserialize graph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            AssemblyGraph instance
        """
        graph = cls()

        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            graph.add_node(node_id, **node_data)

        # Add edges
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            graph.add_edge(source, target, **edge_data)

        # Restore pathways
        graph._pathways = data.get("pathways", [])

        return graph

    def to_json(self, path: str) -> None:
        """
        Save graph to JSON file.

        Args:
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "AssemblyGraph":
        """
        Load graph from JSON file.

        Args:
            path: Input file path

        Returns:
            AssemblyGraph instance
        """
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def compute_statistics(self) -> dict[str, Any]:
        """
        Compute graph statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_nodes": self.number_of_nodes(),
            "num_edges": self.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self),
            "is_connected": nx.is_weakly_connected(self) if self.number_of_nodes() > 0 else False,
            "density": nx.density(self),
        }

        if self.number_of_nodes() > 0:
            # Assembly statistics
            indices = [data.get("assembly_index", 0) for _, data in self.nodes(data=True)]
            stats["avg_assembly_index"] = sum(indices) / len(indices)
            stats["max_assembly_index"] = max(indices) if indices else 0
            stats["min_assembly_index"] = min(indices) if indices else 0

            # Reuse factor
            stats["reuse_factor"] = self.calculate_reuse_factor()

            # Pathway statistics
            try:
                min_path = self.get_min_construction_pathway()
                stats["min_pathway_length"] = len(min_path)
            except Exception:
                stats["min_pathway_length"] = 0

        return stats
