"""
Core Assembly Index Calculator (Story 1.1).

Implements assembly index calculation for strings and graphs
based on Assembly Theory principles.
"""

import hashlib
from collections import defaultdict

import networkx as nx


class AssemblyIndexCalculator:
    """
    Calculate assembly index and copy number for queries and reasoning paths.

    The assembly index quantifies the minimum number of steps required to
    construct an object from elementary parts. Copy number tracks how many
    times substructures are reused.

    Based on Assembly Theory (Cronin, Walker, et al.):
    - Assembly Index (A): Minimum construction steps
    - Copy Number (N): Frequency of substructure reuse
    """

    def __init__(self, cache_enabled: bool = True, max_cache_size: int = 10000):
        """
        Initialize assembly index calculator.

        Args:
            cache_enabled: Enable caching of computed indices
            max_cache_size: Maximum number of cached results
        """
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self._cache: dict[str, tuple[int, float]] = {}

    def calculate(self, input_data: str | nx.DiGraph) -> tuple[int, float]:
        """
        Calculate assembly index and copy number.

        Args:
            input_data: Either a string (query) or a directed graph (reasoning trace)

        Returns:
            Tuple of (assembly_index: int, copy_number: float)

        Raises:
            ValueError: If input is empty or invalid
        """
        if isinstance(input_data, str):
            return self._calculate_string_assembly(input_data)
        elif isinstance(input_data, nx.DiGraph):
            return self._calculate_graph_assembly(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _calculate_string_assembly(self, text: str) -> tuple[int, float]:
        """
        Calculate assembly index for text strings.

        Uses hierarchical decomposition:
        1. Split into tokens
        2. Build dependency structure
        3. Calculate minimal assembly pathway
        4. Track substructure reuse

        Args:
            text: Input text

        Returns:
            (assembly_index, copy_number)
        """
        if not text or not text.strip():
            return (0, 0.0)

        # Check cache
        if self.cache_enabled:
            cache_key = self._hash_string(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Tokenize
        tokens = self._tokenize(text)
        if len(tokens) == 1:
            result = (1, 1.0)
        else:
            # Build token dependency graph
            graph = self._build_token_graph(tokens)

            # Calculate assembly index from graph
            assembly_index, copy_number = self._calculate_graph_assembly(graph)
            result = (assembly_index, copy_number)

        # Cache result
        if self.cache_enabled and len(self._cache) < self.max_cache_size:
            self._cache[cache_key] = result

        return result

    def _calculate_graph_assembly(self, graph: nx.DiGraph) -> tuple[int, float]:
        """
        Calculate assembly index for directed graphs.

        Algorithm:
        1. Find all source nodes (no incoming edges)
        2. For each sink node (no outgoing edges):
           - Find shortest path from any source
           - Calculate assembly steps
        3. Return maximum assembly index (worst case)
        4. Calculate copy number from subgraph frequencies

        Args:
            graph: Directed graph representing construction pathway

        Returns:
            (assembly_index, copy_number)
        """
        if graph.number_of_nodes() == 0:
            return (0, 0.0)

        if graph.number_of_nodes() == 1:
            return (1, 1.0)

        # Handle cyclic graphs by converting to DAG
        if not nx.is_directed_acyclic_graph(graph):
            graph = self._break_cycles(graph)

        # Find sources and sinks
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        if not sources:
            # Cyclic graph fallback
            sources = [list(graph.nodes())[0]]

        if not sinks:
            # No sinks means all nodes are intermediate
            sinks = [max(graph.nodes(), key=lambda n: graph.out_degree(n))]

        # Calculate assembly index as maximum path length
        max_assembly_index = 0

        for sink in sinks:
            for source in sources:
                try:
                    if nx.has_path(graph, source, sink):
                        path = nx.shortest_path(graph, source, sink)
                        # Assembly index is path length - 1 (number of assembly steps)
                        assembly_index = len(path) - 1
                        max_assembly_index = max(max_assembly_index, assembly_index)
                except nx.NetworkXNoPath:
                    continue

        # If no paths found, use graph diameter approximation
        if max_assembly_index == 0:
            try:
                max_assembly_index = nx.dag_longest_path_length(graph)
            except (nx.NetworkXError, nx.NetworkXNotImplemented):
                max_assembly_index = graph.number_of_nodes() - 1

        # Calculate copy number (substructure reuse)
        copy_number = self._calculate_copy_number(graph)

        return (max(1, max_assembly_index), copy_number)

    def _calculate_copy_number(self, graph: nx.DiGraph) -> float:
        """
        Calculate copy number based on subgraph reuse.

        Copy number measures how many times substructures appear:
        - High copy number = lots of reuse (efficient assembly)
        - Low copy number = unique components (complex assembly)

        Args:
            graph: Directed graph

        Returns:
            Copy number (frequency of most common substructure)
        """
        if graph.number_of_nodes() <= 2:
            return 1.0

        # Count node pattern frequencies
        node_patterns: dict[tuple[int, int], int] = defaultdict(int)

        for node in graph.nodes():
            # Pattern: (in_degree, out_degree)
            pattern = (graph.in_degree(node), graph.out_degree(node))
            node_patterns[pattern] += 1

        if not node_patterns:
            return 1.0

        # Copy number is the maximum frequency divided by total nodes
        max_frequency = max(node_patterns.values())
        total_nodes = graph.number_of_nodes()

        # Normalize to [1.0, total_nodes]
        copy_number = max_frequency / total_nodes * total_nodes

        return float(copy_number)

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into meaningful units.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization (can be enhanced with NLP)
        tokens = []

        # Split on whitespace and punctuation
        import re

        # Keep alphanumeric and underscores, split on others
        raw_tokens = re.findall(r"\b\w+\b", text.lower())

        # Filter out very short tokens
        tokens = [t for t in raw_tokens if len(t) > 1 or t.isalnum()]

        return tokens if tokens else [text]

    def _build_token_graph(self, tokens: list[str]) -> nx.DiGraph:
        """
        Build dependency graph from tokens.

        Edges represent sequential dependencies:
        - Each token depends on previous tokens
        - Special patterns create additional dependencies

        Args:
            tokens: List of tokens

        Returns:
            Directed graph
        """
        graph = nx.DiGraph()

        if not tokens:
            return graph

        # Add nodes
        for i, token in enumerate(tokens):
            graph.add_node(i, label=token)

        # Add sequential edges
        for i in range(len(tokens) - 1):
            graph.add_edge(i, i + 1, weight=1)

        # Add dependencies based on token relationships
        # (can be enhanced with NLP dependency parsing)
        for i in range(len(tokens)):
            for j in range(i + 2, min(i + 5, len(tokens))):  # Look ahead 2-4 tokens
                # Add edge if tokens are related (simple heuristic)
                if self._tokens_related(tokens[i], tokens[j]) and not graph.has_edge(i, j):
                    graph.add_edge(i, j, weight=0.5)

        return graph

    def _tokens_related(self, token1: str, token2: str) -> bool:
        """
        Check if two tokens are semantically related.

        Simple heuristic: share common prefix/suffix or same root.

        Args:
            token1: First token
            token2: Second token

        Returns:
            True if related
        """
        if len(token1) < 3 or len(token2) < 3:
            return False

        # Check for common prefix (length >= 3)
        for length in range(3, min(len(token1), len(token2)) + 1):
            if token1[:length] == token2[:length]:
                return True

        # Check for common suffix
        return any(token1[-length:] == token2[-length:] for length in range(3, min(len(token1), len(token2)) + 1))

    def _break_cycles(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Break cycles in graph to create DAG.

        Removes minimum number of edges to eliminate cycles.

        Args:
            graph: Possibly cyclic directed graph

        Returns:
            Acyclic directed graph
        """
        # Create copy
        dag = graph.copy()

        # Find and remove cycles
        try:
            cycles = list(nx.simple_cycles(dag))
            for cycle in cycles:
                # Remove edge with lowest weight in cycle
                min_edge = None
                min_weight = float("inf")

                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]

                    if dag.has_edge(u, v):
                        weight = dag[u][v].get("weight", 1.0)
                        if weight < min_weight:
                            min_weight = weight
                            min_edge = (u, v)

                if min_edge:
                    dag.remove_edge(*min_edge)

        except (nx.NetworkXNoCycle, nx.NetworkXError):
            pass

        return dag

    def _hash_string(self, text: str) -> str:
        """Generate cache key for string."""
        return hashlib.sha256(text.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_cache_size,
            "enabled": self.cache_enabled,
        }
