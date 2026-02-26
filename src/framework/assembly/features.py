"""
Assembly Feature Extraction (Story 2.1).

Extract assembly-based features for ML models and decision-making.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .calculator import AssemblyIndexCalculator
from .concept_extractor import ConceptExtractor
from .config import AssemblyConfig


@dataclass
class AssemblyFeatures:
    """
    Assembly-based features for a query or reasoning path.

    Attributes:
        assembly_index: Core assembly index (complexity measure)
        copy_number: Substructure reuse factor
        decomposability_score: How easily the query can be decomposed
        graph_depth: Depth of the concept dependency graph
        constraint_count: Number of constraints/dependencies
        concept_count: Number of extracted concepts
        technical_complexity: Technical term density
        normalized_assembly_index: Assembly index normalized to [0, 1]
    """

    assembly_index: float
    copy_number: float
    decomposability_score: float
    graph_depth: int
    constraint_count: int
    concept_count: int
    technical_complexity: float
    normalized_assembly_index: float

    def to_dict(self) -> dict[str, float]:
        """
        Convert to dictionary suitable for ML models.

        Returns:
            Dictionary of feature names to values
        """
        return asdict(self)

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array for ML models.

        Returns:
            1D numpy array of feature values
        """
        return np.array(
            [
                self.assembly_index,
                self.copy_number,
                self.decomposability_score,
                float(self.graph_depth),
                float(self.constraint_count),
                float(self.concept_count),
                self.technical_complexity,
                self.normalized_assembly_index,
            ],
            dtype=np.float32,
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get list of feature names."""
        return [
            "assembly_index",
            "copy_number",
            "decomposability_score",
            "graph_depth",
            "constraint_count",
            "concept_count",
            "technical_complexity",
            "normalized_assembly_index",
        ]


class AssemblyFeatureExtractor:
    """
    Extract assembly-based features from queries and reasoning paths.

    Integrates:
    - AssemblyIndexCalculator for complexity measurement
    - ConceptExtractor for concept/dependency analysis
    - Normalization and feature engineering
    """

    def __init__(
        self,
        config: AssemblyConfig | None = None,
        domain: str = "general",
    ):
        """
        Initialize feature extractor.

        Args:
            config: Assembly configuration
            domain: Domain for concept extraction (general, software, data_science)
        """
        self.config = config or AssemblyConfig()
        self.domain = domain

        # Initialize components
        self.calculator = AssemblyIndexCalculator(
            cache_enabled=self.config.cache_assembly_indices,
            max_cache_size=self.config.max_cache_size,
        )

        self.concept_extractor = ConceptExtractor(
            domain=domain,
            min_frequency=self.config.min_concept_frequency,
            max_concepts=self.config.max_concepts,
            use_technical_terms=self.config.use_technical_terms,
        )

        # Normalization constants
        self._max_assembly_index = 20.0  # Reasonable upper bound for normalization
        self._max_depth = 10.0

    def extract(self, query: str, **context) -> AssemblyFeatures:
        """
        Extract all assembly features from a query.

        Args:
            query: Input query text
            **context: Additional context (e.g., previous_queries, domain_hints)

        Returns:
            AssemblyFeatures object

        Example:
            >>> extractor = AssemblyFeatureExtractor()
            >>> features = extractor.extract("How to optimize database queries?")
            >>> features.assembly_index
            5.0
            >>> features.decomposability_score
            0.75
        """
        if not query or not query.strip():
            return self._empty_features()

        # Extract concepts
        concepts = self.concept_extractor.extract_concepts(query)

        # Build dependency graph
        dep_graph = self.concept_extractor.build_dependency_graph(concepts)

        # Calculate assembly index and copy number
        assembly_index, copy_number = self.calculator.calculate(query)

        # Calculate decomposability score
        decomposability = self._calculate_decomposability(concepts, dep_graph)

        # Graph depth
        graph_depth = self._calculate_graph_depth(dep_graph)

        # Constraint count (number of edges in dependency graph)
        constraint_count = dep_graph.number_of_edges()

        # Concept count
        concept_count = len(concepts)

        # Technical complexity (ratio of technical terms to total concepts)
        technical_complexity = self._calculate_technical_complexity(concepts)

        # Normalized assembly index
        normalized_assembly_index = min(assembly_index / self._max_assembly_index, 1.0)

        return AssemblyFeatures(
            assembly_index=float(assembly_index),
            copy_number=float(copy_number),
            decomposability_score=decomposability,
            graph_depth=graph_depth,
            constraint_count=constraint_count,
            concept_count=concept_count,
            technical_complexity=technical_complexity,
            normalized_assembly_index=normalized_assembly_index,
        )

    def extract_batch(self, queries: list[str], **context) -> list[AssemblyFeatures]:
        """
        Extract features for multiple queries.

        Args:
            queries: List of queries
            **context: Additional context

        Returns:
            List of AssemblyFeatures
        """
        return [self.extract(q, **context) for q in queries]

    def _calculate_decomposability(self, concepts, dep_graph) -> float:
        """
        Calculate how easily a query can be decomposed.

        High decomposability = modular structure with clear layers
        Low decomposability = highly interconnected, complex dependencies

        Args:
            concepts: List of Concept objects
            dep_graph: Dependency graph

        Returns:
            Decomposability score (0.0 = hard to decompose, 1.0 = easy)
        """
        if not concepts or dep_graph.number_of_nodes() == 0:
            return 0.0

        # Factors:
        # 1. Graph is a DAG (if not, hard to decompose)
        import networkx as nx

        is_dag = nx.is_directed_acyclic_graph(dep_graph)
        if not is_dag:
            return 0.2  # Low decomposability for cyclic graphs

        # 2. Number of layers (more layers = easier to decompose hierarchically)
        try:
            layers = list(nx.topological_generations(dep_graph))
            num_layers = len(layers)
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            num_layers = 1

        layer_score = min(num_layers / 5.0, 1.0)  # Normalize to max 5 layers

        # 3. Connectivity (lower avg degree = more modular)
        if dep_graph.number_of_nodes() > 0:
            avg_degree = sum(dict(dep_graph.degree()).values()) / dep_graph.number_of_nodes()
            connectivity_score = 1.0 - min(avg_degree / 4.0, 1.0)  # Invert: lower degree = higher score
        else:
            connectivity_score = 1.0

        # 4. Concept clarity (higher importance = clearer structure)
        avg_importance = sum(c.importance for c in concepts) / len(concepts) if concepts else 0.0
        clarity_score = avg_importance

        # Combined score
        decomposability = (
            0.3 * layer_score + 0.3 * connectivity_score + 0.2 * clarity_score + 0.2 * (1.0 if is_dag else 0.0)
        )

        return float(min(decomposability, 1.0))

    def _calculate_graph_depth(self, dep_graph) -> int:
        """
        Calculate depth of dependency graph.

        Args:
            dep_graph: Dependency graph

        Returns:
            Maximum depth (number of layers)
        """
        if dep_graph.number_of_nodes() == 0:
            return 0

        import networkx as nx

        try:
            if nx.is_directed_acyclic_graph(dep_graph):
                # DAG longest path length
                depth = nx.dag_longest_path_length(dep_graph)
            else:
                # For cyclic graphs, use topological sort approximation
                layers = list(nx.topological_generations(dep_graph))
                depth = len(layers)
        except (nx.NetworkXError, nx.NetworkXUnfeasible, nx.NetworkXException):
            # Fallback: estimate from node count
            depth = int(np.sqrt(dep_graph.number_of_nodes()))

        return int(depth)

    def _calculate_technical_complexity(self, concepts) -> float:
        """
        Calculate technical term density.

        Args:
            concepts: List of Concept objects

        Returns:
            Technical complexity score (0.0 = no technical terms, 1.0 = all technical)
        """
        if not concepts:
            return 0.0

        technical_count = sum(1 for c in concepts if c.type == "technical_term")
        return technical_count / len(concepts)

    def _empty_features(self) -> AssemblyFeatures:
        """Return empty/default features."""
        return AssemblyFeatures(
            assembly_index=0.0,
            copy_number=0.0,
            decomposability_score=0.0,
            graph_depth=0,
            constraint_count=0,
            concept_count=0,
            technical_complexity=0.0,
            normalized_assembly_index=0.0,
        )

    def get_feature_importance(self, features: AssemblyFeatures) -> dict[str, float]:
        """
        Analyze which features are most significant.

        Args:
            features: Extracted features

        Returns:
            Dictionary of feature importances
        """
        # Heuristic importances based on domain knowledge
        importances = {
            "assembly_index": 0.25,  # Core metric
            "decomposability_score": 0.20,  # Critical for routing
            "graph_depth": 0.15,
            "constraint_count": 0.10,
            "copy_number": 0.10,
            "technical_complexity": 0.10,
            "concept_count": 0.05,
            "normalized_assembly_index": 0.05,
        }

        # Adjust based on feature values
        if features.decomposability_score > 0.7:
            # High decomposability increases its importance
            importances["decomposability_score"] *= 1.2

        if features.assembly_index > 8:
            # High complexity increases importance of copy number
            importances["copy_number"] *= 1.3

        # Normalize
        total = sum(importances.values())
        return {k: v / total for k, v in importances.items()}

    def explain_features(self, features: AssemblyFeatures) -> str:
        """
        Generate human-readable explanation of features.

        Args:
            features: Extracted features

        Returns:
            Explanation string
        """
        explanations = []

        # Assembly index
        if features.assembly_index < 3:
            complexity = "low"
        elif features.assembly_index < 7:
            complexity = "medium"
        else:
            complexity = "high"

        explanations.append(f"Complexity: {complexity} (assembly index: {features.assembly_index:.1f})")

        # Decomposability
        if features.decomposability_score > 0.7:
            decomp = "highly decomposable (good for hierarchical reasoning)"
        elif features.decomposability_score > 0.4:
            decomp = "moderately decomposable"
        else:
            decomp = "difficult to decompose (may need iterative refinement)"

        explanations.append(f"Decomposability: {decomp}")

        # Reuse
        if features.copy_number > 3:
            reuse = "high pattern reuse (efficient)"
        elif features.copy_number > 1.5:
            reuse = "moderate reuse"
        else:
            reuse = "unique construction (low reuse)"

        explanations.append(f"Pattern reuse: {reuse}")

        # Technical complexity
        if features.technical_complexity > 0.5:
            tech = "highly technical"
        elif features.technical_complexity > 0.2:
            tech = "moderately technical"
        else:
            tech = "non-technical"

        explanations.append(f"Domain: {tech}")

        # Depth
        explanations.append(f"Dependency depth: {features.graph_depth} layers")

        return " | ".join(explanations)

    def clear_cache(self) -> None:
        """Clear feature extraction caches."""
        self.calculator.clear_cache()
