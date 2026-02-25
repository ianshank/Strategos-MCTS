"""
Concept Extraction and Dependency Graph Builder (Story 1.2).

Extracts key concepts from text and builds dependency graphs showing
prerequisite relationships between concepts.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class Concept:
    """
    Represents a concept extracted from text.

    Attributes:
        term: The concept term/phrase
        type: Concept type (noun, verb, technical_term, entity)
        frequency: Number of occurrences
        dependencies: List of prerequisite concept IDs
        importance: Importance score (0.0-1.0)
        context: Surrounding context where concept appears
    """

    term: str
    type: str = "noun"
    frequency: int = 1
    dependencies: list[str] = field(default_factory=list)
    importance: float = 0.0
    context: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.term)

    def __eq__(self, other):
        return isinstance(other, Concept) and self.term == other.term


class ConceptExtractor:
    """
    Extract concepts from text and build dependency graphs.

    Supports:
    - Noun and verb extraction
    - Named entity recognition (basic)
    - Technical term detection
    - Domain-specific concept libraries
    - Dependency graph construction
    """

    def __init__(
        self,
        domain: str = "general",
        min_frequency: int = 1,
        max_concepts: int = 100,
        use_technical_terms: bool = True,
    ):
        """
        Initialize concept extractor.

        Args:
            domain: Domain for specialized extraction (software, data_science, general)
            min_frequency: Minimum frequency for concept inclusion
            max_concepts: Maximum number of concepts to extract
            use_technical_terms: Enable technical term detection
        """
        self.domain = domain
        self.min_frequency = min_frequency
        self.max_concepts = max_concepts
        self.use_technical_terms = use_technical_terms

        # Load domain-specific concept libraries
        self._load_domain_library()

    def extract_concepts(self, text: str) -> list[Concept]:
        """
        Extract key concepts from text.

        Args:
            text: Input text

        Returns:
            List of extracted concepts, sorted by importance

        Example:
            >>> extractor = ConceptExtractor()
            >>> concepts = extractor.extract_concepts("How to optimize database queries?")
            >>> [c.term for c in concepts]
            ['optimize', 'database', 'queries']
        """
        if not text or not text.strip():
            return []

        # Step 1: Extract candidate concepts
        candidates = self._extract_candidates(text)

        # Step 2: Filter and score
        concepts = self._filter_and_score(candidates, text)

        # Step 3: Limit to max_concepts
        concepts = sorted(concepts, key=lambda c: c.importance, reverse=True)[: self.max_concepts]

        return concepts

    def build_dependency_graph(self, concepts: list[Concept]) -> nx.DiGraph:
        """
        Build concept dependency DAG.

        Edges represent "requires" relationships:
        - A -> B means "B requires A" or "B depends on A"

        Args:
            concepts: List of concepts

        Returns:
            Directed acyclic graph of concept dependencies

        Example:
            >>> concepts = extractor.extract_concepts("optimize database queries")
            >>> graph = extractor.build_dependency_graph(concepts)
            >>> list(graph.edges())
            [('database', 'queries'), ('optimize', 'queries')]
        """
        graph = nx.DiGraph()

        # Add nodes
        for concept in concepts:
            graph.add_node(concept.term, type=concept.type, frequency=concept.frequency, importance=concept.importance)

        # Add dependency edges based on:
        # 1. Explicit dependencies
        # 2. Term relationships (compound terms)
        # 3. Domain knowledge
        # 4. Sequential ordering

        for i, concept in enumerate(concepts):
            # Explicit dependencies
            for dep in concept.dependencies:
                if dep in [c.term for c in concepts]:
                    graph.add_edge(dep, concept.term, type="explicit")

            # Infer dependencies from domain knowledge
            prerequisite_concepts = self._infer_prerequisites(concept.term)
            for prereq in prerequisite_concepts:
                if (
                    prereq in [c.term for c in concepts]
                    and prereq != concept.term
                    and not graph.has_edge(prereq, concept.term)
                ):
                    graph.add_edge(prereq, concept.term, type="inferred")

            # Sequential dependencies (weaker)
            if i > 0:
                prev_concept = concepts[i - 1]
                if self._are_related(prev_concept.term, concept.term) and not graph.has_edge(
                    prev_concept.term, concept.term
                ):
                    graph.add_edge(prev_concept.term, concept.term, type="sequential")

        # Ensure DAG (remove cycles)
        if not nx.is_directed_acyclic_graph(graph):
            graph = self._make_dag(graph)

        return graph

    def visualize_graph(self, graph: nx.DiGraph, output_path: str | None = None) -> None:
        """
        Visualize concept dependency graph.

        Args:
            graph: Dependency graph
            output_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Layout
            pos = nx.spring_layout(graph, k=2, iterations=50)

            # Draw nodes
            node_colors = []
            for node in graph.nodes():
                node_type = graph.nodes[node].get("type", "noun")
                if node_type == "verb":
                    node_colors.append("lightblue")
                elif node_type == "technical_term":
                    node_colors.append("lightcoral")
                else:
                    node_colors.append("lightgreen")

            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1000, alpha=0.9)

            # Draw edges
            edge_colors = []
            for u, v in graph.edges():
                edge_type = graph[u][v].get("type", "inferred")
                if edge_type == "explicit":
                    edge_colors.append("red")
                elif edge_type == "sequential":
                    edge_colors.append("gray")
                else:
                    edge_colors.append("black")

            nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrows=True, arrowsize=20, alpha=0.6)

            # Draw labels
            nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")

            plt.title("Concept Dependency Graph", fontsize=14)
            plt.axis("off")
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available. Skipping visualization.")

    def _extract_candidates(self, text: str) -> dict[str, Concept]:
        """Extract candidate concepts from text."""
        candidates = {}

        # Extract nouns (basic pattern matching)
        nouns = self._extract_nouns(text)
        for noun in nouns:
            if noun not in candidates:
                candidates[noun] = Concept(term=noun, type="noun")
            candidates[noun].frequency += 1

        # Extract verbs
        verbs = self._extract_verbs(text)
        for verb in verbs:
            if verb not in candidates:
                candidates[verb] = Concept(term=verb, type="verb")
            candidates[verb].frequency += 1

        # Extract technical terms
        if self.use_technical_terms:
            tech_terms = self._extract_technical_terms(text)
            for term in tech_terms:
                if term not in candidates:
                    candidates[term] = Concept(term=term, type="technical_term")
                candidates[term].frequency += 1
                candidates[term].importance += 0.3  # Boost technical terms

        # Extract named entities (basic)
        entities = self._extract_entities(text)
        for entity in entities:
            if entity not in candidates:
                candidates[entity] = Concept(term=entity, type="entity")
            candidates[entity].frequency += 1

        return candidates

    def _extract_nouns(self, text: str) -> list[str]:
        """Extract nouns using simple patterns."""
        # Common noun patterns
        words = re.findall(r"\b[a-z]+\b", text.lower())

        # Filter using noun indicators (can be enhanced with POS tagging)
        nouns = []
        for word in words:
            if len(word) > 2 and word in self.domain_library.get("nouns", set()) or len(word) > 3:
                nouns.append(word)

        return nouns

    def _extract_verbs(self, text: str) -> list[str]:
        """Extract verbs using simple patterns."""
        # Common verb endings
        verb_patterns = [
            r"\b(\w+(?:ize|ise|ate|ify|en))\b",  # -ize, -ate, -ify endings
            r"\b(create|build|design|implement|optimize|analyze|solve|find|get|set|use|make)\b",
        ]

        verbs = set()
        for pattern in verb_patterns:
            matches = re.findall(pattern, text.lower())
            verbs.update(matches)

        return list(verbs)

    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract technical/domain-specific terms."""
        tech_terms = set()

        # Check domain library
        domain_terms = self.domain_library.get("technical_terms", set())
        words = re.findall(r"\b\w+\b", text.lower())

        for word in words:
            if word in domain_terms:
                tech_terms.add(word)

        # Multi-word technical terms (bigrams/trigrams)
        tokens = text.lower().split()
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if bigram in domain_terms:
                tech_terms.add(bigram)

        return list(tech_terms)

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities (basic)."""
        # Capitalized words (basic NER)
        entities = re.findall(r"\b[A-Z][a-z]+\b", text)
        # Filter out common words at sentence start
        common_starts = {"The", "A", "An", "This", "That", "These", "Those", "What", "How", "Why"}
        entities = [e for e in entities if e not in common_starts]
        return entities

    def _filter_and_score(self, candidates: dict[str, Concept], text: str) -> list[Concept]:
        """Filter and score concepts."""
        concepts = []

        for term, concept in candidates.items():
            # Filter by frequency
            if concept.frequency < self.min_frequency:
                continue

            # Calculate importance score
            importance = self._calculate_importance(concept, text)
            concept.importance = importance

            # Add context
            concept.context = self._extract_context(term, text)

            concepts.append(concept)

        return concepts

    def _calculate_importance(self, concept: Concept, text: str) -> float:
        """
        Calculate concept importance score.

        Factors:
        - Frequency (normalized)
        - Type (verbs and technical terms score higher)
        - Position (earlier concepts often more important)
        - Length (longer terms often more specific/important)
        """
        score = 0.0

        # Frequency (max 0.3)
        word_count = len(text.split())
        freq_score = min(concept.frequency / word_count * 10, 0.3)
        score += freq_score

        # Type bonus
        type_bonuses = {
            "technical_term": 0.3,
            "verb": 0.2,
            "entity": 0.2,
            "noun": 0.1,
        }
        score += type_bonuses.get(concept.type, 0.0)

        # Length bonus (longer terms are more specific)
        if len(concept.term) > 6:
            score += 0.1
        elif len(concept.term) > 10:
            score += 0.2

        # Domain relevance
        if concept.term in self.domain_library.get("important_terms", set()):
            score += 0.2

        return min(score, 1.0)

    def _extract_context(self, term: str, text: str, window: int = 5) -> list[str]:
        """Extract surrounding context for a term."""
        words = text.lower().split()
        contexts = []

        for i, word in enumerate(words):
            if term in word:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                context = " ".join(words[start:end])
                contexts.append(context)

        return contexts

    def _infer_prerequisites(self, term: str) -> list[str]:
        """Infer prerequisite concepts based on domain knowledge."""
        prerequisites = []

        # Domain-specific prerequisite rules
        prereq_rules = self.domain_library.get("prerequisites", {})

        for prereq in prereq_rules.get(term, []):
            prerequisites.append(prereq)

        return prerequisites

    def _are_related(self, term1: str, term2: str) -> bool:
        """Check if two terms are related."""
        # Simple heuristic: share common substring
        if len(term1) < 3 or len(term2) < 3:
            return False

        # Check common prefix
        for i in range(3, min(len(term1), len(term2)) + 1):
            if term1[:i] == term2[:i]:
                return True

        # Check if one contains the other
        return bool(term1 in term2 or term2 in term1)

    def _make_dag(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Convert graph to DAG by removing cycles."""
        dag = graph.copy()

        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = nx.find_cycle(dag)
                # Remove edge with lowest weight in cycle
                if cycle:
                    dag.remove_edge(cycle[0][0], cycle[0][1])
            except nx.NetworkXNoCycle:
                break

        return dag

    def _load_domain_library(self) -> None:
        """Load domain-specific concept library."""
        # Default libraries for different domains
        libraries = {
            "software": {
                "nouns": {
                    "api",
                    "database",
                    "server",
                    "client",
                    "query",
                    "function",
                    "class",
                    "method",
                    "algorithm",
                    "data",
                    "code",
                    "system",
                    "architecture",
                    "service",
                    "endpoint",
                    "request",
                    "response",
                    "cache",
                    "session",
                },
                "technical_terms": {
                    "api",
                    "database",
                    "microservices",
                    "rest",
                    "sql",
                    "nosql",
                    "cache",
                    "queue",
                    "asynchronous",
                    "synchronous",
                    "http",
                    "https",
                    "authentication",
                    "authorization",
                    "crud",
                    "orm",
                    "framework",
                },
                "prerequisites": {
                    "query": ["database"],
                    "endpoint": ["api"],
                    "microservices": ["service", "api"],
                    "orm": ["database", "sql"],
                    "cache": ["database"],
                },
                "important_terms": {"api", "database", "architecture", "system", "algorithm"},
            },
            "data_science": {
                "nouns": {
                    "data",
                    "model",
                    "feature",
                    "prediction",
                    "algorithm",
                    "training",
                    "dataset",
                    "sample",
                    "variable",
                    "parameter",
                    "metric",
                    "accuracy",
                    "precision",
                    "recall",
                    "regression",
                    "classification",
                },
                "technical_terms": {
                    "machine learning",
                    "deep learning",
                    "neural network",
                    "regression",
                    "classification",
                    "clustering",
                    "supervised",
                    "unsupervised",
                    "overfitting",
                    "cross-validation",
                    "hyperparameter",
                },
                "prerequisites": {
                    "model": ["data", "algorithm"],
                    "prediction": ["model"],
                    "training": ["data", "model"],
                    "cross-validation": ["training", "model"],
                },
                "important_terms": {"model", "data", "algorithm", "prediction", "training"},
            },
            "general": {
                "nouns": {
                    "problem",
                    "solution",
                    "method",
                    "approach",
                    "result",
                    "analysis",
                    "process",
                    "system",
                    "component",
                    "structure",
                    "pattern",
                },
                "technical_terms": set(),
                "prerequisites": {},
                "important_terms": {"problem", "solution", "method", "system"},
            },
        }

        self.domain_library: dict[str, Any] = libraries.get(self.domain, libraries["general"])
