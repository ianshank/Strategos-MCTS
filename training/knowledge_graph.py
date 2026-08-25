"""
Knowledge Graph System for MCTS/AI Training

Builds and queries a structured knowledge graph of AI/ML concepts, algorithms,
and their relationships from research papers and code implementations.

Features:
- Multi-backend support (NetworkX, Neo4j)
- LLM-powered entity and relationship extraction
- Hybrid retrieval (vector search + graph traversal)
- Graph-based question answering
- Automated graph construction from papers/code
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
import pickle
from typing import Any

import networkx as nx
import yaml

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from neo4j import GraphDatabase

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


logger = logging.getLogger(__name__)


# ============================================================================
# 1. Schema Definitions
# ============================================================================


class RelationType(str, Enum):
    """Types of relationships between concepts."""

    IS_A = "is_a"  # AlphaZero IS_A MCTS variant
    USES = "uses"  # AlphaZero USES neural networks
    IMPROVES = "improves"  # PUCT IMPROVES UCB1
    EXTENDS = "extends"  # MuZero EXTENDS AlphaZero
    IMPLEMENTED_IN = "implemented_in"  # UCB1 IMPLEMENTED_IN paper X
    COMPARED_TO = "compared_to"  # Method A COMPARED_TO Method B
    REQUIRES = "requires"  # AlphaZero REQUIRES self-play
    PART_OF = "part_of"  # PUCT PART_OF AlphaZero
    RELATED_TO = "related_to"  # Generic relationship
    INFLUENCES = "influences"  # Concept A INFLUENCES Concept B
    PRECEDES = "precedes"  # Temporal/historical ordering


@dataclass
class ConceptNode:
    """Represents an AI/ML concept in the knowledge graph."""

    id: str  # Unique identifier (normalized name)
    name: str  # Display name
    type: str  # algorithm, technique, architecture, metric, etc.
    description: str  # Brief description
    aliases: list[str] = field(default_factory=list)  # Alternative names
    properties: dict[str, Any] = field(default_factory=dict)  # Additional metadata
    source_papers: list[str] = field(default_factory=list)  # arXiv IDs
    code_references: list[str] = field(default_factory=list)  # File paths
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0  # Confidence score (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": self.aliases,
            "properties": self.properties,
            "source_papers": self.source_papers,
            "code_references": self.code_references,
            "created_at": self.created_at,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConceptNode":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Relationship:
    """Represents a relationship between two concepts."""

    source: str  # Source concept ID
    target: str  # Target concept ID
    relation_type: RelationType  # Type of relationship
    properties: dict[str, Any] = field(default_factory=dict)  # Additional metadata
    evidence: list[str] = field(default_factory=list)  # Supporting evidence (paper IDs, etc.)
    confidence: float = 1.0  # Confidence score (0-1)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relationship":
        """Create from dictionary."""
        data["relation_type"] = RelationType(data["relation_type"])
        return cls(**data)


# ============================================================================
# 2. Knowledge Extractor
# ============================================================================


class KnowledgeExtractor:
    """Extract entities and relationships from papers and code using LLM."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize knowledge extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_model = config.get("llm_model", "gpt-4-turbo-preview")
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

        if HAS_OPENAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI client not available for knowledge extraction")

        logger.info(f"KnowledgeExtractor initialized with model: {self.llm_model}")

    async def extract_from_paper(
        self, paper_id: str, title: str, abstract: str, full_text: str | None = None
    ) -> tuple[list[ConceptNode], list[Relationship]]:
        """
        Extract concepts and relationships from a research paper.

        Args:
            paper_id: Paper identifier (e.g., arXiv ID)
            title: Paper title
            abstract: Paper abstract
            full_text: Full paper text (optional)

        Returns:
            Tuple of (concepts, relationships)
        """
        if not self.client:
            logger.warning("LLM client not available, returning empty extraction")
            return [], []

        # Use abstract and title for extraction (full text can be too long)
        text_to_analyze = f"Title: {title}\n\nAbstract: {abstract}"

        if full_text and len(full_text) < 4000:
            text_to_analyze += f"\n\nFull Text: {full_text[:4000]}"

        extraction_prompt = self._build_extraction_prompt(text_to_analyze)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in AI/ML research who extracts structured knowledge from papers.",
                    },
                    {"role": "user", "content": extraction_prompt},
                ],
                temperature=0.3,
            )

            result_text = response.choices[0].message.content
            parsed = self._parse_extraction_result(result_text, paper_id)

            logger.info(f"Extracted {len(parsed[0])} concepts and {len(parsed[1])} relationships from {paper_id}")
            return parsed

        except Exception as e:
            logger.error(f"Error extracting from paper {paper_id}: {e}")
            return [], []

    def _build_extraction_prompt(self, text: str) -> str:
        """Build prompt for LLM extraction."""
        return f"""Extract AI/ML concepts and their relationships from the following research paper.

Paper Text:
{text}

Please identify:
1. Key algorithms, techniques, or methods mentioned
2. Mathematical formulations or metrics
3. Architectural components
4. Relationships between concepts (is_a, uses, improves, extends, requires, etc.)

Return your answer as a JSON object with this structure:
{{
  "concepts": [
    {{
      "name": "AlphaZero",
      "type": "algorithm",
      "description": "Game-playing AI using MCTS with deep neural networks",
      "aliases": ["Alpha Zero"],
      "properties": {{"domain": "game playing", "year": 2017}}
    }}
  ],
  "relationships": [
    {{
      "source": "AlphaZero",
      "target": "MCTS",
      "relation": "uses",
      "confidence": 0.95,
      "description": "AlphaZero uses Monte Carlo Tree Search as its core algorithm"
    }}
  ]
}}

Focus on technical concepts, not general terms. Extract only high-confidence relationships.
"""

    def _parse_extraction_result(self, result_text: str, paper_id: str) -> tuple[list[ConceptNode], list[Relationship]]:
        """Parse LLM extraction result."""
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end]
            elif "```" in result_text:
                json_start = result_text.find("```") + 3
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end]

            data = json.loads(result_text.strip())

            concepts = []
            for c in data.get("concepts", []):
                concept = ConceptNode(
                    id=self._normalize_id(c["name"]),
                    name=c["name"],
                    type=c.get("type", "unknown"),
                    description=c.get("description", ""),
                    aliases=c.get("aliases", []),
                    properties=c.get("properties", {}),
                    source_papers=[paper_id],
                    confidence=c.get("confidence", 0.8),
                )
                concepts.append(concept)

            relationships = []
            for r in data.get("relationships", []):
                confidence = r.get("confidence", 0.8)
                if confidence >= self.confidence_threshold:
                    relation = Relationship(
                        source=self._normalize_id(r["source"]),
                        target=self._normalize_id(r["target"]),
                        relation_type=RelationType(r.get("relation", "related_to")),
                        properties={"description": r.get("description", "")},
                        evidence=[paper_id],
                        confidence=confidence,
                    )
                    relationships.append(relation)

            return concepts, relationships

        except Exception as e:
            logger.error(f"Error parsing extraction result: {e}")
            return [], []

    def extract_from_code(self, file_path: str, code: str) -> tuple[list[ConceptNode], list[Relationship]]:
        """
        Extract concepts and relationships from code implementation.

        Args:
            file_path: Path to code file
            code: Source code

        Returns:
            Tuple of (concepts, relationships)
        """
        # Simple pattern-based extraction for code
        concepts = []
        relationships = []

        # Look for common algorithm patterns
        algorithm_patterns = {
            "MCTS": ["monte_carlo_tree_search", "mcts", "MonteCarloTreeSearch"],
            "UCB1": ["ucb1", "upper_confidence_bound"],
            "PUCT": ["puct", "polynomial_upper_confidence_tree"],
            "AlphaZero": ["alphazero", "alpha_zero"],
            "Neural Network": ["neural_network", "nn.Module", "keras.Model"],
        }

        code_lower = code.lower()

        for algo_name, patterns in algorithm_patterns.items():
            for pattern in patterns:
                if pattern.lower() in code_lower:
                    concept = ConceptNode(
                        id=self._normalize_id(algo_name),
                        name=algo_name,
                        type="algorithm",
                        description=f"Implementation found in {file_path}",
                        code_references=[file_path],
                        confidence=0.9,
                    )
                    concepts.append(concept)

                    # Add IMPLEMENTED_IN relationship
                    relationships.append(
                        Relationship(
                            source=self._normalize_id(algo_name),
                            target=file_path,
                            relation_type=RelationType.IMPLEMENTED_IN,
                            evidence=[file_path],
                            confidence=0.9,
                        )
                    )
                    break

        return concepts, relationships

    def _normalize_id(self, name: str) -> str:
        """Normalize concept name to create consistent ID."""
        return name.lower().replace(" ", "_").replace("-", "_")


# ============================================================================
# 3. Graph Builder (NetworkX Backend)
# ============================================================================


class KnowledgeGraphBuilder:
    """Build and manage knowledge graph using NetworkX."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize graph builder.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backend = config.get("backend", "networkx")
        self.storage_path = Path(config.get("storage", "./cache/knowledge_graph"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize backend
        if self.backend == "networkx":
            self.graph = nx.MultiDiGraph()
        elif self.backend == "neo4j" and HAS_NEO4J:
            self._initialize_neo4j()
        else:
            self.graph = nx.MultiDiGraph()
            logger.warning(f"Backend '{self.backend}' not available, using NetworkX")

        self.concepts = {}  # id -> ConceptNode
        self.relationships = []  # List of Relationship objects

        logger.info(f"KnowledgeGraphBuilder initialized with backend: {self.backend}")

    def _initialize_neo4j(self):
        """Initialize Neo4j connection."""
        neo4j_config = self.config.get("neo4j", {})
        uri = neo4j_config.get("uri", "bolt://localhost:7687")
        user = neo4j_config.get("user", "neo4j")
        password = neo4j_config.get("password", os.environ.get("NEO4J_PASSWORD", ""))

        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Neo4j driver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            self.neo4j_driver = None
            self.graph = nx.MultiDiGraph()

    def add_concept(self, concept: ConceptNode) -> None:
        """
        Add or update a concept node in the graph.

        Args:
            concept: ConceptNode to add
        """
        if concept.id in self.concepts:
            # Merge with existing concept
            existing = self.concepts[concept.id]
            existing.aliases = list(set(existing.aliases + concept.aliases))
            existing.source_papers = list(set(existing.source_papers + concept.source_papers))
            existing.code_references = list(set(existing.code_references + concept.code_references))
            existing.properties.update(concept.properties)
            # Use higher confidence
            existing.confidence = max(existing.confidence, concept.confidence)
        else:
            self.concepts[concept.id] = concept

        # Add to graph
        if self.backend == "networkx":
            self.graph.add_node(
                concept.id,
                name=concept.name,
                type=concept.type,
                description=concept.description,
                aliases=concept.aliases,
                properties=concept.properties,
                confidence=concept.confidence,
            )
        elif self.backend == "neo4j" and hasattr(self, "neo4j_driver") and self.neo4j_driver:
            self._add_concept_neo4j(concept)

    def _add_concept_neo4j(self, concept: ConceptNode):
        """Add concept to Neo4j."""
        with self.neo4j_driver.session() as session:
            session.run(
                """
                MERGE (c:Concept {id: $id})
                SET c.name = $name,
                    c.type = $type,
                    c.description = $description,
                    c.confidence = $confidence
                """,
                id=concept.id,
                name=concept.name,
                type=concept.type,
                description=concept.description,
                confidence=concept.confidence,
            )

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> None:
        """
        Add a relationship between concepts.

        Args:
            source: Source concept ID
            target: Target concept ID
            relation_type: Type of relationship
            properties: Additional properties
            confidence: Confidence score
        """
        relationship = Relationship(
            source=source,
            target=target,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence,
        )
        self.relationships.append(relationship)

        # Add to graph
        if self.backend == "networkx":
            self.graph.add_edge(
                source,
                target,
                relation=relation_type.value,
                properties=properties or {},
                confidence=confidence,
            )
        elif self.backend == "neo4j" and hasattr(self, "neo4j_driver") and self.neo4j_driver:
            self._add_relationship_neo4j(relationship)

    def _add_relationship_neo4j(self, relationship: Relationship):
        """Add relationship to Neo4j."""
        with self.neo4j_driver.session() as session:
            session.run(
                f"""
                MATCH (a:Concept {{id: $source}})
                MATCH (b:Concept {{id: $target}})
                MERGE (a)-[r:{relationship.relation_type.value}]->(b)
                SET r.confidence = $confidence
                """,
                source=relationship.source,
                target=relationship.target,
                confidence=relationship.confidence,
            )

    def build_from_corpus(self, extractor: KnowledgeExtractor, papers: list[dict[str, Any]]) -> dict[str, int]:
        """
        Build knowledge graph from corpus of papers.

        Args:
            extractor: KnowledgeExtractor instance
            papers: List of paper dictionaries with 'id', 'title', 'abstract'

        Returns:
            Statistics dictionary
        """
        total_concepts = 0
        total_relationships = 0

        for paper in papers:
            paper_id = paper.get("id", "unknown")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            try:
                # Extract concepts and relationships
                import asyncio

                concepts, relationships = asyncio.run(extractor.extract_from_paper(paper_id, title, abstract))

                # Add to graph
                for concept in concepts:
                    self.add_concept(concept)
                    total_concepts += 1

                for relationship in relationships:
                    self.add_relationship(
                        relationship.source,
                        relationship.target,
                        relationship.relation_type,
                        relationship.properties,
                        relationship.confidence,
                    )
                    total_relationships += 1

                if len(papers) > 10 and (papers.index(paper) + 1) % 10 == 0:
                    logger.info(f"Processed {papers.index(paper) + 1}/{len(papers)} papers")

            except Exception as e:
                logger.error(f"Error processing paper {paper_id}: {e}")
                continue

        stats = {
            "total_concepts": len(self.concepts),
            "total_relationships": len(self.relationships),
            "papers_processed": len(papers),
        }

        logger.info(f"Built graph: {stats}")
        return stats

    def save(self, filepath: Path | None = None) -> None:
        """Save knowledge graph to disk."""
        if filepath is None:
            filepath = self.storage_path / "knowledge_graph.pkl"

        data = {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "graph": nx.node_link_data(self.graph) if self.backend == "networkx" else None,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        # Also save as JSON for readability
        json_path = filepath.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "concepts": data["concepts"],
                    "relationships": data["relationships"],
                },
                f,
                indent=2,
            )

        logger.info(f"Saved knowledge graph to {filepath}")

    def load(self, filepath: Path | None = None) -> None:
        """Load knowledge graph from disk."""
        if filepath is None:
            filepath = self.storage_path / "knowledge_graph.pkl"

        if not filepath.exists():
            logger.warning(f"Graph file not found: {filepath}")
            return

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.concepts = {k: ConceptNode.from_dict(v) for k, v in data["concepts"].items()}
        self.relationships = [Relationship.from_dict(r) for r in data["relationships"]]

        if self.backend == "networkx" and data["graph"]:
            self.graph = nx.node_link_graph(data["graph"], multigraph=True, directed=True)

        logger.info(
            f"Loaded knowledge graph from {filepath}: {len(self.concepts)} concepts, {len(self.relationships)} relationships"
        )


# ============================================================================
# 4. Graph Query Engine
# ============================================================================


class GraphQueryEngine:
    """Query the knowledge graph."""

    def __init__(self, builder: KnowledgeGraphBuilder):
        """
        Initialize query engine.

        Args:
            builder: KnowledgeGraphBuilder instance
        """
        self.builder = builder
        self.graph = builder.graph
        self.concepts = builder.concepts

    def find_concept(self, name: str) -> ConceptNode | None:
        """
        Find concept by name or alias.

        Args:
            name: Concept name or alias

        Returns:
            ConceptNode if found, None otherwise
        """
        normalized = name.lower().replace(" ", "_").replace("-", "_")

        # Direct lookup
        if normalized in self.concepts:
            return self.concepts[normalized]

        # Search by alias
        for concept in self.concepts.values():
            if name.lower() in [a.lower() for a in concept.aliases]:
                return concept
            if name.lower() in concept.name.lower():
                return concept

        return None

    def get_relationships(
        self, concept: str, relation_type: RelationType | None = None, direction: str = "outgoing"
    ) -> list[dict[str, Any]]:
        """
        Get all relationships for a concept.

        Args:
            concept: Concept ID or name
            relation_type: Filter by relationship type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationship dictionaries
        """
        concept_node = self.find_concept(concept)
        if not concept_node:
            return []

        concept_id = concept_node.id
        results = []

        if direction in ["outgoing", "both"] and concept_id in self.graph:
            for target in self.graph.successors(concept_id):
                for edge_data in self.graph[concept_id][target].values():
                    if relation_type is None or edge_data.get("relation") == relation_type.value:
                        results.append(
                            {
                                "source": concept_id,
                                "target": target,
                                "relation": edge_data.get("relation"),
                                "properties": edge_data.get("properties", {}),
                                "confidence": edge_data.get("confidence", 1.0),
                            }
                        )

        if direction in ["incoming", "both"] and concept_id in self.graph:
            for source in self.graph.predecessors(concept_id):
                for edge_data in self.graph[source][concept_id].values():
                    if relation_type is None or edge_data.get("relation") == relation_type.value:
                        results.append(
                            {
                                "source": source,
                                "target": concept_id,
                                "relation": edge_data.get("relation"),
                                "properties": edge_data.get("properties", {}),
                                "confidence": edge_data.get("confidence", 1.0),
                            }
                        )

        return results

    def find_path(self, source: str, target: str, max_depth: int = 5) -> list[list[str]] | None:
        """
        Find connection path between two concepts.

        Args:
            source: Source concept name
            target: Target concept name
            max_depth: Maximum path length

        Returns:
            List of paths (each path is a list of concept IDs)
        """
        source_node = self.find_concept(source)
        target_node = self.find_concept(target)

        if not source_node or not target_node:
            return None

        try:
            # Find all simple paths up to max_depth
            paths = list(
                nx.all_simple_paths(self.graph.to_undirected(), source_node.id, target_node.id, cutoff=max_depth)
            )
            return paths[:10]  # Return up to 10 paths
        except nx.NetworkXNoPath:
            return None

    def get_related_concepts(
        self, concept: str, depth: int = 2, relation_filter: list[RelationType] | None = None
    ) -> dict[str, Any]:
        """
        Get related concepts via BFS/DFS traversal.

        Args:
            concept: Starting concept name
            depth: Traversal depth
            relation_filter: Filter by relationship types

        Returns:
            Dictionary with related concepts organized by depth
        """
        concept_node = self.find_concept(concept)
        if not concept_node:
            return {}

        result = {
            "root": concept_node.to_dict(),
            "related": defaultdict(list),
        }

        visited = {concept_node.id}
        queue = deque([(concept_node.id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Get neighbors
            if current_id in self.graph:
                for neighbor in self.graph.neighbors(current_id):
                    if neighbor not in visited:
                        # Check relation filter
                        edge_data = list(self.graph[current_id][neighbor].values())[0]
                        relation = edge_data.get("relation")

                        if relation_filter is None or RelationType(relation) in relation_filter:
                            visited.add(neighbor)
                            queue.append((neighbor, current_depth + 1))

                            if neighbor in self.concepts:
                                result["related"][current_depth + 1].append(
                                    {
                                        "concept": self.concepts[neighbor].to_dict(),
                                        "relation": relation,
                                        "confidence": edge_data.get("confidence", 1.0),
                                    }
                                )

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        stats = {
            "total_concepts": len(self.concepts),
            "total_relationships": len(self.builder.relationships),
            "concept_types": defaultdict(int),
            "relation_types": defaultdict(int),
            "avg_degree": 0,
            "connected_components": 0,
        }

        # Count concept types
        for concept in self.concepts.values():
            stats["concept_types"][concept.type] += 1

        # Count relation types
        for rel in self.builder.relationships:
            stats["relation_types"][rel.relation_type.value] += 1

        # Graph metrics
        if len(self.graph) > 0:
            stats["avg_degree"] = sum(dict(self.graph.degree()).values()) / len(self.graph)
            stats["connected_components"] = nx.number_weakly_connected_components(self.graph)

        return dict(stats)


# ============================================================================
# 5. Hybrid Retrieval (Vector + Graph)
# ============================================================================


class HybridKnowledgeRetriever:
    """Combine vector search with graph traversal for enhanced retrieval."""

    def __init__(
        self,
        query_engine: GraphQueryEngine,
        vector_index: Any,  # VectorIndexBuilder from rag_builder.py
        config: dict[str, Any],
    ):
        """
        Initialize hybrid retriever.

        Args:
            query_engine: GraphQueryEngine instance
            vector_index: Vector search index
            config: Configuration dictionary
        """
        self.query_engine = query_engine
        self.vector_index = vector_index
        self.config = config
        self.expansion_depth = config.get("expansion_depth", 2)
        self.vector_weight = config.get("vector_weight", 0.6)
        self.graph_weight = config.get("graph_weight", 0.4)

    def retrieve(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """
        Hybrid retrieval combining vector search and graph expansion.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of enhanced results with relationship context
        """
        # Step 1: Vector search to find initial concepts
        vector_results = self.vector_index.search(query, k=k * 2)

        # Step 2: Extract concepts from results
        mentioned_concepts = self._extract_concepts_from_results(vector_results)

        # Step 3: Graph expansion to find related concepts
        expanded_concepts = set()
        for concept in mentioned_concepts:
            related = self.query_engine.get_related_concepts(concept, depth=self.expansion_depth)
            for depth_concepts in related.get("related", {}).values():
                for item in depth_concepts:
                    expanded_concepts.add(item["concept"]["id"])

        # Step 4: Combine and re-rank
        combined_results = self._combine_and_rerank(query, vector_results, expanded_concepts, k)

        # Step 5: Enrich with relationship context
        enriched_results = self.enrich_with_relationships(combined_results)

        return enriched_results

    def _extract_concepts_from_results(self, results: list) -> set[str]:
        """Extract concept names from search results."""
        concepts = set()
        for result in results:
            text = result.text if hasattr(result, "text") else result.get("text", "")
            # Simple extraction - look for capitalized terms
            words = text.split()
            for word in words:
                if word[0].isupper() and len(word) > 3:
                    concept = self.query_engine.find_concept(word)
                    if concept:
                        concepts.add(concept.name)
        return concepts

    def _combine_and_rerank(
        self, query: str, vector_results: list, graph_concepts: set[str], k: int
    ) -> list[dict[str, Any]]:
        """Combine vector and graph results with reranking."""
        combined = []

        for result in vector_results:
            score = result.score if hasattr(result, "score") else result.get("score", 0)
            text = result.text if hasattr(result, "text") else result.get("text", "")

            # Boost score if result mentions graph-expanded concepts
            graph_boost = 0
            for concept in graph_concepts:
                if concept.lower() in text.lower():
                    graph_boost += 0.1

            combined_score = self.vector_weight * score + self.graph_weight * min(graph_boost, 1.0)

            combined.append(
                {
                    "text": text,
                    "score": combined_score,
                    "doc_id": result.doc_id if hasattr(result, "doc_id") else result.get("doc_id", ""),
                    "metadata": result.metadata if hasattr(result, "metadata") else result.get("metadata", {}),
                }
            )

        # Sort by combined score
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:k]

    def enrich_with_relationships(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add relationship context to results."""
        for result in results:
            text = result.get("text", "")
            relationships = []

            # Find concepts in text
            for concept in self.query_engine.concepts.values():
                if concept.name.lower() in text.lower():
                    # Get relationships for this concept
                    rels = self.query_engine.get_relationships(concept.id, direction="both")
                    relationships.extend(rels[:5])  # Limit to 5 per concept

            result["relationships"] = relationships[:10]  # Max 10 relationships per result

        return results


# ============================================================================
# 6. Graph-based Question Answering
# ============================================================================


class GraphQA:
    """Answer questions using graph reasoning."""

    def __init__(
        self, query_engine: GraphQueryEngine, llm_client: Any | None = None, config: dict[str, Any] | None = None
    ):
        """
        Initialize graph QA system.

        Args:
            query_engine: GraphQueryEngine instance
            llm_client: LLM client for answer generation
            config: Configuration dictionary
        """
        self.query_engine = query_engine
        self.llm_client = llm_client
        self.config = config or {}

    async def answer(self, question: str) -> dict[str, Any]:
        """
        Answer question using graph reasoning.

        Args:
            question: Question string

        Returns:
            Dictionary with answer and supporting evidence
        """
        # Step 1: Identify entities in question
        entities = self._identify_entities(question)

        if len(entities) < 1:
            return {
                "answer": "Could not identify relevant concepts in the question.",
                "confidence": 0.0,
                "entities": [],
                "graph_path": None,
            }

        # Step 2: Determine question type
        question_type = self._classify_question(question)

        # Step 3: Graph traversal based on question type
        if question_type == "relationship" and len(entities) >= 2:
            # Question about relationship between concepts
            graph_info = self._find_relationship_path(entities[0], entities[1])
        elif question_type == "property":
            # Question about properties of a concept
            graph_info = self._get_concept_properties(entities[0])
        elif question_type == "comparison":
            # Question comparing concepts
            graph_info = self._compare_concepts(entities[0], entities[1] if len(entities) > 1 else None)
        else:
            # General exploration
            graph_info = self.query_engine.get_related_concepts(entities[0], depth=2)

        # Step 4: Generate answer from graph information
        answer = self._generate_answer(question, entities, question_type, graph_info)

        return {
            "answer": answer,
            "confidence": 0.8,
            "entities": entities,
            "question_type": question_type,
            "graph_info": graph_info,
        }

    def _identify_entities(self, question: str) -> list[str]:
        """Identify concept entities in question."""
        entities = []
        words = question.split()

        # Try multi-word concepts first
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = " ".join(words[i:j])
                concept = self.query_engine.find_concept(phrase)
                if concept:
                    entities.append(concept.name)
                    break

        return list(set(entities))

    def _classify_question(self, question: str) -> str:
        """Classify question type."""
        q_lower = question.lower()

        if any(word in q_lower for word in ["differ", "difference", "compare", "versus", "vs"]):
            return "comparison"
        elif any(word in q_lower for word in ["relate", "connection", "between"]):
            return "relationship"
        elif any(word in q_lower for word in ["what is", "define", "explain"]):
            return "property"
        else:
            return "general"

    def _find_relationship_path(self, entity1: str, entity2: str) -> dict[str, Any]:
        """Find relationship path between entities."""
        paths = self.query_engine.find_path(entity1, entity2)

        if not paths:
            return {"path_found": False, "entities": [entity1, entity2]}

        # Get details for shortest path
        shortest_path = min(paths, key=len)
        path_details = []

        for i in range(len(shortest_path) - 1):
            source = shortest_path[i]
            target = shortest_path[i + 1]

            # Get relationship
            if source in self.query_engine.graph and target in self.query_engine.graph[source]:
                edge_data = list(self.query_engine.graph[source][target].values())[0]
                path_details.append(
                    {
                        "from": (
                            self.query_engine.concepts[source].name if source in self.query_engine.concepts else source
                        ),
                        "to": (
                            self.query_engine.concepts[target].name if target in self.query_engine.concepts else target
                        ),
                        "relation": edge_data.get("relation", "related_to"),
                    }
                )

        return {
            "path_found": True,
            "path": shortest_path,
            "path_details": path_details,
            "path_length": len(shortest_path),
        }

    def _get_concept_properties(self, entity: str) -> dict[str, Any]:
        """Get properties and relationships of a concept."""
        concept = self.query_engine.find_concept(entity)
        if not concept:
            return {"found": False}

        relationships = self.query_engine.get_relationships(concept.id, direction="both")

        return {
            "found": True,
            "concept": concept.to_dict(),
            "relationships": relationships[:10],
        }

    def _compare_concepts(self, entity1: str, entity2: str | None) -> dict[str, Any]:
        """Compare two concepts."""
        concept1 = self.query_engine.find_concept(entity1)

        if not concept1:
            return {"found": False}

        result = {
            "concept1": concept1.to_dict(),
            "concept1_relationships": self.query_engine.get_relationships(concept1.id, direction="both"),
        }

        if entity2:
            concept2 = self.query_engine.find_concept(entity2)
            if concept2:
                result["concept2"] = concept2.to_dict()
                result["concept2_relationships"] = self.query_engine.get_relationships(concept2.id, direction="both")

                # Find common relationships
                rels1 = {
                    (r["relation"], r["target"]) for r in result["concept1_relationships"] if r["source"] == concept1.id
                }
                rels2 = {
                    (r["relation"], r["target"]) for r in result["concept2_relationships"] if r["source"] == concept2.id
                }
                result["common_relationships"] = list(rels1 & rels2)

        return result

    def _generate_answer(
        self, question: str, entities: list[str], question_type: str, graph_info: dict[str, Any]
    ) -> str:
        """Generate natural language answer from graph information."""
        if question_type == "relationship":
            if graph_info.get("path_found"):
                path_details = graph_info["path_details"]
                answer_parts = []
                for detail in path_details:
                    answer_parts.append(f"{detail['from']} {detail['relation']} {detail['to']}")
                return f"The connection is: {' → '.join(answer_parts)}"
            else:
                return f"No direct relationship found between {entities[0]} and {entities[1]} in the knowledge graph."

        elif question_type == "property":
            if graph_info.get("found"):
                concept = graph_info["concept"]
                return f"{concept['name']}: {concept['description']}"
            else:
                return f"Concept {entities[0]} not found in knowledge graph."

        elif question_type == "comparison":
            if "concept1" in graph_info and "concept2" in graph_info:
                c1 = graph_info["concept1"]
                c2 = graph_info["concept2"]
                return f"{c1['name']} ({c1['type']}): {c1['description']}\n\n{c2['name']} ({c2['type']}): {c2['description']}"
            elif "concept1" in graph_info:
                c1 = graph_info["concept1"]
                return f"{c1['name']}: {c1['description']}"
            else:
                return "Concepts not found for comparison."

        else:
            if "root" in graph_info:
                concept = graph_info["root"]
                related = graph_info.get("related", {})
                answer = f"{concept['name']}: {concept['description']}\n\nRelated concepts: "
                related_names = []
                for depth_items in related.values():
                    for item in depth_items[:5]:
                        related_names.append(item["concept"]["name"])
                answer += ", ".join(related_names)
                return answer
            else:
                return "Unable to generate answer from available graph information."


# ============================================================================
# Main Interface
# ============================================================================


class KnowledgeGraphSystem:
    """Main interface for knowledge graph system."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize knowledge graph system.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            full_config = yaml.safe_load(f)

        self.config = full_config.get("knowledge_graph", {})

        # Initialize components
        self.extractor = KnowledgeExtractor(self.config.get("extraction", {}))
        self.builder = KnowledgeGraphBuilder(self.config)
        self.query_engine = GraphQueryEngine(self.builder)
        self.qa = GraphQA(self.query_engine)

        logger.info("KnowledgeGraphSystem initialized")

    def build_from_papers(self, papers: list[dict[str, Any]]) -> dict[str, int]:
        """Build knowledge graph from papers."""
        return self.builder.build_from_corpus(self.extractor, papers)

    def query(self, query: str) -> list[dict[str, Any]]:
        """Query the knowledge graph."""
        concept = self.query_engine.find_concept(query)
        if concept:
            return self.query_engine.get_relationships(concept.id, direction="both")
        return []

    async def ask(self, question: str) -> dict[str, Any]:
        """Ask a question."""
        return await self.qa.answer(question)

    def save(self):
        """Save knowledge graph."""
        self.builder.save()

    def load(self):
        """Load knowledge graph."""
        self.builder.load()

    def get_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        return self.query_engine.get_statistics()


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Testing Knowledge Graph System")

    # Test with simple manual data
    test_config = {
        "backend": "networkx",
        "storage": "./cache/knowledge_graph_test",
        "extraction": {
            "llm_model": "gpt-4-turbo-preview",
            "confidence_threshold": 0.7,
        },
    }

    # Initialize system
    builder = KnowledgeGraphBuilder(test_config)

    # Add some test concepts
    mcts = ConceptNode(
        id="mcts",
        name="MCTS",
        type="algorithm",
        description="Monte Carlo Tree Search algorithm for decision making",
        aliases=["Monte Carlo Tree Search"],
    )

    alphazero = ConceptNode(
        id="alphazero",
        name="AlphaZero",
        type="algorithm",
        description="Game-playing AI combining MCTS with deep neural networks",
        aliases=["Alpha Zero"],
    )

    ucb1 = ConceptNode(
        id="ucb1",
        name="UCB1",
        type="algorithm",
        description="Upper Confidence Bound algorithm for balancing exploration and exploitation",
        aliases=["Upper Confidence Bound"],
    )

    builder.add_concept(mcts)
    builder.add_concept(alphazero)
    builder.add_concept(ucb1)

    # Add relationships
    builder.add_relationship("alphazero", "mcts", RelationType.USES, confidence=0.95)
    builder.add_relationship("alphazero", "mcts", RelationType.EXTENDS, confidence=0.9)
    builder.add_relationship("mcts", "ucb1", RelationType.USES, confidence=0.85)

    logger.info("Added test concepts and relationships")

    # Test queries
    query_engine = GraphQueryEngine(builder)

    logger.info("\n=== Test 1: Find concept ===")
    concept = query_engine.find_concept("AlphaZero")
    if concept:
        logger.info(f"Found: {concept.name} - {concept.description}")

    logger.info("\n=== Test 2: Get relationships ===")
    rels = query_engine.get_relationships("alphazero", direction="both")
    for rel in rels:
        logger.info(f"{rel['source']} --[{rel['relation']}]--> {rel['target']}")

    logger.info("\n=== Test 3: Find path ===")
    paths = query_engine.find_path("alphazero", "ucb1")
    if paths:
        logger.info(f"Found {len(paths)} path(s): {paths[0]}")

    logger.info("\n=== Test 4: Get related concepts ===")
    related = query_engine.get_related_concepts("mcts", depth=2)
    logger.info(f"Related to MCTS: {list(related.get('related', {}).keys())}")

    logger.info("\n=== Test 5: Statistics ===")
    stats = query_engine.get_statistics()
    logger.info(f"Graph stats: {stats}")

    # Test save/load
    logger.info("\n=== Test 6: Save/Load ===")
    builder.save()
    logger.info("Saved graph")

    builder2 = KnowledgeGraphBuilder(test_config)
    builder2.load()
    logger.info(f"Loaded graph: {len(builder2.concepts)} concepts")

    logger.info("\nKnowledge Graph System test complete!")
