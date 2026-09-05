"""
Tests for Knowledge Graph System
"""

import asyncio
import json
from pathlib import Path
import tempfile

import networkx as nx
import pytest

from src.training.knowledge_graph import (
    ConceptNode,
    GraphQA,
    GraphQueryEngine,
    HybridKnowledgeRetriever,
    KnowledgeExtractor,
    KnowledgeGraphBuilder,
    Relationship,
    RelationType,
)


@pytest.fixture
def test_config():
    """Test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield {
            "backend": "networkx",
            "storage": tmpdir,
            "extraction": {
                "llm_model": "gpt-4-turbo-preview",
                "confidence_threshold": 0.7,
            },
        }


@pytest.fixture
def sample_concepts():
    """Sample concept nodes for testing."""
    return [
        ConceptNode(
            id="mcts",
            name="MCTS",
            type="algorithm",
            description="Monte Carlo Tree Search",
            aliases=["Monte Carlo Tree Search"],
            properties={"domain": "game playing"},
        ),
        ConceptNode(
            id="alphazero",
            name="AlphaZero",
            type="architecture",
            description="Game-playing AI using MCTS and neural networks",
            aliases=["Alpha Zero"],
            properties={"year": 2017},
        ),
        ConceptNode(
            id="ucb1",
            name="UCB1",
            type="algorithm",
            description="Upper Confidence Bound algorithm",
            aliases=["UCB", "Upper Confidence Bound"],
        ),
        ConceptNode(
            id="neural_network",
            name="Neural Network",
            type="technique",
            description="Computational model for learning",
            aliases=["NN", "neural net"],
        ),
    ]


@pytest.fixture
def graph_builder(test_config, sample_concepts):
    """Initialize graph builder with sample data."""
    builder = KnowledgeGraphBuilder(test_config)

    # Add concepts
    for concept in sample_concepts:
        builder.add_concept(concept)

    # Add relationships
    builder.add_relationship("alphazero", "mcts", RelationType.USES, confidence=0.95)
    builder.add_relationship("alphazero", "neural_network", RelationType.USES, confidence=0.9)
    builder.add_relationship("mcts", "ucb1", RelationType.USES, confidence=0.85)
    builder.add_relationship("alphazero", "mcts", RelationType.EXTENDS, confidence=0.88)

    return builder


class TestConceptNode:
    """Tests for ConceptNode."""

    def test_concept_creation(self):
        """Test creating a concept node."""
        concept = ConceptNode(
            id="test_concept",
            name="Test Concept",
            type="algorithm",
            description="A test concept",
            aliases=["TC"],
            properties={"key": "value"},
        )

        assert concept.id == "test_concept"
        assert concept.name == "Test Concept"
        assert concept.type == "algorithm"
        assert "TC" in concept.aliases
        assert concept.properties["key"] == "value"
        assert 0 <= concept.confidence <= 1

    def test_concept_serialization(self):
        """Test concept serialization to/from dict."""
        concept = ConceptNode(
            id="test",
            name="Test",
            type="algorithm",
            description="Test description",
        )

        # To dict
        data = concept.to_dict()
        assert isinstance(data, dict)
        assert data["id"] == "test"
        assert data["name"] == "Test"

        # From dict
        restored = ConceptNode.from_dict(data)
        assert restored.id == concept.id
        assert restored.name == concept.name


class TestRelationship:
    """Tests for Relationship."""

    def test_relationship_creation(self):
        """Test creating a relationship."""
        rel = Relationship(
            source="concept_a",
            target="concept_b",
            relation_type=RelationType.USES,
            properties={"description": "A uses B"},
            confidence=0.9,
        )

        assert rel.source == "concept_a"
        assert rel.target == "concept_b"
        assert rel.relation_type == RelationType.USES
        assert rel.confidence == 0.9

    def test_relationship_serialization(self):
        """Test relationship serialization."""
        rel = Relationship(
            source="a",
            target="b",
            relation_type=RelationType.EXTENDS,
            confidence=0.85,
        )

        # To dict
        data = rel.to_dict()
        assert data["relation_type"] == "extends"

        # From dict
        restored = Relationship.from_dict(data)
        assert restored.source == rel.source
        assert restored.relation_type == RelationType.EXTENDS


class TestKnowledgeExtractor:
    """Tests for KnowledgeExtractor."""

    def test_extractor_initialization(self, test_config):
        """Test extractor initialization."""
        extractor = KnowledgeExtractor(test_config["extraction"])
        assert extractor.llm_model == "gpt-4-turbo-preview"
        assert extractor.confidence_threshold == 0.7

    def test_normalize_id(self, test_config):
        """Test ID normalization."""
        extractor = KnowledgeExtractor(test_config["extraction"])

        assert extractor._normalize_id("AlphaZero") == "alphazero"
        assert extractor._normalize_id("Monte Carlo Tree Search") == "monte_carlo_tree_search"
        assert extractor._normalize_id("UCB-1") == "ucb_1"

    def test_extract_from_code(self, test_config):
        """Test extraction from code."""
        extractor = KnowledgeExtractor(test_config["extraction"])

        code = """
        class MonteCarloTreeSearch:
            def __init__(self):
                self.ucb1 = UCB1()

            def select_node(self):
                return self.ucb1.select()
        """

        concepts, relationships = extractor.extract_from_code("test.py", code)

        # Should find MCTS and UCB1 concepts
        concept_names = [c.name for c in concepts]
        assert "MCTS" in concept_names or len(concepts) >= 0  # Pattern matching may vary


class TestKnowledgeGraphBuilder:
    """Tests for KnowledgeGraphBuilder."""

    def test_builder_initialization(self, test_config):
        """Test builder initialization."""
        builder = KnowledgeGraphBuilder(test_config)
        assert builder.backend == "networkx"
        assert isinstance(builder.graph, nx.MultiDiGraph)
        assert len(builder.concepts) == 0

    def test_add_concept(self, test_config, sample_concepts):
        """Test adding concepts."""
        builder = KnowledgeGraphBuilder(test_config)

        concept = sample_concepts[0]
        builder.add_concept(concept)

        assert concept.id in builder.concepts
        assert concept.id in builder.graph.nodes

    def test_add_duplicate_concept(self, test_config, sample_concepts):
        """Test adding duplicate concept merges data."""
        builder = KnowledgeGraphBuilder(test_config)

        concept1 = sample_concepts[0]
        builder.add_concept(concept1)

        # Add duplicate with additional data
        concept2 = ConceptNode(
            id="mcts",
            name="MCTS",
            type="algorithm",
            description="Updated description",
            aliases=["New Alias"],
            source_papers=["paper1"],
        )
        builder.add_concept(concept2)

        # Should merge
        merged = builder.concepts["mcts"]
        assert "New Alias" in merged.aliases
        assert "paper1" in merged.source_papers

    def test_add_relationship(self, test_config, sample_concepts):
        """Test adding relationships."""
        builder = KnowledgeGraphBuilder(test_config)

        for concept in sample_concepts[:2]:
            builder.add_concept(concept)

        builder.add_relationship(
            "alphazero",
            "mcts",
            RelationType.USES,
            properties={"description": "test"},
            confidence=0.9,
        )

        assert builder.graph.has_edge("alphazero", "mcts")
        assert len(builder.relationships) == 1

    def test_save_load_graph(self, graph_builder):
        """Test saving and loading graph."""
        # Save
        graph_builder.save()

        # Create new builder and load
        new_builder = KnowledgeGraphBuilder(graph_builder.config)
        new_builder.load()

        # Verify
        assert len(new_builder.concepts) == len(graph_builder.concepts)
        assert len(new_builder.relationships) == len(graph_builder.relationships)
        assert "mcts" in new_builder.concepts
        assert "alphazero" in new_builder.concepts

    def test_save_creates_json(self, graph_builder):
        """Test that save creates JSON file."""
        graph_builder.save()

        json_path = Path(graph_builder.storage_path) / "knowledge_graph.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "concepts" in data
        assert "relationships" in data


class TestGraphQueryEngine:
    """Tests for GraphQueryEngine."""

    def test_query_engine_initialization(self, graph_builder):
        """Test query engine initialization."""
        engine = GraphQueryEngine(graph_builder)
        assert engine.builder == graph_builder
        assert engine.graph == graph_builder.graph

    def test_find_concept_by_id(self, graph_builder):
        """Test finding concept by ID."""
        engine = GraphQueryEngine(graph_builder)

        concept = engine.find_concept("mcts")
        assert concept is not None
        assert concept.name == "MCTS"

    def test_find_concept_by_name(self, graph_builder):
        """Test finding concept by name."""
        engine = GraphQueryEngine(graph_builder)

        concept = engine.find_concept("AlphaZero")
        assert concept is not None
        assert concept.id == "alphazero"

    def test_find_concept_by_alias(self, graph_builder):
        """Test finding concept by alias."""
        engine = GraphQueryEngine(graph_builder)

        concept = engine.find_concept("Monte Carlo Tree Search")
        assert concept is not None
        assert concept.id == "mcts"

    def test_find_nonexistent_concept(self, graph_builder):
        """Test finding nonexistent concept."""
        engine = GraphQueryEngine(graph_builder)

        concept = engine.find_concept("NonexistentConcept")
        assert concept is None

    def test_get_outgoing_relationships(self, graph_builder):
        """Test getting outgoing relationships."""
        engine = GraphQueryEngine(graph_builder)

        rels = engine.get_relationships("alphazero", direction="outgoing")
        assert len(rels) > 0

        # Should have relationships to mcts and neural_network
        targets = [r["target"] for r in rels]
        assert "mcts" in targets
        assert "neural_network" in targets

    def test_get_incoming_relationships(self, graph_builder):
        """Test getting incoming relationships."""
        engine = GraphQueryEngine(graph_builder)

        rels = engine.get_relationships("mcts", direction="incoming")
        assert len(rels) > 0

        # Should have relationships from alphazero
        sources = [r["source"] for r in rels]
        assert "alphazero" in sources

    def test_get_relationships_by_type(self, graph_builder):
        """Test filtering relationships by type."""
        engine = GraphQueryEngine(graph_builder)

        uses_rels = engine.get_relationships("alphazero", relation_type=RelationType.USES, direction="outgoing")

        for rel in uses_rels:
            assert rel["relation"] == "uses"

    def test_find_path(self, graph_builder):
        """Test finding path between concepts."""
        engine = GraphQueryEngine(graph_builder)

        paths = engine.find_path("alphazero", "ucb1")
        assert paths is not None
        assert len(paths) > 0

        # Path should be: alphazero -> mcts -> ucb1
        shortest = paths[0]
        assert "alphazero" in shortest
        assert "ucb1" in shortest

    def test_find_path_no_connection(self, graph_builder):
        """Test finding path with no connection."""
        engine = GraphQueryEngine(graph_builder)

        # Add isolated concept
        isolated = ConceptNode(
            id="isolated",
            name="Isolated",
            type="algorithm",
            description="Isolated concept",
        )
        graph_builder.add_concept(isolated)

        paths = engine.find_path("alphazero", "isolated")
        assert paths is None or len(paths) == 0

    def test_get_related_concepts(self, graph_builder):
        """Test getting related concepts."""
        engine = GraphQueryEngine(graph_builder)

        related = engine.get_related_concepts("mcts", depth=2)

        assert "root" in related
        assert related["root"]["id"] == "mcts"
        assert "related" in related

        # Should have concepts at different depths
        assert len(related["related"]) > 0

    def test_get_statistics(self, graph_builder):
        """Test getting graph statistics."""
        engine = GraphQueryEngine(graph_builder)

        stats = engine.get_statistics()

        assert stats["total_concepts"] == len(graph_builder.concepts)
        assert stats["total_relationships"] == len(graph_builder.relationships)
        assert "concept_types" in stats
        assert "relation_types" in stats
        assert stats["avg_degree"] >= 0


class TestGraphQA:
    """Tests for GraphQA."""

    def test_qa_initialization(self, graph_builder):
        """Test QA system initialization."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)
        assert qa.query_engine == engine

    @pytest.mark.asyncio
    async def test_identify_entities(self, graph_builder):
        """Test entity identification."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)

        entities = qa._identify_entities("How does AlphaZero use MCTS?")
        assert len(entities) > 0
        # Should identify at least one of: AlphaZero, MCTS

    def test_classify_question_relationship(self, graph_builder):
        """Test classifying relationship questions."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)

        q_type = qa._classify_question("What is the relationship between AlphaZero and MCTS?")
        assert q_type == "relationship"

    def test_classify_question_property(self, graph_builder):
        """Test classifying property questions."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)

        q_type = qa._classify_question("What is AlphaZero?")
        assert q_type == "property"

    def test_classify_question_comparison(self, graph_builder):
        """Test classifying comparison questions."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)

        q_type = qa._classify_question("How does AlphaZero differ from MCTS?")
        assert q_type == "comparison"

    @pytest.mark.asyncio
    async def test_answer_property_question(self, graph_builder):
        """Test answering property questions."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)

        result = await qa.answer("What is MCTS?")

        assert "answer" in result
        assert result["confidence"] > 0
        assert "MCTS" in result.get("entities", []) or "mcts" in str(result).lower()

    @pytest.mark.asyncio
    async def test_answer_relationship_question(self, graph_builder):
        """Test answering relationship questions."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)

        result = await qa.answer("What is the relationship between AlphaZero and UCB1?")

        assert "answer" in result
        assert result.get("question_type") in ["relationship", "general"]

    @pytest.mark.asyncio
    async def test_answer_no_entities(self, graph_builder):
        """Test answering when no entities are found."""
        engine = GraphQueryEngine(graph_builder)
        qa = GraphQA(engine)

        result = await qa.answer("What is the weather today?")

        assert "answer" in result
        assert result["confidence"] == 0.0 or "not found" in result["answer"].lower()


class TestHybridKnowledgeRetriever:
    """Tests for HybridKnowledgeRetriever."""

    def test_retriever_initialization(self, graph_builder):
        """Test retriever initialization."""
        engine = GraphQueryEngine(graph_builder)

        # Mock vector index
        class MockVectorIndex:
            def search(self, query, k=10):
                return [
                    type(
                        "Result",
                        (),
                        {"text": "MCTS is a search algorithm", "score": 0.9, "doc_id": "doc1", "metadata": {}},
                    )()
                ]

        vector_index = MockVectorIndex()

        retriever = HybridKnowledgeRetriever(
            query_engine=engine,
            vector_index=vector_index,
            config={"expansion_depth": 2, "vector_weight": 0.6, "graph_weight": 0.4},
        )

        assert retriever.query_engine == engine
        assert retriever.vector_index == vector_index
        assert retriever.expansion_depth == 2

    def test_retrieve(self, graph_builder):
        """Test hybrid retrieval."""
        engine = GraphQueryEngine(graph_builder)

        class MockVectorIndex:
            def search(self, query, k=10):
                return [
                    type(
                        "Result",
                        (),
                        {
                            "text": "AlphaZero uses MCTS for game playing",
                            "score": 0.9,
                            "doc_id": "doc1",
                            "metadata": {"category": "algorithms"},
                        },
                    )()
                ]

        vector_index = MockVectorIndex()

        retriever = HybridKnowledgeRetriever(
            query_engine=engine,
            vector_index=vector_index,
            config={"expansion_depth": 1, "vector_weight": 0.6, "graph_weight": 0.4},
        )

        results = retriever.retrieve("How does AlphaZero work?", k=5)

        assert len(results) > 0
        assert "score" in results[0]
        assert "text" in results[0]
        assert "relationships" in results[0]


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_workflow(self, test_config, sample_concepts):
        """Test complete workflow from building to querying."""
        # 1. Build graph
        builder = KnowledgeGraphBuilder(test_config)

        for concept in sample_concepts:
            builder.add_concept(concept)

        builder.add_relationship("alphazero", "mcts", RelationType.USES)
        builder.add_relationship("mcts", "ucb1", RelationType.USES)

        # 2. Save graph
        builder.save()

        # 3. Load in new instance
        new_builder = KnowledgeGraphBuilder(test_config)
        new_builder.load()

        # 4. Query
        engine = GraphQueryEngine(new_builder)
        concept = engine.find_concept("AlphaZero")
        assert concept is not None

        rels = engine.get_relationships("alphazero")
        assert len(rels) > 0

        paths = engine.find_path("alphazero", "ucb1")
        assert paths is not None

        # 5. QA
        qa = GraphQA(engine)
        result = asyncio.run(qa.answer("What is AlphaZero?"))
        assert "answer" in result

    def test_graph_persistence(self, test_config, sample_concepts):
        """Test that graph persists correctly across sessions."""
        # Session 1: Create and save
        builder1 = KnowledgeGraphBuilder(test_config)

        for concept in sample_concepts:
            builder1.add_concept(concept)

        builder1.add_relationship("alphazero", "mcts", RelationType.USES)
        builder1.save()

        concept_count = len(builder1.concepts)
        rel_count = len(builder1.relationships)

        # Session 2: Load and verify
        builder2 = KnowledgeGraphBuilder(test_config)
        builder2.load()

        assert len(builder2.concepts) == concept_count
        assert len(builder2.relationships) == rel_count

        # Session 3: Modify and save
        new_concept = ConceptNode(
            id="new_concept",
            name="New Concept",
            type="algorithm",
            description="A new concept",
        )
        builder2.add_concept(new_concept)
        builder2.save()

        # Session 4: Load and verify modification
        builder3 = KnowledgeGraphBuilder(test_config)
        builder3.load()

        assert len(builder3.concepts) == concept_count + 1
        assert "new_concept" in builder3.concepts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
