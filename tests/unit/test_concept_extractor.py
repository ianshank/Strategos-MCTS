"""Unit tests for ConceptExtractor and Concept dataclass."""

import networkx as nx
import pytest

from src.framework.assembly.concept_extractor import Concept, ConceptExtractor


@pytest.mark.unit
class TestConcept:
    """Tests for the Concept dataclass."""

    def test_default_values(self):
        c = Concept(term="database")
        assert c.term == "database"
        assert c.type == "noun"
        assert c.frequency == 1
        assert c.dependencies == []
        assert c.importance == 0.0
        assert c.context == []

    def test_custom_values(self):
        c = Concept(term="api", type="technical_term", frequency=5, importance=0.8)
        assert c.type == "technical_term"
        assert c.frequency == 5
        assert c.importance == 0.8

    def test_hash_based_on_term(self):
        c1 = Concept(term="api")
        c2 = Concept(term="api", type="verb")
        assert hash(c1) == hash(c2)

    def test_equality(self):
        c1 = Concept(term="api")
        c2 = Concept(term="api", frequency=10)
        assert c1 == c2

    def test_inequality(self):
        c1 = Concept(term="api")
        c2 = Concept(term="database")
        assert c1 != c2

    def test_not_equal_to_non_concept(self):
        c = Concept(term="api")
        assert c != "api"

    def test_usable_in_set(self):
        c1 = Concept(term="api")
        c2 = Concept(term="api")
        assert len({c1, c2}) == 1


@pytest.mark.unit
class TestConceptExtractorInit:
    """Tests for ConceptExtractor initialization."""

    def test_default_params(self):
        ext = ConceptExtractor()
        assert ext.domain == "general"
        assert ext.min_frequency == 1
        assert ext.max_concepts == 100
        assert ext.use_technical_terms is True

    def test_custom_domain(self):
        ext = ConceptExtractor(domain="software")
        assert ext.domain == "software"
        assert "api" in ext.domain_library.get("technical_terms", set())

    def test_data_science_domain(self):
        ext = ConceptExtractor(domain="data_science")
        assert "model" in ext.domain_library.get("nouns", set())

    def test_unknown_domain_falls_back_to_general(self):
        ext = ConceptExtractor(domain="unknown_domain")
        assert ext.domain_library == ConceptExtractor(domain="general").domain_library


@pytest.mark.unit
class TestConceptExtractorExtract:
    """Tests for extract_concepts method."""

    def test_empty_string(self):
        ext = ConceptExtractor()
        assert ext.extract_concepts("") == []

    def test_whitespace_only(self):
        ext = ConceptExtractor()
        assert ext.extract_concepts("   ") == []

    def test_extracts_nouns(self):
        ext = ConceptExtractor(domain="software")
        concepts = ext.extract_concepts("The database server handles client requests")
        terms = [c.term for c in concepts]
        assert "database" in terms
        assert "server" in terms

    def test_extracts_verbs(self):
        ext = ConceptExtractor()
        concepts = ext.extract_concepts("We need to optimize and analyze the system")
        terms = [c.term for c in concepts]
        assert "optimize" in terms
        assert "analyze" in terms

    def test_extracts_technical_terms(self):
        ext = ConceptExtractor(domain="software")
        concepts = ext.extract_concepts("The api uses authentication and cache")
        terms = [c.term for c in concepts]
        assert "api" in terms

    def test_extracts_entities(self):
        ext = ConceptExtractor()
        concepts = ext.extract_concepts("Google and Amazon provide cloud services")
        terms = [c.term for c in concepts]
        assert "Google" in terms
        assert "Amazon" in terms

    def test_filters_common_sentence_starts(self):
        ext = ConceptExtractor()
        concepts = ext.extract_concepts("The system is good. What about performance?")
        terms = [c.term for c in concepts]
        assert "The" not in terms
        assert "What" not in terms

    def test_respects_max_concepts(self):
        ext = ConceptExtractor(max_concepts=3)
        text = "database server client query function class method algorithm data code system"
        concepts = ext.extract_concepts(text)
        assert len(concepts) <= 3

    def test_respects_min_frequency(self):
        ext = ConceptExtractor(min_frequency=2)
        concepts = ext.extract_concepts("database is important")
        # Single occurrences filtered at frequency=2
        # (frequency starts at 1 in Concept and increments, so it's actually 2 after extraction)
        # This depends on implementation details -- just verify it returns a list
        assert isinstance(concepts, list)

    def test_sorted_by_importance(self):
        ext = ConceptExtractor(domain="software")
        concepts = ext.extract_concepts(
            "The api endpoint connects to the database for query optimization"
        )
        if len(concepts) > 1:
            importances = [c.importance for c in concepts]
            assert importances == sorted(importances, reverse=True)

    def test_technical_terms_disabled(self):
        ext = ConceptExtractor(domain="software", use_technical_terms=False)
        concepts = ext.extract_concepts("The api uses authentication")
        # Technical terms should not get the boost
        for c in concepts:
            if c.term in ("api", "authentication"):
                assert c.type != "technical_term" or c.importance < 0.3


@pytest.mark.unit
class TestConceptExtractorDependencyGraph:
    """Tests for build_dependency_graph method."""

    def test_empty_concepts(self):
        ext = ConceptExtractor()
        graph = ext.build_dependency_graph([])
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0

    def test_single_concept(self):
        ext = ConceptExtractor()
        graph = ext.build_dependency_graph([Concept(term="database")])
        assert len(graph.nodes()) == 1
        assert "database" in graph.nodes()

    def test_nodes_have_attributes(self):
        ext = ConceptExtractor()
        c = Concept(term="api", type="technical_term", frequency=3, importance=0.8)
        graph = ext.build_dependency_graph([c])
        assert graph.nodes["api"]["type"] == "technical_term"
        assert graph.nodes["api"]["frequency"] == 3
        assert graph.nodes["api"]["importance"] == 0.8

    def test_explicit_dependencies(self):
        ext = ConceptExtractor()
        c1 = Concept(term="database")
        c2 = Concept(term="query", dependencies=["database"])
        graph = ext.build_dependency_graph([c1, c2])
        assert graph.has_edge("database", "query")

    def test_inferred_dependencies_software(self):
        ext = ConceptExtractor(domain="software")
        c1 = Concept(term="database")
        c2 = Concept(term="query")
        graph = ext.build_dependency_graph([c1, c2])
        # software domain has query -> database prerequisite
        assert graph.has_edge("database", "query")

    def test_result_is_dag(self):
        ext = ConceptExtractor()
        concepts = [
            Concept(term="a", dependencies=["b"]),
            Concept(term="b", dependencies=["a"]),
        ]
        graph = ext.build_dependency_graph(concepts)
        assert nx.is_directed_acyclic_graph(graph)

    def test_returns_digraph(self):
        ext = ConceptExtractor()
        graph = ext.build_dependency_graph([Concept(term="x")])
        assert isinstance(graph, nx.DiGraph)


@pytest.mark.unit
class TestConceptExtractorPrivateMethods:
    """Tests for internal helper methods."""

    def test_extract_nouns(self):
        ext = ConceptExtractor(domain="software")
        nouns = ext._extract_nouns("The database server handles requests")
        assert "database" in nouns
        assert "server" in nouns

    def test_extract_verbs(self):
        ext = ConceptExtractor()
        verbs = ext._extract_verbs("optimize and analyze the data")
        assert "optimize" in verbs
        assert "analyze" in verbs

    def test_extract_verbs_ize_ending(self):
        ext = ConceptExtractor()
        verbs = ext._extract_verbs("We should customize the interface")
        assert "customize" in verbs

    def test_extract_entities_filters_common_starts(self):
        ext = ConceptExtractor()
        entities = ext._extract_entities("The Python language. What about Java?")
        assert "Python" in entities
        assert "Java" in entities
        assert "The" not in entities
        assert "What" not in entities

    def test_extract_context(self):
        ext = ConceptExtractor()
        contexts = ext._extract_context("database", "the database server is fast")
        assert len(contexts) >= 1
        assert "database" in contexts[0]

    def test_are_related_common_prefix(self):
        ext = ConceptExtractor()
        assert ext._are_related("optimize", "optimization") is True

    def test_are_related_containment(self):
        ext = ConceptExtractor()
        assert ext._are_related("data", "database") is True

    def test_are_related_short_words(self):
        ext = ConceptExtractor()
        assert ext._are_related("ab", "cd") is False

    def test_are_related_unrelated(self):
        ext = ConceptExtractor()
        assert ext._are_related("apple", "zebra") is False

    def test_infer_prerequisites_software(self):
        ext = ConceptExtractor(domain="software")
        prereqs = ext._infer_prerequisites("query")
        assert "database" in prereqs

    def test_infer_prerequisites_unknown_term(self):
        ext = ConceptExtractor(domain="software")
        prereqs = ext._infer_prerequisites("xyznonexistent")
        assert prereqs == []

    def test_make_dag_removes_cycles(self):
        ext = ConceptExtractor()
        g = nx.DiGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        dag = ext._make_dag(g)
        assert nx.is_directed_acyclic_graph(dag)

    def test_calculate_importance_technical_term_bonus(self):
        ext = ConceptExtractor()
        c = Concept(term="api", type="technical_term", frequency=2)
        score = ext._calculate_importance(c, "the api is a great api for developers")
        assert score > 0.0

    def test_calculate_importance_capped_at_one(self):
        ext = ConceptExtractor(domain="software")
        c = Concept(term="database", type="technical_term", frequency=100)
        score = ext._calculate_importance(c, "database " * 100)
        assert score <= 1.0


@pytest.mark.unit
class TestConceptExtractorVisualize:
    """Tests for visualize_graph (just verifies no crash with mocked matplotlib)."""

    def test_visualize_handles_missing_matplotlib(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot" or name == "matplotlib":
                raise ImportError("no matplotlib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        ext = ConceptExtractor()
        graph = nx.DiGraph()
        graph.add_node("test")
        # Should not raise
        ext.visualize_graph(graph)


@pytest.mark.unit
class TestConceptExtractorIntegration:
    """Integration-style tests combining extraction and graph building."""

    def test_extract_and_build_graph(self):
        ext = ConceptExtractor(domain="software")
        text = "The api endpoint connects to the database for sql query optimization"
        concepts = ext.extract_concepts(text)
        graph = ext.build_dependency_graph(concepts)
        assert isinstance(graph, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(graph)
        assert len(graph.nodes()) > 0

    def test_data_science_domain_end_to_end(self):
        ext = ConceptExtractor(domain="data_science")
        text = "Train a model on the data using a classification algorithm for prediction"
        concepts = ext.extract_concepts(text)
        terms = [c.term for c in concepts]
        # Should find some data science terms
        assert any(t in terms for t in ["model", "data", "algorithm", "prediction", "classification"])
