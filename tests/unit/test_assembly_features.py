"""
Unit tests for Assembly Feature Extraction module.

Tests AssemblyFeatures dataclass and AssemblyFeatureExtractor class.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.framework.assembly.features import AssemblyFeatureExtractor, AssemblyFeatures


@pytest.mark.unit
class TestAssemblyFeatures:
    """Tests for AssemblyFeatures dataclass."""

    def _make_features(self, **overrides) -> AssemblyFeatures:
        defaults = {
            "assembly_index": 5.0,
            "copy_number": 2.0,
            "decomposability_score": 0.75,
            "graph_depth": 3,
            "constraint_count": 4,
            "concept_count": 6,
            "technical_complexity": 0.5,
            "normalized_assembly_index": 0.25,
        }
        defaults.update(overrides)
        return AssemblyFeatures(**defaults)

    def test_creation(self) -> None:
        """AssemblyFeatures can be created with all required fields."""
        f = self._make_features()
        assert f.assembly_index == 5.0
        assert f.copy_number == 2.0
        assert f.decomposability_score == 0.75
        assert f.graph_depth == 3
        assert f.constraint_count == 4
        assert f.concept_count == 6
        assert f.technical_complexity == 0.5
        assert f.normalized_assembly_index == 0.25

    def test_to_dict(self) -> None:
        """to_dict returns all fields as a dictionary."""
        f = self._make_features()
        d = f.to_dict()
        assert isinstance(d, dict)
        assert d["assembly_index"] == 5.0
        assert d["concept_count"] == 6
        assert len(d) == 8

    def test_to_array(self) -> None:
        """to_array returns a float32 numpy array with correct values."""
        f = self._make_features()
        arr = f.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert len(arr) == 8
        assert arr[0] == pytest.approx(5.0)
        assert arr[3] == pytest.approx(3.0)  # graph_depth cast to float

    def test_to_array_order_matches_feature_names(self) -> None:
        """Array values are in the same order as feature_names."""
        f = self._make_features()
        arr = f.to_array()
        d = f.to_dict()
        names = AssemblyFeatures.feature_names()
        for i, name in enumerate(names):
            assert arr[i] == pytest.approx(float(d[name]))

    def test_feature_names(self) -> None:
        """feature_names returns correct list of names."""
        names = AssemblyFeatures.feature_names()
        assert isinstance(names, list)
        assert len(names) == 8
        assert "assembly_index" in names
        assert "normalized_assembly_index" in names


@pytest.mark.unit
class TestAssemblyFeatureExtractor:
    """Tests for AssemblyFeatureExtractor class."""

    def _make_extractor(self, **config_overrides):
        """Create an extractor with mocked internal components."""
        import networkx as nx

        extractor = AssemblyFeatureExtractor.__new__(AssemblyFeatureExtractor)

        # Set up config
        from src.framework.assembly.config import AssemblyConfig

        extractor.config = AssemblyConfig(**config_overrides)
        extractor.domain = "general"
        extractor._max_assembly_index = 20.0
        extractor._max_depth = 10.0

        # Mock calculator
        extractor.calculator = MagicMock()
        extractor.calculator.calculate.return_value = (5, 2.0)

        # Mock concept extractor
        extractor.concept_extractor = MagicMock()

        # Default: return some concepts and empty graph
        mock_concept = MagicMock()
        mock_concept.type = "noun"
        mock_concept.importance = 0.8
        extractor.concept_extractor.extract_concepts.return_value = [mock_concept]

        graph = nx.DiGraph()
        graph.add_node("concept1")
        extractor.concept_extractor.build_dependency_graph.return_value = graph

        return extractor

    def test_init_default(self) -> None:
        """Extractor initializes with default config."""
        with patch("src.framework.assembly.features.AssemblyIndexCalculator"), \
             patch("src.framework.assembly.features.ConceptExtractor"):
            extractor = AssemblyFeatureExtractor()
            assert extractor.domain == "general"
            assert extractor._max_assembly_index == 20.0

    def test_init_custom_domain(self) -> None:
        """Extractor accepts custom domain."""
        with patch("src.framework.assembly.features.AssemblyIndexCalculator"), \
             patch("src.framework.assembly.features.ConceptExtractor"):
            extractor = AssemblyFeatureExtractor(domain="software")
            assert extractor.domain == "software"

    def test_extract_empty_query(self) -> None:
        """Empty or blank queries return empty features."""
        extractor = self._make_extractor()
        result = extractor.extract("")
        assert result.assembly_index == 0.0
        assert result.concept_count == 0
        assert result.graph_depth == 0

        result2 = extractor.extract("   ")
        assert result2.assembly_index == 0.0

    def test_extract_normal_query(self) -> None:
        """Normal query returns populated features."""
        extractor = self._make_extractor()
        result = extractor.extract("How to optimize database queries?")

        assert result.assembly_index == 5.0
        assert result.copy_number == 2.0
        assert result.concept_count == 1
        assert result.normalized_assembly_index == pytest.approx(5.0 / 20.0)

    def test_extract_normalized_assembly_index_capped(self) -> None:
        """Normalized assembly index is capped at 1.0."""
        extractor = self._make_extractor()
        extractor.calculator.calculate.return_value = (25, 3.0)

        result = extractor.extract("very complex query")
        assert result.normalized_assembly_index == 1.0

    def test_extract_batch(self) -> None:
        """extract_batch processes multiple queries."""
        extractor = self._make_extractor()
        results = extractor.extract_batch(["query1", "query2", "query3"])
        assert len(results) == 3
        assert all(isinstance(r, AssemblyFeatures) for r in results)

    def test_extract_batch_empty(self) -> None:
        """extract_batch with empty list returns empty list."""
        extractor = self._make_extractor()
        results = extractor.extract_batch([])
        assert results == []

    def test_technical_complexity_all_technical(self) -> None:
        """Technical complexity is 1.0 when all concepts are technical."""
        extractor = self._make_extractor()
        tech_concept = MagicMock()
        tech_concept.type = "technical_term"
        tech_concept.importance = 0.9
        extractor.concept_extractor.extract_concepts.return_value = [tech_concept, tech_concept]

        result = extractor.extract("complex technical query")
        assert result.technical_complexity == 1.0

    def test_technical_complexity_no_technical(self) -> None:
        """Technical complexity is 0.0 when no concepts are technical."""
        extractor = self._make_extractor()
        noun = MagicMock()
        noun.type = "noun"
        noun.importance = 0.5
        extractor.concept_extractor.extract_concepts.return_value = [noun]

        result = extractor.extract("simple query")
        assert result.technical_complexity == 0.0

    def test_technical_complexity_no_concepts(self) -> None:
        """Technical complexity is 0.0 when no concepts extracted."""
        extractor = self._make_extractor()
        extractor.concept_extractor.extract_concepts.return_value = []
        import networkx as nx
        extractor.concept_extractor.build_dependency_graph.return_value = nx.DiGraph()

        result = extractor.extract("hello")
        assert result.technical_complexity == 0.0

    def test_graph_depth_empty_graph(self) -> None:
        """Graph depth is 0 for empty graph."""
        import networkx as nx
        extractor = self._make_extractor()
        extractor.concept_extractor.build_dependency_graph.return_value = nx.DiGraph()
        extractor.concept_extractor.extract_concepts.return_value = []

        result = extractor.extract("test")
        assert result.graph_depth == 0

    def test_graph_depth_dag(self) -> None:
        """Graph depth calculated correctly for a DAG."""
        import networkx as nx
        extractor = self._make_extractor()

        g = nx.DiGraph()
        g.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        extractor.concept_extractor.build_dependency_graph.return_value = g

        mock_concept = MagicMock()
        mock_concept.type = "noun"
        mock_concept.importance = 0.5
        extractor.concept_extractor.extract_concepts.return_value = [mock_concept]

        result = extractor.extract("deep query")
        assert result.graph_depth == 3  # longest path length a->b->c->d = 3 edges

    def test_decomposability_empty_concepts(self) -> None:
        """Decomposability is 0.0 with no concepts."""
        import networkx as nx
        extractor = self._make_extractor()
        score = extractor._calculate_decomposability([], nx.DiGraph())
        assert score == 0.0

    def test_decomposability_cyclic_graph(self) -> None:
        """Cyclic graphs get low decomposability score."""
        import networkx as nx
        extractor = self._make_extractor()

        g = nx.DiGraph()
        g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])

        mock_concept = MagicMock()
        mock_concept.type = "noun"
        mock_concept.importance = 0.5
        score = extractor._calculate_decomposability([mock_concept], g)
        assert score == 0.2

    def test_get_feature_importance(self) -> None:
        """Feature importances sum to 1.0."""
        extractor = self._make_extractor()
        features = AssemblyFeatures(
            assembly_index=5.0,
            copy_number=2.0,
            decomposability_score=0.5,
            graph_depth=2,
            constraint_count=3,
            concept_count=4,
            technical_complexity=0.3,
            normalized_assembly_index=0.25,
        )
        importances = extractor.get_feature_importance(features)
        assert isinstance(importances, dict)
        assert pytest.approx(sum(importances.values()), abs=1e-6) == 1.0
        assert "assembly_index" in importances

    def test_get_feature_importance_high_decomposability(self) -> None:
        """High decomposability increases its own importance."""
        extractor = self._make_extractor()
        low_decomp = AssemblyFeatures(
            assembly_index=5.0, copy_number=2.0, decomposability_score=0.3,
            graph_depth=2, constraint_count=3, concept_count=4,
            technical_complexity=0.3, normalized_assembly_index=0.25,
        )
        high_decomp = AssemblyFeatures(
            assembly_index=5.0, copy_number=2.0, decomposability_score=0.8,
            graph_depth=2, constraint_count=3, concept_count=4,
            technical_complexity=0.3, normalized_assembly_index=0.25,
        )
        imp_low = extractor.get_feature_importance(low_decomp)
        imp_high = extractor.get_feature_importance(high_decomp)
        assert imp_high["decomposability_score"] > imp_low["decomposability_score"]

    def test_explain_features_low_complexity(self) -> None:
        """Explain features for low complexity query."""
        extractor = self._make_extractor()
        features = AssemblyFeatures(
            assembly_index=2.0, copy_number=1.0, decomposability_score=0.8,
            graph_depth=1, constraint_count=0, concept_count=2,
            technical_complexity=0.1, normalized_assembly_index=0.1,
        )
        explanation = extractor.explain_features(features)
        assert "low" in explanation.lower()
        assert "highly decomposable" in explanation.lower()

    def test_explain_features_high_complexity(self) -> None:
        """Explain features for high complexity query."""
        extractor = self._make_extractor()
        features = AssemblyFeatures(
            assembly_index=9.0, copy_number=4.0, decomposability_score=0.2,
            graph_depth=5, constraint_count=10, concept_count=15,
            technical_complexity=0.7, normalized_assembly_index=0.45,
        )
        explanation = extractor.explain_features(features)
        assert "high" in explanation.lower()
        assert "difficult to decompose" in explanation.lower()
        assert "highly technical" in explanation.lower()

    def test_clear_cache(self) -> None:
        """clear_cache delegates to calculator."""
        extractor = self._make_extractor()
        extractor.clear_cache()
        extractor.calculator.clear_cache.assert_called_once()

    def test_empty_features(self) -> None:
        """_empty_features returns all-zero features."""
        extractor = self._make_extractor()
        f = extractor._empty_features()
        assert f.assembly_index == 0.0
        assert f.concept_count == 0
        assert f.graph_depth == 0
        assert f.normalized_assembly_index == 0.0
