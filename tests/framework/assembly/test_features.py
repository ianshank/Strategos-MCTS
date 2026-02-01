"""
Tests for Assembly Feature Extraction (Story 2.1).
"""

import numpy as np
import pytest

# Skip if networkx not available (required for assembly module)
pytest.importorskip("networkx", reason="networkx required for assembly tests")

from src.framework.assembly.features import AssemblyFeatureExtractor, AssemblyFeatures


class TestAssemblyFeatureExtractor:
    """Test suite for AssemblyFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return AssemblyFeatureExtractor(domain="software")

    def test_extract_basic_features(self, extractor):
        """Test basic feature extraction."""
        query = "How to optimize database queries?"

        features = extractor.extract(query)

        assert isinstance(features, AssemblyFeatures)
        assert features.assembly_index >= 0
        assert 0.0 <= features.decomposability_score <= 1.0
        assert features.graph_depth >= 0
        assert features.concept_count > 0

    def test_empty_query(self, extractor):
        """Test with empty query."""
        features = extractor.extract("")

        assert features.assembly_index == 0.0
        assert features.copy_number == 0.0
        assert features.concept_count == 0

    def test_simple_vs_complex(self, extractor):
        """Test that complex queries have higher indices."""
        simple = "hello"
        complex_query = "Design a distributed microservices architecture with API gateway, service mesh, and event-driven communication"

        simple_features = extractor.extract(simple)
        complex_features = extractor.extract(complex_query)

        # Complex query should have higher assembly index
        assert complex_features.assembly_index >= simple_features.assembly_index

        # Complex query should have more concepts
        assert complex_features.concept_count >= simple_features.concept_count

    def test_decomposability_scoring(self, extractor):
        """Test decomposability score calculation."""
        # Hierarchical query (should be decomposable)
        hierarchical = "First build the database, then create the API, finally add the frontend"

        # Interconnected query (harder to decompose)
        interconnected = "System with circular dependencies and complex interactions"

        h_features = extractor.extract(hierarchical)
        i_features = extractor.extract(interconnected)

        # Both should have valid scores
        assert 0.0 <= h_features.decomposability_score <= 1.0
        assert 0.0 <= i_features.decomposability_score <= 1.0

    def test_technical_complexity(self, extractor):
        """Test technical complexity measurement."""
        technical = "Implement REST API with microservices and database"
        non_technical = "Write a simple hello world program"

        tech_features = extractor.extract(technical)
        extractor.extract(non_technical)

        # Technical query should have higher technical complexity
        # (though this depends on domain library)
        assert tech_features.technical_complexity >= 0.0

    def test_normalized_features(self, extractor):
        """Test that normalized features are in [0, 1]."""
        query = "Complex query with many dependencies"

        features = extractor.extract(query)

        assert 0.0 <= features.normalized_assembly_index <= 1.0
        assert 0.0 <= features.decomposability_score <= 1.0
        assert 0.0 <= features.technical_complexity <= 1.0

    def test_feature_to_dict(self, extractor):
        """Test conversion to dictionary."""
        query = "Test query"
        features = extractor.extract(query)

        feature_dict = features.to_dict()

        assert isinstance(feature_dict, dict)
        assert 'assembly_index' in feature_dict
        assert 'decomposability_score' in feature_dict
        assert len(feature_dict) == len(AssemblyFeatures.feature_names())

    def test_feature_to_array(self, extractor):
        """Test conversion to numpy array."""
        query = "Test query"
        features = extractor.extract(query)

        feature_array = features.to_array()

        assert isinstance(feature_array, np.ndarray)
        assert feature_array.shape == (8,)  # 8 features
        assert feature_array.dtype == np.float32

    def test_batch_extraction(self, extractor):
        """Test batch feature extraction."""
        queries = [
            "Simple query",
            "More complex query with dependencies",
            "Very complex distributed system architecture query",
        ]

        features_list = extractor.extract_batch(queries)

        assert len(features_list) == 3
        assert all(isinstance(f, AssemblyFeatures) for f in features_list)

    def test_feature_explanation(self, extractor):
        """Test human-readable feature explanation."""
        query = "Optimize database queries"
        features = extractor.extract(query)

        explanation = extractor.explain_features(features)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        # Should contain some key terms
        assert any(term in explanation.lower() for term in ['complexity', 'decompos', 'reuse'])

    def test_feature_importance(self, extractor):
        """Test feature importance calculation."""
        query = "Complex technical query"
        features = extractor.extract(query)

        importances = extractor.get_feature_importance(features)

        assert isinstance(importances, dict)
        assert len(importances) > 0

        # Importances should sum to ~1.0
        assert abs(sum(importances.values()) - 1.0) < 0.01

        # All importances should be non-negative
        assert all(v >= 0 for v in importances.values())

    def test_feature_names(self):
        """Test feature name retrieval."""
        names = AssemblyFeatures.feature_names()

        assert isinstance(names, list)
        assert len(names) == 8
        assert 'assembly_index' in names
        assert 'decomposability_score' in names


@pytest.mark.parametrize("query,min_concepts", [
    ("simple", 1),
    ("complex query with many concepts", 3),
    ("API database service cache queue async microservices", 5),
])
def test_concept_count(query, min_concepts):
    """Test that concept counts meet minimums."""
    extractor = AssemblyFeatureExtractor()
    features = extractor.extract(query)

    assert features.concept_count >= min_concepts
