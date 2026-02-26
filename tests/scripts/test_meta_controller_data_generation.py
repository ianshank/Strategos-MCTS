"""
Tests for meta-controller training data generation (Story 2.4).
"""

import json
import sys
from pathlib import Path

import pytest

# Skip if networkx not available (required for assembly module)
pytest.importorskip("networkx", reason="networkx required for assembly module")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_meta_controller_training_data import (
    fill_template,
    generate_query,
    generate_training_sample,
    validate_dataset,
)
from src.framework.assembly import AssemblyFeatureExtractor


class TestTemplateGeneration:
    """Test query template generation."""

    def test_fill_simple_template(self):
        """Test filling template with vocabularies."""
        template = "What is {concept}?"
        result = fill_template(template)

        assert "{concept}" not in result
        assert "?" in result
        assert len(result) > 0

    def test_fill_complex_template(self):
        """Test filling template with multiple placeholders."""
        template = "Implement {algorithm} with {constraint}"
        result = fill_template(template)

        assert "{algorithm}" not in result
        assert "{constraint}" not in result
        assert len(result) > 10

    @pytest.mark.parametrize("complexity", ["simple", "medium", "complex"])
    def test_generate_query(self, complexity):
        """Test query generation for all complexity levels."""
        query = generate_query(complexity)

        assert len(query) > 0
        assert "{" not in query  # No unfilled placeholders
        assert isinstance(query, str)

    def test_simple_queries_are_short(self):
        """Test that simple queries are generally shorter."""
        simple = generate_query("simple")
        generate_query("complex")

        # Simple queries should generally be shorter (statistical test)
        assert len(simple) < 200  # Simple queries typically < 200 chars


class TestTrainingSampleGeneration:
    """Test training sample generation."""

    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor."""
        return AssemblyFeatureExtractor()

    def test_generate_simple_sample(self, feature_extractor):
        """Test generating simple training sample."""
        sample = generate_training_sample("simple", feature_extractor)

        assert "query" in sample
        assert "features" in sample
        assert "assembly_features" in sample
        assert "ground_truth_agent" in sample
        assert "reasoning" in sample
        assert "complexity" in sample

        # Check agent is valid
        assert sample["ground_truth_agent"] in ["hrm", "trm", "mcts"]

    def test_generate_medium_sample(self, feature_extractor):
        """Test generating medium training sample."""
        sample = generate_training_sample("medium", feature_extractor)

        assert sample["complexity"] == "medium"
        assert sample["ground_truth_agent"] in ["hrm", "trm", "mcts"]

    def test_generate_complex_sample(self, feature_extractor):
        """Test generating complex training sample."""
        sample = generate_training_sample("complex", feature_extractor)

        assert sample["complexity"] == "complex"
        # Complex queries should often route to MCTS or HRM
        assert sample["ground_truth_agent"] in ["hrm", "mcts"]

    def test_assembly_features_complete(self, feature_extractor):
        """Test that all assembly features are present."""
        sample = generate_training_sample("medium", feature_extractor)

        required_assembly_features = [
            "assembly_index",
            "copy_number",
            "decomposability_score",
            "graph_depth",
            "constraint_count",
            "concept_count",
            "technical_complexity",
            "normalized_assembly_index",
        ]

        for feature in required_assembly_features:
            assert feature in sample["assembly_features"]
            # Check no NaN values
            value = sample["assembly_features"][feature]
            if isinstance(value, float):
                assert value == value  # NaN check

    def test_meta_controller_features_complete(self, feature_extractor):
        """Test that all meta-controller features are present."""
        sample = generate_training_sample("medium", feature_extractor)

        required_mc_features = [
            "hrm_confidence",
            "trm_confidence",
            "mcts_value",
            "consensus_score",
            "last_agent",
            "iteration",
            "query_length",
            "has_rag_context",
        ]

        for feature in required_mc_features:
            assert feature in sample["features"]

    def test_confidence_values_valid(self, feature_extractor):
        """Test that confidence values are in valid range."""
        sample = generate_training_sample("medium", feature_extractor)

        features = sample["features"]

        assert 0.0 <= features["hrm_confidence"] <= 1.0
        assert 0.0 <= features["trm_confidence"] <= 1.0
        assert 0.0 <= features["mcts_value"] <= 1.0
        assert 0.0 <= features["consensus_score"] <= 1.0


class TestDatasetValidation:
    """Test dataset validation."""

    def test_validate_empty_dataset(self):
        """Test validation rejects empty dataset."""
        is_valid, errors = validate_dataset([])

        assert not is_valid
        assert "empty" in errors[0].lower()

    def test_validate_missing_fields(self):
        """Test validation catches missing fields."""
        samples = [
            {
                "query": "test",
                "features": {},
                # Missing assembly_features and ground_truth_agent
            }
        ]

        is_valid, errors = validate_dataset(samples)

        assert not is_valid
        assert any("assembly_features" in error for error in errors)
        assert any("ground_truth_agent" in error for error in errors)

    def test_validate_nan_values(self):
        """Test validation catches NaN values."""
        samples = [
            {
                "query": "test",
                "features": {"hrm_confidence": float("nan")},
                "assembly_features": {"assembly_index": 5.0},
                "ground_truth_agent": "hrm",
            }
        ]

        is_valid, errors = validate_dataset(samples)

        assert not is_valid
        assert any("nan" in error.lower() for error in errors)

    def test_validate_good_dataset(self):
        """Test validation passes for good dataset."""
        from scripts.generate_meta_controller_training_data import generate_dataset

        # Generate small dataset
        samples = generate_dataset(
            num_samples=10,
            output_file="data/test_validation.json",
        )

        is_valid, errors = validate_dataset(samples)

        assert is_valid
        assert len(errors) == 0


class TestDatasetDistribution:
    """Test dataset distribution and quality."""

    @pytest.fixture
    def dataset(self):
        """Load generated dataset."""
        dataset_path = Path("data/training_with_assembly.json")
        if not dataset_path.exists():
            pytest.skip("Dataset not generated yet")

        with open(dataset_path) as f:
            return json.load(f)

    def test_dataset_size(self, dataset):
        """Test dataset has sufficient samples."""
        assert len(dataset) >= 1000

    def test_agent_distribution(self, dataset):
        """Test all agents are represented."""
        agents = [s["ground_truth_agent"] for s in dataset]

        assert "hrm" in agents
        assert "trm" in agents
        assert "mcts" in agents

        # Check rough balance (allowing some variance)
        from collections import Counter

        counts = Counter(agents)

        # At least 5% of each agent
        min_count = len(dataset) * 0.05
        assert counts["hrm"] >= min_count
        assert counts["trm"] >= min_count
        assert counts["mcts"] >= min_count

    def test_assembly_index_range(self, dataset):
        """Test assembly index has good range."""
        indices = [s["assembly_features"]["assembly_index"] for s in dataset]

        assert min(indices) >= 0
        assert max(indices) > 5  # Should have some complex queries

        # Should have variance
        import numpy as np

        assert np.std(indices) > 1.0

    def test_complexity_distribution(self, dataset):
        """Test complexity distribution follows curriculum."""
        complexities = [s["complexity"] for s in dataset]

        from collections import Counter

        counts = Counter(complexities)

        total = len(dataset)

        # Check curriculum (10% simple, 30% medium, 60% complex)
        # Allow 5% variance
        assert counts["simple"] >= total * 0.05
        assert counts["simple"] <= total * 0.15

        assert counts["medium"] >= total * 0.25
        assert counts["medium"] <= total * 0.35

        assert counts["complex"] >= total * 0.55
        assert counts["complex"] <= total * 0.65

    def test_no_nan_values(self, dataset):
        """Test dataset has no NaN values."""
        import numpy as np

        for i, sample in enumerate(dataset):
            # Check assembly features
            for key, value in sample["assembly_features"].items():
                if isinstance(value, float):
                    assert not np.isnan(value), f"Sample {i}: NaN in {key}"

            # Check meta-controller features
            for key, value in sample["features"].items():
                if isinstance(value, float):
                    assert not np.isnan(value), f"Sample {i}: NaN in {key}"

    def test_reasoning_quality(self, dataset):
        """Test reasoning explanations are present and meaningful."""
        for sample in dataset[:10]:  # Check first 10
            reasoning = sample["reasoning"]

            assert len(reasoning) > 20  # At least some explanation
            assert sample["ground_truth_agent"] in reasoning.lower() or any(
                word in reasoning.lower() for word in ["trm", "hrm", "mcts"]
            )


@pytest.mark.parametrize(
    "num_samples,simple_ratio,medium_ratio,complex_ratio",
    [
        (100, 0.10, 0.30, 0.60),
        (50, 0.20, 0.40, 0.40),
        (200, 0.15, 0.35, 0.50),
    ],
)
def test_custom_curriculum(num_samples, simple_ratio, medium_ratio, complex_ratio):
    """Test dataset generation with custom curriculum."""
    from scripts.generate_meta_controller_training_data import generate_dataset

    samples = generate_dataset(
        num_samples=num_samples,
        simple_ratio=simple_ratio,
        medium_ratio=medium_ratio,
        complex_ratio=complex_ratio,
        output_file=f"data/test_curriculum_{num_samples}.json",
    )

    assert len(samples) == num_samples

    # Check distribution
    from collections import Counter

    complexities = [s["complexity"] for s in samples]
    counts = Counter(complexities)

    # Allow 10% variance for small datasets
    tolerance = 0.10

    expected_simple = num_samples * simple_ratio
    expected_medium = num_samples * medium_ratio
    expected_complex = num_samples * complex_ratio

    assert abs(counts["simple"] - expected_simple) <= num_samples * tolerance
    assert abs(counts["medium"] - expected_medium) <= num_samples * tolerance
    assert abs(counts["complex"] - expected_complex) <= num_samples * tolerance
