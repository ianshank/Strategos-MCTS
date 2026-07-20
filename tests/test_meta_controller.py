"""
Comprehensive unit tests for the Neural Meta-Controller system.

Tests the core components:
- MetaControllerFeatures dataclass
- MetaControllerPrediction dataclass
- AbstractMetaController base class
- Utility functions (normalize_features, one_hot_encode_agent, features_to_tensor, features_to_text)
"""

from dataclasses import fields

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required")
nn = torch.nn

from src.agents.meta_controller.base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)
from src.agents.meta_controller.utils import (
    features_to_tensor,
    features_to_text,
    normalize_features,
    one_hot_encode_agent,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_features():
    """Create a standard MetaControllerFeatures instance for testing."""
    return MetaControllerFeatures(
        hrm_confidence=0.8,
        trm_confidence=0.6,
        mcts_value=0.75,
        consensus_score=0.7,
        last_agent="hrm",
        iteration=2,
        query_length=150,
        has_rag_context=True,
    )


@pytest.fixture
def features_with_extreme_values():
    """Create MetaControllerFeatures with values outside normal range."""
    return MetaControllerFeatures(
        hrm_confidence=1.5,  # Above 1.0
        trm_confidence=-0.2,  # Below 0.0
        mcts_value=2.0,  # Above 1.0
        consensus_score=0.5,
        last_agent="trm",
        iteration=30,  # Above max of 20
        query_length=15000,  # Above max of 10000
        has_rag_context=False,
    )


@pytest.fixture
def simple_state_dict():
    """Create a simple state dictionary with direct keys."""
    return {
        "hrm_confidence": 0.85,
        "trm_confidence": 0.72,
        "mcts_value": 0.68,
        "consensus_score": 0.75,
        "last_agent": "hrm",
        "iteration": 3,
        "query_length": 200,
        "has_rag_context": True,
    }


@pytest.fixture
def nested_state_dict():
    """Create a nested state dictionary with agent_confidences and mcts_state."""
    return {
        "agent_confidences": {
            "hrm": 0.9,
            "trm": 0.65,
        },
        "mcts_state": {
            "value": 0.78,
        },
        "consensus_score": 0.82,
        "last_agent": "mcts",
        "iteration": 5,
        "query": "What is machine learning and how does it work?",
        "rag_context": "Machine learning is a subset of AI...",
    }


@pytest.fixture
def empty_state_dict():
    """Create an empty state dictionary to test defaults."""
    return {}


@pytest.fixture
def concrete_meta_controller():
    """Create a concrete implementation of AbstractMetaController for testing."""

    class ConcreteMetaController(AbstractMetaController):
        def predict(self, features: MetaControllerFeatures) -> MetaControllerPrediction:
            # Simple implementation for testing
            return MetaControllerPrediction(
                agent="hrm",
                confidence=0.9,
                probabilities={"hrm": 0.6, "trm": 0.3, "mcts": 0.1},
            )

        def load_model(self, path: str) -> None:
            pass

        def save_model(self, path: str) -> None:
            pass

    return ConcreteMetaController(name="test_controller", seed=42)


# ============================================================================
# TEST: MetaControllerFeatures
# ============================================================================


class TestMetaControllerFeatures:
    """Tests for MetaControllerFeatures dataclass."""

    def test_features_creation(self, sample_features):
        """Test creating MetaControllerFeatures instance with valid values."""
        assert sample_features.hrm_confidence == 0.8
        assert sample_features.trm_confidence == 0.6
        assert sample_features.mcts_value == 0.75
        assert sample_features.consensus_score == 0.7
        assert sample_features.last_agent == "hrm"
        assert sample_features.iteration == 2
        assert sample_features.query_length == 150
        assert sample_features.has_rag_context is True

    def test_features_default_values(self):
        """Test that all fields are required (no defaults)."""
        from dataclasses import MISSING

        # Get all field names
        feature_fields = fields(MetaControllerFeatures)

        # Check that fields (except specific optional ones) have no default values
        for field in feature_fields:
            if field.name not in ("rag_relevance_score", "is_technical_query"):
                assert field.default is MISSING
                assert field.default_factory is MISSING

        # Verify that creating without all fields raises TypeError
        with pytest.raises(TypeError):
            MetaControllerFeatures()  # type: ignore

        with pytest.raises(TypeError):
            MetaControllerFeatures(hrm_confidence=0.5)  # type: ignore

        with pytest.raises(TypeError):
            MetaControllerFeatures(  # type: ignore
                hrm_confidence=0.5,
                trm_confidence=0.5,
                mcts_value=0.5,
                # Missing remaining required fields
            )

    def test_features_equality(self):
        """Test that two features with same values are equal."""
        features1 = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.75,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )
        features2 = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.75,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )
        assert features1 == features2

    def test_features_different_values(self):
        """Test that features with different values are not equal."""
        features1 = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.75,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )
        features2 = MetaControllerFeatures(
            hrm_confidence=0.9,  # Different
            trm_confidence=0.6,
            mcts_value=0.75,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )
        assert features1 != features2


# ============================================================================
# TEST: normalize_features
# ============================================================================


class TestNormalizeFeatures:
    """Tests for normalize_features utility function."""

    def test_normalize_basic(self, sample_features):
        """Test that basic normalization returns 10 floats."""
        normalized = normalize_features(sample_features)
        assert isinstance(normalized, list)
        assert len(normalized) == 10
        assert all(isinstance(v, float) for v in normalized)

    def test_normalize_values_in_range(self, sample_features):
        """Test that all normalized values are between 0 and 1."""
        normalized = normalize_features(sample_features)
        assert all(0.0 <= v <= 1.0 for v in normalized)

    def test_normalize_clipping(self, features_with_extreme_values):
        """Test that values >1 or <0 are clipped to valid range."""
        normalized = normalize_features(features_with_extreme_values)

        # All values should be clipped to [0, 1]
        assert all(0.0 <= v <= 1.0 for v in normalized)

        # Specific checks for clipped values
        # hrm_confidence (1.5) should be clipped to 1.0
        assert normalized[0] == 1.0
        # trm_confidence (-0.2) should be clipped to 0.0
        assert normalized[1] == 0.0
        # mcts_value (2.0) should be clipped to 1.0
        assert normalized[2] == 1.0
        # iteration (30) normalized as 30/20=1.5, clipped to 1.0
        assert normalized[7] == 1.0
        # query_length (15000) normalized as 15000/10000=1.5, clipped to 1.0
        assert normalized[8] == 1.0

    def test_normalize_iteration(self):
        """Test iteration/20 normalization."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.5,
            consensus_score=0.5,
            last_agent="none",
            iteration=10,  # Should normalize to 10/20 = 0.5
            query_length=100,
            has_rag_context=False,
        )
        normalized = normalize_features(features)
        # Index 7 is iteration_norm
        assert normalized[7] == pytest.approx(0.5)

        # Test with iteration = 20 (max)
        features.iteration = 20
        normalized = normalize_features(features)
        assert normalized[7] == pytest.approx(1.0)

        # Test with iteration = 0
        features.iteration = 0
        normalized = normalize_features(features)
        assert normalized[7] == pytest.approx(0.0)

    def test_normalize_query_length(self):
        """Test query_length/10000 normalization."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.5,
            consensus_score=0.5,
            last_agent="none",
            iteration=1,
            query_length=5000,  # Should normalize to 5000/10000 = 0.5
            has_rag_context=False,
        )
        normalized = normalize_features(features)
        # Index 8 is query_length_norm
        assert normalized[8] == pytest.approx(0.5)

        # Test with query_length = 10000 (max)
        features.query_length = 10000
        normalized = normalize_features(features)
        assert normalized[8] == pytest.approx(1.0)

        # Test with query_length = 0
        features.query_length = 0
        normalized = normalize_features(features)
        assert normalized[8] == pytest.approx(0.0)

    def test_normalize_vector_structure(self, sample_features):
        """Test the structure of the normalized vector."""
        normalized = normalize_features(sample_features)

        # Expected structure:
        # [hrm_conf, trm_conf, mcts_value, consensus, last_hrm, last_trm, last_mcts,
        #  iteration_norm, query_length_norm, has_rag]

        # Confidence scores (indices 0-3)
        assert normalized[0] == sample_features.hrm_confidence
        assert normalized[1] == sample_features.trm_confidence
        assert normalized[2] == sample_features.mcts_value
        assert normalized[3] == sample_features.consensus_score

        # One-hot encoding for last_agent="hrm" (indices 4-6)
        assert normalized[4] == 1.0  # hrm
        assert normalized[5] == 0.0  # trm
        assert normalized[6] == 0.0  # mcts

        # Normalized iteration (index 7)
        assert normalized[7] == pytest.approx(sample_features.iteration / 20)

        # Normalized query_length (index 8)
        assert normalized[8] == pytest.approx(sample_features.query_length / 10000)

        # Binary has_rag_context (index 9)
        assert normalized[9] == 1.0  # True -> 1.0


# ============================================================================
# TEST: one_hot_encode_agent
# ============================================================================


class TestOneHotEncode:
    """Tests for one_hot_encode_agent utility function."""

    def test_encode_hrm(self):
        """Test encoding 'hrm' returns [1,0,0]."""
        result = one_hot_encode_agent("hrm")
        assert result == [1.0, 0.0, 0.0]

    def test_encode_trm(self):
        """Test encoding 'trm' returns [0,1,0]."""
        result = one_hot_encode_agent("trm")
        assert result == [0.0, 1.0, 0.0]

    def test_encode_mcts(self):
        """Test encoding 'mcts' returns [0,0,1]."""
        result = one_hot_encode_agent("mcts")
        assert result == [0.0, 0.0, 1.0]

    def test_encode_none(self):
        """Test encoding 'none' returns [0,0,0]."""
        result = one_hot_encode_agent("none")
        assert result == [0.0, 0.0, 0.0]

    def test_encode_unknown(self):
        """Test encoding unknown agent returns [0,0,0]."""
        result = one_hot_encode_agent("unknown_agent")
        assert result == [0.0, 0.0, 0.0]

        result = one_hot_encode_agent("random")
        assert result == [0.0, 0.0, 0.0]

        result = one_hot_encode_agent("")
        assert result == [0.0, 0.0, 0.0]

    def test_encode_case_insensitive(self):
        """Test that encoding is case-insensitive."""
        # Test uppercase
        assert one_hot_encode_agent("HRM") == [1.0, 0.0, 0.0]
        assert one_hot_encode_agent("TRM") == [0.0, 1.0, 0.0]
        assert one_hot_encode_agent("MCTS") == [0.0, 0.0, 1.0]

        # Test mixed case
        assert one_hot_encode_agent("Hrm") == [1.0, 0.0, 0.0]
        assert one_hot_encode_agent("Trm") == [0.0, 1.0, 0.0]
        assert one_hot_encode_agent("Mcts") == [0.0, 0.0, 1.0]

        # Test all caps variations
        assert one_hot_encode_agent("hRm") == [1.0, 0.0, 0.0]
        assert one_hot_encode_agent("tRm") == [0.0, 1.0, 0.0]
        assert one_hot_encode_agent("mCts") == [0.0, 0.0, 1.0]

    def test_encode_returns_list_of_floats(self):
        """Test that encoding returns a list of floats."""
        result = one_hot_encode_agent("hrm")
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ============================================================================
# TEST: features_to_tensor
# ============================================================================


class TestFeaturesToTensor:
    """Tests for features_to_tensor utility function."""

    def test_tensor_shape(self, sample_features):
        """Test that tensor has correct shape [10]."""
        tensor = features_to_tensor(sample_features)
        assert tensor.shape == torch.Size([10])

    def test_tensor_dtype(self, sample_features):
        """Test that tensor has float32 dtype."""
        tensor = features_to_tensor(sample_features)
        assert tensor.dtype == torch.float32

    def test_tensor_values(self, sample_features):
        """Test that tensor values match normalize_features output."""
        tensor = features_to_tensor(sample_features)
        normalized = normalize_features(sample_features)

        # Compare each value
        for i, expected_val in enumerate(normalized):
            assert tensor[i].item() == pytest.approx(expected_val)

    def test_tensor_is_1d(self, sample_features):
        """Test that tensor is 1-dimensional."""
        tensor = features_to_tensor(sample_features)
        assert tensor.dim() == 1

    def test_tensor_device(self, sample_features):
        """Test that tensor is on CPU by default."""
        tensor = features_to_tensor(sample_features)
        assert tensor.device.type == "cpu"

    @pytest.mark.slow
    def test_tensor_batch_processing(self):
        """Test converting multiple features to tensors."""
        features_list = [
            MetaControllerFeatures(
                hrm_confidence=0.1 * i,
                trm_confidence=0.2 * i,
                mcts_value=0.3 * i,
                consensus_score=0.4 * i,
                last_agent=["none", "hrm", "trm", "mcts"][i % 4],
                iteration=i,
                query_length=100 * i,
                has_rag_context=i % 2 == 0,
            )
            for i in range(1, 5)
        ]

        tensors = [features_to_tensor(f) for f in features_list]

        # Stack into batch
        batch = torch.stack(tensors)
        assert batch.shape == torch.Size([4, 10])
        assert batch.dtype == torch.float32


# ============================================================================
# TEST: features_to_text
# ============================================================================


class TestFeaturesToText:
    """Tests for features_to_text utility function."""

    def test_text_format(self, sample_features):
        """Test that text contains expected strings."""
        text = features_to_text(sample_features)

        # Check for key components
        assert "Agent State Features:" in text
        assert "HRM confidence:" in text
        assert "TRM confidence:" in text
        assert "MCTS value:" in text
        assert "Consensus score:" in text
        assert "Last agent used:" in text
        assert "Current iteration:" in text
        assert "Query length:" in text
        assert "RAG context:" in text

    def test_rag_available(self, sample_features):
        """Test that RAG shows 'available' when has_rag_context is True."""
        text = features_to_text(sample_features)
        assert "RAG context: available" in text

    def test_rag_not_available(self):
        """Test that RAG shows 'not available' when has_rag_context is False."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.5,
            consensus_score=0.5,
            last_agent="none",
            iteration=1,
            query_length=100,
            has_rag_context=False,
        )
        text = features_to_text(features)
        assert "RAG context: not available" in text

    def test_text_includes_values(self, sample_features):
        """Test that text includes actual feature values."""
        text = features_to_text(sample_features)

        # Check numeric values are formatted
        assert "0.800" in text  # HRM confidence
        assert "0.600" in text  # TRM confidence
        assert "0.750" in text  # MCTS value
        assert "0.700" in text  # Consensus score
        assert "hrm" in text  # Last agent
        assert "2" in text  # Iteration
        assert "150 characters" in text  # Query length

    def test_text_multiline(self, sample_features):
        """Test that text output is multiline."""
        text = features_to_text(sample_features)
        lines = text.strip().split("\n")
        assert len(lines) >= 8  # At least 8 lines of information

    def test_text_formatting_precision(self):
        """Test that numeric values are formatted with 3 decimal places."""
        features = MetaControllerFeatures(
            hrm_confidence=0.123456,
            trm_confidence=0.789012,
            mcts_value=0.345678,
            consensus_score=0.901234,
            last_agent="trm",
            iteration=5,
            query_length=500,
            has_rag_context=True,
        )
        text = features_to_text(features)

        # Should be formatted to 3 decimal places
        assert "0.123" in text
        assert "0.789" in text
        assert "0.346" in text  # Rounded
        assert "0.901" in text


# ============================================================================
# TEST: AbstractMetaController
# ============================================================================


class TestAbstractMetaController:
    """Tests for AbstractMetaController base class."""

    def test_extract_features_basic(self, concrete_meta_controller, simple_state_dict):
        """Test extracting features from simple dictionary with direct keys."""
        features = concrete_meta_controller.extract_features(simple_state_dict)

        assert isinstance(features, MetaControllerFeatures)
        assert features.hrm_confidence == 0.85
        assert features.trm_confidence == 0.72
        assert features.mcts_value == 0.68
        assert features.consensus_score == 0.75
        assert features.last_agent == "hrm"
        assert features.iteration == 3
        assert features.query_length == 200
        assert features.has_rag_context is True

    def test_extract_features_nested(self, concrete_meta_controller, nested_state_dict):
        """Test extracting features from nested dictionary with agent_confidences."""
        features = concrete_meta_controller.extract_features(nested_state_dict)

        assert isinstance(features, MetaControllerFeatures)
        assert features.hrm_confidence == 0.9
        assert features.trm_confidence == 0.65
        assert features.mcts_value == 0.78
        assert features.consensus_score == 0.82
        assert features.last_agent == "mcts"
        assert features.iteration == 5
        # Query length from "What is machine learning and how does it work?"
        assert features.query_length == len("What is machine learning and how does it work?")
        # RAG context is present and non-empty
        assert features.has_rag_context is True

    def test_extract_features_missing_values(self, concrete_meta_controller, empty_state_dict):
        """Test that missing keys use default values."""
        features = concrete_meta_controller.extract_features(empty_state_dict)

        # All missing values should use defaults
        assert features.hrm_confidence == 0.0
        assert features.trm_confidence == 0.0
        assert features.mcts_value == 0.0
        assert features.consensus_score == 0.0
        assert features.last_agent == "none"
        assert features.iteration == 0
        assert features.query_length == 0
        assert features.has_rag_context is False

    def test_agent_names_constant(self, concrete_meta_controller):
        """Test that AGENT_NAMES is ['hrm', 'trm', 'mcts']."""
        assert concrete_meta_controller.AGENT_NAMES == ["hrm", "trm", "mcts"]
        assert AbstractMetaController.AGENT_NAMES == ["hrm", "trm", "mcts"]

    def test_extract_features_with_invalid_last_agent(self, concrete_meta_controller):
        """Test that invalid last_agent is converted to 'none'."""
        state = {
            "last_agent": "invalid_agent",
            "iteration": 1,
        }
        features = concrete_meta_controller.extract_features(state)
        assert features.last_agent == "none"

    def test_extract_features_with_empty_rag_context(self, concrete_meta_controller):
        """Test that empty RAG context is treated as not available."""
        state = {
            "rag_context": "",
        }
        features = concrete_meta_controller.extract_features(state)
        assert features.has_rag_context is False

    def test_extract_features_with_none_rag_context(self, concrete_meta_controller):
        """Test that None RAG context is treated as not available."""
        state = {
            "rag_context": None,
        }
        features = concrete_meta_controller.extract_features(state)
        assert features.has_rag_context is False

    def test_controller_initialization(self, concrete_meta_controller):
        """Test that controller initializes with correct name and seed."""
        assert concrete_meta_controller.name == "test_controller"
        assert concrete_meta_controller.seed == 42

    def test_controller_is_abstract(self):
        """Test that AbstractMetaController cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractMetaController(name="test", seed=42)  # type: ignore

    def test_extract_features_type_conversion(self, concrete_meta_controller):
        """Test that values are properly converted to correct types."""
        state = {
            "hrm_confidence": "0.75",  # String instead of float
            "trm_confidence": 0,  # int instead of float
            "mcts_value": "0.5",  # String
            "consensus_score": 1,  # int
            "last_agent": "hrm",
            "iteration": "3",  # String instead of int
            "query_length": "100",  # String
            "has_rag_context": 1,  # int instead of bool
        }
        features = concrete_meta_controller.extract_features(state)

        assert isinstance(features.hrm_confidence, float)
        assert isinstance(features.trm_confidence, float)
        assert isinstance(features.mcts_value, float)
        assert isinstance(features.consensus_score, float)
        assert isinstance(features.iteration, int)
        assert isinstance(features.query_length, int)
        assert isinstance(features.has_rag_context, bool)

    def test_extract_features_partial_nested(self, concrete_meta_controller):
        """Test extraction with partial nested structure."""
        state = {
            "agent_confidences": {
                "hrm": 0.8,
                # trm is missing
            },
            "mcts_state": {
                # value is missing
            },
            "consensus_score": 0.6,
        }
        features = concrete_meta_controller.extract_features(state)

        assert features.hrm_confidence == 0.8
        assert features.trm_confidence == 0.0  # Default for missing
        assert features.mcts_value == 0.0  # Default for missing


# ============================================================================
# TEST: MetaControllerPrediction
# ============================================================================


class TestMetaControllerPrediction:
    """Tests for MetaControllerPrediction dataclass."""

    def test_prediction_creation(self):
        """Test creating prediction with valid values."""
        pred = MetaControllerPrediction(
            agent="hrm",
            confidence=0.95,
            probabilities={"hrm": 0.6, "trm": 0.3, "mcts": 0.1},
        )
        assert pred.agent == "hrm"
        assert pred.confidence == 0.95
        assert pred.probabilities == {"hrm": 0.6, "trm": 0.3, "mcts": 0.1}

    def test_prediction_default_probabilities(self):
        """Test that probabilities has default empty dict."""
        pred = MetaControllerPrediction(agent="trm", confidence=0.8)
        assert pred.probabilities == {}

    def test_prediction_with_empty_probabilities(self):
        """Test prediction with explicitly empty probabilities."""
        pred = MetaControllerPrediction(
            agent="mcts",
            confidence=0.7,
            probabilities={},
        )
        assert pred.probabilities == {}


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, concrete_meta_controller, nested_state_dict):
        """Test complete pipeline from state dict to normalized features."""
        # Extract features from state
        features = concrete_meta_controller.extract_features(nested_state_dict)

        # Normalize features
        normalized = normalize_features(features)

        # Convert to tensor
        tensor = features_to_tensor(features)

        # Convert to text
        text = features_to_text(features)

        # Verify all outputs
        assert len(normalized) == 10
        assert tensor.shape == torch.Size([10])
        assert "Agent State Features:" in text

    def test_features_consistency(self, sample_features):
        """Test that features are consistent across different representations."""
        normalized = normalize_features(sample_features)
        tensor = features_to_tensor(sample_features)

        # Tensor should match normalized list
        for i in range(10):
            assert tensor[i].item() == pytest.approx(normalized[i])

    @pytest.mark.slow
    def test_multiple_states_extraction(self, concrete_meta_controller):
        """Test extracting features from multiple different state configurations."""
        states = [
            {"hrm_confidence": 0.9, "trm_confidence": 0.1},
            {"agent_confidences": {"hrm": 0.5, "trm": 0.8}},
            {"mcts_state": {"value": 0.95}, "consensus_score": 0.7},
            {},  # Empty state
        ]

        features_list = [concrete_meta_controller.extract_features(state) for state in states]

        assert len(features_list) == 4
        assert all(isinstance(f, MetaControllerFeatures) for f in features_list)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_values(self):
        """Test features with all zero values."""
        features = MetaControllerFeatures(
            hrm_confidence=0.0,
            trm_confidence=0.0,
            mcts_value=0.0,
            consensus_score=0.0,
            last_agent="none",
            iteration=0,
            query_length=0,
            has_rag_context=False,
        )
        normalized = normalize_features(features)
        assert all(v >= 0.0 for v in normalized)

    def test_max_values(self):
        """Test features with maximum valid values."""
        features = MetaControllerFeatures(
            hrm_confidence=1.0,
            trm_confidence=1.0,
            mcts_value=1.0,
            consensus_score=1.0,
            last_agent="hrm",
            iteration=20,
            query_length=10000,
            has_rag_context=True,
        )
        normalized = normalize_features(features)
        assert all(0.0 <= v <= 1.0 for v in normalized)

    def test_very_long_query(self):
        """Test with query length exceeding normal maximum."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.5,
            consensus_score=0.5,
            last_agent="none",
            iteration=1,
            query_length=100000,  # Very long
            has_rag_context=False,
        )
        normalized = normalize_features(features)
        # Should be clipped to 1.0
        assert normalized[8] == 1.0

    def test_negative_iteration(self):
        """Test with negative iteration (should be clipped to 0)."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.5,
            consensus_score=0.5,
            last_agent="none",
            iteration=-5,
            query_length=100,
            has_rag_context=False,
        )
        normalized = normalize_features(features)
        # Should be clipped to 0.0
        assert normalized[7] == 0.0


# ============================================================================
# RNN META-CONTROLLER TESTS
# ============================================================================

from src.agents.meta_controller.bert_controller import BERTMetaController  # noqa: E402
from src.agents.meta_controller.rnn_controller import (  # noqa: E402
    RNNMetaController,
    RNNMetaControllerModel,
)


@pytest.fixture
def rnn_controller() -> RNNMetaController:
    """Create an RNNMetaController instance for testing."""
    return RNNMetaController(name="TestRNN", seed=42)


class TestRNNMetaControllerModel:
    """Tests for RNNMetaControllerModel neural network."""

    def test_model_initialization(self) -> None:
        """Test that model initializes with default parameters correctly."""
        model = RNNMetaControllerModel()

        assert model.hidden_dim == 64
        assert model.num_layers == 1
        assert isinstance(model.gru, nn.GRU)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.fc, nn.Linear)

        # Check GRU configuration
        assert model.gru.input_size == 10
        assert model.gru.hidden_size == 64
        assert model.gru.num_layers == 1
        assert model.gru.batch_first is True

        # Check output layer configuration
        assert model.fc.in_features == 64
        assert model.fc.out_features == 3

    def test_model_forward_2d_input(self) -> None:
        """Test forward pass with 2D input (batch_size, features)."""
        model = RNNMetaControllerModel()
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 10)

        with torch.no_grad():
            output = model(x)

        assert output.shape == torch.Size([batch_size, 3])
        assert output.dtype == torch.float32

    def test_model_forward_3d_input(self) -> None:
        """Test forward pass with 3D input (batch_size, seq_len, features)."""
        model = RNNMetaControllerModel()
        model.eval()

        batch_size = 4
        seq_len = 5
        x = torch.randn(batch_size, seq_len, 10)

        with torch.no_grad():
            output = model(x)

        assert output.shape == torch.Size([batch_size, 3])
        assert output.dtype == torch.float32

    def test_model_output_shape(self) -> None:
        """Test that output shape is (batch_size, 3) for various batch sizes."""
        model = RNNMetaControllerModel()
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 10)
            with torch.no_grad():
                output = model(x)
            assert output.shape == torch.Size([batch_size, 3])

    def test_model_returns_logits(self) -> None:
        """Test that model returns unnormalized logits (not probabilities)."""
        model = RNNMetaControllerModel()
        model.eval()

        x = torch.randn(4, 10)

        with torch.no_grad():
            output = model(x)

        # Logits can be negative or greater than 1
        # They should NOT sum to 1 (that would indicate probabilities)
        row_sums = output.sum(dim=1)

        # At least some rows should not sum to exactly 1
        # (with very high probability given random initialization)
        assert not torch.allclose(row_sums, torch.ones(4))

        # Logits can be any real number
        assert output.min().item() < 1.0 or output.max().item() > 1.0

    def test_model_different_hidden_dims(self) -> None:
        """Test model with different hidden dimensions (32, 128)."""
        for hidden_dim in [32, 128]:
            model = RNNMetaControllerModel(hidden_dim=hidden_dim)
            model.eval()

            assert model.hidden_dim == hidden_dim
            assert model.gru.hidden_size == hidden_dim
            assert model.fc.in_features == hidden_dim

            x = torch.randn(2, 10)
            with torch.no_grad():
                output = model(x)

            assert output.shape == torch.Size([2, 3])

    @pytest.mark.slow
    def test_model_multiple_layers(self) -> None:
        """Test model with multiple GRU layers."""
        model = RNNMetaControllerModel(num_layers=2)
        model.eval()

        assert model.num_layers == 2
        assert model.gru.num_layers == 2

        x = torch.randn(4, 10)
        with torch.no_grad():
            output = model(x)

        assert output.shape == torch.Size([4, 3])

        # Test with 3D input as well
        x_3d = torch.randn(4, 5, 10)
        with torch.no_grad():
            output_3d = model(x_3d)

        assert output_3d.shape == torch.Size([4, 3])


class TestRNNMetaController:
    """Tests for RNNMetaController class."""

    def test_controller_initialization(self, rnn_controller: RNNMetaController) -> None:
        """Test that controller initializes with correct name, seed, and device."""
        assert rnn_controller.name == "TestRNN"
        assert rnn_controller.seed == 42
        assert isinstance(rnn_controller.device, torch.device)
        assert rnn_controller.hidden_dim == 64
        assert rnn_controller.num_layers == 1
        assert rnn_controller.dropout == 0.1
        assert isinstance(rnn_controller.model, RNNMetaControllerModel)

    def test_controller_predict_returns_prediction(
        self, rnn_controller: RNNMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that predict returns MetaControllerPrediction object."""
        prediction = rnn_controller.predict(sample_features)

        assert isinstance(prediction, MetaControllerPrediction)
        assert hasattr(prediction, "agent")
        assert hasattr(prediction, "confidence")
        assert hasattr(prediction, "probabilities")

    def test_controller_predict_valid_agent(
        self, rnn_controller: RNNMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that predicted agent is in AGENT_NAMES."""
        prediction = rnn_controller.predict(sample_features)

        assert prediction.agent in AbstractMetaController.AGENT_NAMES
        assert prediction.agent in ["hrm", "trm", "mcts"]

    def test_controller_predict_confidence_range(
        self, rnn_controller: RNNMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that confidence is between 0 and 1."""
        prediction = rnn_controller.predict(sample_features)

        assert isinstance(prediction.confidence, float)
        assert 0.0 <= prediction.confidence <= 1.0

    def test_controller_predict_probabilities_sum_to_one(
        self, rnn_controller: RNNMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that probabilities sum to approximately 1.0."""
        prediction = rnn_controller.predict(sample_features)

        assert len(prediction.probabilities) == 3
        assert set(prediction.probabilities.keys()) == {"hrm", "trm", "mcts"}

        prob_sum = sum(prediction.probabilities.values())
        assert prob_sum == pytest.approx(1.0, abs=1e-6)

        # All probabilities should be valid
        for prob in prediction.probabilities.values():
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_controller_deterministic_with_seed(self, sample_features: MetaControllerFeatures) -> None:
        """Test that same seed produces same results."""
        controller1 = RNNMetaController(name="Test1", seed=123)
        controller2 = RNNMetaController(name="Test2", seed=123)

        prediction1 = controller1.predict(sample_features)
        prediction2 = controller2.predict(sample_features)

        assert prediction1.agent == prediction2.agent
        assert prediction1.confidence == pytest.approx(prediction2.confidence, abs=1e-6)

        for agent in ["hrm", "trm", "mcts"]:
            assert prediction1.probabilities[agent] == pytest.approx(prediction2.probabilities[agent], abs=1e-6)

    @pytest.mark.slow
    def test_controller_save_and_load_model(
        self,
        rnn_controller: RNNMetaController,
        sample_features: MetaControllerFeatures,
        tmp_path,
    ) -> None:
        """Test save and load roundtrip preserves model weights."""
        # Get prediction before save
        prediction_before = rnn_controller.predict(sample_features)

        # Save model
        model_path = tmp_path / "test_model.pt"
        rnn_controller.save_model(str(model_path))

        # Verify file was created
        assert model_path.exists()

        # Create new controller and load model
        new_controller = RNNMetaController(name="LoadedRNN", seed=99)
        new_controller.load_model(str(model_path))

        # Get prediction after load
        prediction_after = new_controller.predict(sample_features)

        # Predictions should be identical
        assert prediction_before.agent == prediction_after.agent
        assert prediction_before.confidence == pytest.approx(prediction_after.confidence, abs=1e-6)

        for agent in ["hrm", "trm", "mcts"]:
            assert prediction_before.probabilities[agent] == pytest.approx(
                prediction_after.probabilities[agent], abs=1e-6
            )

    def test_controller_reset_hidden_state(self, rnn_controller: RNNMetaController) -> None:
        """Test that reset_hidden_state() sets hidden state to None."""
        # Initially should be None
        assert rnn_controller.hidden_state is None

        # Set to some value
        rnn_controller.hidden_state = torch.randn(1, 64)
        assert rnn_controller.hidden_state is not None

        # Reset
        rnn_controller.reset_hidden_state()
        assert rnn_controller.hidden_state is None

    def test_controller_device_auto_detection(self) -> None:
        """Test that controller auto-detects device correctly."""
        controller = RNNMetaController(name="AutoDevice", seed=42, device=None)

        # Device should be set based on availability
        assert isinstance(controller.device, torch.device)

        # Device type should be one of the expected types
        assert controller.device.type in ["cpu", "cuda", "mps"]

        # If CUDA is available, it should be selected
        if torch.cuda.is_available():
            assert controller.device.type == "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert controller.device.type == "mps"
        else:
            assert controller.device.type == "cpu"

        # Model should be on the same device
        model_device = next(controller.model.parameters()).device
        assert model_device.type == controller.device.type


# ============================================================================
# BERT META-CONTROLLER TESTS
# ============================================================================


@pytest.fixture
def bert_controller() -> BERTMetaController:
    """Create a BERTMetaController instance for testing."""
    return BERTMetaController(name="TestBERT", seed=42)


class TestBERTMetaController:
    """Tests for BERTMetaController class with LoRA support."""

    @pytest.mark.slow
    def test_controller_initialization(self, bert_controller: BERTMetaController) -> None:
        """Test that controller initializes with correct name, seed, device, model_name, and lora settings."""
        assert bert_controller.name == "TestBERT"
        assert bert_controller.seed == 42
        assert isinstance(bert_controller.device, torch.device)
        assert bert_controller.model_name == "prajjwal1/bert-mini"
        assert bert_controller.lora_r == 4
        assert bert_controller.lora_alpha == 16
        assert bert_controller.lora_dropout == 0.1
        assert bert_controller.use_lora is True

    def test_controller_default_model_name(self) -> None:
        """Test that DEFAULT_MODEL_NAME is set to 'prajjwal1/bert-mini'."""
        assert BERTMetaController.DEFAULT_MODEL_NAME == "prajjwal1/bert-mini"

    def test_controller_num_labels(self) -> None:
        """Test that NUM_LABELS is set to 3 (hrm, trm, mcts)."""
        assert BERTMetaController.NUM_LABELS == 3

    @pytest.mark.slow
    def test_controller_predict_returns_prediction(
        self, bert_controller: BERTMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that predict returns MetaControllerPrediction object."""
        prediction = bert_controller.predict(sample_features)

        assert isinstance(prediction, MetaControllerPrediction)
        assert hasattr(prediction, "agent")
        assert hasattr(prediction, "confidence")
        assert hasattr(prediction, "probabilities")

    @pytest.mark.slow
    def test_controller_predict_valid_agent(
        self, bert_controller: BERTMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that predicted agent is in AGENT_NAMES."""
        prediction = bert_controller.predict(sample_features)

        assert prediction.agent in AbstractMetaController.AGENT_NAMES
        assert prediction.agent in ["hrm", "trm", "mcts"]

    @pytest.mark.slow
    def test_controller_predict_confidence_range(
        self, bert_controller: BERTMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that confidence is between 0 and 1."""
        prediction = bert_controller.predict(sample_features)

        assert isinstance(prediction.confidence, float)
        assert 0.0 <= prediction.confidence <= 1.0

    @pytest.mark.slow
    def test_controller_predict_probabilities_sum_to_one(
        self, bert_controller: BERTMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that probabilities sum to approximately 1.0."""
        prediction = bert_controller.predict(sample_features)

        assert len(prediction.probabilities) == 3
        assert set(prediction.probabilities.keys()) == {"hrm", "trm", "mcts"}

        prob_sum = sum(prediction.probabilities.values())
        assert prob_sum == pytest.approx(1.0, abs=1e-6)

        # All probabilities should be valid
        for prob in prediction.probabilities.values():
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    @pytest.mark.slow
    def test_controller_tokenization_caching(
        self, bert_controller: BERTMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that same features use cached tokenization."""
        # Clear cache first
        bert_controller.clear_cache()
        assert bert_controller.get_cache_info()["cache_size"] == 0

        # First prediction should populate cache
        bert_controller.predict(sample_features)
        cache_info_after_first = bert_controller.get_cache_info()
        assert cache_info_after_first["cache_size"] == 1

        # Second prediction with same features should use cache (size unchanged)
        bert_controller.predict(sample_features)
        cache_info_after_second = bert_controller.get_cache_info()
        assert cache_info_after_second["cache_size"] == 1

        # Different features should add new cache entry
        different_features = MetaControllerFeatures(
            hrm_confidence=0.9,
            trm_confidence=0.4,
            mcts_value=0.65,
            consensus_score=0.8,
            last_agent="trm",
            iteration=5,
            query_length=200,
            has_rag_context=False,
        )
        bert_controller.predict(different_features)
        cache_info_after_third = bert_controller.get_cache_info()
        assert cache_info_after_third["cache_size"] == 2

    @pytest.mark.slow
    def test_controller_clear_cache(
        self, bert_controller: BERTMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that clear_cache() empties the tokenization cache."""
        # Populate cache
        bert_controller.predict(sample_features)
        assert bert_controller.get_cache_info()["cache_size"] > 0

        # Clear cache
        bert_controller.clear_cache()
        cache_info = bert_controller.get_cache_info()
        assert cache_info["cache_size"] == 0
        assert cache_info["cache_keys"] == []

    @pytest.mark.slow
    def test_controller_get_cache_info(
        self, bert_controller: BERTMetaController, sample_features: MetaControllerFeatures
    ) -> None:
        """Test that get_cache_info returns dict with cache_size and cache_keys."""
        bert_controller.clear_cache()

        # Empty cache info
        cache_info = bert_controller.get_cache_info()
        assert isinstance(cache_info, dict)
        assert "cache_size" in cache_info
        assert "cache_keys" in cache_info
        assert cache_info["cache_size"] == 0
        assert isinstance(cache_info["cache_keys"], list)

        # After prediction
        bert_controller.predict(sample_features)
        cache_info = bert_controller.get_cache_info()
        assert cache_info["cache_size"] == 1
        assert len(cache_info["cache_keys"]) == 1
        assert isinstance(cache_info["cache_keys"][0], str)

    @pytest.mark.slow
    def test_controller_without_lora(self, sample_features: MetaControllerFeatures) -> None:
        """Test controller initialization with use_lora=False."""
        controller = BERTMetaController(name="NoLoRA", seed=42, use_lora=False)

        assert controller.use_lora is False
        assert controller.name == "NoLoRA"

        # Should still be able to make predictions
        prediction = controller.predict(sample_features)
        assert isinstance(prediction, MetaControllerPrediction)
        assert prediction.agent in ["hrm", "trm", "mcts"]
        assert 0.0 <= prediction.confidence <= 1.0

    @pytest.mark.slow
    def test_controller_get_trainable_parameters(self, bert_controller: BERTMetaController) -> None:
        """Test that get_trainable_parameters returns dict with trainable statistics."""
        params = bert_controller.get_trainable_parameters()

        assert isinstance(params, dict)
        assert "total_params" in params
        assert "trainable_params" in params
        assert "trainable_percentage" in params

        assert isinstance(params["total_params"], int)
        assert isinstance(params["trainable_params"], int)
        assert isinstance(params["trainable_percentage"], float)

        # Total params should be positive
        assert params["total_params"] > 0

        # Trainable params should be less than or equal to total
        assert params["trainable_params"] <= params["total_params"]

        # Percentage should be between 0 and 100
        assert 0.0 <= params["trainable_percentage"] <= 100.0

        # For LoRA model, trainable percentage should be relatively small
        if bert_controller.use_lora:
            assert params["trainable_percentage"] < 50.0

    @pytest.mark.slow
    def test_controller_save_and_load_model(
        self,
        bert_controller: BERTMetaController,
        sample_features: MetaControllerFeatures,
        tmp_path,
    ) -> None:
        """Test save and load roundtrip preserves model behavior."""
        # Get prediction before save
        prediction_before = bert_controller.predict(sample_features)

        # Save model (for LoRA, it's a directory)
        model_path = tmp_path / "bert_lora_adapter"
        bert_controller.save_model(str(model_path))

        # Verify directory was created (for LoRA models)
        assert model_path.exists()

        # Create new controller and load model
        new_controller = BERTMetaController(name="LoadedBERT", seed=99)
        new_controller.load_model(str(model_path))

        # Get prediction after load
        prediction_after = new_controller.predict(sample_features)

        # Predictions should be identical
        assert prediction_before.agent == prediction_after.agent
        assert prediction_before.confidence == pytest.approx(prediction_after.confidence, abs=1e-6)

        for agent in ["hrm", "trm", "mcts"]:
            assert prediction_before.probabilities[agent] == pytest.approx(
                prediction_after.probabilities[agent], abs=1e-6
            )
