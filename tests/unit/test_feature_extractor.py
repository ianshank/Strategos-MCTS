"""
Tests for meta-controller feature extractor.

Tests FeatureExtractorConfig and FeatureExtractor with mocked
sentence-transformers to avoid heavy model loading.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# The feature_extractor module imports sentence_transformers at module level.
# We need to mock it before importing.
mock_st_module = MagicMock()
mock_util_module = MagicMock()
sys.modules.setdefault("sentence_transformers", mock_st_module)

from src.agents.meta_controller.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
)


@pytest.mark.unit
class TestFeatureExtractorConfig:
    """Tests for FeatureExtractorConfig."""

    def test_defaults(self):
        config = FeatureExtractorConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"

    def test_custom(self):
        config = FeatureExtractorConfig(model_name="custom-model", device="cuda")
        assert config.model_name == "custom-model"
        assert config.device == "cuda"

    @patch.dict("os.environ", {"EMBEDDING_MODEL": "paraphrase-mini", "DEVICE": "cuda:0"})
    def test_from_env(self):
        config = FeatureExtractorConfig.from_env()
        assert config.model_name == "paraphrase-mini"
        assert config.device == "cuda:0"

    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_defaults(self):
        config = FeatureExtractorConfig.from_env()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"


@pytest.mark.unit
class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def _make_mock_model(self):
        """Create a mock SentenceTransformer."""
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 384
        model.encode.return_value = np.random.randn(384).astype(np.float32)
        return model

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_init_success(self, mock_st_cls):
        mock_model = self._make_mock_model()
        mock_st_cls.return_value = mock_model

        extractor = FeatureExtractor()
        assert extractor.model is mock_model
        assert extractor.embedding_dim == 384

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_init_failure_sets_model_none(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("Model not found")
        extractor = FeatureExtractor()
        assert extractor.model is None

    @patch("src.agents.meta_controller.feature_extractor.util")
    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_extract_features_with_model(self, mock_st_cls, mock_util):
        mock_model = self._make_mock_model()
        mock_st_cls.return_value = mock_model

        # Mock cosine similarity to return reasonable scores
        import torch
        mock_util.cos_sim.return_value = torch.tensor([[0.8, 0.6, 0.5, 0.4, 0.3]])

        extractor = FeatureExtractor()
        features = extractor.extract_features("complex problem decomposition task")

        assert 0.0 <= features.hrm_confidence <= 1.0
        assert 0.0 <= features.trm_confidence <= 1.0
        assert 0.0 <= features.mcts_value <= 1.0
        assert features.query_length > 0

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_extract_features_fallback_on_model_failure(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("Model not found")
        extractor = FeatureExtractor()
        # Should use heuristic fallback
        features = extractor.extract_features("optimize the best algorithm")
        assert 0.0 <= features.hrm_confidence <= 1.0
        assert features.query_length > 0

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_heuristic_fallback_multiple_questions(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor._heuristic_fallback("What is X? What is Y?", iteration=0, last_agent="none")
        # Multiple questions should boost HRM confidence
        assert features.hrm_confidence > features.trm_confidence

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_heuristic_fallback_comparison(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor._heuristic_fallback("compare X versus Y", iteration=0, last_agent="none")
        # Comparison should boost TRM confidence
        assert features.trm_confidence > features.mcts_value

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_heuristic_fallback_optimization(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor._heuristic_fallback("optimize the best path", iteration=0, last_agent="none")
        assert features.mcts_value > 0

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_heuristic_consensus_score(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor._heuristic_fallback("hello world", iteration=0, last_agent="none")
        assert 0.0 <= features.consensus_score <= 1.0

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_extract_features_passes_iteration_and_last_agent(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor.extract_features("test query", iteration=3, last_agent="hrm")
        assert features.iteration == 3
        assert features.last_agent == "hrm"

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_extract_features_technical_query(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor.extract_features("fix the code bug in the api")
        assert features.is_technical_query is True

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_extract_features_non_technical_query(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor.extract_features("hello world")
        assert features.is_technical_query is False

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_has_rag_context_long_query(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor.extract_features("a" * 60)
        assert features.has_rag_context is True

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_has_rag_context_short_query(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor.extract_features("short")
        assert features.has_rag_context is False

    @patch("src.agents.meta_controller.feature_extractor.util")
    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_extract_features_error_during_encode(self, mock_st_cls, mock_util):
        mock_model = self._make_mock_model()
        mock_st_cls.return_value = mock_model
        # Make encode succeed during init (for prototypes) but fail during extract
        call_count = [0]

        def encode_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 3:  # Fail after prototype encoding
                raise RuntimeError("Encode failed")
            return np.random.randn(384).astype(np.float32)

        mock_model.encode.side_effect = encode_side_effect

        extractor = FeatureExtractor()
        # This should fall back to heuristics
        features = extractor.extract_features("test query")
        assert features.query_length > 0

    @patch("src.agents.meta_controller.feature_extractor.SentenceTransformer")
    def test_normalization_sums_roughly_to_one(self, mock_st_cls):
        mock_st_cls.side_effect = Exception("nope")
        extractor = FeatureExtractor()
        features = extractor._heuristic_fallback("anything", iteration=0, last_agent="none")
        total = features.hrm_confidence + features.trm_confidence + features.mcts_value
        assert total == pytest.approx(1.0, abs=0.01)
