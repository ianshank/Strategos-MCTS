"""
Tests for hybrid meta-controller module.

Tests HybridPrediction, HybridMetaController initialization,
prediction paths (neural only, assembly only, ensemble, fallback),
and statistics tracking.
"""

from unittest.mock import MagicMock

import pytest

from src.agents.meta_controller.base import MetaControllerFeatures, MetaControllerPrediction
from src.agents.meta_controller.hybrid_controller import HybridMetaController, HybridPrediction


def _make_features(**overrides):
    defaults = {
        "hrm_confidence": 0.5,
        "trm_confidence": 0.3,
        "mcts_value": 0.2,
        "consensus_score": 0.7,
        "last_agent": "none",
        "iteration": 0,
        "query_length": 50,
        "has_rag_context": False,
        "rag_relevance_score": 0.0,
        "is_technical_query": False,
    }
    defaults.update(overrides)
    return MetaControllerFeatures(**defaults)


def _make_neural_prediction(agent="hrm", confidence=0.8):
    return MetaControllerPrediction(
        agent=agent,
        confidence=confidence,
        probabilities=(
            {"hrm": 0.7, "trm": 0.2, "mcts": 0.1} if agent == "hrm" else {"hrm": 0.1, "trm": 0.7, "mcts": 0.2}
        ),
    )


@pytest.mark.unit
class TestHybridPrediction:
    """Tests for HybridPrediction dataclass."""

    def test_defaults(self):
        pred = HybridPrediction(
            agent="hrm",
            confidence=0.8,
            probabilities={"hrm": 0.8, "trm": 0.1, "mcts": 0.1},
        )
        assert pred.agent == "hrm"
        assert pred.confidence == 0.8
        assert pred.neural_weight == 0.6
        assert pred.assembly_weight == 0.4
        assert pred.neural_prediction is None
        assert pred.assembly_decision is None


@pytest.mark.unit
class TestHybridMetaController:
    """Tests for HybridMetaController."""

    def test_init_defaults(self):
        controller = HybridMetaController()
        assert controller.neural_weight == pytest.approx(0.6)
        assert controller.assembly_weight == pytest.approx(0.4)
        assert controller.neural_controller is None

    def test_init_custom_weights(self):
        controller = HybridMetaController(neural_weight=0.3, assembly_weight=0.7)
        assert controller.neural_weight == pytest.approx(0.3)
        assert controller.assembly_weight == pytest.approx(0.7)

    def test_weight_normalization(self):
        controller = HybridMetaController(neural_weight=3.0, assembly_weight=7.0)
        assert controller.neural_weight == pytest.approx(0.3)
        assert controller.assembly_weight == pytest.approx(0.7)

    def test_set_query_context(self):
        controller = HybridMetaController()
        controller.set_query_context("test query")
        assert controller._current_query == "test query"
        assert controller._current_assembly_features is None

    def test_predict_fallback_no_neural_no_query(self):
        """When no neural controller and no query, should fallback."""
        controller = HybridMetaController()
        features = _make_features()
        pred = controller.predict(features)
        assert isinstance(pred, HybridPrediction)
        assert pred.agent in ["hrm", "trm", "mcts"]

    def test_predict_neural_only(self):
        """When neural controller available but no query."""
        mock_neural = MagicMock()
        mock_neural.predict.return_value = _make_neural_prediction("hrm", 0.9)
        controller = HybridMetaController(neural_controller=mock_neural)
        features = _make_features()
        pred = controller.predict(features)
        assert isinstance(pred, HybridPrediction)
        assert pred.agent == "hrm"

    def test_predict_assembly_only(self):
        """When no neural controller but query available."""
        controller = HybridMetaController()
        features = _make_features()
        controller.set_query_context("complex problem decomposition task")
        pred = controller.predict(features)
        assert isinstance(pred, HybridPrediction)
        assert pred.agent in ["hrm", "trm", "mcts"]

    def test_predict_ensemble(self):
        """When both neural and assembly available."""
        mock_neural = MagicMock()
        mock_neural.predict.return_value = _make_neural_prediction("hrm", 0.9)
        controller = HybridMetaController(neural_controller=mock_neural)
        features = _make_features()
        controller.set_query_context("optimize the best algorithm")
        pred = controller.predict(features)
        assert isinstance(pred, HybridPrediction)
        assert pred.agent in ["hrm", "trm", "mcts"]

    def test_predict_with_query_parameter(self):
        """Query can be passed as parameter instead of set_query_context."""
        mock_neural = MagicMock()
        mock_neural.predict.return_value = _make_neural_prediction("trm", 0.8)
        controller = HybridMetaController(neural_controller=mock_neural)
        features = _make_features()
        pred = controller.predict(features, query="compare two approaches")
        assert isinstance(pred, HybridPrediction)

    def test_predict_neural_failure_falls_back(self):
        """If neural prediction fails, should fall back to assembly."""
        mock_neural = MagicMock()
        mock_neural.predict.side_effect = RuntimeError("model error")
        controller = HybridMetaController(neural_controller=mock_neural)
        features = _make_features()
        controller.set_query_context("some query")
        pred = controller.predict(features)
        assert isinstance(pred, HybridPrediction)

    def test_stats_tracking(self):
        controller = HybridMetaController()
        features = _make_features()
        controller.predict(features)
        controller.predict(features)
        assert controller._stats["total_predictions"] == 2

    def test_get_stats(self):
        controller = HybridMetaController()
        stats = controller._stats
        assert "total_predictions" in stats
        assert "neural_dominant" in stats
        assert "assembly_dominant" in stats
        assert "neural_override" in stats
        assert "assembly_override" in stats
