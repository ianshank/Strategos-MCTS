"""
Tests for meta-controller base module.

Tests MetaControllerFeatures, MetaControllerPrediction,
AbstractMetaController, and extract_features.
"""

import pytest

from src.agents.meta_controller.base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)


class DummyController(AbstractMetaController):
    """Concrete controller for testing."""

    def predict(self, features):
        return MetaControllerPrediction(
            agent="hrm", confidence=0.9, probabilities={"hrm": 0.9, "trm": 0.05, "mcts": 0.05}
        )

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass


@pytest.mark.unit
class TestMetaControllerFeatures:
    def test_basic(self):
        f = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.5,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=1,
            query_length=100,
            has_rag_context=True,
        )
        assert f.hrm_confidence == 0.8
        assert f.rag_relevance_score == 0.0
        assert f.is_technical_query is False

    def test_optional_fields(self):
        f = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.5,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=1,
            query_length=100,
            has_rag_context=True,
            rag_relevance_score=0.9,
            is_technical_query=True,
        )
        assert f.rag_relevance_score == 0.9
        assert f.is_technical_query is True


@pytest.mark.unit
class TestMetaControllerPrediction:
    def test_basic(self):
        p = MetaControllerPrediction(agent="trm", confidence=0.8)
        assert p.agent == "trm"
        assert p.confidence == 0.8
        assert p.probabilities == {}

    def test_with_probabilities(self):
        p = MetaControllerPrediction(
            agent="mcts",
            confidence=0.7,
            probabilities={"hrm": 0.1, "trm": 0.2, "mcts": 0.7},
        )
        assert p.probabilities["mcts"] == 0.7


@pytest.mark.unit
class TestAbstractMetaController:
    def test_init(self):
        ctrl = DummyController(name="test", seed=99)
        assert ctrl.name == "test"
        assert ctrl.seed == 99

    def test_agent_names(self):
        assert AbstractMetaController.AGENT_NAMES == ["hrm", "trm", "mcts"]

    def test_extract_features_flat(self):
        ctrl = DummyController(name="test")
        state = {
            "hrm_confidence": 0.8,
            "trm_confidence": 0.6,
            "mcts_value": 0.75,
            "consensus_score": 0.7,
            "last_agent": "hrm",
            "iteration": 2,
            "query_length": 150,
            "has_rag_context": True,
        }
        features = ctrl.extract_features(state)
        assert features.hrm_confidence == 0.8
        assert features.trm_confidence == 0.6
        assert features.mcts_value == 0.75
        assert features.iteration == 2
        assert features.has_rag_context is True

    def test_extract_features_nested(self):
        ctrl = DummyController(name="test")
        state = {
            "agent_confidences": {"hrm": 0.9, "trm": 0.4},
            "mcts_state": {"value": 0.6},
            "consensus_score": 0.5,
            "last_agent": "trm",
            "iteration": 1,
            "query": "What is AI?",
            "rag_context": "AI is artificial intelligence",
        }
        features = ctrl.extract_features(state)
        assert features.hrm_confidence == 0.9
        assert features.trm_confidence == 0.4
        assert features.mcts_value == 0.6
        assert features.query_length == len("What is AI?")
        assert features.has_rag_context is True

    def test_extract_features_defaults(self):
        ctrl = DummyController(name="test")
        features = ctrl.extract_features({})
        assert features.hrm_confidence == 0.0
        assert features.trm_confidence == 0.0
        assert features.mcts_value == 0.0
        assert features.last_agent == "none"
        assert features.iteration == 0
        assert features.query_length == 0
        assert features.has_rag_context is False

    def test_extract_features_invalid_last_agent(self):
        ctrl = DummyController(name="test")
        features = ctrl.extract_features({"last_agent": "unknown_agent"})
        assert features.last_agent == "none"

    def test_extract_features_rag_context_none(self):
        ctrl = DummyController(name="test")
        features = ctrl.extract_features({"rag_context": None})
        assert features.has_rag_context is False

    def test_extract_features_rag_context_empty(self):
        ctrl = DummyController(name="test")
        features = ctrl.extract_features({"rag_context": ""})
        assert features.has_rag_context is False
