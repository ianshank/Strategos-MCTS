"""
Tests for BERT meta-controller module.

Tests BERTMetaController initialization, prediction, model save/load,
and ImportError handling when dependencies are missing.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.agents.meta_controller.base import MetaControllerFeatures


def _make_features(**overrides):
    defaults = {
        "hrm_confidence": 0.5, "trm_confidence": 0.3, "mcts_value": 0.2,
        "consensus_score": 0.7, "last_agent": "none", "iteration": 0,
        "query_length": 50, "has_rag_context": False,
        "rag_relevance_score": 0.0, "is_technical_query": False,
    }
    defaults.update(overrides)
    return MetaControllerFeatures(**defaults)


@pytest.mark.unit
class TestBERTMetaControllerImportErrors:
    """Tests for import error handling."""

    def test_raises_when_transformers_missing(self):
        with patch("src.agents.meta_controller.bert_controller._TRANSFORMERS_AVAILABLE", False):
            from src.agents.meta_controller.bert_controller import BERTMetaController
            with pytest.raises(ImportError, match="transformers"):
                BERTMetaController(device="cpu")

    def test_raises_when_peft_missing(self):
        with patch("src.agents.meta_controller.bert_controller._TRANSFORMERS_AVAILABLE", True), \
             patch("src.agents.meta_controller.bert_controller._PEFT_AVAILABLE", False), \
             patch("src.agents.meta_controller.bert_controller.AutoTokenizer"), \
             patch("src.agents.meta_controller.bert_controller.AutoModelForSequenceClassification"):
            from src.agents.meta_controller.bert_controller import BERTMetaController
            with pytest.raises(ImportError, match="peft"):
                BERTMetaController(device="cpu", use_lora=True)


@pytest.mark.unit
class TestBERTMetaControllerWithMocks:
    """Tests with mocked transformers/peft."""

    def _make_controller(self):
        """Create a BERTMetaController with mocked dependencies."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 20)),
            "attention_mask": torch.ones(1, 20, dtype=torch.long),
        }

        mock_model = MagicMock()
        # Simulate model output with logits
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.8, 0.1]])
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.state_dict = MagicMock(return_value={})

        with patch("src.agents.meta_controller.bert_controller._TRANSFORMERS_AVAILABLE", True), \
             patch("src.agents.meta_controller.bert_controller._PEFT_AVAILABLE", False), \
             patch("src.agents.meta_controller.bert_controller.AutoTokenizer") as mock_tok_cls, \
             patch("src.agents.meta_controller.bert_controller.AutoModelForSequenceClassification") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            mock_model_cls.from_pretrained.return_value = mock_model

            from src.agents.meta_controller.bert_controller import BERTMetaController
            ctrl = BERTMetaController(device="cpu", use_lora=False)
            return ctrl

    def test_init(self):
        ctrl = self._make_controller()
        assert ctrl.name == "BERTMetaController"
        assert ctrl.device == torch.device("cpu")
        assert ctrl.use_lora is False

    def test_predict(self):
        ctrl = self._make_controller()
        features = _make_features()
        pred = ctrl.predict(features)
        assert pred.agent in ["hrm", "trm", "mcts"]
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.probabilities) == 3

    def test_constants(self):
        from src.agents.meta_controller.bert_controller import BERTMetaController
        assert BERTMetaController.DEFAULT_MODEL_NAME == "prajjwal1/bert-mini"
        assert BERTMetaController.NUM_LABELS == 3
