"""
Unit tests for src/agents/meta_controller/bert_controller.py - extended coverage.

Covers missed lines: predict cache hit path, load_model (LoRA and base),
save_model (LoRA and base), clear_cache, get_cache_info, get_trainable_parameters,
device auto-detection (cuda/mps), and LoRA initialization.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.agents.meta_controller.base import MetaControllerFeatures

pytestmark = pytest.mark.unit


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


def _make_mock_model():
    """Create a mock model that returns proper logits."""
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.1, 0.8, 0.1]])
    mock_model.return_value = mock_output
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.state_dict = MagicMock(return_value={"w": torch.tensor([1.0])})
    mock_model.parameters = MagicMock(
        return_value=[
            torch.nn.Parameter(torch.randn(10, 10)),
            torch.nn.Parameter(torch.randn(5, 5), requires_grad=False),
        ]
    )
    mock_model.save_pretrained = MagicMock()
    mock_model.get_base_model = MagicMock(return_value=mock_model)
    mock_model.load_state_dict = MagicMock()
    return mock_model


def _make_mock_tokenizer():
    """Create a mock tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 20)),
        "attention_mask": torch.ones(1, 20, dtype=torch.long),
    }
    return mock_tokenizer


def _create_controller(use_lora=False, device="cpu"):
    """Create a BERTMetaController with fully mocked dependencies."""
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()

    patches = {
        "_TRANSFORMERS_AVAILABLE": True,
        "_PEFT_AVAILABLE": use_lora,
        "AutoTokenizer": MagicMock(),
        "AutoModelForSequenceClassification": MagicMock(),
    }

    if use_lora:
        patches["LoraConfig"] = MagicMock()
        patches["TaskType"] = MagicMock()
        patches["get_peft_model"] = MagicMock(return_value=mock_model)

    with patch.multiple("src.agents.meta_controller.bert_controller", **patches):
        # Can't use patch.multiple return for class-level patches; re-import
        import src.agents.meta_controller.bert_controller as mod

        mod.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mod.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model

        ctrl = mod.BERTMetaController(device=device, use_lora=use_lora)
        return ctrl, mock_model, mock_tokenizer


class TestBERTMetaControllerDeviceDetection:
    """Tests for device auto-detection."""

    def test_explicit_cpu(self):
        ctrl, _, _ = _create_controller(device="cpu")
        assert ctrl.device == torch.device("cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_detect_cuda(self, mock_cuda):
        ctrl, _, _ = _create_controller(device=None)
        assert ctrl.device == torch.device("cuda")

    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_detect_cpu_fallback(self, mock_cuda):
        # Also need to ensure MPS is not available
        with patch("torch.backends.mps.is_available", return_value=False, create=True):
            ctrl, _, _ = _create_controller(device=None)
            assert ctrl.device == torch.device("cpu")


class TestBERTMetaControllerLoRA:
    """Tests for LoRA initialization path."""

    def test_init_with_lora(self):
        ctrl, mock_model, _ = _create_controller(use_lora=True)
        assert ctrl.use_lora is True

    def test_init_without_lora(self):
        ctrl, _, _ = _create_controller(use_lora=False)
        assert ctrl.use_lora is False


class TestBERTMetaControllerPredict:
    """Tests for predict method including caching."""

    def test_predict_returns_valid(self):
        ctrl, _, _ = _create_controller(use_lora=False)
        features = _make_features()
        pred = ctrl.predict(features)

        assert pred.agent in ["hrm", "trm", "mcts"]
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.probabilities) == 3

    def test_predict_cache_hit(self):
        """Second call with same features should use cache."""
        ctrl, _, mock_tokenizer = _create_controller(use_lora=False)
        features = _make_features()

        # First predict fills cache
        ctrl.predict(features)
        call_count_1 = mock_tokenizer.call_count

        # Second predict should use cache (tokenizer not called again)
        ctrl.predict(features)
        call_count_2 = mock_tokenizer.call_count

        # Tokenizer should only have been called once
        assert call_count_2 == call_count_1

    def test_predict_different_features(self):
        """Different features should produce predictions."""
        ctrl, _, _ = _create_controller(use_lora=False)

        pred1 = ctrl.predict(_make_features(hrm_confidence=0.9))
        pred2 = ctrl.predict(_make_features(hrm_confidence=0.1))

        assert pred1.agent in ["hrm", "trm", "mcts"]
        assert pred2.agent in ["hrm", "trm", "mcts"]


class TestBERTMetaControllerCache:
    """Tests for cache management methods."""

    def test_clear_cache(self):
        ctrl, _, _ = _create_controller(use_lora=False)
        features = _make_features()
        ctrl.predict(features)

        assert len(ctrl._tokenization_cache) > 0
        ctrl.clear_cache()
        assert len(ctrl._tokenization_cache) == 0

    def test_get_cache_info_empty(self):
        ctrl, _, _ = _create_controller(use_lora=False)
        info = ctrl.get_cache_info()
        assert info["cache_size"] == 0
        assert info["cache_keys"] == []

    def test_get_cache_info_after_predict(self):
        ctrl, _, _ = _create_controller(use_lora=False)
        ctrl.predict(_make_features())

        info = ctrl.get_cache_info()
        assert info["cache_size"] >= 1
        assert len(info["cache_keys"]) >= 1

    def test_get_cache_info_truncates_long_keys(self):
        ctrl, _, _ = _create_controller(use_lora=False)
        # Manually insert a long key
        ctrl._tokenization_cache["a" * 100] = "some_value"

        info = ctrl.get_cache_info()
        for key in info["cache_keys"]:
            assert len(key) <= 53  # 50 + "..."


class TestBERTMetaControllerSaveLoad:
    """Tests for save_model and load_model."""

    def test_save_model_lora(self):
        ctrl, mock_model, _ = _create_controller(use_lora=True)
        ctrl.save_model("/tmp/lora_adapter")
        mock_model.save_pretrained.assert_called_once_with("/tmp/lora_adapter")

    def test_save_model_base(self):
        ctrl, mock_model, _ = _create_controller(use_lora=False)
        with patch("torch.save") as mock_save:
            ctrl.save_model("/tmp/model.pt")
            mock_save.assert_called_once()

    def test_load_model_base(self):
        ctrl, mock_model, _ = _create_controller(use_lora=False)
        with patch("torch.load", return_value={"w": torch.tensor([1.0])}):
            ctrl.load_model("/tmp/model.pt")
            mock_model.load_state_dict.assert_called_once()
            mock_model.eval.assert_called()

    def test_load_model_lora(self):
        ctrl, mock_model, _ = _create_controller(use_lora=True)
        with patch("src.agents.meta_controller.bert_controller.PeftModel", create=True) as mock_peft:
            mock_loaded = MagicMock()
            mock_loaded.to.return_value = mock_loaded
            mock_loaded.eval.return_value = mock_loaded
            mock_peft.from_pretrained.return_value = mock_loaded

            # Need to patch the peft import inside load_model
            with patch.dict("sys.modules", {"peft": MagicMock(PeftModel=mock_peft)}):
                ctrl.load_model("/tmp/lora_adapter")
                mock_loaded.eval.assert_called_once()


class TestBERTMetaControllerTrainableParams:
    """Tests for get_trainable_parameters."""

    def test_get_trainable_parameters(self):
        ctrl, _, _ = _create_controller(use_lora=False)

        # Override parameters to have known values
        param1 = torch.nn.Parameter(torch.randn(10, 10))  # 100 params, trainable
        param2 = torch.nn.Parameter(torch.randn(5, 5))  # 25 params
        param2.requires_grad = False

        ctrl.model.parameters = MagicMock(return_value=[param1, param2])

        result = ctrl.get_trainable_parameters()
        assert result["total_params"] == 125
        assert result["trainable_params"] == 100
        assert result["trainable_percentage"] == 80.0

    def test_get_trainable_parameters_zero_total(self):
        ctrl, _, _ = _create_controller(use_lora=False)
        ctrl.model.parameters = MagicMock(return_value=[])

        result = ctrl.get_trainable_parameters()
        assert result["total_params"] == 0
        assert result["trainable_params"] == 0
        assert result["trainable_percentage"] == 0.0
