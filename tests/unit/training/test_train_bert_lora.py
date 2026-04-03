"""
Tests for BERT LoRA training script.

Tests setup_logging and BERTLoRATrainer initialization behavior
when dependencies are missing.
"""

import logging
from unittest.mock import patch

import pytest

from src.training.train_bert_lora import setup_logging


@pytest.mark.unit
class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_custom_log_level(self):
        logger = setup_logging(log_level=logging.DEBUG)
        assert isinstance(logger, logging.Logger)


@pytest.mark.unit
class TestBERTLoRATrainer:
    """Tests for BERTLoRATrainer."""

    def test_import_error_when_transformers_missing(self):
        """BERTLoRATrainer should raise ImportError when transformers not available."""
        with patch("src.training.train_bert_lora._TRANSFORMERS_AVAILABLE", False):
            from src.training.train_bert_lora import BERTLoRATrainer

            with pytest.raises(ImportError, match="transformers"):
                BERTLoRATrainer()

    def test_import_error_when_datasets_missing(self):
        """BERTLoRATrainer should raise ImportError when datasets not available."""
        with patch("src.training.train_bert_lora._TRANSFORMERS_AVAILABLE", True), patch(
            "src.training.train_bert_lora._DATASETS_AVAILABLE", False
        ):
            from src.training.train_bert_lora import BERTLoRATrainer

            with pytest.raises(ImportError, match="datasets"):
                BERTLoRATrainer()
