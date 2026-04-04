"""
Tests for src/training/train_bert_lora.py

Covers BERTLoRATrainer initialization, dataset preparation,
metrics computation, training, evaluation, model saving, and
the setup_logging helper. All heavy dependencies (torch, transformers,
datasets, BERTMetaController) are mocked.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: mock out the heavy imports that train_bert_lora pulls in
# ---------------------------------------------------------------------------


def _make_mock_controller(**kwargs):
    """Build a mock BERTMetaController."""
    ctrl = MagicMock(name="BERTMetaController")
    ctrl.device = "cpu"
    ctrl.tokenizer = MagicMock(name="tokenizer")
    ctrl.get_trainable_parameters.return_value = {
        "total_params": 1000,
        "trainable_params": 50,
        "trainable_percentage": 5.0,
    }
    ctrl.model = MagicMock(name="model")
    ctrl.model.train = MagicMock()
    ctrl.model.eval = MagicMock()
    ctrl.save_model = MagicMock()
    return ctrl


@pytest.fixture(autouse=True)
def _patch_heavy_deps(monkeypatch):
    """Patch heavy dependencies so tests run without torch/transformers/datasets."""
    # We need the module-level flags to be True inside train_bert_lora
    import src.training.train_bert_lora as mod

    monkeypatch.setattr(mod, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(mod, "_DATASETS_AVAILABLE", True)

    # Patch BERTMetaController constructor
    monkeypatch.setattr(mod, "BERTMetaController", lambda **kw: _make_mock_controller(**kw))


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetupLogging:
    """Tests for the setup_logging helper."""

    def test_returns_logger(self):
        from src.training.train_bert_lora import setup_logging

        logger = setup_logging(logging.DEBUG)
        assert isinstance(logger, logging.Logger)

    def test_default_level(self):
        from src.training.train_bert_lora import setup_logging

        logger = setup_logging()
        assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# BERTLoRATrainer.__init__
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBERTLoRATrainerInit:
    """Tests for BERTLoRATrainer construction."""

    def test_default_init(self):
        from src.training.train_bert_lora import BERTLoRATrainer

        trainer = BERTLoRATrainer()
        assert trainer.model_name == "prajjwal1/bert-mini"
        assert trainer.lora_r == 4
        assert trainer.lora_alpha == 16
        assert trainer.lora_dropout == 0.1
        assert trainer.lr == 1e-3
        assert trainer.batch_size == 32
        assert trainer.epochs == 10
        assert trainer.warmup_steps == 100
        assert trainer.seed == 42
        assert trainer.device == "cpu"
        assert trainer.tokenizer is not None
        assert trainer.controller is not None
        assert trainer._trainer is None

    def test_custom_params(self):
        from src.training.train_bert_lora import BERTLoRATrainer

        trainer = BERTLoRATrainer(
            model_name="test-model",
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.2,
            lr=5e-4,
            batch_size=16,
            epochs=5,
            warmup_steps=50,
            seed=123,
            device="cpu",
        )
        assert trainer.model_name == "test-model"
        assert trainer.lora_r == 8
        assert trainer.epochs == 5
        assert trainer.seed == 123

    def test_raises_without_transformers(self, monkeypatch):
        import src.training.train_bert_lora as mod

        monkeypatch.setattr(mod, "_TRANSFORMERS_AVAILABLE", False)
        with pytest.raises(ImportError, match="transformers"):
            mod.BERTLoRATrainer()

    def test_raises_without_datasets(self, monkeypatch):
        import src.training.train_bert_lora as mod

        monkeypatch.setattr(mod, "_DATASETS_AVAILABLE", False)
        with pytest.raises(ImportError, match="datasets"):
            mod.BERTLoRATrainer()


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrepareDataset:
    """Tests for BERTLoRATrainer.prepare_dataset."""

    def test_mismatched_lengths_raises(self):
        from src.training.train_bert_lora import BERTLoRATrainer

        trainer = BERTLoRATrainer()
        with pytest.raises(ValueError, match="same length"):
            trainer.prepare_dataset(["a", "b"], [0])

    @patch("src.training.train_bert_lora.Dataset")
    def test_prepare_dataset_calls_tokenizer(self, mock_dataset_cls):
        from src.training.train_bert_lora import BERTLoRATrainer

        # Set up mock dataset chain
        mock_ds = MagicMock(name="dataset")
        mock_dataset_cls.from_dict.return_value = mock_ds
        mock_tokenized = MagicMock(name="tokenized_dataset")
        mock_tokenized.__len__ = MagicMock(return_value=2)
        mock_ds.map.return_value = mock_tokenized

        trainer = BERTLoRATrainer()
        result = trainer.prepare_dataset(["text1", "text2"], [0, 1])

        mock_dataset_cls.from_dict.assert_called_once()
        mock_ds.map.assert_called_once()
        mock_tokenized.set_format.assert_called_once_with(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        assert result is mock_tokenized


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeMetrics:
    """Tests for BERTLoRATrainer.compute_metrics."""

    def test_accuracy_perfect(self):
        from src.training.train_bert_lora import BERTLoRATrainer

        trainer = BERTLoRATrainer()
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        labels = np.array([1, 0])
        eval_pred = MagicMock()
        eval_pred.predictions = predictions
        eval_pred.label_ids = labels

        metrics = trainer.compute_metrics(eval_pred)
        assert metrics["accuracy"] == 1.0

    def test_accuracy_partial(self):
        from src.training.train_bert_lora import BERTLoRATrainer

        trainer = BERTLoRATrainer()
        predictions = np.array([[0.9, 0.1], [0.8, 0.2]])
        labels = np.array([1, 0])  # first wrong, second right
        eval_pred = MagicMock()
        eval_pred.predictions = predictions
        eval_pred.label_ids = labels

        metrics = trainer.compute_metrics(eval_pred)
        assert metrics["accuracy"] == 0.5

    def test_tuple_predictions(self):
        from src.training.train_bert_lora import BERTLoRATrainer

        trainer = BERTLoRATrainer()
        preds = np.array([[0.1, 0.9]])
        eval_pred = MagicMock()
        eval_pred.predictions = (preds, "extra")
        eval_pred.label_ids = np.array([1])

        metrics = trainer.compute_metrics(eval_pred)
        assert metrics["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrain:
    """Tests for BERTLoRATrainer.train."""

    @patch("src.training.train_bert_lora.Dataset")
    @patch("src.training.train_bert_lora.TrainingArguments")
    @patch("src.training.train_bert_lora.Trainer")
    def test_train_returns_history(self, mock_trainer_cls, mock_training_args_cls, mock_dataset_cls, tmp_path):
        from src.training.train_bert_lora import BERTLoRATrainer

        # Setup mock dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=3)
        mock_dataset_cls.from_dict.return_value.map.return_value = mock_ds

        # Setup mock trainer
        mock_trainer_instance = MagicMock()
        train_result = MagicMock()
        train_result.training_loss = 0.25
        train_result.metrics = {"train_runtime": 10.0, "train_samples_per_second": 5.0}
        mock_trainer_instance.train.return_value = train_result
        mock_trainer_instance.evaluate.return_value = {"eval_accuracy": 0.9, "eval_loss": 0.1}
        mock_trainer_cls.return_value = mock_trainer_instance

        trainer = BERTLoRATrainer(epochs=2)
        history = trainer.train(
            train_texts=["a", "b", "c"],
            train_labels=[0, 1, 2],
            val_texts=["d"],
            val_labels=[0],
            output_dir=str(tmp_path / "output"),
        )

        assert history["train_loss"] == 0.25
        assert history["epochs"] == 2
        assert "eval_results" in history
        assert history["eval_results"]["eval_accuracy"] == 0.9

    @patch("src.training.train_bert_lora.Dataset")
    @patch("src.training.train_bert_lora.TrainingArguments")
    @patch("src.training.train_bert_lora.Trainer")
    def test_train_creates_output_dir(self, mock_trainer_cls, mock_ta, mock_ds_cls, tmp_path):
        from src.training.train_bert_lora import BERTLoRATrainer

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_ds_cls.from_dict.return_value.map.return_value = mock_ds

        mock_t = MagicMock()
        result = MagicMock()
        result.training_loss = 0.1
        result.metrics = {}
        mock_t.train.return_value = result
        mock_t.evaluate.return_value = {}
        mock_trainer_cls.return_value = mock_t

        output_dir = tmp_path / "nested" / "dir"
        trainer = BERTLoRATrainer()
        trainer.train(["a"], [0], ["b"], [1], str(output_dir))

        assert output_dir.exists()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluate:
    """Tests for BERTLoRATrainer.evaluate."""

    @patch("src.training.train_bert_lora.Dataset")
    def test_evaluate_returns_metrics(self, mock_dataset_cls):
        import torch

        from src.training.train_bert_lora import BERTLoRATrainer

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=2)
        mock_dataset_cls.from_dict.return_value.map.return_value = mock_ds

        trainer = BERTLoRATrainer()

        # Mock the tokenizer to return dict with tensors
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock the model forward pass
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.9, 0.0]])
        trainer.controller.model.return_value = mock_output
        trainer.controller.model.eval = MagicMock()

        results = trainer.evaluate(["test text 1", "test text 2"], [1, 1])

        assert "loss" in results
        assert "accuracy" in results
        assert "predictions" in results
        assert "probabilities" in results
        assert len(results["predictions"]) == 2
        assert len(results["probabilities"]) == 2
        assert 0.0 <= results["accuracy"] <= 1.0

    @patch("src.training.train_bert_lora.Dataset")
    def test_evaluate_correct_predictions(self, mock_dataset_cls):
        import torch

        from src.training.train_bert_lora import BERTLoRATrainer

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_dataset_cls.from_dict.return_value.map.return_value = mock_ds

        trainer = BERTLoRATrainer()
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        mock_output = MagicMock()
        # Model predicts class 0 strongly
        mock_output.logits = torch.tensor([[10.0, -10.0, -10.0]])
        trainer.controller.model.return_value = mock_output
        trainer.controller.model.eval = MagicMock()

        results = trainer.evaluate(["text"], [0])
        assert results["accuracy"] == 1.0
        assert results["predictions"] == [0]


# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveModel:
    """Tests for BERTLoRATrainer.save_model."""

    def test_save_delegates_to_controller(self):
        from src.training.train_bert_lora import BERTLoRATrainer

        trainer = BERTLoRATrainer()
        trainer.save_model("/tmp/test_save")
        trainer.controller.save_model.assert_called_once_with("/tmp/test_save")


# ---------------------------------------------------------------------------
# main (argument parsing smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMainFunction:
    """Smoke tests for the main() entry point argument parsing."""

    @patch("src.training.train_bert_lora.MetaControllerDataGenerator")
    @patch("src.training.train_bert_lora.BERTLoRATrainer")
    def test_main_runs_with_synthetic_data(self, mock_trainer_cls, mock_data_gen_cls, tmp_path):
        from src.training.train_bert_lora import main

        # Mock data generator
        mock_gen = MagicMock()
        mock_gen.generate_dataset.return_value = (["feat1", "feat2"], ["hrm", "trm"])
        mock_gen.to_text_dataset.return_value = (["text1", "text2"], [0, 1])
        mock_gen.split_dataset.return_value = {
            "X_train": ["t1"],
            "y_train": [0],
            "X_val": ["v1"],
            "y_val": [1],
            "X_test": ["e1"],
            "y_test": [0],
        }
        mock_data_gen_cls.return_value = mock_gen

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "train_loss": 0.1,
            "train_runtime": 1.0,
            "train_samples_per_second": 10.0,
            "epochs": 1,
            "final_metrics": {},
            "eval_results": {"eval_accuracy": 0.9},
        }
        mock_trainer.evaluate.return_value = {
            "loss": 0.1,
            "accuracy": 0.9,
            "predictions": [0],
            "probabilities": [[0.9, 0.1]],
        }
        mock_trainer.controller = MagicMock()
        mock_trainer.controller.get_trainable_parameters.return_value = {
            "total_params": 100,
            "trainable_params": 10,
            "trainable_percentage": 10.0,
        }
        mock_trainer_cls.return_value = mock_trainer

        output_dir = str(tmp_path / "output")

        # Patch sys.argv
        with patch(
            "sys.argv",
            ["train_bert_lora.py", "--output_dir", output_dir, "--epochs", "1", "--num_samples", "6"],
        ):
            main()

        mock_trainer.train.assert_called_once()
        mock_trainer.evaluate.assert_called_once()
        mock_trainer.save_model.assert_called_once()

    @patch("src.training.train_bert_lora.MetaControllerDataGenerator")
    @patch("src.training.train_bert_lora.BERTLoRATrainer")
    def test_main_balanced_flag(self, mock_trainer_cls, mock_data_gen_cls, tmp_path):
        from src.training.train_bert_lora import main

        mock_gen = MagicMock()
        mock_gen.generate_balanced_dataset.return_value = (["f1", "f2", "f3"], ["hrm", "trm", "mcts"])
        mock_gen.to_text_dataset.return_value = (["t1", "t2", "t3"], [0, 1, 2])
        mock_gen.split_dataset.return_value = {
            "X_train": ["t1"],
            "y_train": [0],
            "X_val": ["v1"],
            "y_val": [1],
            "X_test": ["e1"],
            "y_test": [0],
        }
        mock_data_gen_cls.return_value = mock_gen

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "train_loss": 0.1,
            "epochs": 1,
            "final_metrics": {},
            "eval_results": {"eval_accuracy": 0.8},
            "train_runtime": 1.0,
            "train_samples_per_second": 5.0,
        }
        mock_trainer.evaluate.return_value = {"loss": 0.1, "accuracy": 0.8, "predictions": [], "probabilities": []}
        mock_trainer.controller = MagicMock()
        mock_trainer.controller.get_trainable_parameters.return_value = {
            "total_params": 100,
            "trainable_params": 10,
            "trainable_percentage": 10.0,
        }
        mock_trainer_cls.return_value = mock_trainer

        with patch(
            "sys.argv",
            ["prog", "--balanced", "--output_dir", str(tmp_path / "out"), "--epochs", "1", "--num_samples", "6"],
        ):
            main()

        mock_gen.generate_balanced_dataset.assert_called_once()

    @patch("src.training.train_bert_lora.MetaControllerDataGenerator")
    @patch("src.training.train_bert_lora.BERTLoRATrainer")
    def test_main_load_from_file(self, mock_trainer_cls, mock_data_gen_cls, tmp_path):
        from src.training.train_bert_lora import main

        mock_gen = MagicMock()
        mock_gen.load_dataset.return_value = (["f1"], ["hrm"])
        mock_gen.to_text_dataset.return_value = (["t1"], [0])
        mock_gen.split_dataset.return_value = {
            "X_train": ["t1"],
            "y_train": [0],
            "X_val": ["v1"],
            "y_val": [1],
            "X_test": ["e1"],
            "y_test": [0],
        }
        mock_data_gen_cls.return_value = mock_gen

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "train_loss": 0.1,
            "epochs": 1,
            "final_metrics": {},
            "eval_results": {"eval_accuracy": 0.8},
            "train_runtime": 1.0,
            "train_samples_per_second": 5.0,
        }
        mock_trainer.evaluate.return_value = {"loss": 0.1, "accuracy": 0.8, "predictions": [], "probabilities": []}
        mock_trainer.controller = MagicMock()
        mock_trainer.controller.get_trainable_parameters.return_value = {
            "total_params": 100,
            "trainable_params": 10,
            "trainable_percentage": 10.0,
        }
        mock_trainer_cls.return_value = mock_trainer

        data_file = tmp_path / "data.json"
        data_file.write_text("{}")

        output_dir = str(tmp_path / "out")
        with patch(
            "sys.argv",
            ["prog", "--data_path", str(data_file), "--output_dir", output_dir, "--epochs", "1"],
        ):
            # Ensure the output directory exists before main() tries to write results
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            main()

        mock_gen.load_dataset.assert_called_once_with(str(data_file))
