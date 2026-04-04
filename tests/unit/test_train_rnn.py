"""Unit tests for src/training/train_rnn.py."""

import pytest
import torch

from src.training.train_rnn import RNNTrainer


@pytest.mark.unit
class TestRNNTrainerClassAttributes:
    """Tests for RNNTrainer class-level attributes."""

    def test_agent_names(self):
        assert RNNTrainer.AGENT_NAMES == ["hrm", "trm", "mcts"]

    def test_label_to_index(self):
        assert RNNTrainer.LABEL_TO_INDEX == {"hrm": 0, "trm": 1, "mcts": 2}

    def test_index_to_label(self):
        assert RNNTrainer.INDEX_TO_LABEL == {0: "hrm", 1: "trm", 2: "mcts"}


@pytest.mark.unit
class TestRNNTrainerInit:
    """Tests for RNNTrainer initialization."""

    def test_default_init(self):
        """Test default initialization stores hyperparameters."""
        trainer = RNNTrainer(device="cpu")
        assert trainer.hidden_dim == 64
        assert trainer.num_layers == 1
        assert trainer.dropout == 0.1
        assert trainer.lr == 1e-3
        assert trainer.batch_size == 32
        assert trainer.epochs == 10
        assert trainer.early_stopping_patience == 3
        assert trainer.seed == 42
        assert trainer.device == torch.device("cpu")

    def test_custom_init(self):
        """Test custom hyperparameters are stored."""
        trainer = RNNTrainer(
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            lr=0.01,
            batch_size=16,
            epochs=5,
            early_stopping_patience=5,
            seed=123,
            device="cpu",
        )
        assert trainer.hidden_dim == 128
        assert trainer.num_layers == 2
        assert trainer.dropout == 0.2
        assert trainer.lr == 0.01
        assert trainer.batch_size == 16
        assert trainer.epochs == 5
        assert trainer.early_stopping_patience == 5
        assert trainer.seed == 123

    def test_model_is_created(self):
        """Test that model is created during init."""
        trainer = RNNTrainer(device="cpu")
        assert trainer.model is not None

    def test_optimizer_is_adam(self):
        """Test that optimizer is Adam."""
        trainer = RNNTrainer(device="cpu")
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_criterion_is_cross_entropy(self):
        """Test that criterion is CrossEntropyLoss."""
        trainer = RNNTrainer(device="cpu")
        assert isinstance(trainer.criterion, torch.nn.CrossEntropyLoss)

    def test_logger_is_setup(self):
        """Test that logger is set up."""
        trainer = RNNTrainer(device="cpu")
        assert trainer.logger is not None
        assert trainer.logger.name == "RNNTrainer"

    def test_braintrust_tracker_none_by_default(self):
        """Test that braintrust_tracker is None by default."""
        trainer = RNNTrainer(device="cpu")
        assert trainer.braintrust_tracker is None


@pytest.mark.unit
class TestRNNTrainerCreateDataloader:
    """Tests for RNNTrainer.create_dataloader method."""

    def test_creates_dataloader(self):
        """Test that create_dataloader returns a DataLoader."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        loader = trainer.create_dataloader(X, y)
        assert len(loader) == 7  # ceil(100/16) = 7

    def test_custom_batch_size(self):
        """Test create_dataloader with custom batch_size override."""
        trainer = RNNTrainer(device="cpu", batch_size=32)
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        loader = trainer.create_dataloader(X, y, batch_size=50)
        assert len(loader) == 2  # ceil(100/50) = 2

    def test_no_shuffle(self):
        """Test create_dataloader with shuffle=False."""
        trainer = RNNTrainer(device="cpu")
        X = torch.randn(10, 10)
        y = torch.randint(0, 3, (10,))
        loader = trainer.create_dataloader(X, y, shuffle=False)
        # With shuffle=False, first batch should be deterministic
        first_batch_X, first_batch_y = next(iter(loader))
        assert first_batch_X.shape[1] == 10


@pytest.mark.unit
class TestRNNTrainerTrainEpoch:
    """Tests for RNNTrainer.train_epoch method."""

    def test_train_epoch_returns_float(self):
        """Test that train_epoch returns a float loss."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(50, 10)
        y = torch.randint(0, 3, (50,))
        loader = trainer.create_dataloader(X, y)
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_epoch_updates_model(self):
        """Test that train_epoch actually updates model weights."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(50, 10)
        y = torch.randint(0, 3, (50,))
        loader = trainer.create_dataloader(X, y)

        # Get initial weights
        initial_weights = {n: p.clone() for n, p in trainer.model.named_parameters()}
        trainer.train_epoch(loader)

        # At least some weights should change
        changed = False
        for n, p in trainer.model.named_parameters():
            if not torch.equal(p, initial_weights[n]):
                changed = True
                break
        assert changed, "Model weights should be updated after training"


@pytest.mark.unit
class TestRNNTrainerValidate:
    """Tests for RNNTrainer.validate method."""

    def test_validate_returns_tuple(self):
        """Test that validate returns (loss, accuracy) tuple."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(50, 10)
        y = torch.randint(0, 3, (50,))
        loader = trainer.create_dataloader(X, y, shuffle=False)
        loss, acc = trainer.validate(loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0


@pytest.mark.unit
class TestRNNTrainerTrain:
    """Tests for RNNTrainer.train method."""

    def test_train_returns_history_dict(self):
        """Test that train returns a history dict with expected keys."""
        trainer = RNNTrainer(device="cpu", batch_size=16, epochs=2)
        X_train = torch.randn(60, 10)
        y_train = torch.randint(0, 3, (60,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 3, (20,))

        history = trainer.train((X_train, y_train), (X_val, y_val))

        assert "train_losses" in history
        assert "val_losses" in history
        assert "val_accuracies" in history
        assert "best_epoch" in history
        assert "best_val_loss" in history
        assert "best_val_accuracy" in history
        assert "stopped_early" in history
        assert "total_epochs" in history
        assert len(history["train_losses"]) <= 2

    def test_train_early_stopping(self):
        """Test early stopping when validation loss does not improve."""
        # Use a very small patience to trigger early stopping quickly
        trainer = RNNTrainer(device="cpu", batch_size=64, epochs=100, early_stopping_patience=1)
        # Use random data so no real learning occurs
        X_train = torch.randn(80, 10)
        y_train = torch.randint(0, 3, (80,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 3, (20,))

        history = trainer.train((X_train, y_train), (X_val, y_val))

        # Should stop before 100 epochs (at most a few epochs)
        assert history["total_epochs"] < 100

    def test_train_saves_model(self, tmp_path):
        """Test that train saves model checkpoint when save_path is provided."""
        trainer = RNNTrainer(device="cpu", batch_size=16, epochs=2)
        X_train = torch.randn(60, 10)
        y_train = torch.randint(0, 3, (60,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 3, (20,))

        save_path = str(tmp_path / "model.pt")
        trainer.train((X_train, y_train), (X_val, y_val), save_path=save_path)

        assert (tmp_path / "model.pt").exists()


@pytest.mark.unit
class TestRNNTrainerEvaluate:
    """Tests for RNNTrainer.evaluate method."""

    def test_evaluate_returns_expected_keys(self):
        """Test that evaluate returns dict with expected keys."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(50, 10)
        y = torch.randint(0, 3, (50,))
        loader = trainer.create_dataloader(X, y, shuffle=False)
        results = trainer.evaluate(loader)

        assert "loss" in results
        assert "accuracy" in results
        assert "per_class_metrics" in results
        assert "confusion_matrix" in results
        assert "total_samples" in results

    def test_evaluate_per_class_metrics(self):
        """Test that per_class_metrics has entries for all agents."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(90, 10)
        # Ensure all 3 classes are present
        y = torch.tensor([0] * 30 + [1] * 30 + [2] * 30)
        loader = trainer.create_dataloader(X, y, shuffle=False)
        results = trainer.evaluate(loader)

        for agent in ["hrm", "trm", "mcts"]:
            assert agent in results["per_class_metrics"]
            metrics = results["per_class_metrics"][agent]
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "support" in metrics
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0

    def test_evaluate_confusion_matrix_shape(self):
        """Test that confusion matrix is 3x3."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(30, 10)
        y = torch.randint(0, 3, (30,))
        loader = trainer.create_dataloader(X, y, shuffle=False)
        results = trainer.evaluate(loader)

        cm = results["confusion_matrix"]
        assert len(cm) == 3
        for row in cm:
            assert len(row) == 3

    def test_evaluate_total_samples(self):
        """Test that total_samples matches input size."""
        trainer = RNNTrainer(device="cpu", batch_size=16)
        X = torch.randn(42, 10)
        y = torch.randint(0, 3, (42,))
        loader = trainer.create_dataloader(X, y, shuffle=False)
        results = trainer.evaluate(loader)
        assert results["total_samples"] == 42
