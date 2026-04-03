"""
Tests for RNN meta-controller training script.

Tests RNNTrainer initialization, data loading, training epoch,
validation, early stopping, and model saving.
"""

from unittest.mock import MagicMock

import pytest
import torch

from src.training.train_rnn import RNNTrainer


@pytest.mark.unit
class TestRNNTrainer:
    """Tests for RNNTrainer class."""

    def test_init_defaults(self):
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

    def test_init_custom(self):
        trainer = RNNTrainer(
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            lr=5e-4,
            batch_size=16,
            epochs=20,
            early_stopping_patience=5,
            seed=99,
            device="cpu",
        )
        assert trainer.hidden_dim == 128
        assert trainer.num_layers == 2
        assert trainer.batch_size == 16
        assert trainer.epochs == 20

    def test_model_initialized(self):
        trainer = RNNTrainer(device="cpu", hidden_dim=32)
        assert trainer.model is not None
        # Model should accept 10-dim input
        x = torch.randn(4, 10)
        output = trainer.model(x)
        assert output.shape == (4, 3)  # 3 agents

    def test_create_dataloader(self):
        trainer = RNNTrainer(device="cpu", batch_size=8)
        X = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        loader = trainer.create_dataloader(X, y)
        assert len(loader) == 3  # 20 / 8 = 2.5 -> 3 batches

    def test_create_dataloader_custom_batch_size(self):
        trainer = RNNTrainer(device="cpu")
        X = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        loader = trainer.create_dataloader(X, y, batch_size=5)
        assert len(loader) == 4

    def test_create_dataloader_no_shuffle(self):
        trainer = RNNTrainer(device="cpu")
        X = torch.randn(10, 10)
        y = torch.randint(0, 3, (10,))
        loader = trainer.create_dataloader(X, y, shuffle=False)
        assert len(loader) > 0

    def test_train_epoch(self):
        trainer = RNNTrainer(device="cpu", hidden_dim=16, batch_size=8)
        X = torch.randn(24, 10)
        y = torch.randint(0, 3, (24,))
        loader = trainer.create_dataloader(X, y)
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert loss > 0

    def test_validate(self):
        trainer = RNNTrainer(device="cpu", hidden_dim=16, batch_size=8)
        X = torch.randn(24, 10)
        y = torch.randint(0, 3, (24,))
        loader = trainer.create_dataloader(X, y, shuffle=False)
        loss, accuracy = trainer.validate(loader)
        assert isinstance(loss, float)
        assert loss > 0
        assert 0.0 <= accuracy <= 1.0

    def test_train_returns_history_dict(self):
        trainer = RNNTrainer(
            device="cpu",
            hidden_dim=16,
            batch_size=16,
            epochs=3,
            early_stopping_patience=2,
        )
        X_train = torch.randn(48, 10)
        y_train = torch.randint(0, 3, (48,))
        X_val = torch.randn(16, 10)
        y_val = torch.randint(0, 3, (16,))

        history = trainer.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
        )
        assert isinstance(history, dict)
        assert "train_losses" in history
        assert "val_losses" in history
        assert "val_accuracies" in history
        assert "total_epochs" in history
        assert history["total_epochs"] <= 3

    def test_train_with_save_path(self, tmp_path):
        trainer = RNNTrainer(
            device="cpu",
            hidden_dim=16,
            batch_size=16,
            epochs=2,
        )
        X_train = torch.randn(32, 10)
        y_train = torch.randint(0, 3, (32,))
        X_val = torch.randn(16, 10)
        y_val = torch.randint(0, 3, (16,))

        save_path = str(tmp_path / "best_model.pt")
        history = trainer.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            save_path=save_path,
        )
        assert isinstance(history, dict)

    def test_agent_names(self):
        assert RNNTrainer.AGENT_NAMES == ["hrm", "trm", "mcts"]
        assert RNNTrainer.LABEL_TO_INDEX == {"hrm": 0, "trm": 1, "mcts": 2}
        assert RNNTrainer.INDEX_TO_LABEL == {0: "hrm", 1: "trm", 2: "mcts"}

    def test_model_state_dict_saveable(self, tmp_path):
        trainer = RNNTrainer(device="cpu", hidden_dim=16)
        save_path = tmp_path / "model.pt"
        torch.save(trainer.model.state_dict(), save_path)
        assert save_path.exists()

    def test_braintrust_tracker_integration(self):
        mock_tracker = MagicMock()
        mock_tracker.is_available = True
        trainer = RNNTrainer(device="cpu", hidden_dim=16, braintrust_tracker=mock_tracker)
        assert trainer.braintrust_tracker is mock_tracker
        mock_tracker.log_hyperparameters.assert_called_once()

    def test_braintrust_tracker_not_available(self):
        mock_tracker = MagicMock()
        mock_tracker.is_available = False
        trainer = RNNTrainer(device="cpu", hidden_dim=16, braintrust_tracker=mock_tracker)
        assert trainer.braintrust_tracker is mock_tracker
