"""
Tests for neural_trainer module.

Tests TrainingConfig, TrainingMetrics, PolicyDataset, ValueDataset,
NeuralTrainer, and convenience functions.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.neural_trainer import (
    NeuralTrainer,
    PolicyDataset,
    TrainingConfig,
    TrainingMetrics,
    ValueDataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model(input_dim: int = 4, output_dim: int = 3) -> nn.Module:
    """Create a small feedforward model for testing."""
    return nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, output_dim))


def _make_simple_loss() -> nn.Module:
    """Create a simple loss module that returns (loss, metrics_dict)."""

    class SimpleLoss(nn.Module):
        def forward(self, *args, **kwargs):
            # Just compute MSE between first two positional args
            pred = args[0] if len(args) > 0 else kwargs.get("predictions", torch.tensor(0.0))
            target = args[1] if len(args) > 1 else kwargs.get("targets", torch.tensor(0.0))
            loss = nn.functional.mse_loss(pred, target)
            return loss, {"mse": loss.item()}

    return SimpleLoss()


def _make_trainer_with_mock_forward(config, model=None, loss_fn=None):
    """Create a NeuralTrainer whose _forward_batch is patched to work with generic models."""
    model = model or _make_simple_model()
    loss_fn = loss_fn or _make_simple_loss()
    trainer = NeuralTrainer(model, loss_fn, config, model_name="test_model")

    # Patch _forward_batch to avoid PolicyNetwork/ValueNetwork isinstance checks

    def generic_forward(batch):
        states = batch[0]
        targets = batch[1]
        output = trainer.model(states)
        loss, metrics = trainer.loss_fn(output, targets)
        return loss, metrics

    trainer._forward_batch = generic_forward
    return trainer


# ---------------------------------------------------------------------------
# TrainingConfig tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.0001
        assert config.batch_size == 64
        assert config.num_epochs == 100
        assert config.gradient_clip == 1.0
        assert config.scheduler_type == "cosine"
        assert config.early_stopping_patience == 10
        assert config.min_delta == 0.0001
        assert config.save_every == 5
        assert config.keep_best_only is True
        assert config.log_every == 10
        assert config.use_wandb is False
        assert config.wandb_project is None

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=128,
            num_epochs=50,
            scheduler_type="step",
        )
        assert config.learning_rate == 0.01
        assert config.batch_size == 128
        assert config.num_epochs == 50
        assert config.scheduler_type == "step"

    def test_scheduler_params_default_empty(self):
        """Test scheduler_params defaults to empty dict."""
        config = TrainingConfig()
        assert config.scheduler_params == {}
        # Ensure each instance gets its own dict
        config2 = TrainingConfig()
        config.scheduler_params["key"] = "value"
        assert "key" not in config2.scheduler_params

    def test_device_field(self):
        """Test device is set based on CUDA availability."""
        config = TrainingConfig()
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected


# ---------------------------------------------------------------------------
# TrainingMetrics tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_required_fields(self):
        """Test creating metrics with required fields."""
        metrics = TrainingMetrics(epoch=1, train_loss=0.5)
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.val_loss is None
        assert metrics.learning_rate == 0.0
        assert metrics.additional_metrics == {}

    def test_all_fields(self):
        """Test creating metrics with all fields."""
        metrics = TrainingMetrics(
            epoch=5,
            train_loss=0.3,
            val_loss=0.4,
            learning_rate=0.001,
            additional_metrics={"accuracy": 0.95},
        )
        assert metrics.epoch == 5
        assert metrics.val_loss == 0.4
        assert metrics.additional_metrics["accuracy"] == 0.95


# ---------------------------------------------------------------------------
# PolicyDataset tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolicyDataset:
    """Tests for PolicyDataset."""

    def test_len(self):
        """Test dataset length."""
        states = torch.randn(10, 4)
        actions = torch.randint(0, 3, (10,))
        ds = PolicyDataset(states, actions)
        assert len(ds) == 10

    def test_getitem_without_values(self):
        """Test indexing without values returns (state, action)."""
        states = torch.randn(5, 4)
        actions = torch.randint(0, 3, (5,))
        ds = PolicyDataset(states, actions)
        item = ds[0]
        assert len(item) == 2
        assert torch.equal(item[0], states[0])
        assert torch.equal(item[1], actions[0])

    def test_getitem_with_values(self):
        """Test indexing with values returns (state, action, value)."""
        states = torch.randn(5, 4)
        actions = torch.randint(0, 3, (5,))
        values = torch.randn(5)
        ds = PolicyDataset(states, actions, values)
        item = ds[0]
        assert len(item) == 3
        assert torch.equal(item[2], values[0])


# ---------------------------------------------------------------------------
# ValueDataset tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValueDataset:
    """Tests for ValueDataset."""

    def test_len(self):
        """Test dataset length."""
        states = torch.randn(8, 4)
        values = torch.randn(8)
        ds = ValueDataset(states, values)
        assert len(ds) == 8

    def test_getitem(self):
        """Test indexing returns (state, value)."""
        states = torch.randn(5, 4)
        values = torch.randn(5)
        ds = ValueDataset(states, values)
        s, v = ds[2]
        assert torch.equal(s, states[2])
        assert torch.equal(v, values[2])


# ---------------------------------------------------------------------------
# NeuralTrainer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeuralTrainer:
    """Tests for NeuralTrainer."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a minimal training config for testing."""
        return TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            num_epochs=2,
            scheduler_type="cosine",
            checkpoint_dir=str(tmp_path / "checkpoints"),
            save_every=1,
            log_every=1,
            early_stopping_patience=3,
            device="cpu",
            use_wandb=False,
        )

    @pytest.fixture
    def model(self):
        return _make_simple_model()

    @pytest.fixture
    def loss_fn(self):
        return _make_simple_loss()

    @pytest.fixture
    def trainer(self, model, loss_fn, config):
        return NeuralTrainer(model, loss_fn, config, model_name="test_model")

    def test_init(self, trainer, config):
        """Test trainer initializes correctly."""
        assert trainer.model_name == "test_model"
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float("inf")
        assert trainer.epochs_without_improvement == 0
        assert trainer.training_history == []
        assert trainer.checkpoint_dir.exists()
        assert trainer.wandb is None

    def test_init_creates_checkpoint_dir(self, model, loss_fn, tmp_path):
        """Test that checkpoint directory is created on init."""
        ckpt_dir = tmp_path / "nested" / "checkpoints"
        config = TrainingConfig(
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
        )
        NeuralTrainer(model, loss_fn, config)
        assert ckpt_dir.exists()

    def test_optimizer_is_adam(self, trainer):
        """Test optimizer is Adam with correct parameters."""
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.defaults["lr"] == 0.01

    def test_create_scheduler_cosine(self, trainer):
        """Test cosine scheduler creation."""
        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_scheduler_step(self, model, loss_fn, tmp_path):
        """Test step scheduler creation."""
        config = TrainingConfig(
            scheduler_type="step",
            scheduler_params={"step_size": 10, "gamma": 0.5},
            checkpoint_dir=str(tmp_path),
            device="cpu",
        )
        trainer = NeuralTrainer(model, loss_fn, config)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_scheduler_plateau(self, model, loss_fn, tmp_path):
        """Test plateau scheduler creation."""
        config = TrainingConfig(
            scheduler_type="plateau",
            checkpoint_dir=str(tmp_path),
            device="cpu",
        )
        trainer = NeuralTrainer(model, loss_fn, config)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_scheduler_none(self, model, loss_fn, tmp_path):
        """Test no scheduler when type is None."""
        config = TrainingConfig(
            scheduler_type=None,
            checkpoint_dir=str(tmp_path),
            device="cpu",
        )
        trainer = NeuralTrainer(model, loss_fn, config)
        assert trainer.scheduler is None

    def test_create_scheduler_invalid(self, model, loss_fn, tmp_path):
        """Test invalid scheduler type raises ValueError."""
        config = TrainingConfig(
            scheduler_type="invalid",
            checkpoint_dir=str(tmp_path),
            device="cpu",
        )
        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            NeuralTrainer(model, loss_fn, config)

    def test_train_epoch(self, config):
        """Test training for one epoch."""
        t = _make_trainer_with_mock_forward(config)
        states = torch.randn(16, 4)
        targets = torch.randn(16, 3)
        dataset = torch.utils.data.TensorDataset(states, targets)
        loader = DataLoader(dataset, batch_size=4)

        avg_loss, metrics = t.train_epoch(loader)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert isinstance(metrics, dict)

    def test_validate(self, config):
        """Test validation step."""
        t = _make_trainer_with_mock_forward(config)
        states = torch.randn(8, 4)
        targets = torch.randn(8, 3)
        dataset = torch.utils.data.TensorDataset(states, targets)
        loader = DataLoader(dataset, batch_size=4)

        avg_loss, metrics = t.validate(loader)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_save_and_load_checkpoint(self, trainer):
        """Test saving and loading checkpoints."""
        trainer.current_epoch = 5
        trainer.best_val_loss = 0.123

        trainer.save_checkpoint("test_ckpt.pt")
        ckpt_path = trainer.checkpoint_dir / "test_ckpt.pt"
        assert ckpt_path.exists()

        # Reset state
        trainer.current_epoch = 0
        trainer.best_val_loss = float("inf")

        trainer.load_checkpoint("test_ckpt.pt")
        assert trainer.current_epoch == 5
        assert trainer.best_val_loss == pytest.approx(0.123)

    def test_load_checkpoint_not_found(self, trainer):
        """Test loading non-existent checkpoint raises error."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            trainer.load_checkpoint("nonexistent.pt")

    def test_save_checkpoint_includes_scheduler(self, trainer):
        """Test checkpoint includes scheduler state when present."""
        trainer.save_checkpoint("with_scheduler.pt")
        ckpt = torch.load(trainer.checkpoint_dir / "with_scheduler.pt", weights_only=False)
        assert "scheduler_state_dict" in ckpt

    def test_save_checkpoint_no_scheduler(self, model, loss_fn, tmp_path):
        """Test checkpoint without scheduler state."""
        config = TrainingConfig(
            scheduler_type=None,
            checkpoint_dir=str(tmp_path),
            device="cpu",
        )
        trainer = NeuralTrainer(model, loss_fn, config)
        trainer.save_checkpoint("no_sched.pt")
        ckpt = torch.load(tmp_path / "no_sched.pt", weights_only=False)
        assert "scheduler_state_dict" not in ckpt

    def test_train_full_loop_no_validation(self, config):
        """Test full training loop without validation."""
        t = _make_trainer_with_mock_forward(config)
        states = torch.randn(16, 4)
        targets = torch.randn(16, 3)
        dataset = torch.utils.data.TensorDataset(states, targets)
        loader = DataLoader(dataset, batch_size=4)

        history = t.train(loader, val_loader=None)
        assert len(history) == t.config.num_epochs
        assert all(isinstance(m, TrainingMetrics) for m in history)
        assert all(m.val_loss is None for m in history)

    def test_train_full_loop_with_validation(self, config):
        """Test full training loop with validation."""
        t = _make_trainer_with_mock_forward(config)
        train_states = torch.randn(16, 4)
        train_targets = torch.randn(16, 3)
        val_states = torch.randn(8, 4)
        val_targets = torch.randn(8, 3)

        train_ds = torch.utils.data.TensorDataset(train_states, train_targets)
        val_ds = torch.utils.data.TensorDataset(val_states, val_targets)
        train_loader = DataLoader(train_ds, batch_size=4)
        val_loader = DataLoader(val_ds, batch_size=4)

        history = t.train(train_loader, val_loader)
        assert len(history) > 0
        assert all(m.val_loss is not None for m in history)

    def test_early_stopping(self, tmp_path):
        """Test early stopping triggers when val_loss stagnates."""
        config = TrainingConfig(
            num_epochs=100,
            early_stopping_patience=2,
            # Large min_delta ensures random losses never "improve" enough
            min_delta=100.0,
            save_every=1,
            checkpoint_dir=str(tmp_path),
            device="cpu",
            batch_size=4,
        )
        t = _make_trainer_with_mock_forward(config)

        # Create loaders
        train_ds = torch.utils.data.TensorDataset(torch.randn(16, 4), torch.randn(16, 3))
        val_ds = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randn(8, 3))
        train_loader = DataLoader(train_ds, batch_size=4)
        val_loader = DataLoader(val_ds, batch_size=4)

        history = t.train(train_loader, val_loader)
        # With min_delta=100, no improvement is ever detected, so early stopping
        # should trigger after patience+1 epochs (first epoch sets baseline,
        # then patience=2 more without improvement)
        assert len(history) == 3  # epoch 0 sets best, epochs 1,2 without improvement -> stop

    def test_wandb_import_error_handled(self, model, loss_fn, tmp_path):
        """Test wandb import error is handled gracefully."""
        config = TrainingConfig(
            use_wandb=True,
            checkpoint_dir=str(tmp_path),
            device="cpu",
        )
        # Remove wandb from sys.modules so the import inside __init__ fails
        import sys

        with patch.dict(sys.modules, {"wandb": None}):
            trainer = NeuralTrainer(model, loss_fn, config)
            assert trainer.wandb is None

    def test_forward_batch_unknown_model_type(self, trainer):
        """Test _forward_batch with unknown model type raises ValueError."""
        batch = (torch.randn(4, 4), torch.randn(4, 3))
        # The trainer has a Sequential model (not PolicyNetwork or ValueNetwork)
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer._forward_batch(batch)
