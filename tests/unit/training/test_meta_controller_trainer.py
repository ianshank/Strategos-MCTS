"""
Unit tests for meta-controller training orchestrator.

Tests configuration, training loop, calibration, and checkpointing.

Based on: NEXT_STEPS_PLAN.md Phase 4.1
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Check PyTorch availability
# =============================================================================

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    pass

skip_if_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed",
)

# sentencepiece / tiktoken — required by the BERT tokenizer inside transformers.
_HAS_SENTENCEPIECE = False
try:
    import sentencepiece  # noqa: F401

    _HAS_SENTENCEPIECE = True
except ImportError:
    pass

_HAS_TIKTOKEN = False
try:
    import tiktoken  # noqa: F401

    _HAS_TIKTOKEN = True
except ImportError:
    pass

_HAS_BERT_TOKENIZER_DEPS = _HAS_SENTENCEPIECE or _HAS_TIKTOKEN
skip_if_no_bert_tokenizer = pytest.mark.skipif(
    not _HAS_BERT_TOKENIZER_DEPS,
    reason="sentencepiece or tiktoken required for BERT tokenizer initialisation",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def training_config():
    """Create default training configuration."""
    pytest.importorskip("torch")
    from src.training.meta_controller_trainer import MetaControllerTrainingConfig

    return MetaControllerTrainingConfig(
        model_type="bert",
        num_agents=4,
        hidden_dim=64,
        num_layers=1,
        epochs=2,
        batch_size=4,
        early_stopping_patience=2,
        use_wandb=False,
        use_braintrust=False,
    )


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader for testing."""
    if not TORCH_AVAILABLE:
        return None

    # Create simple synthetic data
    inputs = torch.randn(16, 10, 768)  # [batch, seq, hidden]
    targets = torch.randint(0, 4, (16,))  # 4 classes

    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def mock_simple_model():
    """Create a simple model for testing."""
    if not TORCH_AVAILABLE:
        return None

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(10 * 768, 4)

        def forward(self, x):
            return self.fc(self.flatten(x))

    return SimpleModel()


# =============================================================================
# Configuration Tests
# =============================================================================


@skip_if_no_torch
class TestMetaControllerTrainingConfig:
    """Tests for training configuration."""

    def test_default_config_values(self, training_config):
        """Test default configuration values."""
        assert training_config.model_type == "bert"
        assert training_config.num_agents == 4
        assert training_config.learning_rate == 2e-5

    def test_config_curriculum_stages(self, training_config):
        """Test curriculum stages are defined."""
        assert training_config.curriculum_stages == ["simple", "medium", "complex"]

    def test_config_checkpoint_dir_default(self, training_config):
        """Test checkpoint dir has default."""
        assert training_config.checkpoint_dir == Path("checkpoints/meta_controller")


# =============================================================================
# TrainingMetrics Tests
# =============================================================================


@skip_if_no_torch
class TestTrainingMetrics:
    """Tests for training metrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics can be created."""
        from src.training.meta_controller_trainer import TrainingMetrics

        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.8,
            val_loss=0.6,
            val_accuracy=0.75,
            calibration_error=0.05,
        )

        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.val_accuracy == 0.75

    def test_metrics_optional_fields(self):
        """Test optional fields have defaults."""
        from src.training.meta_controller_trainer import TrainingMetrics

        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.8,
            val_loss=0.6,
            val_accuracy=0.75,
        )

        assert metrics.calibration_error == 0.0
        assert metrics.stage == ""


# =============================================================================
# CalibrationLoss Tests
# =============================================================================


@skip_if_no_torch
class TestCalibrationLoss:
    """Tests for calibration loss."""

    def test_calibration_loss_creation(self):
        """Test calibration loss can be created."""
        from src.training.meta_controller_trainer import CalibrationLoss

        loss_fn = CalibrationLoss(num_bins=10, weight=0.1)

        assert loss_fn.num_bins == 10
        assert loss_fn.weight == 0.1

    def test_calibration_loss_computation(self):
        """Test calibration loss computes ECE."""
        from src.training.meta_controller_trainer import CalibrationLoss

        loss_fn = CalibrationLoss(num_bins=10, weight=0.1)

        # Perfect calibration case
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.6])
        predictions = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])  # All correct

        loss, ece = loss_fn(confidences, predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(ece, float)

    def test_calibration_loss_returns_zero_for_perfect_calibration(self):
        """Test ECE is low for well-calibrated predictions."""
        from src.training.meta_controller_trainer import CalibrationLoss

        loss_fn = CalibrationLoss(num_bins=10, weight=1.0)

        # All predictions correct with high confidence
        confidences = torch.ones(100)
        predictions = torch.zeros(100, dtype=torch.long)
        targets = torch.zeros(100, dtype=torch.long)

        loss, ece = loss_fn(confidences, predictions, targets)

        # Should be close to 0 (perfect calibration)
        assert ece < 0.1


# =============================================================================
# MetaControllerTrainingOrchestrator Initialization Tests
# =============================================================================


@skip_if_no_torch
class TestOrchestratorInitialization:
    """Tests for orchestrator initialization."""

    def test_orchestrator_creation(self, training_config):
        """Test orchestrator can be created."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        orchestrator = MetaControllerTrainingOrchestrator(training_config)

        assert orchestrator is not None
        assert orchestrator._model is None  # Not initialized yet

    def test_orchestrator_device_detection(self, training_config):
        """Test device auto-detection."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        training_config.device = "auto"
        orchestrator = MetaControllerTrainingOrchestrator(training_config)

        # Should be either cuda or cpu
        assert orchestrator._device.type in ["cuda", "cpu"]

    def test_orchestrator_creates_checkpoint_dir(self, training_config):
        """Test checkpoint directory is created."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            training_config.checkpoint_dir = Path(tmpdir) / "test_checkpoints"
            # Create orchestrator to trigger checkpoint dir creation
            _orchestrator = MetaControllerTrainingOrchestrator(training_config)

            assert training_config.checkpoint_dir.exists()
            assert _orchestrator is not None  # Verify instance was created


# =============================================================================
# Model Initialization Tests
# =============================================================================


@skip_if_no_torch
class TestModelInitialization:
    """Tests for model initialization."""

    @skip_if_no_bert_tokenizer
    def test_initialize_model_bert(self, training_config):
        """Test BERT model initialization.

        Requires ``sentencepiece`` or ``tiktoken`` — skipped when neither is
        installed (the HuggingFace BERT tokenizer raises ValueError otherwise).
        """
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        training_config.model_type = "bert"
        orchestrator = MetaControllerTrainingOrchestrator(training_config)
        orchestrator.initialize_model()

        assert orchestrator._model is not None

    def test_initialize_model_rnn(self, training_config):
        """Test RNN model initialization."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        training_config.model_type = "rnn"
        orchestrator = MetaControllerTrainingOrchestrator(training_config)
        orchestrator.initialize_model()

        assert orchestrator._model is not None

    def test_initialize_with_custom_model(self, training_config, mock_simple_model):
        """Test initialization with custom model."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )
        orchestrator.initialize_model()

        assert orchestrator._model is mock_simple_model


# =============================================================================
# Optimizer Setup Tests
# =============================================================================


@skip_if_no_torch
class TestOptimizerSetup:
    """Tests for optimizer and scheduler setup."""

    def test_setup_optimizer(self, training_config, mock_simple_model):
        """Test optimizer setup."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )
        orchestrator.initialize_model()
        orchestrator.setup_optimizer()

        assert orchestrator._optimizer is not None
        assert orchestrator._scheduler is not None

    def test_optimizer_uses_config_lr(self, training_config, mock_simple_model):
        """Test optimizer uses configured learning rate."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        training_config.learning_rate = 1e-4
        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )
        orchestrator.initialize_model()
        orchestrator.setup_optimizer()

        assert orchestrator._optimizer.param_groups[0]["lr"] == 1e-4


# =============================================================================
# Training Loop Tests
# =============================================================================


@skip_if_no_torch
class TestTrainingLoop:
    """Tests for training loop."""

    def test_train_runs_without_error(self, training_config, mock_simple_model, mock_data_loader):
        """Test training runs without errors."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )

        metrics = orchestrator.train(mock_data_loader, mock_data_loader)

        assert len(metrics) > 0
        assert all(isinstance(m.train_loss, float) for m in metrics)

    def test_train_returns_metrics_per_epoch(self, training_config, mock_simple_model, mock_data_loader):
        """Test training returns metrics for each epoch."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        training_config.epochs = 3
        training_config.early_stopping_patience = 10  # Disable early stopping

        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )

        metrics = orchestrator.train(mock_data_loader, mock_data_loader)

        assert len(metrics) == 3

    def test_train_updates_best_val_loss(self, training_config, mock_simple_model, mock_data_loader):
        """Test training updates best validation loss."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )

        initial_best = orchestrator._best_val_loss
        orchestrator.train(mock_data_loader, mock_data_loader)

        assert orchestrator._best_val_loss < initial_best


# =============================================================================
# Early Stopping Tests
# =============================================================================


@skip_if_no_torch
class TestEarlyStopping:
    """Tests for early stopping."""

    def test_early_stopping_triggers(self, training_config, mock_simple_model, mock_data_loader):
        """Test early stopping can trigger."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        # Set patience to 1 to trigger early stopping quickly
        training_config.epochs = 10
        training_config.early_stopping_patience = 1

        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )

        metrics = orchestrator.train(mock_data_loader, mock_data_loader)

        # Should stop before 10 epochs
        assert len(metrics) < 10

    def test_check_early_stopping_method(self, training_config):
        """Test early stopping check logic."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        training_config.early_stopping_patience = 2
        orchestrator = MetaControllerTrainingOrchestrator(training_config)
        orchestrator._best_val_loss = 1.0

        # First check - improvement (0.5 < 1.0 - delta)
        result = orchestrator._check_early_stopping(0.5)
        assert result is False
        assert orchestrator._patience_counter == 0

        # Simulate best val loss being tracked externally
        orchestrator._best_val_loss = 0.5

        # Second check - no improvement (0.6 >= 0.5 - delta)
        result = orchestrator._check_early_stopping(0.6)
        assert result is False
        assert orchestrator._patience_counter == 1

        # Third check - still no improvement, triggers early stopping
        result = orchestrator._check_early_stopping(0.7)
        assert result is True


# =============================================================================
# Checkpointing Tests
# =============================================================================


@skip_if_no_torch
class TestCheckpointing:
    """Tests for model checkpointing."""

    def test_save_checkpoint(self, training_config, mock_simple_model):
        """Test checkpoint saving."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            training_config.checkpoint_dir = Path(tmpdir)

            orchestrator = MetaControllerTrainingOrchestrator(
                training_config,
                model=mock_simple_model,
            )
            orchestrator.initialize_model()
            orchestrator.setup_optimizer()

            orchestrator._save_checkpoint("test_checkpoint.pt")

            assert (training_config.checkpoint_dir / "test_checkpoint.pt").exists()

    def test_load_checkpoint(self, training_config, mock_simple_model):
        """Test checkpoint loading."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            training_config.checkpoint_dir = Path(tmpdir)

            # Save
            orchestrator = MetaControllerTrainingOrchestrator(
                training_config,
                model=mock_simple_model,
            )
            orchestrator.initialize_model()
            orchestrator.setup_optimizer()
            orchestrator._current_epoch = 5
            orchestrator._best_val_loss = 0.5
            orchestrator._save_checkpoint("test.pt")

            # Load into new orchestrator
            new_orchestrator = MetaControllerTrainingOrchestrator(
                training_config,
                model=mock_simple_model,
            )
            new_orchestrator.initialize_model()
            new_orchestrator.setup_optimizer()
            new_orchestrator.load_checkpoint(training_config.checkpoint_dir / "test.pt")

            assert new_orchestrator._current_epoch == 5
            assert new_orchestrator._best_val_loss == 0.5


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateMetaControllerTrainer:
    """Tests for factory function."""

    @skip_if_no_torch
    def test_factory_creates_orchestrator(self):
        """Test factory creates orchestrator."""
        from src.training.meta_controller_trainer import create_meta_controller_trainer

        trainer = create_meta_controller_trainer(
            model_type="rnn",
            num_agents=3,
            epochs=5,
        )

        assert trainer is not None
        assert trainer.config.model_type == "rnn"
        assert trainer.config.num_agents == 3

    def test_factory_returns_none_without_torch(self):
        """Test factory returns None when PyTorch unavailable."""
        # Import the module directly to avoid __init__.py chain
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "meta_controller_trainer",
            "src/training/meta_controller_trainer.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("Cannot load module directly")

        # We test that the module correctly returns None when _HAS_TORCH is False
        # This is done by importing the function and mocking
        if not TORCH_AVAILABLE:
            # If torch is not available, the factory should return None by default
            from src.training.meta_controller_trainer import create_meta_controller_trainer

            trainer = create_meta_controller_trainer()
            assert trainer is None
        else:
            # If torch IS available, we mock it to test the guard works
            from src.training.meta_controller_trainer import create_meta_controller_trainer

            with patch("src.training.meta_controller_trainer._HAS_TORCH", False):
                trainer = create_meta_controller_trainer()
                assert trainer is None


# =============================================================================
# Training History Tests
# =============================================================================


@skip_if_no_torch
class TestTrainingHistory:
    """Tests for training history tracking."""

    def test_get_training_history(self, training_config, mock_simple_model, mock_data_loader):
        """Test training history is tracked."""
        from src.training.meta_controller_trainer import MetaControllerTrainingOrchestrator

        orchestrator = MetaControllerTrainingOrchestrator(
            training_config,
            model=mock_simple_model,
        )

        orchestrator.train(mock_data_loader, mock_data_loader)

        history = orchestrator.get_training_history()

        assert len(history) > 0
        assert all(hasattr(m, "train_loss") for m in history)
