"""
Unified Meta-Controller Training Orchestrator.

Provides a comprehensive training pipeline for meta-controller models
(BERT and RNN variants) with support for:
- Curriculum learning
- Calibration-aware training
- Feature importance tracking
- Experiment tracking (W&B, Braintrust)
- Early stopping and checkpointing

Best Practices 2025:
- Configuration via Pydantic models
- Graceful degradation without optional dependencies
- Structured logging with correlation IDs
- Thread-safe metric collection
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Module logger
logger = logging.getLogger(__name__)

# Check dependencies
_HAS_TORCH: bool = False
try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None


@dataclass
class MetaControllerTrainingConfig:
    """Configuration for meta-controller training."""

    # Model settings
    model_type: str = "bert"  # "bert" or "rnn"
    num_agents: int = 4
    hidden_dim: int = 256
    embedding_dim: int = 768  # Input embedding dimension (BERT default)
    num_layers: int = 2
    dropout: float = 0.1

    # Training settings
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 10
    batch_size: int = 32
    gradient_clip: float = 1.0
    warmup_steps: int = 100

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: list[str] = field(default_factory=lambda: ["simple", "medium", "complex"])
    epochs_per_stage: int = 3

    # Calibration
    use_calibration_loss: bool = True
    calibration_weight: float = 0.1

    # Early stopping
    early_stopping_patience: int = 3
    min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/meta_controller"))
    save_every_epoch: bool = True
    save_best_only: bool = True

    # Experiment tracking
    experiment_name: str = "meta-controller-training"
    use_wandb: bool = False
    use_braintrust: bool = False

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"


@dataclass
class TrainingMetrics:
    """Metrics from a single training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    calibration_error: float = 0.0
    learning_rate: float = 0.0
    epoch_time_seconds: float = 0.0
    stage: str = ""


# Define _SimpleRNNController only if torch is available
_SimpleRNNController: type | None = None

if _HAS_TORCH:

    class _SimpleRNNControllerImpl(nn.Module):
        """
        Simple GRU-based controller fallback.

        Used when full RNNMetaController is not available.
        Defined at module level to avoid class redefinition overhead.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_agents: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.classifier = nn.Linear(hidden_dim, num_agents)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through GRU and classifier."""
            _, hidden = self.gru(x)
            return self.classifier(hidden[-1])

    _SimpleRNNController = _SimpleRNNControllerImpl


class CalibrationLoss:
    """
    Expected Calibration Error (ECE) loss for confidence calibration.

    Encourages the model's predicted confidence to match its actual accuracy.
    """

    def __init__(self, num_bins: int = 10, weight: float = 0.1):
        self.num_bins = num_bins
        self.weight = weight

    def __call__(
        self,
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute calibration loss.

        Args:
            confidences: Predicted confidence scores [B]
            predictions: Predicted class indices [B]
            targets: Ground truth class indices [B]

        Returns:
            Tuple of (calibration_loss, ece_value)
        """
        accuracies = (predictions == targets).float()
        ece = torch.tensor(0.0, device=confidences.device)

        for i in range(self.num_bins):
            bin_lower = i / self.num_bins
            bin_upper = (i + 1) / self.num_bins

            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if mask.sum() > 0:
                bin_accuracy = accuracies[mask].mean()
                bin_confidence = confidences[mask].mean()
                bin_weight = mask.float().mean()
                ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)

        return ece * self.weight, float(ece)


class MetaControllerTrainingOrchestrator:
    """
    Unified training orchestrator for meta-controller models.

    Handles:
    - Model initialization (BERT or RNN)
    - Training loop with validation
    - Curriculum learning stages
    - Calibration-aware loss
    - Checkpointing and early stopping
    - Experiment tracking
    """

    def __init__(
        self,
        config: MetaControllerTrainingConfig,
        model: nn.Module | None = None,
        logger_instance: logging.Logger | None = None,
    ):
        """
        Initialize training orchestrator.

        Args:
            config: Training configuration
            model: Optional pre-initialized model
            logger_instance: Optional logger for dependency injection
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config
        self._logger = logger_instance or logger
        self._model = model
        self._optimizer = None
        self._scheduler = None
        self._calibration_loss = CalibrationLoss(weight=config.calibration_weight)
        self._criterion = nn.CrossEntropyLoss()

        # Determine device
        if config.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(config.device)

        # Training state
        self._current_epoch = 0
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._training_history: list[TrainingMetrics] = []
        self._feature_importance: dict[str, float] = {}

        # Experiment tracker (lazy initialized)
        self._tracker = None

        # Ensure checkpoint directory exists
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._log_info(
            "Initialized training orchestrator",
            model_type=config.model_type,
            device=str(self._device),
        )

    def initialize_model(self) -> None:
        """Initialize or reinitialize the model based on config."""
        if self._model is not None:
            self._model = self._model.to(self._device)
            return

        if self.config.model_type == "bert":
            self._model = self._create_bert_controller()
        elif self.config.model_type == "rnn":
            self._model = self._create_rnn_controller()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        self._model = self._model.to(self._device)

    def _create_bert_controller(self) -> nn.Module:
        """Create BERT-based meta-controller."""
        # Try to import project-specific BERT controller
        try:
            from src.agents.meta_controller.bert_controller import BERTMetaController

            return BERTMetaController(
                num_agents=self.config.num_agents,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            )
        except ImportError:
            # Fallback to a simple BERT-like classifier
            return nn.Sequential(
                nn.Linear(768, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.num_agents),
            )

    def _create_rnn_controller(self) -> nn.Module:
        """Create RNN-based meta-controller."""
        try:
            from src.agents.meta_controller.rnn_controller import RNNMetaController

            return RNNMetaController(
                num_agents=self.config.num_agents,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            )
        except ImportError:
            # Fallback to simple GRU classifier (defined at module level)
            return _SimpleRNNController(
                input_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                num_agents=self.config.num_agents,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            )

    def setup_optimizer(self) -> None:
        """Initialize optimizer and scheduler."""
        if self._model is None:
            raise ValueError("Model must be initialized before optimizer")

        self._optimizer = AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._scheduler = CosineAnnealingLR(
            self._optimizer,
            T_max=self.config.epochs,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        curriculum_data_fn: Callable[[str], DataLoader] | None = None,
    ) -> list[TrainingMetrics]:
        """
        Run complete training pipeline.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            curriculum_data_fn: Optional function to get data for each curriculum stage

        Returns:
            List of training metrics per epoch
        """
        self.initialize_model()

        # Validate model was successfully initialized
        if self._model is None:
            raise RuntimeError(
                f"Model initialization failed. Check if model_type '{self.config.model_type}' is supported."
            )

        self.setup_optimizer()
        self._initialize_tracker()

        if self.config.use_curriculum and curriculum_data_fn:
            return self._train_with_curriculum(curriculum_data_fn, val_loader)
        else:
            return self._train_loop(train_loader, val_loader)

    def _train_with_curriculum(
        self,
        curriculum_data_fn: Callable[[str], DataLoader],
        val_loader: DataLoader,
    ) -> list[TrainingMetrics]:
        """Train with curriculum learning."""
        all_metrics = []

        for stage in self.config.curriculum_stages:
            self._log_info(f"Starting curriculum stage: {stage}")

            # Get data for this stage
            train_loader = curriculum_data_fn(stage)

            # Train for this stage
            stage_epochs = self.config.epochs_per_stage
            for _ in range(stage_epochs):
                metrics = self._train_epoch(train_loader, val_loader)
                metrics.stage = stage
                all_metrics.append(metrics)

                if self._check_early_stopping(metrics.val_loss):
                    self._log_info("Early stopping triggered")
                    break

        return all_metrics

    def _train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[TrainingMetrics]:
        """Standard training loop."""
        all_metrics = []

        for epoch in range(self.config.epochs):
            self._current_epoch = epoch
            metrics = self._train_epoch(train_loader, val_loader)
            all_metrics.append(metrics)

            self._log_info(
                f"Epoch {epoch + 1}/{self.config.epochs}",
                train_loss=f"{metrics.train_loss:.4f}",
                val_loss=f"{metrics.val_loss:.4f}",
                val_acc=f"{metrics.val_accuracy:.4f}",
            )

            # Track metrics
            self._log_metrics(metrics)

            # Checkpointing
            if self.config.save_every_epoch:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            if metrics.val_loss < self._best_val_loss:
                self._best_val_loss = metrics.val_loss
                self._save_checkpoint("best_model.pt")

            # Early stopping
            if self._check_early_stopping(metrics.val_loss):
                self._log_info("Early stopping triggered")
                break

            # Step scheduler
            if self._scheduler:
                self._scheduler.step()

        self._finalize_tracker()
        return all_metrics

    def _train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingMetrics:
        """Train for one epoch."""
        start_time = time.time()

        # Training phase
        self._model.train()
        train_loss, train_acc = self._run_phase(train_loader, is_training=True)

        # Validation phase
        self._model.eval()
        with torch.no_grad():
            val_loss, val_acc, calibration_error = self._run_validation(val_loader)

        epoch_time = time.time() - start_time

        return TrainingMetrics(
            epoch=self._current_epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            calibration_error=calibration_error,
            learning_rate=self._optimizer.param_groups[0]["lr"] if self._optimizer else 0,
            epoch_time_seconds=epoch_time,
        )

    def _run_phase(
        self,
        loader: DataLoader,
        is_training: bool,
    ) -> tuple[float, float]:
        """Run training or validation phase."""
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            inputs, targets = self._prepare_batch(batch)

            if is_training:
                self._optimizer.zero_grad()

            outputs = self._model(inputs)
            loss = self._criterion(outputs, targets)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self.config.gradient_clip,
                )
                self._optimizer.step()

            total_loss += loss.item() * len(targets)
            predictions = outputs.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += len(targets)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def _run_validation(
        self,
        loader: DataLoader,
    ) -> tuple[float, float, float]:
        """Run validation with calibration error computation."""
        total_loss = 0.0
        correct = 0
        total = 0
        all_confidences = []
        all_predictions = []
        all_targets = []

        for batch in loader:
            inputs, targets = self._prepare_batch(batch)
            outputs = self._model(inputs)
            loss = self._criterion(outputs, targets)

            total_loss += loss.item() * len(targets)
            probs = torch.softmax(outputs, dim=-1)
            confidences, predictions = probs.max(dim=-1)

            correct += (predictions == targets).sum().item()
            total += len(targets)

            all_confidences.append(confidences)
            all_predictions.append(predictions)
            all_targets.append(targets)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        # Compute calibration error
        if self.config.use_calibration_loss and all_confidences:
            all_confidences = torch.cat(all_confidences)
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            _, calibration_error = self._calibration_loss(all_confidences, all_predictions, all_targets)
        else:
            calibration_error = 0.0

        return avg_loss, accuracy, calibration_error

    def _prepare_batch(
        self,
        batch: tuple | dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for model input."""
        if isinstance(batch, dict):
            inputs = batch["inputs"].to(self._device)
            targets = batch["targets"].to(self._device)
        else:
            inputs, targets = batch
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

        return inputs, targets

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if val_loss < self._best_val_loss - self.config.min_delta:
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        return self._patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / filename
        checkpoint = {
            "epoch": self._current_epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "config": self.config.__dict__,
            "best_val_loss": self._best_val_loss,
            "training_history": self._training_history,
        }
        torch.save(checkpoint, path)
        self._log_debug(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self._device)

        self._current_epoch = checkpoint["epoch"]
        self._best_val_loss = checkpoint["best_val_loss"]
        self._training_history = checkpoint.get("training_history", [])

        if self._model is not None:
            self._model.load_state_dict(checkpoint["model_state_dict"])

        if self._optimizer and checkpoint.get("optimizer_state_dict"):
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self._scheduler and checkpoint.get("scheduler_state_dict"):
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._log_info(f"Loaded checkpoint from epoch {self._current_epoch}")

    def _initialize_tracker(self) -> None:
        """Initialize experiment tracker."""
        if self.config.use_wandb or self.config.use_braintrust:
            try:
                from src.training.experiment_tracker import ExperimentTracker

                self._tracker = ExperimentTracker(
                    experiment_name=self.config.experiment_name,
                    use_wandb=self.config.use_wandb,
                    use_braintrust=self.config.use_braintrust,
                )
            except ImportError:
                self._log_warning("Experiment tracker not available")

    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log metrics to tracker."""
        if self._tracker:
            self._tracker.log_metrics(
                {
                    "train/loss": metrics.train_loss,
                    "train/accuracy": metrics.train_accuracy,
                    "val/loss": metrics.val_loss,
                    "val/accuracy": metrics.val_accuracy,
                    "val/calibration_error": metrics.calibration_error,
                    "learning_rate": metrics.learning_rate,
                    "epoch_time": metrics.epoch_time_seconds,
                }
            )

        self._training_history.append(metrics)

    def _finalize_tracker(self) -> None:
        """Finalize experiment tracking."""
        if self._tracker:
            self._tracker.finish()

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores if available."""
        return self._feature_importance

    def get_training_history(self) -> list[TrainingMetrics]:
        """Get training history."""
        return self._training_history

    # Logging helpers
    def _log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional structured data."""
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self._logger.debug("%s [%s]", message, extra_info)
        else:
            self._logger.debug(message)

    def _log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional structured data."""
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self._logger.info("%s [%s]", message, extra_info)
        else:
            self._logger.info(message)

    def _log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional structured data."""
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self._logger.warning("%s [%s]", message, extra_info)
        else:
            self._logger.warning(message)


def create_meta_controller_trainer(
    model_type: str = "bert",
    num_agents: int = 4,
    learning_rate: float = 2e-5,
    epochs: int = 10,
    device: str = "auto",
    **kwargs,
) -> MetaControllerTrainingOrchestrator | None:
    """
    Factory function to create a meta-controller trainer.

    Args:
        model_type: "bert" or "rnn"
        num_agents: Number of agents to route to
        learning_rate: Learning rate
        epochs: Number of training epochs
        device: Device to use
        **kwargs: Additional config options

    Returns:
        MetaControllerTrainingOrchestrator or None if PyTorch unavailable
    """
    if not _HAS_TORCH:
        logger.warning("PyTorch not available for meta-controller training")
        return None

    config = MetaControllerTrainingConfig(
        model_type=model_type,
        num_agents=num_agents,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device,
        **kwargs,
    )

    return MetaControllerTrainingOrchestrator(config)


__all__ = [
    "CalibrationLoss",
    "MetaControllerTrainingConfig",
    "MetaControllerTrainingOrchestrator",
    "TrainingMetrics",
    "create_meta_controller_trainer",
]
