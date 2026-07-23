"""
Distillation Trainer for MCTS Neural Networks.

Provides:
- DistillationTrainer: Main training loop with logging and checkpointing
- TrainingCallback: Hook interface for custom training behavior
- TrainingCheckpoint: Checkpoint management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.observability.logging import get_correlation_id, get_structured_logger

from .metrics import (
    MetricsAccumulator,
    TrainingMetrics,
)

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None

if TYPE_CHECKING:
    from .dataset import TrainingBatch
    from .networks import PolicyNetwork, ValueNetwork

logger = get_structured_logger(__name__)


@dataclass
class DistillationTrainerConfig:
    """Configuration for distillation trainer."""

    # Training parameters
    num_epochs: int = 10
    """Number of training epochs."""

    learning_rate: float = 1e-4
    """Initial learning rate."""

    weight_decay: float = 0.01
    """Weight decay (L2 regularization)."""

    warmup_steps: int = 100
    """Number of warmup steps for learning rate."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    # Loss weights
    policy_loss_weight: float = 1.0
    """Weight for policy loss."""

    value_loss_weight: float = 1.0
    """Weight for value loss."""

    use_mcts_policy: bool = True
    """Use MCTS-improved policy as target (vs LLM policy)."""

    use_outcome_value: bool = True
    """Use episode outcome as value target (vs LLM estimate)."""

    # Checkpointing
    checkpoint_dir: str | Path = "./checkpoints"
    """Directory for saving checkpoints."""

    save_every_epochs: int = 1
    """Save checkpoint every N epochs."""

    keep_last_n_checkpoints: int = 3
    """Number of recent checkpoints to keep."""

    # Logging
    log_every_steps: int = 100
    """Log metrics every N steps."""

    eval_every_epochs: int = 1
    """Evaluate on validation set every N epochs."""

    # Early stopping
    early_stopping_patience: int = 5
    """Epochs without improvement before stopping."""

    early_stopping_metric: str = "total_loss"
    """Metric to monitor for early stopping."""

    # Device
    device: str = "auto"
    """Device to use: 'auto', 'cuda', 'cpu'."""

    # Mixed precision
    use_amp: bool = False
    """Use automatic mixed precision."""

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        if self.num_epochs < 1:
            errors.append("num_epochs must be >= 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if self.warmup_steps < 0:
            errors.append("warmup_steps must be >= 0")
        if self.max_grad_norm <= 0:
            errors.append("max_grad_norm must be > 0")
        if self.policy_loss_weight < 0 or self.value_loss_weight < 0:
            errors.append("loss weights must be >= 0")

        # Validate early_stopping_metric is a valid TrainingMetrics attribute
        valid_metrics = {
            "policy_loss",
            "value_loss",
            "total_loss",
            "policy_accuracy",
            "policy_top3_accuracy",
            "policy_kl_divergence",
            "policy_entropy",
            "value_mse",
            "value_mae",
            "value_correlation",
        }
        if self.early_stopping_metric not in valid_metrics:
            errors.append(f"early_stopping_metric must be one of: {', '.join(sorted(valid_metrics))}")

        if errors:
            raise ValueError("Invalid DistillationTrainerConfig:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class TrainingCheckpoint:
    """Training checkpoint data."""

    epoch: int
    """Epoch number."""

    step: int
    """Global step."""

    policy_state_dict: dict[str, Any] | None = None
    """Policy network state dict."""

    value_state_dict: dict[str, Any] | None = None
    """Value network state dict."""

    optimizer_state_dict: dict[str, Any] | None = None
    """Optimizer state dict."""

    scheduler_state_dict: dict[str, Any] | None = None
    """LR scheduler state dict."""

    best_metric: float = float("inf")
    """Best metric value seen."""

    metrics_history: list[dict[str, float]] = field(default_factory=list)
    """History of training metrics."""

    config: dict[str, Any] = field(default_factory=dict)
    """Training configuration."""

    def save(self, filepath: str | Path) -> None:
        """Save checkpoint to file."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": self.epoch,
                "step": self.step,
                "policy_state_dict": self.policy_state_dict,
                "value_state_dict": self.value_state_dict,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_metric": self.best_metric,
                "metrics_history": self.metrics_history,
                "config": self.config,
            },
            filepath,
        )

        logger.info("Saved checkpoint", path=str(filepath), epoch=self.epoch)

    @classmethod
    def load(cls, filepath: str | Path) -> TrainingCheckpoint:
        """
        Load checkpoint from file.

        Note: Only load checkpoints from trusted sources. The checkpoint format
        uses pickle which can execute arbitrary code during loading.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        # Note: weights_only=False is required for loading full checkpoints with
        # optimizer state dicts and other non-tensor data. Only load from trusted sources.
        data = torch.load(filepath, map_location="cpu", weights_only=False)

        return cls(
            epoch=data["epoch"],
            step=data["step"],
            policy_state_dict=data.get("policy_state_dict"),
            value_state_dict=data.get("value_state_dict"),
            optimizer_state_dict=data.get("optimizer_state_dict"),
            scheduler_state_dict=data.get("scheduler_state_dict"),
            best_metric=data.get("best_metric", float("inf")),
            metrics_history=data.get("metrics_history", []),
            config=data.get("config", {}),
        )


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_epoch_start(self, epoch: int, trainer: DistillationTrainer) -> None:
        """Called at the start of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics, trainer: DistillationTrainer) -> None:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_batch_end(self, step: int, metrics: TrainingMetrics, trainer: DistillationTrainer) -> None:
        """Called after each batch."""
        pass

    @abstractmethod
    def on_training_end(self, trainer: DistillationTrainer) -> None:
        """Called when training completes."""
        pass


class LoggingCallback(TrainingCallback):
    """Default callback that logs training progress."""

    def on_epoch_start(self, epoch: int, trainer: DistillationTrainer) -> None:
        """Log epoch start."""
        logger.info(
            "Starting epoch",
            epoch=epoch + 1,
            total_epochs=trainer._config.num_epochs,
        )

    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics, trainer: DistillationTrainer) -> None:
        """Log epoch metrics."""
        logger.info(
            "Epoch complete",
            epoch=epoch + 1,
            policy_loss=round(metrics.policy_loss, 4),
            value_loss=round(metrics.value_loss, 4),
            policy_accuracy=round(metrics.policy_accuracy, 4),
            value_mse=round(metrics.value_mse, 4),
        )

    def on_batch_end(self, step: int, metrics: TrainingMetrics, trainer: DistillationTrainer) -> None:
        """Log batch metrics periodically."""
        if step % trainer._config.log_every_steps == 0:
            logger.info(
                "Training step",
                step=step,
                total_loss=round(metrics.total_loss, 4),
                learning_rate=metrics.learning_rate,
            )

    def on_training_end(self, trainer: DistillationTrainer) -> None:
        """Log training completion."""
        logger.info(
            "Training complete",
            total_steps=trainer._global_step,
            epochs=trainer._current_epoch + 1,
        )


class DistillationTrainer:
    """
    Trainer for distilling LLM/MCTS knowledge into neural networks.

    Trains policy and value networks using data collected from
    LLM-guided MCTS search.
    """

    def __init__(
        self,
        policy_network: PolicyNetwork | None = None,
        value_network: ValueNetwork | None = None,
        config: DistillationTrainerConfig | None = None,
        callbacks: list[TrainingCallback] | None = None,
    ):
        """
        Initialize trainer.

        Args:
            policy_network: Policy network to train (optional)
            value_network: Value network to train (optional)
            config: Training configuration
            callbacks: List of training callbacks
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self._config = config or DistillationTrainerConfig()
        self._config.validate()

        # Networks
        self._policy_network = policy_network
        self._value_network = value_network

        # Setup device
        self._device = self._get_device()

        # Move networks to device
        if self._policy_network is not None:
            self._policy_network = self._policy_network.to(self._device)
        if self._value_network is not None:
            self._value_network = self._value_network.to(self._device)

        # Optimizers (created during training)
        self._optimizer: optim.Optimizer | None = None
        self._scheduler: optim.lr_scheduler._LRScheduler | None = None

        # AMP scaler - only enabled for CUDA devices
        self._use_amp = self._config.use_amp and self._device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp) if self._use_amp else None

        if self._config.use_amp and not self._use_amp:
            logger.warning(
                "AMP requested but device is not CUDA, disabling AMP",
                device=str(self._device),
            )

        # Callbacks
        self._callbacks = callbacks or [LoggingCallback()]

        # Training state
        self._current_epoch = 0
        self._global_step = 0
        self._best_metric = float("inf")
        self._metrics_history: list[dict[str, float]] = []
        self._early_stopping_counter = 0

        logger.info(
            "Initialized DistillationTrainer",
            device=str(self._device),
            policy_network=self._policy_network is not None,
            value_network=self._value_network is not None,
        )

    def _get_device(self) -> torch.device:
        """Determine device to use."""
        if self._config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self._config.device)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for all networks."""
        params = []
        if self._policy_network is not None:
            params.extend(self._policy_network.parameters())
        if self._value_network is not None:
            params.extend(self._value_network.parameters())

        if not params:
            raise ValueError("At least one network (policy_network or value_network) must be provided for training")

        return optim.AdamW(
            params,
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

    def _create_scheduler(self, num_training_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        # Clamp pct_start to [0, 1] to handle cases where warmup_steps > total_steps
        pct_start = min(1.0, max(0.0, self._config.warmup_steps / max(num_training_steps, 1)))
        return optim.lr_scheduler.OneCycleLR(
            self._optimizer,
            max_lr=self._config.learning_rate,
            total_steps=num_training_steps,
            pct_start=pct_start,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        resume_from: str | Path | None = None,
    ) -> TrainingMetrics:
        """
        Run training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            resume_from: Path to checkpoint to resume from

        Returns:
            Final training metrics
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            self._load_checkpoint(resume_from)

        # Create optimizer and scheduler
        self._optimizer = self._create_optimizer()
        num_training_steps = len(train_loader) * self._config.num_epochs
        self._scheduler = self._create_scheduler(num_training_steps)

        # Restore optimizer state if resuming
        if resume_from is not None:
            checkpoint = TrainingCheckpoint.load(resume_from)
            if checkpoint.optimizer_state_dict:
                self._optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            if checkpoint.scheduler_state_dict:
                self._scheduler.load_state_dict(checkpoint.scheduler_state_dict)

        logger.info(
            "Starting training",
            num_epochs=self._config.num_epochs,
            train_samples=len(train_loader.dataset),
            val_samples=len(val_loader.dataset) if val_loader else 0,
            correlation_id=get_correlation_id(),
        )

        final_metrics = TrainingMetrics()
        start_epoch = self._current_epoch

        try:
            for epoch in range(start_epoch, self._config.num_epochs):
                self._current_epoch = epoch

                # Callbacks
                for callback in self._callbacks:
                    callback.on_epoch_start(epoch, self)

                # Train epoch
                train_metrics = self._train_epoch(train_loader)

                # Evaluate if needed
                if val_loader is not None and (epoch + 1) % self._config.eval_every_epochs == 0:
                    val_metrics = self._evaluate(val_loader)
                    train_metrics.value_mse = val_metrics.value_mse  # Update with val metrics

                # Callbacks
                for callback in self._callbacks:
                    callback.on_epoch_end(epoch, train_metrics, self)

                # Record metrics
                self._metrics_history.append(train_metrics.to_dict())
                final_metrics = train_metrics

                # Checkpointing
                if (epoch + 1) % self._config.save_every_epochs == 0:
                    self._save_checkpoint(epoch)

                # Early stopping
                current_metric = getattr(train_metrics, self._config.early_stopping_metric)
                if current_metric < self._best_metric:
                    self._best_metric = current_metric
                    self._early_stopping_counter = 0
                else:
                    self._early_stopping_counter += 1
                    if self._early_stopping_counter >= self._config.early_stopping_patience:
                        logger.info(
                            "Early stopping triggered",
                            epoch=epoch + 1,
                            patience=self._config.early_stopping_patience,
                        )
                        break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")

        finally:
            # Save final checkpoint
            self._save_checkpoint(self._current_epoch, is_final=True)

            # Callbacks
            for callback in self._callbacks:
                callback.on_training_end(self)

        return final_metrics

    def _train_epoch(self, train_loader: DataLoader) -> TrainingMetrics:
        """Train for one epoch."""
        if self._policy_network is not None:
            self._policy_network.train()
        if self._value_network is not None:
            self._value_network.train()

        accumulator = MetricsAccumulator()

        # These should be set by train() before this method is called
        assert self._optimizer is not None, "Optimizer must be initialized"
        assert self._scheduler is not None, "Scheduler must be initialized"

        for _batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self._device)

            # Forward pass
            with torch.amp.autocast(self._device.type, enabled=self._use_amp):
                loss, batch_metrics = self._compute_loss(batch)

            # Check for NaN/inf loss - skip batch if detected
            if not torch.isfinite(loss):
                logger.warning(
                    "Detected non-finite loss, skipping batch",
                    step=self._global_step,
                    loss_value=loss.item() if loss.numel() == 1 else "tensor",
                )
                continue

            # Backward pass
            self._optimizer.zero_grad()

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self._optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._get_all_parameters(),
                    self._config.max_grad_norm,
                )
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._get_all_parameters(),
                    self._config.max_grad_norm,
                )
                self._optimizer.step()

            self._scheduler.step()
            self._global_step += 1

            # Accumulate metrics
            accumulator.update(
                policy_loss=batch_metrics["policy_loss"],
                value_loss=batch_metrics["value_loss"],
                total_loss=loss.item(),
                policy_correct=batch_metrics["policy_correct"],
                policy_top3_correct=batch_metrics["policy_top3_correct"],
                value_predictions=batch_metrics["value_predictions"],
                value_targets=batch_metrics["value_targets"],
                batch_size=len(batch.episode_ids),
            )

            # Step-level callbacks
            step_metrics = TrainingMetrics(
                total_loss=loss.item(),
                learning_rate=self._scheduler.get_last_lr()[0],
                gradient_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                step=self._global_step,
            )
            for callback in self._callbacks:
                callback.on_batch_end(self._global_step, step_metrics, self)

        # Compute epoch metrics
        return accumulator.compute(
            learning_rate=self._scheduler.get_last_lr()[0],
            epoch=self._current_epoch,
            step=self._global_step,
        )

    def _compute_loss(self, batch: TrainingBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss for a batch."""
        policy_loss = torch.tensor(0.0, device=self._device)
        value_loss = torch.tensor(0.0, device=self._device)
        batch_size = len(batch.episode_ids)
        value_targets = batch.outcome if self._config.use_outcome_value else batch.llm_value
        batch_metrics: dict[str, Any] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "policy_correct": 0,
            "policy_top3_correct": 0,
            # Initialize with zeros matching batch size for proper metric aggregation
            "value_predictions": torch.zeros(batch_size, device=self._device),
            "value_targets": value_targets,
        }

        # Policy loss
        if self._policy_network is not None:
            log_probs = self._policy_network(
                batch.code_tokens,
                batch.code_attention_mask,
                batch.problem_tokens,
                batch.problem_attention_mask,
                batch.action_mask,
            )

            # Target policy
            target_policy = batch.mcts_policy if self._config.use_mcts_policy else batch.llm_policy

            # KL divergence loss
            policy_loss = self._kl_loss(log_probs, target_policy, batch.action_mask)

            # Metrics
            batch_metrics["policy_loss"] = policy_loss.item()
            batch_metrics["policy_correct"] = int(
                (log_probs.argmax(dim=-1) == target_policy.argmax(dim=-1)).sum().item()
            )
            # Top-3 accuracy
            _, top3_preds = log_probs.topk(3, dim=-1)
            target_actions = target_policy.argmax(dim=-1, keepdim=True)
            batch_metrics["policy_top3_correct"] = int((top3_preds == target_actions).any(dim=-1).sum().item())

        # Value loss
        if self._value_network is not None:
            value_preds = self._value_network(
                batch.code_tokens,
                batch.code_attention_mask,
                batch.problem_tokens,
                batch.problem_attention_mask,
            )

            # Target value
            target_value = batch.outcome if self._config.use_outcome_value else batch.llm_value

            # MSE loss
            value_loss = nn.functional.mse_loss(value_preds, target_value)

            # Metrics
            batch_metrics["value_loss"] = value_loss.item()
            batch_metrics["value_predictions"] = value_preds.detach()
            batch_metrics["value_targets"] = target_value

        # Combined loss
        total_loss = self._config.policy_loss_weight * policy_loss + self._config.value_loss_weight * value_loss

        return total_loss, batch_metrics

    def _kl_loss(
        self,
        log_probs: torch.Tensor,
        target_probs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss between predicted and target policies.

        Uses cross-entropy H(target, pred) = -sum(target * log_pred) which is
        equivalent to KL divergence up to the target entropy constant.

        The loss is normalized by the number of valid actions per sample to ensure
        consistent gradient magnitudes regardless of action space size.

        Args:
            log_probs: Log probabilities from policy network [batch, actions]
            target_probs: Target probability distribution [batch, actions]
            action_mask: Mask for valid actions [batch, actions]

        Returns:
            Mean cross-entropy loss across the batch
        """
        cross_entropy = -(target_probs * log_probs)
        cross_entropy = cross_entropy * action_mask
        # Normalize by number of valid actions per sample for consistent scaling
        cross_entropy = cross_entropy.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-8)
        return cross_entropy.mean()

    def _get_all_parameters(self):
        """Get parameters from all networks."""
        params = []
        if self._policy_network is not None:
            params.extend(self._policy_network.parameters())
        if self._value_network is not None:
            params.extend(self._value_network.parameters())
        return params

    def _evaluate(self, val_loader: DataLoader) -> TrainingMetrics:
        """Evaluate on validation set."""
        if self._policy_network is not None:
            self._policy_network.eval()
        if self._value_network is not None:
            self._value_network.eval()

        accumulator = MetricsAccumulator()

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self._device)
                loss, batch_metrics = self._compute_loss(batch)

                accumulator.update(
                    policy_loss=batch_metrics["policy_loss"],
                    value_loss=batch_metrics["value_loss"],
                    total_loss=loss.item(),
                    policy_correct=batch_metrics["policy_correct"],
                    policy_top3_correct=batch_metrics["policy_top3_correct"],
                    value_predictions=batch_metrics["value_predictions"],
                    value_targets=batch_metrics["value_targets"],
                    batch_size=len(batch.episode_ids),
                )

        return accumulator.compute()

    def _save_checkpoint(self, epoch: int, is_final: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self._config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = TrainingCheckpoint(
            epoch=epoch,
            step=self._global_step,
            policy_state_dict=self._policy_network.state_dict() if self._policy_network else None,
            value_state_dict=self._value_network.state_dict() if self._value_network else None,
            optimizer_state_dict=self._optimizer.state_dict() if self._optimizer else None,
            scheduler_state_dict=self._scheduler.state_dict() if self._scheduler else None,
            best_metric=self._best_metric,
            metrics_history=self._metrics_history,
            config=self._config.__dict__,
        )

        # Save checkpoint
        filename = "checkpoint_final.pt" if is_final else f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint.save(checkpoint_dir / filename)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def _load_checkpoint(self, filepath: str | Path) -> None:
        """Load training state from checkpoint."""
        checkpoint = TrainingCheckpoint.load(filepath)

        self._current_epoch = checkpoint.epoch + 1
        self._global_step = checkpoint.step
        self._best_metric = checkpoint.best_metric
        self._metrics_history = checkpoint.metrics_history

        if checkpoint.policy_state_dict and self._policy_network is not None:
            self._policy_network.load_state_dict(checkpoint.policy_state_dict)

        if checkpoint.value_state_dict and self._value_network is not None:
            self._value_network.load_state_dict(checkpoint.value_state_dict)

        logger.info(
            "Resumed from checkpoint",
            epoch=self._current_epoch,
            step=self._global_step,
        )

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        checkpoint_dir = Path(self._config.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

        if len(checkpoints) > self._config.keep_last_n_checkpoints:
            for ckpt in checkpoints[: -self._config.keep_last_n_checkpoints]:
                ckpt.unlink()
                logger.debug("Removed old checkpoint", checkpoint_path=str(ckpt))


def create_trainer(
    policy_network: PolicyNetwork | None = None,
    value_network: ValueNetwork | None = None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    checkpoint_dir: str | Path = "./checkpoints",
    use_mcts_policy: bool = True,
    use_outcome_value: bool = True,
    device: str = "auto",
) -> DistillationTrainer:
    """
    Create a distillation trainer with specified parameters.

    Args:
        policy_network: Policy network to train
        value_network: Value network to train
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        checkpoint_dir: Directory for checkpoints
        use_mcts_policy: Use MCTS-improved policy as target
        use_outcome_value: Use episode outcome as value target
        device: Device to use

    Returns:
        Configured DistillationTrainer
    """
    config = DistillationTrainerConfig(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        use_mcts_policy=use_mcts_policy,
        use_outcome_value=use_outcome_value,
        device=device,
    )

    return DistillationTrainer(
        policy_network=policy_network,
        value_network=value_network,
        config=config,
    )
