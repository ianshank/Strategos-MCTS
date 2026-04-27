"""
Neural Network Trainer for Policy and Value Networks.

Implements training loops, optimization, and evaluation for neural
components of the hybrid LLM-neural MCTS system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.observability.logging import get_logger

from ..models.policy_network import PolicyLoss, PolicyNetwork
from ..models.value_network import ValueLoss, ValueNetwork
from ..utils.device import get_default_device_str

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for neural network training."""

    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    num_epochs: int = 100
    gradient_clip: float = 1.0

    # Learning rate schedule
    scheduler_type: str = "cosine"  # 'cosine', 'step', 'plateau', None
    scheduler_params: dict[str, Any] = field(default_factory=dict)

    # Early stopping
    early_stopping_patience: int = 10
    min_delta: float = 0.0001

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save every N epochs
    keep_best_only: bool = True

    # Logging
    log_every: int = 10  # Log every N batches
    use_wandb: bool = False
    wandb_project: str | None = None

    # Device
    device: str = field(default_factory=get_default_device_str)


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    learning_rate: float = 0.0
    additional_metrics: dict[str, float] = field(default_factory=dict)


class PolicyDataset(Dataset):
    """Dataset for policy network training."""

    def __init__(self, states: torch.Tensor, actions: torch.Tensor, values: torch.Tensor | None = None):
        """
        Initialize policy dataset.

        Args:
            states: [N, state_dim] state tensors
            actions: [N] action indices or [N, action_dim] action distributions
            values: [N] optional value targets
        """
        self.states = states
        self.actions = actions
        self.values = values

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.values is not None:
            return self.states[idx], self.actions[idx], self.values[idx]
        return self.states[idx], self.actions[idx]


class ValueDataset(Dataset):
    """Dataset for value network training."""

    def __init__(self, states: torch.Tensor, values: torch.Tensor):
        """
        Initialize value dataset.

        Args:
            states: [N, state_dim] state tensors
            values: [N] value targets
        """
        self.states = states
        self.values = values

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.values[idx]


class NeuralTrainer:
    """
    Trainer for policy and value networks.

    Handles training loop, optimization, checkpointing, and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        model_name: str = "model",
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network to train
            loss_fn: Loss function
            config: Training configuration
            model_name: Name for checkpoints and logging
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.model_name = model_name

        # Move model to device
        self.model.to(config.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.training_history: list[TrainingMetrics] = []

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup wandb if requested
        if config.use_wandb:
            try:
                import wandb

                wandb.init(project=config.wandb_project or "neural-training", config=vars(config))
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self.wandb = None
        else:
            self.wandb = None

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.scheduler_type is None:
            return None
        elif self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                **self.config.scheduler_params,
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_params.get("step_size", 30),
                gamma=self.config.scheduler_params.get("gamma", 0.1),
            )
        elif self.config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_params.get("factor", 0.5),
                patience=self.config.scheduler_params.get("patience", 5),
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {self.config.scheduler_type}")

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            avg_loss: Average loss for epoch
            metrics: Additional metrics
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        aggregated_metrics: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = tuple(b.to(self.config.device) for b in batch)

            # Forward pass
            loss, metrics = self._forward_batch(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            batch_count += 1

            for key, value in metrics.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value)

            # Log batch metrics
            if batch_idx % self.config.log_every == 0:
                logger.info(f"Epoch {self.current_epoch} Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")

        # Compute average metrics
        avg_loss = total_loss / batch_count
        avg_metrics = {key: sum(values) / len(values) for key, values in aggregated_metrics.items()}

        return avg_loss, avg_metrics

    def validate(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            avg_loss: Average validation loss
            metrics: Additional metrics
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        aggregated_metrics: dict[str, list[float]] = {}

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = tuple(b.to(self.config.device) for b in batch)

                # Forward pass
                loss, metrics = self._forward_batch(batch)

                # Accumulate metrics
                total_loss += loss.item()
                batch_count += 1

                for key, value in metrics.items():
                    if key not in aggregated_metrics:
                        aggregated_metrics[key] = []
                    aggregated_metrics[key].append(value)

        # Compute average metrics
        avg_loss = total_loss / batch_count
        avg_metrics = {key: sum(values) / len(values) for key, values in aggregated_metrics.items()}

        return avg_loss, avg_metrics

    def _forward_batch(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Forward pass for a single batch.

        Handles both policy and value network formats.

        Args:
            batch: Batch of data

        Returns:
            loss: Computed loss
            metrics: Additional metrics
        """
        if isinstance(self.model, PolicyNetwork):
            if len(batch) == 3:
                states, actions, values = batch
            else:
                states, actions = batch
                values = None

            # Forward pass
            output = self.model(states, return_probs=True)

            # Compute loss
            if actions.dim() == 1:
                # Actions are indices
                loss, metrics = self.loss_fn(
                    policy_output=output, target_actions=actions, target_values=values, model=self.model
                )
            else:
                # Actions are distributions
                loss, metrics = self.loss_fn(
                    policy_output=output, target_policy=actions, target_values=values, model=self.model
                )

        elif isinstance(self.model, ValueNetwork):
            states, values = batch

            # Forward pass
            output = self.model(states)

            # Compute loss
            loss, metrics = self.loss_fn(
                predictions=output.value, targets=values, uncertainty=output.uncertainty, model=self.model
            )
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")

        return loss, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> list[TrainingMetrics]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            training_history: List of training metrics per epoch
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader)

            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                additional_metrics=train_metrics,
            )
            self.training_history.append(metrics)

            # Log metrics
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            logger.info(
                f"Epoch {epoch}/{self.config.num_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss_str}, "
                f"lr={current_lr:.6f}"
            )

            if self.wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": current_lr,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                    log_dict.update({f"val_{k}": v for k, v in val_metrics.items()})
                self.wandb.log(log_dict)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save checkpoint
            if epoch % self.config.save_every == 0 or epoch == self.config.num_epochs - 1:
                self.save_checkpoint(f"{self.model_name}_epoch_{epoch}.pt")

            # Check for improvement
            if val_loss is not None:
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(f"{self.model_name}_best.pt")
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"({self.epochs_without_improvement} epochs without improvement)"
                    )
                    break

        logger.info("Training complete")
        return self.training_history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.config.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", [])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded: {filepath}")


def train_policy_network(
    policy_net: PolicyNetwork,
    train_dataset: PolicyDataset,
    val_dataset: PolicyDataset | None = None,
    config: TrainingConfig | None = None,
) -> NeuralTrainer:
    """
    Convenience function to train policy network.

    Args:
        policy_net: Policy network to train
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        config: Training configuration

    Returns:
        trainer: Trained NeuralTrainer instance
    """
    config = config or TrainingConfig()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Create loss function
    loss_fn = PolicyLoss()

    # Create trainer
    trainer = NeuralTrainer(policy_net, loss_fn, config, model_name="policy_network")

    # Train
    trainer.train(train_loader, val_loader)

    return trainer


def train_value_network(
    value_net: ValueNetwork,
    train_dataset: ValueDataset,
    val_dataset: ValueDataset | None = None,
    config: TrainingConfig | None = None,
) -> NeuralTrainer:
    """
    Convenience function to train value network.

    Args:
        value_net: Value network to train
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        config: Training configuration

    Returns:
        trainer: Trained NeuralTrainer instance
    """
    config = config or TrainingConfig()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Create loss function
    loss_fn = ValueLoss()

    # Create trainer
    trainer = NeuralTrainer(value_net, loss_fn, config, model_name="value_network")

    # Train
    trainer.train(train_loader, val_loader)

    return trainer
