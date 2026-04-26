"""
Training script for the RNN Meta-Controller.

This module provides a complete training pipeline for the RNN-based meta-controller,
including data generation/loading, model training with early stopping, validation,
checkpointing, and comprehensive evaluation with per-class metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.agents.meta_controller.rnn_controller import (
    RNNMetaControllerModel,
)
from src.training.data_generator import MetaControllerDataGenerator

# Braintrust integration (optional)
try:
    from src.observability.braintrust_tracker import BraintrustTracker, create_training_tracker

    BRAINTRUST_AVAILABLE = True
except ImportError:
    BRAINTRUST_AVAILABLE = False
    BraintrustTracker = None  # type: ignore


class RNNTrainer:
    """
    Trainer class for the RNN Meta-Controller model.

    Handles the complete training pipeline including data loading, training loops,
    validation, early stopping, model checkpointing, and comprehensive evaluation.

    Attributes:
        hidden_dim: Dimension of the GRU hidden state.
        num_layers: Number of GRU layers.
        dropout: Dropout probability for regularization.
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training and evaluation.
        epochs: Maximum number of training epochs.
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
        seed: Random seed for reproducibility.
        device: PyTorch device for computation.
        model: The RNNMetaControllerModel instance.
        optimizer: Adam optimizer for training.
        criterion: CrossEntropyLoss for classification.
        logger: Logger instance for progress reporting.

    Example:
        >>> trainer = RNNTrainer(hidden_dim=64, epochs=10)
        >>> generator = MetaControllerDataGenerator(seed=42)
        >>> features, labels = generator.generate_balanced_dataset(100)
        >>> X, y = generator.to_tensor_dataset(features, labels)
        >>> splits = generator.split_dataset(X, y)
        >>> history = trainer.train(
        ...     train_data=(splits['X_train'], splits['y_train']),
        ...     val_data=(splits['X_val'], splits['y_val'])
        ... )
    """

    AGENT_NAMES = ["hrm", "trm", "mcts"]
    LABEL_TO_INDEX = {"hrm": 0, "trm": 1, "mcts": 2}
    INDEX_TO_LABEL = {0: "hrm", 1: "trm", 2: "mcts"}

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        seed: int = 42,
        device: str | None = None,
        braintrust_tracker: Any | None = None,
    ) -> None:
        """
        Initialize the RNN trainer.

        Args:
            hidden_dim: Dimension of GRU hidden state. Defaults to 64.
            num_layers: Number of stacked GRU layers. Defaults to 1.
            dropout: Dropout probability for regularization. Defaults to 0.1.
            lr: Learning rate for Adam optimizer. Defaults to 1e-3.
            batch_size: Batch size for training and evaluation. Defaults to 32.
            epochs: Maximum number of training epochs. Defaults to 10.
            early_stopping_patience: Epochs to wait for improvement before early stopping.
                Defaults to 3.
            seed: Random seed for reproducibility. Defaults to 42.
            device: Device to run training on ('cpu', 'cuda', 'mps').
                If None, auto-detects best available device.
            braintrust_tracker: Optional BraintrustTracker for experiment tracking.
        """
        # Store hyperparameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Auto-detect device if not specified
        from src.utils.device import resolve_device

        self.device = resolve_device(device)

        # Setup logging
        self._setup_logging()
        self.logger.info(f"Initializing RNNTrainer with device: {self.device}")

        # Initialize model
        self.model = RNNMetaControllerModel(
            input_dim=10,  # Fixed based on features_to_tensor output
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_agents=len(self.AGENT_NAMES),
            dropout=dropout,
        )
        self.model = self.model.to(self.device)
        self.logger.info(f"Model initialized: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.logger.info(f"Optimizer: Adam with lr={lr}")

        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        self.logger.info("Loss function: CrossEntropyLoss")

        # Braintrust experiment tracking (optional)
        self.braintrust_tracker = braintrust_tracker
        if self.braintrust_tracker and hasattr(self.braintrust_tracker, "is_available"):
            if self.braintrust_tracker.is_available:
                self.logger.info("Braintrust experiment tracking enabled")
                self.braintrust_tracker.log_hyperparameters(
                    {
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "max_epochs": epochs,
                        "early_stopping_patience": early_stopping_patience,
                        "seed": seed,
                        "device": str(self.device),
                    }
                )
            else:
                self.logger.info("Braintrust tracker provided but not available")

    def _setup_logging(self) -> None:
        """
        Setup logging configuration for the trainer.

        Creates a logger with console handler and appropriate formatting.
        """
        self.logger = logging.getLogger("RNNTrainer")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def create_dataloader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from feature and label tensors.

        Args:
            X: Feature tensor of shape (N, 10).
            y: Label tensor of shape (N,).
            batch_size: Batch size for the DataLoader. If None, uses self.batch_size.
            shuffle: Whether to shuffle the data. Defaults to True.

        Returns:
            DataLoader instance for iterating over batches.

        Example:
            >>> trainer = RNNTrainer()
            >>> X = torch.randn(100, 10)
            >>> y = torch.randint(0, 3, (100,))
            >>> loader = trainer.create_dataloader(X, y, batch_size=16)
            >>> len(loader)
            7
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Ensure tensors are on CPU for DataLoader
        if X.device != torch.device("cpu"):
            X = X.cpu()
        if y.device != torch.device("cpu"):
            y = y.cpu()

        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Use main process for data loading
            pin_memory=self.device.type == "cuda",
        )

        return loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader providing training batches.

        Returns:
            Average training loss for the epoch.

        Example:
            >>> trainer = RNNTrainer()
            >>> X = torch.randn(100, 10)
            >>> y = torch.randint(0, 3, (100,))
            >>> loader = trainer.create_dataloader(X, y)
            >>> loss = trainer.train_epoch(loader)
            >>> isinstance(loss, float)
            True
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(batch_X)

            # Compute loss
            loss = self.criterion(logits, batch_y)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return average_loss

    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """
        Evaluate the model on the validation set.

        Args:
            val_loader: DataLoader providing validation batches.

        Returns:
            Tuple of (average_loss, accuracy).
            - average_loss: Mean cross-entropy loss over validation set.
            - accuracy: Classification accuracy as a fraction [0, 1].

        Example:
            >>> trainer = RNNTrainer()
            >>> X = torch.randn(50, 10)
            >>> y = torch.randint(0, 3, (50,))
            >>> loader = trainer.create_dataloader(X, y, shuffle=False)
            >>> loss, acc = trainer.validate(loader)
            >>> 0.0 <= acc <= 1.0
            True
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move data to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                logits = self.model(batch_X)

                # Compute loss
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item()

                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        num_batches = len(val_loader)
        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return average_loss, accuracy

    def train(
        self,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Main training loop with early stopping and model checkpointing.

        Trains the model for the specified number of epochs, monitoring validation
        loss for early stopping. If save_path is provided, saves the best model
        checkpoint based on validation loss.

        Args:
            train_data: Tuple of (X_train, y_train) tensors.
            val_data: Tuple of (X_val, y_val) tensors.
            save_path: Optional path to save the best model checkpoint.

        Returns:
            Dictionary containing training history:
            - 'train_losses': List of training losses per epoch.
            - 'val_losses': List of validation losses per epoch.
            - 'val_accuracies': List of validation accuracies per epoch.
            - 'best_epoch': Epoch with best validation loss.
            - 'best_val_loss': Best validation loss achieved.
            - 'best_val_accuracy': Validation accuracy at best epoch.
            - 'stopped_early': Whether training stopped early.
            - 'total_epochs': Total number of epochs trained.

        Example:
            >>> trainer = RNNTrainer(epochs=5)
            >>> X_train = torch.randn(100, 10)
            >>> y_train = torch.randint(0, 3, (100,))
            >>> X_val = torch.randn(20, 10)
            >>> y_val = torch.randint(0, 3, (20,))
            >>> history = trainer.train((X_train, y_train), (X_val, y_val))
            >>> 'train_losses' in history
            True
            >>> len(history['train_losses']) <= 5
            True
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training samples: {train_data[0].shape[0]}")
        self.logger.info(f"Validation samples: {val_data[0].shape[0]}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Max epochs: {self.epochs}")
        self.logger.info(f"Early stopping patience: {self.early_stopping_patience}")

        # Create data loaders
        train_loader = self.create_dataloader(train_data[0], train_data[1], shuffle=True)
        val_loader = self.create_dataloader(val_data[0], val_data[1], shuffle=False)

        # Initialize tracking variables
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_accuracies: list[float] = []

        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 0
        best_model_state = None
        patience_counter = 0
        stopped_early = False

        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Log progress
            self.logger.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

            # Log to Braintrust if available
            if self.braintrust_tracker and hasattr(self.braintrust_tracker, "log_epoch_summary"):
                self.braintrust_tracker.log_epoch_summary(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                )

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                self.logger.info(f"  -> New best validation loss: {val_loss:.4f}")

                # Save checkpoint if path provided
                if save_path:
                    torch.save(best_model_state, save_path)
                    self.logger.info(f"  -> Model checkpoint saved to {save_path}")
            else:
                patience_counter += 1
                self.logger.info(f"  -> No improvement for {patience_counter} epoch(s)")

                # Check for early stopping
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                    stopped_early = True
                    break

        # Restore best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(
                f"Restored best model from epoch {best_epoch} "
                f"with val_loss={best_val_loss:.4f}, val_accuracy={best_val_accuracy:.4f}"
            )

        # Final save if path provided and not already saved
        if save_path and best_model_state is not None:
            torch.save(best_model_state, save_path)
            self.logger.info(f"Final model saved to {save_path}")

        # Compile history
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "stopped_early": stopped_early,
            "total_epochs": len(train_losses),
        }

        self.logger.info("Training completed!")
        self.logger.info(f"Best epoch: {best_epoch}")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        self.logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

        # Log final model artifact to Braintrust
        if self.braintrust_tracker and hasattr(self.braintrust_tracker, "log_model_artifact"):
            self.braintrust_tracker.log_model_artifact(
                model_path=str(save_path) if save_path else "in_memory",
                model_type="rnn",
                metrics={
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "best_epoch": float(best_epoch),
                    "total_epochs": float(len(train_losses)),
                },
            )

        return history

    def evaluate(self, test_loader: DataLoader) -> dict[str, Any]:
        """
        Comprehensive evaluation on the test set.

        Computes overall metrics and per-class precision, recall, and F1-score.

        Args:
            test_loader: DataLoader providing test batches.

        Returns:
            Dictionary containing:
            - 'loss': Average cross-entropy loss.
            - 'accuracy': Overall classification accuracy.
            - 'per_class_metrics': Dictionary with per-class metrics:
                - For each agent ('hrm', 'trm', 'mcts'):
                    - 'precision': Precision score.
                    - 'recall': Recall score.
                    - 'f1_score': F1 score.
                    - 'support': Number of samples in this class.
            - 'confusion_matrix': 3x3 confusion matrix as nested list.
            - 'total_samples': Total number of test samples.

        Example:
            >>> trainer = RNNTrainer()
            >>> X = torch.randn(50, 10)
            >>> y = torch.randint(0, 3, (50,))
            >>> loader = trainer.create_dataloader(X, y, shuffle=False)
            >>> results = trainer.evaluate(loader)
            >>> 'accuracy' in results
            True
            >>> 'per_class_metrics' in results
            True
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # Move data to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                logits = self.model(batch_X)

                # Compute loss
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

        # Calculate overall metrics
        num_batches = len(test_loader)
        average_loss = total_loss / num_batches if num_batches > 0 else 0.0

        correct = sum(p == label for p, label in zip(all_predictions, all_labels, strict=False))
        total = len(all_labels)
        accuracy = correct / total if total > 0 else 0.0

        # Calculate confusion matrix
        num_classes = len(self.AGENT_NAMES)
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
        for pred, label in zip(all_predictions, all_labels, strict=False):
            confusion_matrix[label][pred] += 1

        # Calculate per-class metrics
        per_class_metrics: dict[str, dict[str, float]] = {}

        for class_idx, agent_name in enumerate(self.AGENT_NAMES):
            # True positives: predicted as this class and actually this class
            tp = confusion_matrix[class_idx][class_idx]

            # False positives: predicted as this class but actually other class
            fp = sum(confusion_matrix[i][class_idx] for i in range(num_classes) if i != class_idx)

            # False negatives: actually this class but predicted as other class
            fn = sum(confusion_matrix[class_idx][j] for j in range(num_classes) if j != class_idx)

            # Support: total number of samples in this class
            support = sum(confusion_matrix[class_idx])

            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_metrics[agent_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }

        results = {
            "loss": average_loss,
            "accuracy": accuracy,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_matrix,
            "total_samples": total,
        }

        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Test Loss: {average_loss:.4f}")
        self.logger.info(f"  Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Total Samples: {total}")
        self.logger.info("  Per-Class Metrics:")
        for agent_name, metrics in per_class_metrics.items():
            self.logger.info(
                f"    {agent_name}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1_score']:.4f}, "
                f"Support={metrics['support']}"
            )

        return results


def main() -> None:
    """
    Main entry point for training the RNN Meta-Controller.

    Parses command-line arguments, generates or loads dataset, trains the model,
    evaluates on test set, and saves results.
    """
    parser = argparse.ArgumentParser(
        description="Train the RNN Meta-Controller for agent selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimension of GRU hidden state",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of GRU layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Data parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3000,
        help="Number of samples to generate (per class for balanced dataset)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to load existing dataset (JSON format). If not provided, generates new data.",
    )

    # Output parameters
    parser.add_argument(
        "--save_path",
        type=str,
        default="rnn_meta_controller.pt",
        help="Path to save the trained model",
    )

    # Experiment tracking
    parser.add_argument(
        "--use_braintrust",
        action="store_true",
        help="Enable Braintrust experiment tracking",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment name for Braintrust (auto-generated if not provided)",
    )

    args = parser.parse_args()

    # Setup logging for main
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("train_rnn")

    logger.info("=" * 60)
    logger.info("RNN Meta-Controller Training")
    logger.info("=" * 60)

    # Print configuration
    logger.info("Configuration:")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"  {arg_name}: {arg_value}")
    logger.info("")

    try:
        # Initialize data generator
        generator = MetaControllerDataGenerator(seed=args.seed)

        # Load or generate dataset
        if args.data_path and Path(args.data_path).exists():
            logger.info(f"Loading dataset from {args.data_path}...")
            features_list, labels_list = generator.load_dataset(args.data_path)
            logger.info(f"Loaded {len(features_list)} samples")
        else:
            logger.info(f"Generating balanced dataset with {args.num_samples} samples per class...")
            features_list, labels_list = generator.generate_balanced_dataset(num_samples_per_class=args.num_samples)
            total_samples = len(features_list)
            logger.info(f"Generated {total_samples} total samples")

            # Optionally save generated dataset
            if args.data_path:
                logger.info(f"Saving generated dataset to {args.data_path}...")
                generator.save_dataset(features_list, labels_list, args.data_path)

        # Convert to tensors
        logger.info("Converting dataset to tensors...")
        X, y = generator.to_tensor_dataset(features_list, labels_list)
        logger.info(f"Feature tensor shape: {X.shape}")
        logger.info(f"Label tensor shape: {y.shape}")

        # Split dataset
        logger.info("Splitting dataset into train/val/test (70%/15%/15%)...")
        splits = generator.split_dataset(X, y, train_ratio=0.7, val_ratio=0.15)

        logger.info(f"Training set size: {splits['X_train'].shape[0]}")
        logger.info(f"Validation set size: {splits['X_val'].shape[0]}")
        logger.info(f"Test set size: {splits['X_test'].shape[0]}")
        logger.info("")

        # Initialize Braintrust tracker if enabled
        braintrust_tracker = None
        if args.use_braintrust and BRAINTRUST_AVAILABLE:
            logger.info("Initializing Braintrust experiment tracker...")
            braintrust_tracker = create_training_tracker(
                model_type="rnn",
                config={
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "seed": args.seed,
                    "num_samples": args.num_samples,
                },
            )
            if braintrust_tracker.is_available:
                logger.info("Braintrust experiment tracking enabled")
            else:
                logger.info("Braintrust not available (check API key)")
        elif args.use_braintrust and not BRAINTRUST_AVAILABLE:
            logger.warning("Braintrust requested but not installed. Install with: pip install braintrust")

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = RNNTrainer(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping_patience=args.patience,
            seed=args.seed,
            braintrust_tracker=braintrust_tracker,
        )
        logger.info("")

        # Train model
        logger.info("Starting training...")
        logger.info("-" * 60)
        history = trainer.train(
            train_data=(splits["X_train"], splits["y_train"]),
            val_data=(splits["X_val"], splits["y_val"]),
            save_path=args.save_path,
        )
        logger.info("-" * 60)
        logger.info("")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        logger.info("-" * 60)
        test_loader = trainer.create_dataloader(splits["X_test"], splits["y_test"], shuffle=False)
        test_results = trainer.evaluate(test_loader)
        logger.info("-" * 60)
        logger.info("")

        # Save training history
        history_path = Path(args.save_path).with_suffix(".history.json")
        logger.info(f"Saving training history to {history_path}...")

        # Combine history and test results
        full_results = {
            "config": {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "seed": args.seed,
                "num_samples": args.num_samples,
            },
            "training_history": history,
            "test_results": test_results,
        }

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        logger.info(f"Training history saved to {history_path}")
        logger.info("")

        # Print final summary
        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {args.save_path}")
        logger.info(f"History saved to: {history_path}")
        logger.info(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
        logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test loss: {test_results['loss']:.4f}")

        if history["stopped_early"]:
            logger.info(f"Training stopped early at epoch {history['total_epochs']}")
        else:
            logger.info(f"Training completed all {history['total_epochs']} epochs")

        logger.info("")
        logger.info("Per-class test performance:")
        for agent_name, metrics in test_results["per_class_metrics"].items():
            logger.info(
                f"  {agent_name}: F1={metrics['f1_score']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}"
            )

        # End Braintrust experiment
        if braintrust_tracker and hasattr(braintrust_tracker, "end_experiment"):
            experiment_url = braintrust_tracker.end_experiment()
            if experiment_url:
                logger.info(f"Braintrust experiment URL: {experiment_url}")

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
