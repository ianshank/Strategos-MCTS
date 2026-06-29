#!/usr/bin/env python3
"""
Train Meta-Controllers with Assembly Features (Story 2.5).

This script trains neural meta-controllers using assembly-augmented training data,
compares performance with baseline, and tracks assembly feature importance.

Usage:
    # Train with default settings
    python scripts/train_assembly_meta_controller.py

    # Train with custom data
    python scripts/train_assembly_meta_controller.py \\
        --data-path data/training_with_assembly.json \\
        --epochs 50 \\
        --batch-size 128

    # Compare baseline vs assembly-augmented
    python scripts/train_assembly_meta_controller.py \\
        --compare-baseline \\
        --save-comparison results/comparison.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class AssemblyMetaControllerDataset(Dataset):
    """Dataset for assembly-augmented meta-controller training."""

    def __init__(
        self,
        data_path: str,
        include_assembly: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to training data JSON
            include_assembly: Whether to include assembly features
            normalize: Whether to normalize features
        """
        self.include_assembly = include_assembly

        # Load data
        with open(data_path) as f:
            self.samples = json.load(f)

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

        # Agent to index mapping
        self.agent_map = {"hrm": 0, "trm": 1, "mcts": 2}

        # Compute normalization stats if needed
        if normalize:
            self._compute_normalization_stats()
        else:
            self.feature_means = None
            self.feature_stds = None

    def _compute_normalization_stats(self):
        """Compute mean and std for feature normalization."""
        # Extract all features
        all_features = []

        for sample in self.samples:
            features = self._extract_features(sample, normalize=False)
            all_features.append(features)

        all_features = np.array(all_features)

        self.feature_means = all_features.mean(axis=0)
        self.feature_stds = all_features.std(axis=0) + 1e-8  # Avoid division by zero

        logger.info(f"Feature normalization: mean={self.feature_means.mean():.3f}, std={self.feature_stds.mean():.3f}")

    def _extract_features(self, sample: dict[str, Any], normalize: bool = True) -> np.ndarray:
        """
        Extract feature vector from sample.

        Args:
            sample: Training sample
            normalize: Whether to normalize

        Returns:
            Feature vector
        """
        mc_features = sample["features"]

        # Standard meta-controller features
        features = [
            mc_features["hrm_confidence"],
            mc_features["trm_confidence"],
            mc_features["mcts_value"],
            mc_features["consensus_score"],
            float(mc_features["last_agent"] == "hrm"),
            float(mc_features["last_agent"] == "trm"),
            float(mc_features["last_agent"] == "mcts"),
            mc_features["iteration"] / 10.0,  # Normalize
            mc_features["query_length"] / 200.0,  # Normalize
            float(mc_features["has_rag_context"]),
        ]

        # Add assembly features if requested
        if self.include_assembly:
            assembly = sample["assembly_features"]
            features.extend(
                [
                    assembly["assembly_index"] / 30.0,  # Normalize to ~[0, 1]
                    assembly["copy_number"] / 20.0,
                    assembly["decomposability_score"],  # Already [0, 1]
                    assembly["graph_depth"] / 10.0,
                    assembly["constraint_count"] / 20.0,
                    assembly["concept_count"] / 30.0,
                    assembly["technical_complexity"],  # Already [0, 1]
                    assembly["normalized_assembly_index"],  # Already [0, 1]
                ]
            )

        feature_array = np.array(features, dtype=np.float32)

        # Normalize if stats available
        if normalize and self.feature_means is not None:
            feature_array = (feature_array - self.feature_means) / self.feature_stds

        return feature_array

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Extract features
        features = self._extract_features(sample)

        # Get label
        agent = sample["ground_truth_agent"]
        label = self.agent_map[agent]

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "query": sample["query"],
            "reasoning": sample["reasoning"],
        }

    def get_feature_dim(self) -> int:
        """Get feature dimensionality."""
        return 10 + (8 if self.include_assembly else 0)


class AssemblyAwareRouter(nn.Module):
    """Neural router with assembly feature support."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64, 32],  # noqa: B006
        num_agents: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize router.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            num_agents: Number of agents
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_agents = num_agents

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_agents))

        self.network = nn.Sequential(*layers)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._last_hidden = None

        logger.info(f"AssemblyAwareRouter: {input_dim} -> {hidden_dims} -> {num_agents}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Dictionary with logits, probabilities, and confidence
        """
        # Pass through hidden layers
        hidden = x
        for i in range(len(self.network) - 1):
            hidden = self.network[i](hidden)

        self._last_hidden = hidden

        # Output logits
        logits = self.network[-1](hidden)

        # Confidence
        confidence = self.confidence_head(hidden).squeeze(-1)

        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "confidence": confidence,
            "hidden": hidden,
        }

    def get_feature_importance(self) -> torch.Tensor:
        """
        Compute feature importance using gradient-based attribution.

        Returns:
            Feature importance scores
        """
        # Average absolute weights from first layer
        first_layer = self.network[0]
        importance = first_layer.weight.abs().mean(dim=0)

        return importance


class MetaControllerTrainer:
    """Trainer for assembly-augmented meta-controllers."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
    ):
        """
        Initialize trainer.

        Args:
            model: Router model to train
            device: Device for training
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            outputs = self.model(features)
            logits = outputs["logits"]

            # Classification loss
            loss = F.cross_entropy(logits, labels)

            # Add confidence calibration loss
            confidences = outputs["confidence"]
            accuracy_per_sample = (logits.argmax(dim=-1) == labels).float()
            calibration_loss = F.mse_loss(confidences, accuracy_per_sample)

            total_loss_value = loss + 0.1 * calibration_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_value.backward()
            self.optimizer.step()

            total_loss += total_loss_value.item()

            # Accuracy
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model.

        Args:
            dataloader: Validation data loader

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        confidences = []
        calibration_errors = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(features)
                logits = outputs["logits"]

                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Confidence calibration
                conf = outputs["confidence"]
                acc = (predictions == labels).float()
                calibration_errors.extend((conf - acc).abs().cpu().tolist())
                confidences.extend(conf.cpu().tolist())

        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total,
            "avg_confidence": np.mean(confidences),
            "calibration_error": np.mean(calibration_errors),
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
    ) -> dict[str, list[float]]:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience

        Returns:
            Training history
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
            )

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return self.history


def compare_models(
    baseline_model: nn.Module,
    assembly_model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Compare baseline and assembly-augmented models.

    Args:
        baseline_model: Model without assembly features
        assembly_model: Model with assembly features
        test_loader: Test data loader
        device: Device

    Returns:
        Comparison results
    """
    logger.info("Comparing baseline vs. assembly-augmented models...")

    # Evaluate both models
    baseline_trainer = MetaControllerTrainer(baseline_model, device)
    assembly_trainer = MetaControllerTrainer(assembly_model, device)

    baseline_metrics = baseline_trainer.evaluate(test_loader)
    assembly_metrics = assembly_trainer.evaluate(test_loader)

    # Compute improvement
    improvement = {
        "accuracy": assembly_metrics["accuracy"] - baseline_metrics["accuracy"],
        "calibration_error": baseline_metrics["calibration_error"] - assembly_metrics["calibration_error"],
    }

    results = {
        "baseline": baseline_metrics,
        "assembly": assembly_metrics,
        "improvement": improvement,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info("\n" + "=" * 70)
    logger.info("Model Comparison Results")
    logger.info("=" * 70)
    logger.info("Baseline:")
    logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"  Calibration Error: {baseline_metrics['calibration_error']:.4f}")
    logger.info("")
    logger.info("Assembly-Augmented:")
    logger.info(f"  Accuracy: {assembly_metrics['accuracy']:.4f}")
    logger.info(f"  Calibration Error: {assembly_metrics['calibration_error']:.4f}")
    logger.info("")
    logger.info("Improvement:")
    logger.info(f"  Accuracy: {improvement['accuracy']:+.4f} ({improvement['accuracy']*100:+.2f}%)")
    logger.info(f"  Calibration Error: {improvement['calibration_error']:+.4f}")
    logger.info("=" * 70)

    return results


def analyze_feature_importance(
    model: AssemblyAwareRouter,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Analyze feature importance.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        Feature importance scores
    """
    importance = model.get_feature_importance().cpu().detach().numpy()

    # Normalize to sum to 1
    importance = importance / importance.sum()

    importance_dict = {name: float(score) for name, score in zip(feature_names, importance)}

    # Sort by importance
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("Feature Importance (Top 10)")
    logger.info("=" * 70)
    for i, (feature, score) in enumerate(sorted_importance[:10]):
        logger.info(f"{i + 1:2d}. {feature:35s}: {score:.4f} ({score*100:.2f}%)")
    logger.info("=" * 70)

    return importance_dict


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train assembly-augmented meta-controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training_with_assembly.json",
        help="Path to training data (default: data/training_with_assembly.json)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    # Comparison
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline (no assembly features)",
    )
    parser.add_argument(
        "--save-comparison",
        type=str,
        help="Path to save comparison results",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/meta_controller",
        help="Output directory for models (default: models/meta_controller)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 70)
    logger.info("Assembly Meta-Controller Training")
    logger.info("=" * 70)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 70)

    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = AssemblyMetaControllerDataset(args.data_path, include_assembly=True)

    # Split into train/val/test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create assembly-augmented model
    input_dim = full_dataset.get_feature_dim()
    assembly_model = AssemblyAwareRouter(input_dim=input_dim)

    logger.info(f"\nTraining assembly-augmented model (input_dim={input_dim})...")
    assembly_trainer = MetaControllerTrainer(
        assembly_model,
        device=args.device,
        learning_rate=args.learning_rate,
    )

    history = assembly_trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
    )

    # Evaluate on test set
    test_metrics = assembly_trainer.evaluate(test_loader)

    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Calibration Error: {test_metrics['calibration_error']:.4f}")

    # Feature importance analysis
    feature_names = [
        "hrm_confidence",
        "trm_confidence",
        "mcts_value",
        "consensus_score",
        "last_agent_hrm",
        "last_agent_trm",
        "last_agent_mcts",
        "iteration",
        "query_length",
        "has_rag_context",
        "assembly_index",
        "copy_number",
        "decomposability_score",
        "graph_depth",
        "constraint_count",
        "concept_count",
        "technical_complexity",
        "normalized_assembly_index",
    ]

    importance_scores = analyze_feature_importance(assembly_model, feature_names)

    # Compare with baseline if requested
    comparison_results = None
    if args.compare_baseline:
        logger.info("\nTraining baseline model (without assembly features)...")

        # Create baseline dataset and loaders
        baseline_dataset = AssemblyMetaControllerDataset(args.data_path, include_assembly=False)
        baseline_train, baseline_val, baseline_test = random_split(
            baseline_dataset,
            [train_size, val_size, test_size],
        )

        baseline_train_loader = DataLoader(baseline_train, batch_size=args.batch_size, shuffle=True)
        baseline_val_loader = DataLoader(baseline_val, batch_size=args.batch_size)
        baseline_test_loader = DataLoader(baseline_test, batch_size=args.batch_size)  # noqa: F841

        # Train baseline
        baseline_input_dim = baseline_dataset.get_feature_dim()
        baseline_model = AssemblyAwareRouter(input_dim=baseline_input_dim)

        baseline_trainer = MetaControllerTrainer(
            baseline_model,
            device=args.device,
            learning_rate=args.learning_rate,
        )

        baseline_trainer.train(
            baseline_train_loader,
            baseline_val_loader,
            num_epochs=args.epochs,
        )

        # Compare
        comparison_results = compare_models(
            baseline_model,
            assembly_model,
            test_loader,
            args.device,
        )

        # Save comparison if requested
        if args.save_comparison:
            comparison_path = Path(args.save_comparison)
            comparison_path.parent.mkdir(parents=True, exist_ok=True)

            with open(comparison_path, "w") as f:
                json.dump(comparison_results, f, indent=2)

            logger.info(f"\nSaved comparison results to {comparison_path}")

    # Save models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assembly_model_path = output_dir / "assembly_meta_controller.pt"
    torch.save(
        {
            "model_state_dict": assembly_model.state_dict(),
            "input_dim": input_dim,
            "test_metrics": test_metrics,
            "feature_importance": importance_scores,
            "history": history,
        },
        assembly_model_path,
    )

    logger.info(f"\n✓ Saved assembly model to {assembly_model_path}")

    if args.compare_baseline and baseline_model is not None:
        baseline_model_path = output_dir / "baseline_meta_controller.pt"
        torch.save(
            {
                "model_state_dict": baseline_model.state_dict(),
                "input_dim": baseline_input_dim,
            },
            baseline_model_path,
        )

        logger.info(f"✓ Saved baseline model to {baseline_model_path}")

    logger.info("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
