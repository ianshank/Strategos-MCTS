"""
Training script for BERT Meta-Controller with LoRA adapters.

This module provides a training pipeline for fine-tuning BERT-based meta-controllers
using Low-Rank Adaptation (LoRA) for parameter-efficient training. It supports
synthetic data generation, dataset preparation, training with HuggingFace Trainer,
and model evaluation.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Handle optional dependencies gracefully
_TRANSFORMERS_AVAILABLE = False
_DATASETS_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer,
        EvalPrediction,
        Trainer,
        TrainingArguments,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "transformers library not installed. Install it with: pip install transformers",
        ImportWarning,
        stacklevel=2,
    )
    Trainer = None
    TrainingArguments = None
    AutoTokenizer = None
    EvalPrediction = None

try:
    from datasets import Dataset

    _DATASETS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "datasets library not installed. Install it with: pip install datasets",
        ImportWarning,
        stacklevel=2,
    )
    Dataset = None

from src.agents.meta_controller.bert_controller import BERTMetaController  # noqa: E402
from src.training.data_generator import MetaControllerDataGenerator  # noqa: E402


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the training script.

    Args:
        log_level: Logging level (default: logging.INFO).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


class BERTLoRATrainer:
    """
    Trainer class for BERT Meta-Controller with LoRA adapters.

    This class provides a complete training pipeline including dataset preparation,
    training with HuggingFace Trainer, evaluation, and model persistence.

    Attributes:
        model_name: Name of the pre-trained BERT model.
        lora_r: LoRA rank parameter.
        lora_alpha: LoRA alpha scaling parameter.
        lora_dropout: LoRA dropout rate.
        lr: Learning rate for training.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        seed: Random seed for reproducibility.
        device: PyTorch device for training.
        controller: BERTMetaController instance.
        tokenizer: BERT tokenizer.
        logger: Logger instance.

    Example:
        >>> trainer = BERTLoRATrainer(
        ...     model_name="prajjwal1/bert-mini",
        ...     lora_r=4,
        ...     epochs=5
        ... )
        >>> # Prepare and train
        >>> train_dataset = trainer.prepare_dataset(train_texts, train_labels)
        >>> val_dataset = trainer.prepare_dataset(val_texts, val_labels)
        >>> results = trainer.train(train_texts, train_labels, val_texts, val_labels, "output")
    """

    def __init__(
        self,
        model_name: str = "prajjwal1/bert-mini",
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
        warmup_steps: int = 100,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        """
        Initialize the BERT LoRA trainer.

        Args:
            model_name: Pre-trained model name from HuggingFace. Defaults to "prajjwal1/bert-mini".
            lora_r: LoRA rank parameter (lower = more compression). Defaults to 4.
            lora_alpha: LoRA alpha scaling parameter. Defaults to 16.
            lora_dropout: Dropout rate for LoRA layers. Defaults to 0.1.
            lr: Learning rate for training. Defaults to 1e-3.
            batch_size: Training batch size. Defaults to 32.
            epochs: Number of training epochs. Defaults to 10.
            warmup_steps: Number of warmup steps. Defaults to 100.
            seed: Random seed for reproducibility. Defaults to 42.
            device: Device for training ('cpu', 'cuda', 'mps'). If None, auto-detects.

        Raises:
            ImportError: If required dependencies are not installed.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for BERTLoRATrainer. Install it with: pip install transformers"
            )

        if not _DATASETS_AVAILABLE:
            raise ImportError("datasets library is required for BERTLoRATrainer. Install it with: pip install datasets")

        # Setup logging
        self.logger = setup_logging()
        self.logger.info("Initializing BERTLoRATrainer")

        # Store training parameters
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.seed = seed

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Initialize BERTMetaController with LoRA enabled
        self.logger.info(f"Loading model: {model_name}")
        self.controller = BERTMetaController(
            name="BERTLoRATrainer",
            seed=seed,
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=device,
            use_lora=True,
        )

        # Store device and tokenizer for convenience
        self.device = self.controller.device
        self.tokenizer = self.controller.tokenizer

        # Log trainable parameters
        params_info = self.controller.get_trainable_parameters()
        self.logger.info(
            f"Model parameters - Total: {params_info['total_params']:,}, "
            f"Trainable: {params_info['trainable_params']:,} "
            f"({params_info['trainable_percentage']:.2f}%)"
        )

        # Store trainer instance (will be created during training)
        self._trainer: Trainer | None = None

    def prepare_dataset(
        self,
        texts: list[str],
        labels: list[int],
    ) -> Dataset:
        """
        Prepare a HuggingFace Dataset from texts and labels.

        Tokenizes all texts and creates a dataset ready for training with
        the HuggingFace Trainer.

        Args:
            texts: List of text inputs (feature descriptions).
            labels: List of integer labels (agent indices: 0=hrm, 1=trm, 2=mcts).

        Returns:
            HuggingFace Dataset with tokenized inputs and labels.

        Raises:
            ValueError: If texts and labels have different lengths.

        Example:
            >>> trainer = BERTLoRATrainer()
            >>> texts = ["HRM confidence: 0.8, TRM confidence: 0.3..."]
            >>> labels = [0]  # hrm
            >>> dataset = trainer.prepare_dataset(texts, labels)
            >>> 'input_ids' in dataset.features
            True
        """
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels must have same length, got {len(texts)} and {len(labels)}")

        self.logger.info(f"Preparing dataset with {len(texts)} samples")

        # Create initial dataset
        dataset = Dataset.from_dict({"text": texts, "labels": labels})

        # Tokenize function
        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        # Set format for PyTorch
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        self.logger.info(f"Dataset prepared with {len(tokenized_dataset)} samples")
        return tokenized_dataset

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]:
        """
        Compute evaluation metrics from predictions.

        Args:
            eval_pred: EvalPrediction object containing predictions and labels.

        Returns:
            Dictionary containing computed metrics (accuracy).

        Example:
            >>> # Called automatically by Trainer during evaluation
            >>> metrics = trainer.compute_metrics(eval_pred)
            >>> 'accuracy' in metrics
            True
        """
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Get predicted class indices
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        preds = predictions.argmax(axis=-1)

        # Calculate accuracy
        accuracy = (preds == labels).astype(float).mean()

        return {"accuracy": float(accuracy)}

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        output_dir: str,
    ) -> dict[str, Any]:
        """
        Train the BERT LoRA model.

        Creates training and validation datasets, configures the HuggingFace Trainer,
        and runs the training loop.

        Args:
            train_texts: List of training text inputs.
            train_labels: List of training labels (integer indices).
            val_texts: List of validation text inputs.
            val_labels: List of validation labels.
            output_dir: Directory to save model checkpoints and outputs.

        Returns:
            Dictionary containing training history and results.

        Example:
            >>> trainer = BERTLoRATrainer(epochs=3)
            >>> history = trainer.train(
            ...     train_texts, train_labels,
            ...     val_texts, val_labels,
            ...     "output/bert_lora"
            ... )
            >>> 'train_loss' in history
            True
        """
        self.logger.info("Starting training")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)

        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            warmup_steps=self.warmup_steps,
            learning_rate=self.lr,
            weight_decay=0.01,
            logging_dir=str(output_path / "logs"),
            logging_steps=10,
            seed=self.seed,
            report_to="none",  # Disable wandb/tensorboard by default
            save_total_limit=3,  # Keep only last 3 checkpoints
        )

        # Set model to training mode
        self.controller.model.train()

        # Create Trainer
        self._trainer = Trainer(
            model=self.controller.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        self.logger.info("Starting training loop")
        train_result = self._trainer.train()

        # Log training results
        self.logger.info("Training completed")
        self.logger.info(f"Final training loss: {train_result.training_loss:.4f}")

        # Get training history
        history = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "epochs": self.epochs,
            "final_metrics": train_result.metrics,
        }

        # Evaluate on validation set
        self.logger.info("Evaluating on validation set")
        eval_results = self._trainer.evaluate()
        history["eval_results"] = eval_results
        self.logger.info(f"Validation accuracy: {eval_results.get('eval_accuracy', 0):.4f}")

        # Set model back to evaluation mode
        self.controller.model.eval()

        return history

    def evaluate(
        self,
        test_texts: list[str],
        test_labels: list[int],
    ) -> dict[str, Any]:
        """
        Evaluate the model on a test set.

        Args:
            test_texts: List of test text inputs.
            test_labels: List of test labels (integer indices).

        Returns:
            Dictionary containing:
            - loss: Average cross-entropy loss.
            - accuracy: Classification accuracy.
            - predictions: List of predicted class indices.
            - probabilities: List of probability distributions.

        Example:
            >>> trainer = BERTLoRATrainer()
            >>> # After training...
            >>> results = trainer.evaluate(test_texts, test_labels)
            >>> 0.0 <= results['accuracy'] <= 1.0
            True
        """
        self.logger.info(f"Evaluating on {len(test_texts)} test samples")

        # Prepare test dataset
        self.prepare_dataset(test_texts, test_labels)

        # Set model to evaluation mode
        self.controller.model.eval()

        # Collect predictions
        all_predictions: list[int] = []
        all_probabilities: list[list[float]] = []
        total_loss = 0.0

        with torch.no_grad():
            for i in range(len(test_texts)):
                # Tokenize single sample
                inputs = self.tokenizer(
                    test_texts[i],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                label_tensor = torch.tensor([test_labels[i]], device=self.device)

                # Forward pass
                outputs = self.controller.model(**inputs)
                logits = outputs.logits

                # Compute loss
                loss = F.cross_entropy(logits, label_tensor)
                total_loss += loss.item()

                # Get predictions
                probs = F.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()

                all_predictions.append(pred_idx)
                all_probabilities.append(probs[0].cpu().tolist())

        # Calculate metrics
        avg_loss = total_loss / len(test_texts)
        correct = sum(1 for pred, label in zip(all_predictions, test_labels, strict=False) if pred == label)
        accuracy = correct / len(test_labels)

        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "predictions": all_predictions,
            "probabilities": all_probabilities,
        }

    def save_model(self, path: str) -> None:
        """
        Save the LoRA adapter weights to disk.

        Args:
            path: Directory path where the adapter weights will be saved.

        Example:
            >>> trainer = BERTLoRATrainer()
            >>> # After training...
            >>> trainer.save_model("models/bert_lora_adapter")
        """
        self.logger.info(f"Saving LoRA adapter weights to {path}")
        self.controller.save_model(path)
        self.logger.info("Model saved successfully")


def main() -> None:
    """
    Main function for training BERT Meta-Controller with LoRA.

    Parses command-line arguments, generates or loads dataset, trains the model,
    and saves results.
    """
    parser = argparse.ArgumentParser(description="Train BERT Meta-Controller with LoRA adapters")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="prajjwal1/bert-mini",
        help="Pre-trained model name from HuggingFace",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=4,
        help="LoRA rank parameter",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling parameter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate",
    )

    # Training arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Data arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate (if not loading from file)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to load existing dataset (JSON format)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Generate balanced dataset (equal samples per class)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/bert_lora",
        help="Directory to save model and results",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("BERT Meta-Controller LoRA Training")
    logger.info("=" * 60)

    # Log configuration
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # Initialize data generator
    data_generator = MetaControllerDataGenerator(seed=args.seed)

    # Generate or load dataset
    if args.data_path is not None:
        logger.info(f"Loading dataset from {args.data_path}")
        features_list, labels_list = data_generator.load_dataset(args.data_path)
        logger.info(f"Loaded {len(features_list)} samples")
    else:
        logger.info(f"Generating synthetic dataset with {args.num_samples} samples")
        if args.balanced:
            samples_per_class = args.num_samples // 3
            features_list, labels_list = data_generator.generate_balanced_dataset(
                num_samples_per_class=samples_per_class
            )
            logger.info(f"Generated balanced dataset with {samples_per_class} samples per class")
        else:
            features_list, labels_list = data_generator.generate_dataset(num_samples=args.num_samples)

        # Save generated dataset
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_path = output_path / "generated_dataset.json"
        data_generator.save_dataset(features_list, labels_list, str(dataset_path))
        logger.info(f"Saved generated dataset to {dataset_path}")

    # Convert to text format
    logger.info("Converting features to text format")
    texts, label_indices = data_generator.to_text_dataset(features_list, labels_list)

    # Log class distribution
    class_counts = {0: 0, 1: 0, 2: 0}
    for label in label_indices:
        class_counts[label] += 1
    logger.info("Class distribution:")
    logger.info(f"  HRM (0): {class_counts[0]} samples")
    logger.info(f"  TRM (1): {class_counts[1]} samples")
    logger.info(f"  MCTS (2): {class_counts[2]} samples")

    # Split dataset
    logger.info("Splitting dataset into train/val/test sets")
    splits = data_generator.split_dataset(texts, label_indices, train_ratio=0.7, val_ratio=0.15)

    train_texts = splits["X_train"]
    train_labels = splits["y_train"]
    val_texts = splits["X_val"]
    val_labels = splits["y_val"]
    test_texts = splits["X_test"]
    test_labels = splits["y_test"]

    logger.info(f"Train set: {len(train_texts)} samples")
    logger.info(f"Validation set: {len(val_texts)} samples")
    logger.info(f"Test set: {len(test_texts)} samples")

    # Initialize trainer
    logger.info("Initializing BERTLoRATrainer")
    trainer = BERTLoRATrainer(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )

    # Train model
    logger.info("Starting training")
    train_history = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=args.output_dir,
    )

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_texts, test_labels)

    # Save final model
    final_model_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))

    # Save training results
    results = {
        "config": vars(args),
        "train_history": train_history,
        "test_results": {
            "loss": test_results["loss"],
            "accuracy": test_results["accuracy"],
        },
        "model_params": trainer.controller.get_trainable_parameters(),
    }

    results_path = Path(args.output_dir) / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved training results to {results_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA Parameters: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"Training Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Final Training Loss: {train_history['train_loss']:.4f}")
    logger.info(f"Validation Accuracy: {train_history['eval_results'].get('eval_accuracy', 0):.4f}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test Loss: {test_results['loss']:.4f}")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
