"""
Braintrust integration for experiment tracking in Neural Meta-Controller training.

Provides experiment logging, metric tracking, and model versioning capabilities.
"""

import os
from datetime import datetime
from typing import Any

from src.observability.logging import get_logger

# Check if braintrust is available
try:
    import braintrust

    BRAINTRUST_AVAILABLE = True
except ImportError:
    BRAINTRUST_AVAILABLE = False
    braintrust = None

logger = get_logger(__name__)


class BraintrustTracker:
    """
    Experiment tracker using Braintrust API for neural meta-controller training.

    Provides:
    - Experiment creation and management
    - Metric logging (loss, accuracy, etc.)
    - Hyperparameter tracking
    - Model evaluation logging
    - Training run comparison
    """

    def __init__(
        self,
        project_name: str = "neural-meta-controller",
        api_key: str | None = None,
        auto_init: bool = True,
    ):
        """
        Initialize Braintrust tracker.

        Args:
            project_name: Name of the Braintrust project
            api_key: Braintrust API key (if None, reads from BRAINTRUST_API_KEY env var)
            auto_init: Whether to initialize Braintrust client immediately
        """
        self.project_name = project_name
        self._api_key = api_key or os.environ.get("BRAINTRUST_API_KEY")
        self._experiment: Any = None
        self._current_span: Any = None
        self._is_initialized = False
        self._metrics_buffer: list[dict[str, Any]] = []

        if not BRAINTRUST_AVAILABLE:
            logger.warning("braintrust package not installed. Install with: pip install braintrust")
            return

        if auto_init and self._api_key:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize Braintrust client with API key."""
        if not BRAINTRUST_AVAILABLE:
            return

        if self._api_key:
            braintrust.login(api_key=self._api_key)
            self._is_initialized = True

    @property
    def is_available(self) -> bool:
        """Check if Braintrust is available and configured."""
        return BRAINTRUST_AVAILABLE and self._is_initialized and self._api_key is not None

    def start_experiment(
        self,
        experiment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any | None:
        """
        Start a new experiment run.

        Args:
            experiment_name: Optional name for the experiment (auto-generated if None)
            metadata: Optional metadata to attach to the experiment

        Returns:
            Braintrust Experiment object or None if not available
        """
        if not self.is_available:
            return None

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"meta_controller_training_{timestamp}"

        try:
            self._experiment = braintrust.init(
                project=self.project_name,
                experiment=experiment_name,
                metadata=metadata or {},
            )
            return self._experiment
        except Exception as e:
            logger.warning("Failed to start Braintrust experiment: %s", e)
            return None

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters for the current experiment.

        Args:
            params: Dictionary of hyperparameters
        """
        if not self.is_available or self._experiment is None:
            self._metrics_buffer.append({"type": "hyperparameters", "data": params})
            return

        try:
            self._experiment.log(metadata=params)
        except Exception as e:
            logger.warning("Failed to log hyperparameters: %s", e)

    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log a single training step.

        Args:
            epoch: Current epoch number
            step: Current step/batch number
            loss: Training loss value
            metrics: Optional additional metrics (accuracy, etc.)
        """
        if not self.is_available or self._experiment is None:
            self._metrics_buffer.append(
                {
                    "type": "training_step",
                    "epoch": epoch,
                    "step": step,
                    "loss": loss,
                    "metrics": metrics or {},
                }
            )
            return

        try:
            log_data = {
                "input": {"epoch": epoch, "step": step},
                "output": {"loss": loss},
                "scores": metrics or {},
            }
            self._experiment.log(**log_data)
        except Exception as e:
            logger.warning("Failed to log training step: %s", e)

    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        train_accuracy: float | None = None,
        val_accuracy: float | None = None,
        additional_metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log summary metrics for a completed epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss for the epoch
            val_loss: Optional validation loss
            train_accuracy: Optional training accuracy
            val_accuracy: Optional validation accuracy
            additional_metrics: Optional additional metrics
        """
        if not self.is_available or self._experiment is None:
            self._metrics_buffer.append(
                {
                    "type": "epoch_summary",
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "additional_metrics": additional_metrics or {},
                }
            )
            return

        try:
            scores = {
                "train_loss": train_loss,
            }
            if val_loss is not None:
                scores["val_loss"] = val_loss
            if train_accuracy is not None:
                scores["train_accuracy"] = train_accuracy
            if val_accuracy is not None:
                scores["val_accuracy"] = val_accuracy
            if additional_metrics:
                scores.update(additional_metrics)

            self._experiment.log(
                input={"epoch": epoch},
                output={"completed": True},
                scores=scores,
            )
        except Exception as e:
            logger.warning("Failed to log epoch summary: %s", e)

    def log_evaluation(
        self,
        eval_type: str,
        predictions: list[str],
        ground_truth: list[str],
        metrics: dict[str, float],
    ) -> None:
        """
        Log model evaluation results.

        Args:
            eval_type: Type of evaluation (e.g., "validation", "test")
            predictions: Model predictions
            ground_truth: Ground truth labels
            metrics: Computed metrics (accuracy, precision, recall, f1, etc.)
        """
        if not self.is_available or self._experiment is None:
            self._metrics_buffer.append(
                {
                    "type": "evaluation",
                    "eval_type": eval_type,
                    "num_samples": len(predictions),
                    "metrics": metrics,
                }
            )
            return

        try:
            self._experiment.log(
                input={
                    "eval_type": eval_type,
                    "num_samples": len(predictions),
                },
                output={
                    "predictions_sample": predictions[:10],
                    "ground_truth_sample": ground_truth[:10],
                },
                scores=metrics,
            )
        except Exception as e:
            logger.warning("Failed to log evaluation: %s", e)

    def log_model_prediction(
        self,
        input_features: dict[str, Any],
        prediction: str,
        confidence: float,
        ground_truth: str | None = None,
    ) -> None:
        """
        Log a single model prediction for analysis.

        Args:
            input_features: Input features used for prediction
            prediction: Model's predicted agent
            confidence: Prediction confidence score
            ground_truth: Optional ground truth label
        """
        if not self.is_available or self._experiment is None:
            self._metrics_buffer.append(
                {
                    "type": "prediction",
                    "input": input_features,
                    "prediction": prediction,
                    "confidence": confidence,
                    "ground_truth": ground_truth,
                }
            )
            return

        try:
            scores = {"confidence": confidence}
            if ground_truth:
                scores["correct"] = float(prediction == ground_truth)

            self._experiment.log(
                input=input_features,
                output={"prediction": prediction},
                expected=ground_truth,
                scores=scores,
            )
        except Exception as e:
            logger.warning("Failed to log prediction: %s", e)

    def log_model_artifact(
        self,
        model_path: str,
        model_type: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a trained model artifact.

        Args:
            model_path: Path to the saved model
            model_type: Type of model (e.g., "rnn", "bert")
            metrics: Final model metrics
            metadata: Optional additional metadata
        """
        if not self.is_available or self._experiment is None:
            self._metrics_buffer.append(
                {
                    "type": "model_artifact",
                    "model_path": model_path,
                    "model_type": model_type,
                    "metrics": metrics,
                    "metadata": metadata or {},
                }
            )
            return

        try:
            self._experiment.log(
                input={
                    "model_path": model_path,
                    "model_type": model_type,
                },
                output={"saved": True},
                scores=metrics,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning("Failed to log model artifact: %s", e)

    def end_experiment(self) -> str | None:
        """
        End the current experiment and return summary URL.

        Returns:
            URL to view the experiment in Braintrust dashboard, or None
        """
        if not self.is_available or self._experiment is None:
            return None

        try:
            summary = self._experiment.summarize()
            self._experiment = None
            return summary.experiment_url if hasattr(summary, "experiment_url") else None
        except Exception as e:
            logger.warning("Failed to end experiment: %s", e)
            return None

    def get_buffered_metrics(self) -> list[dict[str, Any]]:
        """
        Get all buffered metrics (useful when Braintrust is not available).

        Returns:
            List of buffered metric dictionaries
        """
        return self._metrics_buffer.copy()

    def clear_buffer(self) -> None:
        """Clear the metrics buffer."""
        self._metrics_buffer.clear()


class BraintrustContextManager:
    """
    Context manager for Braintrust experiment tracking.

    Usage:
        with BraintrustContextManager(
            project_name="neural-meta-controller",
            experiment_name="training_run_1"
        ) as tracker:
            tracker.log_hyperparameters({"learning_rate": 0.001})
            tracker.log_epoch_summary(1, train_loss=0.5, val_loss=0.4)
    """

    def __init__(
        self,
        project_name: str = "neural-meta-controller",
        experiment_name: str | None = None,
        api_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize context manager.

        Args:
            project_name: Name of the Braintrust project
            experiment_name: Optional experiment name
            api_key: Optional API key
            metadata: Optional experiment metadata
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.api_key = api_key
        self.metadata = metadata
        self.tracker: BraintrustTracker | None = None
        self.experiment_url: str | None = None

    def __enter__(self) -> BraintrustTracker:
        """Start experiment tracking."""
        self.tracker = BraintrustTracker(
            project_name=self.project_name,
            api_key=self.api_key,
        )
        self.tracker.start_experiment(
            experiment_name=self.experiment_name,
            metadata=self.metadata,
        )
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End experiment tracking."""
        if self.tracker:
            self.experiment_url = self.tracker.end_experiment()
        return False


def create_training_tracker(
    model_type: str = "rnn",
    config: dict[str, Any] | None = None,
) -> BraintrustTracker:
    """
    Create a pre-configured tracker for meta-controller training.

    Args:
        model_type: Type of model being trained ("rnn" or "bert")
        config: Optional training configuration

    Returns:
        Configured BraintrustTracker instance
    """
    tracker = BraintrustTracker(project_name="neural-meta-controller")

    if tracker.is_available:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_type}_training_{timestamp}"

        metadata = {
            "model_type": model_type,
            "timestamp": timestamp,
        }
        if config:
            metadata.update(config)

        tracker.start_experiment(
            experiment_name=experiment_name,
            metadata=metadata,
        )

    return tracker


__all__ = [
    "BraintrustTracker",
    "BraintrustContextManager",
    "create_training_tracker",
    "BRAINTRUST_AVAILABLE",
]
