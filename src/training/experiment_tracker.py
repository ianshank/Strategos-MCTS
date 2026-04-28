"""
Experiment Tracking Integration Module.

Provides unified interface for:
- Braintrust experiment tracking
- Weights & Biases (W&B) logging
- Metric collection and visualization
- Model artifact versioning
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    project_name: str
    experiment_name: str
    tags: list[str] = field(default_factory=list)
    description: str = ""
    save_artifacts: bool = True
    log_frequency: int = 1  # Log every N steps


@dataclass
class TrainingMetrics:
    """Standard training metrics."""

    epoch: int
    step: int
    train_loss: float
    val_loss: float | None = None
    accuracy: float | None = None
    learning_rate: float | None = None
    timestamp: float = field(default_factory=time.time)
    custom_metrics: dict[str, float] = field(default_factory=dict)


class BraintrustTracker:
    """
    Braintrust experiment tracking integration.

    Provides:
    - Experiment initialization and management
    - Metric logging with automatic visualization
    - Hyperparameter tracking
    - Model evaluation scoring
    - Artifact versioning
    """

    def __init__(self, api_key: str | None = None, project_name: str = "mcts-neural-meta-controller"):
        """
        Initialize Braintrust tracker.

        Args:
            api_key: Braintrust API key (or from BRAINTRUST_API_KEY env var)
            project_name: Project name in Braintrust
        """
        self.api_key = api_key or os.getenv("BRAINTRUST_API_KEY")
        self.project_name = project_name
        self._experiment: Any = None
        self._experiment_id: str | None = None
        self._metrics_buffer: list[dict[str, Any]] = []
        self._initialized = False

        if not self.api_key:
            logger.warning("BRAINTRUST_API_KEY not set. Using offline mode.")
            self._offline_mode = True
        else:
            self._offline_mode = False
            self._initialize_client()

    def _initialize_client(self):
        """Initialize Braintrust client."""
        try:
            import braintrust

            braintrust.login(api_key=self.api_key)
            self._bt = braintrust
            self._initialized = True
            logger.info(f"Braintrust client initialized for project: {self.project_name}")
        except ImportError:
            logger.error("braintrust library not installed. Run: pip install braintrust")
            self._offline_mode = True
        except Exception as e:
            logger.error(f"Failed to initialize Braintrust: {e}")
            self._offline_mode = True

    def init_experiment(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Initialize a new experiment.

        Args:
            name: Experiment name (e.g., "rnn_meta_controller_v2")
            description: Experiment description
            tags: List of tags for filtering
            metadata: Additional metadata

        Returns:
            Experiment ID
        """
        if self._offline_mode:
            exp_id = f"offline_{int(time.time())}"
            logger.info(f"Created offline experiment: {exp_id}")
            self._experiment_id = exp_id
            self._experiment_config = {
                "name": name,
                "description": description,
                "tags": tags or [],
                "metadata": metadata or {},
                "start_time": datetime.now().isoformat(),
            }
            return exp_id

        try:
            self._experiment = self._bt.init(
                project=self.project_name,
                experiment=name,
            )
            self._experiment_id = self._experiment.id if self._experiment else None
            logger.info(f"Created Braintrust experiment: {name} (ID: {self._experiment_id})")
            return str(self._experiment_id)
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return self.init_experiment(name, description, tags, metadata)  # Fallback to offline

    def log_hyperparameters(self, params: dict[str, Any]):
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameters
        """
        logger.info(f"Logging hyperparameters: {params}")

        if self._offline_mode:
            self._experiment_config["hyperparameters"] = params
            return

        try:
            if self._experiment:
                # Braintrust uses metadata for hyperparameters
                self._experiment.log(
                    input="hyperparameters",
                    output=params,
                    metadata={"type": "hyperparameters"},
                )
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        timestamp: float | None = None,
    ):
        """
        Log a single metric.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
            timestamp: Optional timestamp
        """
        metric_data = {
            "name": name,
            "value": value,
            "step": step or len(self._metrics_buffer),
            "timestamp": timestamp or time.time(),
        }

        self._metrics_buffer.append(metric_data)

        if self._offline_mode:
            logger.debug(f"Metric logged (offline): {name}={value}")
            return

        try:
            if self._experiment:
                self._experiment.log(
                    input=f"metric_{name}",
                    output={"value": value},
                    scores={name: value},
                    metadata={"step": step},
                )
        except Exception as e:
            logger.error(f"Failed to log metric {name}: {e}")

    def log_training_step(self, metrics: TrainingMetrics):
        """
        Log a complete training step.

        Args:
            metrics: TrainingMetrics object
        """
        self.log_metric("train_loss", metrics.train_loss, step=metrics.step)

        if metrics.val_loss is not None:
            self.log_metric("val_loss", metrics.val_loss, step=metrics.step)

        if metrics.accuracy is not None:
            self.log_metric("accuracy", metrics.accuracy, step=metrics.step)

        if metrics.learning_rate is not None:
            self.log_metric("learning_rate", metrics.learning_rate, step=metrics.step)

        for key, value in metrics.custom_metrics.items():
            self.log_metric(key, value, step=metrics.step)

    def log_evaluation(
        self,
        input_data: Any,
        output: Any,
        expected: Any,
        scores: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log an evaluation result.

        Args:
            input_data: Input to the model
            output: Model output
            expected: Expected output
            scores: Dictionary of scores (e.g., accuracy, f1)
            metadata: Additional metadata
        """
        if self._offline_mode:
            logger.info(f"Evaluation logged (offline): scores={scores}")
            return

        try:
            if self._experiment:
                self._experiment.log(
                    input=input_data,
                    output=output,
                    expected=expected,
                    scores=scores,
                    metadata=metadata or {},
                )
        except Exception as e:
            logger.error(f"Failed to log evaluation: {e}")

    def log_artifact(self, path: str | Path, name: str | None = None):
        """
        Log a model artifact.

        Args:
            path: Path to artifact file
            name: Optional artifact name
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact not found: {path}")
            return

        logger.info(f"Logging artifact: {path}")

        if self._offline_mode:
            if "artifacts" not in self._experiment_config:
                self._experiment_config["artifacts"] = []
            artifacts: list[Any] = self._experiment_config["artifacts"]  # type: ignore[assignment]
            artifacts.append(str(path))
            return

        # Braintrust artifact logging would go here
        # For now, just log the path
        try:
            if self._experiment:
                self._experiment.log(
                    input=f"artifact_{name or path.name}",
                    output={"path": str(path), "name": name or path.name},
                    metadata={"artifact_path": str(path), "artifact_name": name or path.name},
                )
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def get_summary(self) -> dict[str, Any]:
        """
        Get experiment summary.

        Returns:
            Dictionary with experiment summary
        """
        if self._offline_mode:
            return {
                "id": self._experiment_id,
                "config": self._experiment_config,
                "metrics_count": len(self._metrics_buffer),
                "offline": True,
            }

        return {
            "id": self._experiment_id,
            "project": self.project_name,
            "metrics_count": len(self._metrics_buffer),
            "offline": False,
        }

    def end_experiment(self):
        """End the current experiment."""
        summary = self.get_summary()
        logger.info(f"Experiment ended: {summary}")

        if not self._offline_mode and self._experiment:
            try:
                # Braintrust experiments auto-close, but we'll try explicit close if available
                if hasattr(self._experiment, "close"):
                    self._experiment.close()
                elif hasattr(self._experiment, "flush"):
                    self._experiment.flush()
            except Exception as e:
                logger.error(f"Failed to end experiment: {e}")

        self._experiment = None
        self._experiment_id = None
        self._metrics_buffer = []

        return summary


class WandBTracker:
    """
    Weights & Biases experiment tracking integration.

    Provides:
    - Real-time metric visualization
    - Hyperparameter sweep management
    - Model artifact logging
    - Collaborative experiment comparison
    """

    def __init__(
        self,
        api_key: str | None = None,
        project_name: str = "mcts-neural-meta-controller",
        entity: str | None = None,
    ):
        """
        Initialize W&B tracker.

        Args:
            api_key: W&B API key (or from WANDB_API_KEY env var)
            project_name: Project name in W&B
            entity: W&B entity (team or username)
        """
        self.api_key = api_key or os.getenv("WANDB_API_KEY")
        self.project_name = project_name
        self.entity = entity
        self._run = None
        self._initialized = False
        self._offline_mode = os.getenv("WANDB_MODE") == "offline"

        if not self.api_key and not self._offline_mode:
            logger.warning("WANDB_API_KEY not set. Using offline mode.")
            self._offline_mode = True
            os.environ["WANDB_MODE"] = "offline"
        else:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize W&B client."""
        try:
            import wandb

            if self.api_key:
                wandb.login(key=self.api_key)

            self._wandb = wandb
            self._initialized = True
            logger.info(f"W&B client initialized for project: {self.project_name}")
        except ImportError:
            logger.error("wandb library not installed. Run: pip install wandb")
            self._offline_mode = True
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self._offline_mode = True

    def init_run(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ):
        """
        Initialize a new W&B run.

        Args:
            name: Run name
            config: Configuration dictionary
            tags: List of tags
            notes: Run notes/description

        Returns:
            Run object
        """
        if self._offline_mode:
            logger.info(f"W&B run initialized (offline mode): {name}")
            self._run_config = config or {}
            return None

        try:
            self._run = self._wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
            )
            logger.info(f"W&B run initialized: {name}")
            return self._run
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            self._offline_mode = True
            return None

    def log(self, metrics: dict[str, Any], step: int | None = None):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if self._offline_mode:
            logger.debug(f"W&B metrics (offline): {metrics}")
            return

        try:
            if self._run:
                self._wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log to W&B: {e}")

    def log_training_step(self, metrics: TrainingMetrics):
        """
        Log a complete training step to W&B.

        Args:
            metrics: TrainingMetrics object
        """
        log_data = {
            "epoch": metrics.epoch,
            "train_loss": metrics.train_loss,
        }

        if metrics.val_loss is not None:
            log_data["val_loss"] = metrics.val_loss

        if metrics.accuracy is not None:
            log_data["accuracy"] = metrics.accuracy

        if metrics.learning_rate is not None:
            log_data["learning_rate"] = metrics.learning_rate

        log_data.update(metrics.custom_metrics)

        self.log(log_data, step=metrics.step)

    def update_config(self, config: dict[str, Any]):
        """
        Update run configuration.

        Args:
            config: Configuration updates
        """
        if self._offline_mode:
            self._run_config.update(config)
            return

        try:
            if self._run:
                self._wandb.config.update(config)
        except Exception as e:
            logger.error(f"Failed to update W&B config: {e}")

    def watch_model(self, model, log_freq: int = 100):
        """
        Watch model gradients and parameters.

        Args:
            model: PyTorch model
            log_freq: Logging frequency
        """
        if self._offline_mode:
            return

        try:
            if self._run:
                self._wandb.watch(model, log="all", log_freq=log_freq)
        except Exception as e:
            logger.error(f"Failed to watch model: {e}")

    def log_artifact(self, path: str | Path, name: str, artifact_type: str = "model"):
        """
        Log artifact to W&B.

        Args:
            path: Path to artifact
            name: Artifact name
            artifact_type: Type of artifact (model, dataset, etc.)
        """
        if self._offline_mode:
            logger.info(f"Artifact logged (offline): {path}")
            return

        try:
            artifact = self._wandb.Artifact(name, type=artifact_type)
            artifact.add_file(str(path))
            if self._run:
                self._run.log_artifact(artifact)
            logger.info(f"Artifact logged: {name}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def finish(self):
        """Finish the W&B run."""
        if self._offline_mode:
            logger.info("W&B run finished (offline)")
            return

        try:
            if self._run:
                self._run.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")


class UnifiedExperimentTracker:
    """
    Unified experiment tracker that coordinates both Braintrust and W&B.

    Provides single interface for:
    - Dual logging to both platforms
    - Fallback handling
    - Consistent metric tracking
    """

    def __init__(
        self,
        braintrust_api_key: str | None = None,
        wandb_api_key: str | None = None,
        project_name: str = "mcts-neural-meta-controller",
    ):
        """
        Initialize unified tracker.

        Args:
            braintrust_api_key: Braintrust API key
            wandb_api_key: W&B API key
            project_name: Project name for both platforms
        """
        self.bt = BraintrustTracker(api_key=braintrust_api_key, project_name=project_name)
        self.wandb = WandBTracker(api_key=wandb_api_key, project_name=project_name)
        self.project_name = project_name

    def init_experiment(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ):
        """
        Initialize experiment on both platforms.

        Args:
            name: Experiment/run name
            config: Configuration dictionary
            description: Description
            tags: List of tags
        """
        self.bt.init_experiment(name, description, tags)
        self.wandb.init_run(name, config, tags, description)

        if config:
            self.bt.log_hyperparameters(config)

        logger.info(f"Unified experiment initialized: {name}")

    def log_metrics(self, metrics: TrainingMetrics):
        """
        Log training metrics to both platforms.

        Args:
            metrics: TrainingMetrics object
        """
        self.bt.log_training_step(metrics)
        self.wandb.log_training_step(metrics)

    def log_evaluation(
        self,
        input_data: Any,
        output: Any,
        expected: Any,
        scores: dict[str, float],
    ):
        """
        Log evaluation to Braintrust.

        Args:
            input_data: Input data
            output: Model output
            expected: Expected output
            scores: Evaluation scores
        """
        self.bt.log_evaluation(input_data, output, expected, scores)
        self.wandb.log(scores)

    def log_artifact(self, path: str | Path, name: str):
        """
        Log artifact to both platforms.

        Args:
            path: Path to artifact
            name: Artifact name
        """
        self.bt.log_artifact(path, name)
        self.wandb.log_artifact(path, name)

    def finish(self):
        """End tracking on both platforms."""
        bt_summary = self.bt.end_experiment()
        self.wandb.finish()
        logger.info("Unified experiment ended")
        return bt_summary
