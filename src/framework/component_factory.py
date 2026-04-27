"""
Reusable Factory Patterns for Dynamic Component Creation.

This module provides comprehensive factory classes for creating:
- Training components (HRMTrainer, TRMTrainer, SelfPlayEvaluator, ReplayBuffer)
- Metrics collectors (MetricsCollector, ExperimentTracker, PerformanceMonitor)
- Data loaders (DABStepLoader, PRIMUSLoader, CombinedDatasetLoader)

Design Patterns Applied:
- Dependency injection via constructor
- Configuration from settings (no hardcoded values)
- Lazy initialization where appropriate
- Singleton pattern for expensive components
- Protocol-based interfaces

Best Practices 2025:
- All parameters configurable via settings
- Comprehensive type hints
- Error handling for missing dependencies
- Logging at component creation

Example:
    >>> from src.config.settings import get_settings
    >>> from src.framework.component_factory import (
    ...     TrainerFactory,
    ...     MetricsFactory,
    ...     DataLoaderFactory,
    ...     ComponentRegistry,
    ... )
    >>>
    >>> settings = get_settings()
    >>> trainer_factory = TrainerFactory(settings)
    >>> hrm_trainer = trainer_factory.create_hrm_trainer(model=my_model)
    >>>
    >>> metrics_factory = MetricsFactory(settings)
    >>> monitor = metrics_factory.create_performance_monitor()
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

from src.config.settings import Settings, get_settings
from src.observability.logging import StructuredLogger, get_structured_logger

if TYPE_CHECKING:
    import torch.optim as optim

    from src.agents.hrm_agent import HRMAgent, HRMLoss
    from src.agents.trm_agent import TRMAgent, TRMLoss
    from src.data.dataset_loader import CombinedDatasetLoader, DABStepLoader, PRIMUSLoader
    from src.framework.mcts.neural_mcts import NeuralMCTS
    from src.training.agent_trainer import HRMTrainer, SelfPlayEvaluator, TRMTrainer
    from src.training.experiment_tracker import BraintrustTracker, UnifiedExperimentTracker, WandBTracker
    from src.training.performance_monitor import PerformanceMonitor
    from src.training.replay_buffer import AugmentedReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer


# Type variable for generic factory protocols
T = TypeVar("T")


@runtime_checkable
class ComponentProtocol(Protocol):
    """Protocol for components created by factories."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize component with configuration."""
        ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for trainer components."""

    async def train_step(self, *args: Any, **kwargs: Any) -> Any:
        """Execute single training step."""
        ...

    async def train_epoch(self, data_loader: Any) -> dict[str, float]:
        """Train for one epoch."""
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection components."""

    def log_metric(self, name: str, value: float, **kwargs: Any) -> None:
        """Log a single metric."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get collected statistics."""
        ...


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for data loader components."""

    def load(self, split: str, **kwargs: Any) -> list[Any]:
        """Load dataset split."""
        ...

    def get_statistics(self) -> Any:
        """Get dataset statistics."""
        ...


@dataclass
class TrainerConfig:
    """Configuration for trainer component creation."""

    batch_size: int = 32
    num_batches: int = 10
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = False
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"

    # HRM-specific
    ponder_weight: float = 0.01
    consistency_weight: float = 0.1

    # TRM-specific
    supervision_weight_decay: float = 0.5

    # Evaluation-specific
    num_eval_games: int = 20
    eval_temperature: float = 0.0
    mcts_iterations: int = 100
    win_threshold: float = 0.55

    # Replay buffer-specific
    buffer_capacity: int = 100000
    prioritized_alpha: float = 0.6
    prioritized_beta_start: float = 0.4
    prioritized_beta_frames: int = 100000

    additional_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> TrainerConfig:
        """Create configuration from settings."""
        return cls(
            batch_size=settings.MCTS_MAX_PARALLEL_ROLLOUTS * 8,
            gradient_clip_norm=1.0,
            use_mixed_precision=False,
            device="cuda" if settings.MCTS_IMPL.value == "neural" else "cpu",
            checkpoint_dir=settings.S3_PREFIX if settings.S3_BUCKET else "checkpoints",
            mcts_iterations=settings.MCTS_ITERATIONS,
        )


@dataclass
class MetricsConfig:
    """Configuration for metrics component creation."""

    window_size: int = 100
    enable_gpu_monitoring: bool = True
    alert_threshold_ms: float = 1000.0
    log_frequency: int = 1
    save_artifacts: bool = True
    project_name: str = "langgraph-mcts"
    wandb_entity: str | None = None
    offline_mode: bool = False

    additional_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> MetricsConfig:
        """Create configuration from settings."""
        return cls(
            project_name=settings.WANDB_PROJECT,
            wandb_entity=settings.WANDB_ENTITY,
            offline_mode=settings.WANDB_MODE == "offline",
            log_frequency=1,
        )


@dataclass
class DataLoaderConfig:
    """Configuration for data loader component creation."""

    cache_dir: str | None = None
    max_samples: int | None = None
    streaming: bool = True
    include_instruct: bool = True
    batch_size: int = 32

    additional_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> DataLoaderConfig:
        """Create configuration from settings."""
        cache_dir = str(Path.home() / ".cache" / "mcts_datasets")
        return cls(
            cache_dir=cache_dir,
            batch_size=settings.MCTS_MAX_PARALLEL_ROLLOUTS * 8,
        )


class TrainerFactory:
    """
    Factory for creating training components.

    Creates:
    - HRMTrainer: Hierarchical Reasoning Model trainer
    - TRMTrainer: Task Refinement Model trainer
    - SelfPlayEvaluator: Model evaluation through self-play
    - ReplayBuffer: Experience replay buffer (uniform, prioritized, augmented)

    Example:
        >>> factory = TrainerFactory(settings=get_settings())
        >>> hrm_trainer = factory.create_hrm_trainer(
        ...     agent=my_hrm_agent,
        ...     optimizer=my_optimizer,
        ...     loss_fn=hrm_loss,
        ... )
        >>> buffer = factory.create_replay_buffer(buffer_type="prioritized")
    """

    # Singleton instance cache for expensive components
    _replay_buffer_instances: dict[str, Any] = {}
    _instance_lock = threading.Lock()

    def __init__(
        self,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        """
        Initialize trainer factory.

        Args:
            settings: Application settings (uses defaults if not provided)
            logger: Optional logger instance
            config: Optional trainer configuration (derived from settings if not provided)
        """
        self._settings = settings or get_settings()
        self._logger = logger or get_structured_logger(__name__)
        self._config = config or TrainerConfig.from_settings(self._settings)

    def create_hrm_trainer(
        self,
        agent: HRMAgent,
        optimizer: optim.Optimizer,
        loss_fn: HRMLoss,
        batch_size: int | None = None,
        num_batches: int | None = None,
        gradient_clip_norm: float | None = None,
        ponder_weight: float | None = None,
        consistency_weight: float | None = None,
        use_mixed_precision: bool | None = None,
        device: str | None = None,
        scaler: Any | None = None,
        **_kwargs: Any,
    ) -> HRMTrainer:
        """
        Create an HRM (Hierarchical Reasoning Model) trainer.

        Args:
            agent: HRM agent to train
            optimizer: PyTorch optimizer
            loss_fn: HRM loss function
            batch_size: Training batch size
            num_batches: Number of batches per epoch
            gradient_clip_norm: Maximum gradient norm for clipping
            ponder_weight: Weight for ponder cost regularization
            consistency_weight: Weight for consistency loss
            use_mixed_precision: Enable mixed precision training
            device: Device for training (cpu/cuda)
            scaler: Optional gradient scaler for mixed precision
            **kwargs: Additional configuration

        Returns:
            Configured HRMTrainer instance

        Example:
            >>> trainer = factory.create_hrm_trainer(
            ...     agent=hrm_agent,
            ...     optimizer=torch.optim.Adam(hrm_agent.parameters(), lr=0.001),
            ...     loss_fn=HRMLoss(),
            ... )
        """
        from src.training.agent_trainer import HRMTrainer, HRMTrainingConfig

        # Build config using factory defaults and overrides
        training_config = HRMTrainingConfig(
            batch_size=batch_size if batch_size is not None else self._config.batch_size,
            num_batches=num_batches if num_batches is not None else self._config.num_batches,
            gradient_clip_norm=(
                gradient_clip_norm if gradient_clip_norm is not None else self._config.gradient_clip_norm
            ),
            ponder_weight=ponder_weight if ponder_weight is not None else self._config.ponder_weight,
            consistency_weight=(
                consistency_weight if consistency_weight is not None else self._config.consistency_weight
            ),
            use_mixed_precision=(
                use_mixed_precision if use_mixed_precision is not None else self._config.use_mixed_precision
            ),
        )

        trainer_device = device if device is not None else self._config.device

        self._logger.info(
            "Creating HRM trainer",
            batch_size=training_config.batch_size,
            device=trainer_device,
            mixed_precision=training_config.use_mixed_precision,
        )

        return HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=training_config,
            device=trainer_device,
            scaler=scaler,
        )

    def create_trm_trainer(
        self,
        agent: TRMAgent,
        optimizer: optim.Optimizer,
        loss_fn: TRMLoss,
        batch_size: int | None = None,
        num_batches: int | None = None,
        gradient_clip_norm: float | None = None,
        supervision_weight_decay: float | None = None,
        use_mixed_precision: bool | None = None,
        device: str | None = None,
        scaler: Any | None = None,
        **_kwargs: Any,
    ) -> TRMTrainer:
        """
        Create a TRM (Task Refinement Model) trainer.

        Args:
            agent: TRM agent to train
            optimizer: PyTorch optimizer
            loss_fn: TRM loss function
            batch_size: Training batch size
            num_batches: Number of batches per epoch
            gradient_clip_norm: Maximum gradient norm for clipping
            supervision_weight_decay: Decay factor for intermediate supervision weights
            use_mixed_precision: Enable mixed precision training
            device: Device for training (cpu/cuda)
            scaler: Optional gradient scaler for mixed precision
            **kwargs: Additional configuration

        Returns:
            Configured TRMTrainer instance

        Example:
            >>> trainer = factory.create_trm_trainer(
            ...     agent=trm_agent,
            ...     optimizer=torch.optim.Adam(trm_agent.parameters(), lr=0.001),
            ...     loss_fn=TRMLoss(),
            ... )
        """
        from src.training.agent_trainer import TRMTrainer, TRMTrainingConfig

        training_config = TRMTrainingConfig(
            batch_size=batch_size if batch_size is not None else self._config.batch_size,
            num_batches=num_batches if num_batches is not None else self._config.num_batches,
            gradient_clip_norm=(
                gradient_clip_norm if gradient_clip_norm is not None else self._config.gradient_clip_norm
            ),
            supervision_weight_decay=(
                supervision_weight_decay
                if supervision_weight_decay is not None
                else self._config.supervision_weight_decay
            ),
            use_mixed_precision=(
                use_mixed_precision if use_mixed_precision is not None else self._config.use_mixed_precision
            ),
        )

        trainer_device = device if device is not None else self._config.device

        self._logger.info(
            "Creating TRM trainer",
            batch_size=training_config.batch_size,
            device=trainer_device,
            supervision_weight_decay=training_config.supervision_weight_decay,
        )

        return TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=training_config,
            device=trainer_device,
            scaler=scaler,
        )

    def create_self_play_evaluator(
        self,
        mcts: NeuralMCTS,
        initial_state_fn: Any,
        num_games: int | None = None,
        temperature: float | None = None,
        mcts_iterations: int | None = None,
        win_threshold: float | None = None,
        device: str | None = None,
        **_kwargs: Any,
    ) -> SelfPlayEvaluator:
        """
        Create a self-play evaluator for model comparison.

        Args:
            mcts: Neural MCTS instance
            initial_state_fn: Function to create initial game states
            num_games: Number of evaluation games
            temperature: Temperature for move selection (0 = deterministic)
            mcts_iterations: Number of MCTS iterations per move
            win_threshold: Minimum win rate to consider model better
            device: Device for evaluation (cpu/cuda)
            **kwargs: Additional configuration

        Returns:
            Configured SelfPlayEvaluator instance

        Example:
            >>> evaluator = factory.create_self_play_evaluator(
            ...     mcts=neural_mcts,
            ...     initial_state_fn=create_initial_state,
            ...     num_games=50,
            ... )
            >>> result = await evaluator.evaluate(current_model, best_model)
        """
        from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator

        eval_config = EvaluationConfig(
            num_games=num_games if num_games is not None else self._config.num_eval_games,
            temperature=temperature if temperature is not None else self._config.eval_temperature,
            mcts_iterations=mcts_iterations if mcts_iterations is not None else self._config.mcts_iterations,
            win_threshold=win_threshold if win_threshold is not None else self._config.win_threshold,
        )

        eval_device = device if device is not None else self._config.device

        self._logger.info(
            "Creating self-play evaluator",
            num_games=eval_config.num_games,
            mcts_iterations=eval_config.mcts_iterations,
            device=eval_device,
        )

        return SelfPlayEvaluator(
            mcts=mcts,
            initial_state_fn=initial_state_fn,
            config=eval_config,
            device=eval_device,
        )

    def create_replay_buffer(
        self,
        buffer_type: str = "uniform",
        capacity: int | None = None,
        alpha: float | None = None,
        beta_start: float | None = None,
        beta_frames: int | None = None,
        augmentation_fn: Any | None = None,
        use_singleton: bool = True,
        **_kwargs: Any,
    ) -> ReplayBuffer | PrioritizedReplayBuffer | AugmentedReplayBuffer:
        """
        Create an experience replay buffer.

        Supports:
        - uniform: Simple uniform sampling replay buffer
        - prioritized: Prioritized experience replay (PER)
        - augmented: Replay buffer with data augmentation

        Args:
            buffer_type: Type of buffer ("uniform", "prioritized", "augmented")
            capacity: Maximum buffer capacity
            alpha: Priority exponent for PER (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling weight for PER
            beta_frames: Number of frames to anneal beta to 1.0 for PER
            augmentation_fn: Augmentation function for augmented buffer
            use_singleton: Whether to use cached singleton instance
            **kwargs: Additional configuration

        Returns:
            Configured replay buffer instance

        Example:
            >>> # Create prioritized replay buffer
            >>> buffer = factory.create_replay_buffer(
            ...     buffer_type="prioritized",
            ...     capacity=100000,
            ...     alpha=0.6,
            ... )
        """
        from src.training.replay_buffer import AugmentedReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer

        buffer_capacity = capacity if capacity is not None else self._config.buffer_capacity
        cache_key = f"{buffer_type}_{buffer_capacity}"

        # Check for cached singleton
        if use_singleton and cache_key in self._replay_buffer_instances:
            self._logger.info(
                "Returning cached replay buffer",
                buffer_type=buffer_type,
                capacity=buffer_capacity,
            )
            return cast(
                "ReplayBuffer | PrioritizedReplayBuffer | AugmentedReplayBuffer",
                self._replay_buffer_instances[cache_key],
            )

        self._logger.info(
            "Creating replay buffer",
            buffer_type=buffer_type,
            capacity=buffer_capacity,
        )

        buffer: ReplayBuffer | PrioritizedReplayBuffer | AugmentedReplayBuffer

        if buffer_type == "uniform":
            buffer = ReplayBuffer(capacity=buffer_capacity)
        elif buffer_type == "prioritized":
            buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                alpha=alpha if alpha is not None else self._config.prioritized_alpha,
                beta_start=beta_start if beta_start is not None else self._config.prioritized_beta_start,
                beta_frames=beta_frames if beta_frames is not None else self._config.prioritized_beta_frames,
            )
        elif buffer_type == "augmented":
            buffer = AugmentedReplayBuffer(
                capacity=buffer_capacity,
                augmentation_fn=augmentation_fn,
            )
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}. Valid types: uniform, prioritized, augmented")

        # Cache singleton if requested
        if use_singleton:
            with self._instance_lock:
                self._replay_buffer_instances[cache_key] = buffer

        return buffer

    @classmethod
    def clear_singleton_cache(cls) -> None:
        """Clear the singleton instance cache for replay buffers."""
        with cls._instance_lock:
            cls._replay_buffer_instances.clear()


class MetricsFactory:
    """
    Factory for creating metrics collection components.

    Creates:
    - MetricsCollector: Unified metrics collection interface
    - ExperimentTracker: Braintrust/W&B experiment tracking
    - PerformanceMonitor: System performance monitoring

    Example:
        >>> factory = MetricsFactory(settings=get_settings())
        >>> monitor = factory.create_performance_monitor(window_size=200)
        >>> tracker = factory.create_experiment_tracker(platform="unified")
    """

    # Singleton instances for expensive trackers
    _tracker_instances: dict[str, Any] = {}
    _monitor_instance: Any = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
        config: MetricsConfig | None = None,
    ) -> None:
        """
        Initialize metrics factory.

        Args:
            settings: Application settings (uses defaults if not provided)
            logger: Optional logger instance
            config: Optional metrics configuration (derived from settings if not provided)
        """
        self._settings = settings or get_settings()
        self._logger = logger or get_structured_logger(__name__)
        self._config = config or MetricsConfig.from_settings(self._settings)

    def create_performance_monitor(
        self,
        window_size: int | None = None,
        enable_gpu_monitoring: bool | None = None,
        alert_threshold_ms: float | None = None,
        use_singleton: bool = True,
        **_kwargs: Any,
    ) -> PerformanceMonitor:
        """
        Create a performance monitoring component.

        Tracks:
        - Inference latency
        - Memory usage (CPU/GPU)
        - Training losses
        - Cache efficiency
        - Throughput statistics

        Args:
            window_size: Number of recent measurements to keep
            enable_gpu_monitoring: Whether to monitor GPU usage
            alert_threshold_ms: Threshold for slow inference alerts
            use_singleton: Whether to use cached singleton instance
            **kwargs: Additional configuration

        Returns:
            Configured PerformanceMonitor instance

        Example:
            >>> monitor = factory.create_performance_monitor(window_size=500)
            >>> monitor.log_inference(total_time_ms=45.2)
            >>> stats = monitor.get_stats()
        """
        from src.training.performance_monitor import PerformanceMonitor

        # Check for cached singleton
        if use_singleton and self._monitor_instance is not None:
            self._logger.info("Returning cached performance monitor")
            return cast("PerformanceMonitor", self._monitor_instance)

        window = window_size if window_size is not None else self._config.window_size
        gpu_monitoring = (
            enable_gpu_monitoring if enable_gpu_monitoring is not None else self._config.enable_gpu_monitoring
        )
        threshold = alert_threshold_ms if alert_threshold_ms is not None else self._config.alert_threshold_ms

        self._logger.info(
            "Creating performance monitor",
            window_size=window,
            gpu_monitoring=gpu_monitoring,
            alert_threshold_ms=threshold,
        )

        monitor = PerformanceMonitor(
            window_size=window,
            enable_gpu_monitoring=gpu_monitoring,
            alert_threshold_ms=threshold,
        )

        # Cache singleton if requested
        if use_singleton:
            with self._instance_lock:
                self._monitor_instance = monitor

        return monitor

    def create_experiment_tracker(
        self,
        platform: str = "unified",
        project_name: str | None = None,
        api_key: str | None = None,
        entity: str | None = None,
        use_singleton: bool = True,
        **_kwargs: Any,
    ) -> BraintrustTracker | WandBTracker | UnifiedExperimentTracker:
        """
        Create an experiment tracking component.

        Supports:
        - braintrust: Braintrust experiment tracking
        - wandb: Weights & Biases tracking
        - unified: Dual logging to both platforms

        Args:
            platform: Tracking platform ("braintrust", "wandb", "unified")
            project_name: Project name for tracking
            api_key: API key (uses settings if not provided)
            entity: W&B entity (team/username)
            use_singleton: Whether to use cached singleton instance
            **kwargs: Additional configuration

        Returns:
            Configured experiment tracker instance

        Example:
            >>> tracker = factory.create_experiment_tracker(platform="wandb")
            >>> tracker.init_run("experiment_v1", config={"lr": 0.001})
            >>> tracker.log({"loss": 0.5, "accuracy": 0.95})
        """
        from src.training.experiment_tracker import BraintrustTracker, UnifiedExperimentTracker, WandBTracker

        project = project_name if project_name is not None else self._config.project_name
        cache_key = f"{platform}_{project}"

        # Check for cached singleton
        if use_singleton and cache_key in self._tracker_instances:
            self._logger.info(
                "Returning cached experiment tracker",
                platform=platform,
                project=project,
            )
            return cast(
                "BraintrustTracker | WandBTracker | UnifiedExperimentTracker",
                self._tracker_instances[cache_key],
            )

        self._logger.info(
            "Creating experiment tracker",
            platform=platform,
            project=project,
        )

        tracker: BraintrustTracker | WandBTracker | UnifiedExperimentTracker

        if platform == "braintrust":
            bt_api_key = api_key or self._settings.get_braintrust_api_key()
            tracker = BraintrustTracker(
                api_key=bt_api_key,
                project_name=project,
            )
        elif platform == "wandb":
            wandb_api_key = api_key or self._settings.get_wandb_api_key()
            tracker = WandBTracker(
                api_key=wandb_api_key,
                project_name=project,
                entity=entity or self._config.wandb_entity,
            )
        elif platform == "unified":
            bt_api_key = self._settings.get_braintrust_api_key()
            wandb_api_key = self._settings.get_wandb_api_key()
            tracker = UnifiedExperimentTracker(
                braintrust_api_key=bt_api_key,
                wandb_api_key=wandb_api_key,
                project_name=project,
            )
        else:
            raise ValueError(f"Unknown platform: {platform}. Valid platforms: braintrust, wandb, unified")

        # Cache singleton if requested
        if use_singleton:
            with self._instance_lock:
                self._tracker_instances[cache_key] = tracker

        return tracker

    def create_metrics_collector(
        self,
        include_performance: bool = True,
        include_experiment_tracking: bool = True,
        tracking_platform: str = "unified",
        **kwargs: Any,
    ) -> MetricsCollector:
        """
        Create a unified metrics collector combining monitoring and tracking.

        Args:
            include_performance: Include performance monitoring
            include_experiment_tracking: Include experiment tracking
            tracking_platform: Platform for experiment tracking
            **kwargs: Additional configuration

        Returns:
            Configured MetricsCollector instance

        Example:
            >>> collector = factory.create_metrics_collector()
            >>> collector.log_training_step(epoch=1, loss=0.5, accuracy=0.95)
        """
        monitor = None
        tracker = None

        if include_performance:
            monitor = self.create_performance_monitor(**kwargs)

        if include_experiment_tracking:
            tracker = self.create_experiment_tracker(platform=tracking_platform, **kwargs)

        self._logger.info(
            "Creating metrics collector",
            include_performance=include_performance,
            include_experiment_tracking=include_experiment_tracking,
        )

        return MetricsCollector(
            performance_monitor=monitor,
            experiment_tracker=tracker,
            logger=self._logger,
        )

    @classmethod
    def clear_singleton_cache(cls) -> None:
        """Clear the singleton instance cache."""
        with cls._instance_lock:
            cls._tracker_instances.clear()
            cls._monitor_instance = None


class MetricsCollector:
    """
    Unified metrics collector combining performance monitoring and experiment tracking.

    Provides a single interface for:
    - Logging training metrics
    - Tracking performance statistics
    - Managing experiments
    """

    def __init__(
        self,
        performance_monitor: PerformanceMonitor | None = None,
        experiment_tracker: BraintrustTracker | WandBTracker | UnifiedExperimentTracker | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """
        Initialize metrics collector.

        Args:
            performance_monitor: Optional performance monitor
            experiment_tracker: Optional experiment tracker
            logger: Optional logger instance
        """
        self._monitor = performance_monitor
        self._tracker = experiment_tracker
        self._logger = logger or get_structured_logger(__name__)

    def log_metric(self, name: str, value: float, step: int | None = None, **kwargs: Any) -> None:
        """
        Log a single metric to all available collectors.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
            **kwargs: Additional context
        """
        if self._tracker is not None:
            if hasattr(self._tracker, "log_metric"):
                self._tracker.log_metric(name, value, step=step)
            elif hasattr(self._tracker, "log"):
                self._tracker.log({name: value}, step=step)

        self._logger.debug(f"Metric logged: {name}={value}", step=step)

    def log_training_step(
        self,
        epoch: int,
        loss: float,
        accuracy: float | None = None,
        learning_rate: float | None = None,
        step: int | None = None,
        **additional_metrics: float,
    ) -> None:
        """
        Log a complete training step.

        Args:
            epoch: Current epoch number
            loss: Training loss
            accuracy: Optional accuracy metric
            learning_rate: Optional learning rate
            step: Optional step number
            **additional_metrics: Additional metrics to log
        """
        from src.training.experiment_tracker import TrainingMetrics

        metrics = TrainingMetrics(
            epoch=epoch,
            step=step or epoch,
            train_loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            custom_metrics=additional_metrics,
        )

        if self._tracker is not None:
            if hasattr(self._tracker, "log_training_step"):
                self._tracker.log_training_step(metrics)
            elif hasattr(self._tracker, "log_metrics"):
                self._tracker.log_metrics(metrics)

        if self._monitor is not None:
            self._monitor.log_loss(loss, 0.0, loss)

    def log_inference(self, total_time_ms: float) -> None:
        """Log inference timing to performance monitor."""
        if self._monitor is not None:
            self._monitor.log_inference(total_time_ms)

    def log_memory(self) -> None:
        """Log current memory usage to performance monitor."""
        if self._monitor is not None:
            self._monitor.log_memory()

    def get_stats(self) -> dict[str, Any]:
        """Get collected statistics from all collectors."""
        stats: dict[str, Any] = {}

        if self._monitor is not None:
            stats["performance"] = self._monitor.get_stats()
            stats["memory"] = self._monitor.get_current_memory()

        if self._tracker is not None and hasattr(self._tracker, "get_summary"):
            stats["experiment"] = self._tracker.get_summary()

        return stats

    def print_summary(self) -> None:
        """Print formatted summary of all collected metrics."""
        if self._monitor is not None:
            self._monitor.print_summary()


class DataLoaderFactory:
    """
    Factory for creating data loader components.

    Creates:
    - DABStepLoader: Multi-step data analysis reasoning dataset
    - PRIMUSLoader: Cybersecurity domain knowledge dataset
    - CombinedDatasetLoader: Unified loader for multiple datasets

    Example:
        >>> factory = DataLoaderFactory(settings=get_settings())
        >>> dabstep = factory.create_dabstep_loader()
        >>> samples = dabstep.load(split="train", difficulty="hard")
    """

    # Singleton instances for dataset loaders
    _loader_instances: dict[str, Any] = {}
    _instance_lock = threading.Lock()

    def __init__(
        self,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
        config: DataLoaderConfig | None = None,
    ) -> None:
        """
        Initialize data loader factory.

        Args:
            settings: Application settings (uses defaults if not provided)
            logger: Optional logger instance
            config: Optional data loader configuration (derived from settings if not provided)
        """
        self._settings = settings or get_settings()
        self._logger = logger or get_structured_logger(__name__)
        self._config = config or DataLoaderConfig.from_settings(self._settings)

    def create_dabstep_loader(
        self,
        cache_dir: str | None = None,
        use_singleton: bool = True,
        **_kwargs: Any,
    ) -> DABStepLoader:
        """
        Create a DABStep dataset loader.

        DABStep contains 450+ data analysis tasks requiring sequential,
        iterative problem-solving. Perfect for training HRM/TRM agents.

        Args:
            cache_dir: Directory to cache downloaded datasets
            use_singleton: Whether to use cached singleton instance
            **kwargs: Additional configuration

        Returns:
            Configured DABStepLoader instance

        Example:
            >>> loader = factory.create_dabstep_loader()
            >>> samples = loader.load(split="train", difficulty="medium")
            >>> reasoning_tasks = loader.get_reasoning_tasks()

        License:
            CC-BY-4.0 (Creative Commons Attribution 4.0)
        """
        from src.data.dataset_loader import DABStepLoader

        cache_key = "dabstep"

        # Check for cached singleton
        if use_singleton and cache_key in self._loader_instances:
            self._logger.info("Returning cached DABStep loader")
            return cast("DABStepLoader", self._loader_instances[cache_key])

        cache = cache_dir if cache_dir is not None else self._config.cache_dir

        self._logger.info(
            "Creating DABStep loader",
            cache_dir=cache,
        )

        loader = DABStepLoader(cache_dir=cache)

        # Cache singleton if requested
        if use_singleton:
            with self._instance_lock:
                self._loader_instances[cache_key] = loader

        return loader

    def create_primus_loader(
        self,
        cache_dir: str | None = None,
        use_singleton: bool = True,
        **_kwargs: Any,
    ) -> PRIMUSLoader:
        """
        Create a PRIMUS dataset loader.

        PRIMUS contains:
        - Seed: 674,848 cybersecurity documents (190M tokens)
        - Instruct: 835 instruction-tuning samples
        - Reasoning: Self-reflection data

        Args:
            cache_dir: Directory to cache downloaded datasets
            use_singleton: Whether to use cached singleton instance
            **kwargs: Additional configuration

        Returns:
            Configured PRIMUSLoader instance

        Example:
            >>> loader = factory.create_primus_loader()
            >>> seed_samples = loader.load_seed(max_samples=10000)
            >>> mitre_samples = loader.get_mitre_attack_samples()

        License:
            ODC-BY (Open Data Commons Attribution)
        """
        from src.data.dataset_loader import PRIMUSLoader

        cache_key = "primus"

        # Check for cached singleton
        if use_singleton and cache_key in self._loader_instances:
            self._logger.info("Returning cached PRIMUS loader")
            return cast("PRIMUSLoader", self._loader_instances[cache_key])

        cache = cache_dir if cache_dir is not None else self._config.cache_dir

        self._logger.info(
            "Creating PRIMUS loader",
            cache_dir=cache,
        )

        loader = PRIMUSLoader(cache_dir=cache)

        # Cache singleton if requested
        if use_singleton:
            with self._instance_lock:
                self._loader_instances[cache_key] = loader

        return loader

    def create_combined_loader(
        self,
        cache_dir: str | None = None,
        use_singleton: bool = True,
        **_kwargs: Any,
    ) -> CombinedDatasetLoader:
        """
        Create a combined dataset loader for multiple datasets.

        Provides unified interface for loading and managing:
        - DABStep (multi-step reasoning)
        - PRIMUS (cybersecurity knowledge)
        - Custom tactical datasets

        Args:
            cache_dir: Directory to cache downloaded datasets
            use_singleton: Whether to use cached singleton instance
            **kwargs: Additional configuration

        Returns:
            Configured CombinedDatasetLoader instance

        Example:
            >>> loader = factory.create_combined_loader()
            >>> all_samples = loader.load_all(
            ...     dabstep_split="train",
            ...     primus_max_samples=10000,
            ...     include_instruct=True,
            ... )
            >>> reasoning_samples = loader.get_multi_step_reasoning_samples()
        """
        from src.data.dataset_loader import CombinedDatasetLoader

        cache_key = "combined"

        # Check for cached singleton
        if use_singleton and cache_key in self._loader_instances:
            self._logger.info("Returning cached combined loader")
            return cast("CombinedDatasetLoader", self._loader_instances[cache_key])

        cache = cache_dir if cache_dir is not None else self._config.cache_dir

        self._logger.info(
            "Creating combined dataset loader",
            cache_dir=cache,
        )

        loader = CombinedDatasetLoader(cache_dir=cache)

        # Cache singleton if requested
        if use_singleton:
            with self._instance_lock:
                self._loader_instances[cache_key] = loader

        return loader

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Convenience method to load a dataset by name.

        Args:
            dataset_name: Dataset name ("dabstep", "primus_seed", "primus_instruct", "combined")
            split: Dataset split to load
            max_samples: Maximum number of samples to load
            **kwargs: Additional configuration

        Returns:
            List of dataset samples

        Example:
            >>> samples = factory.load_dataset("dabstep", split="train")
        """
        max_samples = max_samples if max_samples is not None else self._config.max_samples

        if dataset_name == "dabstep":
            dabstep_loader = self.create_dabstep_loader(**kwargs)
            return list(dabstep_loader.load(split=split))
        elif dataset_name == "primus_seed":
            primus_loader = self.create_primus_loader(**kwargs)
            return list(primus_loader.load_seed(max_samples=max_samples))
        elif dataset_name == "primus_instruct":
            primus_instruct_loader = self.create_primus_loader(**kwargs)
            return list(primus_instruct_loader.load_instruct())
        elif dataset_name == "combined":
            combined_loader = self.create_combined_loader(**kwargs)
            return list(
                combined_loader.load_all(
                    primus_max_samples=max_samples,
                    include_instruct=self._config.include_instruct,
                )
            )
        else:
            raise ValueError(
                f"Unknown dataset_name: {dataset_name}. Valid names: dabstep, primus_seed, primus_instruct, combined"
            )

    @classmethod
    def clear_singleton_cache(cls) -> None:
        """Clear the singleton instance cache."""
        with cls._instance_lock:
            cls._loader_instances.clear()


class ComponentRegistry:
    """
    Central registry for managing component factories.

    Provides:
    - Unified access to all factories
    - Lazy initialization of factories
    - Singleton management
    - Configuration propagation

    Example:
        >>> registry = ComponentRegistry(settings=get_settings())
        >>> trainer = registry.trainers.create_hrm_trainer(agent=my_agent, ...)
        >>> monitor = registry.metrics.create_performance_monitor()
        >>> loader = registry.data_loaders.create_dabstep_loader()
    """

    _instance: ComponentRegistry | None = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        settings: Settings | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """
        Initialize component registry.

        Args:
            settings: Application settings (uses defaults if not provided)
            logger: Optional logger instance
        """
        self._settings = settings or get_settings()
        self._logger = logger or get_structured_logger(__name__)

        # Lazy-initialized factories
        self._trainer_factory: TrainerFactory | None = None
        self._metrics_factory: MetricsFactory | None = None
        self._data_loader_factory: DataLoaderFactory | None = None

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> ComponentRegistry:
        """
        Get or create the singleton registry instance.

        Args:
            settings: Application settings (uses defaults if not provided)

        Returns:
            The singleton ComponentRegistry instance
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(settings=settings)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        with cls._instance_lock:
            cls._instance = None
            TrainerFactory.clear_singleton_cache()
            MetricsFactory.clear_singleton_cache()
            DataLoaderFactory.clear_singleton_cache()

    @property
    def trainers(self) -> TrainerFactory:
        """Get the trainer factory (lazy initialization)."""
        if self._trainer_factory is None:
            self._trainer_factory = TrainerFactory(
                settings=self._settings,
                logger=self._logger,
            )
        return self._trainer_factory

    @property
    def metrics(self) -> MetricsFactory:
        """Get the metrics factory (lazy initialization)."""
        if self._metrics_factory is None:
            self._metrics_factory = MetricsFactory(
                settings=self._settings,
                logger=self._logger,
            )
        return self._metrics_factory

    @property
    def data_loaders(self) -> DataLoaderFactory:
        """Get the data loader factory (lazy initialization)."""
        if self._data_loader_factory is None:
            self._data_loader_factory = DataLoaderFactory(
                settings=self._settings,
                logger=self._logger,
            )
        return self._data_loader_factory

    def clear_caches(self) -> None:
        """Clear all singleton caches across factories."""
        TrainerFactory.clear_singleton_cache()
        MetricsFactory.clear_singleton_cache()
        DataLoaderFactory.clear_singleton_cache()
        self._logger.info("Cleared all factory singleton caches")


# Convenience functions for quick component creation


def create_trainer_factory(settings: Settings | None = None) -> TrainerFactory:
    """
    Create a trainer factory with optional settings.

    Args:
        settings: Application settings (uses defaults if not provided)

    Returns:
        Configured TrainerFactory instance

    Example:
        >>> factory = create_trainer_factory()
        >>> trainer = factory.create_hrm_trainer(agent=my_agent, ...)
    """
    return TrainerFactory(settings=settings)


def create_metrics_factory(settings: Settings | None = None) -> MetricsFactory:
    """
    Create a metrics factory with optional settings.

    Args:
        settings: Application settings (uses defaults if not provided)

    Returns:
        Configured MetricsFactory instance

    Example:
        >>> factory = create_metrics_factory()
        >>> monitor = factory.create_performance_monitor()
    """
    return MetricsFactory(settings=settings)


def create_data_loader_factory(settings: Settings | None = None) -> DataLoaderFactory:
    """
    Create a data loader factory with optional settings.

    Args:
        settings: Application settings (uses defaults if not provided)

    Returns:
        Configured DataLoaderFactory instance

    Example:
        >>> factory = create_data_loader_factory()
        >>> loader = factory.create_dabstep_loader()
    """
    return DataLoaderFactory(settings=settings)


def get_component_registry(settings: Settings | None = None) -> ComponentRegistry:
    """
    Get the global component registry singleton.

    Args:
        settings: Application settings (uses defaults if not provided)

    Returns:
        The singleton ComponentRegistry instance

    Example:
        >>> registry = get_component_registry()
        >>> trainer = registry.trainers.create_hrm_trainer(...)
        >>> monitor = registry.metrics.create_performance_monitor()
    """
    return ComponentRegistry.get_instance(settings=settings)


# Type exports for external use
__all__ = [
    # Protocols
    "ComponentProtocol",
    "TrainerProtocol",
    "MetricsCollectorProtocol",
    "DataLoaderProtocol",
    # Configurations
    "TrainerConfig",
    "MetricsConfig",
    "DataLoaderConfig",
    # Factories
    "TrainerFactory",
    "MetricsFactory",
    "DataLoaderFactory",
    # Components
    "MetricsCollector",
    # Registry
    "ComponentRegistry",
    # Convenience functions
    "create_trainer_factory",
    "create_metrics_factory",
    "create_data_loader_factory",
    "get_component_registry",
]
