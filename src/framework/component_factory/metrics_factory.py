"""Internal submodule (split from component_factory.py)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

from src.config.settings import Settings, get_settings
from src.observability.logging import StructuredLogger, get_structured_logger

from .configs import MetricsConfig

if TYPE_CHECKING:
    from src.training.experiment_tracker import BraintrustTracker, UnifiedExperimentTracker, WandBTracker
    from src.training.performance_monitor import PerformanceMonitor


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
                MetricsFactory._monitor_instance = monitor

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
