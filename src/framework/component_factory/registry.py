"""Internal submodule (split from component_factory.py)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, TypeVar

from src.config.settings import Settings, get_settings
from src.observability.logging import StructuredLogger, get_structured_logger

from .data_loader_factory import DataLoaderFactory
from .metrics_factory import MetricsFactory
from .trainer_factory import TrainerFactory

if TYPE_CHECKING:

    pass

T = TypeVar("T")


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
