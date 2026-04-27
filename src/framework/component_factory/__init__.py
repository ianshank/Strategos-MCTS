"""Public API for src.framework.component_factory (split from former module).

Re-exports keep:
    from src.framework.component_factory import (
        TrainerFactory, MetricsFactory, DataLoaderFactory,
        ComponentRegistry, get_component_registry,
        TrainerConfig, MetricsConfig, DataLoaderConfig,
        ComponentProtocol, TrainerProtocol, MetricsCollectorProtocol, DataLoaderProtocol,
        MetricsCollector,
        create_trainer_factory, create_metrics_factory, create_data_loader_factory,
    )
working unchanged after the split.
"""

from .configs import DataLoaderConfig, MetricsConfig, TrainerConfig
from .data_loader_factory import DataLoaderFactory
from .metrics_factory import MetricsCollector, MetricsFactory
from .protocols import (
    ComponentProtocol,
    DataLoaderProtocol,
    MetricsCollectorProtocol,
    TrainerProtocol,
)
from .registry import (
    ComponentRegistry,
    create_data_loader_factory,
    create_metrics_factory,
    create_trainer_factory,
    get_component_registry,
)
from .trainer_factory import TrainerFactory

__all__ = [
    "ComponentProtocol",
    "ComponentRegistry",
    "DataLoaderConfig",
    "DataLoaderFactory",
    "DataLoaderProtocol",
    "MetricsCollector",
    "MetricsCollectorProtocol",
    "MetricsConfig",
    "MetricsFactory",
    "TrainerConfig",
    "TrainerFactory",
    "TrainerProtocol",
    "create_data_loader_factory",
    "create_metrics_factory",
    "create_trainer_factory",
    "get_component_registry",
]
