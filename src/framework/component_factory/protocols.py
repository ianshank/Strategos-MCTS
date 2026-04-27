"""Internal submodule (split from component_factory.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:

    pass

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
