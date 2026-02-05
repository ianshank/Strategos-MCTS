"""
Factory for creating benchmark system adapters.

Follows the existing factory pattern from src/framework/factories.py.
Enables dependency injection and testability.
"""

from __future__ import annotations

import logging
from typing import Any

from src.benchmark.adapters.protocol import BenchmarkSystemProtocol
from src.benchmark.config.benchmark_settings import BenchmarkSettings, get_benchmark_settings

logger = logging.getLogger(__name__)

# Registry of known adapter types
ADAPTER_REGISTRY: dict[str, str] = {
    "langgraph_mcts": "src.benchmark.adapters.langgraph_adapter.LangGraphBenchmarkAdapter",
    "vertex_adk": "src.benchmark.adapters.adk_adapter.ADKBenchmarkAdapter",
}


class BenchmarkAdapterFactory:
    """
    Factory for creating benchmark system adapters.

    Creates adapters for different multi-agent systems under benchmark.
    Supports dynamic registration of new system types.

    Example:
        >>> factory = BenchmarkAdapterFactory()
        >>> lg_adapter = factory.create("langgraph_mcts")
        >>> adk_adapter = factory.create("vertex_adk")
        >>> all_adapters = factory.create_all_available()
    """

    def __init__(self, settings: BenchmarkSettings | None = None) -> None:
        """
        Initialize adapter factory.

        Args:
            settings: Benchmark settings (uses global if not provided)
        """
        self._settings = settings or get_benchmark_settings()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._custom_adapters: dict[str, type] = {}

    def register_adapter(self, name: str, adapter_class: type) -> None:
        """
        Register a custom adapter type.

        Args:
            name: Unique adapter name
            adapter_class: Adapter class (must implement BenchmarkSystemProtocol)
        """
        self._custom_adapters[name] = adapter_class
        self._logger.info("Registered custom adapter: %s", name)

    def create(self, system_name: str, **kwargs: Any) -> BenchmarkSystemProtocol:
        """
        Create a benchmark adapter for a specific system.

        Args:
            system_name: System identifier (e.g., "langgraph_mcts", "vertex_adk")
            **kwargs: Additional adapter configuration

        Returns:
            Configured adapter instance

        Raises:
            ValueError: If system_name is not recognized
        """
        # Check custom adapters first
        if system_name in self._custom_adapters:
            adapter_class = self._custom_adapters[system_name]
            adapter: BenchmarkSystemProtocol = adapter_class(settings=self._settings, **kwargs)
            return adapter

        # Check built-in registry
        if system_name not in ADAPTER_REGISTRY:
            available = list(ADAPTER_REGISTRY.keys()) + list(self._custom_adapters.keys())
            raise ValueError(f"Unknown system: '{system_name}'. Available: {available}")

        adapter = self._create_builtin(system_name, **kwargs)
        self._logger.info("Created adapter: %s (available=%s)", system_name, adapter.is_available)
        return adapter

    def create_all_available(self, **kwargs: Any) -> list[BenchmarkSystemProtocol]:
        """
        Create adapters for all available systems.

        Only returns adapters whose dependencies are installed and configured.

        Args:
            **kwargs: Additional adapter configuration

        Returns:
            List of available adapter instances
        """
        adapters: list[BenchmarkSystemProtocol] = []

        all_names = list(ADAPTER_REGISTRY.keys()) + list(self._custom_adapters.keys())
        for name in all_names:
            try:
                adapter = self.create(name, **kwargs)
                if adapter.is_available:
                    adapters.append(adapter)
                else:
                    self._logger.info("Adapter '%s' not available, skipping", name)
            except Exception as e:
                self._logger.warning("Failed to create adapter '%s': %s", name, e)

        self._logger.info("Created %d available adapters out of %d registered", len(adapters), len(all_names))
        return adapters

    def get_available_systems(self) -> list[str]:
        """Return names of all registered systems."""
        return list(ADAPTER_REGISTRY.keys()) + list(self._custom_adapters.keys())

    def _create_builtin(self, system_name: str, **kwargs: Any) -> BenchmarkSystemProtocol:
        """Create a built-in adapter by importing and instantiating."""
        if system_name == "langgraph_mcts":
            from src.benchmark.adapters.langgraph_adapter import LangGraphBenchmarkAdapter

            return LangGraphBenchmarkAdapter(settings=self._settings, **kwargs)

        elif system_name == "vertex_adk":
            from src.benchmark.adapters.adk_adapter import ADKBenchmarkAdapter

            return ADKBenchmarkAdapter(settings=self._settings, **kwargs)

        raise ValueError(f"No built-in adapter for: {system_name}")
