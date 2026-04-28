"""Internal submodule (split from component_factory.py)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

from src.config.settings import Settings, get_settings
from src.observability.logging import StructuredLogger, get_structured_logger

from .configs import DataLoaderConfig

if TYPE_CHECKING:
    from src.data.dataset_loader import CombinedDatasetLoader, DABStepLoader, PRIMUSLoader


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
