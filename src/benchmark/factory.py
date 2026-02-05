"""
Benchmark factory for creating fully-wired benchmark pipeline components.

Follows the factory pattern from src/framework/factories.py.
Orchestrates creation of adapters, scorer, cost calculator, harness,
and report generator with all dependencies injected.
"""

from __future__ import annotations

import logging
from typing import Any

from src.benchmark.adapters.factory import BenchmarkAdapterFactory
from src.benchmark.adapters.protocol import BenchmarkSystemProtocol
from src.benchmark.config.benchmark_settings import BenchmarkSettings, get_benchmark_settings
from src.benchmark.evaluation.cost_calculator import CostCalculator
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.scorer import LLMJudgeScorer, ScorerProtocol
from src.benchmark.reporting.metrics_aggregator import MetricsAggregator
from src.benchmark.reporting.report_generator import ReportGenerator
from src.benchmark.tasks.registry import BenchmarkTaskRegistry
from src.observability.logging import get_correlation_id

logger = logging.getLogger(__name__)


class BenchmarkFactory:
    """
    Master factory for creating fully-wired benchmark pipeline components.

    Coordinates adapter creation, scorer wiring, cost calculation setup,
    and harness assembly. All dependencies are injected; no hardcoded values.

    Example:
        >>> factory = BenchmarkFactory()
        >>> harness = factory.create_harness()
        >>> results = await harness.run()
        >>> report = factory.create_report_generator().generate(results)
    """

    def __init__(
        self,
        settings: BenchmarkSettings | None = None,
        llm_client: Any | None = None,
    ) -> None:
        """
        Initialize the benchmark factory.

        Args:
            settings: Benchmark settings (uses global singleton if not provided)
            llm_client: Optional pre-configured LLM client for scoring.
                       If not provided, one will be created from settings.
        """
        self._settings = settings or get_benchmark_settings()
        self._llm_client = llm_client
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def create_llm_client(self, provider: str | None = None, model: str | None = None) -> Any:
        """
        Create an LLM client using the existing LLMClientFactory.

        Args:
            provider: LLM provider override (defaults to scoring config)
            model: Model override (defaults to scoring config)

        Returns:
            Configured LLM client instance
        """
        from src.framework.factories import LLMClientFactory

        scoring = self._settings.scoring
        target_provider = provider or scoring.provider
        target_model = model or scoring.model

        self._logger.info(
            "Creating LLM client: provider=%s, model=%s",
            target_provider,
            target_model,
            extra={"correlation_id": get_correlation_id()},
        )

        factory = LLMClientFactory()
        return factory.create(provider=target_provider, model=target_model)

    def create_scorer(self, llm_client: Any | None = None) -> LLMJudgeScorer:
        """
        Create a scorer wired to an LLM client.

        Args:
            llm_client: Optional LLM client override. If not provided,
                       uses the factory's client or creates one.

        Returns:
            Configured LLMJudgeScorer instance
        """
        client = llm_client or self._llm_client
        scoring_config = self._settings.scoring

        if client is None and scoring_config.enabled:
            try:
                client = self.create_llm_client()
            except Exception as e:
                self._logger.warning("Could not create LLM client for scorer: %s", e)
                client = None

        self._logger.info(
            "Created scorer: enabled=%s, model=%s, client_available=%s",
            scoring_config.enabled,
            scoring_config.model,
            client is not None,
        )

        return LLMJudgeScorer(config=scoring_config, llm_client=client)

    def create_cost_calculator(self) -> CostCalculator:
        """
        Create a cost calculator from settings.

        Returns:
            Configured CostCalculator instance
        """
        return CostCalculator(config=self._settings.cost)

    def create_adapter_factory(self) -> BenchmarkAdapterFactory:
        """
        Create a benchmark adapter factory.

        Returns:
            Configured BenchmarkAdapterFactory
        """
        return BenchmarkAdapterFactory(settings=self._settings)

    def create_adapters(
        self,
        systems: list[str] | None = None,
        llm_client: Any | None = None,
        framework: Any | None = None,
    ) -> list[BenchmarkSystemProtocol]:
        """
        Create benchmark adapters for specified (or all available) systems.

        Args:
            systems: Specific systems to create adapters for.
                    If None, creates all available.
            llm_client: Optional LLM client to inject into adapters.
            framework: Optional IntegratedFramework to inject into LangGraph adapter.

        Returns:
            List of configured adapter instances
        """
        adapter_factory = self.create_adapter_factory()
        client = llm_client or self._llm_client

        if systems:
            adapters: list[BenchmarkSystemProtocol] = []
            for name in systems:
                kwargs: dict[str, Any] = {}
                if client is not None:
                    kwargs["llm_client"] = client
                if framework is not None and name == "langgraph_mcts":
                    kwargs["framework"] = framework
                try:
                    adapter = adapter_factory.create(name, **kwargs)
                    adapters.append(adapter)
                except Exception as e:
                    self._logger.warning("Failed to create adapter '%s': %s", name, e)
            return adapters

        # Create all available
        all_adapters = adapter_factory.create_all_available()
        return all_adapters

    def create_task_registry(self, load_defaults: bool = True) -> BenchmarkTaskRegistry:
        """
        Create a task registry, optionally loading default tasks.

        Args:
            load_defaults: Whether to load the default task sets

        Returns:
            Configured BenchmarkTaskRegistry
        """
        registry = BenchmarkTaskRegistry()
        if load_defaults:
            registry.load_defaults()
            self._logger.info("Loaded %d default benchmark tasks", registry.task_count)
        return registry

    def create_report_generator(self) -> ReportGenerator:
        """
        Create a report generator from settings.

        Returns:
            Configured ReportGenerator
        """
        return ReportGenerator(config=self._settings.report)

    def create_metrics_aggregator(self) -> MetricsAggregator:
        """
        Create a metrics aggregator.

        Returns:
            MetricsAggregator instance
        """
        return MetricsAggregator()

    def create_harness(
        self,
        systems: list[str] | None = None,
        task_ids: list[str] | None = None,
        scorer: ScorerProtocol | None = None,
        llm_client: Any | None = None,
        framework: Any | None = None,
    ) -> EvaluationHarness:
        """
        Create a fully-wired evaluation harness.

        This is the primary entry point for running benchmarks. It assembles
        all components: adapters, registry, scorer, and cost calculator.

        Args:
            systems: Specific systems to benchmark (None = all available)
            task_ids: Specific tasks to load (None = all defaults)
            scorer: Optional custom scorer (created from settings if not provided)
            llm_client: Optional LLM client for adapters and scorer
            framework: Optional IntegratedFramework for LangGraph adapter

        Returns:
            Fully configured EvaluationHarness ready to run

        Example:
            >>> factory = BenchmarkFactory()
            >>> harness = factory.create_harness(systems=["langgraph_mcts"])
            >>> results = await harness.run()
        """
        # Create adapters
        adapters = self.create_adapters(
            systems=systems,
            llm_client=llm_client,
            framework=framework,
        )

        if not adapters:
            self._logger.warning("No adapters created; harness will have no systems to benchmark")

        # Create task registry
        registry = self.create_task_registry()

        # Create scorer
        if scorer is None:
            scorer = self.create_scorer(llm_client=llm_client)

        # Create cost calculator
        cost_calculator = self.create_cost_calculator()

        self._logger.info(
            "Created harness: %d adapters, %d tasks, scorer=%s",
            len(adapters),
            registry.task_count,
            type(scorer).__name__,
            extra={"correlation_id": get_correlation_id()},
        )

        return EvaluationHarness(
            adapters=adapters,
            registry=registry,
            scorer=scorer,
            cost_calculator=cost_calculator,
            settings=self._settings,
        )
