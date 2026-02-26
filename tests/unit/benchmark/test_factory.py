"""
Tests for BenchmarkFactory.

Validates factory creation of harness, adapters, scorer,
registry, report generator, and metrics aggregator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.config.benchmark_settings import BenchmarkSettings, ScoringConfig, reset_benchmark_settings
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.scorer import LLMJudgeScorer
from src.benchmark.factory import BenchmarkFactory
from src.benchmark.reporting.metrics_aggregator import MetricsAggregator
from src.benchmark.reporting.report_generator import ReportGenerator
from src.benchmark.tasks.registry import BenchmarkTaskRegistry


@pytest.mark.unit
class TestBenchmarkFactory:
    """Test BenchmarkFactory creation methods."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()
        self.factory = BenchmarkFactory(settings=self.settings)

    def test_init_default_settings(self) -> None:
        factory = BenchmarkFactory()
        assert isinstance(factory.settings, BenchmarkSettings)

    def test_init_custom_settings(self) -> None:
        settings = BenchmarkSettings()
        factory = BenchmarkFactory(settings=settings)
        assert factory.settings is settings

    def test_init_with_llm_client(self) -> None:
        mock_client = MagicMock()
        factory = BenchmarkFactory(settings=self.settings, llm_client=mock_client)
        assert factory._llm_client is mock_client

    def test_settings_property(self) -> None:
        assert self.factory.settings is self.settings

    def test_create_scorer_disabled(self) -> None:
        self.settings._scoring = ScoringConfig(enabled=False)
        scorer = self.factory.create_scorer()
        assert isinstance(scorer, LLMJudgeScorer)
        assert not scorer._config.enabled

    def test_create_scorer_with_client(self) -> None:
        mock_client = MagicMock()
        scorer = self.factory.create_scorer(llm_client=mock_client)
        assert scorer._llm_client is mock_client

    def test_create_scorer_uses_factory_client(self) -> None:
        mock_client = MagicMock()
        factory = BenchmarkFactory(settings=self.settings, llm_client=mock_client)
        scorer = factory.create_scorer()
        assert scorer._llm_client is mock_client

    def test_create_cost_calculator(self) -> None:
        from src.benchmark.evaluation.cost_calculator import CostCalculator

        calc = self.factory.create_cost_calculator()
        assert isinstance(calc, CostCalculator)

    def test_create_adapter_factory(self) -> None:
        af = self.factory.create_adapter_factory()
        assert "langgraph_mcts" in af.get_available_systems()

    def test_create_task_registry_with_defaults(self) -> None:
        registry = self.factory.create_task_registry(load_defaults=True)
        assert isinstance(registry, BenchmarkTaskRegistry)
        assert registry.task_count > 0

    def test_create_task_registry_empty(self) -> None:
        registry = self.factory.create_task_registry(load_defaults=False)
        assert registry.task_count == 0

    def test_create_report_generator(self) -> None:
        gen = self.factory.create_report_generator()
        assert isinstance(gen, ReportGenerator)

    def test_create_metrics_aggregator(self) -> None:
        agg = self.factory.create_metrics_aggregator()
        assert isinstance(agg, MetricsAggregator)

    def test_create_adapters_all_available(self) -> None:
        adapters = self.factory.create_adapters()
        assert isinstance(adapters, list)

    def test_create_adapters_specific_system(self) -> None:
        adapters = self.factory.create_adapters(systems=["langgraph_mcts"])
        assert len(adapters) == 1
        assert adapters[0].name == "langgraph_mcts"

    def test_create_adapters_with_llm_client(self) -> None:
        mock_client = MagicMock()
        adapters = self.factory.create_adapters(
            systems=["langgraph_mcts"],
            llm_client=mock_client,
        )
        assert len(adapters) == 1
        assert adapters[0]._llm_client is mock_client

    def test_create_adapters_with_framework(self) -> None:
        mock_framework = MagicMock()
        adapters = self.factory.create_adapters(
            systems=["langgraph_mcts"],
            framework=mock_framework,
        )
        assert len(adapters) == 1
        assert adapters[0]._framework is mock_framework

    def test_create_adapters_unknown_system_skipped(self) -> None:
        adapters = self.factory.create_adapters(systems=["nonexistent"])
        assert len(adapters) == 0

    def test_create_harness(self) -> None:
        # Disable scoring to avoid LLM client creation
        self.settings._scoring = ScoringConfig(enabled=False)
        harness = self.factory.create_harness(systems=["langgraph_mcts"])
        assert isinstance(harness, EvaluationHarness)

    def test_create_harness_with_custom_scorer(self) -> None:
        mock_scorer = AsyncMock()
        harness = self.factory.create_harness(
            systems=["langgraph_mcts"],
            scorer=mock_scorer,
        )
        assert isinstance(harness, EvaluationHarness)

    def test_create_harness_no_systems(self) -> None:
        # Should still create harness even with no available systems
        self.settings._scoring = ScoringConfig(enabled=False)
        harness = self.factory.create_harness(systems=["nonexistent_system_xyz"])
        assert isinstance(harness, EvaluationHarness)

    def test_create_llm_client_calls_factory(self) -> None:
        mock_factory_instance = MagicMock()
        mock_factory_instance.create.return_value = MagicMock()

        with patch("src.framework.factories.LLMClientFactory", return_value=mock_factory_instance):
            self.factory.create_llm_client(provider="openai", model="gpt-4")
            mock_factory_instance.create.assert_called_once_with(provider="openai", model="gpt-4")
