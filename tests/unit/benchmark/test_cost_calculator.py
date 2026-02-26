"""
Tests for benchmark cost calculator.

Validates cost estimation, batch processing, and
per-provider pricing accuracy.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.config.benchmark_settings import CostConfig
from src.benchmark.evaluation.cost_calculator import CostBreakdown, CostCalculator
from src.benchmark.evaluation.models import BenchmarkResult


@pytest.mark.unit
class TestCostBreakdown:
    """Test CostBreakdown dataclass."""

    def test_immutability(self) -> None:
        breakdown = CostBreakdown(
            input_cost_usd=0.01,
            output_cost_usd=0.03,
            total_cost_usd=0.04,
            input_tokens=1000,
            output_tokens=1000,
            provider="openai",
            model="gpt-4",
        )
        assert breakdown.total_cost_usd == 0.04
        with pytest.raises(AttributeError):
            breakdown.total_cost_usd = 0.05  # type: ignore[misc]


@pytest.mark.unit
class TestCostCalculator:
    """Test CostCalculator operations."""

    def setup_method(self) -> None:
        self.config = CostConfig(
            openai_input_per_1k=0.01,
            openai_output_per_1k=0.03,
            anthropic_input_per_1k=0.008,
            anthropic_output_per_1k=0.024,
            gemini_pro_input_per_1k=0.00125,
            gemini_pro_output_per_1k=0.005,
            gemini_flash_input_per_1k=0.000075,
            gemini_flash_output_per_1k=0.0003,
        )
        self.calculator = CostCalculator(config=self.config)

    def test_calculate_openai(self) -> None:
        result = BenchmarkResult(input_tokens=1000, output_tokens=500)
        breakdown = self.calculator.calculate(result, "openai", "gpt-4")
        assert breakdown.input_cost_usd == pytest.approx(0.01)
        assert breakdown.output_cost_usd == pytest.approx(0.015)
        assert breakdown.total_cost_usd == pytest.approx(0.025)
        assert breakdown.provider == "openai"

    def test_calculate_anthropic(self) -> None:
        result = BenchmarkResult(input_tokens=2000, output_tokens=1000)
        breakdown = self.calculator.calculate(result, "anthropic", "claude-3")
        assert breakdown.input_cost_usd == pytest.approx(0.016)
        assert breakdown.output_cost_usd == pytest.approx(0.024)
        assert breakdown.total_cost_usd == pytest.approx(0.04)

    def test_calculate_gemini_flash(self) -> None:
        result = BenchmarkResult(input_tokens=10000, output_tokens=5000)
        breakdown = self.calculator.calculate(result, "google", "gemini-2.5-flash")
        assert breakdown.input_cost_usd == pytest.approx(0.00075)
        assert breakdown.output_cost_usd == pytest.approx(0.0015)

    def test_calculate_zero_tokens(self) -> None:
        result = BenchmarkResult(input_tokens=0, output_tokens=0)
        breakdown = self.calculator.calculate(result, "openai", "gpt-4")
        assert breakdown.total_cost_usd == 0.0

    def test_calculate_batch(self) -> None:
        results = [
            BenchmarkResult(input_tokens=1000, output_tokens=500),
            BenchmarkResult(input_tokens=2000, output_tokens=1000),
        ]
        breakdowns = self.calculator.calculate_batch(results, "openai")
        assert len(breakdowns) == 2
        assert all(b.total_cost_usd > 0 for b in breakdowns)

    def test_total_cost(self) -> None:
        results = [
            BenchmarkResult(input_tokens=1000, output_tokens=500),
            BenchmarkResult(input_tokens=1000, output_tokens=500),
        ]
        total = self.calculator.total_cost(results, "openai")
        assert total == pytest.approx(0.05)  # 2 * 0.025

    def test_apply_costs(self) -> None:
        results = [
            BenchmarkResult(system="langgraph_mcts", input_tokens=1000, output_tokens=500),
            BenchmarkResult(system="vertex_adk", input_tokens=1000, output_tokens=500),
        ]
        system_providers = {
            "langgraph_mcts": ("openai", "gpt-4"),
            "vertex_adk": ("google", "gemini-2.5-pro"),
        }

        self.calculator.apply_costs(results, system_providers)

        assert results[0].estimated_cost_usd > 0
        assert results[1].estimated_cost_usd > 0
        # OpenAI should cost more than Gemini
        assert results[0].estimated_cost_usd > results[1].estimated_cost_usd

    def test_apply_costs_unknown_system(self) -> None:
        results = [
            BenchmarkResult(system="unknown_system", input_tokens=1000, output_tokens=500),
        ]
        system_providers: dict[str, tuple[str, str]] = {}
        self.calculator.apply_costs(results, system_providers)
        assert results[0].estimated_cost_usd == 0.0

    def test_apply_costs_empty_results(self) -> None:
        results: list[BenchmarkResult] = []
        system_providers = {"langgraph_mcts": ("openai", "gpt-4")}
        returned = self.calculator.apply_costs(results, system_providers)
        assert returned == []
