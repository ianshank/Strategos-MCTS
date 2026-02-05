"""
Cost calculator for benchmark evaluation.

Estimates USD cost per benchmark run based on token usage
and configurable provider pricing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.benchmark.config.benchmark_settings import CostConfig
from src.benchmark.evaluation.models import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed cost breakdown for a benchmark result."""

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    input_tokens: int
    output_tokens: int
    provider: str
    model: str


class CostCalculator:
    """
    Calculates estimated costs for benchmark runs.

    Uses configurable per-provider pricing from CostConfig.
    All rates are per 1K tokens.

    Example:
        >>> calculator = CostCalculator(cost_config)
        >>> breakdown = calculator.calculate(result, provider="openai", model="gpt-4")
        >>> print(f"Total cost: ${breakdown.total_cost_usd:.4f}")
    """

    def __init__(self, config: CostConfig) -> None:
        """
        Initialize cost calculator.

        Args:
            config: Cost configuration with per-provider pricing
        """
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate(
        self,
        result: BenchmarkResult,
        provider: str,
        model: str = "",
    ) -> CostBreakdown:
        """
        Calculate cost for a single benchmark result.

        Args:
            result: Benchmark result with token counts
            provider: LLM provider name
            model: Model identifier (used for rate selection)

        Returns:
            CostBreakdown with detailed cost information
        """
        input_rate, output_rate = self._config.get_rates(provider, model)

        input_cost = (result.input_tokens / 1000.0) * input_rate
        output_cost = (result.output_tokens / 1000.0) * output_rate
        total_cost = input_cost + output_cost

        breakdown = CostBreakdown(
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=total_cost,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            provider=provider,
            model=model,
        )

        self._logger.debug(
            "Cost for %s/%s: $%.4f (in: %d tokens, out: %d tokens)",
            result.task_id,
            result.system,
            total_cost,
            result.input_tokens,
            result.output_tokens,
        )

        return breakdown

    def calculate_batch(
        self,
        results: list[BenchmarkResult],
        provider: str,
        model: str = "",
    ) -> list[CostBreakdown]:
        """Calculate costs for multiple results."""
        return [self.calculate(r, provider, model) for r in results]

    def total_cost(
        self,
        results: list[BenchmarkResult],
        provider: str,
        model: str = "",
    ) -> float:
        """Calculate total cost across all results."""
        breakdowns = self.calculate_batch(results, provider, model)
        return sum(b.total_cost_usd for b in breakdowns)

    def apply_costs(
        self,
        results: list[BenchmarkResult],
        system_providers: dict[str, tuple[str, str]],
    ) -> list[BenchmarkResult]:
        """
        Apply cost estimates to benchmark results in-place.

        Args:
            results: List of benchmark results
            system_providers: Mapping of system name to (provider, model) tuples

        Returns:
            The same results list with estimated_cost_usd populated
        """
        for result in results:
            provider_info = system_providers.get(result.system)
            if provider_info:
                provider, model = provider_info
                breakdown = self.calculate(result, provider, model)
                result.estimated_cost_usd = breakdown.total_cost_usd

        return results
