"""
Metrics aggregation for benchmark results.

Computes statistical summaries across benchmark runs
for comparison reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.benchmark.evaluation.models import BenchmarkResult


@dataclass(frozen=True)
class AggregatedMetrics:
    """Statistical summary for a group of benchmark results."""

    count: int = 0
    mean: float = 0.0
    std_dev: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    median: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4),
            "median": round(self.median, 4),
        }


@dataclass
class SystemComparison:
    """Side-by-side comparison of two systems on a single metric."""

    metric_name: str
    system_a_name: str
    system_b_name: str
    system_a: AggregatedMetrics
    system_b: AggregatedMetrics
    winner: str = ""
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if self.system_a.mean == self.system_b.mean:
            self.winner = "tie"
        elif self.higher_is_better:
            self.winner = self.system_a_name if self.system_a.mean > self.system_b.mean else self.system_b_name
        else:
            self.winner = self.system_a_name if self.system_a.mean < self.system_b.mean else self.system_b_name


class MetricsAggregator:
    """
    Aggregates and compares benchmark metrics across systems.

    Example:
        >>> aggregator = MetricsAggregator()
        >>> summary = aggregator.aggregate_by_system(results)
        >>> comparison = aggregator.compare_systems(results, "langgraph_mcts", "vertex_adk")
    """

    @staticmethod
    def compute_stats(values: list[float]) -> AggregatedMetrics:
        """
        Compute statistical summary for a list of values.

        Args:
            values: List of numeric values

        Returns:
            AggregatedMetrics with statistical summary
        """
        if not values:
            return AggregatedMetrics()

        n = len(values)
        mean = sum(values) / n
        sorted_vals = sorted(values)

        # Median
        if n % 2 == 0:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            median = sorted_vals[n // 2]

        # Standard deviation
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0

        return AggregatedMetrics(
            count=n,
            mean=mean,
            std_dev=std_dev,
            min_val=min(values),
            max_val=max(values),
            median=median,
        )

    def aggregate_by_system(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, dict[str, AggregatedMetrics]]:
        """
        Aggregate metrics grouped by system.

        Args:
            results: List of benchmark results

        Returns:
            Nested dict: system -> metric_name -> AggregatedMetrics
        """
        by_system: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_system.setdefault(r.system, []).append(r)

        aggregated: dict[str, dict[str, AggregatedMetrics]] = {}
        for system, sys_results in by_system.items():
            valid = [r for r in sys_results if not r.has_error]
            aggregated[system] = self._compute_system_metrics(valid)

        return aggregated

    def aggregate_by_task(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, dict[str, dict[str, AggregatedMetrics]]]:
        """
        Aggregate metrics grouped by task, then by system.

        Returns:
            Nested dict: task_id -> system -> metric_name -> AggregatedMetrics
        """
        by_task: dict[str, dict[str, list[BenchmarkResult]]] = {}
        for r in results:
            task_dict = by_task.setdefault(r.task_id, {})
            task_dict.setdefault(r.system, []).append(r)

        aggregated: dict[str, dict[str, dict[str, AggregatedMetrics]]] = {}
        for task_id, systems in by_task.items():
            aggregated[task_id] = {}
            for system, sys_results in systems.items():
                valid = [r for r in sys_results if not r.has_error]
                aggregated[task_id][system] = self._compute_system_metrics(valid)

        return aggregated

    def compare_systems(
        self,
        results: list[BenchmarkResult],
        system_a: str,
        system_b: str,
    ) -> list[SystemComparison]:
        """
        Generate side-by-side comparisons between two systems.

        Args:
            results: All benchmark results
            system_a: First system name
            system_b: Second system name

        Returns:
            List of SystemComparison instances
        """
        aggregated = self.aggregate_by_system(results)

        metrics_a = aggregated.get(system_a, {})
        metrics_b = aggregated.get(system_b, {})

        # Define which metrics are better when lower
        lower_is_better = {"latency_ms", "cost_usd", "time_to_first_token_ms"}

        comparisons: list[SystemComparison] = []
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())

        for metric_name in sorted(all_metrics):
            a_stats = metrics_a.get(metric_name, AggregatedMetrics())
            b_stats = metrics_b.get(metric_name, AggregatedMetrics())

            comparisons.append(
                SystemComparison(
                    metric_name=metric_name,
                    system_a_name=system_a,
                    system_b_name=system_b,
                    system_a=a_stats,
                    system_b=b_stats,
                    higher_is_better=metric_name not in lower_is_better,
                )
            )

        return comparisons

    def _compute_system_metrics(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, AggregatedMetrics]:
        """Compute all metrics for a set of results from one system."""
        if not results:
            return {}

        return {
            "latency_ms": self.compute_stats([r.total_latency_ms for r in results]),
            "time_to_first_token_ms": self.compute_stats(
                [r.time_to_first_token_ms for r in results if r.time_to_first_token_ms > 0]
            ),
            "task_completion": self.compute_stats([r.scoring.task_completion for r in results]),
            "reasoning_depth": self.compute_stats([r.scoring.reasoning_depth for r in results]),
            "accuracy": self.compute_stats([r.scoring.accuracy for r in results]),
            "coherence": self.compute_stats([r.scoring.coherence for r in results]),
            "average_score": self.compute_stats([r.scoring.average_score for r in results]),
            "agent_calls": self.compute_stats([float(r.num_agent_calls) for r in results]),
            "tool_calls": self.compute_stats([float(r.num_tool_calls) for r in results]),
            "total_tokens": self.compute_stats([float(r.total_tokens) for r in results]),
            "cost_usd": self.compute_stats([r.estimated_cost_usd for r in results]),
        }
