"""
Tests for metrics aggregation and comparison.

Validates statistical computations, system comparisons,
and edge case handling.
"""

from __future__ import annotations

from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult
from src.benchmark.reporting.metrics_aggregator import (
    AggregatedMetrics,
    MetricsAggregator,
    SystemComparison,
)


def _make_result(
    system: str = "system_a",
    task_id: str = "T1",
    latency: float = 1000.0,
    score: float = 4.0,
    tokens: int = 500,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        system=system,
        total_latency_ms=latency,
        input_tokens=tokens,
        output_tokens=tokens,
        scoring=ScoringResult(
            task_completion=score,
            reasoning_depth=score,
            accuracy=score,
            coherence=score,
        ),
    )


class TestAggregatedMetrics:
    """Test AggregatedMetrics dataclass."""

    def test_default(self) -> None:
        metrics = AggregatedMetrics()
        assert metrics.count == 0
        assert metrics.mean == 0.0

    def test_to_dict(self) -> None:
        metrics = AggregatedMetrics(count=10, mean=3.5, std_dev=0.5)
        data = metrics.to_dict()
        assert data["count"] == 10
        assert data["mean"] == 3.5


class TestSystemComparison:
    """Test SystemComparison logic."""

    def test_higher_is_better(self) -> None:
        comp = SystemComparison(
            metric_name="accuracy",
            system_a_name="a",
            system_b_name="b",
            system_a=AggregatedMetrics(mean=4.0),
            system_b=AggregatedMetrics(mean=3.0),
            higher_is_better=True,
        )
        assert comp.winner == "a"

    def test_lower_is_better(self) -> None:
        comp = SystemComparison(
            metric_name="latency_ms",
            system_a_name="a",
            system_b_name="b",
            system_a=AggregatedMetrics(mean=1000.0),
            system_b=AggregatedMetrics(mean=500.0),
            higher_is_better=False,
        )
        assert comp.winner == "b"

    def test_tie(self) -> None:
        comp = SystemComparison(
            metric_name="score",
            system_a_name="a",
            system_b_name="b",
            system_a=AggregatedMetrics(mean=4.0),
            system_b=AggregatedMetrics(mean=4.0),
        )
        assert comp.winner == "tie"


class TestMetricsAggregator:
    """Test MetricsAggregator computations."""

    def test_compute_stats_empty(self) -> None:
        stats = MetricsAggregator.compute_stats([])
        assert stats.count == 0
        assert stats.mean == 0.0

    def test_compute_stats_single(self) -> None:
        stats = MetricsAggregator.compute_stats([5.0])
        assert stats.count == 1
        assert stats.mean == 5.0
        assert stats.std_dev == 0.0
        assert stats.min_val == 5.0
        assert stats.max_val == 5.0
        assert stats.median == 5.0

    def test_compute_stats_multiple(self) -> None:
        stats = MetricsAggregator.compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0
        assert stats.std_dev > 0

    def test_compute_stats_even_count_median(self) -> None:
        stats = MetricsAggregator.compute_stats([1.0, 2.0, 3.0, 4.0])
        assert stats.median == 2.5

    def test_compute_stats_std_dev(self) -> None:
        # Known values
        stats = MetricsAggregator.compute_stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert stats.mean == 5.0
        # Sample std dev should be ~2.0
        assert 1.9 < stats.std_dev < 2.2

    def test_aggregate_by_system(self) -> None:
        results = [
            _make_result("sys_a", latency=1000, score=4.0),
            _make_result("sys_a", latency=1200, score=3.5),
            _make_result("sys_b", latency=800, score=4.5),
        ]

        agg = MetricsAggregator()
        by_system = agg.aggregate_by_system(results)

        assert "sys_a" in by_system
        assert "sys_b" in by_system
        assert by_system["sys_a"]["latency_ms"].count == 2
        assert by_system["sys_b"]["latency_ms"].count == 1

    def test_aggregate_by_task(self) -> None:
        results = [
            _make_result("sys_a", task_id="T1", score=4.0),
            _make_result("sys_b", task_id="T1", score=3.0),
            _make_result("sys_a", task_id="T2", score=5.0),
        ]

        agg = MetricsAggregator()
        by_task = agg.aggregate_by_task(results)

        assert "T1" in by_task
        assert "T2" in by_task
        assert "sys_a" in by_task["T1"]
        assert "sys_b" in by_task["T1"]

    def test_aggregate_skips_errors(self) -> None:
        results = [
            _make_result("sys_a", score=4.0),
            BenchmarkResult(
                task_id="T2",
                system="sys_a",
                raw_response="Error: timeout",
            ),
        ]

        agg = MetricsAggregator()
        by_system = agg.aggregate_by_system(results)
        assert by_system["sys_a"]["latency_ms"].count == 1  # Error excluded

    def test_compare_systems(self) -> None:
        results = [
            _make_result("sys_a", latency=1000, score=4.0),
            _make_result("sys_a", latency=1200, score=4.5),
            _make_result("sys_b", latency=800, score=3.0),
            _make_result("sys_b", latency=700, score=3.5),
        ]

        agg = MetricsAggregator()
        comparisons = agg.compare_systems(results, "sys_a", "sys_b")

        assert len(comparisons) > 0

        # Find latency comparison
        latency_comp = next(c for c in comparisons if c.metric_name == "latency_ms")
        assert latency_comp.winner == "sys_b"  # Lower latency

        # Find average_score comparison
        score_comp = next(c for c in comparisons if c.metric_name == "average_score")
        assert score_comp.winner == "sys_a"  # Higher score

    def test_compare_systems_empty(self) -> None:
        agg = MetricsAggregator()
        comparisons = agg.compare_systems([], "a", "b")
        assert len(comparisons) == 0
