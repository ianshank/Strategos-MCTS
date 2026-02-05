"""
Tests for benchmark report generation.

Validates markdown report structure, content accuracy,
and edge case handling.
"""

from __future__ import annotations

import tempfile

from src.benchmark.config.benchmark_settings import ReportConfig
from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult
from src.benchmark.reporting.report_generator import ReportGenerator


def _make_result(
    system: str,
    task_id: str = "A1",
    latency: float = 1000.0,
    score: float = 4.0,
    cost: float = 0.05,
    tokens: int = 500,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        system=system,
        task_description=f"Task {task_id}",
        total_latency_ms=latency,
        input_tokens=tokens,
        output_tokens=tokens,
        estimated_cost_usd=cost,
        scoring=ScoringResult(
            task_completion=score,
            reasoning_depth=score - 0.5,
            accuracy=score,
            coherence=score + 0.5,
        ),
    )


class TestReportGenerator:
    """Test report generation."""

    def setup_method(self) -> None:
        self.config = ReportConfig()
        self.generator = ReportGenerator(config=self.config)

    def test_generate_single_system(self) -> None:
        results = [_make_result("sys_a", "A1"), _make_result("sys_a", "A2")]
        report = self.generator.generate(results)
        assert "# " in report  # Has title
        assert "sys_a" in report

    def test_generate_two_systems(self) -> None:
        results = [
            _make_result("langgraph_mcts", "A1", latency=1000, score=4.0),
            _make_result("vertex_adk", "A1", latency=800, score=3.5),
            _make_result("langgraph_mcts", "A2", latency=1200, score=4.5),
            _make_result("vertex_adk", "A2", latency=900, score=3.0),
        ]
        report = self.generator.generate(results)

        assert "## Summary Comparison" in report
        assert "langgraph_mcts" in report
        assert "vertex_adk" in report
        assert "| Metric |" in report
        assert "Winner" in report

    def test_generate_per_task_analysis(self) -> None:
        results = [
            _make_result("sys_a", "A1"),
            _make_result("sys_b", "A1"),
        ]
        report = self.generator.generate(results)
        assert "Per-Task Analysis" in report
        assert "Task A1" in report

    def test_generate_scoring_breakdown(self) -> None:
        results = [
            _make_result("sys_a", score=4.0),
            _make_result("sys_b", score=3.0),
        ]
        report = self.generator.generate(results)
        assert "Scoring Breakdown" in report

    def test_generate_cost_analysis(self) -> None:
        results = [
            _make_result("sys_a", cost=0.05),
            _make_result("sys_b", cost=0.01),
        ]
        report = self.generator.generate(results)
        assert "Cost Analysis" in report

    def test_generate_key_findings(self) -> None:
        results = [
            _make_result("sys_a", score=4.5, latency=1000),
            _make_result("sys_b", score=3.0, latency=500),
        ]
        report = self.generator.generate(results)
        assert "Key Findings" in report

    def test_save_report(self) -> None:
        results = [_make_result("sys_a")]
        report = self.generator.generate(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.generator.save(report, tmpdir)
            assert path.exists()
            content = path.read_text()
            assert "sys_a" in content

    def test_generate_empty_results(self) -> None:
        report = self.generator.generate([])
        assert isinstance(report, str)

    def test_custom_title(self) -> None:
        results = [_make_result("sys_a")]
        report = self.generator.generate(results, title="Custom Title")
        assert "# Custom Title" in report

    def test_report_contains_date(self) -> None:
        results = [_make_result("sys_a")]
        report = self.generator.generate(results)
        assert "**Date:**" in report

    def test_report_contains_metadata(self) -> None:
        results = [
            _make_result("sys_a", "A1"),
            _make_result("sys_a", "A2"),
            _make_result("sys_b", "A1"),
        ]
        report = self.generator.generate(results)
        assert "**Tasks Evaluated:** 2" in report
        assert "**Total Results:** 3" in report
