"""
Integration tests for the benchmark framework.

Tests end-to-end flows through multiple components,
including registry -> adapter -> harness -> report pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.benchmark.config.benchmark_settings import (
    BenchmarkSettings,
    ReportConfig,
    reset_benchmark_settings,
)
from src.benchmark.evaluation.cost_calculator import CostCalculator
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult
from src.benchmark.reporting.metrics_aggregator import MetricsAggregator
from src.benchmark.reporting.report_generator import ReportGenerator
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory
from src.benchmark.tasks.registry import BenchmarkTaskRegistry


class MockBenchmarkAdapter:
    """Mock adapter for integration testing."""

    def __init__(self, name: str, response_prefix: str = "Analysis") -> None:
        self._name = name
        self._response_prefix = response_prefix

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        return BenchmarkResult(
            task_id=task.task_id,
            system=self._name,
            task_description=task.description,
            total_latency_ms=500.0 + (100.0 if self._name == "sys_b" else 0.0),
            raw_response=f"{self._response_prefix}: {task.description}",
            input_tokens=200,
            output_tokens=150,
            num_agent_calls=3 if self._name == "sys_a" else 4,
            num_tool_calls=2,
        )

    async def health_check(self) -> bool:
        return True


class MockScorer:
    """Mock scorer for integration testing."""

    def __init__(self, base_score: float = 4.0) -> None:
        self._base_score = base_score

    async def score(self, result: BenchmarkResult, task: BenchmarkTask) -> ScoringResult:
        # Vary score slightly by system
        modifier = 0.5 if result.system == "sys_a" else 0.0
        return ScoringResult(
            task_completion=self._base_score + modifier,
            reasoning_depth=self._base_score - 0.5 + modifier,
            accuracy=self._base_score + modifier,
            coherence=self._base_score + 0.5,
            judge_model="mock-judge",
            judge_explanation="Mock scoring",
        )


@pytest.mark.integration
class TestBenchmarkPipeline:
    """Test the full benchmark pipeline: registry -> harness -> report."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        """Test complete benchmark run with two systems, scoring, and reporting."""
        # Step 1: Set up task registry
        registry = BenchmarkTaskRegistry()
        registry.register(
            BenchmarkTask(
                task_id="A1",
                category=TaskCategory.QE,
                description="Code review task",
                input_data="Review this code for bugs",
                expected_outputs=("bug detection", "severity rating"),
            )
        )
        registry.register(
            BenchmarkTask(
                task_id="B1",
                category=TaskCategory.COMPLIANCE,
                description="Compliance extraction",
                input_data="Extract requirements from regulation",
                expected_outputs=("structured requirements",),
            )
        )

        # Step 2: Create adapters
        adapter_a = MockBenchmarkAdapter("sys_a", "LangGraph MCTS Analysis")
        adapter_b = MockBenchmarkAdapter("sys_b", "ADK Analysis")

        # Step 3: Create scorer
        scorer = MockScorer(base_score=3.5)

        # Step 4: Create cost calculator
        cost_calculator = CostCalculator(config=self.settings.cost)

        # Step 5: Run benchmark
        harness = EvaluationHarness(
            adapters=[adapter_a, adapter_b],
            registry=registry,
            scorer=scorer,
            cost_calculator=cost_calculator,
            settings=self.settings,
        )
        results = await harness.run()

        # Verify results
        assert len(results) == 4  # 2 tasks * 2 systems
        assert all(r.scoring.task_completion > 0 for r in results)
        assert all(r.run_id != "" for r in results)

        # Step 6: Aggregate metrics
        aggregator = MetricsAggregator()
        by_system = aggregator.aggregate_by_system(results)
        assert "sys_a" in by_system
        assert "sys_b" in by_system

        # Step 7: Generate report
        with tempfile.TemporaryDirectory() as tmpdir:
            report_config = ReportConfig(output_dir=tmpdir)
            generator = ReportGenerator(config=report_config, settings=self.settings)
            report = generator.generate(results)

            # Verify report content
            assert "## Summary Comparison" in report
            assert "sys_a" in report
            assert "sys_b" in report
            assert "Per-Task Analysis" in report
            assert "Cost Analysis" in report

            # Save and verify file
            path = generator.save(report, tmpdir)
            assert path.exists()
            assert path.read_text() == report

            # Save results JSON
            results_path = harness.save_results(tmpdir)
            assert results_path.exists()
            data = json.loads(results_path.read_text())
            assert len(data["results"]) == 4

    @pytest.mark.asyncio
    async def test_pipeline_with_default_tasks(self) -> None:
        """Test pipeline loading default task sets."""
        registry = BenchmarkTaskRegistry()
        registry.load_defaults()

        adapter = MockBenchmarkAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=registry,
            settings=self.settings,
        )

        # Run only QE tasks for speed
        qe_tasks = registry.get_by_category(TaskCategory.QE)
        qe_ids = [t.task_id for t in qe_tasks]
        results = await harness.run(task_ids=qe_ids)

        assert len(results) == len(qe_ids)
        assert all(r.task_id in qe_ids for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_summary(self) -> None:
        """Test summary generation after benchmark run."""
        registry = BenchmarkTaskRegistry()
        registry.register(
            BenchmarkTask(
                task_id="T1",
                category=TaskCategory.QE,
                description="Test",
                input_data="Input",
            )
        )

        adapter_a = MockBenchmarkAdapter("sys_a")
        adapter_b = MockBenchmarkAdapter("sys_b")
        scorer = MockScorer()

        harness = EvaluationHarness(
            adapters=[adapter_a, adapter_b],
            registry=registry,
            scorer=scorer,
            settings=self.settings,
        )
        await harness.run()
        summary = harness.summary()

        assert summary["total_results"] == 2
        assert "sys_a" in summary["systems"]
        assert "sys_b" in summary["systems"]
        assert summary["systems"]["sys_a"]["successful"] == 1
        assert summary["systems"]["sys_a"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_comparison_report_winner_detection(self) -> None:
        """Test that comparison correctly identifies winners."""
        registry = BenchmarkTaskRegistry()
        for i in range(3):
            registry.register(
                BenchmarkTask(
                    task_id=f"T{i}",
                    category=TaskCategory.QE,
                    description=f"Task {i}",
                    input_data=f"Input {i}",
                )
            )

        adapter_a = MockBenchmarkAdapter("sys_a")
        adapter_b = MockBenchmarkAdapter("sys_b")
        scorer = MockScorer(base_score=3.5)

        harness = EvaluationHarness(
            adapters=[adapter_a, adapter_b],
            registry=registry,
            scorer=scorer,
            settings=self.settings,
        )
        results = await harness.run()

        # sys_a should score higher (MockScorer adds 0.5 for sys_a)
        aggregator = MetricsAggregator()
        comparisons = aggregator.compare_systems(results, "sys_a", "sys_b")

        score_comp = next(c for c in comparisons if c.metric_name == "average_score")
        assert score_comp.winner == "sys_a"


@pytest.mark.integration
class TestTaskRegistryIntegration:
    """Test task registry with JSON export/import."""

    def test_export_import_roundtrip(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.load_defaults()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "tasks.json"
            registry.export_to_json(export_path)

            # Load into fresh registry
            new_registry = BenchmarkTaskRegistry()
            new_registry.load_from_json(export_path)

            assert new_registry.task_count == registry.task_count

            # Verify each task roundtripped correctly
            for task_id in registry.task_ids:
                original = registry.get(task_id)
                restored = new_registry.get(task_id)
                assert original.task_id == restored.task_id
                assert original.category == restored.category
                assert original.complexity == restored.complexity
