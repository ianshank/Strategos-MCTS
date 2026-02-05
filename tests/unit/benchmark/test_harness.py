"""
Tests for the evaluation harness.

Validates orchestration logic, timeout handling, retry behavior,
and result aggregation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.benchmark.config.benchmark_settings import BenchmarkSettings, reset_benchmark_settings
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory
from src.benchmark.tasks.registry import BenchmarkTaskRegistry


def _make_task(task_id: str = "T1") -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        category=TaskCategory.QE,
        description=f"Task {task_id}",
        input_data=f"Input for {task_id}",
    )


class MockAdapter:
    """Mock adapter for testing the harness."""

    def __init__(self, name: str = "mock_system", available: bool = True, fail: bool = False) -> None:
        self._name = name
        self._available = available
        self._fail = fail
        self.execute_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return self._available

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        self.execute_count += 1
        if self._fail:
            raise RuntimeError("Mock execution failure")
        return BenchmarkResult(
            task_id=task.task_id,
            system=self._name,
            task_description=task.description,
            total_latency_ms=100.0,
            raw_response=f"Mock response for {task.task_id}",
            input_tokens=100,
            output_tokens=50,
        )

    async def health_check(self) -> bool:
        return self._available


class TestEvaluationHarness:
    """Test the evaluation harness orchestration."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()
        self.registry = BenchmarkTaskRegistry()
        self.registry.register(_make_task("T1"))
        self.registry.register(_make_task("T2"))

    @pytest.mark.asyncio
    async def test_run_single_adapter(self) -> None:
        adapter = MockAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        results = await harness.run()

        assert len(results) == 2  # 2 tasks * 1 system
        assert all(r.system == "sys_a" for r in results)
        assert adapter.execute_count == 2

    @pytest.mark.asyncio
    async def test_run_multiple_adapters(self) -> None:
        adapter_a = MockAdapter("sys_a")
        adapter_b = MockAdapter("sys_b")
        harness = EvaluationHarness(
            adapters=[adapter_a, adapter_b],
            registry=self.registry,
            settings=self.settings,
        )

        results = await harness.run()

        assert len(results) == 4  # 2 tasks * 2 systems
        a_results = [r for r in results if r.system == "sys_a"]
        b_results = [r for r in results if r.system == "sys_b"]
        assert len(a_results) == 2
        assert len(b_results) == 2

    @pytest.mark.asyncio
    async def test_run_specific_tasks(self) -> None:
        adapter = MockAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        results = await harness.run(task_ids=["T1"])

        assert len(results) == 1
        assert results[0].task_id == "T1"

    @pytest.mark.asyncio
    async def test_run_with_scorer(self) -> None:
        adapter = MockAdapter("sys_a")
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = ScoringResult(
            task_completion=4.0,
            accuracy=3.5,
        )

        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            scorer=mock_scorer,
            settings=self.settings,
        )

        results = await harness.run()

        assert all(r.scoring.task_completion == 4.0 for r in results)
        assert mock_scorer.score.call_count == 2

    @pytest.mark.asyncio
    async def test_run_unavailable_adapter_skipped(self) -> None:
        adapter_a = MockAdapter("sys_a", available=True)
        adapter_b = MockAdapter("sys_b", available=False)

        harness = EvaluationHarness(
            adapters=[adapter_a, adapter_b],
            registry=self.registry,
            settings=self.settings,
        )

        results = await harness.run()

        assert len(results) == 2  # Only sys_a results
        assert all(r.system == "sys_a" for r in results)

    @pytest.mark.asyncio
    async def test_run_adapter_failure_creates_error_result(self) -> None:
        adapter = MockAdapter("sys_a", fail=True)

        # Set low retry count for fast test
        settings = BenchmarkSettings()
        settings._run = None  # Force re-creation

        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=settings,
        )

        results = await harness.run()

        assert len(results) == 2
        assert all(r.has_error for r in results)

    @pytest.mark.asyncio
    async def test_run_no_adapters(self) -> None:
        harness = EvaluationHarness(
            adapters=[],
            registry=self.registry,
            settings=self.settings,
        )

        results = await harness.run()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_run_no_tasks(self) -> None:
        adapter = MockAdapter("sys_a")
        empty_registry = BenchmarkTaskRegistry()

        harness = EvaluationHarness(
            adapters=[adapter],
            registry=empty_registry,
            settings=self.settings,
        )

        results = await harness.run()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_run_id_generated(self) -> None:
        adapter = MockAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        await harness.run()
        assert harness.run_id != ""
        assert len(harness.run_id) == 8

    @pytest.mark.asyncio
    async def test_run_single(self) -> None:
        adapter = MockAdapter("sys_a")
        task = _make_task("T1")

        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        result = await harness.run_single(adapter, task)
        assert result.task_id == "T1"
        assert result.system == "sys_a"
        assert not result.has_error

    @pytest.mark.asyncio
    async def test_results_property(self) -> None:
        adapter = MockAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        assert harness.results == []
        await harness.run()
        assert len(harness.results) == 2

    @pytest.mark.asyncio
    async def test_get_results_by_system(self) -> None:
        adapter_a = MockAdapter("sys_a")
        adapter_b = MockAdapter("sys_b")
        harness = EvaluationHarness(
            adapters=[adapter_a, adapter_b],
            registry=self.registry,
            settings=self.settings,
        )

        await harness.run()
        by_system = harness.get_results_by_system()
        assert "sys_a" in by_system
        assert "sys_b" in by_system

    @pytest.mark.asyncio
    async def test_get_results_by_task(self) -> None:
        adapter = MockAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        await harness.run()
        by_task = harness.get_results_by_task()
        assert "T1" in by_task
        assert "T2" in by_task

    @pytest.mark.asyncio
    async def test_summary(self) -> None:
        adapter = MockAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        await harness.run()
        summary = harness.summary()

        assert summary["run_id"] != ""
        assert summary["total_results"] == 2
        assert "sys_a" in summary["systems"]
        assert summary["systems"]["sys_a"]["total_tasks"] == 2

    @pytest.mark.asyncio
    async def test_save_results(self, tmp_path: object) -> None:
        import tempfile

        adapter = MockAdapter("sys_a")
        harness = EvaluationHarness(
            adapters=[adapter],
            registry=self.registry,
            settings=self.settings,
        )

        await harness.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = harness.save_results(tmpdir)
            assert path.exists()
            import json

            data = json.loads(path.read_text())
            assert "results" in data
            assert len(data["results"]) == 2
