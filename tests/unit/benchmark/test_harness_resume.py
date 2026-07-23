"""Tests for kill-safe benchmark resume in the evaluation harness.

Validates that incremental persistence records each scored cell and that ``--resume`` skips
already-completed (iteration, system, task) cells without re-executing or duplicating them.
Maps to spec ``strategos_langgraph_hardening`` AC-5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.config.benchmark_settings import BenchmarkRunConfig, BenchmarkSettings, ReportConfig
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.evaluation.run_store import BenchmarkRunStore, JobKey
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory
from src.benchmark.tasks.registry import BenchmarkTaskRegistry


def _make_task(task_id: str) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        category=TaskCategory.QE,
        description=f"Task {task_id}",
        input_data=f"Input for {task_id}",
    )


class RecordingAdapter:
    def __init__(self, name: str = "sys_a") -> None:
        self._name = name
        self.execute_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return True

    async def health_check(self) -> bool:
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        self.execute_count += 1
        return BenchmarkResult(
            task_id=task.task_id,
            system=self._name,
            task_description=task.description,
            raw_response=f"resp {task.task_id}",
        )


def _settings(tmp_path: Path) -> BenchmarkSettings:
    settings = BenchmarkSettings()
    settings._report = ReportConfig(output_dir=str(tmp_path))
    settings._run = BenchmarkRunConfig(
        num_iterations=1,
        incremental_persistence=True,
        checkpoint_every_n_results=1,
        retry_on_failure=False,
    )
    return settings


def _registry() -> BenchmarkTaskRegistry:
    registry = BenchmarkTaskRegistry()
    registry.register(_make_task("T1"))
    registry.register(_make_task("T2"))
    return registry


def _keys(results: list[BenchmarkResult]) -> list[str]:
    return sorted(JobKey.from_result(r).key() for r in results)


@pytest.mark.asyncio
async def test_results_appended_incrementally(tmp_path):
    settings = _settings(tmp_path)
    harness = EvaluationHarness([RecordingAdapter()], _registry(), settings=settings)
    await harness.run()
    jsonl = tmp_path / "runs" / harness.run_id / "results.jsonl"
    assert jsonl.exists()
    assert len(jsonl.read_text().strip().splitlines()) == 2


@pytest.mark.asyncio
async def test_full_resume_skips_all_cells(tmp_path):
    settings = _settings(tmp_path)
    registry = _registry()

    first_adapter = RecordingAdapter()
    first = EvaluationHarness([first_adapter], registry, settings=settings)
    first_results = await first.run()
    assert len(first_results) == 2
    assert first_adapter.execute_count == 2
    run_id = first.run_id

    # Resume with a fresh adapter: every cell is already done, so none re-executes.
    second_adapter = RecordingAdapter()
    second = EvaluationHarness([second_adapter], registry, settings=settings)
    resumed = await second.run(resume_run_id=run_id)

    assert second_adapter.execute_count == 0
    assert _keys(resumed) == ["0:sys_a:T1", "0:sys_a:T2"]
    assert len(resumed) == len(set(_keys(resumed)))  # no duplicates


@pytest.mark.asyncio
async def test_partial_resume_runs_only_missing(tmp_path):
    settings = _settings(tmp_path)
    registry = _registry()

    # Pre-seed the store as if a prior run finished T1 then was killed before T2.
    store = BenchmarkRunStore(tmp_path / "runs" / "resume1", "resume1")
    store.append_result(BenchmarkResult(task_id="T1", system="sys_a", iteration=0, raw_response="preseed"))

    adapter = RecordingAdapter()
    harness = EvaluationHarness([adapter], registry, settings=settings)
    results = await harness.run(resume_run_id="resume1")

    assert adapter.execute_count == 1  # only T2 re-runs
    assert _keys(results) == ["0:sys_a:T1", "0:sys_a:T2"]
    # The pre-seeded T1 is preserved verbatim (not re-executed).
    t1 = next(r for r in results if r.task_id == "T1")
    assert t1.raw_response == "preseed"
