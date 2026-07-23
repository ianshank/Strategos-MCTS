"""Child-process driver for the kill-and-resume integration test (not a test module).

Runs a small benchmark sweep with a deliberately slow adapter, writing durably to a run
store under a fixed run id so the parent can poll progress, SIGKILL it mid-sweep, and resume.

Usage: python tests/integration/_kill_resume_child.py <output_dir> <run_id> <num_tasks> <sleep_s>
"""

from __future__ import annotations

import asyncio
import sys

from src.benchmark.config.benchmark_settings import BenchmarkRunConfig, BenchmarkSettings, ReportConfig
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory
from src.benchmark.tasks.registry import BenchmarkTaskRegistry


class SlowAdapter:
    def __init__(self, name: str, sleep_s: float) -> None:
        self._name = name
        self._sleep_s = sleep_s

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return True

    async def health_check(self) -> bool:
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        await asyncio.sleep(self._sleep_s)
        return BenchmarkResult(task_id=task.task_id, system=self._name, raw_response=f"resp {task.task_id}")


async def _main() -> None:
    output_dir, run_id, num_tasks, sleep_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    settings = BenchmarkSettings()
    settings._report = ReportConfig(output_dir=output_dir)
    settings._run = BenchmarkRunConfig(
        num_iterations=1,
        incremental_persistence=True,
        checkpoint_every_n_results=1,
        retry_on_failure=False,
    )
    registry = BenchmarkTaskRegistry()
    for i in range(num_tasks):
        registry.register(
            BenchmarkTask(task_id=f"T{i}", category=TaskCategory.QE, description=f"Task {i}", input_data="x")
        )
    harness = EvaluationHarness([SlowAdapter("sys_a", sleep_s)], registry, settings=settings)
    await harness.run(resume_run_id=run_id)


if __name__ == "__main__":
    asyncio.run(_main())
