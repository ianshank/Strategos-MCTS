"""Integration test: a benchmark sweep survives a hard SIGKILL and resumes with no lost
and no duplicated results, without re-executing completed cells.

Maps to spec ``strategos_langgraph_hardening`` AC-5. POSIX-only (uses SIGKILL).
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.config.benchmark_settings import BenchmarkRunConfig, BenchmarkSettings, ReportConfig
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.evaluation.run_store import JobKey
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory
from src.benchmark.tasks.registry import BenchmarkTaskRegistry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not hasattr(os, "kill") or sys.platform.startswith("win"), reason="requires POSIX SIGKILL"),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "tests" / "integration" / "_kill_resume_child.py"
_RUN_ID = "killrun"
_NUM_TASKS = 6


class FastAdapter:
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
        return BenchmarkResult(task_id=task.task_id, system=self._name, raw_response=f"resumed {task.task_id}")


def _count_records(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


@pytest.mark.asyncio
async def test_kill_mid_sweep_then_resume_no_loss_no_dupes(tmp_path):
    jsonl = tmp_path / "runs" / _RUN_ID / "results.jsonl"

    # 1. Launch a slow sweep in a child process; each task takes ~0.5s.
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    child = subprocess.Popen(
        [sys.executable, str(_DRIVER), str(tmp_path), _RUN_ID, str(_NUM_TASKS), "0.5"],
        cwd=str(_REPO_ROOT),
        env=env,
    )
    try:
        # 2. Wait until at least 2 cells are durably recorded, then hard-kill mid-sweep.
        deadline = time.time() + 20.0
        while time.time() < deadline and _count_records(jsonl) < 2:
            time.sleep(0.1)
        recorded = _count_records(jsonl)
        assert 2 <= recorded < _NUM_TASKS, f"expected a mid-sweep kill, saw {recorded} records"
    finally:
        child.kill()
        child.wait(timeout=10)

    completed_before = _count_records(jsonl)
    assert 0 < completed_before < _NUM_TASKS

    # 3. Resume in-process with a fast adapter that records how many cells it re-executes.
    settings = BenchmarkSettings()
    settings._report = ReportConfig(output_dir=str(tmp_path))
    settings._run = BenchmarkRunConfig(
        num_iterations=1,
        incremental_persistence=True,
        checkpoint_every_n_results=1,
        retry_on_failure=False,
    )
    registry = BenchmarkTaskRegistry()
    for i in range(_NUM_TASKS):
        registry.register(
            BenchmarkTask(task_id=f"T{i}", category=TaskCategory.QE, description=f"Task {i}", input_data="x")
        )
    adapter = FastAdapter()
    harness = EvaluationHarness([adapter], registry, settings=settings)
    results = await harness.run(resume_run_id=_RUN_ID)

    # 4a. The full matrix is present with no duplicates.
    keys = [JobKey.from_result(r).key() for r in results]
    expected = {f"0:sys_a:T{i}" for i in range(_NUM_TASKS)}
    assert set(keys) == expected
    assert len(keys) == len(set(keys)), "resume produced duplicate cells"

    # 4b. Only the cells not durably completed before the kill were re-executed.
    assert adapter.execute_count == _NUM_TASKS - completed_before
