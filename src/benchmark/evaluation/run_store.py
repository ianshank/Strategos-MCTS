"""Kill-safe incremental persistence for benchmark sweeps.

The evaluation harness holds every result in memory and writes a single JSON file at the
very end, so a crash mid-sweep loses the whole run. :class:`BenchmarkRunStore` appends each
scored result to an append-only JSONL log *before the loop advances*, using the same atomic
``O_APPEND`` idiom as the harness memory log, so a hard kill loses at most the in-flight
result. On resume, :meth:`load_completed` replays the log (tolerating a torn trailing line
from SIGKILL) so already-finished ``(iteration, system, task)`` cells are skipped rather than
re-executed.

This is the sweep counterpart of the self-play trainer's existing checkpoint/resume; it does
not change the final ``results.json`` artifact, which the harness still writes from the full
(resumed + new) result set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.benchmark.evaluation.models import BenchmarkResult

if TYPE_CHECKING:
    from src.benchmark.config.benchmark_settings import BenchmarkSettings

RUN_STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JobKey:
    """Identity of a single sweep cell: one task on one system in one iteration."""

    task_id: str
    system: str
    iteration: int

    def key(self) -> str:
        return f"{self.iteration}:{self.system}:{self.task_id}"

    @classmethod
    def from_result(cls, result: BenchmarkResult) -> JobKey:
        return cls(task_id=result.task_id, system=result.system, iteration=result.iteration)


class BenchmarkRunStore:
    """Append-only, resumable store for one benchmark run's results."""

    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self._results_path = self.run_dir / "results.jsonl"
        self._manifest_path = self.run_dir / "run.json"

    @property
    def results_path(self) -> Path:
        return self._results_path

    def append_result(self, result: BenchmarkResult) -> None:
        """Durably append one scored result (atomic O_APPEND)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(result.to_dict(), default=str) + "\n"
        fd = os.open(self._results_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    def load_completed(self) -> dict[str, BenchmarkResult]:
        """Return already-completed results keyed by :meth:`JobKey.key`.

        A partial trailing line (e.g. from a hard kill mid-write) is skipped, not fatal —
        this is what makes resume safe.
        """
        completed: dict[str, BenchmarkResult] = {}
        if not self._results_path.exists():
            return completed
        for line in self._results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                result = BenchmarkResult.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            completed[JobKey.from_result(result).key()] = result
        return completed

    def write_manifest(self, settings: BenchmarkSettings, task_ids: list[str], systems: list[str]) -> None:
        """Write the run manifest (schema version, run id, matrix, settings snapshot)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "task_ids": list(task_ids),
            "systems": list(systems),
            "settings": settings.safe_dict(),
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    def read_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return {}
        manifest: dict[str, Any] = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return manifest
