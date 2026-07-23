"""Unit tests for the kill-safe benchmark run store.

Covers ``src/benchmark/evaluation/run_store.py`` — job keying, durable append/reload,
torn-trailing-line tolerance, and the versioned manifest. Maps to spec
``strategos_langgraph_hardening`` AC-5.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.config.benchmark_settings import BenchmarkSettings
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.evaluation.run_store import RUN_STORE_SCHEMA_VERSION, BenchmarkRunStore, JobKey


def _result(task_id: str, system: str, iteration: int, response: str = "r") -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        system=system,
        iteration=iteration,
        raw_response=response,
    )


class TestJobKey:
    def test_key_format(self):
        assert JobKey("T1", "sys_a", 2).key() == "2:sys_a:T1"

    def test_from_result(self):
        r = _result("T1", "sys_a", 3)
        assert JobKey.from_result(r).key() == "3:sys_a:T1"


class TestAppendAndLoad:
    def test_roundtrip(self, tmp_path):
        store = BenchmarkRunStore(tmp_path / "runs" / "run1", "run1")
        store.append_result(_result("T1", "sys_a", 0))
        store.append_result(_result("T2", "sys_a", 0))
        completed = store.load_completed()
        assert set(completed) == {"0:sys_a:T1", "0:sys_a:T2"}
        assert completed["0:sys_a:T1"].task_id == "T1"

    def test_missing_file_returns_empty(self, tmp_path):
        store = BenchmarkRunStore(tmp_path / "runs" / "none", "none")
        assert store.load_completed() == {}

    def test_duplicate_key_last_wins(self, tmp_path):
        store = BenchmarkRunStore(tmp_path / "runs" / "run1", "run1")
        store.append_result(_result("T1", "sys_a", 0, response="first"))
        store.append_result(_result("T1", "sys_a", 0, response="second"))
        completed = store.load_completed()
        assert len(completed) == 1
        assert completed["0:sys_a:T1"].raw_response == "second"

    def test_torn_trailing_line_tolerated(self, tmp_path):
        store = BenchmarkRunStore(tmp_path / "runs" / "run1", "run1")
        store.append_result(_result("T1", "sys_a", 0))
        # Simulate a hard kill mid-write.
        store.results_path.open("a").write('{"task_id": "T2", partial')
        completed = store.load_completed()
        assert set(completed) == {"0:sys_a:T1"}


class TestManifest:
    def test_write_and_read(self, tmp_path):
        store = BenchmarkRunStore(tmp_path / "runs" / "run1", "run1")
        store.write_manifest(BenchmarkSettings(), ["T1", "T2"], ["sys_a"])
        manifest = store.read_manifest()
        assert manifest["schema_version"] == RUN_STORE_SCHEMA_VERSION
        assert manifest["run_id"] == "run1"
        assert manifest["task_ids"] == ["T1", "T2"]
        assert manifest["systems"] == ["sys_a"]

    def test_read_missing_manifest_returns_empty(self, tmp_path):
        store = BenchmarkRunStore(tmp_path / "runs" / "none", "none")
        assert store.read_manifest() == {}
