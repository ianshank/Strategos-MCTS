"""
Tests for BenchmarkTaskRegistry.

Validates registration, lookup, filtering, serialization,
and edge cases for the task registry.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.tasks.models import BenchmarkTask, TaskCategory, TaskComplexity
from src.benchmark.tasks.registry import BenchmarkTaskRegistry


def _make_task(
    task_id: str = "T1",
    category: TaskCategory = TaskCategory.QE,
    complexity: TaskComplexity = TaskComplexity.MEDIUM,
    description: str | None = None,
    input_data: str | None = None,
    expected_outputs: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        category=category,
        description=description or f"Task {task_id}",
        input_data=input_data or f"Input for {task_id}",
        expected_outputs=expected_outputs,
        complexity=complexity,
        metadata=metadata or {},
    )


@pytest.fixture
def registry() -> BenchmarkTaskRegistry:
    return BenchmarkTaskRegistry()


@pytest.fixture
def populated_registry() -> BenchmarkTaskRegistry:
    """Registry with a mix of categories and complexities."""
    reg = BenchmarkTaskRegistry()
    reg.register(_make_task("A1", TaskCategory.QE, TaskComplexity.LOW))
    reg.register(_make_task("A2", TaskCategory.QE, TaskComplexity.HIGH))
    reg.register(_make_task("B1", TaskCategory.COMPLIANCE, TaskComplexity.MEDIUM))
    reg.register(_make_task("B2", TaskCategory.COMPLIANCE, TaskComplexity.VERY_HIGH))
    reg.register(_make_task("C1", TaskCategory.STRATEGIC, TaskComplexity.HIGH))
    return reg


@pytest.mark.unit
class TestRegistryRegistration:
    """Test task registration operations."""

    def test_register_single_task(self, registry: BenchmarkTaskRegistry) -> None:
        task = _make_task("T1")
        registry.register(task)
        assert registry.task_count == 1
        assert registry.get("T1") is task

    def test_register_duplicate_raises_value_error(self, registry: BenchmarkTaskRegistry) -> None:
        registry.register(_make_task("T1"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_task("T1"))

    def test_register_many_list(self, registry: BenchmarkTaskRegistry) -> None:
        tasks = [_make_task(f"T{i}") for i in range(5)]
        registry.register_many(tasks)
        assert registry.task_count == 5

    def test_register_many_tuple(self, registry: BenchmarkTaskRegistry) -> None:
        tasks = tuple(_make_task(f"T{i}") for i in range(3))
        registry.register_many(tasks)
        assert registry.task_count == 3

    def test_register_many_with_duplicate_raises(self, registry: BenchmarkTaskRegistry) -> None:
        tasks = [_make_task("T1"), _make_task("T2"), _make_task("T1")]
        with pytest.raises(ValueError, match="already registered"):
            registry.register_many(tasks)
        # First two should have been registered before the error
        assert registry.task_count == 2

    def test_register_many_empty_list(self, registry: BenchmarkTaskRegistry) -> None:
        registry.register_many([])
        assert registry.task_count == 0


@pytest.mark.unit
class TestRegistryLookup:
    """Test task lookup and retrieval."""

    def test_get_existing_task(self, populated_registry: BenchmarkTaskRegistry) -> None:
        task = populated_registry.get("A1")
        assert task.task_id == "A1"
        assert task.category == TaskCategory.QE

    def test_get_missing_task_raises_key_error(self, registry: BenchmarkTaskRegistry) -> None:
        with pytest.raises(KeyError, match="not found"):
            registry.get("MISSING")

    def test_get_missing_task_error_lists_available(self, populated_registry: BenchmarkTaskRegistry) -> None:
        with pytest.raises(KeyError, match="Available"):
            populated_registry.get("MISSING")

    def test_get_all_returns_sorted(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_all()
        ids = [t.task_id for t in tasks]
        assert ids == sorted(ids)

    def test_get_all_empty_registry(self, registry: BenchmarkTaskRegistry) -> None:
        assert registry.get_all() == []

    def test_task_ids_sorted(self, populated_registry: BenchmarkTaskRegistry) -> None:
        ids = populated_registry.task_ids
        assert ids == sorted(ids)
        assert len(ids) == 5

    def test_task_ids_empty_registry(self, registry: BenchmarkTaskRegistry) -> None:
        assert registry.task_ids == []

    def test_task_count(self, populated_registry: BenchmarkTaskRegistry) -> None:
        assert populated_registry.task_count == 5


@pytest.mark.unit
class TestRegistryFiltering:
    """Test category and complexity filtering."""

    def test_get_by_category_qe(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category(TaskCategory.QE)
        assert len(tasks) == 2
        assert all(t.category == TaskCategory.QE for t in tasks)

    def test_get_by_category_compliance(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category(TaskCategory.COMPLIANCE)
        assert len(tasks) == 2

    def test_get_by_category_strategic(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category(TaskCategory.STRATEGIC)
        assert len(tasks) == 1
        assert tasks[0].task_id == "C1"

    def test_get_by_category_no_match(self, registry: BenchmarkTaskRegistry) -> None:
        registry.register(_make_task("T1", TaskCategory.QE))
        tasks = registry.get_by_category(TaskCategory.STRATEGIC)
        assert tasks == []

    def test_get_by_complexity(self, populated_registry: BenchmarkTaskRegistry) -> None:
        high = populated_registry.get_by_complexity(TaskComplexity.HIGH)
        assert len(high) == 2
        assert all(t.complexity == TaskComplexity.HIGH for t in high)

    def test_get_by_complexity_no_match(self, populated_registry: BenchmarkTaskRegistry) -> None:
        # No MEDIUM tasks? Actually B1 is MEDIUM
        low = populated_registry.get_by_complexity(TaskComplexity.LOW)
        assert len(low) == 1

    def test_get_by_category_and_complexity_both(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category_and_complexity(
            category=TaskCategory.QE,
            complexity=TaskComplexity.HIGH,
        )
        assert len(tasks) == 1
        assert tasks[0].task_id == "A2"

    def test_get_by_category_and_complexity_category_only(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category_and_complexity(category=TaskCategory.COMPLIANCE)
        assert len(tasks) == 2

    def test_get_by_category_and_complexity_complexity_only(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category_and_complexity(complexity=TaskComplexity.VERY_HIGH)
        assert len(tasks) == 1
        assert tasks[0].task_id == "B2"

    def test_get_by_category_and_complexity_neither(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category_and_complexity()
        assert len(tasks) == 5

    def test_get_by_category_and_complexity_no_match(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category_and_complexity(
            category=TaskCategory.STRATEGIC,
            complexity=TaskComplexity.LOW,
        )
        assert tasks == []

    def test_filtered_results_are_sorted(self, populated_registry: BenchmarkTaskRegistry) -> None:
        tasks = populated_registry.get_by_category(TaskCategory.COMPLIANCE)
        ids = [t.task_id for t in tasks]
        assert ids == sorted(ids)


@pytest.mark.unit
class TestRegistryDefaults:
    """Test loading default task sets."""

    def test_load_defaults(self, registry: BenchmarkTaskRegistry) -> None:
        registry.load_defaults()
        assert registry.task_count >= 10

    def test_load_defaults_includes_all_categories(self, registry: BenchmarkTaskRegistry) -> None:
        registry.load_defaults()
        for category in TaskCategory:
            tasks = registry.get_by_category(category)
            assert len(tasks) > 0, f"No tasks for category {category}"

    def test_load_defaults_twice_raises(self, registry: BenchmarkTaskRegistry) -> None:
        registry.load_defaults()
        with pytest.raises(ValueError, match="already registered"):
            registry.load_defaults()


@pytest.mark.unit
class TestRegistrySerialization:
    """Test JSON import/export."""

    def test_export_creates_file(self, populated_registry: BenchmarkTaskRegistry) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            populated_registry.export_to_json(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "tasks" in data
            assert len(data["tasks"]) == 5
        finally:
            path.unlink(missing_ok=True)

    def test_export_creates_parent_directories(self, registry: BenchmarkTaskRegistry) -> None:
        registry.register(_make_task("T1"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "tasks.json"
            registry.export_to_json(path)
            assert path.exists()

    def test_load_from_json_list_format(self, registry: BenchmarkTaskRegistry) -> None:
        """Test loading from a JSON file with a top-level list."""
        tasks_data = [
            {
                "task_id": "J1",
                "category": "qe",
                "description": "JSON task 1",
                "input_data": "Input 1",
            },
            {
                "task_id": "J2",
                "category": "compliance",
                "description": "JSON task 2",
                "input_data": "Input 2",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tasks_data, f)
            path = Path(f.name)
        try:
            registry.load_from_json(path)
            assert registry.task_count == 2
            assert registry.get("J1").category == TaskCategory.QE
            assert registry.get("J2").category == TaskCategory.COMPLIANCE
        finally:
            path.unlink(missing_ok=True)

    def test_load_from_json_dict_format(self, registry: BenchmarkTaskRegistry) -> None:
        """Test loading from a JSON file with a 'tasks' key."""
        data = {
            "tasks": [
                {
                    "task_id": "D1",
                    "category": "strategic",
                    "description": "Dict task",
                    "input_data": "Input",
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            registry.load_from_json(path)
            assert registry.task_count == 1
            assert registry.get("D1").category == TaskCategory.STRATEGIC
        finally:
            path.unlink(missing_ok=True)

    def test_load_from_missing_file_raises(self, registry: BenchmarkTaskRegistry) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            registry.load_from_json("/nonexistent/path/tasks.json")

    def test_load_from_invalid_json_raises(self, registry: BenchmarkTaskRegistry) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            path = Path(f.name)
        try:
            with pytest.raises(json.JSONDecodeError):
                registry.load_from_json(path)
        finally:
            path.unlink(missing_ok=True)

    def test_roundtrip_preserves_data(self, populated_registry: BenchmarkTaskRegistry) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            populated_registry.export_to_json(path)

            new_registry = BenchmarkTaskRegistry()
            new_registry.load_from_json(path)

            assert new_registry.task_count == populated_registry.task_count
            for task_id in populated_registry.task_ids:
                original = populated_registry.get(task_id)
                restored = new_registry.get(task_id)
                assert restored.task_id == original.task_id
                assert restored.category == original.category
                assert restored.complexity == original.complexity
                assert restored.description == original.description
        finally:
            path.unlink(missing_ok=True)


@pytest.mark.unit
class TestRegistryClear:
    """Test clearing the registry."""

    def test_clear_removes_all_tasks(self, populated_registry: BenchmarkTaskRegistry) -> None:
        assert populated_registry.task_count > 0
        populated_registry.clear()
        assert populated_registry.task_count == 0

    def test_clear_allows_re_registration(self, registry: BenchmarkTaskRegistry) -> None:
        task = _make_task("T1")
        registry.register(task)
        registry.clear()
        # Should be able to register the same ID again
        registry.register(task)
        assert registry.task_count == 1

    def test_clear_empty_registry(self, registry: BenchmarkTaskRegistry) -> None:
        registry.clear()  # Should not raise
        assert registry.task_count == 0


@pytest.mark.unit
class TestRegistrySummary:
    """Test summary generation."""

    def test_summary_structure(self, populated_registry: BenchmarkTaskRegistry) -> None:
        summary = populated_registry.summary()
        assert "total_tasks" in summary
        assert "by_category" in summary
        assert "by_complexity" in summary
        assert "task_ids" in summary

    def test_summary_total(self, populated_registry: BenchmarkTaskRegistry) -> None:
        summary = populated_registry.summary()
        assert summary["total_tasks"] == 5

    def test_summary_by_category(self, populated_registry: BenchmarkTaskRegistry) -> None:
        summary = populated_registry.summary()
        assert summary["by_category"]["qe"] == 2
        assert summary["by_category"]["compliance"] == 2
        assert summary["by_category"]["strategic"] == 1

    def test_summary_by_complexity(self, populated_registry: BenchmarkTaskRegistry) -> None:
        summary = populated_registry.summary()
        assert summary["by_complexity"]["low"] == 1
        assert summary["by_complexity"]["high"] == 2
        assert summary["by_complexity"]["medium"] == 1
        assert summary["by_complexity"]["very_high"] == 1

    def test_summary_task_ids_sorted(self, populated_registry: BenchmarkTaskRegistry) -> None:
        summary = populated_registry.summary()
        assert summary["task_ids"] == sorted(summary["task_ids"])

    def test_summary_empty_registry(self, registry: BenchmarkTaskRegistry) -> None:
        summary = registry.summary()
        assert summary["total_tasks"] == 0
        assert summary["by_category"] == {}
        assert summary["by_complexity"] == {}
        assert summary["task_ids"] == []
