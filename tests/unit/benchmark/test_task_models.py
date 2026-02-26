"""
Tests for benchmark task data models and registry.

Validates task creation, serialization, registry operations,
and data-driven task loading.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.tasks.models import BenchmarkTask, TaskCategory, TaskComplexity
from src.benchmark.tasks.registry import BenchmarkTaskRegistry
from src.benchmark.tasks.task_sets import (
    ALL_TASKS,
    TASK_A1_CODE_REVIEW,
    TASK_SET_A,
    TASK_SET_B,
    TASK_SET_C,
)


@pytest.mark.unit
class TestTaskCategory:
    """Test TaskCategory enum."""

    def test_values(self) -> None:
        assert TaskCategory.QE == "qe"
        assert TaskCategory.COMPLIANCE == "compliance"
        assert TaskCategory.STRATEGIC == "strategic"

    def test_from_string(self) -> None:
        assert TaskCategory("qe") == TaskCategory.QE
        assert TaskCategory("compliance") == TaskCategory.COMPLIANCE
        assert TaskCategory("strategic") == TaskCategory.STRATEGIC


@pytest.mark.unit
class TestTaskComplexity:
    """Test TaskComplexity enum."""

    def test_values(self) -> None:
        assert TaskComplexity.LOW == "low"
        assert TaskComplexity.MEDIUM == "medium"
        assert TaskComplexity.HIGH == "high"
        assert TaskComplexity.VERY_HIGH == "very_high"


@pytest.mark.unit
class TestBenchmarkTask:
    """Test BenchmarkTask dataclass."""

    def test_create_task(self) -> None:
        task = BenchmarkTask(
            task_id="T1",
            category=TaskCategory.QE,
            description="Test task",
            input_data="Test input",
        )
        assert task.task_id == "T1"
        assert task.category == TaskCategory.QE
        assert task.description == "Test task"
        assert task.input_data == "Test input"

    def test_default_fields(self) -> None:
        task = BenchmarkTask(
            task_id="T1",
            category=TaskCategory.QE,
            description="Test",
            input_data="Input",
        )
        assert task.expected_outputs == ()
        assert task.complexity == TaskComplexity.MEDIUM
        assert task.metadata == {}

    def test_immutability(self) -> None:
        task = BenchmarkTask(
            task_id="T1",
            category=TaskCategory.QE,
            description="Test",
            input_data="Input",
        )
        with pytest.raises(AttributeError):
            task.task_id = "T2"  # type: ignore[misc]

    def test_validation_empty_task_id(self) -> None:
        with pytest.raises(ValueError, match="task_id cannot be empty"):
            BenchmarkTask(
                task_id="",
                category=TaskCategory.QE,
                description="Test",
                input_data="Input",
            )

    def test_validation_empty_description(self) -> None:
        with pytest.raises(ValueError, match="description cannot be empty"):
            BenchmarkTask(
                task_id="T1",
                category=TaskCategory.QE,
                description="",
                input_data="Input",
            )

    def test_validation_empty_input(self) -> None:
        with pytest.raises(ValueError, match="input_data cannot be empty"):
            BenchmarkTask(
                task_id="T1",
                category=TaskCategory.QE,
                description="Test",
                input_data="",
            )

    def test_category_label(self) -> None:
        task = BenchmarkTask(
            task_id="T1",
            category=TaskCategory.QE,
            description="Test",
            input_data="Input",
        )
        assert task.category_label == "Software Quality Engineering"

    def test_to_dict(self) -> None:
        task = BenchmarkTask(
            task_id="T1",
            category=TaskCategory.QE,
            description="Test",
            input_data="Input",
            expected_outputs=("out1", "out2"),
            complexity=TaskComplexity.HIGH,
        )
        data = task.to_dict()
        assert data["task_id"] == "T1"
        assert data["category"] == "qe"
        assert data["expected_outputs"] == ["out1", "out2"]
        assert data["complexity"] == "high"

    def test_from_dict(self) -> None:
        data = {
            "task_id": "T1",
            "category": "qe",
            "description": "Test task",
            "input_data": "Test input",
            "expected_outputs": ["out1"],
            "complexity": "high",
        }
        task = BenchmarkTask.from_dict(data)
        assert task.task_id == "T1"
        assert task.category == TaskCategory.QE
        assert task.expected_outputs == ("out1",)
        assert task.complexity == TaskComplexity.HIGH

    def test_roundtrip_serialization(self) -> None:
        original = BenchmarkTask(
            task_id="T1",
            category=TaskCategory.COMPLIANCE,
            description="Test",
            input_data="Input",
            expected_outputs=("a", "b", "c"),
            complexity=TaskComplexity.VERY_HIGH,
            metadata={"key": "value"},
        )
        restored = BenchmarkTask.from_dict(original.to_dict())
        assert restored.task_id == original.task_id
        assert restored.category == original.category
        assert restored.expected_outputs == original.expected_outputs
        assert restored.complexity == original.complexity


@pytest.mark.unit
class TestBenchmarkTaskRegistry:
    """Test BenchmarkTaskRegistry operations."""

    def _make_task(self, task_id: str = "T1") -> BenchmarkTask:
        return BenchmarkTask(
            task_id=task_id,
            category=TaskCategory.QE,
            description=f"Task {task_id}",
            input_data=f"Input for {task_id}",
        )

    def test_register_and_get(self) -> None:
        registry = BenchmarkTaskRegistry()
        task = self._make_task()
        registry.register(task)
        assert registry.get("T1") is task

    def test_register_duplicate_raises(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.register(self._make_task("T1"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(self._make_task("T1"))

    def test_get_missing_raises(self) -> None:
        registry = BenchmarkTaskRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("missing")

    def test_task_count(self) -> None:
        registry = BenchmarkTaskRegistry()
        assert registry.task_count == 0
        registry.register(self._make_task("T1"))
        assert registry.task_count == 1
        registry.register(self._make_task("T2"))
        assert registry.task_count == 2

    def test_task_ids_sorted(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.register(self._make_task("C1"))
        registry.register(self._make_task("A1"))
        registry.register(self._make_task("B1"))
        assert registry.task_ids == ["A1", "B1", "C1"]

    def test_get_all_sorted(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.register(self._make_task("B1"))
        registry.register(self._make_task("A1"))
        tasks = registry.get_all()
        assert [t.task_id for t in tasks] == ["A1", "B1"]

    def test_get_by_category(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.register(
            BenchmarkTask(task_id="Q1", category=TaskCategory.QE, description="QE task", input_data="input")
        )
        registry.register(
            BenchmarkTask(
                task_id="C1", category=TaskCategory.COMPLIANCE, description="Compliance task", input_data="input"
            )
        )
        qe_tasks = registry.get_by_category(TaskCategory.QE)
        assert len(qe_tasks) == 1
        assert qe_tasks[0].task_id == "Q1"

    def test_get_by_complexity(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.register(
            BenchmarkTask(
                task_id="E1",
                category=TaskCategory.QE,
                description="Easy",
                input_data="in",
                complexity=TaskComplexity.LOW,
            )
        )
        registry.register(
            BenchmarkTask(
                task_id="H1",
                category=TaskCategory.QE,
                description="Hard",
                input_data="in",
                complexity=TaskComplexity.HIGH,
            )
        )
        high = registry.get_by_complexity(TaskComplexity.HIGH)
        assert len(high) == 1
        assert high[0].task_id == "H1"

    def test_register_many(self) -> None:
        registry = BenchmarkTaskRegistry()
        tasks = [self._make_task(f"T{i}") for i in range(5)]
        registry.register_many(tasks)
        assert registry.task_count == 5

    def test_clear(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.register(self._make_task())
        registry.clear()
        assert registry.task_count == 0

    def test_load_defaults(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.load_defaults()
        assert registry.task_count == len(ALL_TASKS)
        assert registry.task_count >= 10  # Expect at least 10 default tasks

    def test_json_roundtrip(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.register(self._make_task("T1"))
        registry.register(self._make_task("T2"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            registry.export_to_json(path)
            assert path.exists()

            # Load into fresh registry
            new_registry = BenchmarkTaskRegistry()
            new_registry.load_from_json(path)
            assert new_registry.task_count == 2
            assert new_registry.get("T1").description == "Task T1"
        finally:
            path.unlink(missing_ok=True)

    def test_load_from_missing_file(self) -> None:
        registry = BenchmarkTaskRegistry()
        with pytest.raises(FileNotFoundError):
            registry.load_from_json("/nonexistent/file.json")

    def test_summary(self) -> None:
        registry = BenchmarkTaskRegistry()
        registry.load_defaults()
        summary = registry.summary()
        assert summary["total_tasks"] > 0
        assert "by_category" in summary
        assert "by_complexity" in summary
        assert "qe" in summary["by_category"]


@pytest.mark.unit
class TestDefaultTaskSets:
    """Test the default task set definitions."""

    def test_task_set_a_count(self) -> None:
        assert len(TASK_SET_A) == 4

    def test_task_set_b_count(self) -> None:
        assert len(TASK_SET_B) == 3

    def test_task_set_c_count(self) -> None:
        assert len(TASK_SET_C) == 3

    def test_all_tasks_total(self) -> None:
        assert len(ALL_TASKS) == 10

    def test_all_task_ids_unique(self) -> None:
        ids = [t.task_id for t in ALL_TASKS]
        assert len(ids) == len(set(ids))

    def test_task_a1_properties(self) -> None:
        assert TASK_A1_CODE_REVIEW.task_id == "A1"
        assert TASK_A1_CODE_REVIEW.category == TaskCategory.QE
        assert TASK_A1_CODE_REVIEW.complexity == TaskComplexity.MEDIUM
        assert len(TASK_A1_CODE_REVIEW.expected_outputs) > 0
        assert "division by zero" in TASK_A1_CODE_REVIEW.expected_outputs[0]

    def test_all_tasks_have_expected_outputs(self) -> None:
        for task in ALL_TASKS:
            assert len(task.expected_outputs) > 0, f"Task {task.task_id} has no expected outputs"

    def test_all_tasks_have_nonempty_input(self) -> None:
        for task in ALL_TASKS:
            assert len(task.input_data) > 10, f"Task {task.task_id} has too short input"

    def test_set_a_all_qe_category(self) -> None:
        for task in TASK_SET_A:
            assert task.category == TaskCategory.QE

    def test_set_b_all_compliance_category(self) -> None:
        for task in TASK_SET_B:
            assert task.category == TaskCategory.COMPLIANCE

    def test_set_c_all_strategic_category(self) -> None:
        for task in TASK_SET_C:
            assert task.category == TaskCategory.STRATEGIC
