"""
Tests for default benchmark task set definitions.

Validates task properties, category groupings, complexity assignments,
and data integrity of the predefined task sets.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.tasks.models import BenchmarkTask, TaskCategory, TaskComplexity
from src.benchmark.tasks.task_sets import (
    ALL_TASKS,
    TASK_A1_CODE_REVIEW,
    TASK_A2_SECURITY_ANALYSIS,
    TASK_A3_TEST_PLAN,
    TASK_A4_ARCHITECTURE_REVIEW,
    TASK_B1_REQUIREMENT_EXTRACTION,
    TASK_B2_CONTROL_GAP_ANALYSIS,
    TASK_B3_REMEDIATION_PLAN,
    TASK_C1_INVESTMENT_STRATEGY,
    TASK_C2_PROJECT_PLANNING,
    TASK_C3_COMPETITIVE_ANALYSIS,
    TASK_SET_A,
    TASK_SET_B,
    TASK_SET_C,
)


@pytest.mark.unit
class TestTaskSetCollections:
    """Test task set collection structure and composition."""

    def test_task_set_a_contains_four_tasks(self) -> None:
        assert len(TASK_SET_A) == 4

    def test_task_set_b_contains_three_tasks(self) -> None:
        assert len(TASK_SET_B) == 3

    def test_task_set_c_contains_three_tasks(self) -> None:
        assert len(TASK_SET_C) == 3

    def test_all_tasks_is_union_of_sets(self) -> None:
        assert ALL_TASKS == TASK_SET_A + TASK_SET_B + TASK_SET_C

    def test_all_tasks_total_count(self) -> None:
        assert len(ALL_TASKS) == 10

    def test_all_tasks_are_tuples(self) -> None:
        assert isinstance(TASK_SET_A, tuple)
        assert isinstance(TASK_SET_B, tuple)
        assert isinstance(TASK_SET_C, tuple)
        assert isinstance(ALL_TASKS, tuple)

    def test_all_elements_are_benchmark_tasks(self) -> None:
        for task in ALL_TASKS:
            assert isinstance(task, BenchmarkTask)


@pytest.mark.unit
class TestTaskIdUniqueness:
    """Verify all task IDs are unique across sets."""

    def test_all_task_ids_unique(self) -> None:
        ids = [t.task_id for t in ALL_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_set_a_ids_start_with_a(self) -> None:
        for task in TASK_SET_A:
            assert task.task_id.startswith("A"), f"Task {task.task_id} in set A does not start with 'A'"

    def test_set_b_ids_start_with_b(self) -> None:
        for task in TASK_SET_B:
            assert task.task_id.startswith("B"), f"Task {task.task_id} in set B does not start with 'B'"

    def test_set_c_ids_start_with_c(self) -> None:
        for task in TASK_SET_C:
            assert task.task_id.startswith("C"), f"Task {task.task_id} in set C does not start with 'C'"


@pytest.mark.unit
class TestTaskSetCategories:
    """Verify category assignments per set."""

    def test_set_a_all_qe(self) -> None:
        for task in TASK_SET_A:
            assert task.category == TaskCategory.QE, f"Task {task.task_id} is not QE"

    def test_set_b_all_compliance(self) -> None:
        for task in TASK_SET_B:
            assert task.category == TaskCategory.COMPLIANCE, f"Task {task.task_id} is not COMPLIANCE"

    def test_set_c_all_strategic(self) -> None:
        for task in TASK_SET_C:
            assert task.category == TaskCategory.STRATEGIC, f"Task {task.task_id} is not STRATEGIC"

    def test_all_categories_represented(self) -> None:
        categories = {t.category for t in ALL_TASKS}
        assert categories == {TaskCategory.QE, TaskCategory.COMPLIANCE, TaskCategory.STRATEGIC}


@pytest.mark.unit
class TestTaskDataIntegrity:
    """Verify each task has required non-empty fields and valid data."""

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.task_id for t in ALL_TASKS])
    def test_task_has_nonempty_description(self, task: BenchmarkTask) -> None:
        assert len(task.description) > 0

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.task_id for t in ALL_TASKS])
    def test_task_has_substantial_input(self, task: BenchmarkTask) -> None:
        assert len(task.input_data) > 50, f"Task {task.task_id} input too short ({len(task.input_data)} chars)"

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.task_id for t in ALL_TASKS])
    def test_task_has_expected_outputs(self, task: BenchmarkTask) -> None:
        assert len(task.expected_outputs) > 0, f"Task {task.task_id} has no expected outputs"

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.task_id for t in ALL_TASKS])
    def test_task_has_valid_complexity(self, task: BenchmarkTask) -> None:
        assert task.complexity in TaskComplexity

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.task_id for t in ALL_TASKS])
    def test_task_is_frozen(self, task: BenchmarkTask) -> None:
        with pytest.raises(AttributeError):
            task.task_id = "CHANGED"  # type: ignore[misc]


@pytest.mark.unit
class TestIndividualTaskProperties:
    """Test specific properties of individual task definitions."""

    def test_a1_code_review(self) -> None:
        assert TASK_A1_CODE_REVIEW.task_id == "A1"
        assert TASK_A1_CODE_REVIEW.complexity == TaskComplexity.MEDIUM
        assert "division by zero" in TASK_A1_CODE_REVIEW.expected_outputs[0]
        assert TASK_A1_CODE_REVIEW.metadata.get("domain") == "mcts"

    def test_a2_security_analysis(self) -> None:
        assert TASK_A2_SECURITY_ANALYSIS.task_id == "A2"
        assert TASK_A2_SECURITY_ANALYSIS.complexity == TaskComplexity.HIGH
        assert any("SQL injection" in o for o in TASK_A2_SECURITY_ANALYSIS.expected_outputs)
        assert TASK_A2_SECURITY_ANALYSIS.metadata.get("domain") == "security"

    def test_a3_test_plan(self) -> None:
        assert TASK_A3_TEST_PLAN.task_id == "A3"
        assert TASK_A3_TEST_PLAN.complexity == TaskComplexity.HIGH
        assert any("unit test" in o for o in TASK_A3_TEST_PLAN.expected_outputs)

    def test_a4_architecture_review(self) -> None:
        assert TASK_A4_ARCHITECTURE_REVIEW.task_id == "A4"
        assert TASK_A4_ARCHITECTURE_REVIEW.complexity == TaskComplexity.VERY_HIGH
        assert TASK_A4_ARCHITECTURE_REVIEW.metadata.get("pattern") == "adr"

    def test_b1_requirement_extraction(self) -> None:
        assert TASK_B1_REQUIREMENT_EXTRACTION.task_id == "B1"
        assert TASK_B1_REQUIREMENT_EXTRACTION.complexity == TaskComplexity.MEDIUM
        assert TASK_B1_REQUIREMENT_EXTRACTION.metadata.get("regulation") == "eu_ai_act"

    def test_b2_control_gap_analysis(self) -> None:
        assert TASK_B2_CONTROL_GAP_ANALYSIS.task_id == "B2"
        assert TASK_B2_CONTROL_GAP_ANALYSIS.complexity == TaskComplexity.HIGH
        assert any("gap" in o for o in TASK_B2_CONTROL_GAP_ANALYSIS.expected_outputs)

    def test_b3_remediation_plan(self) -> None:
        assert TASK_B3_REMEDIATION_PLAN.task_id == "B3"
        assert TASK_B3_REMEDIATION_PLAN.complexity == TaskComplexity.VERY_HIGH
        assert len(TASK_B3_REMEDIATION_PLAN.expected_outputs) >= 5

    def test_c1_investment_strategy(self) -> None:
        assert TASK_C1_INVESTMENT_STRATEGY.task_id == "C1"
        assert TASK_C1_INVESTMENT_STRATEGY.complexity == TaskComplexity.HIGH
        assert TASK_C1_INVESTMENT_STRATEGY.metadata.get("sector") == "ai_infrastructure"

    def test_c2_project_planning(self) -> None:
        assert TASK_C2_PROJECT_PLANNING.task_id == "C2"
        assert TASK_C2_PROJECT_PLANNING.complexity == TaskComplexity.VERY_HIGH
        assert TASK_C2_PROJECT_PLANNING.metadata.get("product") == "mangomas"

    def test_c3_competitive_analysis(self) -> None:
        assert TASK_C3_COMPETITIVE_ANALYSIS.task_id == "C3"
        assert TASK_C3_COMPETITIVE_ANALYSIS.complexity == TaskComplexity.VERY_HIGH
        assert TASK_C3_COMPETITIVE_ANALYSIS.metadata.get("frameworks") == 5


@pytest.mark.unit
class TestTaskMetadata:
    """Verify metadata structure across tasks."""

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.task_id for t in ALL_TASKS])
    def test_metadata_is_dict(self, task: BenchmarkTask) -> None:
        assert isinstance(task.metadata, dict)

    def test_all_tasks_have_metadata(self) -> None:
        for task in ALL_TASKS:
            assert len(task.metadata) > 0, f"Task {task.task_id} has empty metadata"

    def test_set_a_tasks_have_domain(self) -> None:
        for task in TASK_SET_A:
            assert "domain" in task.metadata, f"Task {task.task_id} missing 'domain' in metadata"

    def test_set_b_tasks_have_regulation(self) -> None:
        for task in TASK_SET_B:
            assert "regulation" in task.metadata, f"Task {task.task_id} missing 'regulation' in metadata"


@pytest.mark.unit
class TestTaskSerialization:
    """Test that all predefined tasks survive serialization roundtrip."""

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.task_id for t in ALL_TASKS])
    def test_to_dict_roundtrip(self, task: BenchmarkTask) -> None:
        data = task.to_dict()
        restored = BenchmarkTask.from_dict(data)
        assert restored.task_id == task.task_id
        assert restored.category == task.category
        assert restored.complexity == task.complexity
        assert restored.description == task.description
        assert restored.expected_outputs == task.expected_outputs
