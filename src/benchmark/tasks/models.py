"""
Benchmark task data models.

Defines the schema for benchmark tasks used across all evaluation runs.
Tasks are data-driven and loaded from configuration, not hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskCategory(StrEnum):
    """Task domain categories for benchmark organization."""

    QE = "qe"  # Software Quality Engineering
    COMPLIANCE = "compliance"  # Regulatory Compliance
    STRATEGIC = "strategic"  # Strategic Decision Making


class TaskComplexity(StrEnum):
    """Task complexity levels for analysis."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass(frozen=True)
class BenchmarkTask:
    """
    Immutable benchmark task definition.

    Attributes:
        task_id: Unique task identifier (e.g., "A1", "B2")
        category: Task domain category
        description: Human-readable task description
        input_data: Full input text/prompt for the task
        expected_outputs: Key elements expected in a good response
        complexity: Task complexity rating
        metadata: Additional task metadata for analysis
    """

    task_id: str
    category: TaskCategory
    description: str
    input_data: str
    expected_outputs: tuple[str, ...] = field(default_factory=tuple)
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task fields."""
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if not self.description:
            raise ValueError("description cannot be empty")
        if not self.input_data:
            raise ValueError("input_data cannot be empty")

    @property
    def category_label(self) -> str:
        """Human-readable category label."""
        labels = {
            TaskCategory.QE: "Software Quality Engineering",
            TaskCategory.COMPLIANCE: "Regulatory Compliance",
            TaskCategory.STRATEGIC: "Strategic Decision Making",
        }
        return labels.get(self.category, self.category.value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "task_id": self.task_id,
            "category": self.category.value,
            "description": self.description,
            "input_data": self.input_data,
            "expected_outputs": list(self.expected_outputs),
            "complexity": self.complexity.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkTask:
        """Deserialize task from dictionary."""
        return cls(
            task_id=data["task_id"],
            category=TaskCategory(data["category"]),
            description=data["description"],
            input_data=data["input_data"],
            expected_outputs=tuple(data.get("expected_outputs", [])),
            complexity=TaskComplexity(data.get("complexity", "medium")),
            metadata=data.get("metadata", {}),
        )
