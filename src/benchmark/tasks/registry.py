"""
Data-driven benchmark task registry.

Manages task collections with filtering, lookup, and serialization.
Tasks can be loaded from code (task_sets.py) or external JSON files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.benchmark.tasks.models import BenchmarkTask, TaskCategory, TaskComplexity

logger = logging.getLogger(__name__)


class BenchmarkTaskRegistry:
    """
    Registry for benchmark task management.

    Supports registration, lookup, filtering, and serialization
    of benchmark tasks. Thread-safe for concurrent access.

    Example:
        >>> registry = BenchmarkTaskRegistry()
        >>> registry.load_defaults()
        >>> qe_tasks = registry.get_by_category(TaskCategory.QE)
        >>> task = registry.get("A1")
    """

    def __init__(self) -> None:
        self._tasks: dict[str, BenchmarkTask] = {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def task_count(self) -> int:
        """Return total number of registered tasks."""
        return len(self._tasks)

    @property
    def task_ids(self) -> list[str]:
        """Return sorted list of registered task IDs."""
        return sorted(self._tasks.keys())

    def register(self, task: BenchmarkTask) -> None:
        """
        Register a benchmark task.

        Args:
            task: Task to register

        Raises:
            ValueError: If a task with the same ID already exists
        """
        if task.task_id in self._tasks:
            raise ValueError(f"Task '{task.task_id}' already registered")
        self._tasks[task.task_id] = task
        self._logger.debug("Registered task: %s (%s)", task.task_id, task.description)

    def register_many(self, tasks: tuple[BenchmarkTask, ...] | list[BenchmarkTask]) -> None:
        """Register multiple tasks at once."""
        for task in tasks:
            self.register(task)

    def get(self, task_id: str) -> BenchmarkTask:
        """
        Retrieve a task by ID.

        Args:
            task_id: Task identifier

        Returns:
            The requested BenchmarkTask

        Raises:
            KeyError: If task_id not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found. Available: {self.task_ids}")
        return self._tasks[task_id]

    def get_all(self) -> list[BenchmarkTask]:
        """Return all registered tasks sorted by ID."""
        return [self._tasks[tid] for tid in sorted(self._tasks.keys())]

    def get_by_category(self, category: TaskCategory) -> list[BenchmarkTask]:
        """Filter tasks by category."""
        return [t for t in self.get_all() if t.category == category]

    def get_by_complexity(self, complexity: TaskComplexity) -> list[BenchmarkTask]:
        """Filter tasks by complexity."""
        return [t for t in self.get_all() if t.complexity == complexity]

    def get_by_category_and_complexity(
        self,
        category: TaskCategory | None = None,
        complexity: TaskComplexity | None = None,
    ) -> list[BenchmarkTask]:
        """Filter tasks by category and/or complexity."""
        tasks = self.get_all()
        if category is not None:
            tasks = [t for t in tasks if t.category == category]
        if complexity is not None:
            tasks = [t for t in tasks if t.complexity == complexity]
        return tasks

    def load_defaults(self) -> None:
        """Load default task sets from task_sets module."""
        from src.benchmark.tasks.task_sets import ALL_TASKS

        self.register_many(ALL_TASKS)
        self._logger.info("Loaded %d default benchmark tasks", len(ALL_TASKS))

    def load_from_json(self, file_path: str | Path) -> None:
        """
        Load tasks from a JSON file.

        Args:
            file_path: Path to JSON file containing task definitions

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        tasks_data = data if isinstance(data, list) else data.get("tasks", [])
        loaded = 0
        for task_data in tasks_data:
            task = BenchmarkTask.from_dict(task_data)
            self.register(task)
            loaded += 1

        self._logger.info("Loaded %d tasks from %s", loaded, path)

    def export_to_json(self, file_path: str | Path) -> None:
        """
        Export all tasks to a JSON file.

        Args:
            file_path: Output path for the JSON file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"tasks": [t.to_dict() for t in self.get_all()]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._logger.info("Exported %d tasks to %s", self.task_count, path)

    def clear(self) -> None:
        """Remove all registered tasks."""
        count = len(self._tasks)
        self._tasks.clear()
        self._logger.debug("Cleared %d tasks from registry", count)

    def summary(self) -> dict[str, Any]:
        """Generate a summary of registered tasks."""
        by_category: dict[str, int] = {}
        by_complexity: dict[str, int] = {}

        for task in self._tasks.values():
            cat = task.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            comp = task.complexity.value
            by_complexity[comp] = by_complexity.get(comp, 0) + 1

        return {
            "total_tasks": self.task_count,
            "by_category": by_category,
            "by_complexity": by_complexity,
            "task_ids": self.task_ids,
        }
