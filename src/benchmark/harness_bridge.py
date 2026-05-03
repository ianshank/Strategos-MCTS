"""Bridge from :class:`BenchmarkTask` to harness :class:`Task` / spec markdown.

This module is the single integration point that lets the agent harness
consume tasks defined for the benchmark framework without duplicating the
task content.

Two output forms are supported:

* :meth:`BenchmarkTaskAdapter.to_task` — direct in-memory conversion to
  the harness :class:`Task` dataclass. Preferred when the harness is
  embedded in the same Python process as the benchmark.
* :meth:`BenchmarkTaskAdapter.to_spec_text` — render a markdown spec
  string compatible with :class:`SpecLoader`. Used when invoking the
  harness CLI (``harness run --spec``) which requires a spec file.

The adapter is intentionally side-effect free; callers who need a file
on disk should write the returned text themselves. This keeps the
adapter trivially testable and avoids hidden filesystem coupling.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from src.benchmark.tasks.models import BenchmarkTask
from src.benchmark.tasks.task_sets import ALL_TASKS
from src.framework.harness.state import AcceptanceCriterion, Task

# Template lives next to the harness — one parametric file used for
# every benchmark task. Resolved relative to *this* module so it works
# regardless of the CWD or how the package is installed.
DEFAULT_TEMPLATE_PATH: Path = (
    Path(__file__).resolve().parents[1] / "framework" / "harness" / "specs" / "benchmark_task_template.md"
)


@dataclass
class BenchmarkTaskAdapter:
    """Convert a :class:`BenchmarkTask` to a harness :class:`Task` or spec markdown.

    The adapter is configured with three knobs:

    * ``template_path`` — override the markdown template; defaults to
      :data:`DEFAULT_TEMPLATE_PATH`.
    * ``constraints_metadata_key`` — the key under which constraints are
      stored in :attr:`BenchmarkTask.metadata`. Defaults to
      ``"constraints"``.
    * ``criterion_id_prefix`` — prefix for synthesised acceptance
      criterion ids. Defaults to ``"AC"`` (``AC-1``, ``AC-2``, ...).
    """

    template_path: Path | None = None
    constraints_metadata_key: str = "constraints"
    criterion_id_prefix: str = "AC"

    # ─────────────────────────── public API ───────────────────────────

    def to_task(self, bt: BenchmarkTask) -> Task:
        """Return a harness :class:`Task` built from ``bt``.

        The mapping is direct and lossless: every field on the benchmark
        task is reflected in the harness task either by name or via
        ``Task.metadata`` so downstream phases can recover the original
        category/complexity if needed.
        """
        constraints = self._build_constraints(bt.metadata)
        criteria = self._build_acceptance_criteria(bt.expected_outputs)
        # Preserve original metadata (do not mutate) and tag with
        # benchmark-derived keys so analytics can group by category.
        merged_metadata: dict[str, Any] = dict(bt.metadata)
        merged_metadata.update(
            {
                "benchmark_category": bt.category.value,
                "benchmark_complexity": bt.complexity.value,
                "benchmark_description": bt.description,
            }
        )
        return Task(
            id=bt.task_id,
            goal=bt.input_data,
            acceptance_criteria=criteria,
            constraints=constraints,
            metadata=merged_metadata,
            raw="",
        )

    def to_spec_text(self, bt: BenchmarkTask) -> str:
        """Render ``bt`` as markdown text compatible with :class:`SpecLoader`.

        The body of the ``# Goal`` section is wrapped in a triple-backtick
        fence (or a longer fence if the input itself contains backticks)
        so that any ATX headers or ``---`` lines inside the prompt do not
        confuse the spec parser.
        """
        template = Template(self._load_template())
        constraints = self._build_constraints(bt.metadata)
        criteria = self._build_acceptance_criteria(bt.expected_outputs)
        # ``safe_substitute`` is required because user-controlled fields
        # (input_data, description) may contain ``$`` characters that are
        # NOT placeholders.
        return template.safe_substitute(
            task_id=bt.task_id,
            category=bt.category.value,
            complexity=bt.complexity.value,
            description=self._sanitize_single_line(bt.description),
            goal=self._fence_goal(bt.input_data),
            acceptance_criteria=self._render_bullets(c.description for c in criteria),
            constraints=self._render_bullets(constraints),
        )

    def lookup(self, task_id: str) -> BenchmarkTask:
        """Find a benchmark task by id (case-insensitive).

        Raises :class:`KeyError` with a list of (up to) ten known ids if
        no match is found.
        """
        needle = task_id.casefold()
        for task in ALL_TASKS:
            if task.task_id.casefold() == needle:
                return task
        known = ", ".join(t.task_id for t in ALL_TASKS[:10])
        raise KeyError(f"Unknown benchmark task_id: {task_id!r}. Known: {known}")

    # ────────────────────────────── helpers ──────────────────────────

    def _build_acceptance_criteria(self, expected_outputs: Iterable[str]) -> tuple[AcceptanceCriterion, ...]:
        """Map each expected output to an indexed :class:`AcceptanceCriterion`."""
        return tuple(
            AcceptanceCriterion(
                id=f"{self.criterion_id_prefix}-{i + 1}",
                description=expected,
                check="",
            )
            for i, expected in enumerate(expected_outputs)
        )

    def _build_constraints(self, metadata: dict[str, Any]) -> tuple[str, ...]:
        """Pull constraints out of ``metadata`` and coerce to ``tuple[str, ...]``.

        Accepts a tuple, list, or single string. Anything else (including
        ``None`` or absent key) yields an empty tuple.
        """
        raw = metadata.get(self.constraints_metadata_key)
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, (list, tuple)):
            return tuple(str(item) for item in raw)
        return ()

    def _load_template(self) -> str:
        """Read the markdown template from disk, honouring the override."""
        path = self.template_path or DEFAULT_TEMPLATE_PATH
        return path.read_text(encoding="utf-8")

    # The following helpers are private utilities that keep
    # ``to_spec_text`` readable; each is a small, focused transformation.

    @staticmethod
    def _fence_goal(goal: str) -> str:
        """Wrap ``goal`` in a fence at least one backtick longer than any
        run of backticks already inside the value.

        This protects the spec parser from goals that themselves contain
        ATX headers, ``---`` separators, or triple-backtick code fences.
        """
        # Detect the longest run of backticks inside the goal so we can
        # pick a fence that won't be terminated prematurely.
        max_run = 0
        run = 0
        for ch in goal:
            if ch == "`":
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        fence = "`" * max(3, max_run + 1)
        return f"{fence}\n{goal}\n{fence}"

    @staticmethod
    def _render_bullets(items: Iterable[str]) -> str:
        """Render an iterable as ``- ``-prefixed markdown bullets.

        Embedded newlines in any single bullet are collapsed to spaces so
        the bullet stays on one line — :class:`SpecLoader` only matches
        bullets at the start of a line.
        """
        bullets = [f"- {str(item).replace(chr(10), ' ').strip()}" for item in items]
        return "\n".join(bullets)

    @staticmethod
    def _sanitize_single_line(text: str) -> str:
        """Collapse newlines so a value renders safely inside frontmatter."""
        return text.replace("\n", " ").strip()


__all__ = ["BenchmarkTaskAdapter", "DEFAULT_TEMPLATE_PATH"]
