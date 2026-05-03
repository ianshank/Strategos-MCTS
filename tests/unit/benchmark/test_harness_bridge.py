"""Tests for :mod:`src.benchmark.harness_bridge`.

These tests pin the contract of :class:`BenchmarkTaskAdapter`:

* in-memory conversion (`to_task`) is total and lossless;
* markdown rendering (`to_spec_text`) round-trips through
  :class:`SpecLoader` even when the prompt contains ``#`` headers,
  ``---`` separators, or triple-backtick fences;
* lookup is case-insensitive and raises a discoverable error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.harness_bridge import (
    DEFAULT_TEMPLATE_PATH,
    BenchmarkTaskAdapter,
)
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory
from src.benchmark.tasks.task_sets import (
    TASK_A1_CODE_REVIEW,
    TASK_A2_SECURITY_ANALYSIS,
    TASK_A3_TEST_PLAN,
)
from src.framework.harness.intent.spec_loader import SpecLoader

pytestmark = pytest.mark.unit


# ─────────────────────────── to_task contract ───────────────────────────


def test_to_task_id_matches_bt_task_id() -> None:
    """The harness ``Task.id`` mirrors the benchmark ``task_id``."""
    task = BenchmarkTaskAdapter().to_task(TASK_A1_CODE_REVIEW)
    assert task.id == TASK_A1_CODE_REVIEW.task_id


def test_to_task_goal_is_verbatim_input_data() -> None:
    """Goal must be the raw prompt — adapters cannot silently transform it."""
    task = BenchmarkTaskAdapter().to_task(TASK_A1_CODE_REVIEW)
    assert task.goal == TASK_A1_CODE_REVIEW.input_data


def test_to_task_maps_each_expected_output_to_acceptance_criterion_with_indexed_id() -> None:
    """Each expected output becomes ``<prefix>-<n>`` with description preserved."""
    bt = BenchmarkTask(
        task_id="X1",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        expected_outputs=("alpha", "beta", "gamma"),
    )
    task = BenchmarkTaskAdapter().to_task(bt)
    assert tuple(c.id for c in task.acceptance_criteria) == ("AC-1", "AC-2", "AC-3")
    assert tuple(c.description for c in task.acceptance_criteria) == ("alpha", "beta", "gamma")
    assert all(c.check == "" for c in task.acceptance_criteria)


def test_to_task_constraints_pulled_from_metadata() -> None:
    """A tuple under ``metadata['constraints']`` becomes ``Task.constraints``."""
    bt = BenchmarkTask(
        task_id="X2",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        metadata={"constraints": ("X", "Y")},
    )
    task = BenchmarkTaskAdapter().to_task(bt)
    assert task.constraints == ("X", "Y")


def test_to_task_constraints_default_empty_tuple_when_absent() -> None:
    """Missing key yields an empty tuple, not ``None`` or ``KeyError``."""
    task = BenchmarkTaskAdapter().to_task(TASK_A1_CODE_REVIEW)
    assert task.constraints == ()


def test_to_task_constraints_accepts_list_input() -> None:
    """A list is normalised to ``tuple[str, ...]``."""
    bt = BenchmarkTask(
        task_id="X3",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        metadata={"constraints": ["X"]},
    )
    task = BenchmarkTaskAdapter().to_task(bt)
    assert task.constraints == ("X",)


def test_to_task_constraints_accepts_single_string() -> None:
    """A single string is wrapped in a one-element tuple."""
    bt = BenchmarkTask(
        task_id="X4",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        metadata={"constraints": "only-one"},
    )
    task = BenchmarkTaskAdapter().to_task(bt)
    assert task.constraints == ("only-one",)


def test_to_task_constraints_unknown_type_yields_empty_tuple() -> None:
    """An unsupported type (e.g. dict) is treated as no constraints."""
    bt = BenchmarkTask(
        task_id="X5",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        metadata={"constraints": {"k": "v"}},
    )
    task = BenchmarkTaskAdapter().to_task(bt)
    assert task.constraints == ()


def test_to_task_metadata_preserves_original_keys() -> None:
    """All original metadata keys survive the bridge."""
    task = BenchmarkTaskAdapter().to_task(TASK_A1_CODE_REVIEW)
    for key, value in TASK_A1_CODE_REVIEW.metadata.items():
        assert task.metadata[key] == value


def test_to_task_metadata_adds_benchmark_category_and_complexity_and_description() -> None:
    """Benchmark-derived metadata is tagged so analytics can group by task type."""
    task = BenchmarkTaskAdapter().to_task(TASK_A1_CODE_REVIEW)
    assert task.metadata["benchmark_category"] == TASK_A1_CODE_REVIEW.category.value
    assert task.metadata["benchmark_complexity"] == TASK_A1_CODE_REVIEW.complexity.value
    assert task.metadata["benchmark_description"] == TASK_A1_CODE_REVIEW.description


def test_to_task_handles_empty_expected_outputs() -> None:
    """No expected outputs means no acceptance criteria — never an error."""
    bt = BenchmarkTask(
        task_id="X6",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        expected_outputs=(),
    )
    task = BenchmarkTaskAdapter().to_task(bt)
    assert task.acceptance_criteria == ()


def test_to_task_uses_custom_criterion_prefix() -> None:
    """A bespoke prefix flows through to all generated ids."""
    bt = BenchmarkTask(
        task_id="X7",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        expected_outputs=("a", "b"),
    )
    adapter = BenchmarkTaskAdapter(criterion_id_prefix="REQ")
    task = adapter.to_task(bt)
    assert tuple(c.id for c in task.acceptance_criteria) == ("REQ-1", "REQ-2")


def test_to_task_uses_custom_constraints_metadata_key() -> None:
    """A non-default key still resolves constraints from metadata."""
    bt = BenchmarkTask(
        task_id="X8",
        category=TaskCategory.QE,
        description="d",
        input_data="i",
        metadata={"limits": ("budget=100",)},
    )
    adapter = BenchmarkTaskAdapter(constraints_metadata_key="limits")
    task = adapter.to_task(bt)
    assert task.constraints == ("budget=100",)


def test_to_task_raw_field_is_empty() -> None:
    """``Task.raw`` is intentionally left blank — adapter is internal."""
    task = BenchmarkTaskAdapter().to_task(TASK_A1_CODE_REVIEW)
    assert task.raw == ""


def test_to_task_does_not_mutate_source_metadata() -> None:
    """Adapter must not mutate the (shared) source metadata dict."""
    snapshot = dict(TASK_A1_CODE_REVIEW.metadata)
    BenchmarkTaskAdapter().to_task(TASK_A1_CODE_REVIEW)
    assert TASK_A1_CODE_REVIEW.metadata == snapshot


# ───────────────────────── to_spec_text contract ─────────────────────────


@pytest.mark.parametrize(
    "bt",
    [TASK_A1_CODE_REVIEW, TASK_A2_SECURITY_ANALYSIS, TASK_A3_TEST_PLAN],
    ids=["A1", "A2", "A3"],
)
def test_to_spec_text_round_trips_through_spec_loader(bt: BenchmarkTask) -> None:
    """A1/A2/A3 (which contain ``#`` headers and code fences) must round-trip.

    This is the critical regression test for the fenced-code-block
    protection: SpecLoader's section detection treats any ``#``-prefixed
    line as a header, so the goal body must be hidden inside a fence.
    """
    text = BenchmarkTaskAdapter().to_spec_text(bt)
    spec = SpecLoader().parse(text)
    assert spec.frontmatter["task_id"] == bt.task_id
    # Every expected_output should round-trip as one acceptance bullet.
    assert len(spec.acceptance_criteria) == len(bt.expected_outputs)
    for original, parsed in zip(bt.expected_outputs, spec.acceptance_criteria):
        # Newlines collapse to spaces but leading/trailing whitespace
        # is stripped — assert the bullet contains the original text.
        assert original.replace("\n", " ").strip() == parsed
    # The first non-fence line of the goal section should appear verbatim.
    first_line = bt.input_data.splitlines()[0]
    assert first_line in spec.goal


def test_to_spec_text_escapes_input_with_triple_backticks() -> None:
    """If the input itself contains ``````` fences the adapter
    switches to a longer fence so the spec parser cannot truncate the body."""
    bt = BenchmarkTask(
        task_id="FENCE",
        category=TaskCategory.QE,
        description="fence test",
        input_data="```\nimport os\n```\n",
    )
    text = BenchmarkTaskAdapter().to_spec_text(bt)
    # Adapter must use a 4-backtick fence (one more than the longest run).
    assert "````\n" in text
    # Round-trip: SpecLoader recovers the body.
    spec = SpecLoader().parse(text)
    assert "import os" in spec.goal


def test_to_spec_text_escapes_input_with_dollar_signs() -> None:
    """``$`` is the placeholder marker for :class:`string.Template`; the
    adapter uses ``safe_substitute`` so user content with ``$1`` does not
    raise."""
    bt = BenchmarkTask(
        task_id="DOLLAR",
        category=TaskCategory.QE,
        description="cost is $5 and group is $1",
        input_data="echo $1 $PATH",
    )
    text = BenchmarkTaskAdapter().to_spec_text(bt)
    # Frontmatter must include the literal $-bearing description and the
    # body must include the literal command.
    assert "$1" in text
    assert "$PATH" in text


def test_to_spec_text_renders_frontmatter_fields() -> None:
    """Frontmatter exposes id/category/complexity for downstream tooling."""
    text = BenchmarkTaskAdapter().to_spec_text(TASK_A1_CODE_REVIEW)
    assert "task_id: A1" in text
    assert f"category: {TASK_A1_CODE_REVIEW.category.value}" in text
    assert f"complexity: {TASK_A1_CODE_REVIEW.complexity.value}" in text


def test_to_spec_text_with_empty_constraints_renders_empty_section() -> None:
    """No constraints means an empty ``# Constraints`` body — still parses."""
    bt = BenchmarkTask(
        task_id="EMPTY",
        category=TaskCategory.QE,
        description="d",
        input_data="hello",
        expected_outputs=("ok",),
    )
    text = BenchmarkTaskAdapter().to_spec_text(bt)
    spec = SpecLoader().parse(text)
    assert spec.constraints == []
    assert spec.acceptance_criteria == ["ok"]


def test_to_spec_text_renders_constraints_from_metadata() -> None:
    """Constraints in metadata appear as bullets in the spec body."""
    bt = BenchmarkTask(
        task_id="CONS",
        category=TaskCategory.QE,
        description="d",
        input_data="hello",
        metadata={"constraints": ("budget", "deadline")},
    )
    text = BenchmarkTaskAdapter().to_spec_text(bt)
    spec = SpecLoader().parse(text)
    assert spec.constraints == ["budget", "deadline"]


def test_to_spec_text_collapses_newlines_in_acceptance_bullets() -> None:
    """Multiline expected outputs collapse to single-line bullets."""
    bt = BenchmarkTask(
        task_id="MULTI",
        category=TaskCategory.QE,
        description="d",
        input_data="hello",
        expected_outputs=("line one\nline two",),
    )
    text = BenchmarkTaskAdapter().to_spec_text(bt)
    spec = SpecLoader().parse(text)
    assert spec.acceptance_criteria == ["line one line two"]


# ─────────────────────────── lookup contract ───────────────────────────


def test_lookup_returns_task_by_id_case_insensitive() -> None:
    """Both ``a1`` and ``A1`` resolve to :data:`TASK_A1_CODE_REVIEW`."""
    adapter = BenchmarkTaskAdapter()
    assert adapter.lookup("A1") is TASK_A1_CODE_REVIEW
    assert adapter.lookup("a1") is TASK_A1_CODE_REVIEW


def test_lookup_unknown_raises_keyerror_with_known_list() -> None:
    """The error message lists known ids so users can self-correct."""
    adapter = BenchmarkTaskAdapter()
    with pytest.raises(KeyError) as exc:
        adapter.lookup("Z99")
    message = str(exc.value)
    assert "Z99" in message
    assert "A1" in message  # at least one real id appears in the hint


# ───────────────────────── template-loading helpers ─────────────────────────


def test_template_loads_from_default_path_when_none_passed() -> None:
    """When ``template_path`` is ``None`` the bundled template is used."""
    adapter = BenchmarkTaskAdapter(template_path=None)
    text = adapter._load_template()
    assert "$task_id" in text
    assert "$goal" in text


def test_template_loads_from_custom_path(tmp_path: Path) -> None:
    """A caller-provided path overrides the default."""
    custom = tmp_path / "custom.md"
    custom.write_text(
        "---\ntask_id: $task_id\n---\n# Goal\n$goal\n# Acceptance Criteria\n$acceptance_criteria\n# Constraints\n$constraints\n",
        encoding="utf-8",
    )
    adapter = BenchmarkTaskAdapter(template_path=custom)
    bt = BenchmarkTask(
        task_id="CUSTOM",
        category=TaskCategory.QE,
        description="d",
        input_data="hi",
        expected_outputs=("ok",),
    )
    text = adapter.to_spec_text(bt)
    assert "task_id: CUSTOM" in text
    spec = SpecLoader().parse(text)
    assert spec.acceptance_criteria == ["ok"]


def test_template_path_resolves_relative_to_module() -> None:
    """The default template must exist on disk at runtime."""
    assert DEFAULT_TEMPLATE_PATH.is_file()
    text = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "$task_id" in text
