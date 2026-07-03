"""Unit tests for intent normalisation and SPEC parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.harness import HarnessSettings
from src.framework.harness.intent import DefaultIntentNormalizer, SpecCriterion, SpecLoader, SpecParseError

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_normalize_string_intent() -> None:
    """A string becomes a Task with ``goal`` populated."""
    n = DefaultIntentNormalizer()
    task = await n.normalize("ship the feature", HarnessSettings())
    assert task.goal == "ship the feature"
    assert task.id


@pytest.mark.asyncio
async def test_normalize_dict_intent_with_criteria() -> None:
    """Dict input copies criteria, constraints, and metadata."""
    n = DefaultIntentNormalizer()
    task = await n.normalize(
        {
            "id": "T1",
            "goal": "fix bug",
            "acceptance_criteria": [
                "lints clean",
                {"id": "c2", "description": "tests pass", "check": "passed"},
            ],
            "constraints": ["no new deps"],
            "metadata": {"owner": "alice"},
        },
        HarnessSettings(),
    )
    assert task.id == "T1"
    assert len(task.acceptance_criteria) == 2
    assert task.acceptance_criteria[0].id == "c0"
    assert task.acceptance_criteria[1].check == "passed"
    assert task.constraints == ("no new deps",)
    assert task.metadata == {"owner": "alice"}


@pytest.mark.asyncio
async def test_normalize_rejects_empty() -> None:
    """An empty intent is a programming error."""
    n = DefaultIntentNormalizer()
    with pytest.raises(ValueError):
        await n.normalize("   ", HarnessSettings())
    with pytest.raises(ValueError):
        await n.normalize({"goal": ""}, HarnessSettings())


@pytest.mark.asyncio
async def test_normalize_unsupported_type() -> None:
    """Non-str, non-dict payloads are rejected."""
    n = DefaultIntentNormalizer()
    with pytest.raises(TypeError):
        await n.normalize(42, HarnessSettings())  # type: ignore[arg-type]


def test_spec_loader_parses_frontmatter_and_sections(tmp_path: Path) -> None:
    """Frontmatter, goal, criteria, and constraints are extracted."""
    text = (
        "---\n"
        "owner: alice\n"
        "version: 1\n"
        "---\n"
        "# Goal\n"
        "Add the feature.\n\n"
        "# Acceptance Criteria\n"
        "- ruff clean\n"
        "- tests pass\n\n"
        "# Constraints\n"
        "- no new deps\n"
    )
    spec_file = tmp_path / "spec.md"
    spec_file.write_text(text)
    spec = SpecLoader().load(spec_file)
    assert spec.goal == "Add the feature."
    assert spec.acceptance_criteria == ["ruff clean", "tests pass"]
    assert spec.constraints == ["no new deps"]
    assert spec.frontmatter == {"owner": "alice", "version": "1"}


def test_spec_loader_handles_missing_frontmatter() -> None:
    """No frontmatter is fine — body is parsed as-is."""
    spec = SpecLoader().parse("# Goal\nDo stuff.\n# Acceptance\n- a\n")
    assert spec.goal == "Do stuff."
    assert spec.acceptance_criteria == ["a"]


def test_spec_loader_missing_file_raises(tmp_path: Path) -> None:
    """Missing files raise a clear, typed error."""
    with pytest.raises(SpecParseError):
        SpecLoader().load(tmp_path / "absent.md")


def test_spec_loader_handles_multi_digit_numbered_lists() -> None:
    """Numbered lists with two-or-more-digit indices must parse correctly."""
    text = "# Acceptance Criteria\n" "1. first\n" "2. second\n" "10. tenth\n" "11) eleventh\n" "100. hundredth\n"
    spec = SpecLoader().parse(text)
    assert spec.acceptance_criteria == ["first", "second", "tenth", "eleventh", "hundredth"]


def test_spec_loader_mixes_bullets_and_numbers() -> None:
    """A section with mixed bullet and numeric markers is parsed faithfully."""
    text = "# Constraints\n" "- bulleted\n" "12. twelve\n" "* asterisk\n" "13) thirteen\n"
    spec = SpecLoader().parse(text)
    assert spec.constraints == ["bulleted", "twelve", "asterisk", "thirteen"]


# ---------------------------------------------------------------------------
# Schema v2: frontmatter fields, authored criterion IDs, new sections
# ---------------------------------------------------------------------------


def test_spec_loader_parses_v2_frontmatter_fields() -> None:
    """id/module/status/supersedes populate from frontmatter, non-destructively."""
    text = (
        "---\n"
        "id: my_spec\n"
        "goal: Do it\n"
        "module: src/api/\n"
        "status: approved\n"
        "supersedes: old_spec\n"
        "---\n"
        "# Acceptance Criteria\n- AC-1: works\n"
    )
    spec = SpecLoader().parse(text)
    assert spec.id == "my_spec"
    assert spec.module == "src/api/"
    assert spec.status == "approved"
    assert spec.supersedes == "old_spec"
    # Frontmatter reads are non-destructive: the dict keeps every key.
    assert spec.frontmatter["id"] == "my_spec"
    assert spec.frontmatter["status"] == "approved"


def test_spec_loader_v2_fields_default_empty() -> None:
    """Legacy specs without v2 frontmatter read empty strings, not errors."""
    spec = SpecLoader().parse("# Goal\nDo stuff.\n# Acceptance Criteria\n- a\n")
    assert spec.id == ""
    assert spec.module == ""
    assert spec.status == ""
    assert spec.supersedes == ""


def test_spec_loader_extracts_authored_criterion_ids() -> None:
    """``- AC-n: ...`` bullets yield authored IDs and prefix-stripped descriptions."""
    spec = SpecLoader().parse("# Acceptance Criteria\n- AC-1: ruff clean\n- AC-2: tests pass\n")
    assert spec.criteria == [
        SpecCriterion(id="AC-1", description="ruff clean"),
        SpecCriterion(id="AC-2", description="tests pass"),
    ]
    assert spec.acceptance_criteria == ["ruff clean", "tests pass"]
    assert spec.criteria_payload() == [
        {"id": "AC-1", "description": "ruff clean"},
        {"id": "AC-2", "description": "tests pass"},
    ]


def test_spec_loader_positional_fallback_ids() -> None:
    """Unprefixed bullets keep the historical positional IDs in order."""
    spec = SpecLoader().parse("# Acceptance Criteria\n- one\n- two\n")
    assert [c.id for c in spec.criteria] == ["c0", "c1"]
    assert spec.acceptance_criteria == ["one", "two"]


def test_spec_loader_mixed_bullets_keep_authored_and_fallback() -> None:
    """The parser is tolerant of mixed prefixes; the validator is the gate."""
    spec = SpecLoader().parse("# Acceptance Criteria\n- AC-1: one\n- two\n")
    assert [c.id for c in spec.criteria] == ["AC-1", "c1"]


def test_spec_loader_constraints_keep_ac_looking_prefixes() -> None:
    """AC-ID stripping applies to acceptance bullets only — constraints stay verbatim."""
    spec = SpecLoader().parse("# Acceptance Criteria\n- AC-1: one\n# Constraints\n- AC-1: not a criterion\n")
    assert spec.constraints == ["AC-1: not a criterion"]


def test_spec_loader_parses_invariants_and_out_of_scope() -> None:
    """The optional v2 sections extract as bullet lists."""
    text = (
        "# Goal\nDo it.\n"
        "# Acceptance Criteria\n- AC-1: works\n"
        "# Invariants\n- coverage never drops\n"
        "# Out of Scope\n- hooks\n- slash commands\n"
    )
    spec = SpecLoader().parse(text)
    assert spec.invariants == ["coverage never drops"]
    assert spec.out_of_scope == ["hooks", "slash commands"]


def test_spec_loader_unterminated_frontmatter_is_body() -> None:
    """A ``---`` opener with no closing delimiter is treated as plain body."""
    spec = SpecLoader().parse("---\nid: x\n# Goal\nDo it.\n")
    assert spec.frontmatter == {}
    assert spec.id == ""
    assert spec.body == spec.raw


def test_spec_loader_sections_are_fence_aware() -> None:
    """A ``#`` comment inside a ``` fence is section content, not a new header."""
    text = (
        "# Goal\nDo it.\n"
        "# Acceptance Criteria\n- AC-1: works\n"
        "```bash\n# not a header\necho hi\n```\n"
        "- AC-2: still in the same section\n"
    )
    spec = SpecLoader().parse(text)
    assert [c.id for c in spec.criteria] == ["AC-1", "AC-2"]
    assert "# not a header" in spec.sections["acceptance criteria"]


def test_spec_loader_body_excludes_frontmatter() -> None:
    """``Spec.body`` is the frontmatter-stripped text (header walks must use it)."""
    text = "---\n# a frontmatter comment\nid: x\n---\n# Goal\nDo it.\n"
    spec = SpecLoader().parse(text)
    assert "# a frontmatter comment" not in spec.body
    assert spec.body.startswith("# Goal")


@pytest.mark.asyncio
async def test_migrated_criterion_still_verifies() -> None:
    """Prefix stripping is load-bearing: the verifier uses descriptions as match
    needles when ``check`` is empty, so a migrated ``AC-n:`` criterion must match
    an observation containing only the description text."""
    from src.framework.harness.state import Observation
    from src.framework.harness.verifier import AcceptanceCriteriaVerifier

    spec = SpecLoader().parse("# Goal\nDo it.\n# Acceptance Criteria\n- AC-1: lints clean\n")
    task = await DefaultIntentNormalizer().normalize(
        {"id": "T", "goal": spec.goal, "acceptance_criteria": spec.criteria_payload()},
        HarnessSettings(),
    )
    assert task.acceptance_criteria[0].id == "AC-1"
    obs = (Observation(invocation_id="i", tool_name="t", success=True, payload="output: lints clean"),)
    result = await AcceptanceCriteriaVerifier().verify(obs, task, None)
    assert result.passed is True
