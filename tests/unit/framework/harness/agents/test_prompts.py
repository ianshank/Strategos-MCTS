"""Unit tests for the provider-agnostic prompt templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.harness.agents import prompts
from src.framework.harness.agents.prompts import (
    render_constraints_block,
    render_criteria_block,
    render_feedback_block,
    render_producer_user_message,
    render_reviewer_user_message,
)
from src.framework.harness.state import AcceptanceCriterion, Task

pytestmark = pytest.mark.unit


def _task(
    *,
    goal: str = "Build a thing",
    criteria: tuple[AcceptanceCriterion, ...] = (),
    constraints: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> Task:
    return Task(
        id="t-1",
        goal=goal,
        acceptance_criteria=criteria,
        constraints=constraints,
        metadata=metadata or {},
    )


# --- render_criteria_block ---------------------------------------------------


def test_render_criteria_block_empty_returns_empty_string() -> None:
    assert render_criteria_block(()) == ""
    assert render_criteria_block([]) == ""


def test_render_criteria_block_formats_each_criterion_with_id_and_description() -> None:
    block = render_criteria_block(
        [
            AcceptanceCriterion(id="C1", description="must compile"),
            AcceptanceCriterion(id="C2", description="must pass tests"),
        ]
    )
    assert block.startswith("# Acceptance Criteria\n")
    assert "- [C1] must compile" in block
    assert "- [C2] must pass tests" in block
    # Trailing blank line so subsequent blocks render cleanly.
    assert block.endswith("\n\n")


# --- render_constraints_block ------------------------------------------------


def test_render_constraints_block_empty_returns_empty_string() -> None:
    assert render_constraints_block(()) == ""
    assert render_constraints_block([""]) == ""


def test_render_constraints_block_formats_bullets() -> None:
    block = render_constraints_block(["no network", "deterministic"])
    assert block.startswith("# Constraints\n")
    assert "- no network" in block
    assert "- deterministic" in block
    assert block.endswith("\n\n")


# --- render_feedback_block ---------------------------------------------------


def test_render_feedback_block_empty_when_both_none() -> None:
    assert render_feedback_block(None, None) == ""
    assert render_feedback_block("", "") == ""
    assert render_feedback_block("   ", "\n\t") == ""


def test_render_feedback_block_with_only_previous_draft() -> None:
    block = render_feedback_block("old draft", None)
    assert "Draft:" in block
    assert "old draft" in block
    assert "Reviewer feedback:" not in block
    assert block.endswith("\n\n")


def test_render_feedback_block_with_only_feedback() -> None:
    block = render_feedback_block(None, "fix the imports")
    assert "Reviewer feedback:" in block
    assert "fix the imports" in block
    assert "Draft:" not in block


def test_render_feedback_block_with_both() -> None:
    block = render_feedback_block("old", "fix")
    assert "Draft:" in block
    assert "Reviewer feedback:" in block
    assert "old" in block
    assert "fix" in block


# --- render_producer_user_message --------------------------------------------


def test_render_producer_user_message_includes_goal_and_blocks() -> None:
    task = _task(
        goal="Write a haiku",
        criteria=(AcceptanceCriterion(id="A1", description="5-7-5 syllables"),),
        constraints=("no profanity",),
        metadata={"previous_draft": "old try", "review_feedback": "more vivid"},
    )
    msg = render_producer_user_message(task)
    assert "# Goal" in msg
    assert "Write a haiku" in msg
    assert "# Acceptance Criteria" in msg
    assert "[A1] 5-7-5 syllables" in msg
    assert "# Constraints" in msg
    assert "no profanity" in msg
    assert "# Previous Round" in msg
    assert "old try" in msg
    assert "more vivid" in msg
    assert msg.rstrip().endswith("Produce your response now.")


def test_render_producer_user_message_omits_optional_when_empty() -> None:
    task = _task(goal="Just do it")
    msg = render_producer_user_message(task)
    assert "Just do it" in msg
    assert "# Acceptance Criteria" not in msg
    assert "# Constraints" not in msg
    assert "# Previous Round" not in msg
    assert msg.rstrip().endswith("Produce your response now.")


# --- render_reviewer_user_message --------------------------------------------


def test_render_reviewer_user_message_uses_explicit_draft_arg() -> None:
    task = _task(goal="Goal text", metadata={"draft": "metadata-draft"})
    msg = render_reviewer_user_message(task, draft="explicit-draft")
    assert "explicit-draft" in msg
    assert "metadata-draft" not in msg
    assert "# Draft" in msg
    assert msg.rstrip().endswith("Now review.")


def test_render_reviewer_user_message_uses_metadata_draft_when_no_explicit_arg() -> None:
    task = _task(goal="Goal text", metadata={"draft": "from-metadata"})
    msg = render_reviewer_user_message(task)
    assert "from-metadata" in msg


def test_render_reviewer_user_message_falls_back_to_scraping_goal_after_DRAFT_marker() -> None:
    goal = (
        "Review the following draft against the task. "
        "Reply with ACCEPT or REJECT plus feedback.\n\nDRAFT:\nthe-actual-draft-body"
    )
    task = _task(goal=goal)
    msg = render_reviewer_user_message(task)
    assert "the-actual-draft-body" in msg


def test_render_reviewer_user_message_has_empty_draft_when_nothing_available() -> None:
    task = _task(goal="just a goal")
    msg = render_reviewer_user_message(task)
    # Section header still rendered, body is empty.
    assert "# Draft" in msg
    assert "Now review." in msg


# --- escaping ----------------------------------------------------------------


def test_render_escapes_curly_braces_in_user_data() -> None:
    """Inputs containing literal ``{`` / ``}`` must not crash ``str.format``."""
    task = _task(goal="format like {this} not {that}", metadata={"draft": "use {value} here"})
    producer_msg = render_producer_user_message(task)
    reviewer_msg = render_reviewer_user_message(task)
    assert "{this}" in producer_msg
    assert "{that}" in producer_msg
    assert "{value}" in reviewer_msg


# --- provider-agnostic guard -------------------------------------------------


def test_no_provider_specific_strings_in_module() -> None:
    source = Path(prompts.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("phi", "lmstudio", "openai", "anthropic"):
        assert forbidden not in source, f"Provider-specific string {forbidden!r} leaked into prompts.py"
