"""Provider-agnostic prompt templates for producer / reviewer agents.

Templates use Python ``str.format`` placeholders: ``{goal}``, ``{criteria}``,
``{constraints}``, ``{previous_draft}``, ``{review_feedback}``, ``{draft}``.
Empty optional sections are elided by the rendering helpers.

All user-controlled strings are escaped (``{`` → ``{{``, ``}`` → ``}}``)
before being substituted so that braces in the input cannot collide with
:py:meth:`str.format` placeholders.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.framework.harness.state import AcceptanceCriterion, Task

PRODUCER_SYSTEM_PROMPT: str = (
    "You are a careful, structured producer agent. Read the goal, acceptance criteria, "
    "and constraints. Produce a complete, well-organized response that satisfies every "
    "acceptance criterion. If reviewer feedback from a previous round is provided, "
    "address each point explicitly."
)

PRODUCER_USER_TEMPLATE: str = """\
# Goal
{goal}

{criteria_block}{constraints_block}{feedback_block}
Produce your response now."""

REVIEWER_SYSTEM_PROMPT: str = (
    "You are a strict reviewer. Compare the producer's draft against the goal, "
    "acceptance criteria, and constraints. Reply with EXACTLY this format:\n\n"
    "ACCEPT\n"
    "score: <0.0-1.0>\n"
    "notes: <one or two sentences>\n\n"
    "OR\n\n"
    "REJECT\n"
    "score: <0.0-1.0>\n"
    "issues:\n"
    "- <criterion-id or short label>: <what is wrong>\n"
    "- ...\n"
    "suggestions:\n"
    "- <concrete revision>\n"
    "- ...\n\n"
    "Use ACCEPT only if every criterion is met."
)

REVIEWER_USER_TEMPLATE: str = """\
# Goal
{goal}

{criteria_block}{constraints_block}
# Draft
{draft}

Now review."""

# Marker substring placed by the producer-reviewer topology in the reviewer's
# prompt when no explicit draft is forwarded via metadata. Used as a final
# fall-back when extracting the draft to review.
_DRAFT_MARKER: str = "DRAFT:\n"


def _escape_braces(value: str) -> str:
    """Escape ``{`` and ``}`` so user input can be safely passed to ``str.format``."""
    return value.replace("{", "{{").replace("}", "}}")


def render_criteria_block(criteria: Iterable[AcceptanceCriterion]) -> str:
    """Render the acceptance criteria as a markdown bullet list.

    Returns the empty string when ``criteria`` is empty.
    """
    items = list(criteria)
    if not items:
        return ""
    lines = ["# Acceptance Criteria"]
    for criterion in items:
        lines.append(f"- [{criterion.id}] {criterion.description}")
    return "\n".join(lines) + "\n\n"


def render_constraints_block(constraints: Iterable[str]) -> str:
    """Render constraints as a markdown bullet list.

    Returns the empty string when ``constraints`` is empty.
    """
    items = [c for c in constraints if c]
    if not items:
        return ""
    lines = ["# Constraints"]
    for constraint in items:
        lines.append(f"- {constraint}")
    return "\n".join(lines) + "\n\n"


def render_feedback_block(
    previous_draft: str | None,
    review_feedback: str | None,
) -> str:
    """Render previous-round draft + reviewer feedback (omitted when both empty)."""
    has_draft = bool(previous_draft and previous_draft.strip())
    has_feedback = bool(review_feedback and review_feedback.strip())
    if not (has_draft or has_feedback):
        return ""
    parts: list[str] = ["# Previous Round"]
    if has_draft:
        parts.append("Draft:")
        parts.append(str(previous_draft).rstrip())
    if has_feedback:
        if has_draft:
            parts.append("")  # blank line between draft and feedback
        parts.append("Reviewer feedback:")
        parts.append(str(review_feedback).rstrip())
    return "\n".join(parts) + "\n\n"


def render_producer_user_message(task: Task) -> str:
    """Render the user message handed to the producer.

    Empty optional sections (criteria, constraints, feedback) are elided so
    the resulting prompt has no awkward blank headings.
    """
    criteria_block = render_criteria_block(task.acceptance_criteria)
    constraints_block = render_constraints_block(task.constraints)
    feedback_block = render_feedback_block(
        task.metadata.get("previous_draft"),
        task.metadata.get("review_feedback"),
    )
    return PRODUCER_USER_TEMPLATE.format(
        goal=_escape_braces(task.goal),
        criteria_block=criteria_block,
        constraints_block=constraints_block,
        feedback_block=feedback_block,
    )


def render_reviewer_user_message(task: Task, *, draft: str | None = None) -> str:
    """Render the user message handed to the reviewer.

    The draft is sourced in this order:

    1. Explicit ``draft`` keyword argument.
    2. ``task.metadata['draft']`` (the producer-reviewer topology stores it
       there for review rounds).
    3. Substring scraped after a literal ``"DRAFT:\\n"`` marker in
       ``task.goal``. This is what the topology embeds when no metadata
       channel is wired up.
    4. Empty string fall-back.
    """
    if draft is None:
        metadata_draft = task.metadata.get("draft")
        if isinstance(metadata_draft, str) and metadata_draft:
            draft = metadata_draft
        else:
            marker_index = task.goal.find(_DRAFT_MARKER)
            if marker_index >= 0:
                draft = task.goal[marker_index + len(_DRAFT_MARKER) :]
            else:
                draft = ""

    criteria_block = render_criteria_block(task.acceptance_criteria)
    constraints_block = render_constraints_block(task.constraints)
    return REVIEWER_USER_TEMPLATE.format(
        goal=_escape_braces(task.goal),
        criteria_block=criteria_block,
        constraints_block=constraints_block,
        draft=_escape_braces(draft),
    )


__all__ = [
    "PRODUCER_SYSTEM_PROMPT",
    "PRODUCER_USER_TEMPLATE",
    "REVIEWER_SYSTEM_PROMPT",
    "REVIEWER_USER_TEMPLATE",
    "render_constraints_block",
    "render_criteria_block",
    "render_feedback_block",
    "render_producer_user_message",
    "render_reviewer_user_message",
]
