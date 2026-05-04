"""Provider-agnostic LLM agents for the harness producer-reviewer topology.

This package exposes two ready-to-use ``AgentLike`` implementations and the
prompt templates they use. Both agents are constructed from any object
satisfying the :class:`~src.adapters.llm.base.LLMClient` protocol and
therefore work uniformly against every provider that implements it.
"""

from __future__ import annotations

from src.framework.harness.agents.llm_producer import LLMProducerAgent
from src.framework.harness.agents.llm_reviewer import LLMReviewerAgent, parse_review_decision
from src.framework.harness.agents.prompts import (
    PRODUCER_SYSTEM_PROMPT,
    PRODUCER_USER_TEMPLATE,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    render_constraints_block,
    render_criteria_block,
    render_feedback_block,
    render_producer_user_message,
    render_reviewer_user_message,
)

__all__ = [
    "LLMProducerAgent",
    "LLMReviewerAgent",
    "PRODUCER_SYSTEM_PROMPT",
    "PRODUCER_USER_TEMPLATE",
    "REVIEWER_SYSTEM_PROMPT",
    "REVIEWER_USER_TEMPLATE",
    "parse_review_decision",
    "render_constraints_block",
    "render_criteria_block",
    "render_feedback_block",
    "render_producer_user_message",
    "render_reviewer_user_message",
]
