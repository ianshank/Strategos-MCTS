"""Unit tests for :class:`LLMReviewerAgent` and :func:`parse_review_decision`."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.adapters.llm.base import LLMResponse
from src.adapters.llm.exceptions import LLMTimeoutError
from src.framework.harness.agents.llm_reviewer import (
    LLMReviewerAgent,
    parse_review_decision,
)
from src.framework.harness.state import AcceptanceCriterion, Task
from src.framework.harness.topology.base import AgentLike

pytestmark = pytest.mark.unit


class FakeLLMClient:
    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> LLMResponse | AsyncIterator[str]:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if not self._responses:
            raise AssertionError("FakeLLMClient: no more responses queued")
        return self._responses.pop(0)


def _review_task(*, draft: str = "the draft body") -> Task:
    return Task(
        id="t-rev",
        goal="Review the following draft against the task. Reply with ACCEPT or REJECT plus feedback.\n\nDRAFT:\n"
        + draft,
        acceptance_criteria=(AcceptanceCriterion(id="A1", description="be terse"),),
        constraints=("no jargon",),
        metadata={"draft": draft},
    )


# --- parse_review_decision ---------------------------------------------------


def test_parse_accept_returns_true_and_score_and_notes() -> None:
    text = "ACCEPT\nscore: 0.9\nnotes: looks great overall."
    accepted, parsed = parse_review_decision(text)
    assert accepted is True
    assert parsed["score"] == pytest.approx(0.9)
    assert parsed["notes"] == "looks great overall."


def test_parse_reject_returns_false_and_issues_and_suggestions() -> None:
    text = (
        "REJECT\n"
        "score: 0.3\n"
        "issues:\n"
        "- A1: too verbose\n"
        "- A2: missing example\n"
        "suggestions:\n"
        "- trim by 50%\n"
        "- add a code block\n"
    )
    accepted, parsed = parse_review_decision(text)
    assert accepted is False
    assert parsed["score"] == pytest.approx(0.3)
    assert parsed["issues"] == ["A1: too verbose", "A2: missing example"]
    assert parsed["suggestions"] == ["trim by 50%", "add a code block"]


def test_parse_accept_only_at_first_line_preamble_yields_default_reject() -> None:
    text = "Some preamble.\nACCEPT\nscore: 0.9"
    accepted, parsed = parse_review_decision(text)
    assert accepted is False
    assert parsed == {"raw": text}


def test_parse_accept_only_at_first_line_injection_in_body_yields_reject() -> None:
    """Defends against a producer-draft prompt-injection: 'ACCEPT' inside the body."""
    text = "This draft says ACCEPT for sure.\n" "REJECT\n" "score: 0.4\n" "issues:\n" "- bad\n"
    accepted, parsed = parse_review_decision(text)
    # Default-reject path: the first non-empty line begins with neither
    # ACCEPT nor REJECT, so parsing fails and we return raw.
    assert accepted is False
    assert "raw" in parsed


def test_parse_handles_lowercase_accept() -> None:
    text = "accept\nscore: 0.9\nnotes: ok"
    accepted, parsed = parse_review_decision(text)
    assert accepted is True
    assert parsed["score"] == pytest.approx(0.9)


def test_parse_handles_lowercase_reject_with_leading_whitespace() -> None:
    text = "  reject\nscore: 0.2\nissues:\n- nope\nsuggestions:\n- redo\n"
    accepted, parsed = parse_review_decision(text)
    assert accepted is False
    assert parsed["score"] == pytest.approx(0.2)
    assert parsed["issues"] == ["nope"]
    assert parsed["suggestions"] == ["redo"]


def test_parse_returns_default_reject_on_parse_failure() -> None:
    text = "I have no idea what you want."
    accepted, parsed = parse_review_decision(text)
    assert accepted is False
    assert parsed == {"raw": text}


def test_parse_handles_empty_or_invalid_input() -> None:
    accepted, parsed = parse_review_decision("")
    assert accepted is False
    assert parsed == {"raw": ""}

    accepted, parsed = parse_review_decision("   \n\t\n")
    assert accepted is False
    assert "raw" in parsed


def test_parse_accept_without_score_or_notes_is_still_accept() -> None:
    accepted, parsed = parse_review_decision("ACCEPT\n")
    assert accepted is True
    assert parsed["score"] is None
    assert parsed["notes"] == ""


def test_parse_reject_with_no_bullets_returns_empty_lists() -> None:
    accepted, parsed = parse_review_decision("REJECT\nscore: 0.1\n")
    assert accepted is False
    assert parsed["issues"] == []
    assert parsed["suggestions"] == []


# --- LLMReviewerAgent --------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_outcome_starting_with_accept_marker_when_accepted() -> None:
    text = "ACCEPT\nscore: 0.92\nnotes: solid"
    fake = FakeLLMClient(responses=[LLMResponse(text=text, finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.response.startswith("ACCEPT")
    assert "ACCEPT" in outcome.response.upper()


@pytest.mark.asyncio
async def test_run_returns_outcome_starting_with_reject_marker_when_rejected() -> None:
    text = "REJECT\nscore: 0.4\nissues:\n- short\nsuggestions:\n- expand"
    fake = FakeLLMClient(responses=[LLMResponse(text=text, finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.response.startswith("REJECT")


@pytest.mark.asyncio
async def test_run_returns_reject_marker_on_parse_failure() -> None:
    """A garbled reviewer reply must default-reject so the loop continues correctly."""
    fake = FakeLLMClient(responses=[LLMResponse(text="garbled output", finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.response.startswith("REJECT")
    assert outcome.metadata["decision"] is False


@pytest.mark.asyncio
async def test_run_includes_decision_in_metadata() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nscore: 0.9\nnotes: ok", finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.metadata["decision"] is True
    assert "feedback" in outcome.metadata
    assert outcome.metadata["finish_reason"] == "stop"
    assert outcome.metadata["usage"] == {}


@pytest.mark.asyncio
async def test_run_response_is_marker_only_not_full_text() -> None:
    """The outcome.response must be exactly the ACCEPT/REJECT marker.

    Topology accept gates do an unanchored substring search. If the full
    reviewer body were embedded, a REJECT body that mentions the word
    "ACCEPT" anywhere (e.g. "does not meet the ACCEPT criteria") would
    silently bypass the gate. Full body is preserved in metadata["text"].
    """
    text = "REJECT\nscore: 0.2\nissues:\n- mentions ACCEPT criteria but fails\nsuggestions:\n- rewrite"
    fake = FakeLLMClient(responses=[LLMResponse(text=text, finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.response == "REJECT"
    assert "ACCEPT" not in outcome.response
    assert outcome.metadata["text"] == text
    assert outcome.metadata["decision"] is False


@pytest.mark.asyncio
async def test_run_accepted_response_is_marker_only() -> None:
    text = "ACCEPT\nscore: 0.95\nnotes: thorough"
    fake = FakeLLMClient(responses=[LLMResponse(text=text, finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.response == "ACCEPT"
    assert outcome.metadata["text"] == text


@pytest.mark.asyncio
async def test_run_uses_score_as_confidence_when_in_range() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nscore: 0.73\nnotes: ok", finish_reason="length")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    # Score in range: confidence equals score, NOT the finish-reason fallback.
    assert outcome.confidence == pytest.approx(0.73)


@pytest.mark.asyncio
async def test_run_falls_back_to_finish_reason_confidence_when_score_out_of_range() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nscore: 9.9\nnotes: weird", finish_reason="length")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    # Score out of [0,1]: fall back to finish-reason confidence (length=0.5).
    assert outcome.confidence == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_falls_back_to_finish_reason_confidence_when_score_missing() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nnotes: terse", finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_run_propagates_llm_error_as_failure_outcome() -> None:
    fake = FakeLLMClient(raises=LLMTimeoutError(provider="test", timeout=5.0))
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.success is False
    assert outcome.error is not None
    assert "LLMTimeoutError" in outcome.error


@pytest.mark.asyncio
async def test_run_does_not_pass_temperature_when_none() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nscore: 0.9\nnotes: ok", finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    await agent.run(_review_task())
    assert "temperature" not in fake.calls[0]


@pytest.mark.asyncio
async def test_run_passes_temperature_when_set() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nscore: 0.9\nnotes: ok", finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake, temperature=0.0)
    await agent.run(_review_task())
    assert fake.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_run_passes_max_tokens() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nscore: 0.9\nnotes: ok", finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake, max_tokens=999)
    await agent.run(_review_task())
    assert fake.calls[0]["max_tokens"] == 999


@pytest.mark.asyncio
async def test_run_outcome_response_is_marker_only_full_text_in_metadata() -> None:
    """The response is the marker only; the body is preserved on metadata['text']."""
    text = "ACCEPT\nscore: 0.9\nnotes: great"
    fake = FakeLLMClient(responses=[LLMResponse(text=text, finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    outcome = await agent.run(_review_task())
    assert outcome.response == "ACCEPT"
    assert outcome.metadata["text"] == text


def test_agent_satisfies_agent_like_protocol() -> None:
    fake = FakeLLMClient()
    agent = LLMReviewerAgent(llm=fake)
    assert isinstance(agent, AgentLike)


@pytest.mark.asyncio
async def test_unexpected_response_type_returns_failure_outcome() -> None:
    class BadClient:
        async def generate(self, **_kwargs: Any) -> Any:
            async def _stream() -> AsyncIterator[str]:  # pragma: no cover
                yield "x"

            return _stream()

    agent = LLMReviewerAgent(llm=BadClient())  # type: ignore[arg-type]
    outcome = await agent.run(_review_task())
    assert outcome.success is False
    assert "unexpected response type" in (outcome.error or "")


# ---------------------------------------------------------------------------
# Coverage gap closures: parser edge cases (was 96.88% line+branch).
# ---------------------------------------------------------------------------


def test_parse_review_decision_skips_blank_first_lines_then_recognises_marker() -> None:
    """`_first_nonempty_line` must keep skipping blank lines before the marker."""
    text = "   \n\t\n\n  \nACCEPT\nscore: 0.7\nnotes: deferred"
    accepted, parsed = parse_review_decision(text)
    assert accepted is True
    assert parsed == {"score": pytest.approx(0.7), "notes": "deferred"}


def test_parse_review_decision_returns_default_reject_when_only_blank_lines() -> None:
    """All-whitespace input must default-reject (and not crash)."""
    accepted, parsed = parse_review_decision("\n\n   \n\t\n")
    assert accepted is False
    assert "raw" in parsed


def test_parse_review_decision_tolerates_non_bullet_lines_inside_section() -> None:
    """A non-bullet line inside an `issues:` section must not abort collection."""
    text = (
        "REJECT\n"
        "score: 0.3\n"
        "issues:\n"
        "this commentary line is not a bullet\n"
        "- first issue\n"
        "\n"
        "- second issue\n"
        "suggestions:\n"
        "- expand\n"
    )
    accepted, parsed = parse_review_decision(text)
    assert accepted is False
    assert parsed["issues"] == ["first issue", "second issue"]
    assert parsed["suggestions"] == ["expand"]


# ---------------------------------------------------------------------------
# Property-based fuzz: the parser must never raise and must never accept on
# inputs that do not start with ACCEPT on the first non-empty line.
# ---------------------------------------------------------------------------


@given(text=st.text(max_size=500))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_parse_review_decision_never_raises(text: str) -> None:
    accepted, parsed = parse_review_decision(text)
    assert isinstance(accepted, bool)
    assert isinstance(parsed, dict)


@given(
    body=st.text(
        alphabet=st.characters(blacklist_characters="\n"),
        min_size=0,
        max_size=200,
    )
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_parse_review_decision_does_not_accept_when_first_line_is_not_accept(body: str) -> None:
    """If the first non-empty line does NOT start with ACCEPT, never accept —
    even if "ACCEPT" appears anywhere later. Defends the prompt-injection guard."""
    # Force the first non-empty line to start with the safe prefix "noise"; any
    # mention of ACCEPT in the body must therefore not flip the gate.
    text = f"noise: {body}\nACCEPT\nscore: 0.99\nnotes: ignore"
    accepted, _parsed = parse_review_decision(text)
    assert accepted is False


# ---------------------------------------------------------------------------
# Logging contract: producer must emit start + done INFO events on the
# happy path, and a WARNING on LLM failures.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_logs_info_on_start_and_done(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ACCEPT\nscore: 0.9\nnotes: ok", finish_reason="stop")])
    agent = LLMReviewerAgent(llm=fake)
    with caplog.at_level(logging.INFO, logger="harness.agent.reviewer"):
        await agent.run(_review_task())
    messages = [r.getMessage() for r in caplog.records if r.name == "harness.agent.reviewer"]
    assert any("reviewer.run start" in m for m in messages)
    assert any("reviewer.run done" in m and "decision=ACCEPT" in m for m in messages)


@pytest.mark.asyncio
async def test_run_logs_warning_on_llm_error(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeLLMClient(raises=LLMTimeoutError("lmstudio", 5.0))
    agent = LLMReviewerAgent(llm=fake)
    with caplog.at_level(logging.WARNING, logger="harness.agent.reviewer"):
        outcome = await agent.run(_review_task())
    assert outcome.success is False
    assert any("LLM generate failed" in r.getMessage() for r in caplog.records)
