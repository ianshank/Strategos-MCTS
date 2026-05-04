"""Unit tests for :class:`LLMProducerAgent`."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from src.adapters.llm.base import LLMResponse
from src.adapters.llm.exceptions import LLMTimeoutError
from src.framework.harness.agents.llm_producer import LLMProducerAgent
from src.framework.harness.state import AcceptanceCriterion, Task
from src.framework.harness.topology.base import AgentLike, AgentOutcome

pytestmark = pytest.mark.unit


class FakeLLMClient:
    """Minimal ``LLMClient``-shaped test double.

    Records every ``generate`` call's keyword arguments and either returns the
    next pre-loaded response or raises a pre-loaded exception.
    """

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


def _basic_task(**overrides: Any) -> Task:
    base = {
        "id": "t-prod",
        "goal": "Draft something",
        "acceptance_criteria": (AcceptanceCriterion(id="A1", description="be terse"),),
        "constraints": ("no jargon",),
        "metadata": {},
    }
    base.update(overrides)
    return Task(**base)


@pytest.mark.asyncio
async def test_run_returns_outcome_with_response_text() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="hello world", finish_reason="stop", usage={"total_tokens": 12})])
    agent = LLMProducerAgent(llm=fake)
    outcome = await agent.run(_basic_task())
    assert isinstance(outcome, AgentOutcome)
    assert outcome.agent_name == "producer"
    assert outcome.response == "hello world"
    assert outcome.success is True
    assert outcome.confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_run_marks_failure_on_empty_response() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="", finish_reason="stop")])
    agent = LLMProducerAgent(llm=fake)
    outcome = await agent.run(_basic_task())
    assert outcome.response == ""
    assert outcome.success is False


@pytest.mark.asyncio
async def test_run_propagates_llm_error_as_outcome_with_error_field() -> None:
    fake = FakeLLMClient(raises=LLMTimeoutError(provider="test", timeout=5.0))
    agent = LLMProducerAgent(llm=fake)
    outcome = await agent.run(_basic_task())
    assert outcome.success is False
    assert outcome.error is not None
    assert "LLMTimeoutError" in outcome.error
    assert outcome.confidence == 0.0


@pytest.mark.asyncio
async def test_run_passes_max_tokens() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ok", finish_reason="stop")])
    agent = LLMProducerAgent(llm=fake, max_tokens=321)
    await agent.run(_basic_task())
    assert fake.calls[0]["max_tokens"] == 321


@pytest.mark.asyncio
async def test_run_does_not_pass_temperature_when_none() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ok", finish_reason="stop")])
    agent = LLMProducerAgent(llm=fake)  # default: temperature=None
    await agent.run(_basic_task())
    assert "temperature" not in fake.calls[0]


@pytest.mark.asyncio
async def test_run_passes_temperature_when_set() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="ok", finish_reason="stop")])
    agent = LLMProducerAgent(llm=fake, temperature=0.25)
    await agent.run(_basic_task())
    assert fake.calls[0]["temperature"] == 0.25


@pytest.mark.asyncio
async def test_run_includes_review_feedback_in_user_message() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="revised", finish_reason="stop")])
    agent = LLMProducerAgent(llm=fake)
    task = _basic_task(metadata={"previous_draft": "old", "review_feedback": "fix X"})
    await agent.run(task)
    messages = fake.calls[0]["messages"]
    user = next(m for m in messages if m["role"] == "user")
    assert "old" in user["content"]
    assert "fix X" in user["content"]


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", 0.85),
        ("length", 0.5),
        ("tool_calls", 0.6),
        ("content_filter", 0.4),
        ("other_unknown_value", 0.4),
    ],
)
@pytest.mark.asyncio
async def test_confidence_maps_finish_reason_correctly(finish_reason: str, expected: float) -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="x", finish_reason=finish_reason)])
    agent = LLMProducerAgent(llm=fake)
    outcome = await agent.run(_basic_task())
    assert outcome.confidence == pytest.approx(expected)


@pytest.mark.asyncio
async def test_outcome_metadata_includes_usage() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    fake = FakeLLMClient(responses=[LLMResponse(text="x", finish_reason="stop", usage=usage)])
    agent = LLMProducerAgent(llm=fake)
    outcome = await agent.run(_basic_task())
    assert outcome.metadata["usage"] == usage
    assert outcome.metadata["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_run_uses_system_prompt_in_messages() -> None:
    fake = FakeLLMClient(responses=[LLMResponse(text="x", finish_reason="stop")])
    agent = LLMProducerAgent(llm=fake)
    await agent.run(_basic_task())
    messages = fake.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "producer agent" in messages[0]["content"].lower()


def test_agent_satisfies_agent_like_protocol() -> None:
    """Runtime ``Protocol`` check — guards the topology contract."""
    fake = FakeLLMClient()
    agent = LLMProducerAgent(llm=fake)
    assert isinstance(agent, AgentLike)


@pytest.mark.asyncio
async def test_unexpected_response_type_returns_failure_outcome() -> None:
    """If the client returns a non-``LLMResponse`` (e.g. an iterator), surface as failure."""

    class BadClient:
        async def generate(self, **_kwargs: Any) -> Any:
            async def _stream() -> AsyncIterator[str]:  # pragma: no cover - never iterated
                yield "x"

            return _stream()

    agent = LLMProducerAgent(llm=BadClient())  # type: ignore[arg-type]
    outcome = await agent.run(_basic_task())
    assert outcome.success is False
    assert outcome.error is not None
    assert "unexpected response type" in outcome.error


# ---------------------------------------------------------------------------
# Logging contract: INFO entry + done on success, WARNING on error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_logs_info_on_start_and_done(caplog: pytest.LogCaptureFixture) -> None:
    """Producer must emit start + done INFO records with task_id and confidence."""
    from tests.unit.framework.harness.agents.test_llm_producer import (  # type: ignore[import-self]  # noqa: F401
        _basic_task,
    )

    fake_response = LLMResponse(text="some draft body", finish_reason="stop")

    class _Stub:
        async def generate(self, **_kwargs: Any) -> LLMResponse:
            return fake_response

    agent = LLMProducerAgent(llm=_Stub())  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO, logger="harness.agent.producer"):
        await agent.run(_basic_task())
    messages = [r.getMessage() for r in caplog.records if r.name == "harness.agent.producer"]
    assert any("producer.run start" in m for m in messages)
    assert any("producer.run done" in m and "success=True" in m for m in messages)


@pytest.mark.asyncio
async def test_run_logs_warning_on_llm_error(caplog: pytest.LogCaptureFixture) -> None:
    class _Boom:
        async def generate(self, **_kwargs: Any) -> LLMResponse:
            raise LLMTimeoutError("lmstudio", 5.0)

    agent = LLMProducerAgent(llm=_Boom())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="harness.agent.producer"):
        outcome = await agent.run(_basic_task())
    assert outcome.success is False
    assert any("LLM generate failed" in r.getMessage() for r in caplog.records)
