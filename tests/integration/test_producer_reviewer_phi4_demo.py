"""Integration tests for the producer-reviewer Phi-4 demo and factory helpers.

The harness uses an in-process :class:`FakeLLMClient` that records every
call. No network traffic is performed and no real LM Studio instance is
required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from demos import producer_reviewer_phi4_demo as demo
from src.adapters.llm.base import LLMResponse
from src.framework.harness.agents import LLMProducerAgent, LLMReviewerAgent
from src.framework.harness.factories import HarnessFactory
from src.framework.harness.settings import HarnessSettings, reset_harness_settings
from src.framework.harness.topology.base import AgentOutcome
from src.framework.harness.topology.producer_reviewer import ProducerReviewerTopology

pytestmark = pytest.mark.integration


# ──────────────────────────── fake LLM client ────────────────────────────


@dataclass
class FakeLLMClient:
    """A scripted, deterministic LLM client implementing the protocol surface.

    ``responses`` is consumed in order; each entry maps to one ``generate``
    call. The fake stores every call's keyword arguments in ``calls`` so
    tests can introspect message ordering and per-agent budgets.
    """

    responses: list[str]
    finish_reason: str = "stop"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursor: int = 0

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        self.calls.append(
            {
                "messages": messages,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
                "stream": stream,
                "stop": stop,
                **kwargs,
            }
        )
        if self._cursor >= len(self.responses):
            text = self.responses[-1] if self.responses else ""
        else:
            text = self.responses[self._cursor]
            self._cursor += 1
        return LLMResponse(
            text=text,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="fake",
            finish_reason=self.finish_reason,
        )


# ───────────────────────── fixtures / helpers ────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the env var that influences demo defaults between tests."""
    monkeypatch.delenv("HARNESS_BENCHMARK_TASK_ID", raising=False)
    reset_harness_settings()
    yield  # type: ignore[misc]
    reset_harness_settings()


def _accept_text(score: float = 0.9, notes: str = "looks good") -> str:
    return f"ACCEPT\nscore: {score}\nnotes: {notes}"


def _reject_text() -> str:
    return (
        "REJECT\n"
        "score: 0.3\n"
        "issues:\n"
        "- AC-1: missing edge case handling\n"
        "suggestions:\n"
        "- add nil-guard before division"
    )


# ─────────────────────────── run_pipeline tests ──────────────────────────


async def test_run_pipeline_completes_and_returns_successful_outcome() -> None:
    """A producer draft followed by an ACCEPT review yields a successful outcome."""
    fake = FakeLLMClient(
        responses=[
            "DRAFT: comprehensive review of the diff",
            _accept_text(),
        ]
    )
    outcome = await demo.run_pipeline(llm=fake, task_id="A1", rounds=3)
    assert outcome.success is True
    assert outcome.response
    assert len(fake.calls) == 2


async def test_run_pipeline_iterates_when_reviewer_rejects_then_accepts() -> None:
    """Rejection feeds review feedback back into the next producer prompt."""
    fake = FakeLLMClient(
        responses=[
            "DRAFT: first attempt",
            _reject_text(),
            "DRAFT: revised attempt addressing feedback",
            _accept_text(score=0.85),
        ]
    )
    outcome = await demo.run_pipeline(llm=fake, task_id="A1", rounds=3)
    assert outcome.success is True
    assert len(fake.calls) == 4

    # Second producer call (index 2) must contain the review feedback that the
    # topology threaded through ``Task.metadata['review_feedback']``.
    second_producer_call = fake.calls[2]
    user_msg = next(m for m in second_producer_call["messages"] if m["role"] == "user")
    assert "Reviewer feedback" in user_msg["content"]


async def test_run_pipeline_lookup_unknown_task_id_raises_keyerror() -> None:
    fake = FakeLLMClient(responses=[])
    with pytest.raises(KeyError):
        await demo.run_pipeline(llm=fake, task_id="ZZ99", rounds=1)


async def test_run_pipeline_uses_explicit_max_tokens_overrides() -> None:
    fake = FakeLLMClient(responses=["DRAFT: x", _accept_text()])
    await demo.run_pipeline(
        llm=fake,
        task_id="A1",
        rounds=1,
        producer_max_tokens=512,
        reviewer_max_tokens=64,
    )
    assert fake.calls[0]["max_tokens"] == 512
    assert fake.calls[1]["max_tokens"] == 64


async def test_run_pipeline_default_max_tokens_when_none() -> None:
    """When no override is supplied, the agent's defaults (which mirror
    ``HarnessSettings``) apply — confirming the demo does NOT hardcode them
    in its own body."""
    fake = FakeLLMClient(responses=["DRAFT: x", _accept_text()])
    await demo.run_pipeline(
        llm=fake,
        task_id="A1",
        rounds=1,
        producer_max_tokens=None,
        reviewer_max_tokens=None,
    )
    hs = HarnessSettings()
    assert fake.calls[0]["max_tokens"] == hs.PRODUCER_MAX_TOKENS
    assert fake.calls[1]["max_tokens"] == hs.REVIEWER_MAX_TOKENS


# ─────────────────────────── main() integration ──────────────────────────


def test_main_exits_2_when_no_task_id_and_no_env_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HARNESS_BENCHMARK_TASK_ID", raising=False)
    reset_harness_settings()
    rc = demo.main([])
    captured = capsys.readouterr()
    assert rc == demo.EXIT_USAGE_ERROR
    assert "task" in captured.err.lower()


def test_main_resolves_task_from_env_when_flag_absent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeLLMClient(
        responses=[
            "DRAFT: env-driven run",
            _accept_text(),
        ]
    )

    def _fake_create_llm(self: Any) -> Any:
        return fake

    monkeypatch.setenv("HARNESS_BENCHMARK_TASK_ID", "A1")
    reset_harness_settings()
    monkeypatch.setattr(
        "src.framework.harness.factories.HarnessFactory.create_llm",
        _fake_create_llm,
    )

    rc = demo.main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out  # human prose was emitted
    assert len(fake.calls) == 2

    roles = [m["role"] for call in fake.calls for m in call["messages"] if m["role"] in {"system", "user"}]
    # System+user for both producer and reviewer ⇒ at least 4 entries.
    assert roles.count("system") >= 2
    assert roles.count("user") >= 2


def test_main_json_flag_emits_pure_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeLLMClient(responses=["DRAFT: machine-readable", _accept_text()])

    def _fake_create_llm(self: Any) -> Any:
        return fake

    monkeypatch.setattr(
        "src.framework.harness.factories.HarnessFactory.create_llm",
        _fake_create_llm,
    )

    rc = demo.main(["--task", "A1", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    parsed = json.loads(captured.out)
    for key in ("success", "response", "agent_name"):
        assert key in parsed


def test_format_outcome_includes_intermediate_history() -> None:
    intermediate = [
        {"agent_name": "producer", "success": True, "confidence": 0.8},
        {"agent_name": "reviewer", "success": True, "confidence": 0.9},
    ]
    outcome = AgentOutcome(
        agent_name="reviewer",
        response="ACCEPT\nscore: 0.9",
        confidence=0.9,
        success=True,
        metadata={"intermediate": intermediate, "policy": "verifier_ranked"},
    )
    rendered = demo.format_outcome(outcome)
    assert "producer" in rendered
    assert "reviewer" in rendered
    assert "intermediate" in rendered


# ─────────────────────────── factory helpers ─────────────────────────────


def test_factory_create_producer_reviewer_agents_default_tokens_from_settings() -> None:
    fake = FakeLLMClient(responses=[])
    factory = HarnessFactory()
    producer, reviewer = factory.create_producer_reviewer_agents(llm=fake)
    hs = HarnessSettings()
    assert producer.max_tokens == hs.PRODUCER_MAX_TOKENS
    assert reviewer.max_tokens == hs.REVIEWER_MAX_TOKENS
    # Sequential-execution constraint: producer and reviewer share one client.
    assert producer.llm is reviewer.llm
    assert isinstance(producer, LLMProducerAgent)
    assert isinstance(reviewer, LLMReviewerAgent)


def test_factory_create_producer_reviewer_agents_explicit_overrides() -> None:
    fake = FakeLLMClient(responses=[])
    factory = HarnessFactory()
    producer, reviewer = factory.create_producer_reviewer_agents(
        llm=fake,
        producer_max_tokens=321,
        reviewer_max_tokens=123,
        temperature=0.1,
    )
    assert producer.max_tokens == 321
    assert reviewer.max_tokens == 123
    assert producer.temperature == pytest.approx(0.1)
    assert reviewer.temperature == pytest.approx(0.1)


def test_factory_create_producer_reviewer_topology_uses_settings_default() -> None:
    factory = HarnessFactory()
    topology = factory.create_producer_reviewer_topology()
    assert isinstance(topology, ProducerReviewerTopology)
    assert topology.max_rounds == HarnessSettings().PRODUCER_REVIEWER_ROUNDS


def test_factory_topology_max_rounds_explicit_override() -> None:
    factory = HarnessFactory()
    topology = factory.create_producer_reviewer_topology(max_rounds=5)
    assert topology.max_rounds == 5
