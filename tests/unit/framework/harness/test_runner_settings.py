"""Verify the runner consumes the right token budgets per phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.adapters.llm.base import LLMResponse, LLMToolResponse, ToolCall
from src.framework.harness import HarnessSettings
from src.framework.harness.context import DefaultContextInjector
from src.framework.harness.hooks import HookChain
from src.framework.harness.intent import DefaultIntentNormalizer
from src.framework.harness.loop.runner import HarnessRunner
from src.framework.harness.planner import HeuristicPlanner
from src.framework.harness.replay.clock import DeterministicClock
from src.framework.harness.tools import AsyncToolExecutor, ToolRegistry
from src.framework.harness.verifier import AcceptanceCriteriaVerifier
from tests.fixtures.harness_fixtures import echo_tool

pytestmark = pytest.mark.unit


class _CapturingLLM:
    """Records every ``generate`` call's kwargs so tests can inspect them."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return self._responses.pop(0) if self._responses else LLMResponse(text="", model="m")


def _make_runner(tmp_path: Path, llm: _CapturingLLM) -> HarnessRunner:
    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "runs",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
        MAX_ITERATIONS=1,
        PLANNER_MAX_TOKENS=111,
        REASON_MAX_TOKENS=222,
        PLANNER_ENABLED=False,  # Use heuristic planner so all LLM calls are reasoning.
    )
    registry = ToolRegistry()
    schema, handler = echo_tool()
    registry.register(schema, handler)
    return HarnessRunner(
        settings=settings,
        llm=llm,
        intent=DefaultIntentNormalizer(),
        planner=HeuristicPlanner(),
        context_injector=DefaultContextInjector(),
        tool_executor=AsyncToolExecutor(registry, settings),
        verifier=AcceptanceCriteriaVerifier(),
        hooks=HookChain(),
        clock=DeterministicClock(seed=1),
    )


@pytest.mark.asyncio
async def test_reason_phase_uses_reason_max_tokens(tmp_path: Path) -> None:
    """The Reason-phase LLM call must use ``REASON_MAX_TOKENS``, not the planner setting."""
    llm = _CapturingLLM(
        responses=[
            LLMToolResponse(
                text="",
                usage={"total_tokens": 10},
                model="m",
                finish_reason="tool_use",
                tool_calls=[ToolCall(id="c", name="echo", arguments={"value": "ok"})],
            )
        ]
    )
    runner = _make_runner(tmp_path, llm)
    await runner.run("test reason budget")
    # Exactly one Reason call (planner is disabled, max_iterations=1).
    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == 222


@pytest.mark.asyncio
async def test_reason_phase_provides_tool_schemas_directly(tmp_path: Path) -> None:
    """The runner now calls ``tool_executor.tool_schemas()`` directly — schemas reach the LLM."""
    llm = _CapturingLLM(
        responses=[
            LLMToolResponse(
                text="",
                usage={"total_tokens": 5},
                model="m",
                finish_reason="tool_use",
                tool_calls=[ToolCall(id="c", name="echo", arguments={"value": "ok"})],
            )
        ]
    )
    runner = _make_runner(tmp_path, llm)
    await runner.run("test schemas")
    assert llm.calls[0]["tools"]
    schemas = llm.calls[0]["tools"]
    assert any(t["function"]["name"] == "echo" for t in schemas)
