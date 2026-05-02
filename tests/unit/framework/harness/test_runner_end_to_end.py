"""End-to-end runner test against a scripted LLM and in-memory verifier."""

from __future__ import annotations

import pytest

from src.framework.harness.outcomes import BudgetExhausted, Terminal
from src.framework.harness.verifier import AcceptanceCriteriaVerifier, AlwaysReject
from tests.fixtures.harness_fixtures import (
    ScriptedLLM,
    echo_tool,
    llm_text,
    llm_tool_call,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_runner_terminates_when_verifier_accepts(make_runner, tool_registry) -> None:
    """A single tool call satisfying the criterion ends the loop terminally."""
    schema, handler = echo_tool()
    tool_registry.register(schema, handler)

    llm = ScriptedLLM(responses=[llm_tool_call("echo", {"value": "passes-tests"})])
    runner = make_runner(llm, verifier=AcceptanceCriteriaVerifier())

    intent = {
        "id": "t1",
        "goal": "echo the success marker",
        "acceptance_criteria": [
            {"id": "c1", "description": "passes-tests"},
        ],
    }
    result = await runner.run(intent)

    assert isinstance(result.outcome, Terminal)
    assert result.outcome.accepted is True
    assert result.iterations == 1
    assert result.confidence == pytest.approx(1.0)
    assert any(step["phase"] == "plan" for step in result.state.history)
    assert any(step["phase"] == "execute" for step in result.state.history)


@pytest.mark.asyncio
async def test_runner_runs_out_of_iterations(make_runner, tool_registry) -> None:
    """When the verifier never accepts, the iteration budget terminates the loop."""
    schema, handler = echo_tool()
    tool_registry.register(schema, handler)

    # AlwaysReject ensures the verifier never accepts.
    responses = [llm_tool_call("echo", {"value": str(i)}) for i in range(20)]
    llm = ScriptedLLM(responses=responses)
    runner = make_runner(llm, verifier=AlwaysReject())

    intent = "force iteration exhaustion"
    result = await runner.run(intent)

    assert isinstance(result.outcome, BudgetExhausted)
    assert result.outcome.budget == "iterations"
    assert result.iterations >= runner.settings.MAX_ITERATIONS


@pytest.mark.asyncio
async def test_runner_handles_unknown_tool(make_runner) -> None:
    """The executor returns a failing observation for unknown tools and the
    verifier treats that as not-yet-accepted."""
    llm = ScriptedLLM(
        responses=[llm_tool_call("nonexistent", {}), llm_text("done")] * 5,
    )
    runner = make_runner(llm, verifier=AlwaysReject())
    result = await runner.run("call a missing tool")
    assert isinstance(result.outcome, BudgetExhausted)
    # Every iteration should have produced an unsuccessful observation.
    failed_iterations = [
        step for step in result.state.history if step["phase"] == "execute" and step.get("successes", 0) == 0
    ]
    assert failed_iterations


@pytest.mark.asyncio
async def test_runner_persists_events_when_callback_supplied(make_runner, tool_registry) -> None:
    """The optional ``persist`` callback receives one event per phase."""
    schema, handler = echo_tool()
    tool_registry.register(schema, handler)

    captured: list[dict] = []

    async def persist(event: dict) -> None:
        captured.append(event)

    llm = ScriptedLLM(responses=[llm_tool_call("echo", {"value": "ok"})])
    runner = make_runner(llm, verifier=AcceptanceCriteriaVerifier())
    runner.persist = persist

    await runner.run({"goal": "go", "acceptance_criteria": [{"id": "c", "description": "ok"}]})
    phases = [event.get("phase") for event in captured]
    assert "intent" in phases
    assert "plan" in phases
    assert "verify" in phases
