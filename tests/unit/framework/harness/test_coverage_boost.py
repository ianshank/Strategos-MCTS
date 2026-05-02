"""Targeted tests to lift coverage on lower-tested harness modules.

This file consolidates small, focused tests that round out coverage on:

* :class:`AlwaysAccept` / :class:`AlwaysReject` verifiers
* memory ``query_episodic`` and ``view`` (and the markdown facade error paths)
* memory_query / memory_compact tools
* shell builtins ``test_run`` / ``lint_run`` / ``type_check_run``
* runner persistence error path
* expert pool with empty agent list
* hierarchical topology with empty agents
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from src.framework.harness import HarnessSettings
from src.framework.harness.memory.markdown import MarkdownMemoryStore
from src.framework.harness.memory.tools import register_memory_tools
from src.framework.harness.settings import AggregationPolicy, HarnessPermissions
from src.framework.harness.state import (
    AcceptanceCriterion,
    Observation,
    Task,
)
from src.framework.harness.tools import ToolRegistry
from src.framework.harness.tools.builtins.shell import (
    lint_run_tool as build_lint_run_tool,
)
from src.framework.harness.tools.builtins.shell import (
    test_run_tool as build_test_run_tool,
)
from src.framework.harness.tools.builtins.shell import (
    type_check_run_tool as build_type_check_run_tool,
)
from src.framework.harness.topology import (
    AgentOutcome,
    ExpertPoolTopology,
    HierarchicalTopology,
    SupervisorTopology,
    aggregate,
)
from src.framework.harness.verifier import AlwaysAccept, AlwaysReject

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- verifier


@pytest.mark.asyncio
async def test_always_accept_returns_passing_result() -> None:
    v = AlwaysAccept()
    out = await v.verify((), Task(id="t", goal="g"), None)
    assert out.passed is True
    assert out.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_always_reject_returns_failing_result() -> None:
    v = AlwaysReject()
    out = await v.verify((), Task(id="t", goal="g"), None)
    assert out.passed is False
    assert out.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_acceptance_criteria_verifier_no_observations() -> None:
    """No observations + no criteria → not accepted (nothing succeeded)."""
    from src.framework.harness.verifier import AcceptanceCriteriaVerifier

    v = AcceptanceCriteriaVerifier()
    out = await v.verify((), Task(id="t", goal="g"), None)
    assert out.passed is False


@pytest.mark.asyncio
async def test_acceptance_criteria_verifier_no_criteria_with_success() -> None:
    """No criteria + at least one successful observation → accepted."""
    from src.framework.harness.verifier import AcceptanceCriteriaVerifier

    v = AcceptanceCriteriaVerifier()
    obs = (Observation(invocation_id="i", tool_name="t", success=True, payload="ok"),)
    out = await v.verify(obs, Task(id="t", goal="g"), None)
    assert out.passed is True


@pytest.mark.asyncio
async def test_acceptance_criteria_verifier_handles_invalid_regex() -> None:
    """A criterion whose ``check`` is not a valid regex falls back to substring."""
    from src.framework.harness.verifier import AcceptanceCriteriaVerifier

    v = AcceptanceCriteriaVerifier()
    obs = (Observation(invocation_id="i", tool_name="t", success=True, payload="hello (world)"),)
    task = Task(
        id="t",
        goal="g",
        acceptance_criteria=(AcceptanceCriterion(id="c", description="d", check="(unbalanced"),),
    )
    out = await v.verify(obs, task, None)
    # Invalid regex falls back to substring match — '(unbalanced' is not a substring of the haystack.
    assert out.passed is False


# -------------------------------------------------------------------- memory


@pytest.mark.asyncio
async def test_memory_store_query_episodic_with_filter(tmp_path: Path) -> None:
    """``query_episodic`` filters events older than the cutoff."""
    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "out",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
    )
    store = MarkdownMemoryStore(settings=settings)
    await store.append_event({"correlation_id": "a", "phase": "intent"})
    await store.append_event({"correlation_id": "b", "phase": "plan"})
    # Filter that excludes everything (far-future cutoff).
    excluded = await store.query_episodic(since_iso="2099-01-01T00:00:00+00:00")
    assert excluded == []
    included = await store.query_episodic()
    assert len(included) == 2


@pytest.mark.asyncio
async def test_memory_store_view_returns_empty_when_missing(tmp_path: Path) -> None:
    """``view`` returns empty string when the file doesn't exist."""
    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "out",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
    )
    store = MarkdownMemoryStore(settings=settings)
    assert await store.view("nonexistent.md") == ""


@pytest.mark.asyncio
async def test_memory_query_tool_renders_events(tmp_path: Path) -> None:
    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "out",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
    )
    store = MarkdownMemoryStore(settings=settings)
    await store.append_event({"correlation_id": "abc12345xyz", "phase": "intent"})
    registry = ToolRegistry()
    register_memory_tools(registry, store)
    out = await registry.get_handler("memory_query")({})
    assert "[intent]" in out
    assert "abc12345" in out  # truncated correlation prefix


@pytest.mark.asyncio
async def test_memory_compact_tool_writes_index(tmp_path: Path) -> None:
    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "out",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
    )
    store = MarkdownMemoryStore(settings=settings)
    await store.append_event({"correlation_id": "x", "phase": "p"})
    registry = ToolRegistry()
    register_memory_tools(registry, store)
    rendered = await registry.get_handler("memory_compact")({})
    assert "Correlation `x`" in rendered
    assert (tmp_path / "mem" / "MEMORY.md").exists()


# ---------------------------------------------------------------- shell builtins


@pytest.mark.asyncio
async def test_test_run_tool_executes_pytest(tmp_path: Path) -> None:
    """``test_run`` runs pytest and reports verdict + exit code."""
    perms = HarnessPermissions(SHELL=True)
    _, handler = build_test_run_tool(cwd=tmp_path, perms=perms, timeout=30.0)
    # Empty directory → pytest reports no tests collected (exit 5).
    out = await handler({})
    assert out.startswith(("verdict=pass", "verdict=fail", "not_found"))


@pytest.mark.asyncio
async def test_lint_run_tool_runs_ruff(tmp_path: Path) -> None:
    perms = HarnessPermissions(SHELL=True)
    _, handler = build_lint_run_tool(cwd=tmp_path, perms=perms, timeout=30.0)
    out = await handler({})
    assert out.startswith(("verdict=pass", "verdict=fail", "not_found"))


@pytest.mark.asyncio
async def test_type_check_run_tool_runs_mypy(tmp_path: Path) -> None:
    perms = HarnessPermissions(SHELL=True)
    _, handler = build_type_check_run_tool(cwd=tmp_path, perms=perms, timeout=30.0)
    out = await handler({})
    assert out.startswith(("verdict=pass", "verdict=fail", "not_found"))


@pytest.mark.asyncio
async def test_check_tools_respect_disabled_permission(tmp_path: Path) -> None:
    perms = HarnessPermissions(SHELL=False)
    for builder in (build_test_run_tool, build_lint_run_tool, build_type_check_run_tool):
        _, handler = builder(cwd=tmp_path, perms=perms)
        out = await handler({})
        assert "permission denied" in out


@pytest.mark.asyncio
async def test_shell_check_tools_reject_bad_extra_args(tmp_path: Path) -> None:
    """``extra_args`` must be a list of strings; anything else is a contract error."""
    perms = HarnessPermissions(SHELL=True)
    _, handler = build_lint_run_tool(cwd=tmp_path, perms=perms)
    out = await handler({"extra_args": "not-a-list"})
    assert out.startswith("error:")


# --------------------------------------------------------------- topology edges


@pytest.mark.asyncio
async def test_expert_pool_with_empty_agents_returns_failure() -> None:
    topo = ExpertPoolTopology()
    out = await topo.run(Task(id="t", goal="g"), [])
    assert out.success is False


@pytest.mark.asyncio
async def test_hierarchical_with_empty_agents_returns_failure() -> None:
    topo = HierarchicalTopology()
    out = await topo.run(Task(id="t", goal="g"), [])
    assert out.success is False


@pytest.mark.asyncio
async def test_supervisor_with_empty_agents_returns_failure() -> None:
    topo = SupervisorTopology()
    out = await topo.run(Task(id="t", goal="g"), [])
    assert out.success is False


@pytest.mark.asyncio
async def test_supervisor_breaks_when_no_delegations_or_done() -> None:
    """A supervisor that emits neither DELEGATE nor DONE halts the loop."""

    class _Sup:
        name = "sup"

        async def run(self, task: Task) -> AgentOutcome:
            return AgentOutcome(agent_name=self.name, response="thinking", success=True, confidence=0.5)

    topo = SupervisorTopology(max_rounds=3)
    out = await topo.run(Task(id="t", goal="g"), [_Sup()])
    # Loop broke after the first round; outcome is the supervisor's own response.
    assert out.success is True


@pytest.mark.asyncio
async def test_aggregate_first_success_with_no_successes_returns_last() -> None:
    """``FIRST_SUCCESS`` falls back to the last outcome when nothing succeeded."""
    outcomes = [
        AgentOutcome(agent_name="a", response="x", success=False, error="e"),
        AgentOutcome(agent_name="b", response="y", success=False, error="e"),
    ]
    out = aggregate(outcomes, AggregationPolicy.FIRST_SUCCESS)
    assert out.agent_name == "b"


# --------------------------------------------------------------- runner persist


@pytest.mark.asyncio
async def test_runner_persist_failure_is_logged_not_raised(tmp_path: Path) -> None:
    """A persist callback that raises must not abort the loop."""
    from src.framework.harness.context import DefaultContextInjector
    from src.framework.harness.hooks import HookChain
    from src.framework.harness.intent import DefaultIntentNormalizer
    from src.framework.harness.loop.runner import HarnessRunner
    from src.framework.harness.planner import HeuristicPlanner
    from src.framework.harness.replay.clock import DeterministicClock
    from src.framework.harness.tools import AsyncToolExecutor
    from src.framework.harness.verifier import AlwaysAccept

    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "out",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
        MAX_ITERATIONS=1,
        PLANNER_ENABLED=False,
    )

    class _BadLLM:
        async def generate(self, **kwargs: Any) -> Any:
            from src.adapters.llm.base import LLMResponse

            return LLMResponse(text="ok", model="m", finish_reason="stop")

    async def bad_persist(_event: dict[str, Any]) -> None:
        raise RuntimeError("persist exploded")

    runner = HarnessRunner(
        settings=settings,
        llm=_BadLLM(),
        intent=DefaultIntentNormalizer(),
        planner=HeuristicPlanner(),
        context_injector=DefaultContextInjector(),
        tool_executor=AsyncToolExecutor(ToolRegistry(), settings),
        verifier=AlwaysAccept(),
        hooks=HookChain(),
        clock=DeterministicClock(seed=1),
        persist=bad_persist,
        logger=logging.getLogger("test_persist"),
    )
    result = await runner.run("test persist failure")
    # Persist failure does not turn into a runner failure — the loop completed.
    assert result.iterations == 1


# ---------------------------------------------------------------- intent edges


@pytest.mark.asyncio
async def test_intent_normalizer_rejects_invalid_criterion_type() -> None:
    """Acceptance criteria of unsupported types raise ``TypeError``."""
    from src.framework.harness.intent import DefaultIntentNormalizer

    with pytest.raises(TypeError):
        await DefaultIntentNormalizer().normalize(
            {"goal": "g", "acceptance_criteria": [42]},
            HarnessSettings(),
        )


# --------------------------------------------------------------- planner edges


def test_llm_planner_strip_codefence() -> None:
    """Helper round-trip — codefence wrappers come off cleanly."""
    from src.framework.harness.planner.planner import LLMPlanner

    cleaned = LLMPlanner._strip_codefence('```json\n{"a":1}\n```')
    assert cleaned == '{"a":1}'


@pytest.mark.asyncio
async def test_llm_planner_handles_non_dict_payload(tmp_path: Path) -> None:
    """A JSON list (instead of object) falls back to the heuristic planner."""
    from src.adapters.llm.base import LLMResponse
    from src.framework.harness.planner import LLMPlanner

    class _C:
        async def generate(self, **kwargs: Any) -> LLMResponse:
            return LLMResponse(text='["not", "an", "object"]', model="m", finish_reason="stop")

    planner = LLMPlanner(_C(), max_tokens=200)  # type: ignore[arg-type]
    plan = await planner.plan(Task(id="t", goal="g"))
    # Falls back to a single-step heuristic plan.
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_llm_planner_drops_steps_without_description(tmp_path: Path) -> None:
    """Steps missing ``description`` are skipped; if none remain, fall back."""
    import json

    from src.adapters.llm.base import LLMResponse
    from src.framework.harness.planner import LLMPlanner

    class _C:
        async def generate(self, **kwargs: Any) -> LLMResponse:
            payload = json.dumps({"summary": "x", "steps": [{"id": "s1"}]})
            return LLMResponse(text=payload, model="m", finish_reason="stop")

    planner = LLMPlanner(_C(), max_tokens=200)  # type: ignore[arg-type]
    plan = await planner.plan(Task(id="t", goal="g", acceptance_criteria=()))
    # Heuristic fallback produced a single step.
    assert len(plan.steps) >= 1
