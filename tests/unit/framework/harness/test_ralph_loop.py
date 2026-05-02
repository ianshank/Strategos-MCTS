"""Integration tests for the Ralph outer loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.harness.ralph import RalphLoop
from src.framework.harness.verifier import AcceptanceCriteriaVerifier, AlwaysReject
from tests.fixtures.harness_fixtures import (
    ScriptedLLM,
    echo_tool,
    llm_tool_call,
)

pytestmark = pytest.mark.unit


def _write_spec(tmp_path: Path, *, marker: str = "", criterion: str = "passes") -> Path:
    spec = tmp_path / "spec.md"
    text = f"# Goal\nMake stuff work.\n\n# Acceptance Criteria\n- {criterion}\n"
    if marker:
        text += f"\n{marker}\n"
    spec.write_text(text)
    return spec


@pytest.mark.asyncio
async def test_ralph_halts_on_completion_marker(make_runner, tool_registry, tmp_path: Path) -> None:
    """When the spec already contains the completion marker, status is ``done``."""
    schema, handler = echo_tool()
    tool_registry.register(schema, handler)

    llm = ScriptedLLM(responses=[llm_tool_call("echo", {"value": "passes"})] * 5)
    runner = make_runner(llm)
    spec_path = _write_spec(tmp_path, marker=runner.settings.RALPH_COMPLETION_MARKER)
    loop = RalphLoop(runner=runner, settings=runner.settings, spec_path=spec_path)
    result = await loop.run()
    assert result.status == "done"


@pytest.mark.asyncio
async def test_ralph_halts_on_acceptance(make_runner, tool_registry, tmp_path: Path) -> None:
    """A successful inner run halts the outer loop with ``accepted``."""
    schema, handler = echo_tool()
    tool_registry.register(schema, handler)

    llm = ScriptedLLM(responses=[llm_tool_call("echo", {"value": "passes"})] * 5)
    runner = make_runner(llm, verifier=AcceptanceCriteriaVerifier())
    spec_path = _write_spec(tmp_path, criterion="passes")
    loop = RalphLoop(runner=runner, settings=runner.settings, spec_path=spec_path)
    result = await loop.run()
    assert result.status == "accepted"
    assert result.rounds >= 1


@pytest.mark.asyncio
async def test_ralph_declares_stuck_on_repeated_outcome(
    make_runner, tool_registry, tmp_path: Path, monkeypatch
) -> None:
    """When the same outcome repeats ``RALPH_STUCK_THRESHOLD`` times, halt with stuck."""
    monkeypatch.setenv("HARNESS_RALPH_STUCK_THRESHOLD", "2")
    monkeypatch.setenv("HARNESS_RALPH_MAX_LOOPS", "5")

    from src.framework.harness import HarnessSettings

    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "runs",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
        MAX_ITERATIONS=2,
        RALPH_STUCK_THRESHOLD=2,
        RALPH_MAX_LOOPS=5,
    )

    schema, handler = echo_tool()
    tool_registry.register(schema, handler)
    llm = ScriptedLLM(responses=[llm_tool_call("echo", {"value": "x"})] * 30)
    runner = make_runner(llm, verifier=AlwaysReject())
    runner.settings = settings
    spec_path = _write_spec(tmp_path, criterion="never matches")
    loop = RalphLoop(runner=runner, settings=settings, spec_path=spec_path)
    result = await loop.run()
    assert result.status == "stuck"
    assert result.stuck_kind == "budget_exhausted"
