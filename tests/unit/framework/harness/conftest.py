"""Pytest fixtures for harness integration tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.framework.harness import HarnessSettings
from src.framework.harness.context import DefaultContextInjector
from src.framework.harness.hooks import HookChain
from src.framework.harness.intent import DefaultIntentNormalizer
from src.framework.harness.loop.runner import HarnessRunner
from src.framework.harness.planner import HeuristicPlanner
from src.framework.harness.replay.clock import DeterministicClock
from src.framework.harness.tools import AsyncToolExecutor, ToolRegistry
from src.framework.harness.verifier import AcceptanceCriteriaVerifier
from tests.fixtures.harness_fixtures import ScriptedLLM


@pytest.fixture
def harness_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HarnessSettings:
    """Function-scoped settings rooted in a fresh tmp directory."""
    monkeypatch.setenv("HARNESS_MEMORY_ROOT", str(tmp_path / "mem"))
    monkeypatch.setenv("HARNESS_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("HARNESS_TOOL_OUTPUT_SPILLOVER_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("HARNESS_MAX_ITERATIONS", "3")
    monkeypatch.setenv("HARNESS_TOTAL_BUDGET_SECONDS", "30")
    monkeypatch.setenv("HARNESS_ITERATION_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("HARNESS_PLANNER_ENABLED", "true")
    return HarnessSettings()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Empty registry — tests register the tools they need."""
    return ToolRegistry()


@pytest.fixture
def make_runner(
    harness_settings: HarnessSettings,
    tool_registry: ToolRegistry,
) -> Callable[..., HarnessRunner]:
    """Build a :class:`HarnessRunner` from a scripted LLM and an optional verifier."""

    def _make(
        llm: ScriptedLLM,
        *,
        verifier: Any | None = None,
        hooks: HookChain | None = None,
    ) -> HarnessRunner:
        executor = AsyncToolExecutor(tool_registry, harness_settings)
        return HarnessRunner(
            settings=harness_settings,
            llm=llm,
            intent=DefaultIntentNormalizer(),
            planner=HeuristicPlanner(),
            context_injector=DefaultContextInjector(),
            tool_executor=executor,
            verifier=verifier or AcceptanceCriteriaVerifier(),
            hooks=hooks or HookChain(),
            clock=DeterministicClock(seed=1234),
        )

    return _make
