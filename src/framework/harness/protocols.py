"""Protocols for every pluggable harness collaborator.

Using ``Protocol`` (PEP 544) gives us structural typing and dependency
injection without forcing a class hierarchy on consumers. Every protocol is
``runtime_checkable`` so contract tests can use ``isinstance`` to verify
compliance.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.framework.harness.outcomes import HookViolation
from src.framework.harness.settings import AggregationPolicy, HarnessSettings
from src.framework.harness.state import (
    ContextPayload,
    HarnessState,
    Observation,
    Plan,
    Task,
    ToolInvocation,
    VerificationResult,
)


class HookCost(str, Enum):
    """Relative execution cost — used to order the hook chain."""

    CHEAP = "cheap"
    MEDIUM = "medium"
    EXPENSIVE = "expensive"


_HOOK_COST_RANK = {HookCost.CHEAP: 0, HookCost.MEDIUM: 1, HookCost.EXPENSIVE: 2}


def hook_cost_rank(cost: HookCost) -> int:
    """Return the numeric ordering rank for a :class:`HookCost`."""
    return _HOOK_COST_RANK[cost]


@runtime_checkable
class IntentNormalizer(Protocol):
    """Translate raw user intent into a typed :class:`Task`."""

    async def normalize(self, raw: str | dict[str, Any], settings: HarnessSettings) -> Task: ...


@runtime_checkable
class Planner(Protocol):
    """Produce a strategic :class:`Plan` consumed by every later phase."""

    async def plan(self, task: Task, ctx: ContextPayload | None = None) -> Plan: ...


@runtime_checkable
class ContextInjector(Protocol):
    """Compose a :class:`ContextPayload` for the Reason phase."""

    async def build(self, task: Task, plan: Plan | None, state: HarnessState) -> ContextPayload: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Run a tool invocation and return a structured observation."""

    async def execute(self, call: ToolInvocation, *, correlation_id: str) -> Observation: ...

    def list_tools(self) -> Sequence[str]: ...

    def tool_schemas(self) -> Sequence[dict[str, Any]]: ...


@runtime_checkable
class OutputVerifier(Protocol):
    """Decide whether the iteration's outputs satisfy acceptance criteria."""

    async def verify(
        self,
        observations: Sequence[Observation],
        task: Task,
        plan: Plan | None,
    ) -> VerificationResult: ...


@runtime_checkable
class Hook(Protocol):
    """A guardrail in the hook chain.

    ``cost_class`` orders hooks cheap-first; ``short_circuit`` controls
    whether a failure halts the chain immediately or accumulates.
    """

    name: str
    cost_class: HookCost
    short_circuit: bool

    async def __call__(self, state: HarnessState) -> HookViolation | None: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Append-only event log + read-only views over compacted markdown."""

    async def append_event(self, event: dict[str, Any]) -> None: ...
    async def read_index(self) -> str: ...
    async def view(self, relative_path: str) -> str: ...
    async def query_episodic(self, *, since_iso: str | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class TopologyRunner(Protocol):
    """Multi-agent orchestration pattern."""

    name: str

    async def run(
        self,
        task: Task,
        agents: Sequence[Any],  # AsyncAgentBase, but avoid the circular import
        *,
        policy: AggregationPolicy,
    ) -> Any: ...


__all__ = [
    "ContextInjector",
    "Hook",
    "HookCost",
    "IntentNormalizer",
    "MemoryStore",
    "OutputVerifier",
    "Planner",
    "TopologyRunner",
    "ToolExecutor",
    "hook_cost_rank",
]
