"""Topology base + aggregation policy implementation.

Topologies operate on objects that quack like an agent: anything with an
async ``run(task) -> AgentOutcome`` method. This intentionally is *not*
``AsyncAgentBase`` — that abstraction is too coupled to ``AgentContext`` and
returns dict-shaped results. Topology consumers wrap their agents with a
small adapter (or use :class:`HarnessAgentAdapter`) before composition.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from typing import Any, Protocol, runtime_checkable

from src.framework.harness.settings import AggregationPolicy
from src.framework.harness.state import Task
from src.observability.logging import get_logger


@dataclass(frozen=True)
class AgentOutcome:
    """Uniform outcome shape returned by every agent in a topology."""

    agent_name: str
    response: str
    confidence: float = 0.0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentLike(Protocol):
    """Anything callable as ``await agent.run(task) -> AgentOutcome``."""

    async def run(self, task: Task) -> AgentOutcome: ...


def aggregate(
    outcomes: Sequence[AgentOutcome],
    policy: AggregationPolicy,
    *,
    verifier_score: float | None = None,
) -> AgentOutcome:
    """Reduce many outcomes to one according to ``policy``.

    Returns an :class:`AgentOutcome` with the aggregated response and a
    ``metadata['intermediate']`` list capturing every contributing outcome.
    """
    if not outcomes:
        return AgentOutcome(agent_name="aggregate", response="", confidence=0.0, success=False, error="no outcomes")

    intermediate = [_outcome_to_dict(o) for o in outcomes]
    base_metadata = {"policy": policy.value, "intermediate": intermediate}

    if policy is AggregationPolicy.FIRST_SUCCESS:
        for outcome in outcomes:
            if outcome.success:
                return _replace(outcome, metadata=base_metadata)
        # Nothing succeeded — return the last failure.
        return _replace(outcomes[-1], metadata=base_metadata)

    if policy is AggregationPolicy.ALL_MUST_PASS:
        all_passed = all(o.success for o in outcomes)
        chosen = outcomes[0]
        return _replace(
            chosen,
            success=all_passed,
            confidence=min(o.confidence for o in outcomes),
            metadata=base_metadata,
            error=None if all_passed else "one or more agents failed",
        )

    if policy is AggregationPolicy.VERIFIER_RANKED:
        ranked = sorted(
            outcomes,
            key=lambda o: (
                o.success,
                verifier_score if (verifier_score is not None and o.success) else o.confidence,
            ),
            reverse=True,
        )
        chosen = ranked[0]
        return _replace(chosen, metadata=base_metadata)

    # ConfidenceWeighted: pick the highest-confidence successful outcome.
    successes = [o for o in outcomes if o.success]
    candidates = successes or list(outcomes)
    chosen = max(candidates, key=lambda o: o.confidence)
    return _replace(chosen, metadata=base_metadata)


def _replace(o: AgentOutcome, **fields: Any) -> AgentOutcome:
    """Functional replace helper preserving frozen-dataclass semantics."""
    metadata: dict[str, Any] = dict(o.metadata)
    if "metadata" in fields:
        metadata = {**o.metadata, **fields["metadata"]}
    return AgentOutcome(
        agent_name=str(fields.get("agent_name", o.agent_name)),
        response=str(fields.get("response", o.response)),
        confidence=float(fields.get("confidence", o.confidence)),
        success=bool(fields.get("success", o.success)),
        error=fields.get("error", o.error),
        metadata=metadata,
    )


def _outcome_to_dict(o: AgentOutcome) -> dict[str, Any]:
    return {
        "agent_name": o.agent_name,
        "response": o.response,
        "confidence": o.confidence,
        "success": o.success,
        "error": o.error,
    }


@dataclass
class BaseTopology:
    """Common scaffolding for the concrete topology implementations."""

    name: str
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))

    async def run(  # pragma: no cover - subclasses override
        self,
        task: Task,
        agents: Sequence[AgentLike],
        *,
        policy: AggregationPolicy = AggregationPolicy.VERIFIER_RANKED,
    ) -> AgentOutcome:
        raise NotImplementedError

    async def _run_one(self, agent: AgentLike, task: Task) -> AgentOutcome:
        """Run a single agent with structured error capture."""
        try:
            return await agent.run(task)
        except Exception as exc:  # noqa: BLE001
            name = getattr(agent, "name", agent.__class__.__name__)
            self.logger.warning("agent %s raised %s during topology run", name, type(exc).__name__)
            return AgentOutcome(
                agent_name=name,
                response="",
                confidence=0.0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _run_all_parallel(self, agents: Sequence[AgentLike], task: Task) -> list[AgentOutcome]:
        if not agents:
            return []
        return list(await asyncio.gather(*(self._run_one(a, task) for a in agents)))


__all__ = ["AgentLike", "AgentOutcome", "BaseTopology", "aggregate"]
