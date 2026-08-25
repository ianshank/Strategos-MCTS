"""Supervisor topology: a manager agent dispatches each step to a worker pool.

The supervisor is the first agent. It is invoked once per round; its
response is parsed for ``DELEGATE: <worker_index> <subtask>`` lines. Each
delegation runs the named worker. The loop terminates when the supervisor
emits ``DONE`` or when the round cap is reached.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

from src.framework.harness.settings import AggregationPolicy
from src.framework.harness.state import Task
from src.framework.harness.topology.base import (
    AgentLike,
    AgentOutcome,
    BaseTopology,
    aggregate,
)

_DELEGATE_PATTERN = re.compile(r"DELEGATE:\s+(\d+)\s+(.+)", re.IGNORECASE)
_DONE_MARKER = "DONE"


@dataclass
class SupervisorTopology(BaseTopology):
    """A single supervisor coordinates a fixed pool of workers."""

    name: str = "supervisor"
    max_rounds: int = 5

    async def run(
        self,
        task: Task,
        agents: Sequence[AgentLike],
        *,
        policy: AggregationPolicy = AggregationPolicy.VERIFIER_RANKED,
    ) -> AgentOutcome:
        if not agents:
            return aggregate([], policy)
        supervisor, *workers = agents
        outcomes: list[AgentOutcome] = []
        current_task = task
        for round_index in range(self.max_rounds):
            sup_outcome = await self._run_one(supervisor, current_task)
            outcomes.append(sup_outcome)
            if not sup_outcome.success:
                break
            if _DONE_MARKER in sup_outcome.response.upper():
                return aggregate(outcomes, policy, verifier_score=sup_outcome.confidence)
            delegations = _DELEGATE_PATTERN.findall(sup_outcome.response)
            if not delegations:
                # No delegations and no DONE → break to avoid runaway looping.
                break
            for raw_index, subtask_text in delegations:
                index = int(raw_index)
                if 0 <= index < len(workers):
                    sub_task = Task(
                        id=f"{current_task.id}-r{round_index}-w{index}",
                        goal=subtask_text.strip(),
                        acceptance_criteria=current_task.acceptance_criteria,
                        constraints=current_task.constraints,
                        metadata={**current_task.metadata, "round": round_index, "worker_index": index},
                        raw=current_task.raw,
                    )
                    outcomes.append(await self._run_one(workers[index], sub_task))
            current_task = Task(
                id=current_task.id,
                goal=current_task.goal,
                acceptance_criteria=current_task.acceptance_criteria,
                constraints=current_task.constraints,
                metadata={
                    **current_task.metadata,
                    "round": round_index + 1,
                    "previous_supervisor_response": sup_outcome.response,
                },
                raw=current_task.raw,
            )
        return aggregate(outcomes, policy)


__all__ = ["SupervisorTopology"]
