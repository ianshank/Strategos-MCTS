"""Planner implementations.

The planner runs *once* per task before any worker. Two implementations are
provided:

* :class:`HeuristicPlanner` — derives a plan deterministically from the task's
  acceptance criteria. Useful as a default and in tests where determinism
  matters more than creative decomposition.
* :class:`LLMPlanner` — uses an :class:`LLMClient` to draft a plan. Output is
  parsed defensively; the planner returns a :class:`Plan` even when the model
  produces malformed JSON, by falling back to a single-step plan.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.adapters.llm.base import LLMClient, LLMResponse
from src.framework.harness.state import ContextPayload, Plan, PlanStep, Task
from src.observability.logging import get_logger


class HeuristicPlanner:
    """Build a plan from acceptance criteria without an LLM call."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or get_logger(__name__)

    async def plan(self, task: Task, ctx: ContextPayload | None = None) -> Plan:
        """Map each acceptance criterion to a single plan step."""
        if task.acceptance_criteria:
            steps = tuple(
                PlanStep(id=f"step-{i}", description=f"Satisfy criterion: {c.description}")
                for i, c in enumerate(task.acceptance_criteria)
            )
        else:
            steps = (PlanStep(id="step-0", description=f"Achieve goal: {task.goal}"),)
        plan = Plan(
            task_id=task.id,
            summary=f"Heuristic plan for task {task.id}",
            steps=steps,
            rationale="Derived directly from acceptance criteria.",
        )
        self._logger.debug("heuristic plan task=%s steps=%d", task.id, len(steps))
        return plan


class LLMPlanner:
    """Use an :class:`LLMClient` to produce a plan, with a safe fallback."""

    SYSTEM_PROMPT = (
        "You are a planner. Given a task, return a JSON object with keys "
        "'summary' (string), 'rationale' (string), and 'steps' (list of "
        "{'id', 'description'} objects). Output JSON only — no prose."
    )

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._logger = logger or get_logger(__name__)
        self._fallback = HeuristicPlanner(logger=self._logger)

    async def plan(self, task: Task, ctx: ContextPayload | None = None) -> Plan:
        """Ask the model for a JSON plan; fall back heuristically on any error."""
        prompt = self._build_prompt(task)
        try:
            response = await self._llm.generate(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            self._logger.warning("planner LLM error task=%s err=%s; falling back", task.id, type(exc).__name__)
            return await self._fallback.plan(task, ctx)
        if not isinstance(response, LLMResponse):
            self._logger.warning("planner unexpected response type %s; falling back", type(response).__name__)
            return await self._fallback.plan(task, ctx)
        return await self._parse_or_fallback(task, response.text)

    @staticmethod
    def _build_prompt(task: Task) -> str:
        criteria = "\n".join(f"- {c.description}" for c in task.acceptance_criteria) or "(none specified)"
        constraints = "\n".join(f"- {c}" for c in task.constraints) or "(none)"
        return (
            f"Task ID: {task.id}\n"
            f"Goal: {task.goal}\n\n"
            f"Acceptance Criteria:\n{criteria}\n\n"
            f"Constraints:\n{constraints}\n"
        )

    async def _parse_or_fallback(self, task: Task, text: str) -> Plan:
        """Best-effort JSON parse; returns heuristic plan on failure."""
        cleaned = self._strip_codefence(text)
        try:
            payload: Any = json.loads(cleaned)
        except json.JSONDecodeError:
            self._logger.warning("planner JSON decode failed task=%s; falling back", task.id)
            return await self._fallback.plan(task, None)
        if not isinstance(payload, dict):
            return await self._fallback.plan(task, None)
        steps_raw = payload.get("steps") or []
        steps = tuple(
            PlanStep(
                id=str(s.get("id") or f"step-{i}"),
                description=str(s.get("description") or "").strip(),
            )
            for i, s in enumerate(steps_raw)
            if isinstance(s, dict) and s.get("description")
        )
        if not steps:
            return await self._fallback.plan(task, None)
        return Plan(
            task_id=task.id,
            summary=str(payload.get("summary") or "")[:500],
            steps=steps,
            rationale=str(payload.get("rationale") or "")[:1000],
        )

    @staticmethod
    def _strip_codefence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove leading ```json (or similar) and trailing ```
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        return stripped.strip()


__all__ = ["HeuristicPlanner", "LLMPlanner"]
