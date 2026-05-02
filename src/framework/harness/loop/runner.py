"""``HarnessRunner`` — the deterministic six-phase control loop.

The phases are: **Intent → Plan → Context → Reason → Execute → Verify**. The
runner orchestrates them, enforces budgets, runs the hook chain, persists
events, and returns a typed :class:`HarnessOutcome`. It is intentionally
small; collaborators do the heavy lifting.

Design notes:

* The Reason phase binds *directly* to :class:`LLMClient` rather than wrapping
  an :class:`AsyncAgentBase`. The wrap-as-agent story lives in
  :mod:`src.framework.harness.loop.facade`.
* All non-determinism (clock, UUIDs, RNG) is funnelled through
  :class:`RecordingClock`.
* Tool-call dispatch parses :class:`LLMToolResponse` and emits
  :class:`Observation`\\s; tool-less responses produce a synthetic
  ``response`` observation so the verifier still has something to inspect.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.adapters.llm.base import LLMClient, LLMResponse, LLMToolResponse
from src.framework.harness.hooks import HookChain, HookOutcome
from src.framework.harness.loop.budget import Budget, BudgetBundle, IterationBudget, TimeBudget
from src.framework.harness.outcomes import (
    BudgetExhausted,
    Continue,
    HarnessOutcome,
    HookViolation,
    PhaseResult,
    Retryable,
    Terminal,
    is_terminal,
)
from src.framework.harness.protocols import (
    ContextInjector,
    IntentNormalizer,
    OutputVerifier,
    Planner,
    ToolExecutor,
)
from src.framework.harness.replay.clock import RecordingClock, SystemClock
from src.framework.harness.settings import HarnessSettings
from src.framework.harness.state import (
    HarnessState,
    Observation,
    ToolInvocation,
)
from src.observability.logging import get_correlation_id, set_correlation_id


@dataclass
class RunResult:
    """Final output returned by :meth:`HarnessRunner.run`."""

    outcome: HarnessOutcome
    state: HarnessState
    iterations: int = 0
    duration_ms: float = 0.0
    response_text: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# Optional persistence callback signature used by the runner. We don't take a
# hard dependency on a MemoryStore here so the runner can be exercised in
# tests without one.
PersistFn = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class HarnessRunner:
    """Deterministic six-phase control loop."""

    settings: HarnessSettings
    llm: LLMClient
    intent: IntentNormalizer
    planner: Planner
    context_injector: ContextInjector
    tool_executor: ToolExecutor
    verifier: OutputVerifier
    hooks: HookChain = field(default_factory=HookChain)
    clock: RecordingClock = field(default_factory=SystemClock)
    persist: PersistFn | None = None
    logger: logging.Logger = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, raw_intent: str | dict[str, Any]) -> RunResult:
        """Drive the loop until terminal verdict or budget exhaustion."""
        run_started = time.perf_counter()
        state = HarnessState(correlation_id=self.clock.uuid())
        set_correlation_id(state.correlation_id)
        self._log("info", state, phase="run.start", payload={"intent_type": type(raw_intent).__name__})

        budgets = BudgetBundle(
            tokens=Budget(limit=float(self.settings.TOKEN_BUDGET_PER_LOOP), name="tokens"),
            time=TimeBudget(
                limit_seconds=self.settings.TOTAL_BUDGET_SECONDS,
                started_at=self.clock.monotonic(),
                clock=self.clock.monotonic,
                name="time",
            ),
            iterations=IterationBudget(limit=self.settings.MAX_ITERATIONS, name="iterations"),
        )

        # Phase 1 (once): Intent capture
        intent_result = await self._intent_phase(raw_intent, state)
        if is_terminal(intent_result.outcome):
            return self._finalise(state, intent_result.outcome, run_started)
        budgets.tokens.consume(intent_result.tokens_used)

        # Phase 2 (once): Plan
        if self.settings.PLANNER_ENABLED:
            plan_result = await self._plan_phase(state)
            if is_terminal(plan_result.outcome):
                return self._finalise(state, plan_result.outcome, run_started)
            budgets.tokens.consume(plan_result.tokens_used)

        # Phases 3-6: per-iteration loop
        outcome: HarnessOutcome = Continue()
        while True:
            exhausted = budgets.first_exhausted()
            if exhausted is not None:
                outcome = BudgetExhausted(
                    budget=exhausted,  # type: ignore[arg-type]
                    consumed=self._budget_consumed(budgets, exhausted),
                    limit=self._budget_limit(budgets, exhausted),
                )
                break
            state.iteration += 1
            budgets.iterations.consume(1)
            self._log("info", state, phase="iteration.begin", payload={})

            # Pre-iteration hooks
            hook_outcome = await self.hooks.run(state)
            violation = self._first_short_circuit(hook_outcome)
            if violation is not None:
                outcome = violation
                break

            # Context
            ctx_result = await self._context_phase(state)
            budgets.tokens.consume(ctx_result.tokens_used)
            if is_terminal(ctx_result.outcome):
                outcome = ctx_result.outcome
                break

            # Reason (LLM call)
            reason_result = await self._reason_phase(state)
            budgets.tokens.consume(reason_result.tokens_used)
            if is_terminal(reason_result.outcome):
                outcome = reason_result.outcome
                break
            if isinstance(reason_result.outcome, Retryable):
                self._log("warning", state, phase="reason.retryable", payload={"reason": reason_result.outcome.reason})
                continue

            # Execute (tool dispatch)
            exec_result = await self._execute_phase(state)
            if is_terminal(exec_result.outcome):
                outcome = exec_result.outcome
                break

            # Verify
            verify_result = await self._verify_phase(state)
            if is_terminal(verify_result.outcome):
                outcome = verify_result.outcome
                break

        return self._finalise(state, outcome, run_started)

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    async def _intent_phase(self, raw: str | dict[str, Any], state: HarnessState) -> PhaseResult:
        try:
            task = await self.intent.normalize(raw, self.settings)
        except Exception as exc:  # noqa: BLE001
            self._log("error", state, phase="intent.error", payload={"error": type(exc).__name__})
            return PhaseResult(outcome=Terminal(accepted=False, note=f"intent error: {exc}"))
        state.task = task
        state.record("intent", {"task_id": task.id, "goal_chars": len(task.goal)})
        await self._persist({"phase": "intent", "task_id": task.id, "goal": task.goal})
        return PhaseResult()

    async def _plan_phase(self, state: HarnessState) -> PhaseResult:
        assert state.task is not None  # narrowed by intent phase
        try:
            plan = await self.planner.plan(state.task)
        except Exception as exc:  # noqa: BLE001
            self._log("error", state, phase="plan.error", payload={"error": type(exc).__name__})
            return PhaseResult(outcome=Terminal(accepted=False, note=f"plan error: {exc}"))
        state.plan = plan
        state.record("plan", {"steps": len(plan.steps)})
        await self._persist({"phase": "plan", "task_id": plan.task_id, "steps": len(plan.steps)})
        return PhaseResult()

    async def _context_phase(self, state: HarnessState) -> PhaseResult:
        assert state.task is not None
        payload = await self.context_injector.build(state.task, state.plan, state)
        state.last_context = payload
        state.record("context", {"system_chars": len(payload.system_prompt)})
        return PhaseResult()

    async def _reason_phase(self, state: HarnessState) -> PhaseResult:
        ctx = state.last_context
        assert ctx is not None
        messages = ctx.render()
        tools = list(self.tool_executor.tool_schemas())
        try:
            response = await self.llm.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=self.settings.REASON_MAX_TOKENS,
                tools=tools or None,
            )
        except Exception as exc:  # noqa: BLE001
            return PhaseResult(outcome=Retryable(reason="llm_error", cause=type(exc).__name__))
        if not isinstance(response, LLMResponse):
            return PhaseResult(outcome=Retryable(reason="llm_streaming_unexpected"))
        state.last_response_text = response.text
        if isinstance(response, LLMToolResponse):
            state.pending_tool_calls = tuple(
                ToolInvocation(id=tc.id, tool_name=tc.name, arguments=tc.arguments) for tc in response.tool_calls
            )
        else:
            state.pending_tool_calls = ()
        tokens = response.total_tokens
        state.tokens_consumed += tokens
        state.record(
            "reason",
            {"tokens": tokens, "tool_calls": len(state.pending_tool_calls), "response_chars": len(response.text)},
        )
        return PhaseResult(tokens_used=tokens)

    async def _execute_phase(self, state: HarnessState) -> PhaseResult:
        observations: list[Observation] = []
        if not state.pending_tool_calls:
            # No tools requested → synthesise an observation from the response.
            observations.append(
                Observation(
                    invocation_id=self.clock.uuid(),
                    tool_name="response",
                    success=True,
                    payload=state.last_response_text,
                    metadata={"synthetic": "true"},
                )
            )
        else:
            for call in state.pending_tool_calls:
                obs = await self.tool_executor.execute(call, correlation_id=state.correlation_id)
                observations.append(obs)
                await self._persist(
                    {
                        "phase": "execute",
                        "tool": call.tool_name,
                        "success": obs.success,
                        "duration_ms": obs.duration_ms,
                    }
                )
        state.last_observations = tuple(observations)
        state.pending_tool_calls = ()
        state.record(
            "execute",
            {"observations": len(observations), "successes": sum(1 for o in observations if o.success)},
        )
        return PhaseResult()

    async def _verify_phase(self, state: HarnessState) -> PhaseResult:
        assert state.task is not None
        verification = await self.verifier.verify(state.last_observations, state.task, state.plan)
        state.last_verification = verification
        state.record(
            "verify",
            {"passed": verification.passed, "score": verification.score, "failed": list(verification.failed_criteria)},
        )
        await self._persist({"phase": "verify", "passed": verification.passed, "score": verification.score})
        if verification.passed:
            return PhaseResult(outcome=Terminal(accepted=True, verification=verification))
        # On rejection, return Continue so the iteration budget — not the
        # verifier — owns the decision to halt. This keeps budget exhaustion
        # the canonical signal for "tried hard enough".
        return PhaseResult(outcome=Continue(note="verification failed; continuing"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finalise(self, state: HarnessState, outcome: HarnessOutcome, started_at: float) -> RunResult:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        confidence = state.last_verification.score if state.last_verification is not None else 0.0
        result = RunResult(
            outcome=outcome,
            state=state,
            iterations=state.iteration,
            duration_ms=duration_ms,
            response_text=state.last_response_text,
            confidence=confidence,
            metadata={
                "correlation_id": state.correlation_id,
                "tokens_consumed": state.tokens_consumed,
                "outcome_kind": outcome.kind,
            },
        )
        self._log(
            "info",
            state,
            phase="run.end",
            payload={
                "outcome": outcome.kind,
                "iterations": state.iteration,
                "duration_ms": duration_ms,
                "confidence": confidence,
            },
        )
        return result

    @staticmethod
    def _first_short_circuit(outcome: HookOutcome) -> HookViolation | None:
        if outcome.short_circuited and outcome.violations:
            return outcome.violations[0]
        return None

    @staticmethod
    def _budget_consumed(bundle: BudgetBundle, name: str) -> float:
        if name == "tokens":
            return bundle.tokens.consumed
        if name == "iterations":
            return float(bundle.iterations.consumed)
        return float(bundle.time.elapsed())

    @staticmethod
    def _budget_limit(bundle: BudgetBundle, name: str) -> float:
        if name == "tokens":
            return float(bundle.tokens.limit)
        if name == "iterations":
            return float(bundle.iterations.limit)
        return float(bundle.time.limit_seconds)

    async def _persist(self, event: dict[str, Any]) -> None:
        if self.persist is None:
            return
        try:
            payload = {"correlation_id": get_correlation_id(), **event}
            await self.persist(payload)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("persistence failed err=%s", type(exc).__name__)

    def _log(self, level: str, state: HarnessState, *, phase: str, payload: dict[str, Any]) -> None:
        if not self.logger.isEnabledFor(getattr(logging, level.upper(), logging.INFO)):
            return
        message = "harness phase=%s iteration=%d cid=%s payload=%s"
        getattr(self.logger, level)(
            message,
            phase,
            state.iteration,
            state.correlation_id,
            payload,
        )


__all__ = ["HarnessRunner", "RunResult"]
