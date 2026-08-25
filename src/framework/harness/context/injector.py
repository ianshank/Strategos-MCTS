"""Default :class:`ContextInjector` implementation.

Composes a :class:`ContextPayload` from task, plan, memory, and any
caller-provided RAG snippets. Memory excerpts are passed through the
:class:`EpisodicCompressor` so prompts respect a hard size budget.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import logging

from src.framework.harness.context.compressor import EpisodicCompressor
from src.framework.harness.protocols import MemoryStore
from src.framework.harness.state import ContextPayload, HarnessState, Plan, Task
from src.observability.logging import get_logger

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful, deterministic software engineer running inside an "
    "agent harness. Use tools rather than prose. Honour acceptance criteria. "
    "Stop when the verifier accepts."
)


class DefaultContextInjector:
    """Compose context payloads with optional memory and RAG."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        compressor: EpisodicCompressor | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        rag_provider: Callable[[Task], Awaitable[tuple[str, ...]]] | None = None,
        spec_text: str = "",
        logger: logging.Logger | None = None,
    ) -> None:
        self._memory = memory
        self._compressor = compressor or EpisodicCompressor()
        self._system_prompt = system_prompt
        self._rag_provider = rag_provider
        self._spec_text = spec_text
        self._logger = logger or get_logger(__name__)

    async def build(
        self,
        task: Task,
        plan: Plan | None,
        state: HarnessState,
    ) -> ContextPayload:
        """Compose a fresh :class:`ContextPayload`."""
        memory_excerpt = ""
        if self._memory is not None:
            try:
                index = await self._memory.read_index()
                memory_excerpt = self._compressor.compress(index)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("memory read failed err=%s; continuing without", type(exc).__name__)

        rag_snippets: tuple[str, ...] = ()
        if self._rag_provider is not None:
            try:
                rag_snippets = await self._rag_provider(task)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("rag provider failed err=%s; continuing without", type(exc).__name__)

        plan_brief = self._render_plan(plan)
        task_brief = self._render_task(task)
        payload = ContextPayload(
            system_prompt=self._system_prompt,
            task_brief=task_brief,
            plan_brief=plan_brief,
            memory_excerpt=memory_excerpt,
            spec_excerpt=self._spec_text,
            rag_snippets=rag_snippets,
            extra={"iteration": state.iteration},
        )
        self._logger.debug(
            "context payload task=%s iter=%d memory_chars=%d rag=%d",
            task.id,
            state.iteration,
            len(memory_excerpt),
            len(rag_snippets),
        )
        return payload

    @staticmethod
    def _render_plan(plan: Plan | None) -> str:
        if plan is None:
            return ""
        lines = [f"Summary: {plan.summary}"]
        if plan.rationale:
            lines.append(f"Rationale: {plan.rationale}")
        for step in plan.steps:
            lines.append(f"- [{step.id}] {step.description}")
        return "\n".join(lines)

    @staticmethod
    def _render_task(task: Task) -> str:
        lines = [f"Goal: {task.goal}"]
        if task.acceptance_criteria:
            lines.append("Acceptance criteria:")
            for crit in task.acceptance_criteria:
                lines.append(f"- {crit.description}")
        if task.constraints:
            lines.append("Constraints:")
            for cstr in task.constraints:
                lines.append(f"- {cstr}")
        return "\n".join(lines)


__all__ = ["DEFAULT_SYSTEM_PROMPT", "DefaultContextInjector"]
