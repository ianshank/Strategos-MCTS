"""LLM-only producer agent.

Drafts a response against a :class:`~src.framework.harness.state.Task` using
any object that satisfies the :class:`~src.adapters.llm.base.LLMClient`
protocol. Returns a uniform :class:`AgentOutcome` so it composes cleanly with
the producer-reviewer topology.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.adapters.llm.base import LLMClient, LLMResponse
from src.framework.harness.agents.prompts import (
    PRODUCER_SYSTEM_PROMPT,
    render_producer_user_message,
)
from src.framework.harness.state import Task
from src.framework.harness.topology.base import AgentOutcome

# Confidence values exposed as a module-level mapping so they remain
# inspectable and trivially overridable for tuning. Anything not in this
# table maps to ``_UNKNOWN_FINISH_REASON_CONFIDENCE``.
_CONFIDENCE_BY_FINISH_REASON: Mapping[str, float] = {
    "stop": 0.85,
    "length": 0.5,
    "tool_calls": 0.6,
}
_UNKNOWN_FINISH_REASON_CONFIDENCE: float = 0.4

# Default budget for the producer's draft. Mirrors ``HarnessSettings``
# defaults (``HARNESS_PRODUCER_MAX_TOKENS``) so unconfigured callers still
# get sensible behaviour.
_DEFAULT_MAX_TOKENS: int = 4_000


def _confidence_from_finish_reason(finish_reason: str) -> float:
    """Map an LLM ``finish_reason`` to a 0..1 confidence score."""
    return _CONFIDENCE_BY_FINISH_REASON.get(finish_reason, _UNKNOWN_FINISH_REASON_CONFIDENCE)


@dataclass
class LLMProducerAgent:
    """Producer agent that drafts a response via an injected ``LLMClient``.

    Implements the :class:`~src.framework.harness.topology.base.AgentLike`
    protocol: ``async def run(task) -> AgentOutcome``.
    """

    llm: LLMClient
    name: str = "producer"
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float | None = None  # ``None`` → defer to the client/preset
    system_prompt: str = PRODUCER_SYSTEM_PROMPT
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("harness.agent.producer"))

    async def run(self, task: Task) -> AgentOutcome:
        """Draft a response for ``task`` and return a uniform outcome."""
        user_message = render_producer_user_message(task)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        self.logger.info(
            "producer.run start task_id=%s max_tokens=%d temperature=%s user_message_chars=%d",
            task.id,
            self.max_tokens,
            "default" if self.temperature is None else f"{self.temperature:.3f}",
            len(user_message),
        )
        # Full prompt at DEBUG so it does not bloat normal logs but is
        # available when reproducing a problematic round.
        self.logger.debug("producer.run user_message=%r", user_message)

        try:
            response = await self.llm.generate(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface as outcome, never raise
            self.logger.warning(
                "producer.run task_id=%s LLM generate failed: %s: %s",
                task.id,
                type(exc).__name__,
                exc,
            )
            return AgentOutcome(
                agent_name=self.name,
                response="",
                confidence=0.0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(response, LLMResponse):
            # ``stream=False`` should always yield ``LLMResponse``; treat
            # anything else as a contract violation and return failure.
            self.logger.warning(
                "producer.run task_id=%s LLM returned unexpected type: %s",
                task.id,
                type(response).__name__,
            )
            return AgentOutcome(
                agent_name=self.name,
                response="",
                confidence=0.0,
                success=False,
                error=f"unexpected response type: {type(response).__name__}",
            )

        text = response.text or ""
        confidence = _confidence_from_finish_reason(response.finish_reason)
        success = bool(text.strip())
        self.logger.info(
            "producer.run done task_id=%s success=%s finish_reason=%s response_chars=%d confidence=%.2f usage=%s",
            task.id,
            success,
            response.finish_reason,
            len(text),
            confidence,
            response.usage or {},
        )
        return AgentOutcome(
            agent_name=self.name,
            response=text,
            confidence=confidence,
            success=success,
            metadata={
                "finish_reason": response.finish_reason,
                "usage": dict(response.usage),
            },
        )


__all__ = ["LLMProducerAgent"]
