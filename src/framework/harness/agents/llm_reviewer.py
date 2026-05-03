"""LLM-only reviewer agent.

Critiques a producer draft against a task and returns an outcome whose
``response`` field begins with the literal ``ACCEPT`` or ``REJECT`` marker so
the producer-reviewer topology's substring-based acceptance check works
deterministically. Robust to prompt-injection: only an ``ACCEPT`` /
``REJECT`` literal at the start of the first non-empty line is recognised.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.adapters.llm.base import LLMClient, LLMResponse
from src.framework.harness.agents.prompts import (
    REVIEWER_SYSTEM_PROMPT,
    render_reviewer_user_message,
)
from src.framework.harness.state import Task
from src.framework.harness.topology.base import AgentOutcome

# Same confidence table as the producer — kept local so the two agents
# evolve independently if needed and so a typo here can't silently break the
# producer.
_CONFIDENCE_BY_FINISH_REASON: Mapping[str, float] = {
    "stop": 0.85,
    "length": 0.5,
    "tool_calls": 0.6,
}
_UNKNOWN_FINISH_REASON_CONFIDENCE: float = 0.4

_DEFAULT_MAX_TOKENS: int = 1_500

_ACCEPT_MARKER: str = "ACCEPT"
_REJECT_MARKER: str = "REJECT"

# Bounds for a parsed score; anything outside is treated as "no usable score".
_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 1.0

_SCORE_RE = re.compile(r"^\s*score:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
_NOTES_RE = re.compile(r"^\s*notes:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*-\s+(.+?)\s*$")
_SECTION_HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]*):\s*$")


def _confidence_from_finish_reason(finish_reason: str) -> float:
    """Map an LLM ``finish_reason`` to a 0..1 confidence score."""
    return _CONFIDENCE_BY_FINISH_REASON.get(finish_reason, _UNKNOWN_FINISH_REASON_CONFIDENCE)


def _first_nonempty_line(text: str) -> str:
    """Return the first non-empty stripped line of ``text`` (or ``""``)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_score(text: str) -> float | None:
    """Parse ``score: <float>`` (case-insensitive). Returns None when absent."""
    match = _SCORE_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:  # pragma: no cover - regex already filters non-numeric
        return None


def _collect_section_bullets(text: str, section_name: str) -> list[str]:
    """Collect ``- item`` bullets that follow a ``<section_name>:`` header.

    Bullet collection stops at the next section header or at end-of-text.
    Lines blank or non-bullets are tolerated within the section.
    """
    target = section_name.lower()
    items: list[str] = []
    in_section = False
    for line in text.splitlines():
        header = _SECTION_HEADER_RE.match(line)
        if header is not None:
            in_section = header.group(1).strip().lower() == target
            continue
        if not in_section:
            continue
        bullet = _BULLET_RE.match(line)
        if bullet is not None:
            items.append(bullet.group(1).strip())
    return items


def parse_review_decision(text: str) -> tuple[bool, dict[str, Any]]:
    """Parse a reviewer response into ``(accepted, structured_feedback)``.

    Returns:
        ``(True, {"score": float | None, "notes": str})`` on ACCEPT.
        ``(False, {"score": float | None, "issues": [...], "suggestions": [...]})``
        on REJECT.
        ``(False, {"raw": text})`` on parse failure (default-reject).

    The literal ``ACCEPT`` / ``REJECT`` MUST appear at the start of the first
    non-empty line (case-insensitive). Producer drafts that mention either
    word in their body cannot trigger acceptance via prompt injection.
    """
    if not isinstance(text, str) or not text.strip():
        return False, {"raw": text if isinstance(text, str) else ""}

    first_line = _first_nonempty_line(text).upper()

    if first_line.startswith(_ACCEPT_MARKER):
        score = _extract_score(text)
        notes_match = _NOTES_RE.search(text)
        notes = notes_match.group(1).strip() if notes_match else ""
        return True, {"score": score, "notes": notes}

    if first_line.startswith(_REJECT_MARKER):
        score = _extract_score(text)
        issues = _collect_section_bullets(text, "issues")
        suggestions = _collect_section_bullets(text, "suggestions")
        return False, {
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
        }

    return False, {"raw": text}


def _decision_marker(accepted: bool) -> str:
    return _ACCEPT_MARKER if accepted else _REJECT_MARKER


@dataclass
class LLMReviewerAgent:
    """Reviewer agent that judges a draft via an injected ``LLMClient``.

    Implements the :class:`~src.framework.harness.topology.base.AgentLike`
    protocol: ``async def run(task) -> AgentOutcome``.
    """

    llm: LLMClient
    name: str = "reviewer"
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float | None = None  # ``None`` → defer to the client/preset
    system_prompt: str = REVIEWER_SYSTEM_PROMPT
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("harness.agent.reviewer"))

    async def run(self, task: Task) -> AgentOutcome:
        """Review the draft embedded in ``task`` and return a uniform outcome."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_reviewer_user_message(task)},
        ]
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        try:
            response = await self.llm.generate(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("reviewer LLM generate failed: %s: %s", type(exc).__name__, exc)
            return AgentOutcome(
                agent_name=self.name,
                response="",
                confidence=0.0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(response, LLMResponse):
            self.logger.warning("reviewer LLM returned unexpected type: %s", type(response).__name__)
            return AgentOutcome(
                agent_name=self.name,
                response="",
                confidence=0.0,
                success=False,
                error=f"unexpected response type: {type(response).__name__}",
            )

        text = response.text or ""
        accepted, parsed = parse_review_decision(text)
        marker = _decision_marker(accepted)

        score = parsed.get("score") if isinstance(parsed, dict) else None
        if isinstance(score, (int, float)) and _SCORE_MIN <= float(score) <= _SCORE_MAX:
            confidence = float(score)
        else:
            confidence = _confidence_from_finish_reason(response.finish_reason)

        return AgentOutcome(
            agent_name=self.name,
            response=f"{marker}\n{text}",
            confidence=confidence,
            success=bool(text.strip()),
            metadata={
                "decision": accepted,
                "feedback": parsed,
                "finish_reason": response.finish_reason,
                "usage": dict(response.usage),
            },
        )


__all__ = ["LLMReviewerAgent", "parse_review_decision"]
