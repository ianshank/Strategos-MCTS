"""Ralph loop: outer iteration around the harness runner.

Each outer cycle:

1. Re-read the spec (if present) to refresh acceptance criteria.
2. Build a :class:`Task` from the spec.
3. Run the harness once.
4. If the spec contains the completion marker, halt with ``done``.
5. If the verifier accepted, halt with ``accepted``.
6. If the same outcome repeats N times, declare ``stuck`` per
   ``HARNESS_RALPH_STUCK_BEHAVIOR``.

The loop is intentionally thin — all heavy lifting lives in the runner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Literal

from src.framework.harness.intent import SpecLoader, SpecParseError
from src.framework.harness.loop.runner import HarnessRunner, RunResult
from src.framework.harness.outcomes import Terminal
from src.framework.harness.ralph.completion import is_complete
from src.framework.harness.settings import HarnessSettings
from src.observability.logging import get_logger

RalphStatus = Literal["accepted", "done", "stuck", "exhausted"]


@dataclass
class RalphResult:
    """Final outcome of the outer loop."""

    status: RalphStatus
    rounds: int
    last_run: RunResult | None = None
    stuck_kind: str = ""


@dataclass
class RalphLoop:
    """Drive the harness runner until completion or stuck-state."""

    runner: HarnessRunner
    settings: HarnessSettings
    spec_path: Path | None = None
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))

    async def run(self) -> RalphResult:
        """Execute the outer loop. Returns the terminal status."""
        recent_outcomes: deque[str] = deque(maxlen=self.settings.RALPH_STUCK_THRESHOLD)
        loader = SpecLoader()
        last: RunResult | None = None
        for round_index in range(self.settings.RALPH_MAX_LOOPS):
            spec_text = ""
            spec_intent: str | dict[str, object]
            if self.spec_path is not None:
                try:
                    spec = loader.load(self.spec_path)
                except SpecParseError as exc:
                    self.logger.error("ralph spec error round=%d err=%s", round_index, exc)
                    return RalphResult(status="stuck", rounds=round_index, stuck_kind="spec_error")
                spec_text = spec.raw
                if is_complete(self.spec_path, self.settings.RALPH_COMPLETION_MARKER, content=spec_text):
                    return RalphResult(status="done", rounds=round_index, last_run=last)
                spec_intent = {
                    "id": f"ralph-{round_index}",
                    "goal": spec.goal or "Make the spec pass.",
                    "acceptance_criteria": spec.criteria_payload(),
                    "constraints": list(spec.constraints),
                    "metadata": {"round": round_index, "spec_path": str(self.spec_path)},
                }
            else:
                spec_intent = f"Ralph round {round_index}"

            last = await self.runner.run(spec_intent)
            outcome_kind = last.outcome.kind
            recent_outcomes.append(outcome_kind)
            self.logger.info(
                "ralph round=%d outcome=%s iterations=%d",
                round_index,
                outcome_kind,
                last.iterations,
            )

            if isinstance(last.outcome, Terminal) and last.outcome.accepted:
                return RalphResult(status="accepted", rounds=round_index + 1, last_run=last)

            if len(recent_outcomes) == recent_outcomes.maxlen and len(set(recent_outcomes)) == 1:
                stuck_kind = recent_outcomes[0]
                if self.settings.RALPH_STUCK_BEHAVIOR == "abort":
                    return RalphResult(status="stuck", rounds=round_index + 1, last_run=last, stuck_kind=stuck_kind)
                # 'escalate' and 'pivot' currently behave identically — they
                # break out so the caller can take corrective action; this
                # leaves room for future enhancement (e.g. swap in a stronger
                # planner) without breaking the contract.
                return RalphResult(status="stuck", rounds=round_index + 1, last_run=last, stuck_kind=stuck_kind)

        return RalphResult(status="exhausted", rounds=self.settings.RALPH_MAX_LOOPS, last_run=last)


__all__ = ["RalphLoop", "RalphResult", "RalphStatus"]
