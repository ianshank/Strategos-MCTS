"""Default :class:`OutputVerifier` implementations.

The verifier decides whether the iteration's outputs satisfy the task's
acceptance criteria. Two implementations:

* :class:`AcceptanceCriteriaVerifier` — searches observation payloads for
  configured success markers; accepts when all criteria match.
* :class:`AlwaysAccept` / :class:`AlwaysReject` — degenerate verifiers used
  in tests to drive the loop into specific terminal states.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
import re

from src.framework.harness.state import (
    AcceptanceCriterion,
    Observation,
    Plan,
    Task,
    VerificationResult,
)
from src.observability.logging import get_logger


@dataclass
class AcceptanceCriteriaVerifier:
    """Pattern-based verifier: each criterion matches if its ``check`` regex
    appears in any observation's payload (or, if ``check`` is empty, the
    description itself is used as a literal needle).
    """

    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))

    async def verify(
        self,
        observations: Sequence[Observation],
        task: Task,
        plan: Plan | None,
    ) -> VerificationResult:
        if not task.acceptance_criteria:
            # No criteria → accept only when at least one observation exists
            # AND every observation succeeded. This is intentionally strict:
            # without explicit criteria, any partial failure should block
            # acceptance so callers must opt in by satisfying the whole batch.
            passed = bool(observations) and all(o.success for o in observations)
            return VerificationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                notes="no acceptance criteria defined; accepted only when every observation succeeded",
            )

        haystack = "\n".join(o.payload for o in observations if o.success)
        failed: list[str] = []
        passes = 0
        for crit in task.acceptance_criteria:
            if self._matches(crit, haystack):
                passes += 1
            else:
                failed.append(crit.id)
        total = len(task.acceptance_criteria)
        score = passes / total if total else 0.0
        passed = passes == total
        self.logger.debug(
            "verifier task=%s passed=%s score=%.2f failed=%d/%d",
            task.id,
            passed,
            score,
            len(failed),
            total,
        )
        return VerificationResult(
            passed=passed,
            score=score,
            failed_criteria=tuple(failed),
            notes=f"{passes}/{total} criteria satisfied",
        )

    def _matches(self, crit: AcceptanceCriterion, haystack: str) -> bool:
        needle = crit.check or crit.description
        if not needle:
            return False
        try:
            return re.search(needle, haystack, re.IGNORECASE) is not None
        except re.error:
            self.logger.debug(
                "criterion %s check is not valid regex; falling back to substring match",
                crit.id,
            )
            return needle.lower() in haystack.lower()


@dataclass
class AlwaysAccept:
    """Test-only verifier that accepts unconditionally."""

    async def verify(self, observations: Sequence[Observation], task: Task, plan: Plan | None) -> VerificationResult:
        return VerificationResult(passed=True, score=1.0, notes="always-accept")


@dataclass
class AlwaysReject:
    """Test-only verifier that rejects unconditionally."""

    async def verify(self, observations: Sequence[Observation], task: Task, plan: Plan | None) -> VerificationResult:
        return VerificationResult(passed=False, score=0.0, notes="always-reject")


__all__ = ["AcceptanceCriteriaVerifier", "AlwaysAccept", "AlwaysReject"]
