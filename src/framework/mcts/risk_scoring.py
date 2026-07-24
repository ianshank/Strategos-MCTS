"""Risk-averse candidate scoring: ``score = value - lambda * dispersion``.

This composes the scoring seam (:mod:`src.framework.mcts.scoring`) with an uncertainty
*dispersion* signal (produced by the coarse-dynamics MDN) to penalise high-uncertainty
candidates. It is off by default — a graph builder with no risk scorer keeps the seam's
identity default, byte-for-byte unchanged.

Dispersion is supplied by a pluggable :class:`DispersionSource` so the scorer is decoupled
from where the number comes from: synthetic values in tests, a precomputed value carried on
``CandidateRecord.metadata`` (the bridge the MDN populates), or a custom callable. Attaching
*real* MDN dispersion to the graph's (currently placeholder) candidates needs real coarse
state sequences and is deferred to a follow-up; this module ships the scorer and the sources.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from src.config.constants import (
    CANDIDATE_SCORER_RISK_AVERSE,
    DEFAULT_SUBGOAL_UNCERTAINTY_LAMBDA,
    RISK_DISPERSION_METADATA_KEY,
)
from src.framework.mcts.scoring import CandidateRecord
from src.observability.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class DispersionSource(Protocol):
    """Supply a non-negative dispersion (uncertainty) value for a candidate."""

    def dispersion_for(self, candidate: CandidateRecord) -> float:
        """Return the candidate's dispersion (``>= 0``); 0 means no penalty."""
        ...


class ZeroDispersionSource:
    """Dispersion source that always returns 0 (no penalty; the safe null source)."""

    def dispersion_for(self, candidate: CandidateRecord) -> float:
        return 0.0


class MetadataDispersionSource:
    """Read a precomputed dispersion from ``candidate.metadata[key]`` (0.0 if absent).

    This is the bridge the coarse-dynamics MDN populates: once dispersion is attached to a
    candidate's metadata upstream, the risk scorer consumes it without importing torch.
    """

    def __init__(self, key: str = RISK_DISPERSION_METADATA_KEY) -> None:
        self.key = key

    def dispersion_for(self, candidate: CandidateRecord) -> float:
        # Clamp to the non-negative contract: a negative dispersion would *increase* the score.
        return max(0.0, float(candidate.metadata.get(self.key, 0.0)))


class CallableDispersionSource:
    """Adapt an arbitrary ``candidate -> dispersion`` callable to a :class:`DispersionSource`."""

    def __init__(self, fn: Callable[[CandidateRecord], float]) -> None:
        self._fn = fn

    def dispersion_for(self, candidate: CandidateRecord) -> float:
        # Clamp to the non-negative contract regardless of what the callable returns.
        return max(0.0, float(self._fn(candidate)))


class RiskAverseSubgoalScorer:
    """Candidate scorer ranking by ``value - lambda * dispersion`` (first-wins tie-break).

    Implements the seam's ``CandidateScorer`` protocol. Higher dispersion is penalised in
    proportion to ``lambda_weight`` (``>= 0``); with ``lambda_weight == 0`` it reduces to a
    pure value ranking. Deterministic: no RNG, no I/O.
    """

    name = CANDIDATE_SCORER_RISK_AVERSE

    def __init__(
        self,
        lambda_weight: float = DEFAULT_SUBGOAL_UNCERTAINTY_LAMBDA,
        dispersion_source: DispersionSource | None = None,
    ) -> None:
        if lambda_weight < 0:
            raise ValueError(f"lambda_weight must be >= 0, got {lambda_weight}")
        self.lambda_weight = lambda_weight
        self.dispersion_source: DispersionSource = dispersion_source or MetadataDispersionSource()
        # One-shot latch for the "active but no dispersion signal" misconfiguration warning.
        self._warned_all_zero_dispersion = False

    def score(self, candidate: CandidateRecord) -> float:
        """Risk-adjusted score for a single candidate: ``value - lambda * dispersion``."""
        return candidate.value - self.lambda_weight * self.dispersion_source.dispersion_for(candidate)

    def _warn_if_no_dispersion_signal(self, candidates: Sequence[CandidateRecord]) -> None:
        """Warn once if the penalty is active (lambda > 0) yet every dispersion is 0.

        In that case ``score == value`` for all candidates and the risk scorer silently
        collapses to a pure value ranking — a no-op that usually means the coarse-dynamics
        dispersion was never wired onto candidate metadata. Logged once per scorer instance
        so a hot loop does not spam, and only when it could plausibly matter.
        """
        if self._warned_all_zero_dispersion or self.lambda_weight <= 0:
            return
        if all(self.dispersion_source.dispersion_for(c) <= 0.0 for c in candidates):
            self._warned_all_zero_dispersion = True
            logger.warning(
                "RiskAverseSubgoalScorer active (lambda_weight=%s) but all candidate dispersions "
                "are 0 -> scoring collapses to pure value ranking (no uncertainty penalty applied). "
                "Ensure the coarse-dynamics dispersion is attached to candidate metadata[%r].",
                self.lambda_weight,
                getattr(self.dispersion_source, "key", RISK_DISPERSION_METADATA_KEY),
            )

    def select_best(
        self,
        candidates: Sequence[CandidateRecord],
        *,
        engine_choice: str | None,
    ) -> str | None:
        if not candidates:
            return engine_choice
        self._warn_if_no_dispersion_signal(candidates)
        best = max(candidates, key=self.score)
        if best.candidate_id != engine_choice:
            logger.debug(
                "RiskAverseSubgoalScorer re-ranked selection",
                extra={
                    "engine_choice": engine_choice,
                    "selected": best.candidate_id,
                    "lambda_weight": self.lambda_weight,
                    "selected_score": self.score(best),
                    "num_candidates": len(candidates),
                },
            )
        return best.candidate_id


__all__ = [
    "CallableDispersionSource",
    "DispersionSource",
    "MetadataDispersionSource",
    "RiskAverseSubgoalScorer",
    "ZeroDispersionSource",
]
