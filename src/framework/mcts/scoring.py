"""Pluggable candidate scoring for MCTS-backed decision nodes.

This module is the seam between an MCTS search result and the final action
choice. The search engine produces per-candidate statistics (visit counts and
mean values); a :class:`CandidateScorer` decides which candidate wins.

The default :class:`IdentityCandidateScorer` returns the engine's own selection
unchanged, so introducing the seam is behaviour-preserving. Alternative scorers
(e.g. a value-argmax scorer here, or an uncertainty-penalised risk-averse scorer
in a later spec) can re-rank candidates without touching the search core.

Design invariants (relied on by the "bit-for-bit identical default" guarantee):

* **Pure and deterministic.** No RNG, no I/O, no ``torch``. Given the same
  candidates and engine choice, a scorer returns the same result every call.
* **First-wins tie-breaks.** Candidates are compared with :func:`max`, which
  returns the *first* maximal element — matching ``MCTSEngine._select_best_action``
  and preserving the engine's insertion order on ties, never a dict/hash order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.config.constants import (
    CANDIDATE_SCORER_IDENTITY,
    CANDIDATE_SCORER_NAMES,
    CANDIDATE_SCORER_VALUE,
    DEFAULT_CANDIDATE_SCORER,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CandidateRecord:
    """One MCTS candidate action exposed to the scoring seam.

    Attributes:
        candidate_id: The action identifier (as produced by the search engine).
        value: Mean value estimate for the candidate (``child.value``).
        visits: Visit count for the candidate.
        metadata: Optional extra per-candidate data for richer scorers; empty by
            default and never required by the built-in scorers.
    """

    candidate_id: str
    value: float
    visits: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def candidates_from_action_stats(
    action_stats: Mapping[str, Mapping[str, Any]],
) -> list[CandidateRecord]:
    """Build ordered candidate records from an engine ``action_stats`` mapping.

    ``action_stats`` preserves child-insertion order (a plain ``dict``), so the
    returned list is order-stable — a prerequisite for deterministic tie-breaks.

    Args:
        action_stats: Mapping of ``action -> {"visits", "value", ...}`` as
            emitted by ``MCTSEngine`` statistics.

    Returns:
        Candidate records in the engine's original action order.
    """
    return [
        CandidateRecord(
            candidate_id=action,
            value=float(stats.get("value", 0.0)),
            visits=int(stats.get("visits", 0)),
        )
        for action, stats in action_stats.items()
    ]


@runtime_checkable
class CandidateScorer(Protocol):
    """Decide the winning candidate from MCTS per-candidate statistics.

    ``engine_choice`` is the action the search engine already selected via its own
    selection policy (``MAX_VISITS`` by default). A scorer that wants to preserve
    baseline behaviour returns it unchanged.
    """

    #: Stable, human-readable scorer name (matches the settings enum value).
    name: str

    def select_best(
        self,
        candidates: Sequence[CandidateRecord],
        *,
        engine_choice: str | None,
    ) -> str | None:
        """Return the winning ``candidate_id`` (or ``None`` when there are none)."""
        ...


class IdentityCandidateScorer:
    """Behaviour-preserving default: return the engine's own selection.

    The search engine has already chosen ``engine_choice`` via its selection
    policy; the seam is a pass-through, so the node's output is byte-for-byte
    identical to the pre-seam behaviour.
    """

    name = CANDIDATE_SCORER_IDENTITY

    def select_best(
        self,
        candidates: Sequence[CandidateRecord],
        *,
        engine_choice: str | None,
    ) -> str | None:
        return engine_choice


class ValueCandidateScorer:
    """Opt-in scorer that selects the candidate with the highest mean value.

    Ties resolve first-wins (via :func:`max`), matching the engine's tie
    behaviour and keeping selection deterministic. Falls back to
    ``engine_choice`` only when there are no candidates.
    """

    name = CANDIDATE_SCORER_VALUE

    def select_best(
        self,
        candidates: Sequence[CandidateRecord],
        *,
        engine_choice: str | None,
    ) -> str | None:
        if not candidates:
            return engine_choice
        best = max(candidates, key=lambda candidate: candidate.value)
        if best.candidate_id != engine_choice:
            logger.debug(
                "ValueCandidateScorer overrode engine choice",
                extra={
                    "engine_choice": engine_choice,
                    "selected": best.candidate_id,
                    "selected_value": best.value,
                    "num_candidates": len(candidates),
                },
            )
        return best.candidate_id


# Registry of built-in scorers keyed by their settings-enum name. New scorers are
# registered here (and in ``CANDIDATE_SCORER_NAMES``) so the factory stays closed
# over a single source of truth.
_SCORER_REGISTRY: dict[str, type[CandidateScorer]] = {
    CANDIDATE_SCORER_IDENTITY: IdentityCandidateScorer,
    CANDIDATE_SCORER_VALUE: ValueCandidateScorer,
}


def create_candidate_scorer(name: str | None = None) -> CandidateScorer:
    """Construct a :class:`CandidateScorer` by name (factory / DI entry point).

    Args:
        name: Scorer name; ``None`` selects the default (``identity``).

    Returns:
        A ready-to-use scorer instance.

    Raises:
        ValueError: If ``name`` is not a recognized scorer. Fails loud rather than
            silently degrading to the default, so a typo can never quietly disable
            an intended scorer.
    """
    resolved = (name or DEFAULT_CANDIDATE_SCORER).strip().lower()
    scorer_cls = _SCORER_REGISTRY.get(resolved)
    if scorer_cls is None:
        raise ValueError(f"Unknown candidate scorer '{name}'; expected one of {sorted(CANDIDATE_SCORER_NAMES)}")
    logger.debug("Created candidate scorer", extra={"scorer": resolved})
    return scorer_cls()


__all__ = [
    "CandidateRecord",
    "CandidateScorer",
    "IdentityCandidateScorer",
    "ValueCandidateScorer",
    "candidates_from_action_stats",
    "create_candidate_scorer",
]
