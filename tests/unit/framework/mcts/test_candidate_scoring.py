"""Unit tests for the MCTS candidate-scoring seam (pure module).

Covers strategos_subgoal_scoring_seam AC-2 (opt-in value scorer re-ranks with a
deterministic first-wins tie-break; unknown scorer names fail loud) and the
determinism/torch-free constraints underpinning AC-3.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from src.config.constants import (
    CANDIDATE_SCORER_IDENTITY,
    CANDIDATE_SCORER_NAMES,
    CANDIDATE_SCORER_VALUE,
    DEFAULT_CANDIDATE_SCORER,
)
from src.framework.mcts.scoring import (
    CandidateRecord,
    CandidateScorer,
    IdentityCandidateScorer,
    ValueCandidateScorer,
    candidates_from_action_stats,
    create_candidate_scorer,
)


def _stats(order):
    """Build an ordered action_stats mapping from (action, value, visits) tuples."""
    return {a: {"visits": v, "value": val, "value_sum": val * v} for a, val, v in order}


class TestCandidateRecord:
    def test_fields_and_default_metadata(self):
        rec = CandidateRecord(candidate_id="a", value=0.5, visits=3)
        assert rec.candidate_id == "a"
        assert rec.value == 0.5
        assert rec.visits == 3
        assert rec.metadata == {}

    def test_is_frozen(self):
        rec = CandidateRecord(candidate_id="a", value=0.5, visits=3)
        with pytest.raises(FrozenInstanceError):
            rec.value = 1.0  # type: ignore[misc]


class TestCandidatesFromActionStats:
    def test_preserves_order_and_coerces_types(self):
        stats = _stats([("x", 0.1, 2), ("y", 0.9, 5), ("z", 0.3, 1)])
        records = candidates_from_action_stats(stats)
        assert [r.candidate_id for r in records] == ["x", "y", "z"]  # insertion order preserved
        assert [r.value for r in records] == [0.1, 0.9, 0.3]
        assert [r.visits for r in records] == [2, 5, 1]
        assert all(isinstance(r.value, float) and isinstance(r.visits, int) for r in records)

    def test_missing_keys_default_to_zero(self):
        records = candidates_from_action_stats({"a": {}})
        assert records == [CandidateRecord(candidate_id="a", value=0.0, visits=0)]

    def test_empty_mapping(self):
        assert candidates_from_action_stats({}) == []


class TestIdentityCandidateScorer:
    def test_returns_engine_choice(self):  # strategos_subgoal_scoring_seam AC-1
        scorer = IdentityCandidateScorer()
        cands = candidates_from_action_stats(_stats([("a", 0.1, 9), ("b", 0.9, 1)]))
        # Even though 'b' has the higher value, identity preserves the engine's choice.
        assert scorer.select_best(cands, engine_choice="a") == "a"

    def test_returns_none_when_engine_choice_none(self):
        assert IdentityCandidateScorer().select_best([], engine_choice=None) is None

    def test_name_matches_settings_enum(self):
        assert IdentityCandidateScorer().name == CANDIDATE_SCORER_IDENTITY

    def test_satisfies_protocol(self):
        assert isinstance(IdentityCandidateScorer(), CandidateScorer)


class TestValueCandidateScorer:
    def test_selects_highest_value(self):  # strategos_subgoal_scoring_seam AC-2
        scorer = ValueCandidateScorer()
        cands = candidates_from_action_stats(_stats([("a", 0.1, 9), ("b", 0.9, 1)]))
        # Engine chose 'a' (most visits); value scorer overrides to the higher-value 'b'.
        assert scorer.select_best(cands, engine_choice="a") == "b"

    def test_tie_break_is_first_wins(self):  # strategos_subgoal_scoring_seam AC-3
        scorer = ValueCandidateScorer()
        cands = candidates_from_action_stats(_stats([("a", 0.9, 1), ("b", 0.9, 9)]))
        # Equal max value -> first in insertion order wins (matches max(...) first-wins).
        assert scorer.select_best(cands, engine_choice="b") == "a"

    def test_empty_candidates_fall_back_to_engine_choice(self):
        assert ValueCandidateScorer().select_best([], engine_choice="a") == "a"

    def test_name_matches_settings_enum(self):
        assert ValueCandidateScorer().name == CANDIDATE_SCORER_VALUE

    def test_satisfies_protocol(self):
        assert isinstance(ValueCandidateScorer(), CandidateScorer)

    def test_is_deterministic_across_calls(self):
        scorer = ValueCandidateScorer()
        cands = candidates_from_action_stats(_stats([("a", 0.2, 3), ("b", 0.7, 1), ("c", 0.7, 2)]))
        results = {scorer.select_best(cands, engine_choice="a") for _ in range(50)}
        assert results == {"b"}  # highest value, first-wins on the 0.7 tie


class TestFactory:
    def test_default_is_identity(self):
        scorer = create_candidate_scorer()
        assert isinstance(scorer, IdentityCandidateScorer)
        assert DEFAULT_CANDIDATE_SCORER == CANDIDATE_SCORER_IDENTITY

    def test_creates_identity(self):
        assert isinstance(create_candidate_scorer(CANDIDATE_SCORER_IDENTITY), IdentityCandidateScorer)

    def test_creates_value(self):
        assert isinstance(create_candidate_scorer(CANDIDATE_SCORER_VALUE), ValueCandidateScorer)

    def test_is_case_and_whitespace_insensitive(self):
        assert isinstance(create_candidate_scorer("  VALUE  "), ValueCandidateScorer)

    def test_unknown_name_raises_no_silent_fallback(self):  # strategos_subgoal_scoring_seam AC-2
        with pytest.raises(ValueError, match="Unknown candidate scorer"):
            create_candidate_scorer("bogus")

    def test_every_registered_name_constructs(self):
        for name in CANDIDATE_SCORER_NAMES:
            assert isinstance(create_candidate_scorer(name), CandidateScorer)


def test_scoring_module_is_torch_optional():  # strategos_subgoal_scoring_seam AC-3
    """The baseline scoring path must import without torch and must not pull torch in.

    Runs in a clean subprocess: importing the scoring module must succeed even with no
    torch installed (proving torch-optionality); and when torch is not installed it must
    stay out of ``sys.modules`` (the baseline path never imports it).
    """
    code = (
        "import importlib.util, sys\n"
        "import src.framework.mcts.scoring  # must not raise without torch\n"
        "if importlib.util.find_spec('torch') is None:\n"
        "    tmods = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "    assert not tmods, tmods\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
