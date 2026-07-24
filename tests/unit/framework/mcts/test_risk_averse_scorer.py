"""Unit tests for the risk-averse subgoal scorer (strategos_risk_averse_subgoal_scorer AC-1).

Torch-free: dispersion is supplied synthetically, so the ranking math is fully pinned locally.
"""

from __future__ import annotations

import pytest

from src.config.constants import CANDIDATE_SCORER_RISK_AVERSE
from src.framework.mcts.risk_scoring import (
    CallableDispersionSource,
    MetadataDispersionSource,
    RiskAverseSubgoalScorer,
    ZeroDispersionSource,
)
from src.framework.mcts.scoring import CandidateRecord, CandidateScorer


def _rec(cid, value, dispersion, visits=1):
    return CandidateRecord(candidate_id=cid, value=value, visits=visits, metadata={"dispersion": dispersion})


class TestScore:
    def test_score_is_value_minus_lambda_times_dispersion(self):  # AC-1
        scorer = RiskAverseSubgoalScorer(lambda_weight=2.0, dispersion_source=MetadataDispersionSource())
        assert scorer.score(_rec("a", value=1.0, dispersion=0.3)) == pytest.approx(1.0 - 2.0 * 0.3)

    def test_lambda_zero_is_pure_value(self):  # AC-1
        scorer = RiskAverseSubgoalScorer(lambda_weight=0.0, dispersion_source=MetadataDispersionSource())
        assert scorer.score(_rec("a", value=0.7, dispersion=9.9)) == pytest.approx(0.7)


class TestRanking:
    def test_equal_value_lower_dispersion_wins_for_any_positive_lambda(self):  # AC-1
        candidates = [_rec("A", 1.0, 0.1), _rec("B", 1.0, 0.9)]
        for lam in (0.01, 0.5, 2.0, 50.0):
            scorer = RiskAverseSubgoalScorer(lambda_weight=lam, dispersion_source=MetadataDispersionSource())
            assert scorer.select_best(candidates, engine_choice="B") == "A"
            assert scorer.score(candidates[0]) > scorer.score(candidates[1])

    def test_penalty_flips_selection_as_lambda_grows(self):  # AC-1
        # B has higher value but higher dispersion; a large enough lambda flips B -> A.
        candidates = [_rec("A", 1.0, 0.1), _rec("B", 1.5, 0.9)]
        assert (
            RiskAverseSubgoalScorer(0.0, MetadataDispersionSource()).select_best(candidates, engine_choice="A") == "B"
        )  # pure value -> B
        assert (
            RiskAverseSubgoalScorer(0.5, MetadataDispersionSource()).select_best(candidates, engine_choice="A") == "B"
        )  # small penalty, B still ahead
        assert (
            RiskAverseSubgoalScorer(2.0, MetadataDispersionSource()).select_best(candidates, engine_choice="A") == "A"
        )  # large penalty flips to the safer A

    def test_crossover_boundary_selection(self):  # AC-1
        # A=(value 1.0, disp 0.1), B=(value 1.5, disp 0.9). score(A)==score(B) when
        # 1.0 - 0.1*lam == 1.5 - 0.9*lam  ->  lam* = 0.5/0.8 = 0.625.
        candidates = [_rec("A", 1.0, 0.1), _rec("B", 1.5, 0.9)]
        # Just below the crossover: higher-value B still wins.
        assert (
            RiskAverseSubgoalScorer(0.624, MetadataDispersionSource()).select_best(candidates, engine_choice="A") == "B"
        )
        # Just above the crossover: the safer A wins.
        assert (
            RiskAverseSubgoalScorer(0.626, MetadataDispersionSource()).select_best(candidates, engine_choice="A") == "A"
        )
        # Exactly at the crossover: equal scores -> first-wins tie-break -> A.
        scorer_at = RiskAverseSubgoalScorer(0.625, MetadataDispersionSource())
        assert scorer_at.score(candidates[0]) == pytest.approx(scorer_at.score(candidates[1]))
        assert scorer_at.select_best(candidates, engine_choice="A") == "A"

    def test_tie_breaks_first_wins(self):  # AC-1
        # Equal risk-adjusted score -> first candidate in order wins (matches max()).
        candidates = [_rec("A", 1.0, 0.5), _rec("B", 1.0, 0.5)]
        scorer = RiskAverseSubgoalScorer(1.0, MetadataDispersionSource())
        assert scorer.select_best(candidates, engine_choice="B") == "A"

    def test_empty_candidates_return_engine_choice(self):
        assert RiskAverseSubgoalScorer(1.0).select_best([], engine_choice="X") == "X"

    def test_is_deterministic(self):  # AC-1
        candidates = [_rec("A", 0.8, 0.2), _rec("B", 0.9, 0.5), _rec("C", 0.85, 0.1)]
        scorer = RiskAverseSubgoalScorer(1.0, MetadataDispersionSource())
        results = {scorer.select_best(candidates, engine_choice="A") for _ in range(50)}
        assert len(results) == 1


class TestValidationAndProtocol:
    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_weight must be"):
            RiskAverseSubgoalScorer(lambda_weight=-0.5)

    def test_name_is_risk_averse(self):
        assert RiskAverseSubgoalScorer().name == CANDIDATE_SCORER_RISK_AVERSE

    def test_satisfies_candidate_scorer_protocol(self):
        assert isinstance(RiskAverseSubgoalScorer(), CandidateScorer)


class TestDispersionSources:
    def test_zero_source_disables_penalty(self):
        scorer = RiskAverseSubgoalScorer(lambda_weight=5.0, dispersion_source=ZeroDispersionSource())
        assert scorer.score(_rec("a", 1.0, 9.9)) == pytest.approx(1.0)

    def test_metadata_source_missing_key_is_zero(self):
        source = MetadataDispersionSource()
        assert source.dispersion_for(CandidateRecord("a", 1.0, 1)) == 0.0

    def test_callable_source(self):
        source = CallableDispersionSource(lambda c: c.visits * 0.1)
        assert source.dispersion_for(CandidateRecord("a", 1.0, 4)) == pytest.approx(0.4)

    def test_default_source_is_metadata(self):
        scorer = RiskAverseSubgoalScorer(lambda_weight=1.0)
        # default MetadataDispersionSource reads metadata['dispersion']
        assert scorer.score(_rec("a", 1.0, 0.25)) == pytest.approx(0.75)

    def test_metadata_source_clamps_negative_dispersion(self):
        # A negative dispersion must not reward uncertainty: clamp to 0.
        assert MetadataDispersionSource().dispersion_for(_rec("a", 1.0, -5.0)) == 0.0

    def test_callable_source_clamps_negative_dispersion(self):
        source = CallableDispersionSource(lambda c: -1.0)
        assert source.dispersion_for(CandidateRecord("a", 1.0, 1)) == 0.0

    def test_negative_dispersion_does_not_increase_score(self):
        scorer = RiskAverseSubgoalScorer(lambda_weight=10.0, dispersion_source=MetadataDispersionSource())
        # metadata dispersion = -0.5 clamps to 0 -> score == value (never > value).
        assert scorer.score(_rec("a", 1.0, -0.5)) == pytest.approx(1.0)


class TestMisconfigurationWarning:
    """A2: warn once when the penalty is active (lambda > 0) but no dispersion signal exists."""

    def test_warns_once_when_all_dispersions_zero(self, caplog):
        scorer = RiskAverseSubgoalScorer(lambda_weight=1.0, dispersion_source=ZeroDispersionSource())
        candidates = [_rec("A", 1.0, 0.0), _rec("B", 0.9, 0.0)]
        with caplog.at_level("WARNING", logger="src.framework.mcts.risk_scoring"):
            scorer.select_best(candidates, engine_choice="A")
            scorer.select_best(candidates, engine_choice="A")  # second call must not re-warn
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "pure value ranking" in warnings[0].getMessage()

    def test_no_warning_when_dispersion_present(self, caplog):
        scorer = RiskAverseSubgoalScorer(lambda_weight=1.0, dispersion_source=MetadataDispersionSource())
        candidates = [_rec("A", 1.0, 0.0), _rec("B", 0.9, 0.5)]  # B carries real dispersion
        with caplog.at_level("WARNING", logger="src.framework.mcts.risk_scoring"):
            scorer.select_best(candidates, engine_choice="A")
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_no_warning_when_lambda_zero(self, caplog):
        # lambda == 0 is a legitimate pure-value config, not a misconfiguration.
        scorer = RiskAverseSubgoalScorer(lambda_weight=0.0, dispersion_source=ZeroDispersionSource())
        candidates = [_rec("A", 1.0, 0.0), _rec("B", 0.9, 0.0)]
        with caplog.at_level("WARNING", logger="src.framework.mcts.risk_scoring"):
            scorer.select_best(candidates, engine_choice="A")
        assert not [r for r in caplog.records if r.levelname == "WARNING"]
