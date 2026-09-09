"""local_distillation_contract AC-9/AC-10/AC-11: search vs no-search arms and promotion reject."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.local_distillation.eval_arms import (
    COMMITTED_RESULTS_RELATIVE_PATH,
    PRIMARY_ENDPOINT,
    compare_search_vs_no_search,
    decide_promotion,
)
from src.config.constants import EVIDENCE_PROVENANCES
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.system_config import MCTSConfig
from src.utils.seeding import new_rng
from tests.unit.scripts.local_distillation.toys import CountingNet, TwoPlyState

pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_compare_search_vs_no_search_records_expansions_and_wall_clock() -> None:
    network = CountingNet()
    mcts = NeuralMCTS(
        network,
        MCTSConfig(num_simulations=4, virtual_loss=0.0, dirichlet_epsilon=0.0),
        device="cpu",
        seed=0,
    )
    comparison = await compare_search_vs_no_search(
        mcts,
        TwoPlyState(),
        num_simulations=4,
        device="cpu",
        rng=new_rng(0),
    )
    assert comparison.no_search.expansions == 0
    assert comparison.search.expansions == 4
    assert comparison.no_search.wall_clock_s >= 0.0
    assert comparison.search.wall_clock_s >= 0.0
    assert comparison.no_search.provenance in EVIDENCE_PROVENANCES
    assert comparison.search.provenance in EVIDENCE_PROVENANCES
    assert comparison.no_search_repeats_in_search_budget >= 0
    assert comparison.primary_endpoint == PRIMARY_ENDPOINT


def test_decide_promotion_rejects_degraded_checkpoint() -> None:
    decision = decide_promotion(0.40, 0.55, min_delta=0.0)
    assert decision.promote is False
    assert "rejected" in decision.reason


def test_decide_promotion_accepts_clear_improvement() -> None:
    decision = decide_promotion(0.70, 0.55, min_delta=0.05)
    assert decision.promote is True


def test_primary_endpoint_declared_and_committed_json_is_not_this_pr() -> None:
    assert PRIMARY_ENDPOINT == "search_minus_no_search"
    assert COMMITTED_RESULTS_RELATIVE_PATH == "benchmarks/results/local_distillation_c4.json"
    repo = Path(__file__).resolve().parents[4]
    artifact = repo / COMMITTED_RESULTS_RELATIVE_PATH
    assert not artifact.exists(), "this PR must not treat a holdout JSON as evidence"
