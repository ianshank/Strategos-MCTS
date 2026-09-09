"""local_distillation_contract AC-6: schema, grouped splits, zero-visit, empty legal."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.local_distillation.policy_targets import visits_to_policy
from scripts.local_distillation.schema import TrajectoryRow, grouped_split, validate_row
from scripts.local_distillation.settings import DistillationSettings
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.system_config import MCTSConfig
from src.utils.seeding import new_rng
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, CountingNet, TwoPlyState

pytestmark = [pytest.mark.unit]


def _row(game_id: str, token: float, visits: np.ndarray | None = None) -> TrajectoryRow:
    visits = np.array([4.0, 1.0]) if visits is None else visits
    return TrajectoryRow(
        game_id=game_id,
        ply=0,
        state=torch.tensor([token, 1.0]),
        visit_counts=visits,
        policy_target=visits / visits.sum() if visits.sum() > 0 else visits,
        legal_mask=np.array([True, True]),
        value_target=1.0,
        current_player=1,
    )


def test_zero_visit_legal_action_is_kept() -> None:
    row = _row("g0", token=0.0, visits=np.array([5.0, 0.0]))
    validate_row(row, action_size=ACTION_SPACE)
    assert row.policy_target[1] == pytest.approx(0.0)
    assert bool(row.legal_mask[1])


def test_all_zero_and_nan_rows_are_rejected() -> None:
    zero = _row("g0", token=0.0, visits=np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="all-zero"):
        validate_row(zero, action_size=ACTION_SPACE)
    nan_row = _row("g1", token=1.0, visits=np.array([1.0, 1.0]))
    nan_row.policy_target = np.array([np.nan, 0.0])
    with pytest.raises(ValueError, match="finite"):
        validate_row(nan_row, action_size=ACTION_SPACE)


def test_grouped_split_keeps_games_together() -> None:
    rows = [
        _row("game-a", token=0.0),
        _row("game-a", token=0.1),
        _row("game-b", token=1.0),
        _row("game-b", token=1.1),
        _row("game-c", token=2.0),
        _row("game-c", token=2.1),
        _row("game-d", token=3.0),
        _row("game-d", token=3.1),
    ]
    settings = DistillationSettings(train_frac=0.5, val_frac=0.25)
    train, val, test = grouped_split(rows, settings, new_rng(0))
    for split in (train, val, test):
        ids = {row.game_id for row in split}
        for gid in ids:
            n_in_split = sum(1 for row in split if row.game_id == gid)
            n_total = sum(1 for row in rows if row.game_id == gid)
            assert n_in_split == n_total


def test_overlapping_state_hashes_across_splits_fail() -> None:
    rows = [
        _row("game-a", token=0.0),
        _row("game-b", token=0.0),
    ]
    settings = DistillationSettings(train_frac=0.5, val_frac=0.25)
    with pytest.raises(ValueError, match="hash overlap"):
        grouped_split(rows, settings, new_rng(0))


@pytest.mark.asyncio
async def test_empty_legal_evaluate_state_returns_empty_policy() -> None:
    network = CountingNet()
    mcts = NeuralMCTS(
        network,
        MCTSConfig(num_simulations=2, virtual_loss=0.0),
        device="cpu",
        seed=0,
    )
    probs, _value = await mcts.evaluate_state(TwoPlyState(ply=2), add_noise=False)
    assert list(probs) == []
    np.testing.assert_array_equal(visits_to_policy(np.array([])), np.array([]))
