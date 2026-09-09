"""local_distillation_contract AC-4: STM z from stored current_player, including P2-to-move."""

from __future__ import annotations

import pytest

from scripts.local_distillation.collector import HygienicCollector, _assign_stm_values
from scripts.local_distillation.schema import TrajectoryRow, validate_row
from scripts.local_distillation.settings import DistillationSettings
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.system_config import MCTSConfig
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, CountingNet, TwoPlyState

pytestmark = [pytest.mark.unit]


def _row(*, current_player: int, token: float) -> TrajectoryRow:
    import numpy as np
    import torch

    visits = np.array([2.0, 1.0])
    return TrajectoryRow(
        game_id="mid",
        ply=1,
        state=torch.tensor([token, float(current_player)]),
        visit_counts=visits,
        policy_target=visits / visits.sum(),
        legal_mask=np.array([True, True]),
        value_target=0.0,
        current_player=current_player,
    )


def test_assign_stm_values_uses_row_player_not_p1_counter() -> None:
    terminal = TwoPlyState(ply=2, current_player=1)
    assert terminal.is_terminal()
    row = _row(current_player=-1, token=0.5)
    _assign_stm_values([row], terminal=terminal, single_agent=False)
    # P1 always wins the toy; STM at the stored ply is P2, so z = -1.
    assert row.value_target == pytest.approx(-1.0)
    validate_row(row, action_size=ACTION_SPACE)
    # Independent player=1 counter would have assigned +1.
    assert terminal.get_reward(player=1) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_play_game_from_midgame_p2_root() -> None:
    network = CountingNet()
    mcts = NeuralMCTS(
        network,
        MCTSConfig(num_simulations=4, virtual_loss=0.0, dirichlet_epsilon=0.0),
        device="cpu",
        seed=0,
        single_agent=False,
    )
    collector = HygienicCollector(
        mcts,
        DistillationSettings(default_simulations=4),
        action_space_size=ACTION_SPACE,
    )
    rows = await collector.play_game(TwoPlyState(ply=1, current_player=-1), game_id="p2")
    assert len(rows) == 1
    assert rows[0].current_player == -1
    assert rows[0].value_target == pytest.approx(-1.0)
    validate_row(rows[0], action_size=ACTION_SPACE)
