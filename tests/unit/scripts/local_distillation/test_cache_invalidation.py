"""local_distillation_contract AC-3: train_step clears the network-blind eval cache."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.local_distillation.collector import HygienicCollector
from scripts.local_distillation.driver import HygienicTrainer
from scripts.local_distillation.schema import TrajectoryRow
from scripts.local_distillation.settings import DistillationSettings
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.system_config import MCTSConfig
from src.utils.seeding import new_rng
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, CountingNet, TwoPlyState

pytestmark = [pytest.mark.unit]


def _row() -> TrajectoryRow:
    visits = np.array([3.0, 1.0])
    return TrajectoryRow(
        game_id="g0",
        ply=0,
        state=TwoPlyState().to_tensor(),
        visit_counts=visits,
        policy_target=visits / visits.sum(),
        legal_mask=np.array([True, True]),
        value_target=1.0,
        current_player=1,
    )


@pytest.mark.asyncio
async def test_train_step_clears_eval_cache() -> None:
    network = CountingNet()
    mcts = NeuralMCTS(
        network,
        MCTSConfig(num_simulations=4, virtual_loss=0.0, dirichlet_epsilon=0.0),
        device="cpu",
        seed=0,
    )
    state = TwoPlyState()
    await mcts.evaluate_state(state, add_noise=False)
    assert len(mcts.cache) > 0
    settings = DistillationSettings(default_simulations=4, batch_size=1)
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    trainer = HygienicTrainer(network, collector, settings, new_rng(0), device="cpu")
    trainer.buffer.append(_row())
    losses = trainer.train_step()
    assert losses is not None
    assert len(mcts.cache) == 0
    assert network.training is False
