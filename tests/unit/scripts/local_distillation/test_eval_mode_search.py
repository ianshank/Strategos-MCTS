"""local_distillation_contract AC-2: search forwards run under eval()."""

from __future__ import annotations

import pytest

from scripts.local_distillation.collector import HygienicCollector
from scripts.local_distillation.settings import DistillationSettings
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.system_config import MCTSConfig
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, CountingNet, TwoPlyState

pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_search_forwards_see_eval_mode_then_train_is_restored() -> None:
    network = CountingNet()
    network.train()
    mcts = NeuralMCTS(
        network,
        MCTSConfig(num_simulations=4, virtual_loss=0.0, dirichlet_epsilon=0.0),
        device="cpu",
        seed=0,
    )
    collector = HygienicCollector(
        mcts,
        DistillationSettings(default_simulations=4),
        action_space_size=ACTION_SPACE,
    )
    await collector.play_game(TwoPlyState(), game_id="eval")
    assert network.modes, "search must call the network"
    assert all(mode is False for mode in network.modes)
    assert network.training is True
