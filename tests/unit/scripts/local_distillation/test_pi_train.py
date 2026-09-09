"""local_distillation_contract AC-1: π_train is visit/sum, not play temperature."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.local_distillation.collector import HygienicCollector
from scripts.local_distillation.policy_targets import visits_to_policy
from scripts.local_distillation.settings import DistillationSettings
from src.framework.mcts.neural_mcts import NeuralMCTS, NeuralMCTSNode
from src.training.system_config import MCTSConfig
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, BiasedNet, TwoPlyState

pytestmark = [pytest.mark.unit]


def test_visits_to_policy_tau1_is_visit_over_sum() -> None:
    visits = np.array([100.0, 50.0, 10.0, 0.0])
    expected = np.array([0.625, 0.3125, 0.0625, 0.0])
    np.testing.assert_allclose(visits_to_policy(visits, temperature=1.0), expected)


def test_play_temperature_is_not_the_training_target() -> None:
    visits = np.array([100.0, 50.0, 10.0, 0.0])
    train = visits_to_policy(visits, temperature=1.0)
    play = visits_to_policy(visits, temperature=0.1)
    assert not np.allclose(train, play)
    # τ=0.1 ⇒ N^10; 100/50 = 2 ⇒ 2^10 = 1024 between the top two masses.
    assert play[0] / play[1] == pytest.approx(1024.0)


def test_empty_legal_yields_empty_policy() -> None:
    empty = visits_to_policy(np.array([], dtype=np.float64))
    assert empty.shape == (0,)


def test_neural_mcts_node_play_probs_diverge_from_visit_sum() -> None:
    root = NeuralMCTSNode(TwoPlyState())
    for action, count in ((0, 100), (1, 10)):
        child = NeuralMCTSNode(TwoPlyState(ply=1), parent=root, action=action)
        child.visit_count = count
        root.children[action] = child
    play = root.get_action_probs(0.1)
    train = visits_to_policy(np.array([100.0, 10.0]), temperature=1.0)
    assert play[0] != pytest.approx(train[0])
    np.testing.assert_allclose(visits_to_policy(np.array([100.0, 10.0]), temperature=1.0), [100 / 110, 10 / 110])


@pytest.mark.asyncio
async def test_collector_stores_visit_sum_after_temperature_threshold() -> None:
    settings = DistillationSettings(
        default_simulations=16,
        temperature_threshold=0,
        temperature_final=0.1,
        temperature_init=1.0,
    )
    network = BiasedNet(bias=(0.5, 0.0))
    mcts = NeuralMCTS(
        network,
        MCTSConfig(num_simulations=16, virtual_loss=0.0),
        device="cpu",
        seed=0,
    )
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    rows = await collector.play_game(TwoPlyState(), game_id="skew")
    assert rows
    row = rows[0]
    train = visits_to_policy(row.visit_counts, temperature=1.0)
    np.testing.assert_allclose(row.policy_target, train)
    play = visits_to_policy(row.visit_counts, temperature=settings.temperature_final)
    assert not np.allclose(row.policy_target, play)
