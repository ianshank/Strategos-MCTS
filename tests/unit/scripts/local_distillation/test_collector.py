"""Collector contract: NeuralMCTS-only teacher, visit/sum rows, injected RNG."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.local_distillation.collector import HygienicCollector, assert_neural_mcts_teacher, build_mcts
from scripts.local_distillation.settings import DistillationSettings
from src.framework.mcts.core import MCTSEngine
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.system_config import MCTSConfig
from src.utils.seeding import new_rng
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, CountingNet, TwoPlyState

pytestmark = [pytest.mark.unit]


def test_assert_neural_mcts_teacher_rejects_core_engine() -> None:
    with pytest.raises(TypeError, match="NeuralMCTS"):
        assert_neural_mcts_teacher(MCTSEngine())  # type: ignore[arg-type]


def test_zero_simulations_refused() -> None:
    settings = DistillationSettings(min_simulations=1, default_simulations=8)
    network = CountingNet()
    mcts = NeuralMCTS(network, MCTSConfig(num_simulations=0), device="cpu", seed=0)
    with pytest.raises(ValueError, match="num_simulations"):
        HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)


@pytest.mark.asyncio
async def test_play_game_rows_have_visit_sum_and_legal_zeros() -> None:
    settings = DistillationSettings(default_simulations=8, temperature_threshold=30)
    network = CountingNet()
    mcts = build_mcts(network, settings, device="cpu", seed=0, single_agent=False, rng=new_rng(0))
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    rows = await collector.play_game(TwoPlyState(), game_id="g0")
    assert len(rows) == 2
    for row in rows:
        assert row.game_id == "g0"
        assert row.visit_counts.shape == (ACTION_SPACE,)
        assert row.policy_target.sum() == pytest.approx(1.0)
        assert float(row.visit_counts.sum()) > 0
        np.testing.assert_allclose(row.policy_target, row.visit_counts / row.visit_counts.sum())


@pytest.mark.asyncio
async def test_generate_batch_uses_injected_rng_for_game_ids() -> None:
    settings = DistillationSettings(default_simulations=4)
    network = CountingNet()
    mcts = build_mcts(network, settings, device="cpu", seed=1, rng=new_rng(1))
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    rows_a = await collector.generate_batch(1, TwoPlyState, new_rng(7))
    mcts.clear_cache()
    rows_b = await collector.generate_batch(1, TwoPlyState, new_rng(7))
    assert rows_a[0].game_id == rows_b[0].game_id
    rows_c = await collector.generate_batch(1, TwoPlyState, new_rng(8))
    assert rows_c[0].game_id != rows_a[0].game_id
