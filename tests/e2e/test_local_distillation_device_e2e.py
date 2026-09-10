"""Distillation trainer/collector placement on the e2e device matrix."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.local_distillation.collector import HygienicCollector, build_mcts
from scripts.local_distillation.driver import HygienicTrainer
from scripts.local_distillation.schema import TrajectoryRow
from scripts.local_distillation.settings import DistillationSettings
from scripts.local_distillation.toy_domain import ACTION_SPACE, CountingNet
from src.utils.seeding import new_rng

pytestmark = [pytest.mark.e2e, pytest.mark.neural]


def _toy_row() -> TrajectoryRow:
    return TrajectoryRow(
        game_id="e2e",
        ply=0,
        state=torch.tensor([0.0, 1.0]),
        visit_counts=np.array([1.0, 1.0], dtype=np.float64),
        policy_target=np.array([0.5, 0.5], dtype=np.float64),
        legal_mask=np.array([True, True]),
        value_target=1.0,
        current_player=1,
    )


def test_train_step_and_module_live_on_device_case(device: str) -> None:
    settings = DistillationSettings(
        device=device,
        batch_size=1,
        default_simulations=2,
        min_simulations=1,
    )
    network = CountingNet()
    mcts = build_mcts(network, settings, device=device, seed=0)
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    trainer = HygienicTrainer(network, collector, settings, new_rng(0), device=device)
    param_device = next(trainer.network.parameters()).device
    assert param_device.type == torch.device(device).type
    assert trainer.collector.mcts.device == device
    trainer.buffer.append(_toy_row())
    metrics = trainer.train_step()
    assert metrics is not None
    assert all(np.isfinite(value) for value in metrics.values())
