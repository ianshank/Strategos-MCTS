"""Student/trainer tensors must land on DistillationSettings.device."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.local_distillation.collector import HygienicCollector, build_mcts
from scripts.local_distillation.device import place_network
from scripts.local_distillation.driver import HygienicTrainer
from scripts.local_distillation.schema import TrajectoryRow
from scripts.local_distillation.settings import DistillationSettings
from scripts.local_distillation.student import build_student
from src.utils.seeding import new_rng
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, CountingNet

pytestmark = [pytest.mark.unit]


class _TrackingNet(CountingNet):
    """Records ``to()`` so CPU hosts can still catch a missing placement call."""

    def __init__(self) -> None:
        super().__init__()
        self.to_devices: list[str] = []

    def to(self, device, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.to_devices.append(str(device))
        return super().to(device, *args, **kwargs)


def _toy_row() -> TrajectoryRow:
    return TrajectoryRow(
        game_id="g0",
        ply=0,
        state=torch.tensor([0.0, 1.0]),
        visit_counts=np.array([1.0, 1.0], dtype=np.float64),
        policy_target=np.array([0.5, 0.5], dtype=np.float64),
        legal_mask=np.array([True, True]),
        value_target=1.0,
        current_player=1,
    )


def test_place_network_records_to_and_binds_mcts() -> None:
    network = _TrackingNet()

    class _Tree:
        def __init__(self) -> None:
            self.device = "cpu"
            self.network = network

    tree = _Tree()
    placed = place_network(network, "cpu", mcts=tree)
    assert network.to_devices == ["cpu"]
    assert tree.device == "cpu"
    assert tree.network is placed


def test_build_mcts_moves_network_to_device() -> None:
    settings = DistillationSettings(device="cpu", default_simulations=2, min_simulations=1)
    network = _TrackingNet()
    mcts = build_mcts(network, settings, device="cpu", seed=0)
    assert network.to_devices, "build_mcts must call network.to(device)"
    assert next(mcts.network.parameters()).device.type == "cpu"


def test_trainer_moves_network_to_configured_device() -> None:
    settings = DistillationSettings(device="cpu", batch_size=1, default_simulations=2, min_simulations=1)
    network = _TrackingNet()
    mcts = build_mcts(network, settings, device="cpu", seed=0)
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    trainer = HygienicTrainer(network, collector, settings, new_rng(0), device="cpu")
    assert "cpu" in "".join(network.to_devices)
    trainer.buffer.append(_toy_row())
    metrics = trainer.train_step()
    assert metrics is not None
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_build_student_parameters_match_settings_device() -> None:
    settings = DistillationSettings(device="cpu", num_res_blocks=1, num_channels=8)
    student = build_student(settings)
    assert next(student.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cross-device train_step")
def test_train_step_runs_when_settings_device_is_cuda() -> None:
    settings = DistillationSettings(device="cuda", batch_size=1, default_simulations=2, min_simulations=1)
    network = CountingNet()
    mcts = build_mcts(network, settings, device="cuda", seed=0)
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    trainer = HygienicTrainer(network, collector, settings, new_rng(0))
    assert next(trainer.network.parameters()).device.type == "cuda"
    assert trainer.collector.mcts.device == "cuda"
    trainer.buffer.append(_toy_row())
    metrics = trainer.train_step()
    assert metrics is not None
