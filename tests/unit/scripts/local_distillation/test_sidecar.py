"""local_distillation_contract AC-7: C4 sidecar must pin 3×6×7; missing board_rows errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.local_distillation.collector import HygienicCollector
from scripts.local_distillation.driver import HygienicTrainer
from scripts.local_distillation.settings import DistillationSettings
from scripts.local_distillation.sidecar import SidecarError, c4_network_architecture, validate_c4_sidecar
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.training.system_config import MCTSConfig
from src.utils.seeding import new_rng
from tests.unit.scripts.local_distillation.toys import ACTION_SPACE, CountingNet

pytestmark = [pytest.mark.unit]


def test_c4_architecture_pins_shape_and_blocks() -> None:
    settings = DistillationSettings()
    arch = c4_network_architecture(settings)
    assert arch["type"] == "resnet"
    assert arch["input_channels"] == 3
    assert arch["board_rows"] == 6
    assert arch["board_cols"] == 7
    assert arch["action_size"] == 7
    assert arch["num_res_blocks"] == settings.num_res_blocks
    assert arch["num_channels"] == settings.num_channels
    validate_c4_sidecar({"network": arch}, settings)


def test_missing_board_rows_is_error() -> None:
    settings = DistillationSettings()
    network = c4_network_architecture(settings)
    del network["board_rows"]
    with pytest.raises(SidecarError, match="board_rows"):
        validate_c4_sidecar({"network": network}, settings)


def test_missing_network_object_forbids_chess_fallback() -> None:
    settings = DistillationSettings()
    with pytest.raises(SidecarError, match="chess_default_architecture"):
        validate_c4_sidecar({}, settings)


def test_save_checkpoint_writes_c4_sidecar(tmp_path: Path) -> None:
    settings = DistillationSettings(default_simulations=4, batch_size=1)
    network = CountingNet()
    mcts = NeuralMCTS(network, MCTSConfig(num_simulations=4), device="cpu", seed=0)
    collector = HygienicCollector(mcts, settings, action_space_size=ACTION_SPACE)
    trainer = HygienicTrainer(network, collector, settings, new_rng(0), device="cpu")
    ckpt = tmp_path / "student.pt"
    trainer.save_checkpoint(ckpt)
    sidecar = ckpt.with_name(ckpt.name + ".meta.json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    validate_c4_sidecar(meta, settings)
