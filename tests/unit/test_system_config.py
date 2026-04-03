"""Unit tests for src/training/system_config.py."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
import torch

from src.training.system_config import (
    HRMConfig,
    MCTSConfig,
    NeuralNetworkConfig,
    SystemConfig,
    TrainingConfig,
    TRMConfig,
    get_arc_agi_config,
    get_large_config,
    get_medium_config,
    get_small_config,
)


@pytest.mark.unit
class TestHRMConfig:
    """Tests for HRMConfig dataclass."""

    def test_defaults(self):
        cfg = HRMConfig()
        assert cfg.h_dim == 512
        assert cfg.l_dim == 256
        assert cfg.num_h_layers == 2
        assert cfg.num_l_layers == 4
        assert cfg.max_outer_steps == 10
        assert cfg.halt_threshold == 0.95
        assert cfg.use_augmentation is True
        assert cfg.dropout == 0.1
        assert cfg.ponder_epsilon == 0.01
        assert cfg.max_ponder_steps == 16
        assert cfg.ponder_weight == 0.01
        assert cfg.consistency_weight == 0.1

    def test_custom_values(self):
        cfg = HRMConfig(h_dim=1024, l_dim=512, dropout=0.2)
        assert cfg.h_dim == 1024
        assert cfg.l_dim == 512
        assert cfg.dropout == 0.2
        # Other defaults unchanged
        assert cfg.num_h_layers == 2


@pytest.mark.unit
class TestTRMConfig:
    """Tests for TRMConfig dataclass."""

    def test_defaults(self):
        cfg = TRMConfig()
        assert cfg.latent_dim == 256
        assert cfg.num_recursions == 16
        assert cfg.hidden_dim == 512
        assert cfg.deep_supervision is True
        assert cfg.supervision_weight_decay == 0.5
        assert cfg.convergence_threshold == 0.01
        assert cfg.min_recursions == 3
        assert cfg.dropout == 0.1
        assert cfg.use_layer_norm is True

    def test_custom_values(self):
        cfg = TRMConfig(latent_dim=128, num_recursions=8, deep_supervision=False)
        assert cfg.latent_dim == 128
        assert cfg.num_recursions == 8
        assert cfg.deep_supervision is False


@pytest.mark.unit
class TestMCTSConfig:
    """Tests for MCTSConfig dataclass."""

    def test_defaults(self):
        cfg = MCTSConfig()
        assert cfg.num_simulations == 1600
        assert cfg.c_puct == 1.25
        assert cfg.dirichlet_epsilon == 0.25
        assert cfg.dirichlet_alpha == 0.3
        assert cfg.temperature_threshold == 30
        assert cfg.temperature_init == 1.0
        assert cfg.temperature_final == 0.1
        assert cfg.virtual_loss == 3.0
        assert cfg.num_parallel == 8
        assert cfg.use_progressive_widening is True
        assert cfg.pw_k == 1.0
        assert cfg.pw_alpha == 0.5

    def test_custom_values(self):
        cfg = MCTSConfig(num_simulations=800, c_puct=2.0)
        assert cfg.num_simulations == 800
        assert cfg.c_puct == 2.0


@pytest.mark.unit
class TestNeuralNetworkConfig:
    """Tests for NeuralNetworkConfig dataclass."""

    def test_defaults(self):
        cfg = NeuralNetworkConfig()
        assert cfg.num_res_blocks == 19
        assert cfg.num_channels == 256
        assert cfg.policy_conv_channels == 2
        assert cfg.policy_fc_dim == 256
        assert cfg.value_conv_channels == 1
        assert cfg.value_fc_hidden == 256
        assert cfg.use_batch_norm is True
        assert cfg.dropout == 0.0
        assert cfg.weight_decay == 1e-4
        assert cfg.input_channels == 17
        assert cfg.action_size == 362

    def test_custom_values(self):
        cfg = NeuralNetworkConfig(num_res_blocks=39, action_size=100)
        assert cfg.num_res_blocks == 39
        assert cfg.action_size == 100


@pytest.mark.unit
class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.games_per_iteration == 25_000
        assert cfg.num_actors == 128
        assert cfg.buffer_size == 500_000
        assert cfg.batch_size == 2048
        assert cfg.learning_rate == 0.2
        assert cfg.momentum == 0.9
        assert cfg.weight_decay == 1e-4
        assert cfg.lr_schedule == "cosine"
        assert cfg.lr_decay_steps == 100
        assert cfg.lr_decay_gamma == 0.1
        assert cfg.epochs_per_iteration == 1
        assert cfg.checkpoint_interval == 10
        assert cfg.hrm_train_batches == 10
        assert cfg.trm_train_batches == 10
        assert cfg.gradient_clip_norm == 1.0
        assert cfg.evaluation_games == 400
        assert cfg.win_rate_threshold == 0.55
        assert cfg.win_threshold == 0.55
        assert cfg.eval_temperature == 0.0
        assert cfg.patience == 20
        assert cfg.min_delta == 0.01

    def test_custom_values(self):
        cfg = TrainingConfig(batch_size=512, learning_rate=0.01)
        assert cfg.batch_size == 512
        assert cfg.learning_rate == 0.01


@pytest.mark.unit
class TestSystemConfig:
    """Tests for SystemConfig dataclass."""

    def test_defaults(self):
        cfg = SystemConfig()
        assert isinstance(cfg.hrm, HRMConfig)
        assert isinstance(cfg.trm, TRMConfig)
        assert isinstance(cfg.mcts, MCTSConfig)
        assert isinstance(cfg.neural_net, NeuralNetworkConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert cfg.seed == 42
        assert cfg.device in ("cpu", "cuda")
        assert cfg.gradient_checkpointing is False
        assert cfg.compile_model is False
        assert cfg.world_size == 1
        assert cfg.rank == 0
        assert cfg.log_interval == 10
        assert cfg.use_wandb is False
        assert cfg.wandb_project == "langgraph-mcts-deepmind"
        assert cfg.wandb_entity is None
        assert cfg.checkpoint_dir == "./checkpoints"
        assert cfg.data_dir == "./data"
        assert cfg.log_dir == "./logs"

    def test_cpu_post_init_disables_mixed_precision(self):
        """When device is cpu, mixed precision and distributed should be off."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            cfg = SystemConfig(device="cpu")
            assert cfg.use_mixed_precision is False
            assert cfg.distributed is False
            assert cfg.backend == "gloo"

    def test_cuda_fallback_when_unavailable(self):
        """Requesting cuda when unavailable should fall back to cpu."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            cfg = SystemConfig(device="cuda")
            assert cfg.device == "cpu"
            assert cfg.use_mixed_precision is False

    def test_to_dict(self):
        cfg = SystemConfig()
        d = cfg.to_dict()
        assert "hrm" in d
        assert "trm" in d
        assert "mcts" in d
        assert "neural_net" in d
        assert "training" in d
        assert "device" in d
        assert "seed" in d
        assert "use_mixed_precision" in d
        assert "gradient_checkpointing" in d
        assert "compile_model" in d
        assert "distributed" in d
        # Nested dicts have correct keys
        assert "h_dim" in d["hrm"]
        assert "latent_dim" in d["trm"]
        assert "num_simulations" in d["mcts"]

    def test_from_dict(self):
        original = SystemConfig()
        d = original.to_dict()
        d["hrm"]["h_dim"] = 1024
        d["seed"] = 99
        restored = SystemConfig.from_dict(d)
        assert restored.hrm.h_dim == 1024
        assert restored.seed == 99

    def test_from_dict_partial(self):
        """from_dict should work with partial config dicts."""
        restored = SystemConfig.from_dict({"seed": 7})
        assert restored.seed == 7
        # Defaults preserved
        assert restored.hrm.h_dim == 512

    def test_from_dict_empty(self):
        """from_dict with empty dict gives defaults."""
        restored = SystemConfig.from_dict({})
        assert restored.hrm.h_dim == 512
        assert restored.seed == 42

    def test_save_and_load(self):
        cfg = SystemConfig()
        cfg.hrm.h_dim = 1024
        cfg.seed = 99

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            cfg.save(path)
            loaded = SystemConfig.load(path)
            assert loaded.hrm.h_dim == 1024
            assert loaded.seed == 99
        finally:
            os.unlink(path)

    def test_to_dict_round_trip_json_serializable(self):
        """to_dict output should be JSON-serializable."""
        cfg = SystemConfig()
        d = cfg.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


@pytest.mark.unit
class TestPresetConfigs:
    """Tests for preset configuration functions."""

    def test_get_small_config(self):
        cfg = get_small_config()
        assert cfg.hrm.h_dim == 256
        assert cfg.hrm.l_dim == 128
        assert cfg.trm.latent_dim == 128
        assert cfg.neural_net.num_res_blocks == 9
        assert cfg.neural_net.num_channels == 128
        assert cfg.mcts.num_simulations == 400
        assert cfg.training.games_per_iteration == 1000
        assert cfg.training.num_actors == 16

    def test_get_medium_config(self):
        cfg = get_medium_config()
        assert cfg.neural_net.num_res_blocks == 19
        assert cfg.mcts.num_simulations == 800
        assert cfg.training.games_per_iteration == 10_000
        assert cfg.training.num_actors == 64

    def test_get_large_config(self):
        cfg = get_large_config()
        assert cfg.hrm.h_dim == 768
        assert cfg.hrm.l_dim == 384
        assert cfg.trm.latent_dim == 384
        assert cfg.neural_net.num_res_blocks == 39
        assert cfg.neural_net.num_channels == 384
        assert cfg.mcts.num_simulations == 3200
        assert cfg.training.games_per_iteration == 50_000
        assert cfg.training.num_actors == 256
        assert cfg.gradient_checkpointing is True

    def test_get_arc_agi_config(self):
        cfg = get_arc_agi_config()
        assert cfg.hrm.max_outer_steps == 20
        assert cfg.trm.num_recursions == 20
        assert cfg.trm.convergence_threshold == 0.005
        assert cfg.mcts.num_simulations == 1600
        assert cfg.mcts.c_puct == 1.5
        assert cfg.neural_net.input_channels == 11
        assert cfg.neural_net.action_size == 100

    def test_presets_return_system_config(self):
        for fn in [get_small_config, get_medium_config, get_large_config, get_arc_agi_config]:
            cfg = fn()
            assert isinstance(cfg, SystemConfig)
