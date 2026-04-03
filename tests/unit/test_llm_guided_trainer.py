"""Unit tests for src/framework/mcts/llm_guided/training/trainer.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# We need to handle torch being optional
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available"),
]

from src.framework.mcts.llm_guided.training.metrics import TrainingMetrics
from src.framework.mcts.llm_guided.training.trainer import (
    DistillationTrainer,
    DistillationTrainerConfig,
    LoggingCallback,
    TrainingCallback,
    TrainingCheckpoint,
    create_trainer,
)

# ---------------------------------------------------------------------------
# DistillationTrainerConfig tests
# ---------------------------------------------------------------------------


class TestDistillationTrainerConfig:
    def test_default_values(self):
        cfg = DistillationTrainerConfig()
        assert cfg.num_epochs == 10
        assert cfg.learning_rate == 1e-4
        assert cfg.weight_decay == 0.01
        assert cfg.warmup_steps == 100
        assert cfg.max_grad_norm == 1.0
        assert cfg.policy_loss_weight == 1.0
        assert cfg.value_loss_weight == 1.0
        assert cfg.use_mcts_policy is True
        assert cfg.use_outcome_value is True
        assert cfg.checkpoint_dir == "./checkpoints"
        assert cfg.save_every_epochs == 1
        assert cfg.keep_last_n_checkpoints == 3
        assert cfg.log_every_steps == 100
        assert cfg.eval_every_epochs == 1
        assert cfg.early_stopping_patience == 5
        assert cfg.early_stopping_metric == "total_loss"
        assert cfg.device == "auto"
        assert cfg.use_amp is False

    def test_validate_passes_with_defaults(self):
        cfg = DistillationTrainerConfig()
        cfg.validate()  # Should not raise

    def test_validate_num_epochs(self):
        cfg = DistillationTrainerConfig(num_epochs=0)
        with pytest.raises(ValueError, match="num_epochs"):
            cfg.validate()

    def test_validate_learning_rate(self):
        cfg = DistillationTrainerConfig(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate"):
            cfg.validate()

    def test_validate_negative_learning_rate(self):
        cfg = DistillationTrainerConfig(learning_rate=-1)
        with pytest.raises(ValueError, match="learning_rate"):
            cfg.validate()

    def test_validate_warmup_steps(self):
        cfg = DistillationTrainerConfig(warmup_steps=-1)
        with pytest.raises(ValueError, match="warmup_steps"):
            cfg.validate()

    def test_validate_max_grad_norm(self):
        cfg = DistillationTrainerConfig(max_grad_norm=0)
        with pytest.raises(ValueError, match="max_grad_norm"):
            cfg.validate()

    def test_validate_negative_loss_weight(self):
        cfg = DistillationTrainerConfig(policy_loss_weight=-1)
        with pytest.raises(ValueError, match="loss weights"):
            cfg.validate()

    def test_validate_invalid_metric(self):
        cfg = DistillationTrainerConfig(early_stopping_metric="invalid_metric")
        with pytest.raises(ValueError, match="early_stopping_metric"):
            cfg.validate()

    def test_validate_multiple_errors(self):
        cfg = DistillationTrainerConfig(
            num_epochs=0, learning_rate=-1, warmup_steps=-5
        )
        with pytest.raises(ValueError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "num_epochs" in msg
        assert "learning_rate" in msg
        assert "warmup_steps" in msg


# ---------------------------------------------------------------------------
# TrainingCheckpoint tests
# ---------------------------------------------------------------------------


class TestTrainingCheckpoint:
    def test_default_values(self):
        ckpt = TrainingCheckpoint(epoch=0, step=0)
        assert ckpt.epoch == 0
        assert ckpt.step == 0
        assert ckpt.policy_state_dict is None
        assert ckpt.value_state_dict is None
        assert ckpt.optimizer_state_dict is None
        assert ckpt.scheduler_state_dict is None
        assert ckpt.best_metric == float("inf")
        assert ckpt.metrics_history == []
        assert ckpt.config == {}

    def test_save_and_load(self, tmp_path):
        ckpt = TrainingCheckpoint(
            epoch=5,
            step=500,
            policy_state_dict={"weight": torch.tensor([1.0, 2.0])},
            value_state_dict={"weight": torch.tensor([3.0])},
            optimizer_state_dict={"lr": 0.001},
            best_metric=0.25,
            metrics_history=[{"loss": 0.5}],
            config={"num_epochs": 10},
        )
        filepath = tmp_path / "test_checkpoint.pt"
        ckpt.save(filepath)

        loaded = TrainingCheckpoint.load(filepath)
        assert loaded.epoch == 5
        assert loaded.step == 500
        assert loaded.best_metric == 0.25
        assert loaded.metrics_history == [{"loss": 0.5}]
        assert loaded.config == {"num_epochs": 10}
        assert torch.allclose(loaded.policy_state_dict["weight"], torch.tensor([1.0, 2.0]))

    def test_save_creates_parent_dirs(self, tmp_path):
        ckpt = TrainingCheckpoint(epoch=0, step=0)
        filepath = tmp_path / "nested" / "dir" / "ckpt.pt"
        ckpt.save(filepath)
        assert filepath.exists()

    def test_load_without_torch(self):
        with patch("src.framework.mcts.llm_guided.training.trainer._TORCH_AVAILABLE", False):
            with pytest.raises(ImportError):
                TrainingCheckpoint.load("fake.pt")

    def test_save_without_torch(self):
        with patch("src.framework.mcts.llm_guided.training.trainer._TORCH_AVAILABLE", False):
            ckpt = TrainingCheckpoint(epoch=0, step=0)
            with pytest.raises(ImportError):
                ckpt.save("fake.pt")


# ---------------------------------------------------------------------------
# LoggingCallback tests
# ---------------------------------------------------------------------------


class TestLoggingCallback:
    def test_on_epoch_start(self):
        cb = LoggingCallback()
        trainer = MagicMock()
        trainer._config.num_epochs = 10
        cb.on_epoch_start(0, trainer)  # Should not raise

    def test_on_epoch_end(self):
        cb = LoggingCallback()
        trainer = MagicMock()
        metrics = TrainingMetrics(
            policy_loss=0.5, value_loss=0.3, policy_accuracy=0.7, value_mse=0.1
        )
        cb.on_epoch_end(0, metrics, trainer)  # Should not raise

    def test_on_batch_end_logged(self):
        cb = LoggingCallback()
        trainer = MagicMock()
        trainer._config.log_every_steps = 100
        metrics = TrainingMetrics(total_loss=0.5, learning_rate=0.001)
        cb.on_batch_end(100, metrics, trainer)  # step == log_every_steps, should log

    def test_on_batch_end_not_logged(self):
        cb = LoggingCallback()
        trainer = MagicMock()
        trainer._config.log_every_steps = 100
        metrics = TrainingMetrics(total_loss=0.5, learning_rate=0.001)
        cb.on_batch_end(50, metrics, trainer)  # step != log_every_steps

    def test_on_training_end(self):
        cb = LoggingCallback()
        trainer = MagicMock()
        trainer._global_step = 500
        trainer._current_epoch = 9
        cb.on_training_end(trainer)  # Should not raise


# ---------------------------------------------------------------------------
# DistillationTrainer tests
# ---------------------------------------------------------------------------


class TestDistillationTrainer:
    def _make_simple_network(self):
        """Create a simple network for testing."""
        return torch.nn.Linear(10, 5)

    def test_init_defaults(self):
        trainer = DistillationTrainer()
        assert trainer._config.num_epochs == 10
        assert trainer._policy_network is None
        assert trainer._value_network is None
        assert trainer._current_epoch == 0
        assert trainer._global_step == 0
        assert trainer._best_metric == float("inf")

    def test_init_with_config(self):
        cfg = DistillationTrainerConfig(
            num_epochs=5, learning_rate=0.01, device="cpu"
        )
        trainer = DistillationTrainer(config=cfg)
        assert trainer._config.num_epochs == 5
        assert trainer._config.learning_rate == 0.01

    def test_init_invalid_config_raises(self):
        cfg = DistillationTrainerConfig(num_epochs=0)
        with pytest.raises(ValueError, match="num_epochs"):
            DistillationTrainer(config=cfg)

    def test_init_without_torch(self):
        with patch("src.framework.mcts.llm_guided.training.trainer._TORCH_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyTorch"):
                DistillationTrainer()

    def test_get_device_cpu(self):
        cfg = DistillationTrainerConfig(device="cpu")
        trainer = DistillationTrainer(config=cfg)
        assert trainer._device == torch.device("cpu")

    def test_get_device_auto_falls_to_cpu(self):
        cfg = DistillationTrainerConfig(device="auto")
        trainer = DistillationTrainer(config=cfg)
        # On most CI environments, this will be cpu
        assert trainer._device.type in ("cpu", "cuda", "mps")

    def test_create_optimizer_no_networks_raises(self):
        cfg = DistillationTrainerConfig(device="cpu")
        trainer = DistillationTrainer(config=cfg)
        with pytest.raises(ValueError, match="At least one network"):
            trainer._create_optimizer()

    def test_create_optimizer_with_policy(self):
        cfg = DistillationTrainerConfig(device="cpu")
        net = self._make_simple_network()
        trainer = DistillationTrainer(policy_network=net, config=cfg)
        optimizer = trainer._create_optimizer()
        assert optimizer is not None

    def test_create_optimizer_with_value(self):
        cfg = DistillationTrainerConfig(device="cpu")
        net = self._make_simple_network()
        trainer = DistillationTrainer(value_network=net, config=cfg)
        optimizer = trainer._create_optimizer()
        assert optimizer is not None

    def test_create_scheduler(self):
        cfg = DistillationTrainerConfig(device="cpu")
        net = self._make_simple_network()
        trainer = DistillationTrainer(policy_network=net, config=cfg)
        trainer._optimizer = trainer._create_optimizer()
        scheduler = trainer._create_scheduler(num_training_steps=100)
        assert scheduler is not None

    def test_create_scheduler_warmup_exceeds_total(self):
        cfg = DistillationTrainerConfig(device="cpu", warmup_steps=200)
        net = self._make_simple_network()
        trainer = DistillationTrainer(policy_network=net, config=cfg)
        trainer._optimizer = trainer._create_optimizer()
        # warmup_steps (200) > total (50): pct_start clamped to 1.0
        scheduler = trainer._create_scheduler(num_training_steps=50)
        assert scheduler is not None

    def test_get_all_parameters(self):
        cfg = DistillationTrainerConfig(device="cpu")
        net1 = self._make_simple_network()
        net2 = self._make_simple_network()
        trainer = DistillationTrainer(
            policy_network=net1, value_network=net2, config=cfg
        )
        params = trainer._get_all_parameters()
        assert len(params) == 4  # 2 params per Linear (weight + bias) x 2

    def test_get_all_parameters_none_networks(self):
        cfg = DistillationTrainerConfig(device="cpu")
        trainer = DistillationTrainer(config=cfg)
        params = trainer._get_all_parameters()
        assert params == []

    def test_callbacks_default(self):
        trainer = DistillationTrainer(config=DistillationTrainerConfig(device="cpu"))
        assert len(trainer._callbacks) == 1
        assert isinstance(trainer._callbacks[0], LoggingCallback)

    def test_callbacks_custom(self):
        cb = MagicMock(spec=TrainingCallback)
        trainer = DistillationTrainer(
            config=DistillationTrainerConfig(device="cpu"), callbacks=[cb]
        )
        assert len(trainer._callbacks) == 1
        assert trainer._callbacks[0] is cb

    def test_save_checkpoint(self, tmp_path):
        cfg = DistillationTrainerConfig(device="cpu", checkpoint_dir=str(tmp_path))
        net = self._make_simple_network()
        trainer = DistillationTrainer(policy_network=net, config=cfg)
        trainer._save_checkpoint(epoch=0)
        assert (tmp_path / "checkpoint_epoch_1.pt").exists()

    def test_save_checkpoint_final(self, tmp_path):
        cfg = DistillationTrainerConfig(device="cpu", checkpoint_dir=str(tmp_path))
        net = self._make_simple_network()
        trainer = DistillationTrainer(policy_network=net, config=cfg)
        trainer._save_checkpoint(epoch=0, is_final=True)
        assert (tmp_path / "checkpoint_final.pt").exists()

    def test_cleanup_old_checkpoints(self, tmp_path):
        cfg = DistillationTrainerConfig(
            device="cpu", checkpoint_dir=str(tmp_path), keep_last_n_checkpoints=2
        )
        net = self._make_simple_network()
        trainer = DistillationTrainer(policy_network=net, config=cfg)
        # Create several checkpoints
        for i in range(5):
            trainer._save_checkpoint(epoch=i)
        remaining = list(tmp_path.glob("checkpoint_epoch_*.pt"))
        assert len(remaining) == 2

    def test_load_checkpoint(self, tmp_path):
        cfg = DistillationTrainerConfig(device="cpu", checkpoint_dir=str(tmp_path))
        net = self._make_simple_network()
        trainer = DistillationTrainer(policy_network=net, config=cfg)
        trainer._global_step = 100
        trainer._best_metric = 0.5
        trainer._metrics_history = [{"loss": 0.3}]
        trainer._save_checkpoint(epoch=2)

        # Create a new trainer and load
        net2 = self._make_simple_network()
        trainer2 = DistillationTrainer(policy_network=net2, config=cfg)
        trainer2._load_checkpoint(tmp_path / "checkpoint_epoch_3.pt")
        assert trainer2._current_epoch == 3  # epoch + 1
        assert trainer2._global_step == 100
        assert trainer2._best_metric == 0.5

    def test_amp_disabled_on_cpu(self):
        cfg = DistillationTrainerConfig(device="cpu", use_amp=True)
        trainer = DistillationTrainer(config=cfg)
        assert trainer._use_amp is False
        assert trainer._scaler is None

    def test_kl_loss(self):
        cfg = DistillationTrainerConfig(device="cpu")
        trainer = DistillationTrainer(config=cfg)
        log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]))
        target_probs = torch.tensor([[0.6, 0.3, 0.1], [0.3, 0.5, 0.2]])
        action_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        loss = trainer._kl_loss(log_probs, target_probs, action_mask)
        assert loss.item() > 0
        assert isinstance(loss, torch.Tensor)


# ---------------------------------------------------------------------------
# create_trainer factory tests
# ---------------------------------------------------------------------------


class TestCreateTrainer:
    def test_create_with_defaults(self):
        trainer = create_trainer(device="cpu")
        assert isinstance(trainer, DistillationTrainer)
        assert trainer._config.learning_rate == 1e-4
        assert trainer._config.num_epochs == 10

    def test_create_with_custom_params(self):
        trainer = create_trainer(
            learning_rate=0.01,
            num_epochs=5,
            checkpoint_dir="/tmp/test",
            use_mcts_policy=False,
            use_outcome_value=False,
            device="cpu",
        )
        assert trainer._config.learning_rate == 0.01
        assert trainer._config.num_epochs == 5
        assert trainer._config.use_mcts_policy is False
        assert trainer._config.use_outcome_value is False
