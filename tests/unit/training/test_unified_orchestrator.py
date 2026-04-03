"""
Tests for unified_orchestrator module.

Tests UnifiedTrainingOrchestrator initialization, configuration,
path setup, memory utilities, checkpointing, early stopping,
and training iteration logic.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.training.system_config import (
    SystemConfig,
)

# ---------------------------------------------------------------------------
# Fixtures: mock all heavy components so tests are fast and isolated
# ---------------------------------------------------------------------------

def _make_mock_model(param_count: int = 100):
    """Create a mock nn.Module with parameter count."""
    model = MagicMock()
    # Make parameters() return actual tensors so sum(p.numel()) works
    param = torch.randn(param_count)
    param.requires_grad = True
    model.parameters.return_value = [param]
    model.get_parameter_count.return_value = param_count
    model.state_dict.return_value = {"weight": torch.randn(4, 4)}
    model.load_state_dict = MagicMock()
    model.train = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


def _make_test_config(tmp_path: Path) -> SystemConfig:
    """Create a minimal SystemConfig pointing to tmp directories."""
    config = SystemConfig(
        device="cpu",
        seed=42,
        use_mixed_precision=False,
        use_wandb=False,
        log_interval=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        data_dir=str(tmp_path / "data"),
        log_dir=str(tmp_path / "logs"),
    )
    config.training.games_per_iteration = 2
    config.training.batch_size = 4
    config.training.buffer_size = 100
    config.training.checkpoint_interval = 1
    config.training.evaluation_games = 2
    config.training.patience = 3
    config.training.min_delta = 0.01
    config.training.hrm_train_batches = 2
    config.training.trm_train_batches = 2
    config.mcts.num_simulations = 4
    return config


@pytest.fixture
def tmp_config(tmp_path):
    """Create a test config with tmp paths."""
    return _make_test_config(tmp_path)


# ---------------------------------------------------------------------------
# Helper to build an orchestrator with mocked components
# ---------------------------------------------------------------------------

_MODULE = "src.training.unified_orchestrator"


def _build_orchestrator_with_mocks(config):
    """Build an orchestrator with all heavy components mocked out."""
    mock_pv_net = _make_mock_model(200)
    mock_hrm = _make_mock_model(100)
    mock_trm = _make_mock_model(100)
    mock_mcts = MagicMock()
    mock_self_play = MagicMock()
    mock_replay_buffer = MagicMock()
    mock_replay_buffer.is_ready.return_value = False
    mock_replay_buffer.__len__ = MagicMock(return_value=0)

    initial_state_fn = MagicMock(return_value=MagicMock())

    with (
        patch(f"{_MODULE}.create_policy_value_network", return_value=mock_pv_net),
        patch(f"{_MODULE}.create_hrm_agent", return_value=mock_hrm),
        patch(f"{_MODULE}.create_trm_agent", return_value=mock_trm),
        patch(f"{_MODULE}.NeuralMCTS", return_value=mock_mcts),
        patch(f"{_MODULE}.SelfPlayCollector", return_value=mock_self_play),
        patch(f"{_MODULE}.AlphaZeroLoss", return_value=MagicMock()),
        patch(f"{_MODULE}.HRMLoss", return_value=MagicMock()),
        patch(f"{_MODULE}.TRMLoss", return_value=MagicMock()),
        patch(f"{_MODULE}.PrioritizedReplayBuffer", return_value=mock_replay_buffer),
        patch(f"{_MODULE}.GradScaler", return_value=MagicMock()),
    ):
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(
            config=config,
            initial_state_fn=initial_state_fn,
            board_size=9,
        )

    # Attach mocks for inspection
    orch._mock_pv_net = mock_pv_net
    orch._mock_hrm = mock_hrm
    orch._mock_trm = mock_trm
    orch._mock_mcts = mock_mcts
    orch._mock_self_play = mock_self_play
    return orch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUnifiedTrainingOrchestratorInit:
    """Tests for UnifiedTrainingOrchestrator initialization."""

    def test_init_sets_config(self, tmp_config):
        """Test orchestrator stores config."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.config is tmp_config

    def test_init_sets_device(self, tmp_config):
        """Test device is set from config."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.device == "cpu"

    def test_init_sets_board_size(self, tmp_config):
        """Test board_size is stored."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.board_size == 9

    def test_init_training_state(self, tmp_config):
        """Test initial training state."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.current_iteration == 0
        assert orch.best_win_rate == 0.0
        assert orch.best_model_path is None

    def test_init_creates_directories(self, tmp_config):
        """Test directories are created."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.checkpoint_dir.exists()
        assert orch.data_dir.exists()
        assert orch.log_dir.exists()

    def test_init_monitor_created(self, tmp_config):
        """Test performance monitor is created."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.monitor is not None


@pytest.mark.unit
class TestSetupPaths:
    """Tests for _setup_paths method."""

    def test_paths_created(self, tmp_config):
        """Test all directories are created."""
        _build_orchestrator_with_mocks(tmp_config)
        assert Path(tmp_config.checkpoint_dir).exists()
        assert Path(tmp_config.data_dir).exists()
        assert Path(tmp_config.log_dir).exists()


@pytest.mark.unit
class TestSetupWandb:
    """Tests for _setup_wandb method."""

    def test_wandb_disabled_on_import_error(self, tmp_config):
        """Test wandb is disabled when import fails."""
        tmp_config.use_wandb = True
        orch = _build_orchestrator_with_mocks(tmp_config)

        import sys
        # Set wandb to None in sys.modules so import raises ImportError
        with patch.dict(sys.modules, {"wandb": None}):
            orch._setup_wandb()
        assert orch.config.use_wandb is False

    def test_wandb_disabled_on_generic_error(self, tmp_config):
        """Test wandb is disabled on generic initialization error."""
        tmp_config.use_wandb = True
        orch = _build_orchestrator_with_mocks(tmp_config)

        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = RuntimeError("wandb error")

        import sys
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            orch._setup_wandb()
        assert orch.config.use_wandb is False


@pytest.mark.unit
class TestGetMemoryUtilization:
    """Tests for _get_memory_utilization method."""

    def test_cpu_memory_reported(self, tmp_config):
        """Test CPU memory is always reported."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        memory = orch._get_memory_utilization()
        assert "cpu_memory_mb" in memory
        assert memory["cpu_memory_mb"] > 0

    def test_cpu_percent_reported(self, tmp_config):
        """Test CPU percentage is reported."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        memory = orch._get_memory_utilization()
        assert "cpu_percent" in memory

    def test_no_gpu_on_cpu_device(self, tmp_config):
        """Test no GPU metrics on CPU device."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        orch.device = "cpu"
        memory = orch._get_memory_utilization()
        assert "gpu_memory_allocated_mb" not in memory


@pytest.mark.unit
class TestComputeGradientNorm:
    """Tests for _compute_gradient_norm method."""

    def test_gradient_norm_computation(self, tmp_config):
        """Test gradient norm is computed correctly."""
        orch = _build_orchestrator_with_mocks(tmp_config)

        # Create a simple model with known gradients
        model = nn.Linear(4, 2, bias=False)
        x = torch.randn(2, 4)
        y = model(x)
        y.sum().backward()

        norm = orch._compute_gradient_norm(model)
        assert isinstance(norm, float)
        assert norm > 0

    def test_gradient_norm_no_grad(self, tmp_config):
        """Test gradient norm is 0 when no gradients exist."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        model = nn.Linear(4, 2, bias=False)
        # No backward pass, so no gradients
        norm = orch._compute_gradient_norm(model)
        assert norm == 0.0


@pytest.mark.unit
class TestSaveCheckpoint:
    """Tests for _save_checkpoint method."""

    def test_checkpoint_saved(self, tmp_config):
        """Test checkpoint file is created."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        orch._save_checkpoint(iteration=1, metrics={"loss": 0.5})

        ckpt_path = orch.checkpoint_dir / "checkpoint_iter_1.pt"
        assert ckpt_path.exists()

    def test_best_checkpoint_saved(self, tmp_config):
        """Test best model checkpoint is saved when is_best=True."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        orch._save_checkpoint(iteration=1, metrics={"loss": 0.5}, is_best=True)

        best_path = orch.checkpoint_dir / "best_model.pt"
        assert best_path.exists()
        assert orch.best_model_path == best_path

    def test_non_best_no_best_file(self, tmp_config):
        """Test best model file is not created when is_best=False."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        orch._save_checkpoint(iteration=1, metrics={"loss": 0.5}, is_best=False)

        best_path = orch.checkpoint_dir / "best_model.pt"
        assert not best_path.exists()

    def test_checkpoint_contents(self, tmp_config):
        """Test checkpoint contains expected keys."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        orch._save_checkpoint(iteration=5, metrics={"policy_loss": 0.3})

        ckpt = torch.load(orch.checkpoint_dir / "checkpoint_iter_5.pt", weights_only=False)
        assert ckpt["iteration"] == 5
        assert "policy_value_net" in ckpt
        assert "hrm_agent" in ckpt
        assert "trm_agent" in ckpt
        assert "metrics" in ckpt
        assert ckpt["metrics"]["policy_loss"] == 0.3


@pytest.mark.unit
class TestEarlyStopping:
    """Tests for _should_early_stop method."""

    def test_no_stop_before_checkpoint_interval(self, tmp_config):
        """Test early stopping only checked at checkpoint intervals."""
        tmp_config.training.checkpoint_interval = 5
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch._should_early_stop(iteration=3) is False

    def test_no_stop_on_first_evaluation(self, tmp_config):
        """Test no early stop on first evaluation."""
        tmp_config.training.checkpoint_interval = 1
        orch = _build_orchestrator_with_mocks(tmp_config)
        orch.best_win_rate = 0.6
        assert orch._should_early_stop(iteration=1) is False

    def test_stop_after_patience_exhausted(self, tmp_config):
        """Test early stopping triggers after patience is exhausted."""
        tmp_config.training.checkpoint_interval = 1
        tmp_config.training.patience = 2
        tmp_config.training.min_delta = 0.01
        orch = _build_orchestrator_with_mocks(tmp_config)

        # First evaluation sets baseline
        orch.best_win_rate = 0.5
        assert orch._should_early_stop(iteration=1) is False

        # No improvement for patience iterations
        assert orch._should_early_stop(iteration=2) is False
        assert orch._should_early_stop(iteration=3) is True

    def test_no_stop_with_improvement(self, tmp_config):
        """Test no early stop when win rate improves."""
        tmp_config.training.checkpoint_interval = 1
        tmp_config.training.patience = 2
        tmp_config.training.min_delta = 0.01
        orch = _build_orchestrator_with_mocks(tmp_config)

        orch.best_win_rate = 0.5
        assert orch._should_early_stop(iteration=1) is False

        # Win rate improves
        orch.best_win_rate = 0.6
        assert orch._should_early_stop(iteration=2) is False


@pytest.mark.unit
class TestLogMetrics:
    """Tests for _log_metrics method."""

    def test_log_metrics_no_wandb(self, tmp_config):
        """Test logging metrics without wandb does not error."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        # Should not raise
        orch._log_metrics(iteration=1, metrics={"loss": 0.5, "win_rate": 0.6})

    def test_log_metrics_with_wandb(self, tmp_config):
        """Test logging metrics with wandb calls wandb.log."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        # Set use_wandb after construction (construction disables it)
        orch.config.use_wandb = True

        mock_wandb = MagicMock()
        import sys
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            orch._log_metrics(iteration=1, metrics={"loss": 0.5})
            mock_wandb.log.assert_called_once()


@pytest.mark.unit
class TestSetupOptimizers:
    """Tests for _setup_optimizers method."""

    def test_optimizers_created(self, tmp_config):
        """Test all optimizers are created."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.pv_optimizer is not None
        assert orch.hrm_optimizer is not None
        assert orch.trm_optimizer is not None

    def test_cosine_scheduler(self, tmp_config):
        """Test cosine scheduler is created by default."""
        tmp_config.training.lr_schedule = "cosine"
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.pv_scheduler is not None

    def test_step_scheduler(self, tmp_config):
        """Test step scheduler creation."""
        tmp_config.training.lr_schedule = "step"
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.pv_scheduler is not None

    def test_no_scheduler(self, tmp_config):
        """Test no scheduler when schedule is constant."""
        tmp_config.training.lr_schedule = "constant"
        orch = _build_orchestrator_with_mocks(tmp_config)
        assert orch.pv_scheduler is None


@pytest.mark.unit
class TestLoadCheckpoint:
    """Tests for load_checkpoint method."""

    def test_load_nonexistent_file(self, tmp_config):
        """Test loading nonexistent checkpoint raises error."""
        orch = _build_orchestrator_with_mocks(tmp_config)
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            orch.load_checkpoint("/nonexistent/checkpoint.pt")

    def test_load_checkpoint_restores_state(self, tmp_config):
        """Test loading a saved checkpoint restores iteration and win rate."""
        orch = _build_orchestrator_with_mocks(tmp_config)

        # Save a checkpoint first
        orch.best_win_rate = 0.75
        orch._save_checkpoint(iteration=10, metrics={"test": 1.0}, is_best=False)
        ckpt_path = str(orch.checkpoint_dir / "checkpoint_iter_10.pt")

        # Reset state
        orch.current_iteration = 0
        orch.best_win_rate = 0.0

        # Load checkpoint
        orch.load_checkpoint(ckpt_path)
        assert orch.current_iteration == 10
        assert orch.best_win_rate == pytest.approx(0.75)


@pytest.mark.unit
class TestSystemConfigIntegration:
    """Test SystemConfig interaction with orchestrator."""

    def test_config_to_dict(self, tmp_config):
        """Test config can be serialized to dict."""
        d = tmp_config.to_dict()
        assert "hrm" in d
        assert "trm" in d
        assert "mcts" in d
        assert "neural_net" in d
        assert "training" in d
        assert d["device"] == "cpu"
