"""
Extended unit tests for src/training/unified_orchestrator.py.

Covers previously uncovered paths including:
- GPU memory utilization branch
- Policy-Value network training when buffer is ready (full training loop)
- Exception loading best model during evaluation
- Exception saving best model checkpoint
- Exception logging wandb metrics
- KeyboardInterrupt during training
- KeyError and generic Exception in load_checkpoint
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from src.training.system_config import SystemConfig

_MODULE = "src.training.unified_orchestrator"


def _make_config(tmp_path=None):
    """Create a minimal SystemConfig for testing."""
    config = SystemConfig(device="cpu", use_wandb=False, use_mixed_precision=False)
    config.training.games_per_iteration = 2
    config.training.batch_size = 4
    config.training.buffer_size = 100
    config.training.checkpoint_interval = 1
    config.training.evaluation_games = 2
    config.training.patience = 3
    config.training.min_delta = 0.01
    config.training.hrm_train_batches = 2
    config.training.trm_train_batches = 2
    config.training.gradient_clip_norm = 1.0
    config.training.eval_temperature = 0.1
    config.training.win_threshold = 0.55
    config.log_interval = 1
    if tmp_path:
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")
    return config


def _make_orchestrator(tmp_path=None):
    """Create an orchestrator with all heavy components mocked."""
    from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

    dummy_param = torch.nn.Parameter(torch.randn(2, 2))

    mock_pv_net = MagicMock()
    mock_pv_net.get_parameter_count.return_value = 100
    mock_pv_net.parameters.return_value = iter([dummy_param])
    mock_pv_net.state_dict.return_value = {"w": torch.randn(2, 2)}

    mock_hrm = MagicMock()
    mock_hrm.get_parameter_count.return_value = 50
    mock_hrm.parameters.return_value = iter([dummy_param])
    mock_hrm.state_dict.return_value = {"w": torch.randn(2, 2)}

    mock_trm = MagicMock()
    mock_trm.get_parameter_count.return_value = 50
    mock_trm.parameters.return_value = iter([dummy_param])
    mock_trm.state_dict.return_value = {"w": torch.randn(2, 2)}

    config = _make_config(tmp_path)

    with (
        patch(f"{_MODULE}.create_policy_value_network", return_value=mock_pv_net),
        patch(f"{_MODULE}.create_hrm_agent", return_value=mock_hrm),
        patch(f"{_MODULE}.create_trm_agent", return_value=mock_trm),
        patch(f"{_MODULE}.NeuralMCTS"),
        patch(f"{_MODULE}.SelfPlayCollector"),
        patch(f"{_MODULE}.PrioritizedReplayBuffer") as mock_buf_cls,
        patch(f"{_MODULE}.PerformanceMonitor"),
    ):
        mock_buf = MagicMock()
        mock_buf.__len__ = MagicMock(return_value=100)
        mock_buf.is_ready.return_value = True
        mock_buf_cls.return_value = mock_buf

        orch = UnifiedTrainingOrchestrator(
            config=config,
            initial_state_fn=lambda: MagicMock(),
        )
        orch.replay_buffer = mock_buf
    return orch


# ---------------------------------------------------------------------------
# GPU memory utilization branch (lines 482-492)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGPUMemoryUtilization:
    """Tests for GPU memory reporting in _get_memory_utilization."""

    def test_gpu_memory_reported_when_cuda_available(self):
        """Test GPU metrics are included when device is cuda and cuda is available."""
        orch = _make_orchestrator()
        orch.device = "cuda"

        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8 GB

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 1024),
            patch("torch.cuda.memory_reserved", return_value=2 * 1024 * 1024 * 1024),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            memory = orch._get_memory_utilization()

        assert "gpu_memory_allocated_mb" in memory
        assert "gpu_memory_reserved_mb" in memory
        assert "gpu_memory_total_mb" in memory
        assert "gpu_utilization_percent" in memory
        assert memory["gpu_memory_allocated_mb"] == pytest.approx(1024.0, rel=0.01)
        assert memory["gpu_memory_total_mb"] == pytest.approx(8192.0, rel=0.01)

    def test_gpu_memory_exception_handled(self):
        """Test GPU memory exception is caught gracefully (line 491-492)."""
        orch = _make_orchestrator()
        orch.device = "cuda"

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", side_effect=RuntimeError("CUDA error")),
        ):
            memory = orch._get_memory_utilization()

        # Should still have CPU metrics, no GPU metrics
        assert "cpu_memory_mb" in memory
        assert "gpu_memory_allocated_mb" not in memory


# ---------------------------------------------------------------------------
# Policy-Value network training with buffer ready (lines 584-713)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainPolicyValueNetworkReady:
    """Tests for _train_policy_value_network when buffer is ready."""

    @pytest.mark.asyncio
    async def test_pv_training_loop_non_mixed_precision(self):
        """Test full PV training loop without mixed precision (lines 634-646)."""
        orch = _make_orchestrator()
        orch.replay_buffer.is_ready.return_value = True

        batch_size = orch.config.training.batch_size
        states = torch.randn(batch_size, 3, 9, 9)
        policies = torch.randn(batch_size, 81)
        values = torch.randn(batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        indices = list(range(batch_size))

        orch.replay_buffer.sample.return_value = (
            [MagicMock() for _ in range(batch_size)],
            indices,
            weights,
        )

        with patch(f"{_MODULE}.collate_experiences", return_value=(states, policies, values)):
            policy_logits = torch.randn(batch_size, 81)
            value_pred = torch.randn(batch_size, 1)
            orch.policy_value_net.return_value = (policy_logits, value_pred)
            orch.policy_value_net.train = MagicMock()

            real_param = torch.nn.Parameter(torch.randn(2, 2))
            loss_dict = {"policy": 0.3, "value": 0.2, "total": 0.5}

            mock_loss_fn = MagicMock()

            def make_loss(*args, **kwargs):
                per_element = real_param.sum().expand(batch_size) * 0 + 0.5
                return per_element, loss_dict

            mock_loss_fn.side_effect = make_loss
            orch.pv_loss_fn = mock_loss_fn

            orch.pv_optimizer = torch.optim.SGD([real_param], lr=0.01)
            orch.pv_scheduler = None

            orch._compute_gradient_norm = MagicMock(return_value=0.5)

            result = await orch._train_policy_value_network()

        assert "policy_loss" in result
        assert "value_loss" in result
        assert result["policy_loss"] == pytest.approx(0.3, rel=0.01)
        assert result["value_loss"] == pytest.approx(0.2, rel=0.01)
        assert orch.monitor.log_loss.call_count == 10  # 10 batches

    @pytest.mark.asyncio
    async def test_pv_training_loop_mixed_precision(self):
        """Test PV training with mixed precision path (lines 616-633)."""
        orch = _make_orchestrator()
        orch.config.use_mixed_precision = True
        orch.replay_buffer.is_ready.return_value = True

        batch_size = orch.config.training.batch_size
        states = torch.randn(batch_size, 3, 9, 9)
        policies = torch.randn(batch_size, 81)
        values = torch.randn(batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        indices = list(range(batch_size))

        orch.replay_buffer.sample.return_value = (
            [MagicMock() for _ in range(batch_size)],
            indices,
            weights,
        )

        with patch(f"{_MODULE}.collate_experiences", return_value=(states, policies, values)):
            policy_logits = torch.randn(batch_size, 81)
            value_pred = torch.randn(batch_size, 1)
            orch.policy_value_net.return_value = (policy_logits, value_pred)
            orch.policy_value_net.train = MagicMock()

            real_param = torch.nn.Parameter(torch.randn(2, 2))
            loss_dict = {"policy": 0.3, "value": 0.2, "total": 0.5}

            mock_loss_fn = MagicMock()

            def make_loss(*args, **kwargs):
                per_element = real_param.sum().expand(batch_size) * 0 + 0.5
                return per_element, loss_dict

            mock_loss_fn.side_effect = make_loss
            orch.pv_loss_fn = mock_loss_fn

            # Mock scaler for mixed precision
            mock_scaler = MagicMock()
            scaled_loss = MagicMock()
            mock_scaler.scale.return_value = scaled_loss
            orch.scaler = mock_scaler

            orch.pv_optimizer = torch.optim.SGD([real_param], lr=0.01)
            orch.pv_scheduler = None

            orch._compute_gradient_norm = MagicMock(return_value=0.5)

            with patch(f"{_MODULE}.autocast"):
                result = await orch._train_policy_value_network()

        assert result["policy_loss"] == pytest.approx(0.3, rel=0.01)
        assert mock_scaler.scale.call_count == 10
        assert mock_scaler.step.call_count == 10
        assert mock_scaler.update.call_count == 10

    @pytest.mark.asyncio
    async def test_pv_training_with_scheduler_step(self):
        """Test LR scheduler is stepped after training (lines 680-691)."""
        orch = _make_orchestrator()
        orch.replay_buffer.is_ready.return_value = True

        batch_size = orch.config.training.batch_size
        states = torch.randn(batch_size, 3, 9, 9)
        policies = torch.randn(batch_size, 81)
        values = torch.randn(batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        indices = list(range(batch_size))

        orch.replay_buffer.sample.return_value = (
            [MagicMock() for _ in range(batch_size)],
            indices,
            weights,
        )

        with patch(f"{_MODULE}.collate_experiences", return_value=(states, policies, values)):
            policy_logits = torch.randn(batch_size, 81)
            value_pred = torch.randn(batch_size, 1)
            orch.policy_value_net.return_value = (policy_logits, value_pred)
            orch.policy_value_net.train = MagicMock()

            real_param = torch.nn.Parameter(torch.randn(2, 2))
            loss_dict = {"policy": 0.3, "value": 0.2, "total": 0.5}

            mock_loss_fn = MagicMock()

            def make_loss(*args, **kwargs):
                per_element = real_param.sum().expand(batch_size) * 0 + 0.5
                return per_element, loss_dict

            mock_loss_fn.side_effect = make_loss
            orch.pv_loss_fn = mock_loss_fn

            orch.pv_optimizer = torch.optim.SGD([real_param], lr=0.01)

            mock_scheduler = MagicMock()
            orch.pv_scheduler = mock_scheduler

            orch._compute_gradient_norm = MagicMock(return_value=0.5)

            await orch._train_policy_value_network()

        mock_scheduler.step.assert_called_once()


# ---------------------------------------------------------------------------
# Evaluation: exception loading best model (lines 963-969)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvaluateBestModelLoadError:
    """Tests for _evaluate when loading best model fails."""

    @pytest.mark.asyncio
    async def test_evaluate_best_model_load_exception(self, tmp_path):
        """Test evaluation continues when best model load fails (lines 963-969)."""
        orch = _make_orchestrator()

        # Create a fake best model path that exists but causes load error
        best_path = tmp_path / "best_model.pt"
        best_path.write_bytes(b"invalid data")
        orch.best_model_path = best_path

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value={"win_rate": 0.6, "wins": 6, "losses": 3, "draws": 1}
        )

        with (
            patch(f"{_MODULE}.torch.load", side_effect=RuntimeError("corrupt checkpoint")),
            patch("src.training.agent_trainer.SelfPlayEvaluator", return_value=mock_evaluator),
            patch("src.training.agent_trainer.EvaluationConfig"),
        ):
            result = await orch._evaluate()

        # Should still return results (best_model=None fallback)
        assert result["win_rate"] == 0.6
        # Evaluator should be called with best_model=None
        call_kwargs = mock_evaluator.evaluate.call_args[1]
        assert call_kwargs["best_model"] is None


# ---------------------------------------------------------------------------
# Save checkpoint: exception saving best model (lines 1077-1078)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSaveCheckpointBestModelError:
    """Tests for _save_checkpoint when saving best model fails."""

    def test_save_best_model_exception(self, tmp_path):
        """Test exception saving best_model.pt is handled (lines 1077-1082)."""
        orch = _make_orchestrator()
        orch.checkpoint_dir = tmp_path

        call_count = 0
        original_save = torch.save

        def save_side_effect(obj, path, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (regular checkpoint) succeeds
                original_save(obj, path)
            else:
                # Second call (best model) fails
                raise OSError("disk full")

        with patch(f"{_MODULE}.torch.save", side_effect=save_side_effect):
            orch._save_checkpoint(1, {"policy_loss": 0.5}, is_best=True)

        # Regular checkpoint was saved, best model was not
        assert (tmp_path / "checkpoint_iter_1.pt").exists()
        assert not (tmp_path / "best_model.pt").exists()


# ---------------------------------------------------------------------------
# Log metrics: wandb exception (lines 1106-1107)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLogMetricsWandbError:
    """Tests for _log_metrics when wandb logging fails."""

    def test_wandb_log_exception_handled(self):
        """Test exception during wandb.log is caught (lines 1106-1111)."""
        orch = _make_orchestrator()
        orch.config.use_wandb = True

        mock_wandb = MagicMock()
        mock_wandb.log.side_effect = RuntimeError("wandb connection error")
        orch.monitor.export_to_wandb = MagicMock(return_value={"metric": 1.0})

        import sys

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            # Should not raise
            orch._log_metrics(1, {"policy_loss": 0.5})


# ---------------------------------------------------------------------------
# Train: KeyboardInterrupt (lines 1164-1171)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainKeyboardInterrupt:
    """Tests for train() when KeyboardInterrupt is raised."""

    @pytest.mark.asyncio
    async def test_train_keyboard_interrupt(self):
        """Test training handles KeyboardInterrupt gracefully (lines 1164-1171)."""
        orch = _make_orchestrator()
        call_count = 0

        async def mock_train_iteration(iteration):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise KeyboardInterrupt()
            return {"policy_loss": 0.5}

        orch.train_iteration = mock_train_iteration
        orch._should_early_stop = MagicMock(return_value=False)

        await orch.train(5)

        # Should have stopped after 2 iterations (second raised interrupt)
        assert call_count == 2
        assert orch.current_iteration == 2


# ---------------------------------------------------------------------------
# Load checkpoint: KeyError and generic Exception (lines 1312-1326)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadCheckpointErrors:
    """Tests for load_checkpoint error handling."""

    def test_load_checkpoint_missing_key(self, tmp_path):
        """Test load_checkpoint raises on missing key (lines 1312-1319)."""
        orch = _make_orchestrator()

        # Save a checkpoint with missing keys
        incomplete_checkpoint = {
            "iteration": 5,
            # Missing policy_value_net, hrm_agent, etc.
        }
        ckpt_path = tmp_path / "incomplete.pt"
        torch.save(incomplete_checkpoint, ckpt_path)

        with pytest.raises(KeyError):
            orch.load_checkpoint(str(ckpt_path))

    def test_load_checkpoint_state_dict_error(self, tmp_path):
        """Test load_checkpoint raises on state dict restore failure (lines 1320-1326)."""
        orch = _make_orchestrator()

        # Save a valid-looking checkpoint but with incompatible state dicts
        checkpoint = {
            "iteration": 5,
            "policy_value_net": {"bad_key": torch.randn(2)},
            "hrm_agent": {"bad_key": torch.randn(2)},
            "trm_agent": {"bad_key": torch.randn(2)},
            "pv_optimizer": {},
            "hrm_optimizer": {},
            "trm_optimizer": {},
            "best_win_rate": 0.5,
        }
        ckpt_path = tmp_path / "bad_state.pt"
        torch.save(checkpoint, ckpt_path)

        # Make load_state_dict raise a RuntimeError (not KeyError)
        orch.policy_value_net.load_state_dict.side_effect = RuntimeError(
            "Error(s) in loading state_dict"
        )

        with pytest.raises(RuntimeError, match="Error.*state_dict"):
            orch.load_checkpoint(str(ckpt_path))

    def test_load_checkpoint_file_not_loadable(self, tmp_path):
        """Test load_checkpoint raises when torch.load fails."""
        orch = _make_orchestrator()

        bad_path = tmp_path / "corrupt.pt"
        bad_path.write_bytes(b"not a valid checkpoint")

        with pytest.raises((RuntimeError, OSError, ValueError, Exception)):  # noqa: B017
            orch.load_checkpoint(str(bad_path))


# ---------------------------------------------------------------------------
# Train: early stopping path in train() (lines 1154-1162)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainEarlyStopping:
    """Tests for early stopping within the train() loop."""

    @pytest.mark.asyncio
    async def test_train_early_stop_logs_and_breaks(self):
        """Test that early stopping triggers break in train loop (lines 1154-1162)."""
        orch = _make_orchestrator()
        orch.train_iteration = AsyncMock(return_value={"policy_loss": 0.5})

        # Early stop after iteration 2
        orch._should_early_stop = MagicMock(side_effect=[False, True])

        await orch.train(10)

        assert orch.train_iteration.call_count == 2
        assert orch.current_iteration == 2


# ---------------------------------------------------------------------------
# Train: exception in train_iteration within train() (lines 1172-1179)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainIterationException:
    """Tests for train() when train_iteration raises an unexpected exception."""

    @pytest.mark.asyncio
    async def test_train_exception_breaks_loop(self):
        """Test generic exception in train_iteration breaks loop (lines 1172-1179)."""
        orch = _make_orchestrator()

        call_count = 0

        async def failing_iteration(iteration):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise ValueError("Unexpected training error")
            return {"policy_loss": 0.5}

        orch.train_iteration = failing_iteration
        orch._should_early_stop = MagicMock(return_value=False)

        await orch.train(10)

        assert call_count == 3
        assert orch.current_iteration == 3


# ---------------------------------------------------------------------------
# _evaluate with no best model path (line 970-974)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvaluateNoBestModel:
    """Tests for _evaluate when best_model_path is None."""

    @pytest.mark.asyncio
    async def test_evaluate_no_best_model_path(self):
        """Test evaluation when no best model exists (line 970-974 debug log)."""
        orch = _make_orchestrator()
        orch.best_model_path = None

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value={"win_rate": 0.55, "wins": 5, "losses": 4, "draws": 1}
        )

        with (
            patch("src.training.agent_trainer.SelfPlayEvaluator", return_value=mock_evaluator),
            patch("src.training.agent_trainer.EvaluationConfig"),
        ):
            result = await orch._evaluate()

        assert result["win_rate"] == 0.55
        call_kwargs = mock_evaluator.evaluate.call_args[1]
        assert call_kwargs["best_model"] is None

    @pytest.mark.asyncio
    async def test_evaluate_best_model_path_does_not_exist(self):
        """Test evaluation when best_model_path is set but file does not exist."""
        orch = _make_orchestrator()
        orch.best_model_path = Path("/nonexistent/best_model.pt")

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value={"win_rate": 0.5, "wins": 5, "losses": 5, "draws": 0}
        )

        with (
            patch("src.training.agent_trainer.SelfPlayEvaluator", return_value=mock_evaluator),
            patch("src.training.agent_trainer.EvaluationConfig"),
        ):
            result = await orch._evaluate()

        assert result["win_rate"] == 0.5
        call_kwargs = mock_evaluator.evaluate.call_args[1]
        assert call_kwargs["best_model"] is None


# ---------------------------------------------------------------------------
# _save_checkpoint regular failure (line 1056-1063)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSaveCheckpointRegularFailure:
    """Test _save_checkpoint when the regular checkpoint save fails."""

    def test_regular_checkpoint_save_failure(self):
        """Test exception on regular checkpoint save returns early (lines 1056-1063)."""
        orch = _make_orchestrator()
        orch.checkpoint_dir = Path("/nonexistent/path/that/does/not/exist")

        # Should not raise, just log error
        orch._save_checkpoint(1, {"policy_loss": 0.5}, is_best=True)
        # best_model_path should NOT be set since regular save failed first
        # (the method returns early)
