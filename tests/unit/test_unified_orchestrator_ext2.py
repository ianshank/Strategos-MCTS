"""Extended tests for src/training/unified_orchestrator.py covering async methods."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from src.training.system_config import SystemConfig


def _make_config():
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
    return config


def _make_orchestrator():
    """Create an orchestrator with all components mocked."""
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

    config = _make_config()

    with (
        patch("src.training.unified_orchestrator.create_policy_value_network", return_value=mock_pv_net),
        patch("src.training.unified_orchestrator.create_hrm_agent", return_value=mock_hrm),
        patch("src.training.unified_orchestrator.create_trm_agent", return_value=mock_trm),
        patch("src.training.unified_orchestrator.NeuralMCTS"),
        patch("src.training.unified_orchestrator.SelfPlayCollector"),
        patch("src.training.unified_orchestrator.PrioritizedReplayBuffer") as mock_buf_cls,
        patch("src.training.unified_orchestrator.PerformanceMonitor"),
    ):
        mock_buf = MagicMock()
        mock_buf.__len__ = MagicMock(return_value=100)
        mock_buf.is_ready.return_value = True
        mock_buf_cls.return_value = mock_buf

        orch = UnifiedTrainingOrchestrator(
            config=config,
            initial_state_fn=lambda: MagicMock(),
        )
        # Ensure replay_buffer is our mock
        orch.replay_buffer = mock_buf
    return orch


@pytest.mark.unit
class TestGenerateSelfPlayData:
    @pytest.mark.asyncio
    async def test_generates_examples(self):
        orch = _make_orchestrator()
        mock_example = MagicMock()
        mock_example.state = torch.randn(4)
        mock_example.policy_target = torch.randn(4)
        mock_example.value_target = 0.5
        orch.self_play_collector.play_game = AsyncMock(return_value=[mock_example])

        examples = await orch._generate_self_play_data()
        assert len(examples) == 2  # 2 games * 1 example each
        orch.replay_buffer.add_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_game_failure(self):
        orch = _make_orchestrator()
        orch.self_play_collector.play_game = AsyncMock(side_effect=RuntimeError("game error"))

        examples = await orch._generate_self_play_data()
        assert len(examples) == 0  # All games failed gracefully


@pytest.mark.unit
class TestTrainPolicyValueNetwork:
    @pytest.mark.asyncio
    async def test_returns_zeros_when_buffer_not_ready(self):
        orch = _make_orchestrator()
        orch.replay_buffer.is_ready.return_value = False

        result = await orch._train_policy_value_network()
        assert result["policy_loss"] == 0.0
        assert result["value_loss"] == 0.0

    @pytest.mark.asyncio
    async def test_returns_metrics_structure(self):
        """Verify _train_policy_value_network returns expected metric keys when buffer not ready."""
        orch = _make_orchestrator()
        orch.replay_buffer.is_ready.return_value = False
        result = await orch._train_policy_value_network()
        assert isinstance(result, dict)
        assert "policy_loss" in result
        assert "value_loss" in result


@pytest.mark.unit
class TestTrainHRMAgent:
    @pytest.mark.asyncio
    async def test_train_hrm_success(self):
        orch = _make_orchestrator()

        mock_trainer = MagicMock()
        mock_trainer.train_epoch = AsyncMock(
            return_value={"loss": 0.1, "hrm_halt_step": 3.0, "hrm_ponder_cost": 0.01, "gradient_norm": 0.5}
        )

        with (
            patch("src.training.agent_trainer.HRMTrainer", return_value=mock_trainer),
            patch("src.training.agent_trainer.create_data_loader_from_buffer", return_value=MagicMock()),
        ):
            result = await orch._train_hrm_agent()
            assert result["hrm_loss"] == 0.1
            assert result["hrm_halt_step"] == 3.0

    @pytest.mark.asyncio
    async def test_train_hrm_data_loader_failure(self):
        orch = _make_orchestrator()

        with patch("src.training.agent_trainer.create_data_loader_from_buffer", side_effect=RuntimeError("data error")):
            result = await orch._train_hrm_agent()
            assert result["hrm_loss"] == 0.0

    @pytest.mark.asyncio
    async def test_train_hrm_epoch_failure(self):
        orch = _make_orchestrator()
        mock_trainer = MagicMock()
        mock_trainer.train_epoch = AsyncMock(side_effect=RuntimeError("train error"))

        with (
            patch("src.training.agent_trainer.HRMTrainer", return_value=mock_trainer),
            patch("src.training.agent_trainer.create_data_loader_from_buffer", return_value=MagicMock()),
        ):
            result = await orch._train_hrm_agent()
            assert result["hrm_loss"] == 0.0


@pytest.mark.unit
class TestTrainTRMAgent:
    @pytest.mark.asyncio
    async def test_train_trm_success(self):
        orch = _make_orchestrator()

        mock_trainer = MagicMock()
        mock_trainer.train_epoch = AsyncMock(
            return_value={"loss": 0.2, "trm_convergence_step": 5.0, "trm_final_residual": 0.01, "gradient_norm": 0.3}
        )

        with (
            patch("src.training.agent_trainer.TRMTrainer", return_value=mock_trainer),
            patch("src.training.agent_trainer.create_data_loader_from_buffer", return_value=MagicMock()),
        ):
            result = await orch._train_trm_agent()
            assert result["trm_loss"] == 0.2
            assert result["trm_convergence_step"] == 5.0

    @pytest.mark.asyncio
    async def test_train_trm_data_loader_failure(self):
        orch = _make_orchestrator()

        with patch("src.training.agent_trainer.create_data_loader_from_buffer", side_effect=RuntimeError("error")):
            result = await orch._train_trm_agent()
            assert result["trm_loss"] == 0.0


@pytest.mark.unit
class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        orch = _make_orchestrator()
        orch.best_model_path = None

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"win_rate": 0.6, "wins": 6, "losses": 3, "draws": 1})

        with (
            patch("src.training.agent_trainer.SelfPlayEvaluator", return_value=mock_evaluator),
            patch("src.training.agent_trainer.EvaluationConfig"),
        ):
            result = await orch._evaluate()
            assert result["win_rate"] == 0.6
            assert result["wins"] == 6

    @pytest.mark.asyncio
    async def test_evaluate_with_best_model(self, tmp_path):
        orch = _make_orchestrator()
        # Save a fake checkpoint
        best_path = tmp_path / "best_model.pt"
        torch.save({"policy_value_net": orch.policy_value_net.state_dict()}, best_path)
        orch.best_model_path = best_path

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"win_rate": 0.55, "wins": 5, "losses": 4, "draws": 1})

        with (
            patch("src.training.agent_trainer.SelfPlayEvaluator", return_value=mock_evaluator),
            patch("src.training.agent_trainer.EvaluationConfig"),
        ):
            result = await orch._evaluate()
            assert result["win_rate"] == 0.55

    @pytest.mark.asyncio
    async def test_evaluate_failure(self):
        orch = _make_orchestrator()
        orch.best_model_path = None

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(side_effect=RuntimeError("eval error"))

        with (
            patch("src.training.agent_trainer.SelfPlayEvaluator", return_value=mock_evaluator),
            patch("src.training.agent_trainer.EvaluationConfig"),
        ):
            result = await orch._evaluate()
            assert result["win_rate"] == 0.0


@pytest.mark.unit
class TestSaveCheckpoint:
    def test_save_checkpoint_regular(self, tmp_path):
        orch = _make_orchestrator()
        orch.checkpoint_dir = tmp_path

        orch._save_checkpoint(1, {"policy_loss": 0.5}, is_best=False)
        assert (tmp_path / "checkpoint_iter_1.pt").exists()
        assert not (tmp_path / "best_model.pt").exists()

    def test_save_checkpoint_best(self, tmp_path):
        orch = _make_orchestrator()
        orch.checkpoint_dir = tmp_path

        orch._save_checkpoint(1, {"policy_loss": 0.5}, is_best=True)
        assert (tmp_path / "checkpoint_iter_1.pt").exists()
        assert (tmp_path / "best_model.pt").exists()
        assert orch.best_model_path == tmp_path / "best_model.pt"

    def test_save_checkpoint_error(self, tmp_path):
        orch = _make_orchestrator()
        orch.checkpoint_dir = Path("/nonexistent/path")

        # Should not raise, just log error
        orch._save_checkpoint(1, {"policy_loss": 0.5}, is_best=False)


@pytest.mark.unit
class TestLogMetrics:
    def test_log_metrics_without_wandb(self):
        orch = _make_orchestrator()
        orch.config.use_wandb = False
        # Should not raise
        orch._log_metrics(1, {"policy_loss": 0.5, "value_loss": 0.3})

    def test_log_metrics_with_wandb(self):
        orch = _make_orchestrator()
        orch.config.use_wandb = True
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            orch._log_metrics(1, {"policy_loss": 0.5})


@pytest.mark.unit
class TestTrain:
    @pytest.mark.asyncio
    async def test_train_completes_iterations(self):
        orch = _make_orchestrator()
        orch.train_iteration = AsyncMock(return_value={"policy_loss": 0.5})
        orch._should_early_stop = MagicMock(return_value=False)

        await orch.train(2)
        assert orch.train_iteration.call_count == 2
        assert orch.current_iteration == 2

    @pytest.mark.asyncio
    async def test_train_early_stops(self):
        orch = _make_orchestrator()
        orch.train_iteration = AsyncMock(return_value={"policy_loss": 0.5})
        orch._should_early_stop = MagicMock(side_effect=[False, True])

        await orch.train(5)
        assert orch.train_iteration.call_count == 2

    @pytest.mark.asyncio
    async def test_train_handles_error(self):
        orch = _make_orchestrator()
        orch.train_iteration = AsyncMock(side_effect=RuntimeError("train error"))

        await orch.train(3)
        assert orch.train_iteration.call_count == 1


@pytest.mark.unit
class TestExecuteIteration:
    @pytest.mark.asyncio
    async def test_train_iteration_basic(self):
        orch = _make_orchestrator()
        orch._generate_self_play_data = AsyncMock(return_value=[MagicMock()] * 5)
        orch._train_policy_value_network = AsyncMock(return_value={"policy_loss": 0.3, "value_loss": 0.2})
        orch._evaluate = AsyncMock(return_value={"win_rate": 0.6, "wins": 6, "losses": 3, "draws": 1})
        orch._save_checkpoint = MagicMock()
        orch._log_metrics = MagicMock()
        orch.monitor.alert_if_slow = MagicMock()
        orch.best_win_rate = 0.0

        # Iteration at checkpoint interval
        metrics = await orch.train_iteration(1)
        assert "games_generated" in metrics
        assert metrics["games_generated"] == 5
        orch._generate_self_play_data.assert_called_once()
        orch._train_policy_value_network.assert_called_once()
        orch._evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_iteration_with_hrm_trm(self):
        orch = _make_orchestrator()
        orch._generate_self_play_data = AsyncMock(return_value=[])
        orch._train_policy_value_network = AsyncMock(return_value={"policy_loss": 0.3, "value_loss": 0.2})
        orch._train_hrm_agent = AsyncMock(return_value={"hrm_loss": 0.1, "hrm_halt_step": 2.0})
        orch._train_trm_agent = AsyncMock(return_value={"trm_loss": 0.2, "trm_convergence_step": 3.0})
        orch._evaluate = AsyncMock(return_value={"win_rate": 0.5})
        orch._save_checkpoint = MagicMock()
        orch._log_metrics = MagicMock()
        orch.monitor.alert_if_slow = MagicMock()
        orch.best_win_rate = 0.0

        await orch.train_iteration(1)
        # HRM and TRM should have been called since orch has hrm_agent and trm_agent
        orch._train_hrm_agent.assert_called_once()
        orch._train_trm_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_iteration_skips_evaluation(self):
        orch = _make_orchestrator()
        orch.config.training.checkpoint_interval = 5  # Only evaluate every 5th
        orch._generate_self_play_data = AsyncMock(return_value=[])
        orch._train_policy_value_network = AsyncMock(return_value={"policy_loss": 0.3, "value_loss": 0.2})
        orch._evaluate = AsyncMock()
        orch._save_checkpoint = MagicMock()
        orch._log_metrics = MagicMock()
        orch.monitor.alert_if_slow = MagicMock()

        await orch.train_iteration(2)  # Not at checkpoint interval
        orch._evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_train_iteration_saves_best(self):
        orch = _make_orchestrator()
        orch._generate_self_play_data = AsyncMock(return_value=[])
        orch._train_policy_value_network = AsyncMock(return_value={"policy_loss": 0.3, "value_loss": 0.2})
        orch._evaluate = AsyncMock(return_value={"win_rate": 0.8})
        orch._save_checkpoint = MagicMock()
        orch._log_metrics = MagicMock()
        orch.monitor.alert_if_slow = MagicMock()
        orch.best_win_rate = 0.5

        await orch.train_iteration(1)
        # Verify save was called with is_best=True
        call_args = orch._save_checkpoint.call_args
        assert call_args[1].get("is_best", call_args[0][2] if len(call_args[0]) > 2 else None) is True
        assert orch.best_win_rate == 0.8


@pytest.mark.unit
class TestGetMemoryUtilization:
    def test_returns_dict(self):
        orch = _make_orchestrator()
        result = orch._get_memory_utilization()
        assert isinstance(result, dict)
        assert "cpu_memory_mb" in result or "cpu_percent" in result
