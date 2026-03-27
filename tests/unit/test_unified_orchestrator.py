"""Unit tests for src/training/unified_orchestrator.py."""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest


@pytest.mark.unit
class TestUnifiedTrainingOrchestratorInit:
    """Tests for UnifiedTrainingOrchestrator initialization and setup methods."""

    def _make_config(self):
        """Create a minimal SystemConfig mock for testing."""
        from src.training.system_config import SystemConfig

        config = SystemConfig(device="cpu", use_wandb=False, use_mixed_precision=False)
        # Override to small values for fast tests
        config.training.games_per_iteration = 2
        config.training.batch_size = 4
        config.training.buffer_size = 100
        config.training.checkpoint_interval = 1
        config.training.evaluation_games = 2
        config.training.patience = 3
        config.training.min_delta = 0.01
        config.log_interval = 1
        return config

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_init_creates_components(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
    ):
        """Test that __init__ creates all required components."""
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        # Setup mocks
        import torch

        dummy_param = torch.nn.Parameter(torch.randn(2, 2))

        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        mock_pv_net.parameters.return_value = iter([dummy_param])
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = iter([dummy_param])
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = iter([dummy_param])
        mock_trm.return_value = mock_trm_agent

        config = self._make_config()
        initial_state_fn = MagicMock()

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=initial_state_fn, board_size=9)

        assert orch.config is config
        assert orch.initial_state_fn is initial_state_fn
        assert orch.board_size == 9
        assert orch.current_iteration == 0
        assert orch.best_win_rate == 0.0
        assert orch.best_model_path is None

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_setup_paths_creates_directories(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
        tmp_path,
    ):
        """Test that _setup_paths creates checkpoint, data, and log directories."""
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        import torch

        dummy_param = torch.nn.Parameter(torch.randn(2, 2))

        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        mock_pv_net.parameters.return_value = iter([dummy_param])
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = iter([dummy_param])
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = iter([dummy_param])
        mock_trm.return_value = mock_trm_agent

        config = self._make_config()
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        assert orch.checkpoint_dir.exists()
        assert orch.data_dir.exists()
        assert orch.log_dir.exists()

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_setup_optimizers_cosine_schedule(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
        tmp_path,
    ):
        """Test that _setup_optimizers creates correct scheduler for cosine LR schedule."""
        import torch
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        # Provide real parameters for optimizer
        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_pv_net.parameters.return_value = [param]
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_trm.return_value = mock_trm_agent

        config = self._make_config()
        config.training.lr_schedule = "cosine"
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        assert orch.pv_scheduler is not None
        assert isinstance(orch.pv_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_setup_optimizers_step_schedule(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
        tmp_path,
    ):
        """Test that _setup_optimizers creates StepLR scheduler for step schedule."""
        import torch
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        mock_pv_net.parameters.return_value = [param]
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_trm.return_value = mock_trm_agent

        config = self._make_config()
        config.training.lr_schedule = "step"
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        assert orch.pv_scheduler is not None
        assert isinstance(orch.pv_scheduler, torch.optim.lr_scheduler.StepLR)

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_setup_optimizers_constant_schedule(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
        tmp_path,
    ):
        """Test that constant LR schedule results in no scheduler."""
        import torch
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        mock_pv_net.parameters.return_value = [param]
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_trm.return_value = mock_trm_agent

        config = self._make_config()
        config.training.lr_schedule = "constant"
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        assert orch.pv_scheduler is None


@pytest.mark.unit
class TestComputeGradientNorm:
    """Tests for _compute_gradient_norm method."""

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_gradient_norm_computation(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
        tmp_path,
    ):
        """Test gradient norm computation returns correct value."""
        import torch
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator
        from src.training.system_config import SystemConfig

        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        mock_pv_net.parameters.return_value = [param]
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_trm.return_value = mock_trm_agent

        config = SystemConfig(device="cpu", use_wandb=False, use_mixed_precision=False)
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        # Create a simple model with known gradients
        model = torch.nn.Linear(3, 2)
        x = torch.randn(1, 3)
        y = model(x)
        y.sum().backward()

        norm = orch._compute_gradient_norm(model)
        assert isinstance(norm, float)
        assert norm >= 0.0

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_gradient_norm_no_grads(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
        tmp_path,
    ):
        """Test gradient norm when no gradients are computed returns 0."""
        import torch
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator
        from src.training.system_config import SystemConfig

        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        mock_pv_net.parameters.return_value = [param]
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_trm.return_value = mock_trm_agent

        config = SystemConfig(device="cpu", use_wandb=False, use_mixed_precision=False)
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        # Model with no gradients computed
        model = torch.nn.Linear(3, 2)
        norm = orch._compute_gradient_norm(model)
        assert norm == 0.0


@pytest.mark.unit
class TestShouldEarlyStop:
    """Tests for _should_early_stop method."""

    def _make_orchestrator(self, tmp_path, checkpoint_interval=1, patience=3, min_delta=0.01):
        """Helper to create an orchestrator with mocked components."""
        import torch
        from unittest.mock import patch as _patch

        patches = [
            _patch("src.training.unified_orchestrator.create_policy_value_network"),
            _patch("src.training.unified_orchestrator.create_hrm_agent"),
            _patch("src.training.unified_orchestrator.create_trm_agent"),
            _patch("src.training.unified_orchestrator.NeuralMCTS"),
            _patch("src.training.unified_orchestrator.SelfPlayCollector"),
            _patch("src.training.unified_orchestrator.PrioritizedReplayBuffer"),
            _patch("src.training.unified_orchestrator.PerformanceMonitor"),
        ]

        for p in patches:
            mock = p.start()
            if "policy_value" in p.attribute:
                m = MagicMock()
                m.get_parameter_count.return_value = 100
                m.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
                mock.return_value = m
            elif "hrm" in p.attribute or "trm" in p.attribute:
                m = MagicMock()
                m.get_parameter_count.return_value = 50
                m.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
                mock.return_value = m

        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator
        from src.training.system_config import SystemConfig

        config = SystemConfig(device="cpu", use_wandb=False, use_mixed_precision=False)
        config.training.checkpoint_interval = checkpoint_interval
        config.training.patience = patience
        config.training.min_delta = min_delta
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        for p in patches:
            p.stop()

        return orch

    def test_no_early_stop_at_non_checkpoint_interval(self, tmp_path):
        """Test that early stopping is not triggered when not at checkpoint interval."""
        orch = self._make_orchestrator(tmp_path, checkpoint_interval=5)
        # Iteration 3 is not a multiple of 5
        assert orch._should_early_stop(3) is False

    def test_no_early_stop_when_improving(self, tmp_path):
        """Test that early stopping is not triggered when win rate improves."""
        orch = self._make_orchestrator(tmp_path, checkpoint_interval=1, patience=3)
        orch.best_win_rate = 0.5
        assert orch._should_early_stop(1) is False
        orch.best_win_rate = 0.6
        assert orch._should_early_stop(2) is False

    def test_early_stop_after_patience_exhausted(self, tmp_path):
        """Test that early stopping triggers after patience is exhausted."""
        orch = self._make_orchestrator(tmp_path, checkpoint_interval=1, patience=2)
        orch.best_win_rate = 0.5

        # First call initializes tracking
        assert orch._should_early_stop(1) is False
        # No improvement
        assert orch._should_early_stop(2) is False
        # Still no improvement, patience exhausted
        assert orch._should_early_stop(3) is True


@pytest.mark.unit
class TestGetMemoryUtilization:
    """Tests for _get_memory_utilization method."""

    @patch("src.training.unified_orchestrator.create_policy_value_network")
    @patch("src.training.unified_orchestrator.create_hrm_agent")
    @patch("src.training.unified_orchestrator.create_trm_agent")
    @patch("src.training.unified_orchestrator.NeuralMCTS")
    @patch("src.training.unified_orchestrator.SelfPlayCollector")
    @patch("src.training.unified_orchestrator.PrioritizedReplayBuffer")
    @patch("src.training.unified_orchestrator.PerformanceMonitor")
    def test_memory_utilization_returns_cpu_info(
        self,
        mock_monitor,
        mock_buffer,
        mock_collector,
        mock_mcts,
        mock_trm,
        mock_hrm,
        mock_pv,
        tmp_path,
    ):
        """Test that _get_memory_utilization returns CPU memory metrics."""
        import torch
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator
        from src.training.system_config import SystemConfig

        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_pv_net = MagicMock()
        mock_pv_net.get_parameter_count.return_value = 100
        mock_pv_net.parameters.return_value = [param]
        mock_pv.return_value = mock_pv_net

        mock_hrm_agent = MagicMock()
        mock_hrm_agent.get_parameter_count.return_value = 50
        mock_hrm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_hrm.return_value = mock_hrm_agent

        mock_trm_agent = MagicMock()
        mock_trm_agent.get_parameter_count.return_value = 50
        mock_trm_agent.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_trm.return_value = mock_trm_agent

        config = SystemConfig(device="cpu", use_wandb=False, use_mixed_precision=False)
        config.checkpoint_dir = str(tmp_path / "ckpts")
        config.data_dir = str(tmp_path / "data")
        config.log_dir = str(tmp_path / "logs")

        orch = UnifiedTrainingOrchestrator(config=config, initial_state_fn=MagicMock(), board_size=9)

        mem_info = orch._get_memory_utilization()
        assert "cpu_memory_mb" in mem_info
        assert "cpu_percent" in mem_info
        assert isinstance(mem_info["cpu_memory_mb"], float)
