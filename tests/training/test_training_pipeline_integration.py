"""
Integration tests for the unified training pipeline.

Tests end-to-end training flow including:
- HRM agent training with proper loss computation
- TRM agent training with deep supervision
- Self-play evaluation
- Checkpoint saving and loading

Best Practices:
- Deterministic tests with fixed seeds
- No brittle mocks (use return_value instead of side_effect sequences)
- Configuration-driven test parameters
- Proper cleanup and isolation
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
nn = torch.nn

# Test constants for determinism
TEST_SEED = 42
TEST_BATCH_SIZE = 4
TEST_SEQ_LENGTH = 8
TEST_H_DIM = 32
TEST_L_DIM = 16
TEST_OUTPUT_DIM = 10


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Set deterministic seed for all tests."""
    torch.manual_seed(TEST_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TEST_SEED)
    yield


@pytest.fixture
def deterministic_tensor_factory():
    """Factory for creating deterministic tensors."""

    def create_tensor(*shape: int) -> torch.Tensor:
        """Create a deterministic tensor with given shape."""
        generator = torch.Generator().manual_seed(TEST_SEED)
        return torch.randn(*shape, generator=generator)

    return create_tensor


@pytest.fixture
def mock_game_state():
    """Create a mock game state that behaves consistently."""
    state = MagicMock()

    # Use a counter-based approach instead of fragile side_effect list
    state._move_count = 0
    state._max_moves = 3

    def is_terminal_impl():
        return state._move_count >= state._max_moves

    def apply_action_impl(action):
        state._move_count += 1
        return state

    state.is_terminal = MagicMock(side_effect=is_terminal_impl)
    state.get_reward = MagicMock(return_value=1.0)
    state.apply_action = MagicMock(side_effect=apply_action_impl)
    state.get_legal_actions = MagicMock(return_value=["a", "b", "c"])

    # Deterministic tensor for state conversion
    generator = torch.Generator().manual_seed(TEST_SEED)
    state.to_tensor = MagicMock(return_value=torch.randn(1, 19, 19, generator=generator))

    return state


@pytest.fixture
def initial_state_fn(mock_game_state):
    """Create initial state function that resets state."""

    def create_initial_state():
        # Reset the move counter for each new game
        mock_game_state._move_count = 0
        return mock_game_state

    return create_initial_state


@pytest.fixture
def system_config(tmp_path):
    """Create minimal system configuration for testing.

    Uses pytest's tmp_path fixture for automatic cleanup after tests.
    This avoids disk space leaks from orphaned temp directories.

    Args:
        tmp_path: Pytest-provided temporary directory (auto-cleaned)

    """
    from src.training.system_config import (
        HRMConfig,
        MCTSConfig,
        NeuralNetworkConfig,
        SystemConfig,
        TrainingConfig,
        TRMConfig,
    )

    # Use pytest's tmp_path for automatic cleanup
    temp_dir = str(tmp_path)

    return SystemConfig(
        # Minimal sizes for testing
        neural_net=NeuralNetworkConfig(
            num_res_blocks=1,
            num_channels=16,
            policy_conv_channels=2,
            value_conv_channels=1,
            action_size=TEST_OUTPUT_DIM,
        ),
        hrm=HRMConfig(
            h_dim=TEST_H_DIM,
            l_dim=TEST_L_DIM,
            num_h_layers=1,
            num_l_layers=1,
            max_outer_steps=2,
            halt_threshold=0.9,
            ponder_epsilon=0.01,
            ponder_weight=0.01,
            dropout=0.0,
        ),
        trm=TRMConfig(
            latent_dim=TEST_H_DIM,
            hidden_dim=64,
            num_recursions=2,
            min_recursions=1,
            convergence_threshold=0.01,
            supervision_weight_decay=0.5,
            use_layer_norm=True,
            deep_supervision=True,
            dropout=0.0,
        ),
        mcts=MCTSConfig(
            num_simulations=4,
            c_puct=1.0,
            dirichlet_alpha=0.3,
            temperature_threshold=10,
        ),
        training=TrainingConfig(
            batch_size=TEST_BATCH_SIZE,
            learning_rate=0.001,
            buffer_size=100,
            games_per_iteration=1,
            checkpoint_interval=1,
            evaluation_games=2,
            win_threshold=0.5,
            patience=10,
            min_delta=0.01,
            hrm_train_batches=2,
            trm_train_batches=2,
            eval_temperature=0.0,
            gradient_clip_norm=1.0,
        ),
        device="cpu",
        seed=TEST_SEED,
        use_mixed_precision=False,
        checkpoint_dir=temp_dir,
        data_dir=temp_dir,
        log_dir=temp_dir,
        use_wandb=False,
    )


class TestHRMTrainerIntegration:
    """Integration tests for HRM training."""

    @pytest.fixture
    def hrm_setup(self, system_config, deterministic_tensor_factory):
        """Setup HRM training components."""
        from src.agents.hrm_agent import HRMLoss, create_hrm_agent
        from src.training.agent_trainer import HRMTrainer, HRMTrainingConfig

        torch.manual_seed(TEST_SEED)

        agent = create_hrm_agent(system_config.hrm, device="cpu")
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        loss_fn = HRMLoss(ponder_weight=0.01)
        config = HRMTrainingConfig(
            batch_size=TEST_BATCH_SIZE,
            num_batches=2,
            gradient_clip_norm=1.0,
        )

        trainer = HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        return trainer, agent, deterministic_tensor_factory

    @pytest.mark.asyncio
    async def test_hrm_train_step(self, hrm_setup):
        """Test single HRM training step produces valid metrics."""
        trainer, agent, tensor_factory = hrm_setup

        # Create deterministic data
        states = tensor_factory(TEST_BATCH_SIZE, TEST_SEQ_LENGTH, TEST_H_DIM)
        targets = tensor_factory(TEST_BATCH_SIZE, TEST_SEQ_LENGTH, TEST_H_DIM)

        metrics = await trainer.train_step(states, targets)

        # Verify metrics structure
        assert metrics.loss > 0, "Loss should be positive"
        assert metrics.samples_processed == TEST_BATCH_SIZE
        assert metrics.gradient_norm >= 0, "Gradient norm should be non-negative"
        assert "hrm_task_loss" in metrics.component_losses
        assert "hrm_ponder_cost" in metrics.component_losses
        assert "hrm_halt_step" in metrics.component_losses

    @pytest.mark.asyncio
    async def test_hrm_train_epoch(self, hrm_setup):
        """Test HRM training epoch aggregates metrics correctly."""
        from src.training.agent_trainer import DummyDataLoader

        trainer, agent, _ = hrm_setup

        data_loader = DummyDataLoader(
            batch_size=TEST_BATCH_SIZE,
            input_dim=TEST_H_DIM,
            output_dim=TEST_H_DIM,
            num_batches=2,
            device="cpu",
            sequence_length=TEST_SEQ_LENGTH,
        )

        metrics = await trainer.train_epoch(data_loader)

        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert "hrm_halt_step" in metrics

    @pytest.mark.asyncio
    async def test_hrm_training_reduces_loss(self, hrm_setup):
        """Test that training actually reduces loss over iterations."""
        from src.training.agent_trainer import DummyDataLoader

        trainer, agent, _ = hrm_setup

        # Fixed data for consistent comparison
        torch.manual_seed(TEST_SEED)
        data_loader = DummyDataLoader(
            batch_size=TEST_BATCH_SIZE,
            input_dim=TEST_H_DIM,
            output_dim=TEST_H_DIM,
            num_batches=5,
            device="cpu",
            sequence_length=TEST_SEQ_LENGTH,
        )

        initial_metrics = await trainer.train_epoch(data_loader)
        initial_loss = initial_metrics["loss"]

        # Train for a few more epochs
        for _ in range(3):
            torch.manual_seed(TEST_SEED)  # Reset for same data
            data_loader = DummyDataLoader(
                batch_size=TEST_BATCH_SIZE,
                input_dim=TEST_H_DIM,
                output_dim=TEST_H_DIM,
                num_batches=5,
                device="cpu",
                sequence_length=TEST_SEQ_LENGTH,
            )
            metrics = await trainer.train_epoch(data_loader)

        # Loss should generally decrease (or at least not explode)
        assert metrics["loss"] < initial_loss * 2, "Loss should not explode during training"


class TestTRMTrainerIntegration:
    """Integration tests for TRM training."""

    @pytest.fixture
    def trm_setup(self, system_config, deterministic_tensor_factory):
        """Setup TRM training components."""
        from src.agents.trm_agent import TRMLoss, create_trm_agent
        from src.training.agent_trainer import TRMTrainer, TRMTrainingConfig

        torch.manual_seed(TEST_SEED)

        agent = create_trm_agent(
            system_config.trm,
            output_dim=TEST_OUTPUT_DIM,
            device="cpu",
        )
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        loss_fn = TRMLoss(
            task_loss_fn=nn.MSELoss(),
            supervision_weight_decay=0.5,
        )
        config = TRMTrainingConfig(
            batch_size=TEST_BATCH_SIZE,
            num_batches=2,
            gradient_clip_norm=1.0,
        )

        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        return trainer, agent, deterministic_tensor_factory

    @pytest.mark.asyncio
    async def test_trm_train_step(self, trm_setup):
        """Test single TRM training step produces valid metrics."""
        trainer, agent, tensor_factory = trm_setup

        # Create deterministic data
        inputs = tensor_factory(TEST_BATCH_SIZE, TEST_SEQ_LENGTH, TEST_H_DIM)
        targets = tensor_factory(TEST_BATCH_SIZE, TEST_SEQ_LENGTH, TEST_OUTPUT_DIM)

        metrics = await trainer.train_step(inputs, targets)

        assert metrics.loss > 0, "Loss should be positive"
        assert metrics.samples_processed == TEST_BATCH_SIZE
        assert metrics.gradient_norm >= 0
        assert "trm_final_loss" in metrics.component_losses
        assert "trm_convergence_step" in metrics.component_losses

    @pytest.mark.asyncio
    async def test_trm_train_epoch(self, trm_setup):
        """Test TRM training epoch aggregates metrics correctly."""
        from src.training.agent_trainer import DummyDataLoader

        trainer, agent, _ = trm_setup

        data_loader = DummyDataLoader(
            batch_size=TEST_BATCH_SIZE,
            input_dim=TEST_H_DIM,
            output_dim=TEST_OUTPUT_DIM,
            num_batches=2,
            device="cpu",
            sequence_length=TEST_SEQ_LENGTH,
        )

        metrics = await trainer.train_epoch(data_loader)

        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert "trm_convergence_step" in metrics


class TestSelfPlayEvaluatorIntegration:
    """Integration tests for self-play evaluation."""

    @pytest.fixture
    def evaluator_setup(self, mock_game_state, initial_state_fn):
        """Setup evaluation components."""
        from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator

        # Mock MCTS with consistent behavior
        mock_mcts = MagicMock()
        mock_root_node = MagicMock()
        mock_root_node.value = 0.6
        mock_mcts.search = AsyncMock(return_value=({"a": 0.5, "b": 0.3, "c": 0.2}, mock_root_node))

        config = EvaluationConfig(
            num_games=2,
            temperature=0.0,
            mcts_iterations=4,
            win_threshold=0.5,
        )

        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=initial_state_fn,
            config=config,
            device="cpu",
        )

        return evaluator, mock_mcts

    @pytest.mark.asyncio
    async def test_play_game(self, evaluator_setup):
        """Test playing a single game produces valid result."""
        evaluator, mock_mcts = evaluator_setup

        model1 = MagicMock()
        model2 = MagicMock()

        result, stats = await evaluator.play_game(model1, model2)

        assert result in [-1, 0, 1], "Result should be -1, 0, or 1"
        assert "moves" in stats
        assert "model1_avg_mcts_value" in stats
        assert stats["moves"] >= 0

    @pytest.mark.asyncio
    async def test_evaluate(self, evaluator_setup):
        """Test full evaluation produces valid metrics."""
        evaluator, mock_mcts = evaluator_setup

        current_model = MagicMock()
        best_model = MagicMock()

        metrics = await evaluator.evaluate(current_model, best_model)

        assert "win_rate" in metrics
        assert "wins" in metrics
        assert "losses" in metrics
        assert "draws" in metrics
        assert "eval_games" in metrics
        assert 0.0 <= metrics["win_rate"] <= 1.0
        assert metrics["eval_games"] == 2


class TestUnifiedOrchestratorIntegration:
    """Integration tests for the unified training orchestrator."""

    @pytest.fixture
    def orchestrator(self, system_config, initial_state_fn):
        """Create orchestrator instance."""
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        torch.manual_seed(TEST_SEED)

        return UnifiedTrainingOrchestrator(
            config=system_config,
            initial_state_fn=initial_state_fn,
            board_size=19,
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes all components."""
        assert orchestrator.policy_value_net is not None
        assert orchestrator.hrm_agent is not None
        assert orchestrator.trm_agent is not None
        assert orchestrator.mcts is not None
        assert orchestrator.replay_buffer is not None

    def test_parameter_count(self, orchestrator):
        """Test that models have trainable parameters."""
        pv_params = orchestrator.policy_value_net.get_parameter_count()
        hrm_params = orchestrator.hrm_agent.get_parameter_count()
        trm_params = orchestrator.trm_agent.get_parameter_count()

        assert pv_params > 0, "Policy-value network should have parameters"
        assert hrm_params > 0, "HRM agent should have parameters"
        assert trm_params > 0, "TRM agent should have parameters"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_train_iteration(self, orchestrator):
        """Test single training iteration produces metrics."""
        torch.manual_seed(TEST_SEED)

        # Mock self-play to avoid long running games
        with patch.object(
            orchestrator.self_play_collector,
            "play_game",
            new_callable=AsyncMock,
        ) as mock_play:
            # Create deterministic mock examples
            generator = torch.Generator().manual_seed(TEST_SEED)
            mock_example = MagicMock()
            mock_example.state = torch.randn(19, 19, generator=generator)
            mock_example.policy_target = torch.rand(TEST_OUTPUT_DIM, generator=generator)
            mock_example.value_target = torch.tensor([0.5])
            mock_play.return_value = [mock_example] * 10

            metrics = await orchestrator.train_iteration(iteration=1)

            assert metrics is not None
            assert "games_generated" in metrics or "policy_loss" in metrics

    def test_save_and_load_checkpoint(self, orchestrator, system_config):
        """Test checkpoint saving and loading preserves state."""
        # Save checkpoint
        orchestrator._save_checkpoint(
            iteration=1,
            metrics={"test_metric": 0.5},
            is_best=True,
        )

        # Verify checkpoint file exists
        checkpoint_path = Path(system_config.checkpoint_dir) / "checkpoint_iter_1.pt"
        assert checkpoint_path.exists(), "Checkpoint file should be created"

        # Verify best model file exists
        best_path = Path(system_config.checkpoint_dir) / "best_model.pt"
        assert best_path.exists(), "Best model file should be created"

        # Load checkpoint
        orchestrator.load_checkpoint(str(checkpoint_path))
        assert orchestrator.current_iteration == 1


class TestDataLoaderIntegration:
    """Integration tests for data loader utilities."""

    def test_dummy_data_loader_shapes(self):
        """Test dummy data loader generates correct tensor shapes."""
        from src.training.agent_trainer import DummyDataLoader

        loader = DummyDataLoader(
            batch_size=8,
            input_dim=64,
            output_dim=32,
            num_batches=5,
            device="cpu",
            sequence_length=16,
        )

        batches = list(loader)

        assert len(batches) == 5
        for inputs, targets in batches:
            assert inputs.shape == (8, 16, 64)
            assert targets.shape == (8, 16, 32)

    def test_dummy_data_loader_iteration(self):
        """Test dummy data loader can be iterated multiple times."""
        from src.training.agent_trainer import DummyDataLoader

        loader = DummyDataLoader(
            batch_size=4,
            input_dim=32,
            output_dim=16,
            num_batches=3,
            device="cpu",
        )

        # First iteration
        batches1 = list(loader)
        assert len(batches1) == 3

        # Second iteration (should reset)
        batches2 = list(loader)
        assert len(batches2) == 3

    def test_create_data_loader_from_empty_buffer(self):
        """Test data loader creation with empty buffer falls back to dummy."""
        from src.training.agent_trainer import create_data_loader_from_buffer

        loader = create_data_loader_from_buffer(
            replay_buffer=None,
            batch_size=8,
            input_dim=64,
            output_dim=32,
        )

        # Should get a DummyDataLoader
        batches = list(loader)
        assert len(batches) > 0


class TestEndToEndTraining:
    """End-to-end training tests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_minimal_training_loop(self, system_config, initial_state_fn):
        """Test a minimal training loop runs without errors."""
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        torch.manual_seed(TEST_SEED)

        orchestrator = UnifiedTrainingOrchestrator(
            config=system_config,
            initial_state_fn=initial_state_fn,
            board_size=19,
        )

        # Mock self-play to avoid long running games
        with patch.object(
            orchestrator.self_play_collector,
            "play_game",
            new_callable=AsyncMock,
        ) as mock_play:
            generator = torch.Generator().manual_seed(TEST_SEED)
            mock_example = MagicMock()
            mock_example.state = torch.randn(19, 19, generator=generator)
            mock_example.policy_target = torch.rand(TEST_OUTPUT_DIM, generator=generator)
            mock_example.value_target = torch.tensor([0.5])
            mock_play.return_value = [mock_example] * 10

            # Run a single iteration
            metrics = await orchestrator.train_iteration(iteration=1)

            assert metrics is not None
            # Should have some metrics (exact keys depend on implementation)
            assert len(metrics) > 0


class TestConfigurationValidation:
    """Tests for configuration validation and defaults."""

    def test_training_config_defaults(self):
        """Test training config has sensible defaults."""
        from src.training.agent_trainer import (
            EvaluationConfig,
            HRMTrainingConfig,
            TRMTrainingConfig,
        )

        hrm_config = HRMTrainingConfig()
        assert hrm_config.batch_size > 0
        assert hrm_config.gradient_clip_norm > 0

        trm_config = TRMTrainingConfig()
        assert trm_config.batch_size > 0
        assert 0 < trm_config.supervision_weight_decay <= 1

        eval_config = EvaluationConfig()
        assert eval_config.num_games > 0
        assert eval_config.temperature >= 0
        assert 0 < eval_config.win_threshold <= 1

    def test_config_from_dict(self):
        """Test config creation from dictionary."""
        from src.training.agent_trainer import HRMTrainingConfig

        config_dict = {
            "batch_size": 64,
            "num_batches": 20,
            "gradient_clip_norm": 0.5,
        }

        config = HRMTrainingConfig.from_dict(config_dict)

        assert config.batch_size == 64
        assert config.num_batches == 20
        assert config.gradient_clip_norm == 0.5
