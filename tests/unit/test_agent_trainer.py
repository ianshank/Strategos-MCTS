"""
Unit tests for Agent Trainer module.

Tests:
- HRMTrainer functionality
- TRMTrainer functionality
- SelfPlayEvaluator functionality
- Configuration handling
- Data loader utilities
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
nn = torch.nn

from src.training.agent_trainer import (
    DummyDataLoader,
    EvaluationConfig,
    HRMTrainer,
    HRMTrainingConfig,
    SelfPlayEvaluator,
    TrainingMetrics,
    TRMTrainer,
    TRMTrainingConfig,
    create_data_loader_from_buffer,
)


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = TrainingMetrics()
        assert metrics.loss == 0.0
        assert metrics.samples_processed == 0
        assert metrics.gradient_norm == 0.0
        assert metrics.learning_rate == 0.0
        assert isinstance(metrics.component_losses, dict)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TrainingMetrics(
            loss=0.5,
            component_losses={"task_loss": 0.3, "reg_loss": 0.2},
            samples_processed=100,
            gradient_norm=1.5,
            learning_rate=0.001,
        )

        result = metrics.to_dict()

        assert result["loss"] == 0.5
        assert result["samples_processed"] == 100
        assert result["gradient_norm"] == 1.5
        assert result["learning_rate"] == 0.001
        assert result["task_loss"] == 0.3
        assert result["reg_loss"] == 0.2


class TestHRMTrainingConfig:
    """Tests for HRM training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HRMTrainingConfig()

        assert config.batch_size == 32
        assert config.num_batches == 10
        assert config.gradient_clip_norm == 1.0
        assert config.ponder_weight == 0.01
        assert config.consistency_weight == 0.1
        assert config.use_mixed_precision is False

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "batch_size": 64,
            "num_batches": 20,
            "gradient_clip_norm": 0.5,
            "ponder_weight": 0.02,
        }

        config = HRMTrainingConfig.from_dict(config_dict)

        assert config.batch_size == 64
        assert config.num_batches == 20
        assert config.gradient_clip_norm == 0.5
        assert config.ponder_weight == 0.02
        # Defaults for missing keys
        assert config.use_mixed_precision is False

    def test_from_empty_dict(self):
        """Test creating config from empty dictionary uses defaults."""
        config = HRMTrainingConfig.from_dict({})

        assert config.batch_size == 32
        assert config.num_batches == 10


class TestTRMTrainingConfig:
    """Tests for TRM training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TRMTrainingConfig()

        assert config.batch_size == 32
        assert config.num_batches == 10
        assert config.gradient_clip_norm == 1.0
        assert config.supervision_weight_decay == 0.5
        assert config.use_mixed_precision is False

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "batch_size": 128,
            "supervision_weight_decay": 0.3,
        }

        config = TRMTrainingConfig.from_dict(config_dict)

        assert config.batch_size == 128
        assert config.supervision_weight_decay == 0.3


class TestEvaluationConfig:
    """Tests for evaluation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()

        assert config.num_games == 20
        assert config.temperature == 0.0
        assert config.mcts_iterations == 100
        assert config.win_threshold == 0.55

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "num_games": 50,
            "temperature": 0.1,
            "win_threshold": 0.60,
        }

        config = EvaluationConfig.from_dict(config_dict)

        assert config.num_games == 50
        assert config.temperature == 0.1
        assert config.win_threshold == 0.60


class TestDummyDataLoader:
    """Tests for DummyDataLoader."""

    def test_generates_correct_batch_size(self):
        """Test that loader generates batches of correct size."""
        loader = DummyDataLoader(
            batch_size=16,
            input_dim=64,
            output_dim=32,
            num_batches=5,
        )

        for inputs, targets in loader:
            assert inputs.shape[0] == 16
            assert inputs.shape[2] == 64
            assert targets.shape[0] == 16
            assert targets.shape[2] == 32
            break  # Just check first batch

    def test_generates_correct_number_of_batches(self):
        """Test that loader generates correct number of batches."""
        loader = DummyDataLoader(
            batch_size=8,
            input_dim=32,
            output_dim=16,
            num_batches=3,
        )

        batch_count = sum(1 for _ in loader)
        assert batch_count == 3

    def test_iterator_resets(self):
        """Test that iterator can be reused."""
        loader = DummyDataLoader(
            batch_size=4,
            input_dim=16,
            output_dim=8,
            num_batches=2,
        )

        # First pass
        count1 = sum(1 for _ in loader)

        # Second pass
        count2 = sum(1 for _ in loader)

        assert count1 == 2
        assert count2 == 2


class TestCreateDataLoaderFromBuffer:
    """Tests for create_data_loader_from_buffer utility."""

    def test_returns_dummy_loader_when_buffer_none(self):
        """Test that dummy loader is returned when buffer is None."""
        loader = create_data_loader_from_buffer(
            replay_buffer=None,
            batch_size=32,
            input_dim=64,
            output_dim=32,
        )

        assert isinstance(loader, DummyDataLoader)

    def test_returns_dummy_loader_when_buffer_not_ready(self):
        """Test that dummy loader is returned when buffer not ready."""
        mock_buffer = MagicMock()
        mock_buffer.is_ready.return_value = False

        loader = create_data_loader_from_buffer(
            replay_buffer=mock_buffer,
            batch_size=32,
            input_dim=64,
            output_dim=32,
        )

        assert isinstance(loader, DummyDataLoader)


class TestHRMTrainer:
    """Tests for HRM trainer."""

    @pytest.fixture
    def mock_hrm_agent(self):
        """Create a mock HRM agent."""
        agent = MagicMock()
        agent.train = MagicMock()
        agent.parameters = MagicMock(return_value=[torch.randn(10, 10, requires_grad=True)])

        # Mock forward pass output
        mock_output = MagicMock()
        mock_output.final_state = torch.randn(8, 16, 64)
        mock_output.total_ponder_cost = 0.5
        mock_output.halt_step = 5
        mock_output.convergence_path = [0.3, 0.5, 0.7, 0.9]
        agent.return_value = mock_output

        return agent

    @pytest.fixture
    def mock_loss_fn(self):
        """Create a mock loss function."""
        loss_fn = MagicMock()
        loss_fn.return_value = (
            torch.tensor(0.5, requires_grad=True),
            {
                "total": 0.5,
                "task": 0.3,
                "ponder": 0.15,
                "consistency": 0.05,
                "halt_step": 5,
            },
        )
        return loss_fn

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = MagicMock()
        optimizer.zero_grad = MagicMock()
        optimizer.step = MagicMock()
        optimizer.param_groups = [{"lr": 0.001}]
        return optimizer

    @pytest.mark.asyncio
    async def test_train_step_returns_metrics(self, mock_hrm_agent, mock_loss_fn, mock_optimizer):
        """Test that train_step returns proper metrics."""
        config = HRMTrainingConfig(batch_size=8)
        trainer = HRMTrainer(
            agent=mock_hrm_agent,
            optimizer=mock_optimizer,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        states = torch.randn(8, 16, 64)
        targets = torch.randn(8, 16, 64)

        metrics = await trainer.train_step(states, targets)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.samples_processed == 8


class TestTRMTrainer:
    """Tests for TRM trainer."""

    @pytest.fixture
    def mock_trm_agent(self):
        """Create a mock TRM agent."""
        agent = MagicMock()
        agent.train = MagicMock()
        agent.parameters = MagicMock(return_value=[torch.randn(10, 10, requires_grad=True)])

        # Mock forward pass output
        mock_output = MagicMock()
        mock_output.final_prediction = torch.randn(8, 16, 32)
        mock_output.intermediate_predictions = [torch.randn(8, 16, 32) for _ in range(3)]
        mock_output.recursion_depth = 3
        mock_output.converged = True
        mock_output.convergence_step = 3
        mock_output.residual_norms = [0.5, 0.2, 0.05]
        agent.return_value = mock_output

        return agent

    @pytest.fixture
    def mock_loss_fn(self):
        """Create a mock loss function."""
        loss_fn = MagicMock()
        loss_fn.return_value = (
            torch.tensor(0.4, requires_grad=True),
            {
                "total": 0.4,
                "final": 0.3,
                "intermediate_mean": 0.1,
                "recursion_depth": 3,
                "converged": True,
                "convergence_step": 3,
            },
        )
        return loss_fn

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = MagicMock()
        optimizer.zero_grad = MagicMock()
        optimizer.step = MagicMock()
        optimizer.param_groups = [{"lr": 0.001}]
        return optimizer

    @pytest.mark.asyncio
    async def test_train_step_returns_metrics(self, mock_trm_agent, mock_loss_fn, mock_optimizer):
        """Test that train_step returns proper metrics."""
        config = TRMTrainingConfig(batch_size=8)
        trainer = TRMTrainer(
            agent=mock_trm_agent,
            optimizer=mock_optimizer,
            loss_fn=mock_loss_fn,
            config=config,
            device="cpu",
        )

        inputs = torch.randn(8, 16, 64)
        targets = torch.randn(8, 16, 32)

        metrics = await trainer.train_step(inputs, targets)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.samples_processed == 8


class TestSelfPlayEvaluator:
    """Tests for SelfPlayEvaluator."""

    @pytest.fixture
    def mock_mcts(self):
        """Create a mock MCTS object."""
        mcts = MagicMock()
        mcts.policy_value_network = MagicMock()
        mcts.search = AsyncMock(return_value=({"action_0": 0.5, "action_1": 0.5}, 0.5))
        return mcts

    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state factory."""

        def create_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=1)
            return state

        return create_state

    @pytest.mark.asyncio
    async def test_evaluate_returns_metrics(self, mock_mcts, mock_game_state):
        """Test that evaluate returns proper metrics dictionary."""
        config = EvaluationConfig(num_games=2)
        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=mock_game_state,
            config=config,
            device="cpu",
        )

        model = MagicMock()
        model.eval = MagicMock()

        # The mcts.search needs to be awaitable
        mock_mcts.search = AsyncMock(return_value=({"action_0": 1.0}, 0.5))

        # Mock state properly
        mock_state = MagicMock()
        terminal_sequence = [False, False, True, False, False, True]  # Two games
        mock_state.is_terminal = MagicMock(side_effect=terminal_sequence)
        mock_state.apply_action = MagicMock(return_value=mock_state)
        mock_state.get_reward = MagicMock(return_value=1)

        def create_fresh_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=1)
            return state

        evaluator.initial_state_fn = create_fresh_state

        metrics = await evaluator.evaluate(model)

        assert "win_rate" in metrics
        assert "eval_games" in metrics
        assert "wins" in metrics
        assert "losses" in metrics
        assert "draws" in metrics
        assert isinstance(metrics["win_rate"], float)


class TestIntegration:
    """Integration tests for training components."""

    @pytest.mark.asyncio
    async def test_hrm_training_epoch_with_dummy_data(self):
        """Test HRM training with dummy data loader."""

        # Create a simple HRM-like model for testing
        class SimpleHRM(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def forward(self, x, return_decomposition=False):
                output = MagicMock()
                output.final_state = self.fc(x)
                output.total_ponder_cost = 0.1
                output.halt_step = 3
                output.convergence_path = [0.5, 0.7, 0.9]
                return output

        class SimpleLoss(nn.Module):
            def forward(self, hrm_output, predictions, targets, task_loss_fn):
                loss = task_loss_fn(predictions, targets)
                return loss, {
                    "total": loss.item(),
                    "task": loss.item(),
                    "ponder": hrm_output.total_ponder_cost,
                    "consistency": 0.0,
                    "halt_step": hrm_output.halt_step,
                }

        model = SimpleHRM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = SimpleLoss()
        config = HRMTrainingConfig(batch_size=4, num_batches=2)

        trainer = HRMTrainer(
            agent=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=64,
            output_dim=64,
            num_batches=2,
        )

        metrics = await trainer.train_epoch(data_loader)

        assert "loss" in metrics
        assert "hrm_halt_step" in metrics
        assert metrics["samples_processed"] == 8  # 4 * 2 batches

    @pytest.mark.asyncio
    async def test_trm_training_epoch_with_dummy_data(self):
        """Test TRM training with dummy data loader."""

        class SimpleTRM(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 32)

            def forward(self, x, check_convergence=False):
                output = MagicMock()
                output.final_prediction = self.fc(x)
                output.intermediate_predictions = [self.fc(x), self.fc(x)]
                output.recursion_depth = 2
                output.converged = True
                output.convergence_step = 2
                output.residual_norms = [0.1, 0.05]
                return output

        class SimpleLoss(nn.Module):
            def forward(self, trm_output, targets):
                # Handle shape mismatch by averaging
                pred = trm_output.final_prediction
                loss = nn.functional.mse_loss(pred, targets[..., :32])
                return loss, {
                    "total": loss.item(),
                    "final": loss.item(),
                    "intermediate_mean": 0.1,
                    "recursion_depth": trm_output.recursion_depth,
                    "converged": trm_output.converged,
                    "convergence_step": trm_output.convergence_step,
                }

        model = SimpleTRM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = SimpleLoss()
        config = TRMTrainingConfig(batch_size=4, num_batches=2)

        trainer = TRMTrainer(
            agent=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=64,
            output_dim=32,
            num_batches=2,
        )

        metrics = await trainer.train_epoch(data_loader)

        assert "loss" in metrics
        assert "trm_convergence_step" in metrics
        assert metrics["samples_processed"] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
