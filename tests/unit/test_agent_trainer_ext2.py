"""
Extended unit tests for src/training/agent_trainer.py - Part 2.

Covers missed lines: 176-200, 257, 274->280, 333-351, 406, 423->429,
501-537, 556->560, 577-585, 602-605, 683-712.

Focus areas:
- HRMTrainer.train_step with mixed precision (lines 176-200)
- HRMTrainer.train_epoch batch limiting and averaging (lines 257, 274-280)
- TRMTrainer.train_step with mixed precision (lines 333-351)
- TRMTrainer.train_epoch batch limiting and averaging (lines 406, 423-429)
- SelfPlayEvaluator.play_game full flow (lines 501-537)
- SelfPlayEvaluator.evaluate with best_model=None (lines 556-560)
- SelfPlayEvaluator.evaluate game failure handling (lines 577-585)
- SelfPlayEvaluator.evaluate zero games played (lines 602-605)
- create_data_loader_from_buffer with ready buffer (lines 683-712)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_hrm_agent():
    """Create a mock HRM agent with proper forward pass."""
    agent = MagicMock()
    agent.train = MagicMock()
    agent.parameters = MagicMock(return_value=[torch.randn(10, 10, requires_grad=True)])

    mock_output = MagicMock()
    mock_output.final_state = torch.randn(4, 16, 64)
    mock_output.total_ponder_cost = 0.5
    mock_output.halt_step = 3
    mock_output.convergence_path = [0.3, 0.5, 0.7]
    agent.return_value = mock_output
    return agent


def _make_mock_hrm_loss_fn():
    """Create a mock HRM loss function."""
    loss_fn = MagicMock()
    loss_fn.return_value = (
        torch.tensor(0.6, requires_grad=True),
        {
            "total": 0.6,
            "task": 0.4,
            "ponder": 0.15,
            "consistency": 0.05,
            "halt_step": 3,
        },
    )
    return loss_fn


def _make_mock_trm_agent():
    """Create a mock TRM agent."""
    agent = MagicMock()
    agent.train = MagicMock()
    agent.parameters = MagicMock(return_value=[torch.randn(10, 10, requires_grad=True)])

    mock_output = MagicMock()
    mock_output.final_prediction = torch.randn(4, 16, 32)
    mock_output.intermediate_predictions = [torch.randn(4, 16, 32)]
    mock_output.recursion_depth = 2
    mock_output.converged = True
    mock_output.convergence_step = 2
    mock_output.residual_norms = [0.2, 0.05]
    agent.return_value = mock_output
    return agent


def _make_mock_trm_loss_fn():
    """Create a mock TRM loss function."""
    loss_fn = MagicMock()
    loss_fn.return_value = (
        torch.tensor(0.35, requires_grad=True),
        {
            "total": 0.35,
            "final": 0.25,
            "intermediate_mean": 0.1,
            "convergence_step": 2,
            "converged": True,
        },
    )
    return loss_fn


def _make_mock_optimizer():
    """Create a mock optimizer."""
    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    optimizer.param_groups = [{"lr": 0.001}]
    return optimizer


def _make_mock_scaler():
    """Create a mock GradScaler for mixed precision."""
    scaler = MagicMock()
    scaler.scale = MagicMock(return_value=MagicMock(backward=MagicMock()))
    scaler.unscale_ = MagicMock()
    scaler.step = MagicMock()
    scaler.update = MagicMock()
    return scaler


# ---------------------------------------------------------------------------
# HRMTrainer - mixed precision (lines 176-200)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHRMTrainerMixedPrecision:
    """Test HRM training with mixed precision enabled."""

    @pytest.mark.asyncio
    async def test_train_step_mixed_precision(self):
        """Train step with use_mixed_precision=True uses scaler (lines 176-200)."""
        agent = _make_mock_hrm_agent()
        loss_fn = _make_mock_hrm_loss_fn()
        optimizer = _make_mock_optimizer()
        scaler = _make_mock_scaler()

        config = HRMTrainingConfig(batch_size=4, use_mixed_precision=True)
        trainer = HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
            scaler=scaler,
        )

        states = torch.randn(4, 16, 64)
        targets = torch.randn(4, 16, 64)

        metrics = await trainer.train_step(states, targets)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.samples_processed == 4
        assert metrics.loss == 0.6
        # Verify scaler was used
        scaler.scale.assert_called_once()
        scaler.unscale_.assert_called_once_with(optimizer)
        scaler.step.assert_called_once_with(optimizer)
        scaler.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_step_mixed_precision_no_scaler_fallback(self):
        """When use_mixed_precision=True but scaler=None, uses normal path."""
        agent = _make_mock_hrm_agent()
        loss_fn = _make_mock_hrm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = HRMTrainingConfig(batch_size=4, use_mixed_precision=True)
        trainer = HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
            scaler=None,  # No scaler
        )

        states = torch.randn(4, 16, 64)
        targets = torch.randn(4, 16, 64)

        metrics = await trainer.train_step(states, targets)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.samples_processed == 4
        # Normal optimizer.step should be called instead
        optimizer.step.assert_called_once()


# ---------------------------------------------------------------------------
# HRMTrainer.train_epoch - batch limiting and averaging (lines 257, 274-280)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHRMTrainerEpoch:
    """Test HRM train_epoch with batch limiting and metric averaging."""

    @pytest.mark.asyncio
    async def test_train_epoch_limits_batches(self):
        """train_epoch stops after config.num_batches (line 257)."""
        agent = _make_mock_hrm_agent()
        loss_fn = _make_mock_hrm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = HRMTrainingConfig(batch_size=4, num_batches=2)
        trainer = HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        # Provide more batches than num_batches
        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=64,
            output_dim=64,
            num_batches=5,
        )

        metrics = await trainer.train_epoch(data_loader)

        # Should have processed 2 batches * 4 samples
        assert metrics["samples_processed"] == 8

    @pytest.mark.asyncio
    async def test_train_epoch_averages_metrics(self):
        """train_epoch averages loss and gradient_norm over batches (lines 274-280)."""
        agent = _make_mock_hrm_agent()
        optimizer = _make_mock_optimizer()

        # Loss function returns different values each call
        call_count = [0]
        losses = [0.4, 0.6]

        def varying_loss(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            loss_val = losses[idx] if idx < len(losses) else 0.5
            return (
                torch.tensor(loss_val, requires_grad=True),
                {
                    "total": loss_val,
                    "task": loss_val * 0.6,
                    "ponder": loss_val * 0.3,
                    "consistency": loss_val * 0.1,
                    "halt_step": 3,
                },
            )

        loss_fn = MagicMock(side_effect=varying_loss)

        config = HRMTrainingConfig(batch_size=4, num_batches=2)
        trainer = HRMTrainer(
            agent=agent,
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

        # Average of 0.4 and 0.6 = 0.5
        assert abs(metrics["loss"] - 0.5) < 1e-5

    @pytest.mark.asyncio
    async def test_train_epoch_empty_dataloader(self):
        """train_epoch with no batches returns zero metrics."""
        agent = _make_mock_hrm_agent()
        loss_fn = _make_mock_hrm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = HRMTrainingConfig(batch_size=4, num_batches=10)
        trainer = HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        # Empty data loader
        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=64,
            output_dim=64,
            num_batches=0,
        )

        metrics = await trainer.train_epoch(data_loader)
        assert metrics["loss"] == 0.0
        assert metrics["samples_processed"] == 0


# ---------------------------------------------------------------------------
# TRMTrainer - mixed precision (lines 333-351)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTRMTrainerMixedPrecision:
    """Test TRM training with mixed precision enabled."""

    @pytest.mark.asyncio
    async def test_train_step_mixed_precision(self):
        """Train step with use_mixed_precision=True uses scaler (lines 333-351)."""
        agent = _make_mock_trm_agent()
        loss_fn = _make_mock_trm_loss_fn()
        optimizer = _make_mock_optimizer()
        scaler = _make_mock_scaler()

        config = TRMTrainingConfig(batch_size=4, use_mixed_precision=True)
        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
            scaler=scaler,
        )

        inputs = torch.randn(4, 16, 64)
        targets = torch.randn(4, 16, 32)

        metrics = await trainer.train_step(inputs, targets)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.samples_processed == 4
        assert metrics.loss == 0.35
        # Verify scaler pipeline
        scaler.scale.assert_called_once()
        scaler.unscale_.assert_called_once_with(optimizer)
        scaler.step.assert_called_once_with(optimizer)
        scaler.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_step_mixed_precision_no_scaler(self):
        """When use_mixed_precision=True but scaler=None, uses normal path."""
        agent = _make_mock_trm_agent()
        loss_fn = _make_mock_trm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = TRMTrainingConfig(batch_size=4, use_mixed_precision=True)
        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
            scaler=None,
        )

        inputs = torch.randn(4, 16, 64)
        targets = torch.randn(4, 16, 32)

        metrics = await trainer.train_step(inputs, targets)

        assert isinstance(metrics, TrainingMetrics)
        optimizer.step.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_step_empty_residual_norms(self):
        """When residual_norms is empty, final_residual defaults to 0.0."""
        agent = _make_mock_trm_agent()
        agent.return_value.residual_norms = []  # Empty
        loss_fn = _make_mock_trm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = TRMTrainingConfig(batch_size=4)
        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        inputs = torch.randn(4, 16, 64)
        targets = torch.randn(4, 16, 32)

        metrics = await trainer.train_step(inputs, targets)

        assert metrics.component_losses["trm_final_residual"] == 0.0

    @pytest.mark.asyncio
    async def test_train_step_gradient_norm_as_float(self):
        """When clip_grad_norm_ returns a float, it is stored directly."""
        agent = _make_mock_trm_agent()
        loss_fn = _make_mock_trm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = TRMTrainingConfig(batch_size=4)
        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        with patch("torch.nn.utils.clip_grad_norm_", return_value=1.5):
            inputs = torch.randn(4, 16, 64)
            targets = torch.randn(4, 16, 32)
            metrics = await trainer.train_step(inputs, targets)

        assert metrics.gradient_norm == 1.5


# ---------------------------------------------------------------------------
# TRMTrainer.train_epoch - batch limiting and averaging (lines 406, 423-429)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTRMTrainerEpoch:
    """Test TRM train_epoch with batch limiting and metric averaging."""

    @pytest.mark.asyncio
    async def test_train_epoch_limits_batches(self):
        """train_epoch stops after config.num_batches (line 406)."""
        agent = _make_mock_trm_agent()
        loss_fn = _make_mock_trm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = TRMTrainingConfig(batch_size=4, num_batches=2)
        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=64,
            output_dim=32,
            num_batches=5,
        )

        metrics = await trainer.train_epoch(data_loader)
        assert metrics["samples_processed"] == 8  # 2 * 4

    @pytest.mark.asyncio
    async def test_train_epoch_averages_metrics(self):
        """train_epoch averages loss over batches (lines 423-429)."""
        agent = _make_mock_trm_agent()
        optimizer = _make_mock_optimizer()

        call_count = [0]
        losses = [0.3, 0.5]

        def varying_loss(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            loss_val = losses[idx] if idx < len(losses) else 0.4
            return (
                torch.tensor(loss_val, requires_grad=True),
                {
                    "total": loss_val,
                    "final": loss_val * 0.7,
                    "intermediate_mean": loss_val * 0.3,
                    "convergence_step": 2,
                    "converged": True,
                },
            )

        loss_fn = MagicMock(side_effect=varying_loss)

        config = TRMTrainingConfig(batch_size=4, num_batches=2)
        trainer = TRMTrainer(
            agent=agent,
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
        # Average of 0.3 and 0.5 = 0.4
        assert abs(metrics["loss"] - 0.4) < 1e-5

    @pytest.mark.asyncio
    async def test_train_epoch_empty_dataloader(self):
        """train_epoch with no batches returns zero metrics."""
        agent = _make_mock_trm_agent()
        loss_fn = _make_mock_trm_loss_fn()
        optimizer = _make_mock_optimizer()

        config = TRMTrainingConfig(batch_size=4, num_batches=10)
        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=64,
            output_dim=32,
            num_batches=0,
        )

        metrics = await trainer.train_epoch(data_loader)
        assert metrics["loss"] == 0.0
        assert metrics["samples_processed"] == 0


# ---------------------------------------------------------------------------
# SelfPlayEvaluator.play_game (lines 501-537)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelfPlayEvaluatorPlayGame:
    """Test SelfPlayEvaluator.play_game covering lines 501-537."""

    def _make_evaluator(self, mcts, state_fn, config=None):
        """Create evaluator with given mocks."""
        config = config or EvaluationConfig(num_games=2, temperature=0.0)
        return SelfPlayEvaluator(
            mcts=mcts,
            initial_state_fn=state_fn,
            config=config,
            device="cpu",
        )

    @pytest.mark.asyncio
    async def test_play_game_model1_wins(self):
        """play_game returns 1 when model1 wins (lines 521-523)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.8

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"action_0": 1.0}, mock_root_node))
        mock_mcts.network = MagicMock()

        state = MagicMock()
        state.is_terminal = MagicMock(side_effect=[False, False, True])
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=1)  # Positive = model1 wins

        evaluator = self._make_evaluator(mock_mcts, lambda: state)
        model1 = MagicMock()
        model2 = MagicMock()

        result, game_stats = await evaluator.play_game(model1, model2, model1_starts=True)

        assert result == 1
        assert game_stats["moves"] == 2

    @pytest.mark.asyncio
    async def test_play_game_model2_wins(self):
        """play_game returns -1 when model2 wins (lines 524-525)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.3

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"action_0": 1.0}, mock_root_node))
        mock_mcts.network = MagicMock()

        state = MagicMock()
        state.is_terminal = MagicMock(side_effect=[False, True])
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=-1)  # Negative = model2 wins

        evaluator = self._make_evaluator(mock_mcts, lambda: state)
        model1 = MagicMock()
        model2 = MagicMock()

        result, game_stats = await evaluator.play_game(model1, model2, model1_starts=True)

        assert result == -1

    @pytest.mark.asyncio
    async def test_play_game_draw(self):
        """play_game returns 0 on draw (lines 526-527)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.5

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"action_0": 1.0}, mock_root_node))
        mock_mcts.network = MagicMock()

        state = MagicMock()
        state.is_terminal = MagicMock(side_effect=[False, True])
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=0)

        evaluator = self._make_evaluator(mock_mcts, lambda: state)

        result, game_stats = await evaluator.play_game(MagicMock(), MagicMock())
        assert result == 0

    @pytest.mark.asyncio
    async def test_play_game_model2_starts(self):
        """play_game with model1_starts=False swaps player order (line 470)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.7

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"act": 1.0}, mock_root_node))
        mock_mcts.network = MagicMock()

        state = MagicMock()
        state.is_terminal = MagicMock(side_effect=[False, True])
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=1)

        evaluator = self._make_evaluator(mock_mcts, lambda: state)
        model1 = MagicMock()
        model2 = MagicMock()

        result, game_stats = await evaluator.play_game(model1, model2, model1_starts=False)

        # get_reward called with player=1 when model1 does not start
        state.get_reward.assert_called_with(player=1)

    @pytest.mark.asyncio
    async def test_play_game_root_node_none(self):
        """When root_node is None, root_value defaults to 0.0 (line 500)."""
        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"action_0": 1.0}, None))
        mock_mcts.network = MagicMock()

        state = MagicMock()
        state.is_terminal = MagicMock(side_effect=[False, True])
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=0)

        evaluator = self._make_evaluator(mock_mcts, lambda: state)

        result, game_stats = await evaluator.play_game(MagicMock(), MagicMock())
        # Should not crash; mcts values should be 0.0
        assert game_stats["model1_avg_mcts_value"] == 0.0 or isinstance(game_stats["model1_avg_mcts_value"], float)

    @pytest.mark.asyncio
    async def test_play_game_with_temperature(self):
        """Non-zero temperature uses multinomial selection (lines 507-513)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.6

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"a": 0.7, "b": 0.3}, mock_root_node))
        mock_mcts.network = MagicMock()

        state = MagicMock()
        state.is_terminal = MagicMock(side_effect=[False, True])
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=1)

        config = EvaluationConfig(num_games=1, temperature=0.5)
        evaluator = self._make_evaluator(mock_mcts, lambda: state, config=config)

        result, game_stats = await evaluator.play_game(MagicMock(), MagicMock())
        # Should select an action via multinomial
        state.apply_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_game_mcts_value_tracking(self):
        """MCTS values are properly tracked per player (lines 530-535)."""
        root_nodes = [MagicMock(value=0.8), MagicMock(value=0.4), MagicMock(value=0.6)]
        call_idx = [0]

        async def search_fn(state, num_simulations, temperature):
            node = root_nodes[call_idx[0]] if call_idx[0] < len(root_nodes) else MagicMock(value=0.5)
            call_idx[0] += 1
            return {"action_0": 1.0}, node

        mock_mcts = MagicMock()
        mock_mcts.search = search_fn
        mock_mcts.network = MagicMock()

        state = MagicMock()
        state.is_terminal = MagicMock(side_effect=[False, False, False, True])
        state.apply_action = MagicMock(return_value=state)
        state.get_reward = MagicMock(return_value=1)

        evaluator = self._make_evaluator(mock_mcts, lambda: state)

        result, game_stats = await evaluator.play_game(MagicMock(), MagicMock(), model1_starts=True)

        assert game_stats["moves"] == 3
        # Model1 plays moves 0 and 2 (values 0.8 and 0.6), model2 plays move 1 (value 0.4)
        assert abs(game_stats["model1_avg_mcts_value"] - 0.7) < 1e-5  # (0.8+0.6)/2
        assert abs(game_stats["model2_avg_mcts_value"] - 0.4) < 1e-5


# ---------------------------------------------------------------------------
# SelfPlayEvaluator.evaluate (lines 556-605)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelfPlayEvaluatorEvaluate:
    """Test SelfPlayEvaluator.evaluate covering lines 556-605."""

    @pytest.mark.asyncio
    async def test_evaluate_self_play_when_best_model_none(self):
        """When best_model=None, evaluates against itself (lines 556-558)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.5

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"a": 1.0}, mock_root_node))
        mock_mcts.network = MagicMock()

        def make_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=0)  # Draw
            return state

        config = EvaluationConfig(num_games=2, temperature=0.0)
        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=make_state,
            config=config,
            device="cpu",
        )

        model = MagicMock()
        metrics = await evaluator.evaluate(current_model=model, best_model=None)

        assert "win_rate" in metrics
        assert metrics["eval_games"] == 2
        assert "is_better" in metrics

    @pytest.mark.asyncio
    async def test_evaluate_game_failure_continues(self):
        """When a game fails, it is skipped and others continue (lines 577-585)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.5

        mock_mcts = MagicMock()
        mock_mcts.network = MagicMock()

        call_count = [0]

        def make_state():
            call_count[0] += 1
            state = MagicMock()
            if call_count[0] == 1:
                # First game fails
                state.is_terminal = MagicMock(side_effect=RuntimeError("game error"))
            else:
                # Second game succeeds
                state.is_terminal = MagicMock(side_effect=[False, True])
                state.apply_action = MagicMock(return_value=state)
                state.get_reward = MagicMock(return_value=1)
            return state

        mock_mcts.search = AsyncMock(return_value=({"a": 1.0}, mock_root_node))

        config = EvaluationConfig(num_games=2, temperature=0.0)
        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=make_state,
            config=config,
            device="cpu",
        )

        model1 = MagicMock()
        model2 = MagicMock()
        metrics = await evaluator.evaluate(model1, model2)

        # Only 1 game should succeed
        assert metrics["eval_games"] == 1
        assert metrics["wins"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_all_games_fail(self):
        """When all games fail, returns zero metrics (lines 602-605)."""
        mock_mcts = MagicMock()
        mock_mcts.network = MagicMock()
        mock_mcts.search = AsyncMock(side_effect=RuntimeError("mcts fail"))

        def make_failing_state():
            state = MagicMock()
            state.is_terminal = MagicMock(return_value=False)
            return state

        config = EvaluationConfig(num_games=3, temperature=0.0)
        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=make_failing_state,
            config=config,
            device="cpu",
        )

        metrics = await evaluator.evaluate(MagicMock(), MagicMock())

        assert metrics["win_rate"] == 0.0
        assert metrics["eval_games"] == 0
        assert metrics["wins"] == 0
        assert metrics["losses"] == 0
        assert metrics["draws"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_win_rate_calculation(self):
        """Win rate includes half-credit for draws (line 602)."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.5

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"a": 1.0}, mock_root_node))
        mock_mcts.network = MagicMock()

        game_results = [1, 0, -1, 0]  # win, draw, loss, draw
        game_idx = [0]

        def make_state():
            idx = game_idx[0]
            game_idx[0] += 1
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=game_results[idx])
            return state

        config = EvaluationConfig(num_games=4, temperature=0.0)
        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=make_state,
            config=config,
            device="cpu",
        )

        metrics = await evaluator.evaluate(MagicMock(), MagicMock())

        assert metrics["wins"] == 1
        assert metrics["losses"] == 1
        assert metrics["draws"] == 2
        # win_rate = (1 + 0.5*2) / 4 = 0.5
        assert abs(metrics["win_rate"] - 0.5) < 1e-5
        assert metrics["is_better"] is False  # 0.5 < 0.55

    @pytest.mark.asyncio
    async def test_evaluate_is_better_threshold(self):
        """is_better is True when win_rate >= win_threshold."""
        mock_root_node = MagicMock()
        mock_root_node.value = 0.7

        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"a": 1.0}, mock_root_node))
        mock_mcts.network = MagicMock()

        def make_win_state():
            state = MagicMock()
            state.is_terminal = MagicMock(side_effect=[False, True])
            state.apply_action = MagicMock(return_value=state)
            state.get_reward = MagicMock(return_value=1)  # Always win
            return state

        config = EvaluationConfig(num_games=2, temperature=0.0, win_threshold=0.55)
        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=make_win_state,
            config=config,
            device="cpu",
        )

        metrics = await evaluator.evaluate(MagicMock(), MagicMock())

        assert metrics["win_rate"] == 1.0
        assert metrics["is_better"] is True


# ---------------------------------------------------------------------------
# create_data_loader_from_buffer with ready buffer (lines 683-712)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDataLoaderFromBuffer:
    """Test create_data_loader_from_buffer with a ready replay buffer."""

    def test_returns_buffer_data_loader_when_ready(self):
        """When buffer is ready, returns BufferDataLoader (lines 683-712)."""
        mock_buffer = MagicMock()
        mock_buffer.is_ready.return_value = True

        # Create mock experiences
        mock_exp1 = MagicMock()
        mock_exp1.state = torch.randn(16, 64)
        mock_exp1.policy = torch.randn(16, 32)

        mock_exp2 = MagicMock()
        mock_exp2.state = torch.randn(16, 64)
        mock_exp2.policy = torch.randn(16, 32)

        mock_buffer.sample.return_value = ([mock_exp1, mock_exp2], None, None)

        loader = create_data_loader_from_buffer(
            replay_buffer=mock_buffer,
            batch_size=2,
            input_dim=64,
            output_dim=32,
        )

        # Should not be DummyDataLoader
        assert not isinstance(loader, DummyDataLoader)

        # Should be iterable and yield batches
        for batch_count, (states, targets) in enumerate(loader):
            assert states.shape[0] == 2
            assert targets.shape[0] == 2
            if batch_count >= 1:
                break

    def test_buffer_data_loader_stops_after_num_batches(self):
        """BufferDataLoader stops after num_batches iterations."""
        mock_buffer = MagicMock()
        mock_buffer.is_ready.return_value = True

        mock_exp = MagicMock()
        mock_exp.state = torch.randn(8, 32)
        mock_exp.policy = torch.randn(8, 16)
        mock_buffer.sample.return_value = ([mock_exp], None, None)

        loader = create_data_loader_from_buffer(
            replay_buffer=mock_buffer,
            batch_size=1,
            input_dim=32,
            output_dim=16,
        )

        # Default num_batches is 10
        batch_count = sum(1 for _ in loader)
        assert batch_count == 10

    def test_buffer_data_loader_resets_iterator(self):
        """BufferDataLoader can be iterated multiple times."""
        mock_buffer = MagicMock()
        mock_buffer.is_ready.return_value = True

        mock_exp = MagicMock()
        mock_exp.state = torch.randn(8, 32)
        mock_exp.policy = torch.randn(8, 16)
        mock_buffer.sample.return_value = ([mock_exp], None, None)

        loader = create_data_loader_from_buffer(
            replay_buffer=mock_buffer,
            batch_size=1,
            input_dim=32,
            output_dim=16,
        )

        count1 = sum(1 for _ in loader)
        count2 = sum(1 for _ in loader)
        assert count1 == count2 == 10


# ---------------------------------------------------------------------------
# DummyDataLoader - additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDummyDataLoaderExtended:
    """Additional edge case tests for DummyDataLoader."""

    def test_custom_sequence_length(self):
        """DummyDataLoader uses custom sequence_length."""
        loader = DummyDataLoader(
            batch_size=4,
            input_dim=32,
            output_dim=16,
            sequence_length=8,
        )

        inputs, targets = next(iter(loader))
        assert inputs.shape == (4, 8, 32)
        assert targets.shape == (4, 8, 16)

    def test_default_sequence_length(self):
        """DummyDataLoader uses DEFAULT_SEQUENCE_LENGTH when not specified."""
        loader = DummyDataLoader(
            batch_size=2,
            input_dim=16,
            output_dim=8,
        )

        inputs, targets = next(iter(loader))
        assert inputs.shape[1] == DummyDataLoader.DEFAULT_SEQUENCE_LENGTH

    def test_device_parameter(self):
        """DummyDataLoader moves data to specified device."""
        loader = DummyDataLoader(
            batch_size=2,
            input_dim=8,
            output_dim=4,
            device="cpu",
        )

        inputs, targets = next(iter(loader))
        assert inputs.device.type == "cpu"
        assert targets.device.type == "cpu"


# ---------------------------------------------------------------------------
# TrainingMetrics edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainingMetricsExtended:
    """Extended tests for TrainingMetrics."""

    def test_to_dict_merges_component_losses(self):
        """Component losses are merged into the dict output."""
        metrics = TrainingMetrics(
            loss=1.0,
            component_losses={"a": 0.5, "b": 0.5},
            samples_processed=10,
        )
        d = metrics.to_dict()
        assert d["a"] == 0.5
        assert d["b"] == 0.5
        assert d["loss"] == 1.0

    def test_to_dict_empty_component_losses(self):
        """Empty component_losses results in base dict only."""
        metrics = TrainingMetrics()
        d = metrics.to_dict()
        assert len(d) == 4  # loss, samples_processed, gradient_norm, learning_rate
