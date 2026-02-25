"""
Agent Training Module for HRM and TRM.

Implements proper training loops for:
- HRM Agent with Adaptive Computation Time
- TRM Agent with Deep Supervision
- Real evaluation through self-play

Best Practices 2025:
- No hardcoded values (all from config)
- Comprehensive metrics tracking
- Gradient clipping and numerical stability
- Mixed precision support
- Proper device handling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

if TYPE_CHECKING:
    from ..agents.hrm_agent import HRMAgent, HRMLoss
    from ..agents.trm_agent import TRMAgent, TRMLoss
    from ..framework.mcts.neural_mcts import NeuralMCTS

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    loss: float = 0.0
    component_losses: dict[str, float] = field(default_factory=dict)
    samples_processed: int = 0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "loss": self.loss,
            "samples_processed": self.samples_processed,
            "gradient_norm": self.gradient_norm,
            "learning_rate": self.learning_rate,
        }
        result.update(self.component_losses)
        return result


@dataclass
class HRMTrainingConfig:
    """Configuration for HRM training."""

    batch_size: int = 32
    num_batches: int = 10
    gradient_clip_norm: float = 1.0
    ponder_weight: float = 0.01
    consistency_weight: float = 0.1
    use_mixed_precision: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> HRMTrainingConfig:
        """Create from dictionary with defaults."""
        return cls(
            batch_size=config.get("batch_size", 32),
            num_batches=config.get("num_batches", 10),
            gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
            ponder_weight=config.get("ponder_weight", 0.01),
            consistency_weight=config.get("consistency_weight", 0.1),
            use_mixed_precision=config.get("use_mixed_precision", False),
        )


@dataclass
class TRMTrainingConfig:
    """Configuration for TRM training."""

    batch_size: int = 32
    num_batches: int = 10
    gradient_clip_norm: float = 1.0
    supervision_weight_decay: float = 0.5
    use_mixed_precision: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> TRMTrainingConfig:
        """Create from dictionary with defaults."""
        return cls(
            batch_size=config.get("batch_size", 32),
            num_batches=config.get("num_batches", 10),
            gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
            supervision_weight_decay=config.get("supervision_weight_decay", 0.5),
            use_mixed_precision=config.get("use_mixed_precision", False),
        )


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    num_games: int = 20
    temperature: float = 0.0  # Deterministic play for evaluation
    mcts_iterations: int = 100
    win_threshold: float = 0.55  # Minimum win rate to be considered better

    @classmethod
    def from_dict(cls, config: dict) -> EvaluationConfig:
        """Create from dictionary with defaults."""
        return cls(
            num_games=config.get("num_games", 20),
            temperature=config.get("temperature", 0.0),
            mcts_iterations=config.get("mcts_iterations", 100),
            win_threshold=config.get("win_threshold", 0.55),
        )


class HRMTrainer:
    """
    Trainer for Hierarchical Reasoning Model.

    Implements proper training with:
    - Adaptive Computation Time loss
    - Ponder cost regularization
    - Convergence consistency loss
    """

    def __init__(
        self,
        agent: HRMAgent,
        optimizer: torch.optim.Optimizer,
        loss_fn: HRMLoss,
        config: HRMTrainingConfig,
        device: str = "cpu",
        scaler: Any | None = None,
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.scaler = scaler

        # Task-specific loss (e.g., for decomposition quality)
        self.task_loss_fn = nn.MSELoss()

    async def train_step(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> TrainingMetrics:
        """
        Execute single training step.

        Args:
            states: Input states [batch, seq, h_dim]
            targets: Target outputs [batch, seq, h_dim]

        Returns:
            TrainingMetrics with loss and component values
        """
        self.agent.train()
        self.optimizer.zero_grad()

        states = states.to(self.device)
        targets = targets.to(self.device)

        metrics = TrainingMetrics()

        if self.config.use_mixed_precision and self.scaler is not None:
            with autocast():
                # Forward pass
                hrm_output = self.agent(states, return_decomposition=True)
                predictions = hrm_output.final_state

                # Compute combined loss
                loss, loss_dict = self.loss_fn(
                    hrm_output=hrm_output,
                    predictions=predictions,
                    targets=targets,
                    task_loss_fn=self.task_loss_fn,
                )

            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.gradient_clip_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward pass
            hrm_output = self.agent(states, return_decomposition=True)
            predictions = hrm_output.final_state

            # Compute combined loss
            loss, loss_dict = self.loss_fn(
                hrm_output=hrm_output,
                predictions=predictions,
                targets=targets,
                task_loss_fn=self.task_loss_fn,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.gradient_clip_norm,
            )

            self.optimizer.step()

        # Collect metrics
        metrics.loss = loss_dict["total"]
        metrics.component_losses = {
            "hrm_task_loss": loss_dict["task"],
            "hrm_ponder_cost": loss_dict["ponder"],
            "hrm_consistency_loss": loss_dict["consistency"],
            "hrm_halt_step": float(loss_dict["halt_step"]),
        }
        metrics.gradient_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        metrics.samples_processed = states.size(0)
        metrics.learning_rate = self.optimizer.param_groups[0]["lr"]

        return metrics

    async def train_epoch(
        self,
        data_loader: Any,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            data_loader: DataLoader providing (states, targets) batches

        Returns:
            Aggregated metrics dictionary
        """
        total_metrics = TrainingMetrics()
        batch_count = 0

        for batch_idx, (states, targets) in enumerate(data_loader):
            if batch_idx >= self.config.num_batches:
                break

            metrics = await self.train_step(states, targets)

            # Aggregate metrics
            total_metrics.loss += metrics.loss
            total_metrics.samples_processed += metrics.samples_processed
            total_metrics.gradient_norm += metrics.gradient_norm

            for key, value in metrics.component_losses.items():
                if key not in total_metrics.component_losses:
                    total_metrics.component_losses[key] = 0.0
                total_metrics.component_losses[key] += value

            batch_count += 1

        # Average metrics
        if batch_count > 0:
            total_metrics.loss /= batch_count
            total_metrics.gradient_norm /= batch_count
            for key in total_metrics.component_losses:
                total_metrics.component_losses[key] /= batch_count

        return total_metrics.to_dict()


class TRMTrainer:
    """
    Trainer for Tiny Recursive Model.

    Implements proper training with:
    - Deep supervision at all recursion levels
    - Convergence monitoring
    - Residual norm tracking
    """

    def __init__(
        self,
        agent: TRMAgent,
        optimizer: torch.optim.Optimizer,
        loss_fn: TRMLoss,
        config: TRMTrainingConfig,
        device: str = "cpu",
        scaler: Any | None = None,
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.scaler = scaler

    async def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> TrainingMetrics:
        """
        Execute single training step.

        Args:
            inputs: Input tensor [batch, ..., latent_dim]
            targets: Target outputs [batch, ..., output_dim]

        Returns:
            TrainingMetrics with loss and component values
        """
        self.agent.train()
        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        metrics = TrainingMetrics()

        if self.config.use_mixed_precision and self.scaler is not None:
            with autocast():
                # Forward pass
                trm_output = self.agent(inputs, check_convergence=True)

                # Compute deep supervision loss
                loss, loss_dict = self.loss_fn(trm_output, targets)

            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.gradient_clip_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward pass
            trm_output = self.agent(inputs, check_convergence=True)

            # Compute deep supervision loss
            loss, loss_dict = self.loss_fn(trm_output, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.gradient_clip_norm,
            )

            self.optimizer.step()

        # Get final residual from output
        final_residual = trm_output.residual_norms[-1] if trm_output.residual_norms else 0.0

        # Collect metrics
        metrics.loss = loss_dict["total"]
        metrics.component_losses = {
            "trm_final_loss": loss_dict["final"],
            "trm_intermediate_mean": loss_dict["intermediate_mean"],
            "trm_convergence_step": float(loss_dict["convergence_step"]),
            "trm_converged": float(loss_dict["converged"]),
            "trm_final_residual": final_residual,
        }
        metrics.gradient_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        metrics.samples_processed = inputs.size(0)
        metrics.learning_rate = self.optimizer.param_groups[0]["lr"]

        return metrics

    async def train_epoch(
        self,
        data_loader: Any,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            data_loader: DataLoader providing (inputs, targets) batches

        Returns:
            Aggregated metrics dictionary
        """
        total_metrics = TrainingMetrics()
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx >= self.config.num_batches:
                break

            metrics = await self.train_step(inputs, targets)

            # Aggregate metrics
            total_metrics.loss += metrics.loss
            total_metrics.samples_processed += metrics.samples_processed
            total_metrics.gradient_norm += metrics.gradient_norm

            for key, value in metrics.component_losses.items():
                if key not in total_metrics.component_losses:
                    total_metrics.component_losses[key] = 0.0
                total_metrics.component_losses[key] += value

            batch_count += 1

        # Average metrics
        if batch_count > 0:
            total_metrics.loss /= batch_count
            total_metrics.gradient_norm /= batch_count
            for key in total_metrics.component_losses:
                total_metrics.component_losses[key] /= batch_count

        return total_metrics.to_dict()


class SelfPlayEvaluator:
    """
    Evaluator using self-play for model comparison.

    Implements arena-style evaluation between current and previous best models.
    """

    def __init__(
        self,
        mcts: NeuralMCTS,
        initial_state_fn: Any,
        config: EvaluationConfig,
        device: str = "cpu",
    ):
        self.mcts = mcts
        self.initial_state_fn = initial_state_fn
        self.config = config
        self.device = device

    async def play_game(
        self,
        model1: nn.Module,
        model2: nn.Module,
        model1_starts: bool = True,
    ) -> tuple[int, dict[str, Any]]:
        """
        Play a single game between two models.

        Args:
            model1: First model (current candidate)
            model2: Second model (previous best / opponent)
            model1_starts: Whether model1 plays first

        Returns:
            result: 1 if model1 wins, -1 if model2 wins, 0 for draw
            game_stats: Dictionary with game statistics
        """
        state = self.initial_state_fn()
        models = [model1, model2] if model1_starts else [model2, model1]
        current_model_idx = 0
        move_count = 0

        game_stats = {
            "moves": 0,
            "model1_avg_mcts_value": 0.0,
            "model2_avg_mcts_value": 0.0,
        }

        mcts_values: list[list[float]] = [[], []]

        while not state.is_terminal():
            current_model = models[current_model_idx]

            # Temporarily set the model for MCTS
            original_model = self.mcts.policy_value_network
            self.mcts.policy_value_network = current_model

            # Run MCTS search
            action_probs, root_value = await self.mcts.search(
                state,
                num_iterations=self.config.mcts_iterations,
                temperature=self.config.temperature,
            )

            # Restore original model
            self.mcts.policy_value_network = original_model

            # Track MCTS values
            mcts_values[current_model_idx].append(root_value)

            # Select action (deterministic for evaluation)
            if self.config.temperature == 0:
                action = max(action_probs, key=action_probs.get)
            else:
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action_idx = torch.multinomial(
                    torch.tensor(probs),
                    num_samples=1,
                ).item()
                action = actions[action_idx]

            # Apply action
            state = state.apply_action(action)
            move_count += 1
            current_model_idx = 1 - current_model_idx  # Switch player

        # Determine result
        reward = state.get_reward(player=0 if model1_starts else 1)
        if reward > 0:
            result = 1  # Model1 wins
        elif reward < 0:
            result = -1  # Model2 wins
        else:
            result = 0  # Draw

        # Calculate average MCTS values
        model1_values = mcts_values[0] if model1_starts else mcts_values[1]
        model2_values = mcts_values[1] if model1_starts else mcts_values[0]

        game_stats["moves"] = move_count
        game_stats["model1_avg_mcts_value"] = sum(model1_values) / len(model1_values) if model1_values else 0.0
        game_stats["model2_avg_mcts_value"] = sum(model2_values) / len(model2_values) if model2_values else 0.0

        return result, game_stats

    async def evaluate(
        self,
        current_model: nn.Module,
        best_model: nn.Module | None = None,
    ) -> dict[str, float]:
        """
        Evaluate current model against previous best.

        Plays multiple games with alternating starting positions.

        Args:
            current_model: Current model to evaluate
            best_model: Previous best model (uses current if None)

        Returns:
            Evaluation metrics including win rate
        """
        if best_model is None:
            # Self-play against itself (sanity check)
            best_model = current_model

        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        avg_mcts_values = []

        # Play games with alternating starting positions
        for game_idx in range(self.config.num_games):
            model1_starts = game_idx % 2 == 0

            try:
                result, game_stats = await self.play_game(
                    current_model,
                    best_model,
                    model1_starts=model1_starts,
                )

                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1

                total_moves += game_stats["moves"]
                avg_mcts_values.append(game_stats["model1_avg_mcts_value"])

            except Exception as e:
                logger.warning(f"Game {game_idx} failed: {e}")
                continue

        # Calculate metrics
        games_played = wins + losses + draws
        if games_played == 0:
            return {
                "win_rate": 0.0,
                "eval_games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
            }

        win_rate = (wins + 0.5 * draws) / games_played
        avg_game_length = total_moves / games_played

        return {
            "win_rate": win_rate,
            "eval_games": games_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_game_length": avg_game_length,
            "avg_mcts_value": (sum(avg_mcts_values) / len(avg_mcts_values) if avg_mcts_values else 0.0),
            "is_better": win_rate >= self.config.win_threshold,
        }


class DummyDataLoader:
    """
    Dummy data loader for generating synthetic training data.

    Used when replay buffer is not ready or for testing.
    """

    # Default sequence length for synthetic data generation
    DEFAULT_SEQUENCE_LENGTH: int = 16

    def __init__(
        self,
        batch_size: int,
        input_dim: int,
        output_dim: int,
        num_batches: int = 10,
        device: str = "cpu",
        sequence_length: int | None = None,
    ):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_batches = num_batches
        self.device = device
        self.sequence_length = sequence_length or self.DEFAULT_SEQUENCE_LENGTH
        self._current = 0

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        if self._current >= self.num_batches:
            raise StopIteration

        self._current += 1

        # Generate synthetic data and move to configured device
        inputs = torch.randn(self.batch_size, self.sequence_length, self.input_dim).to(self.device)
        targets = torch.randn(self.batch_size, self.sequence_length, self.output_dim).to(self.device)

        return inputs, targets


def create_data_loader_from_buffer(
    replay_buffer: Any,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    device: str = "cpu",
) -> Any:
    """
    Create a data loader from replay buffer.

    Falls back to dummy data loader if buffer is not ready.
    """
    if replay_buffer is None or not replay_buffer.is_ready(batch_size):
        logger.warning("Replay buffer not ready, using synthetic data")
        return DummyDataLoader(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
        )

    # Create a generator that samples from replay buffer
    class BufferDataLoader:
        def __init__(self, buffer, batch_size, input_dim, output_dim, num_batches=10):
            self.buffer = buffer
            self.batch_size = batch_size
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_batches = num_batches
            self._current = 0

        def __iter__(self):
            self._current = 0
            return self

        def __next__(self):
            if self._current >= self.num_batches:
                raise StopIteration

            self._current += 1

            # Sample from buffer
            experiences, _, _ = self.buffer.sample(self.batch_size)

            # Convert to tensors
            states = torch.stack([e.state for e in experiences])
            # Use policy as target (simplified)
            targets = torch.stack([e.policy for e in experiences])

            return states, targets

    return BufferDataLoader(
        replay_buffer,
        batch_size,
        input_dim,
        output_dim,
    )
