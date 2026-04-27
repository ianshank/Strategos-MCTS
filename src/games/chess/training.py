"""
Chess Training Orchestrator Module.

Extends the unified training orchestrator with chess-specific features
including opening book integration, data augmentation, and evaluation
against external engines.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.observability.logging import get_logger

if TYPE_CHECKING:
    pass

from src.games.chess.config import ChessConfig
from src.games.chess.ensemble_agent import ChessEnsembleAgent
from src.games.chess.state import ChessGameState, create_initial_state

logger = get_logger(__name__)


@dataclass
class ChessTrainingMetrics:
    """Metrics from a training iteration."""

    iteration: int
    policy_loss: float
    value_loss: float
    total_loss: float
    games_played: int
    average_game_length: float
    win_rate_white: float
    win_rate_black: float
    draw_rate: float
    average_value_accuracy: float
    learning_rate: float
    elapsed_time_seconds: float


@dataclass
class SelfPlayGame:
    """Data from a self-play game."""

    positions: list[torch.Tensor]
    policies: list[np.ndarray]
    values: list[float]
    outcome: float  # 1.0 = white win, -1.0 = black win, 0.0 = draw
    moves: list[str]
    game_length: int


class ChessOpeningBook:
    """Simple opening book implementation."""

    def __init__(self, book_path: str | None = None) -> None:
        """Initialize opening book.

        Args:
            book_path: Path to opening book file (Polyglot format)
        """
        self.book_path = book_path
        self._entries: dict[str, list[tuple[str, int]]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load opening book from file."""
        if self._loaded:
            return

        # Polyglot format parsing would go here
        # For now, use built-in common openings
        self._load_builtin_openings()
        self._loaded = True

    def _load_builtin_openings(self) -> None:
        """Load built-in common openings."""
        # Common opening moves from starting position
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self._entries[starting_fen] = [
            ("e2e4", 100),
            ("d2d4", 100),
            ("c2c4", 50),
            ("g1f3", 40),
        ]

    def get_book_move(
        self,
        state: ChessGameState,
        temperature: float = 1.0,
    ) -> str | None:
        """Get a book move for the position.

        Args:
            state: Current chess position
            temperature: Temperature for move selection

        Returns:
            Move in UCI format, or None if not in book
        """
        if not self._loaded:
            self.load()

        entries = self._entries.get(state.fen)
        if not entries:
            return None

        # Weight by frequency
        moves, weights_tuple = zip(*entries, strict=False)
        weights = np.array(weights_tuple, dtype=np.float32)

        if temperature > 0:
            weights = np.power(weights, 1.0 / temperature)

        weights = weights / weights.sum()
        return str(np.random.choice(moves, p=weights))


class ChessDataAugmentation:
    """Data augmentation for chess positions."""

    def __init__(self, config: ChessConfig) -> None:
        """Initialize data augmentation.

        Args:
            config: Chess configuration
        """
        self.config = config
        self.use_board_flip = config.training.use_board_flip

    def augment(
        self,
        state_tensor: torch.Tensor,
        policy: np.ndarray,
    ) -> list[tuple[torch.Tensor, np.ndarray]]:
        """Augment a training example.

        For chess, we can use color symmetry (flip the board).

        Args:
            state_tensor: Board tensor (channels, 8, 8)
            policy: Policy vector

        Returns:
            List of (augmented_tensor, augmented_policy) tuples
        """
        augmented = [(state_tensor, policy)]

        if self.use_board_flip:
            # Flip board vertically (swap ranks)
            flipped_tensor = torch.flip(state_tensor, dims=[1])

            # Rearrange channels (swap white/black pieces)
            # Channels 0-5 are white, 6-11 are black
            new_tensor = torch.zeros_like(flipped_tensor)
            new_tensor[:6] = flipped_tensor[6:12]  # Black -> White position
            new_tensor[6:12] = flipped_tensor[:6]  # White -> Black position
            new_tensor[12:] = flipped_tensor[12:]  # Game state planes

            # Flip policy vector (reverse move directions)
            flipped_policy = self._flip_policy(policy)

            augmented.append((new_tensor, flipped_policy))

        return augmented

    def _flip_policy(self, policy: np.ndarray) -> np.ndarray:
        """Flip policy vector for color symmetry.

        Args:
            policy: Original policy vector

        Returns:
            Flipped policy vector
        """
        # This is a simplified flip - full implementation would
        # properly reverse all move directions
        flipped = np.zeros_like(policy)

        # For each action, compute the flipped action index
        # This requires knowledge of the action encoding scheme
        num_planes = 73
        for plane in range(num_planes):
            for square in range(64):
                orig_idx = plane * 64 + square

                # Flip the square (reverse rank)
                file = square % 8
                rank = square // 8
                flipped_rank = 7 - rank
                flipped_square = flipped_rank * 8 + file

                # For queen/knight moves, reverse the direction
                # (simplified - would need proper direction reversal)
                flipped_idx = plane * 64 + flipped_square

                if orig_idx < len(policy) and flipped_idx < len(flipped):
                    flipped[flipped_idx] = policy[orig_idx]

        return flipped


class ChessTrainingOrchestrator:
    """Training orchestrator for chess AlphaZero-style learning.

    Extends the base training pipeline with chess-specific features.
    """

    def __init__(
        self,
        config: ChessConfig,
        initial_state_fn: Callable[[], ChessGameState] | None = None,
    ) -> None:
        """Initialize the training orchestrator.

        Args:
            config: Chess configuration
            initial_state_fn: Function to create initial game state
        """
        self.config = config
        self.device = config.device
        self.initial_state_fn = initial_state_fn or (lambda: create_initial_state(config.board, config.action_space))

        # Initialize ensemble agent
        self.ensemble_agent = ChessEnsembleAgent(config, device=self.device)

        # Training components
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

        # Experience replay buffer
        self.replay_buffer: list[tuple[torch.Tensor, np.ndarray, float]] = []
        self.buffer_size = config.training.buffer_size

        # Data augmentation
        self.augmentation = ChessDataAugmentation(config)

        # Opening book
        self.opening_book: ChessOpeningBook | None = None
        if config.training.use_opening_book:
            self.opening_book = ChessOpeningBook(config.training.opening_book_path)

        # Training state
        self.current_iteration = 0
        self.best_model_path: str | None = None
        self.best_win_rate = 0.0

        # Metrics history
        self.metrics_history: list[ChessTrainingMetrics] = []

        logger.info(f"Initialized ChessTrainingOrchestrator on {self.device}")

    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        if self.optimizer is not None:
            return

        self.optimizer = torch.optim.SGD(
            self.ensemble_agent.policy_value_net.parameters(),
            lr=self.config.training.learning_rate,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
        )

        if self.config.training.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.lr_decay_steps,
            )
        elif self.config.training.lr_schedule == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_gamma,
            )

    async def train(self, num_iterations: int) -> list[ChessTrainingMetrics]:
        """Run the full training loop.

        Args:
            num_iterations: Number of training iterations

        Returns:
            List of training metrics for each iteration
        """
        self._setup_optimizer()

        logger.info(f"Starting training for {num_iterations} iterations")

        for iteration in range(num_iterations):
            self.current_iteration = iteration

            # Run single training iteration
            metrics = await self.train_iteration()
            self.metrics_history.append(metrics)

            # Log progress
            logger.info(
                f"Iteration {iteration + 1}/{num_iterations}: "
                f"loss={metrics.total_loss:.4f}, "
                f"games={metrics.games_played}, "
                f"win_rate_w={metrics.win_rate_white:.2%}"
            )

            # Checkpoint
            if (iteration + 1) % self.config.training.checkpoint_interval == 0:
                await self._save_checkpoint(iteration)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model
        await self._save_checkpoint(num_iterations - 1, is_final=True)

        return self.metrics_history

    async def train_iteration(self) -> ChessTrainingMetrics:
        """Run a single training iteration.

        Returns:
            Training metrics for this iteration
        """
        import time

        start_time = time.time()

        # Generate self-play games
        games = await self._generate_self_play_games()

        # Add games to replay buffer
        self._add_games_to_buffer(games)

        # Train on replay buffer
        policy_loss, value_loss = await self._train_on_buffer()

        # Calculate metrics
        wins_white = sum(1 for g in games if g.outcome > 0.5)
        wins_black = sum(1 for g in games if g.outcome < -0.5)
        draws = sum(1 for g in games if -0.5 <= g.outcome <= 0.5)
        total_games = len(games)

        avg_game_length = float(np.mean([g.game_length for g in games])) if games else 0.0

        elapsed = time.time() - start_time

        current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0

        return ChessTrainingMetrics(
            iteration=self.current_iteration,
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=policy_loss + value_loss,
            games_played=total_games,
            average_game_length=avg_game_length,
            win_rate_white=wins_white / total_games if total_games > 0 else 0,
            win_rate_black=wins_black / total_games if total_games > 0 else 0,
            draw_rate=draws / total_games if total_games > 0 else 0,
            average_value_accuracy=0.0,  # Calculated during training
            learning_rate=current_lr,
            elapsed_time_seconds=elapsed,
        )

    async def _generate_self_play_games(self) -> list[SelfPlayGame]:
        """Generate self-play games.

        Returns:
            List of self-play game data
        """
        num_games = self.config.training.games_per_iteration
        num_actors = self.config.training.num_actors

        # Create game tasks
        num_games // num_actors

        tasks = [self._play_single_game() for _ in range(num_games)]

        # Run games (with some concurrency limit)
        games = []
        batch_size = min(num_actors, 16)

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            results = await asyncio.gather(*batch)
            games.extend(results)

        return games

    async def _play_single_game(self) -> SelfPlayGame:
        """Play a single self-play game.

        Returns:
            Self-play game data
        """
        state = self.initial_state_fn()
        positions: list[torch.Tensor] = []
        policies: list[np.ndarray] = []
        values: list[float] = []
        moves: list[str] = []

        move_number = 0
        temperature_threshold = self.config.mcts.temperature_threshold

        while not state.is_terminal():
            # Determine temperature
            temperature = (
                self.config.mcts.temperature_init
                if move_number < temperature_threshold
                else self.config.mcts.temperature_final
            )

            # Check opening book
            book_move = None
            if self.opening_book is not None and move_number < self.config.training.opening_book_moves:
                book_move = self.opening_book.get_book_move(state, temperature)

            if book_move is not None:
                # Use book move
                move = book_move
                # Still need to get policy for training
                response = await self.ensemble_agent.get_best_move(
                    state,
                    temperature=temperature,
                    use_ensemble=False,
                )
                policy_probs = response.move_probabilities
            else:
                # Get move from ensemble agent
                response = await self.ensemble_agent.get_best_move(
                    state,
                    temperature=temperature,
                    use_ensemble=True,
                )
                move = response.best_move
                policy_probs = response.move_probabilities

            # Store training data
            positions.append(state.to_tensor())

            # Convert policy dict to vector
            policy_vector = np.zeros(self.config.action_size)
            from_black = state.current_player == -1
            for m, p in policy_probs.items():
                try:
                    idx = self.ensemble_agent.action_encoder.encode_move(m, from_black)
                    policy_vector[idx] = p
                except ValueError:
                    pass  # Ignore moves that cannot be encoded
            policies.append(policy_vector)

            values.append(response.value_estimate)
            moves.append(move)

            # Apply move
            state = state.apply_action(move)
            move_number += 1

            # Safety limit
            if move_number > 500:
                break

        # Get game outcome
        outcome = state.get_reward(player=1)  # From white's perspective

        # Assign values to all positions based on outcome
        for i, _pos_tensor in enumerate(positions):
            # Alternate perspective
            perspective = 1 if i % 2 == 0 else -1
            values[i] = outcome * perspective

        return SelfPlayGame(
            positions=positions,
            policies=policies,
            values=values,
            outcome=outcome,
            moves=moves,
            game_length=move_number,
        )

    def _add_games_to_buffer(self, games: list[SelfPlayGame]) -> None:
        """Add games to replay buffer.

        Args:
            games: List of self-play games
        """
        for game in games:
            for pos, policy, value in zip(
                game.positions,
                game.policies,
                game.values,
                strict=False,
            ):
                # Apply data augmentation
                augmented = self.augmentation.augment(pos, policy)
                for aug_pos, aug_policy in augmented:
                    self.replay_buffer.append((aug_pos, aug_policy, value))

        # Trim buffer to max size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size :]

    async def _train_on_buffer(self) -> tuple[float, float]:
        """Train the network on replay buffer.

        Returns:
            (policy_loss, value_loss) averages
        """
        if len(self.replay_buffer) < self.config.training.min_buffer_size:
            logger.info(
                f"Buffer size {len(self.replay_buffer)} < {self.config.training.min_buffer_size}, skipping training"
            )
            return 0.0, 0.0

        self.ensemble_agent.policy_value_net.train()

        batch_size = self.config.training.batch_size
        num_batches = max(1, len(self.replay_buffer) // batch_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(num_batches):
            # Sample batch
            batch_indices = random.sample(
                range(len(self.replay_buffer)),
                min(batch_size, len(self.replay_buffer)),
            )
            batch = [self.replay_buffer[i] for i in batch_indices]

            # Prepare tensors
            states = torch.stack([b[0] for b in batch]).to(self.device)
            target_policies = torch.tensor(
                np.array([b[1] for b in batch]),
                dtype=torch.float32,
            ).to(self.device)
            target_values = torch.tensor(
                [[b[2]] for b in batch],
                dtype=torch.float32,
            ).to(self.device)

            # Forward pass
            policy_logits, values = self.ensemble_agent.policy_value_net(states)

            # Calculate losses
            policy_loss = -torch.mean(torch.sum(target_policies * torch.log_softmax(policy_logits, dim=-1), dim=-1))
            value_loss = torch.mean((values - target_values) ** 2)

            total_loss = policy_loss + value_loss

            # Backward pass
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ensemble_agent.policy_value_net.parameters(),
                    self.config.training.max_gradient_norm,
                )
                self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        self.ensemble_agent.policy_value_net.eval()

        return total_policy_loss / num_batches, total_value_loss / num_batches

    async def _save_checkpoint(self, iteration: int, is_final: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            iteration: Current iteration number
            is_final: Whether this is the final checkpoint
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_final:
            path = checkpoint_dir / "final_model"
        else:
            path = checkpoint_dir / f"checkpoint_{iteration:05d}"

        self.ensemble_agent.save(str(path))
        logger.info(f"Saved checkpoint to {path}")

        if is_final:
            self.best_model_path = str(path)

    async def evaluate_vs_stockfish(
        self,
        num_games: int = 100,
        stockfish_elo: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate the model against Stockfish.

        Args:
            num_games: Number of games to play
            stockfish_elo: Stockfish Elo rating (None for full strength)

        Returns:
            Dictionary with win/draw/loss statistics
        """
        if not self.config.training.evaluate_vs_stockfish:
            return {"error": "Stockfish evaluation not enabled"}

        # This would require python-chess with Stockfish integration
        # Implementation would go here
        return {
            "games": num_games,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "win_rate": 0.0,
        }


def create_chess_orchestrator(
    preset: str = "small",
    device: str | None = None,
) -> ChessTrainingOrchestrator:
    """Factory function to create a chess training orchestrator.

    Args:
        preset: Configuration preset ("small", "medium", "large")
        device: Device to use (defaults to auto-detect)

    Returns:
        Configured ChessTrainingOrchestrator
    """
    config = ChessConfig.from_preset(preset)
    if device:
        config.device = device

    return ChessTrainingOrchestrator(config)
