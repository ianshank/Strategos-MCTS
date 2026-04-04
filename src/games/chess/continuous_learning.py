"""
Continuous Learning Module for Chess.

Implements AlphaZero-style continuous self-play learning where agents
play against each other, learn from outcomes, and continuously improve.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

# Optional torch import for neural network operations
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from src.games.chess.ensemble_agent import ChessEnsembleAgent

from src.games.chess.config import ChessConfig, get_chess_small_config
from src.games.chess.state import ChessGameState

logger = logging.getLogger(__name__)


class GameResult(Enum):
    """Possible game results."""

    WHITE_WIN = "white_win"
    BLACK_WIN = "black_win"
    DRAW = "draw"
    IN_PROGRESS = "in_progress"
    TIMEOUT = "timeout"


@dataclass
class GameRecord:
    """Record of a completed game."""

    game_id: str
    white_agent: str
    black_agent: str
    result: GameResult
    moves: list[str]
    positions: list[str]  # FEN strings
    move_times_ms: list[float]
    total_time_ms: float
    start_time: datetime
    end_time: datetime
    final_fen: str
    termination_reason: str


@dataclass
class ScoreCard:
    """Tracks wins, losses, and draws for agents."""

    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_games: int = 0
    total_moves: int = 0
    avg_game_length: float = 0.0
    avg_game_time_ms: float = 0.0
    elo_estimate: float = 1500.0
    win_streak: int = 0
    loss_streak: int = 0
    current_streak: int = 0
    streak_type: str = ""  # "win", "loss", or ""

    # Learning metrics
    games_since_last_update: int = 0
    total_positions_learned: int = 0
    last_loss: float = 0.0
    avg_value_accuracy: float = 0.0

    def record_game(
        self,
        result: GameResult,
        num_moves: int,
        game_time_ms: float,
        player_color: str = "white",
    ) -> None:
        """Record a game result.

        Args:
            result: Game result
            num_moves: Number of moves in the game
            game_time_ms: Total game time in milliseconds
            player_color: Color the tracked player was playing
        """
        self.total_games += 1
        self.total_moves += num_moves
        self.games_since_last_update += 1

        # Update averages
        self.avg_game_length = self.total_moves / self.total_games
        self.avg_game_time_ms = (self.avg_game_time_ms * (self.total_games - 1) + game_time_ms) / self.total_games

        # Update win/loss/draw counts
        if result == GameResult.WHITE_WIN:
            self.white_wins += 1
            is_win = player_color == "white"
        elif result == GameResult.BLACK_WIN:
            self.black_wins += 1
            is_win = player_color == "black"
        else:
            self.draws += 1
            is_win = None

        # Update streaks
        if is_win is True:
            if self.streak_type == "win":
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.streak_type = "win"
            self.win_streak = max(self.win_streak, self.current_streak)
            self.loss_streak = 0
        elif is_win is False:
            if self.streak_type == "loss":
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.streak_type = "loss"
            self.loss_streak = max(self.loss_streak, self.current_streak)
        else:
            self.current_streak = 0
            self.streak_type = ""

        # Simple Elo estimate update
        self._update_elo(is_win)

    def _update_elo(self, is_win: bool | None) -> None:
        """Update Elo estimate based on game result."""
        k_factor = 32  # Standard K-factor
        expected = 0.5  # Assume opponent is equal strength

        if is_win is True:
            actual = 1.0
        elif is_win is False:
            actual = 0.0
        else:
            actual = 0.5

        self.elo_estimate += k_factor * (actual - expected)

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        if self.total_games == 0:
            return 0.0
        wins = self.white_wins + self.black_wins
        return wins / self.total_games

    @property
    def white_win_rate(self) -> float:
        """Calculate white win rate."""
        if self.total_games == 0:
            return 0.0
        return self.white_wins / self.total_games

    @property
    def draw_rate(self) -> float:
        """Calculate draw rate."""
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
            "total_games": self.total_games,
            "total_moves": self.total_moves,
            "avg_game_length": self.avg_game_length,
            "avg_game_time_ms": self.avg_game_time_ms,
            "elo_estimate": self.elo_estimate,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "games_since_last_update": self.games_since_last_update,
            "total_positions_learned": self.total_positions_learned,
            "last_loss": self.last_loss,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_games = 0
        self.total_moves = 0
        self.avg_game_length = 0.0
        self.avg_game_time_ms = 0.0
        self.elo_estimate = 1500.0
        self.win_streak = 0
        self.loss_streak = 0
        self.current_streak = 0
        self.streak_type = ""
        self.games_since_last_update = 0


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning."""

    # Session time limits
    max_session_minutes: int = 60
    max_games: int = 100
    max_moves_per_game: int = 150

    # Learning parameters
    learn_every_n_games: int = 5
    min_games_before_learning: int = 10
    learning_batch_size: int = 256
    learning_rate: float = 0.001

    # Self-play parameters
    temperature_schedule: str = "linear_decay"  # "constant", "linear_decay", "step"
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_decay_games: int = 50

    # Exploration
    add_noise: bool = True
    noise_weight: float = 0.25

    # Time control per move (milliseconds)
    time_per_move_ms: int = 5000
    increment_per_move_ms: int = 0

    # Callbacks
    on_game_complete: Callable[[GameRecord], None] | None = None
    on_learning_update: Callable[[float, int], None] | None = None
    on_session_complete: Callable[[ScoreCard], None] | None = None


class OnlineLearner:
    """Handles online learning from game outcomes."""

    def __init__(
        self,
        config: ChessConfig,
        device: str = "cpu",
    ) -> None:
        """Initialize the online learner.

        Args:
            config: Chess configuration
            device: Device for training
        """
        self.config = config
        self.device = device

        # Experience buffer for online learning
        self.experience_buffer: list[tuple[torch.Tensor, np.ndarray, float]] = []
        self.max_buffer_size = 10000

        # Optimizer (will be initialized when network is provided)
        self.optimizer: torch.optim.Optimizer | None = None
        self.network: torch.nn.Module | None = None

    def set_network(self, network: torch.nn.Module) -> None:
        """Set the network to train.

        Args:
            network: Policy-value network
        """
        self.network = network
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.training.learning_rate * 0.1,  # Lower LR for online learning
            weight_decay=self.config.training.weight_decay,
        )

    def add_experience(
        self,
        state_tensor: torch.Tensor,
        policy: np.ndarray,
        value: float,
    ) -> None:
        """Add experience to buffer.

        Args:
            state_tensor: Board state tensor
            policy: Move probability distribution
            value: Game outcome value
        """
        self.experience_buffer.append((state_tensor, policy, value))

        # Trim buffer if too large
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size :]

    def add_game_experience(
        self,
        positions: list[torch.Tensor],
        policies: list[np.ndarray],
        outcome: float,
    ) -> None:
        """Add all positions from a game to buffer.

        Args:
            positions: List of state tensors
            policies: List of policy vectors
            outcome: Game outcome (1.0 = white win, -1.0 = black win, 0 = draw)
        """
        for i, (pos, pol) in enumerate(zip(positions, policies, strict=False)):
            # Value from perspective of player to move
            perspective = 1 if i % 2 == 0 else -1
            value = outcome * perspective
            self.add_experience(pos, pol, value)

    def learn(self, batch_size: int = 256) -> float:
        """Perform a learning update.

        Args:
            batch_size: Number of samples to train on

        Returns:
            Training loss
        """
        if self.network is None or self.optimizer is None:
            logger.warning("Network not set, skipping learning")
            return 0.0

        if len(self.experience_buffer) < batch_size:
            logger.info(f"Buffer size {len(self.experience_buffer)} < {batch_size}, skipping")
            return 0.0

        # Sample batch
        batch_indices = random.sample(range(len(self.experience_buffer)), batch_size)
        batch = [self.experience_buffer[i] for i in batch_indices]

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
        self.network.train()
        policy_logits, values = self.network(states)

        # Calculate losses
        policy_loss = -torch.mean(torch.sum(target_policies * F.log_softmax(policy_logits, dim=-1), dim=-1))
        value_loss = F.mse_loss(values, target_values)
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self.network.eval()

        return float(total_loss.item())

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.experience_buffer)

    def clear_buffer(self) -> None:
        """Clear the experience buffer."""
        self.experience_buffer.clear()


class ContinuousLearningSession:
    """Manages a continuous learning session with self-play."""

    def __init__(
        self,
        chess_config: ChessConfig | None = None,
        learning_config: ContinuousLearningConfig | None = None,
    ) -> None:
        """Initialize the continuous learning session.

        Args:
            chess_config: Chess configuration
            learning_config: Learning configuration
        """
        self.chess_config = chess_config or get_chess_small_config()
        self.learning_config = learning_config or ContinuousLearningConfig()

        # Score tracking
        self.scorecard = ScoreCard()
        self.game_history: list[GameRecord] = []

        # Learning components
        self.learner = OnlineLearner(self.chess_config, self.chess_config.device)

        # Session state
        self.is_running = False
        self.is_paused = False
        self.session_start_time: datetime | None = None
        self.current_game_id = 0

        # Agents (lazy loaded)
        self._agent: ChessEnsembleAgent | None = None

        # Live state for UI
        self.current_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.current_last_move: str | None = None
        self.current_game_id_display: str = ""

    @property
    def agent(self) -> ChessEnsembleAgent:
        """Lazy load the ensemble agent."""
        if self._agent is None:
            from src.games.chess.ensemble_agent import ChessEnsembleAgent

            self._agent = ChessEnsembleAgent(self.chess_config)
            self.learner.set_network(self._agent.policy_value_net)
        return self._agent

    def get_temperature(self, game_number: int) -> float:
        """Get temperature for current game number.

        Args:
            game_number: Current game number

        Returns:
            Temperature value
        """
        config = self.learning_config

        if config.temperature_schedule == "constant":
            return config.initial_temperature

        elif config.temperature_schedule == "linear_decay":
            progress = min(1.0, game_number / config.temperature_decay_games)
            return config.initial_temperature - progress * (config.initial_temperature - config.final_temperature)

        elif config.temperature_schedule == "step":
            if game_number < config.temperature_decay_games // 2:
                return config.initial_temperature
            elif game_number < config.temperature_decay_games:
                return (config.initial_temperature + config.final_temperature) / 2
            else:
                return config.final_temperature

        return config.initial_temperature

    async def play_single_game(
        self,
        game_id: str,
        temperature: float = 1.0,
    ) -> GameRecord:
        """Play a single self-play game.

        Args:
            game_id: Unique game identifier
            temperature: Temperature for move selection

        Returns:
            GameRecord with game details
        """
        state = ChessGameState.initial()
        moves: list[str] = []
        positions: list[str] = [state.fen]
        position_tensors: list[torch.Tensor] = [state.to_tensor()]
        policies: list[np.ndarray] = []
        move_times: list[float] = []

        start_time = datetime.now()
        move_count = 0

        while not state.is_terminal() and move_count < self.learning_config.max_moves_per_game:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue

            if not self.is_running:
                break

            move_start = time.time()

            # Get move from agent
            try:
                response = await self.agent.get_best_move(
                    state,
                    temperature=temperature,
                    use_ensemble=True,
                )
                move = response.best_move
                policy = self._response_to_policy(response, state)
            except Exception as e:
                logger.error(f"Error getting move: {e}")
                # Fallback to random legal move
                legal_moves = state.get_legal_actions()
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                policy = np.zeros(self.chess_config.action_size)

            move_time = (time.time() - move_start) * 1000
            move_times.append(move_time)

            # Record move
            moves.append(move)
            policies.append(policy)

            # Apply move
            state = state.apply_action(move)
            positions.append(state.fen)
            position_tensors.append(state.to_tensor())

            # Update live state
            self.current_fen = state.fen
            self.current_last_move = move
            self.current_game_id_display = game_id

            move_count += 1

            # Check for 3-fold repetition
            # A correct implementation would require full FEN matching including castling rights/en passant
            # Here we do a simpler check on the full FEN string for exact state repetition
            count = positions.count(state.fen)
            if count >= 3:
                # Force draw
                break

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds() * 1000

        # Determine result
        if state.is_terminal():
            if state.is_checkmate():
                # Previous player won
                result = GameResult.BLACK_WIN if state.current_player == 1 else GameResult.WHITE_WIN
                termination = "checkmate"
            elif state.is_stalemate():
                result = GameResult.DRAW
                termination = "stalemate"
            else:
                result = GameResult.DRAW
                termination = "draw"
        elif move_count >= self.learning_config.max_moves_per_game:
            result = GameResult.DRAW
            termination = "max_moves"
        elif positions.count(state.fen) >= 3:
            result = GameResult.DRAW
            termination = "repetition"
        else:
            result = GameResult.IN_PROGRESS
            termination = "interrupted"

        # Get outcome value for learning
        if result == GameResult.WHITE_WIN:
            outcome = 1.0
        elif result == GameResult.BLACK_WIN:
            outcome = -1.0
        else:
            outcome = 0.0

        # Add experiences to learner
        self.learner.add_game_experience(position_tensors[:-1], policies, outcome)

        record = GameRecord(
            game_id=game_id,
            white_agent="ensemble",
            black_agent="ensemble",
            result=result,
            moves=moves,
            positions=positions,
            move_times_ms=move_times,
            total_time_ms=total_time,
            start_time=start_time,
            end_time=end_time,
            final_fen=state.fen,
            termination_reason=termination,
        )

        return record

    def _response_to_policy(self, response: Any, state: ChessGameState) -> np.ndarray:
        """Convert agent response to policy vector.

        Args:
            response: EnsembleResponse
            state: Current game state

        Returns:
            Policy vector
        """
        policy = np.zeros(self.chess_config.action_size)
        from_black = state.current_player == -1

        for move, prob in response.move_probabilities.items():
            try:
                idx = self.agent.action_encoder.encode_move(move, from_black)
                policy[idx] = prob
            except ValueError:
                pass  # Ignore invalid moves in response

        # Normalize
        if policy.sum() > 0:
            policy = policy / policy.sum()

        return policy

    async def run_session(
        self,
        max_minutes: int | None = None,
        max_games: int | None = None,
        progress_callback: Callable[[int, int, ScoreCard], None] | None = None,
    ) -> ScoreCard:
        """Run a continuous learning session.

        Args:
            max_minutes: Maximum session duration in minutes
            max_games: Maximum number of games
            progress_callback: Called after each game with (game_num, total, scorecard)

        Returns:
            Final scorecard
        """
        max_minutes = max_minutes or self.learning_config.max_session_minutes
        max_games = max_games or self.learning_config.max_games

        self.is_running = True
        self.is_paused = False
        self.session_start_time = datetime.now()
        session_end_time = self.session_start_time + timedelta(minutes=max_minutes)

        game_count = 0

        logger.info(f"Starting continuous learning session: {max_minutes}min, {max_games} games max")

        try:
            while self.is_running and game_count < max_games:
                # Check time limit
                if datetime.now() >= session_end_time:
                    logger.info("Session time limit reached")
                    break

                # Generate game ID
                self.current_game_id += 1
                game_id = f"game_{self.current_game_id:06d}"

                # Get temperature for this game
                temperature = self.get_temperature(game_count)

                # Play game
                record = await self.play_single_game(game_id, temperature)

                if record.result != GameResult.IN_PROGRESS:
                    # Record result
                    self.scorecard.record_game(
                        record.result,
                        len(record.moves),
                        record.total_time_ms,
                    )
                    self.game_history.append(record)
                    game_count += 1

                    # Callback
                    if self.learning_config.on_game_complete:
                        self.learning_config.on_game_complete(record)

                    if progress_callback:
                        progress_callback(game_count, max_games, self.scorecard)

                    logger.info(
                        f"Game {game_count}: {record.result.value} in {len(record.moves)} moves "
                        f"({record.termination_reason})"
                    )

                    # Learning update
                    if (
                        game_count >= self.learning_config.min_games_before_learning
                        and game_count % self.learning_config.learn_every_n_games == 0
                    ):
                        loss = self.learner.learn(self.learning_config.learning_batch_size)
                        self.scorecard.last_loss = loss
                        self.scorecard.total_positions_learned += self.learning_config.learning_batch_size
                        self.scorecard.games_since_last_update = 0

                        if self.learning_config.on_learning_update:
                            self.learning_config.on_learning_update(loss, game_count)

                        logger.info(f"Learning update: loss={loss:.4f}")

        except Exception as e:
            logger.exception(f"Session error: {e}")

        finally:
            self.is_running = False

        # Session complete callback
        if self.learning_config.on_session_complete:
            self.learning_config.on_session_complete(self.scorecard)

        logger.info(
            f"Session complete: {game_count} games, "
            f"W:{self.scorecard.white_wins} B:{self.scorecard.black_wins} D:{self.scorecard.draws}"
        )

        return self.scorecard

    def pause(self) -> None:
        """Pause the session."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume the session."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop the session."""
        self.is_running = False

    def reset_scorecard(self) -> None:
        """Reset the scorecard."""
        self.scorecard.reset()

    def get_session_duration(self) -> timedelta:
        """Get current session duration."""
        if self.session_start_time is None:
            return timedelta(0)
        return datetime.now() - self.session_start_time

    def get_remaining_time(self, max_minutes: int) -> timedelta:
        """Get remaining session time.

        Args:
            max_minutes: Maximum session minutes

        Returns:
            Remaining time
        """
        if self.session_start_time is None:
            return timedelta(minutes=max_minutes)

        elapsed = self.get_session_duration()
        remaining = timedelta(minutes=max_minutes) - elapsed
        return max(timedelta(0), remaining)


# Convenience function
def create_learning_session(
    preset: str = "small",
    max_minutes: int = 30,
    max_games: int = 50,
) -> ContinuousLearningSession:
    """Create a configured learning session.

    Args:
        preset: Chess config preset ("small", "medium", "large")
        max_minutes: Maximum session duration
        max_games: Maximum number of games

    Returns:
        Configured ContinuousLearningSession
    """
    from src.games.chess.config import ChessConfig

    chess_config = ChessConfig.from_preset(preset)
    learning_config = ContinuousLearningConfig(
        max_session_minutes=max_minutes,
        max_games=max_games,
    )

    return ContinuousLearningSession(chess_config, learning_config)
