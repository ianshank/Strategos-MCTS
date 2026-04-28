"""
Data Collector for Neural Network Training.

Collects training data from MCTS rollouts, self-play games, and LLM interactions
to train policy and value networks.
"""

from __future__ import annotations

import asyncio
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Experience:
    """Single experience for neural network training."""

    state: torch.Tensor  # State representation
    action: int | torch.Tensor  # Action taken or action distribution
    value: float  # Value estimate or actual outcome
    policy: torch.Tensor | None = None  # MCTS policy (visit counts)
    reward: float = 0.0  # Immediate reward
    next_state: torch.Tensor | None = None  # Next state (for TD learning)
    done: bool = False  # Episode termination flag
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional info


@dataclass
class GameTrajectory:
    """Complete game trajectory for training."""

    experiences: list[Experience]
    outcome: float  # Final outcome (1=win, 0=loss, 0.5=draw)
    game_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceBuffer:
    """
    Circular buffer for storing training experiences.

    Implements efficient storage, sampling, and persistence for
    neural network training data.
    """

    def __init__(self, max_size: int = 100000, save_dir: str | None = None):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum buffer size (oldest removed when full)
            save_dir: Directory for saving/loading buffer
        """
        self.buffer: deque[Experience] = deque(maxlen=max_size)
        self.max_size = max_size
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def add(self, experience: Experience) -> None:
        """Add single experience to buffer."""
        self.buffer.append(experience)

    def add_batch(self, experiences: list[Experience]) -> None:
        """Add multiple experiences."""
        self.buffer.extend(experiences)

    def add_trajectory(self, trajectory: GameTrajectory) -> None:
        """Add complete game trajectory."""
        # Update all experiences with final outcome
        for exp in trajectory.experiences:
            if exp.value == 0:  # If not already set
                exp.value = trajectory.outcome
            exp.metadata.update(trajectory.metadata)

        self.add_batch(trajectory.experiences)

    def sample(self, batch_size: int, replace: bool = False) -> list[Experience]:
        """
        Sample random batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            replace: Whether to sample with replacement

        Returns:
            sampled_experiences: List of experiences
        """
        import random

        if replace:
            return random.choices(list(self.buffer), k=batch_size)
        else:
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def get_recent(self, n: int) -> list[Experience]:
        """Get n most recent experiences."""
        return list(self.buffer)[-n:]

    def get_all(self) -> list[Experience]:
        """Get all experiences as list."""
        return list(self.buffer)

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()

    def save(self, filename: str) -> None:
        """Save buffer to disk."""
        if self.save_dir is None:
            raise ValueError("save_dir not specified")

        filepath = self.save_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(list(self.buffer), f)

        logger.info(f"Saved {len(self.buffer)} experiences to {filepath}")

    def load(self, filename: str) -> None:
        """Load buffer from disk."""
        if self.save_dir is None:
            raise ValueError("save_dir not specified")

        filepath = self.save_dir / filename
        with open(filepath, "rb") as f:
            experiences = pickle.load(f)
            self.buffer.extend(experiences)

        logger.info(f"Loaded {len(experiences)} experiences from {filepath}")

    def __len__(self) -> int:
        return len(self.buffer)

    def statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {"size": 0}

        values = [exp.value for exp in self.buffer]
        rewards = [exp.reward for exp in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.max_size,
            "utilization": len(self.buffer) / self.max_size,
            "avg_value": sum(values) / len(values),
            "avg_reward": sum(rewards) / len(rewards),
        }


class DataCollector:
    """
    Collect training data from MCTS rollouts and self-play.

    Orchestrates game playing, experience collection, and dataset creation
    for training policy and value networks.
    """

    def __init__(self, buffer: ExperienceBuffer, state_encoder: Any | None = None):
        """
        Initialize data collector.

        Args:
            buffer: Experience buffer for storage
            state_encoder: Optional encoder to convert states to tensors
        """
        self.buffer = buffer
        self.state_encoder = state_encoder

        # Statistics
        self.games_played = 0
        self.total_experiences = 0

    def encode_state(self, state: Any) -> torch.Tensor:
        """
        Encode state to tensor representation.

        Args:
            state: Raw state object

        Returns:
            state_tensor: Encoded state tensor
        """
        if self.state_encoder is not None:
            return self.state_encoder(state)
        elif isinstance(state, torch.Tensor):
            return state
        else:
            raise ValueError("state_encoder required for non-tensor states")

    async def collect_mcts_game(self, mcts_engine: Any, num_simulations: int = 100) -> GameTrajectory:
        """
        Play one game using MCTS and collect experiences.

        Args:
            mcts_engine: MCTS engine with search/select/apply methods
            num_simulations: Number of MCTS simulations per move

        Returns:
            trajectory: Complete game trajectory
        """
        experiences = []
        state = mcts_engine.reset()
        move_num = 0

        while not mcts_engine.is_terminal(state):
            # Run MCTS to get action distribution
            mcts_result = await mcts_engine.search(state, num_simulations=num_simulations)

            # Extract policy and value
            policy = mcts_result.get("policy")  # Visit count distribution
            value = mcts_result.get("value", 0.0)

            # Select action
            action = mcts_engine.select_action(policy)

            # Encode current state
            state_tensor = self.encode_state(state)

            # Create experience
            experience = Experience(
                state=state_tensor,
                action=action,
                value=value,
                policy=torch.tensor(policy) if policy is not None else None,
                metadata={"move_num": move_num, "game_id": self.games_played},
            )
            experiences.append(experience)

            # Apply action and get next state
            next_state = mcts_engine.apply_action(state, action)
            state = next_state
            move_num += 1

        # Get final outcome
        outcome = mcts_engine.get_outcome(state)

        # Create trajectory
        trajectory = GameTrajectory(
            experiences=experiences, outcome=outcome, game_id=self.games_played, metadata={"num_moves": move_num}
        )

        self.games_played += 1
        self.total_experiences += len(experiences)

        return trajectory

    async def collect_self_play_game(self, policy_network: Any, value_network: Any, env: Any) -> GameTrajectory:
        """
        Play one self-play game using neural networks.

        Args:
            policy_network: Policy network for action selection
            value_network: Value network for position evaluation
            env: Game environment

        Returns:
            trajectory: Complete game trajectory
        """
        experiences = []
        state = env.reset()
        move_num = 0

        while not env.is_done():
            # Encode state
            state_tensor = self.encode_state(state)

            # Get action from policy network
            action_selection = policy_network.select_action(state_tensor)
            action = action_selection.action

            # Get value estimate
            value = value_network.evaluate(state_tensor)

            # Create experience
            experience = Experience(
                state=state_tensor,
                action=action,
                value=value,
                metadata={
                    "move_num": move_num,
                    "game_id": self.games_played,
                    "confidence": action_selection.confidence,
                },
            )
            experiences.append(experience)

            # Apply action
            next_state, reward, done = env.step(action)
            experience.reward = reward
            experience.next_state = self.encode_state(next_state) if not done else None
            experience.done = done

            state = next_state
            move_num += 1

        # Get final outcome
        outcome = env.get_outcome()

        trajectory = GameTrajectory(
            experiences=experiences, outcome=outcome, game_id=self.games_played, metadata={"num_moves": move_num}
        )

        self.games_played += 1
        self.total_experiences += len(experiences)

        return trajectory

    async def collect_batch(
        self,
        num_games: int,
        game_fn: Any,
        save_every: int = 100,
        parallel: bool = True,
    ) -> int:
        """
        Collect multiple games.

        Args:
            num_games: Number of games to collect
            game_fn: Coroutine function that returns GameTrajectory
            save_every: Save buffer every N games
            parallel: Whether to collect games in parallel

        Returns:
            total_experiences: Number of experiences collected
        """
        from tqdm import tqdm

        initial_experiences = self.total_experiences

        for i in tqdm(range(0, num_games, save_every if not parallel else 1)):
            if parallel:
                # Collect games in parallel batches
                batch_size = min(save_every, num_games - i)
                trajectories = await asyncio.gather(*[game_fn() for _ in range(batch_size)])

                # Add to buffer
                for trajectory in trajectories:
                    self.buffer.add_trajectory(trajectory)

                # Save checkpoint
                if i % save_every == 0:
                    self.buffer.save(f"checkpoint_{self.games_played}.pkl")
            else:
                # Collect games sequentially
                trajectory = await game_fn()
                self.buffer.add_trajectory(trajectory)

                if (i + 1) % save_every == 0:
                    self.buffer.save(f"checkpoint_{self.games_played}.pkl")

        # Final save
        self.buffer.save(f"final_{self.games_played}.pkl")

        collected = self.total_experiences - initial_experiences
        logger.info(f"Collected {collected} experiences from {num_games} games")

        return collected

    def create_policy_dataset(self, use_visit_counts: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create dataset for policy network training.

        Args:
            use_visit_counts: Use MCTS visit counts as targets (vs. actions)

        Returns:
            states: [N, state_dim] state tensors
            targets: [N] actions or [N, action_dim] visit count distributions
            values: [N] value targets
        """
        experiences = self.buffer.get_all()

        states = torch.stack([exp.state for exp in experiences])
        values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32)

        if use_visit_counts and experiences[0].policy is not None:
            # Use MCTS visit counts as soft targets
            targets = torch.stack([exp.policy for exp in experiences])
        else:
            # Use actions as hard targets
            targets = torch.tensor([exp.action for exp in experiences], dtype=torch.long)

        return states, targets, values

    def create_value_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create dataset for value network training.

        Returns:
            states: [N, state_dim] state tensors
            values: [N] value targets
        """
        experiences = self.buffer.get_all()

        states = torch.stack([exp.state for exp in experiences])
        values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32)

        return states, values

    def create_td_dataset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create dataset for temporal difference learning.

        Returns:
            states: [N, state_dim] current states
            rewards: [N] rewards
            next_states: [N, state_dim] next states
            dones: [N] done flags
        """
        experiences = [exp for exp in self.buffer.get_all() if exp.next_state is not None]

        states = torch.stack([exp.state for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)

        return states, rewards, next_states, dones

    def get_statistics(self) -> dict[str, Any]:
        """Get collection statistics."""
        return {
            "games_played": self.games_played,
            "total_experiences": self.total_experiences,
            "buffer_stats": self.buffer.statistics(),
        }


class LLMDataCollector(DataCollector):
    """
    Data collector that uses LLM to generate high-quality training data.

    Collects expert demonstrations from LLM interactions to bootstrap
    neural network training.
    """

    def __init__(self, buffer: ExperienceBuffer, llm_client: Any, state_encoder: Any | None = None):
        """
        Initialize LLM data collector.

        Args:
            buffer: Experience buffer
            llm_client: LLM client for action selection
            state_encoder: State encoder
        """
        super().__init__(buffer, state_encoder)
        self.llm_client = llm_client
        self.llm_calls = 0
        self.llm_cost = 0.0

    async def collect_llm_game(self, env: Any, track_cost: bool = True) -> GameTrajectory:
        """
        Play one game using LLM for all decisions.

        Args:
            env: Game environment
            track_cost: Whether to track LLM API costs

        Returns:
            trajectory: Game trajectory with LLM actions
        """
        experiences = []
        state = env.reset()
        move_num = 0

        while not env.is_done():
            # Encode state
            state_tensor = self.encode_state(state)

            # Get action from LLM
            llm_response = await self.llm_client.select_action(state)
            action = llm_response["action"]

            self.llm_calls += 1
            if track_cost and "cost" in llm_response:
                self.llm_cost += llm_response["cost"]

            # Create experience
            experience = Experience(
                state=state_tensor,
                action=action,
                value=0.0,  # Will be updated with final outcome
                metadata={
                    "move_num": move_num,
                    "game_id": self.games_played,
                    "source": "llm",
                    "llm_response": llm_response.get("reasoning"),
                },
            )
            experiences.append(experience)

            # Apply action
            next_state, reward, done = env.step(action)
            experience.reward = reward
            experience.next_state = self.encode_state(next_state) if not done else None
            experience.done = done

            state = next_state
            move_num += 1

        # Get final outcome
        outcome = env.get_outcome()

        trajectory = GameTrajectory(
            experiences=experiences,
            outcome=outcome,
            game_id=self.games_played,
            metadata={"num_moves": move_num, "source": "llm", "cost": self.llm_cost},
        )

        self.games_played += 1
        self.total_experiences += len(experiences)

        return trajectory

    def get_llm_statistics(self) -> dict[str, Any]:
        """Get LLM-specific statistics."""
        stats = self.get_statistics()
        stats.update(
            {
                "llm_calls": self.llm_calls,
                "llm_cost": self.llm_cost,
                "avg_cost_per_call": self.llm_cost / max(1, self.llm_calls),
            }
        )
        return stats
