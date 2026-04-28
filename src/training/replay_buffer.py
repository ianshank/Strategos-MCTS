"""
Experience Replay Buffer for LangGraph Multi-Agent MCTS Training.

Implements:
- Prioritized experience replay
- Uniform sampling
- Efficient circular buffer storage
- Data augmentation support
"""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Experience:
    """Single training example from self-play."""

    state: torch.Tensor  # State representation
    policy: np.ndarray  # MCTS visit count distribution
    value: float  # Game outcome from this state's perspective
    metadata: dict | None = None  # Optional metadata (e.g., game_id, move_number)


class ReplayBuffer:
    """
    Simple uniform sampling replay buffer.

    Stores recent experiences in a circular buffer and samples uniformly.
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        logger.debug("ReplayBuffer initialized: capacity=%d", capacity)

    def add(self, experience: Experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def add_batch(self, experiences: list[Experience]):
        """Add multiple experiences."""
        for exp in experiences:
            self.add(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample a batch of experiences uniformly.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= min_size

    def clear(self):
        """Clear all experiences from buffer."""
        logger.debug("Clearing replay buffer (had %d experiences)", len(self.buffer))
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Samples experiences with probability proportional to their TD error.
    Helps focus training on more informative examples.

    Based on: "Prioritized Experience Replay" (Schaul et al., 2015)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        # Storage
        self.buffer: list[Experience | None] = [None] * capacity
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

        logger.debug(
            "PrioritizedReplayBuffer initialized: capacity=%d, alpha=%.2f, beta_start=%.2f",
            capacity,
            alpha,
            beta_start,
        )

    def _get_beta(self) -> float:
        """Get current beta value (anneals from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def add(self, experience: Experience, priority: float | None = None):
        """
        Add experience with priority.

        Args:
            experience: Experience to add
            priority: Priority value (uses max priority if None)
        """
        if priority is None:
            # New experiences get max priority
            priority = self.priorities.max() if self.size > 0 else 1.0

        self.buffer[self.position] = experience
        self.priorities[self.position] = priority**self.alpha

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, experiences: list[Experience], priorities: list[float] | None = None):
        """
        Add multiple experiences.

        Args:
            experiences: List of experiences
            priorities: Optional list of priorities (same length as experiences)
        """
        resolved_priorities: list[float | None]
        if priorities is None:
            resolved_priorities = [None] * len(experiences)
        else:
            resolved_priorities = list(priorities)

        for exp, priority in zip(experiences, resolved_priorities, strict=True):
            self.add(exp, priority)

    def sample(self, batch_size: int) -> tuple[list[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            (experiences, indices, weights) tuple
            - experiences: Sampled experiences
            - indices: Buffer indices (for updating priorities)
            - weights: Importance sampling weights
        """
        if self.size < batch_size:
            batch_size = self.size

        # Compute sampling probabilities
        priorities = self.priorities[: self.size]
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        self.frame += 1

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.

        Args:
            indices: Buffer indices to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities, strict=True):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha  # Small epsilon for stability

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= min_size


class AugmentedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with data augmentation support.

    Applies symmetries/transformations to states during sampling
    (e.g., rotations and flips for Go/chess boards).
    """

    def __init__(self, capacity: int, augmentation_fn=None):
        """
        Initialize augmented replay buffer.

        Args:
            capacity: Maximum buffer size
            augmentation_fn: Function to apply augmentations
                            Should take (state, policy) and return augmented versions
        """
        super().__init__(capacity)
        self.augmentation_fn = augmentation_fn

    def sample(self, batch_size: int, apply_augmentation: bool = True) -> list[Experience]:
        """
        Sample batch with optional augmentation.

        Args:
            batch_size: Number of experiences to sample
            apply_augmentation: Whether to apply data augmentation

        Returns:
            List of (possibly augmented) experiences
        """
        experiences = super().sample(batch_size)

        if apply_augmentation and self.augmentation_fn is not None:
            augmented = []
            for exp in experiences:
                aug_state, aug_policy = self.augmentation_fn(exp.state, exp.policy)
                augmented.append(
                    Experience(
                        state=aug_state,
                        policy=aug_policy,
                        value=exp.value,
                        metadata=exp.metadata,
                    )
                )
            return augmented

        return experiences


# Data augmentation utilities for board games
class BoardGameAugmentation:
    """
    Data augmentation for board games (Go, Chess, etc.).

    Applies symmetry transformations: rotations and reflections.
    """

    @staticmethod
    def rotate_90(state: torch.Tensor, policy: np.ndarray, board_size: int = 19) -> tuple[torch.Tensor, np.ndarray]:
        """
        Rotate state and policy 90 degrees clockwise.

        Args:
            state: State tensor [channels, height, width]
            policy: Policy array [action_size]
            board_size: Board dimension

        Returns:
            (rotated_state, rotated_policy) tuple
        """
        # Rotate state
        rotated_state = torch.rot90(state, k=1, dims=[1, 2])

        # Rotate policy (assuming policy corresponds to board positions)
        # This is game-specific; here's a simple version for square boards
        if len(policy) == board_size * board_size + 1:  # +1 for pass action
            policy_board = policy[:-1].reshape(board_size, board_size)
            rotated_policy_board = np.rot90(policy_board, k=1)
            rotated_policy = np.append(rotated_policy_board.flatten(), policy[-1])
        else:
            rotated_policy = policy  # Can't rotate, return original

        return rotated_state, rotated_policy

    @staticmethod
    def flip_horizontal(
        state: torch.Tensor, policy: np.ndarray, board_size: int = 19
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Flip state and policy horizontally."""
        flipped_state = torch.flip(state, dims=[2])  # Flip width dimension

        if len(policy) == board_size * board_size + 1:
            policy_board = policy[:-1].reshape(board_size, board_size)
            flipped_policy_board = np.fliplr(policy_board)
            flipped_policy = np.append(flipped_policy_board.flatten(), policy[-1])
        else:
            flipped_policy = policy

        return flipped_state, flipped_policy

    @staticmethod
    def random_symmetry(
        state: torch.Tensor, policy: np.ndarray, board_size: int = 19
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Apply random symmetry transformation.

        Randomly selects from:
        - Identity
        - 90° rotation
        - 180° rotation
        - 270° rotation
        - Horizontal flip
        - Vertical flip
        - Diagonal flip
        - Anti-diagonal flip
        """
        transform = random.randint(0, 7)

        if transform == 0:
            # Identity
            return state, policy
        elif transform == 1:
            # 90° rotation
            return BoardGameAugmentation.rotate_90(state, policy, board_size)
        elif transform == 2:
            # 180° rotation
            s, p = BoardGameAugmentation.rotate_90(state, policy, board_size)
            return BoardGameAugmentation.rotate_90(s, p, board_size)
        elif transform == 3:
            # 270° rotation
            s, p = BoardGameAugmentation.rotate_90(state, policy, board_size)
            s, p = BoardGameAugmentation.rotate_90(s, p, board_size)
            return BoardGameAugmentation.rotate_90(s, p, board_size)
        elif transform == 4:
            # Horizontal flip
            return BoardGameAugmentation.flip_horizontal(state, policy, board_size)
        elif transform == 5:
            # Vertical flip
            return torch.flip(state, dims=[1]), policy  # Simplified
        elif transform == 6:
            # Diagonal flip (transpose)
            return state.transpose(1, 2), policy  # Simplified
        else:
            # Anti-diagonal flip
            return torch.flip(state.transpose(1, 2), dims=[1, 2]), policy  # Simplified


def collate_experiences(experiences: list[Experience]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate list of experiences into batched tensors.

    Args:
        experiences: List of Experience objects

    Returns:
        (states, policies, values) tuple of batched tensors
    """
    states = torch.stack([exp.state for exp in experiences])

    # Handle variable-sized policies by padding to max size
    max_policy_size = max(len(exp.policy) for exp in experiences)
    padded_policies = []
    for exp in experiences:
        policy = exp.policy
        if len(policy) < max_policy_size:
            # Pad with zeros
            padded = np.zeros(max_policy_size, dtype=policy.dtype)
            padded[: len(policy)] = policy
            padded_policies.append(padded)
        else:
            padded_policies.append(policy)

    policies = torch.from_numpy(np.stack(padded_policies))
    values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32)

    return states, policies, values
