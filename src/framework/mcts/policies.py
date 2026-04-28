"""
MCTS Policies Module - Selection, rollout, and evaluation policies.

Provides:
- UCB1 with configurable exploration weight
- Rollout heuristics (random, greedy, hybrid)
- Action selection policies (max visits, max value, robust child)
- Progressive widening parameters
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src.observability.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .core import MCTSState


def ucb1(
    value_sum: float,
    visits: int,
    parent_visits: int,
    c: float = 1.414,
) -> float:
    """
    Upper Confidence Bound 1 (UCB1) formula for tree selection.

    Formula: Q(s,a) + c * sqrt(N(s)) / sqrt(N(s,a))

    Args:
        value_sum: Total accumulated value for the node
        visits: Number of visits to the node
        parent_visits: Number of visits to the parent node
        c: Exploration weight constant (default sqrt(2))

    Returns:
        UCB1 score for node selection
    """
    if visits == 0:
        # Encourage exploring unvisited nodes.
        return float("inf")

    if parent_visits <= 0:
        # Should not happen in standard flow. Avoid ln(0) by treating as no parent info.
        return float("inf")

    exploitation = value_sum / visits
    exploration = c * math.sqrt(math.log(parent_visits) / visits)

    return exploitation + exploration


class SelectionPolicy(Enum):
    """Policy for selecting the final action after MCTS search."""

    MAX_VISITS = "max_visits"
    """Select action with most visits (most robust)."""

    MAX_VALUE = "max_value"
    """Select action with highest average value (greedy)."""

    ROBUST_CHILD = "robust_child"
    """Select action balancing visits and value."""

    SECURE_CHILD = "secure_child"
    """Select action with lowest lower confidence bound."""


class RolloutPolicy(ABC):
    """Abstract base class for rollout/simulation policies."""

    @abstractmethod
    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """
        Evaluate a state through rollout simulation.

        Args:
            state: State to evaluate
            rng: Seeded random number generator
            max_depth: Maximum rollout depth

        Returns:
            Estimated value in [0, 1] range
        """
        pass


class RandomRolloutPolicy(RolloutPolicy):
    """Random rollout policy - uniform random evaluation."""

    def __init__(self, base_value: float = 0.5, noise_scale: float = 0.3):
        """
        Initialize random rollout policy.

        Args:
            base_value: Base value for evaluations
            noise_scale: Scale of random noise
        """
        self.base_value = base_value
        self.noise_scale = noise_scale

    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """Generate random evaluation with noise."""
        noise = float(rng.uniform(-self.noise_scale, self.noise_scale))
        value = self.base_value + noise
        return max(0.0, min(1.0, value))


class GreedyRolloutPolicy(RolloutPolicy):
    """Greedy rollout policy using domain heuristics."""

    def __init__(
        self,
        heuristic_fn: Callable[[MCTSState], float],
        noise_scale: float = 0.05,
    ):
        """
        Initialize greedy rollout policy.

        Args:
            heuristic_fn: Function to evaluate state heuristically
            noise_scale: Small noise for tie-breaking
        """
        self.heuristic_fn = heuristic_fn
        self.noise_scale = noise_scale

    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """Evaluate using heuristic with small noise."""
        base_value = self.heuristic_fn(state)
        noise = float(rng.uniform(-self.noise_scale, self.noise_scale))
        value = base_value + noise
        return max(0.0, min(1.0, value))


class HybridRolloutPolicy(RolloutPolicy):
    """Hybrid policy combining random and heuristic evaluation."""

    def __init__(
        self,
        heuristic_fn: Callable[[MCTSState], float] | None = None,
        heuristic_weight: float = 0.7,
        random_weight: float = 0.3,
        base_random_value: float = 0.5,
        noise_scale: float = 0.2,
    ):
        """
        Initialize hybrid rollout policy.

        Args:
            heuristic_fn: Optional heuristic evaluation function
            heuristic_weight: Weight for heuristic component
            random_weight: Weight for random component
            base_random_value: Base value for random component
            noise_scale: Noise scale for random component
        """
        self.heuristic_fn = heuristic_fn
        self.heuristic_weight = heuristic_weight
        self.random_weight = random_weight
        self.base_random_value = base_random_value
        self.noise_scale = noise_scale

        # Normalize weights
        total_weight = heuristic_weight + random_weight
        if total_weight > 0:
            self.heuristic_weight /= total_weight
            self.random_weight /= total_weight

    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """Combine heuristic and random evaluation."""
        # Random component
        random_noise = float(rng.uniform(-self.noise_scale, self.noise_scale))
        random_value = self.base_random_value + random_noise

        # Heuristic component
        heuristic_value = self.heuristic_fn(state) if self.heuristic_fn is not None else self.base_random_value

        # Combine
        value = self.heuristic_weight * heuristic_value + self.random_weight * random_value

        return max(0.0, min(1.0, value))


class LLMRolloutPolicy(RolloutPolicy):
    """Rollout policy that uses an LLM for state evaluation."""

    def __init__(
        self,
        evaluate_fn: Callable[[MCTSState], Awaitable[float]],
        cache_results: bool = True,
    ):
        """
        Initialize LLM rollout policy.

        Args:
            evaluate_fn: Async function to evaluate state with LLM
            cache_results: Whether to cache evaluation results
        """
        self.evaluate_fn = evaluate_fn
        self.cache_results = cache_results
        self._cache: dict = {}

    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """Evaluate state using LLM."""
        state_key = state.to_hash_key()

        if self.cache_results and state_key in self._cache:
            return float(self._cache[state_key])

        value = await self.evaluate_fn(state)
        value = max(0.0, min(1.0, value))

        if self.cache_results:
            self._cache[state_key] = value

        return value


class ProgressiveWideningConfig:
    """Configuration for progressive widening in MCTS."""

    def __init__(
        self,
        k: float = 1.0,
        alpha: float = 0.5,
    ):
        """
        Configure progressive widening parameters.

        Progressive widening expands when: visits > k * num_children^alpha

        Args:
            k: Coefficient controlling expansion threshold
            alpha: Exponent controlling growth rate

        Common configurations:
        - k=1.0, alpha=0.5: Moderate widening (default)
        - k=2.0, alpha=0.5: Conservative (fewer expansions)
        - k=0.5, alpha=0.5: Aggressive (more expansions)
        - k=1.0, alpha=0.3: Very aggressive
        - k=1.0, alpha=0.7: Very conservative
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        self.k = k
        self.alpha = alpha

    def should_expand(self, visits: int, num_children: int) -> bool:
        """
        Check if expansion should occur.

        Args:
            visits: Number of visits to node
            num_children: Current number of children

        Returns:
            True if should expand, False otherwise
        """
        threshold = self.k * (num_children**self.alpha)
        return bool(visits > threshold)

    def min_visits_for_expansion(self, num_children: int) -> int:
        """
        Calculate minimum visits needed to expand to next child.

        Args:
            num_children: Current number of children

        Returns:
            Minimum visit count for expansion
        """
        threshold = self.k * (num_children**self.alpha)
        return int(math.ceil(threshold))

    def __repr__(self) -> str:
        return f"ProgressiveWideningConfig(k={self.k}, alpha={self.alpha})"
