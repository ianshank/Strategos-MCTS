"""
Neural MCTS Policies Module - Neural network integration for MCTS.

Provides:
- PUCT (Polynomial Upper Confidence Trees) selection
- NeuralRolloutPolicy for neural network state evaluation
- ActionFilter protocol for legal move filtering
- Batch evaluation support for GPU efficiency
- Domain adapter interface for state conversion

This module bridges the gap between standard MCTS (UCB1-based) and
neural-guided MCTS (PUCT-based), allowing neural networks to be used
with the standard MCTSEngine.

Based on:
- AlphaGo Zero (Silver et al., 2017)
- AlphaZero (Silver et al., 2018)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .core import MCTSNode, MCTSState

# Lazy imports for optional neural dependencies
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment,unused-ignore]
    nn = None  # type: ignore[assignment,unused-ignore]

_logger = logging.getLogger(__name__)


def is_torch_available() -> bool:
    """Check if PyTorch is available for neural policies."""
    return _TORCH_AVAILABLE


# =============================================================================
# PUCT Selection Policy
# =============================================================================


def puct(
    q_value: float,
    prior: float,
    visit_count: int,
    parent_visits: int,
    c_puct: float = 1.25,
) -> float:
    """
    Polynomial Upper Confidence Trees (PUCT) selection formula.

    Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    PUCT is the selection formula used in AlphaGo Zero and AlphaZero.
    It balances exploitation (Q-value) with exploration (prior-guided).

    Args:
        q_value: Average value of the action Q(s,a) in range [0, 1]
        prior: Prior probability P(s,a) from policy network
        visit_count: Number of visits to child N(s,a)
        parent_visits: Number of visits to parent N(s)
        c_puct: Exploration constant (default 1.25, AlphaZero used 1.0-2.5)

    Returns:
        PUCT score for action selection

    Example:
        >>> score = puct(q_value=0.6, prior=0.3, visit_count=10, parent_visits=100)
        >>> print(f"PUCT score: {score:.3f}")
    """
    if visit_count == 0:
        # Encourage exploring unvisited nodes with high prior
        return float("inf") if prior > 0 else c_puct

    # Exploration bonus: decreases as node is visited more
    exploration = c_puct * prior * math.sqrt(parent_visits) / (1 + visit_count)

    return q_value + exploration


def puct_with_virtual_loss(
    q_value: float,
    prior: float,
    visit_count: int,
    parent_visits: int,
    virtual_loss: float = 0.0,
    c_puct: float = 1.25,
) -> float:
    """
    PUCT with virtual loss for parallel tree search.

    Virtual loss temporarily decreases a node's value during parallel search
    to encourage exploration of different paths by concurrent threads.

    Args:
        q_value: Average action value
        prior: Prior probability from policy network
        visit_count: Child visit count
        parent_visits: Parent visit count
        virtual_loss: Temporary loss added during parallel search
        c_puct: Exploration constant

    Returns:
        PUCT score with virtual loss adjustment
    """
    effective_visits = visit_count + virtual_loss

    if effective_visits == 0:
        return float("inf") if prior > 0 else c_puct

    # Adjust Q-value for virtual loss (pessimistic)
    adjusted_q = (q_value * visit_count) / (visit_count + virtual_loss)

    exploration = c_puct * prior * math.sqrt(parent_visits) / (1 + effective_visits)

    return adjusted_q + exploration


# =============================================================================
# Neural Network State Adapter Protocol
# =============================================================================


@runtime_checkable
class StateAdapter(Protocol):
    """
    Protocol for converting MCTS states to neural network inputs.

    Implementations should handle domain-specific state representations
    and convert them to tensors suitable for neural network evaluation.

    This is a Protocol class (PEP 544) - implement the methods without
    inheriting from this class for structural subtyping.
    """

    def state_to_tensor(self, state: MCTSState) -> Any:
        """
        Convert MCTS state to neural network input tensor.

        Args:
            state: MCTS state to convert

        Returns:
            Tensor suitable for neural network input (typically torch.Tensor)
        """
        ...

    def get_action_mask(self, state: MCTSState) -> Any:
        """
        Get mask for legal actions in state.

        Args:
            state: MCTS state to get mask for

        Returns:
            Boolean mask or None if all actions are legal
            True = legal action, False = illegal action
        """
        ...

    def tensor_to_action_priors(
        self,
        policy_output: Any,
        state: MCTSState,
    ) -> dict[str, float]:
        """
        Convert neural network policy output to action priors.

        Args:
            policy_output: Raw policy network output
            state: State for context (may need for action mapping)

        Returns:
            Dictionary mapping action strings to prior probabilities
        """
        ...


# =============================================================================
# Action Filter Protocol
# =============================================================================


@runtime_checkable
class ActionFilter(Protocol):
    """
    Protocol for filtering legal actions during rollout.

    Use this to restrict MCTS expansion and rollout to valid moves only.
    """

    def get_legal_actions(self, state: MCTSState) -> list[str]:
        """
        Get list of legal actions from state.

        Args:
            state: Current MCTS state

        Returns:
            List of legal action strings
        """
        ...

    def is_terminal(self, state: MCTSState) -> bool:
        """
        Check if state is terminal.

        Args:
            state: Current MCTS state

        Returns:
            True if state is terminal (no valid actions)
        """
        ...


# =============================================================================
# Neural Rollout Policy Configuration
# =============================================================================


@dataclass
class NeuralPolicyConfig:
    """
    Configuration for neural rollout policies.

    Attributes:
        device: Device to run inference on ("cpu", "cuda", "mps")
        batch_size: Batch size for batched evaluation (0 = no batching)
        use_action_mask: Whether to apply action masking
        temperature: Temperature for action sampling (0 = argmax)
        cache_evaluations: Whether to cache network evaluations
        cache_max_size: Maximum cache size (0 = unlimited)
        normalize_value: Whether to normalize value to [0, 1]
    """

    device: str = "cpu"
    batch_size: int = 0
    use_action_mask: bool = True
    temperature: float = 0.0
    cache_evaluations: bool = True
    cache_max_size: int = 10000
    normalize_value: bool = True

    def __post_init__(self):
        """Validate configuration."""
        valid_devices = {"cpu", "cuda", "mps"}
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            _logger.warning(f"Unrecognized device '{self.device}', defaulting to 'cpu'")
            self.device = "cpu"

        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative")

        if self.cache_max_size < 0:
            raise ValueError("cache_max_size must be non-negative")


# =============================================================================
# Base Neural Rollout Policy
# =============================================================================


class NeuralRolloutPolicy(ABC):
    """
    Abstract base class for neural network rollout policies.

    Subclasses implement domain-specific state conversion and
    network evaluation.
    """

    def __init__(
        self,
        config: NeuralPolicyConfig | None = None,
    ):
        """
        Initialize neural rollout policy.

        Args:
            config: Policy configuration (uses defaults if None)
        """
        self.config = config or NeuralPolicyConfig()
        self._cache: dict[str, float] = {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def _get_network_value(self, state: MCTSState) -> float:
        """
        Get value estimate from neural network.

        Implementations should handle tensor conversion and inference.

        Args:
            state: MCTS state to evaluate

        Returns:
            Value estimate in [0, 1] range (or normalized)
        """
        pass

    def _get_cache_key(self, state: MCTSState) -> str:
        """Get cache key for state."""
        return state.to_hash_key()

    def _check_cache(self, state: MCTSState) -> float | None:
        """Check if evaluation is cached."""
        if not self.config.cache_evaluations:
            return None

        key = self._get_cache_key(state)
        return self._cache.get(key)

    def _store_cache(self, state: MCTSState, value: float) -> None:
        """Store evaluation in cache."""
        if not self.config.cache_evaluations:
            return

        # Enforce cache size limit (simple LRU approximation)
        if self.config.cache_max_size > 0 and len(self._cache) >= self.config.cache_max_size:
            # Remove oldest entries (simple strategy)
            keys_to_remove = list(self._cache.keys())[: len(self._cache) // 4]
            for key in keys_to_remove:
                del self._cache[key]

        key = self._get_cache_key(state)
        self._cache[key] = value

    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,  # noqa: ARG002 - Interface compatibility
    ) -> float:
        """
        Evaluate state using neural network.

        Args:
            state: State to evaluate
            rng: Random number generator (unused for deterministic eval)
            max_depth: Maximum depth (unused, for interface compatibility)

        Returns:
            Value estimate in [0, 1] range
        """
        # Check cache first
        cached = self._check_cache(state)
        if cached is not None:
            self._logger.debug("Cache hit for state evaluation")
            return cached

        # Get network evaluation
        value = self._get_network_value(state)

        # Normalize to [0, 1] if configured
        if self.config.normalize_value:
            # Assuming network outputs in [-1, 1] (tanh)
            value = (value + 1.0) / 2.0

        # Clamp to valid range
        value = max(0.0, min(1.0, value))

        # Store in cache
        self._store_cache(state, value)

        return value

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.config.cache_max_size,
        }


# =============================================================================
# PyTorch Neural Rollout Policy
# =============================================================================


if _TORCH_AVAILABLE:

    class TorchNeuralRolloutPolicy(NeuralRolloutPolicy):
        """
        Neural rollout policy using PyTorch models.

        Wraps a PyTorch policy-value network to provide rollout evaluation
        compatible with the standard MCTSEngine.

        Example:
            >>> from src.models.policy_value_net import PolicyValueNetwork
            >>> from src.training.system_config import NeuralNetworkConfig
            >>>
            >>> # Create network
            >>> net_config = NeuralNetworkConfig(input_channels=3, action_size=9)
            >>> network = PolicyValueNetwork(net_config, board_size=3)
            >>>
            >>> # Create adapter for your domain
            >>> adapter = MyDomainAdapter()
            >>>
            >>> # Create rollout policy
            >>> policy = TorchNeuralRolloutPolicy(
            ...     network=network,
            ...     state_adapter=adapter,
            ... )
            >>>
            >>> # Use with MCTSEngine
            >>> value = await policy.evaluate(state, rng)
        """

        def __init__(
            self,
            network: nn.Module,
            state_adapter: StateAdapter,
            config: NeuralPolicyConfig | None = None,
        ):
            """
            Initialize PyTorch neural rollout policy.

            Args:
                network: PyTorch neural network (policy-value or value network)
                state_adapter: Adapter for converting states to tensors
                config: Policy configuration
            """
            super().__init__(config)
            self.network = network
            self.state_adapter = state_adapter

            # Move network to configured device
            self._device = torch.device(self.config.device)
            self.network = self.network.to(self._device)
            self.network.eval()  # Set to evaluation mode

            self._logger.info(f"Initialized TorchNeuralRolloutPolicy on device: {self._device}")

        def _get_network_value(self, state: MCTSState) -> float:
            """Get value from PyTorch network."""
            # Convert state to tensor
            state_tensor = self.state_adapter.state_to_tensor(state)

            if not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.tensor(state_tensor, dtype=torch.float32)

            # Add batch dimension if needed
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)

            # Move to device
            state_tensor = state_tensor.to(self._device)

            # Inference without gradients
            with torch.no_grad():
                output = self.network(state_tensor)

                # Handle different network output formats
                if isinstance(output, tuple):
                    # Policy-value network: (policy, value)
                    _, value = output
                else:
                    # Value-only network
                    value = output

                # Extract scalar value
                if isinstance(value, torch.Tensor):
                    value = value.squeeze().item()

            return float(value)

        def get_policy_priors(self, state: MCTSState) -> dict[str, float]:
            """
            Get action priors from policy network.

            Args:
                state: State to get priors for

            Returns:
                Dictionary mapping actions to prior probabilities
            """
            state_tensor = self.state_adapter.state_to_tensor(state)

            if not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.tensor(state_tensor, dtype=torch.float32)

            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)

            state_tensor = state_tensor.to(self._device)

            with torch.no_grad():
                output = self.network(state_tensor)

                if isinstance(output, tuple):
                    policy_logits, _ = output
                else:
                    # Network doesn't output policy
                    return {}

                # Apply action masking if configured
                if self.config.use_action_mask:
                    action_mask = self.state_adapter.get_action_mask(state)
                    if action_mask is not None:
                        if not isinstance(action_mask, torch.Tensor):
                            action_mask = torch.tensor(action_mask, dtype=torch.bool)
                        action_mask = action_mask.to(self._device)
                        # Set illegal actions to -inf
                        policy_logits = policy_logits.masked_fill(~action_mask, float("-inf"))

                # Convert to probabilities
                policy_probs = torch.softmax(policy_logits, dim=-1)

            # Convert to action dictionary
            return self.state_adapter.tensor_to_action_priors(
                policy_probs.cpu().numpy().squeeze(),
                state,
            )


# =============================================================================
# Fallback Policy (No PyTorch)
# =============================================================================


class FallbackNeuralRolloutPolicy(NeuralRolloutPolicy):
    """
    Fallback rollout policy when PyTorch is not available.

    Uses heuristic evaluation instead of neural network.
    """

    def __init__(
        self,
        heuristic_fn: Callable[[MCTSState], float] | None = None,
        config: NeuralPolicyConfig | None = None,
    ):
        """
        Initialize fallback policy.

        Args:
            heuristic_fn: Optional heuristic function for evaluation
            config: Policy configuration
        """
        super().__init__(config)
        self.heuristic_fn = heuristic_fn
        self._logger.warning("PyTorch not available, using fallback heuristic policy")

    def _get_network_value(self, state: MCTSState) -> float:
        """Get heuristic value estimate."""
        if self.heuristic_fn is not None:
            return self.heuristic_fn(state)

        # Default: return neutral value
        return 0.0


# =============================================================================
# Factory Function
# =============================================================================


def create_neural_rollout_policy(
    network: Any = None,
    state_adapter: StateAdapter | None = None,
    heuristic_fn: Callable[[MCTSState], float] | None = None,
    config: NeuralPolicyConfig | None = None,
) -> NeuralRolloutPolicy:
    """
    Factory function to create appropriate neural rollout policy.

    Creates TorchNeuralRolloutPolicy if PyTorch is available and network
    is provided, otherwise falls back to FallbackNeuralRolloutPolicy.

    Args:
        network: Optional neural network (PyTorch nn.Module)
        state_adapter: Optional state adapter for tensor conversion
        heuristic_fn: Optional heuristic function for fallback
        config: Policy configuration

    Returns:
        Appropriate neural rollout policy instance

    Example:
        >>> # With PyTorch network
        >>> policy = create_neural_rollout_policy(
        ...     network=my_network,
        ...     state_adapter=my_adapter,
        ... )
        >>>
        >>> # Fallback to heuristic
        >>> policy = create_neural_rollout_policy(
        ...     heuristic_fn=lambda s: 0.5,
        ... )
    """
    if _TORCH_AVAILABLE and network is not None and state_adapter is not None:
        return TorchNeuralRolloutPolicy(
            network=network,
            state_adapter=state_adapter,
            config=config,
        )

    return FallbackNeuralRolloutPolicy(
        heuristic_fn=heuristic_fn,
        config=config,
    )


# =============================================================================
# Priors Manager for Standard MCTS
# =============================================================================


@dataclass
class PriorsManager:
    """
    Manages action priors for PUCT selection in standard MCTS.

    This class bridges neural network policy outputs with the standard
    MCTSEngine by storing and retrieving action priors.

    Example:
        >>> manager = PriorsManager()
        >>> manager.set_priors("state_hash", {"action1": 0.6, "action2": 0.4})
        >>> prior = manager.get_prior("state_hash", "action1")  # Returns 0.6
    """

    _priors: dict[str, dict[str, float]] = field(default_factory=dict)
    default_prior: float = 1.0 / 100  # Uniform over ~100 actions

    def set_priors(self, state_hash: str, priors: dict[str, float]) -> None:
        """
        Set action priors for a state.

        Args:
            state_hash: Hash key for the state
            priors: Dictionary mapping actions to prior probabilities
        """
        self._priors[state_hash] = priors

    def get_prior(self, state_hash: str, action: str) -> float:
        """
        Get prior probability for an action.

        Args:
            state_hash: Hash key for the state
            action: Action to get prior for

        Returns:
            Prior probability (default_prior if not found)
        """
        if state_hash not in self._priors:
            return self.default_prior

        return self._priors[state_hash].get(action, self.default_prior)

    def get_all_priors(self, state_hash: str) -> dict[str, float]:
        """Get all priors for a state."""
        return self._priors.get(state_hash, {})

    def clear(self) -> None:
        """Clear all stored priors."""
        self._priors.clear()

    def __len__(self) -> int:
        """Number of states with stored priors."""
        return len(self._priors)


# =============================================================================
# Selection Policy Utilities
# =============================================================================


def select_child_puct(
    node: MCTSNode,
    priors_manager: PriorsManager,
    c_puct: float = 1.25,
) -> tuple[str, MCTSNode] | None:
    """
    Select child using PUCT algorithm.

    Utility function for integrating PUCT with standard MCTSNode.

    Args:
        node: Parent MCTS node
        priors_manager: Manager containing action priors
        c_puct: PUCT exploration constant

    Returns:
        (action, child) tuple or None if no children
    """
    if not node.children:
        return None

    state_hash = node.state.to_hash_key()
    best_score = -float("inf")
    best_action = None
    best_child = None

    sqrt_parent = math.sqrt(node.visits) if node.visits > 0 else 1.0

    for child in node.children:
        action = child.action or ""  # Default to empty string if None
        prior = priors_manager.get_prior(state_hash, action)

        # Calculate PUCT score
        if child.visits == 0:
            score = float("inf") if prior > 0 else c_puct * sqrt_parent
        else:
            q_value = child.value / child.visits if child.visits > 0 else 0.0
            exploration = c_puct * prior * sqrt_parent / (1 + child.visits)
            score = q_value + exploration

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    if best_child is not None and best_action is not None:
        return (best_action, best_child)
    return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # PUCT functions
    "puct",
    "puct_with_virtual_loss",
    # Protocols
    "StateAdapter",
    "ActionFilter",
    # Configuration
    "NeuralPolicyConfig",
    # Policies
    "NeuralRolloutPolicy",
    "FallbackNeuralRolloutPolicy",
    "create_neural_rollout_policy",
    # Priors management
    "PriorsManager",
    "select_child_puct",
    # Utilities
    "is_torch_available",
]

# Conditionally export TorchNeuralRolloutPolicy
if _TORCH_AVAILABLE:
    __all__.append("TorchNeuralRolloutPolicy")
