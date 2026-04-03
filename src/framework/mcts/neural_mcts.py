"""
Neural-Guided Monte Carlo Tree Search (MCTS).

Implements AlphaZero-style MCTS with:
- Policy and value network guidance
- PUCT (Predictor + UCT) selection
- Dirichlet noise for exploration
- Virtual loss for parallel search
- Temperature-based action selection

Based on:
- "Mastering the Game of Go with Deep Neural Networks and Tree Search" (AlphaGo)
- "Mastering Chess and Shogi by Self-Play with a General RL Algorithm" (AlphaZero)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ...training.system_config import MCTSConfig


@dataclass
class GameState:
    """
    Abstract game/problem state interface.

    Users should subclass this for their specific domain.
    """

    def get_legal_actions(self) -> list[Any]:
        """Return list of legal actions from this state."""
        raise NotImplementedError

    def apply_action(self, action: Any) -> GameState:
        """Apply action and return new state."""
        raise NotImplementedError

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        raise NotImplementedError

    def get_reward(self, player: int = 1) -> float:
        """Get reward for the player (1 or -1)."""
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        raise NotImplementedError

    def get_canonical_form(self, player: int) -> GameState:  # noqa: ARG002
        """Get state from perspective of given player."""
        return self

    def get_hash(self) -> str:
        """Get unique hash for this state (for caching)."""
        raise NotImplementedError

    def action_to_index(self, action: Any) -> int:
        """
        Map action to its index in the neural network's action space.

        This method should return the index corresponding to the action
        in the network's policy output vector.

        Default implementation uses string-based mapping for Tic-Tac-Toe style
        actions (e.g., "0,0" -> 0, "0,1" -> 1, etc.). Override this method
        for custom action mappings.

        Args:
            action: The action to map

        Returns:
            Index in the action space (0 to action_size-1)
        """
        # Default implementation for grid-based actions like "row,col"
        if isinstance(action, str) and "," in action:
            row, col = map(int, action.split(","))
            # Assume 3x3 grid by default - override for different sizes
            return row * 3 + col
        # For other action types, assume they are already indices
        return int(action)


class NeuralMCTSNode:
    """
    MCTS node with neural network guidance.

    Stores statistics for PUCT selection and backpropagation.
    """

    def __init__(
        self,
        state: GameState,
        parent: NeuralMCTSNode | None = None,
        action: Any | None = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior = prior  # Prior probability from policy network

        # Statistics
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.virtual_loss: float = 0.0

        # Children: action -> NeuralMCTSNode
        self.children: dict[Any, NeuralMCTSNode] = {}

        # Caching
        self.is_expanded: bool = False
        self.is_terminal: bool = state.is_terminal()

    @property
    def value(self) -> float:
        """Average value (Q-value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(
        self,
        policy_probs: np.ndarray,
        valid_actions: list[Any],
    ):
        """
        Expand node by creating children for all legal actions.

        Args:
            policy_probs: Prior probabilities from policy network
            valid_actions: List of legal actions
        """
        self.is_expanded = True

        for action, prior in zip(valid_actions, policy_probs, strict=True):
            if action not in self.children:
                next_state = self.state.apply_action(action)
                self.children[action] = NeuralMCTSNode(
                    state=next_state,
                    parent=self,
                    action=action,
                    prior=prior,
                )

    def select_child(self, c_puct: float) -> tuple[Any | None, NeuralMCTSNode | None]:
        """
        Select best child using PUCT algorithm.

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant

        Returns:
            (action, child_node) tuple
        """
        best_score = -float("inf")
        best_action = None
        best_child = None

        # Precompute sqrt term for efficiency
        sqrt_parent_visits = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            # Q-value (average value)
            q_value = child.value

            # U-value (exploration bonus)
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count + child.virtual_loss)

            # PUCT score
            puct_score = q_value + u_value

            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def add_virtual_loss(self, virtual_loss: float):
        """Add virtual loss for parallel search."""
        self.virtual_loss += virtual_loss

    def revert_virtual_loss(self, virtual_loss: float):
        """Remove virtual loss after search completes."""
        self.virtual_loss -= virtual_loss

    def update(self, value: float):
        """Update node statistics with search result."""
        self.visit_count += 1
        self.value_sum += value

    def get_action_probs(self, temperature: float = 1.0) -> dict[Any, float]:
        """
        Get action selection probabilities based on visit counts.

        Args:
            temperature: Temperature parameter
                - temperature -> 0: argmax (deterministic)
                - temperature = 1: proportional to visits
                - temperature -> inf: uniform

        Returns:
            Dictionary mapping actions to probabilities
        """
        if not self.children:
            return {}

        if temperature == 0:
            # Deterministic: select most visited
            visits = {action: child.visit_count for action, child in self.children.items()}
            max_visits = max(visits.values())
            best_actions = [a for a, v in visits.items() if v == max_visits]

            # Uniform over best actions
            prob = 1.0 / len(best_actions)
            return {a: (prob if a in best_actions else 0.0) for a in self.children}

        # Temperature-scaled visits
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature != 1.0:
            visit_counts = visit_counts ** (1.0 / temperature)

        # Normalize to probabilities
        probs = visit_counts / visit_counts.sum()

        return dict(zip(actions, probs, strict=True))


class NeuralMCTS:
    """
    Neural-guided MCTS for decision making.

    Combines tree search with neural network evaluation
    using the AlphaZero algorithm.
    """

    def __init__(
        self,
        policy_value_network: nn.Module,
        config: MCTSConfig,
        device: str = "cpu",
    ):
        """
        Initialize neural MCTS.

        Args:
            policy_value_network: Network that outputs (policy, value)
            config: MCTS configuration
            device: Device for neural network
        """
        self.network = policy_value_network
        self.config = config
        self.device = device

        # Caching for network evaluations
        self.cache: dict[str, tuple[np.ndarray, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def add_dirichlet_noise(
        self,
        policy_probs: np.ndarray,
        epsilon: float | None = None,
        alpha: float | None = None,
    ) -> np.ndarray:
        """
        Add Dirichlet noise to policy for exploration (at root only).

        Policy' = (1 - epsilon) * Policy + epsilon * Noise

        Args:
            policy_probs: Original policy probabilities
            epsilon: Mixing parameter (defaults to config)
            alpha: Dirichlet concentration parameter (defaults to config)

        Returns:
            Noised policy probabilities
        """
        epsilon = epsilon or self.config.dirichlet_epsilon
        alpha = alpha or self.config.dirichlet_alpha

        noise = np.random.dirichlet([alpha] * len(policy_probs))
        return (1 - epsilon) * policy_probs + epsilon * noise

    @torch.no_grad()
    async def evaluate_state(self, state: GameState, add_noise: bool = False) -> tuple[np.ndarray, float]:
        """
        Evaluate state using neural network.

        Args:
            state: Game state to evaluate
            add_noise: Whether to add Dirichlet noise (for root exploration)

        Returns:
            (policy_probs, value) tuple
        """
        # Check cache
        state_hash = state.get_hash()
        if not add_noise and state_hash in self.cache:
            self.cache_hits += 1
            return self.cache[state_hash]

        self.cache_misses += 1

        # Get legal actions
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return np.array([]), 0.0

        # Convert state to tensor
        state_tensor = state.to_tensor().unsqueeze(0).to(self.device)

        # Network forward pass
        policy_logits, value = self.network(state_tensor)

        # Convert to numpy (detach to remove gradients)
        policy_logits = policy_logits.squeeze(0).detach().cpu().numpy()
        value = value.item()

        # Proper action masking: Map legal actions to their indices in the action space
        # Create a mask for legal actions
        action_mask = np.full_like(policy_logits, -np.inf)  # Mask all actions initially
        action_indices = []

        # Map legal actions to their network output indices
        for action in legal_actions:
            try:
                action_idx = state.action_to_index(action)
                if 0 <= action_idx < len(policy_logits):
                    action_mask[action_idx] = 0  # Unmask legal actions
                    action_indices.append(action_idx)
            except (ValueError, IndexError, AttributeError) as e:
                # Fallback: if action_to_index fails, use sequential mapping
                print(f"Warning: action_to_index failed for action {action}: {e}")
                action_indices = list(range(len(legal_actions)))
                action_mask = np.full_like(policy_logits, -np.inf)
                action_mask[action_indices] = 0
                break

        # Apply mask before softmax for numerical stability
        masked_logits = policy_logits + action_mask

        # Compute softmax over legal actions only
        exp_logits = np.exp(masked_logits - np.max(masked_logits))  # Subtract max for stability
        policy_probs_full = exp_logits / exp_logits.sum()

        # Extract probabilities for legal actions in order
        policy_probs = policy_probs_full[action_indices]

        # Normalize to ensure probabilities sum to 1 (handle numerical errors)
        if policy_probs.sum() > 0:
            policy_probs = policy_probs / policy_probs.sum()
        else:
            # Fallback: uniform distribution over legal actions
            policy_probs = np.ones(len(legal_actions)) / len(legal_actions)

        # Add Dirichlet noise if requested (root exploration)
        if add_noise:
            policy_probs = self.add_dirichlet_noise(policy_probs)

        # Cache result (without noise)
        if not add_noise:
            self.cache[state_hash] = (policy_probs, value)

        return policy_probs, value

    async def search(
        self,
        root_state: GameState,
        num_simulations: int | None = None,
        temperature: float = 1.0,
        add_root_noise: bool = True,
    ) -> tuple[dict[Any, float], NeuralMCTSNode]:
        """
        Run MCTS search from root state.

        Args:
            root_state: Initial state
            num_simulations: Number of MCTS simulations
            temperature: Temperature for action selection
            add_root_noise: Whether to add Dirichlet noise to root

        Returns:
            (action_probs, root_node) tuple
        """
        num_simulations = num_simulations or self.config.num_simulations

        # Create root node
        root = NeuralMCTSNode(state=root_state)

        # Expand root
        policy_probs, _ = await self.evaluate_state(root_state, add_noise=add_root_noise)
        legal_actions = root_state.get_legal_actions()
        root.expand(policy_probs, legal_actions)

        # Run simulations
        for _ in range(num_simulations):
            await self._simulate(root)

        # Get action probabilities
        action_probs = root.get_action_probs(temperature)

        return action_probs, root

    async def _simulate(self, node: NeuralMCTSNode) -> float:
        """
        Run single MCTS simulation (select, expand, evaluate, backpropagate).

        Args:
            node: Root node for this simulation

        Returns:
            Value from this simulation
        """
        path: list[NeuralMCTSNode] = []

        # Selection: traverse tree using PUCT
        current = node
        while current.is_expanded and not current.is_terminal:
            # Add virtual loss for parallel search
            current.add_virtual_loss(self.config.virtual_loss)
            path.append(current)

            # Select best child
            _, selected = current.select_child(self.config.c_puct)
            assert selected is not None, "select_child returned None"
            current = selected

        # Add leaf to path
        path.append(current)
        current.add_virtual_loss(self.config.virtual_loss)

        # Evaluate leaf node
        if current.is_terminal:
            # Terminal node: use game result
            value = current.state.get_reward()
        else:
            # Non-terminal: expand and evaluate with network
            policy_probs, value = await self.evaluate_state(current.state, add_noise=False)

            if not current.is_expanded:
                legal_actions = current.state.get_legal_actions()
                current.expand(policy_probs, legal_actions)

        # Backpropagate
        for node_in_path in reversed(path):
            node_in_path.update(value)
            node_in_path.revert_virtual_loss(self.config.virtual_loss)

            # Flip value for opponent
            value = -value

        return value

    def select_action(
        self,
        action_probs: dict[Any, float],
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Any:
        """
        Select action from probability distribution.

        Args:
            action_probs: Action probability dictionary
            temperature: Temperature (unused if deterministic=True)
            deterministic: If True, select action with highest probability

        Returns:
            Selected action
        """
        if not action_probs:
            return None

        actions = list(action_probs.keys())
        probs = list(action_probs.values())

        if deterministic or temperature == 0:
            return actions[np.argmax(probs)]

        # Sample from distribution
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def clear_cache(self):
        """Clear the evaluation cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }


# Training data collection
@dataclass
class MCTSExample:
    """Training example from MCTS self-play."""

    state: torch.Tensor  # State representation
    policy_target: np.ndarray  # Target policy (visit counts)
    value_target: float  # Target value (game outcome)
    player: int  # Player to move (1 or -1)


class SelfPlayCollector:
    """
    Collect training data from self-play games.

    Uses MCTS to generate high-quality training examples.
    """

    def __init__(
        self,
        mcts: NeuralMCTS,
        config: MCTSConfig,
    ):
        self.mcts = mcts
        self.config = config

    async def play_game(
        self,
        initial_state: GameState,
        temperature_threshold: int | None = None,
    ) -> list[MCTSExample]:
        """
        Play a single self-play game.

        Args:
            initial_state: Starting game state
            temperature_threshold: Move number to switch to greedy play

        Returns:
            List of training examples from the game
        """
        temperature_threshold = temperature_threshold or self.config.temperature_threshold

        examples: list[MCTSExample] = []
        state = initial_state
        player = 1  # Current player (1 or -1)
        move_count = 0

        while not state.is_terminal():
            # Determine temperature
            temperature = (
                self.config.temperature_init if move_count < temperature_threshold else self.config.temperature_final
            )

            # Run MCTS
            action_probs, root = await self.mcts.search(state, temperature=temperature, add_root_noise=True)

            # Store training example
            # Convert action probs to array for all actions
            probs = np.array(list(action_probs.values()))

            examples.append(
                MCTSExample(
                    state=state.to_tensor(),
                    policy_target=probs,
                    value_target=0.0,  # Will be filled with game outcome
                    player=player,
                )
            )

            # Select and apply action
            action = self.mcts.select_action(action_probs, temperature=temperature)
            state = state.apply_action(action)

            # Switch player
            player = -player
            move_count += 1

        # Get game outcome
        outcome = state.get_reward()

        # Assign values to examples
        for example in examples:
            # Value is from perspective of the player who made the move
            example.value_target = outcome if example.player == 1 else -outcome

        return examples

    async def generate_batch(self, num_games: int, initial_state_fn: Callable[[], GameState]) -> list[MCTSExample]:
        """
        Generate a batch of training examples from multiple games.

        Args:
            num_games: Number of games to play
            initial_state_fn: Function that returns initial game state

        Returns:
            Combined list of training examples
        """
        all_examples = []

        for _ in range(num_games):
            initial_state = initial_state_fn()
            examples = await self.play_game(initial_state)
            all_examples.extend(examples)

            # Clear cache periodically
            if len(self.mcts.cache) > 10000:
                self.mcts.clear_cache()

        return all_examples
