"""
Progressive Widening and RAVE for MCTS.

This module implements:
1. Progressive Widening: Control branching factor in large/continuous action spaces
2. RAVE (Rapid Action Value Estimation): Share value estimates via AMAF
3. Hybrid UCB + RAVE selection for faster convergence

Progressive Widening expands children based on visit count:
    Expand when: N(s) > k * |C(s)|^α

RAVE uses All-Moves-As-First (AMAF) to accelerate value learning:
    Q(s,a) = (1 - β) * Q_UCB(s,a) + β * Q_AMAF(s,a)

Features:
- Adaptive progressive widening with dynamic k parameter
- RAVE statistics tracking and β decay
- Hybrid selection combining UCB1 and RAVE
- Action filtering for large action spaces
- Integration with standard MCTS

Based on:
- "Bandit Based Monte-Carlo Planning" (Kocsis & Szepesvári, 2006)
- "Modification of UCT with Patterns in Monte-Carlo Go" (Gelly & Silver, 2007)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .core import MCTSNode, MCTSState
from .policies import RolloutPolicy


@dataclass
class ProgressiveWideningConfig:
    """Configuration for progressive widening."""

    k: float = 1.0
    """Coefficient controlling expansion threshold."""

    alpha: float = 0.5
    """Exponent controlling growth rate (0 < alpha < 1)."""

    adaptive: bool = False
    """Whether to adapt k based on search progress."""

    k_min: float = 0.5
    """Minimum k value for adaptive widening."""

    k_max: float = 3.0
    """Maximum k value for adaptive widening."""

    def should_expand(self, visits: int, num_children: int) -> bool:
        """
        Check if node should expand another child.

        Args:
            visits: Number of visits to node
            num_children: Current number of children

        Returns:
            True if should expand, False otherwise
        """
        if num_children == 0:
            return True  # Always expand first child

        threshold = self.k * (num_children**self.alpha)
        return bool(visits > threshold)

    def min_visits_for_next_child(self, num_children: int) -> int:
        """
        Calculate minimum visits needed to expand next child.

        Args:
            num_children: Current number of children

        Returns:
            Minimum visit count for expansion
        """
        threshold = self.k * (num_children**self.alpha)
        return int(math.ceil(threshold)) + 1


@dataclass
class RAVEConfig:
    """Configuration for RAVE (Rapid Action Value Estimation)."""

    rave_constant: float = 300.0
    """Equivalence parameter for β computation (k in literature)."""

    enable_rave: bool = True
    """Whether RAVE is enabled."""

    min_visits_for_rave: int = 5
    """Minimum visits before using RAVE."""

    def compute_beta(self, node_visits: int, rave_visits: int) -> float:
        """
        Compute mixing parameter β for RAVE.

        β = rave_visits / (visits + rave_visits + 4*k²*visits*rave_visits/10000)

        As visits increase, β → 0 (rely more on UCB).

        Args:
            node_visits: Number of standard visits
            rave_visits: Number of RAVE visits

        Returns:
            Beta value in [0, 1]
        """
        if not self.enable_rave or rave_visits < self.min_visits_for_rave:
            return 0.0

        denominator = node_visits + rave_visits + 4 * (self.rave_constant**2) * node_visits * rave_visits / 10000.0

        if denominator == 0:
            return 0.0

        beta = rave_visits / denominator
        return min(1.0, max(0.0, beta))


class RAVENode(MCTSNode):
    """
    MCTS node extended with RAVE statistics.

    Tracks both standard UCB statistics and AMAF (All-Moves-As-First) statistics
    for faster value learning through move-ordering heuristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RAVE/AMAF statistics per action
        self.rave_visits: dict[str, int] = {}
        self.rave_value_sum: dict[str, float] = {}

    def update_rave(self, action: str, value: float) -> None:
        """
        Update RAVE statistics for action.

        Args:
            action: Action to update
            value: Value to add
        """
        if action not in self.rave_visits:
            self.rave_visits[action] = 0
            self.rave_value_sum[action] = 0.0

        self.rave_visits[action] += 1
        self.rave_value_sum[action] += value

    def get_rave_value(self, action: str) -> float:
        """
        Get AMAF value estimate for action.

        Args:
            action: Action to query

        Returns:
            AMAF value in [0, 1] or 0.5 if no data
        """
        if action not in self.rave_visits or self.rave_visits[action] == 0:
            return 0.5

        return self.rave_value_sum[action] / self.rave_visits[action]

    def get_rave_visits(self, action: str) -> int:
        """
        Get RAVE visit count for action.

        Args:
            action: Action to query

        Returns:
            RAVE visit count
        """
        return self.rave_visits.get(action, 0)

    def select_child_rave(
        self,
        rave_config: RAVEConfig,
        exploration_weight: float = 1.414,
    ) -> RAVENode:
        """
        Select child using hybrid UCB + RAVE.

        Args:
            rave_config: RAVE configuration
            exploration_weight: UCB exploration constant

        Returns:
            Selected child node
        """
        if not self.children:
            raise ValueError("No children to select from")

        best_score = float("-inf")
        best_child: RAVENode | None = None

        for child in self.children:
            rave_child = cast(RAVENode, child)
            # Unvisited nodes get priority
            if rave_child.visits == 0:
                return rave_child

            # Standard UCB score
            ucb_exploitation = rave_child.value
            ucb_exploration = exploration_weight * math.sqrt(math.log(self.visits) / rave_child.visits)
            ucb_score = ucb_exploitation + ucb_exploration

            # RAVE score
            child_action = rave_child.action or ""
            rave_visits = self.get_rave_visits(child_action)
            rave_value = self.get_rave_value(child_action)

            # Compute β mixing parameter
            beta = rave_config.compute_beta(rave_child.visits, rave_visits)

            # Hybrid score: (1-β)*UCB + β*RAVE
            score = (1 - beta) * ucb_score + beta * rave_value

            if score > best_score:
                best_score = score
                best_child = rave_child

        assert best_child is not None, "No best child found despite non-empty children"
        return best_child


class ProgressiveWideningEngine:
    """
    MCTS engine with progressive widening and optional RAVE.

    Combines:
    - Progressive widening for controlled expansion
    - RAVE for faster value learning
    - Adaptive parameters based on search progress
    """

    def __init__(
        self,
        pw_config: ProgressiveWideningConfig | None = None,
        rave_config: RAVEConfig | None = None,
        exploration_weight: float = 1.414,
        seed: int = 42,
    ):
        """
        Initialize progressive widening MCTS engine.

        Args:
            pw_config: Progressive widening configuration
            rave_config: RAVE configuration
            exploration_weight: UCB1 exploration constant
            seed: Random seed for deterministic behavior
        """
        self.pw_config = pw_config or ProgressiveWideningConfig()
        self.rave_config = rave_config or RAVEConfig()
        self.exploration_weight = exploration_weight
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Adaptive PW state
        self.value_variance_history: list[float] = []

    def should_expand(self, node: RAVENode) -> bool:
        """
        Check if node should expand based on progressive widening.

        Args:
            node: Node to check

        Returns:
            True if should expand, False otherwise
        """
        if node.terminal:
            return False

        if not node.available_actions:
            return False

        if len(node.children) >= len(node.available_actions):
            return False  # Already fully expanded

        return self.pw_config.should_expand(node.visits, len(node.children))

    def select(self, node: RAVENode) -> RAVENode:
        """
        MCTS Selection Phase with RAVE.

        Args:
            node: Current node

        Returns:
            Selected leaf node
        """
        while node.children and not node.terminal:
            # Check if we should expand instead
            if self.should_expand(node):
                break

            # Select child using RAVE-enhanced UCB
            node = node.select_child_rave(self.rave_config, self.exploration_weight)

        return node

    def expand(
        self,
        node: RAVENode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
    ) -> RAVENode:
        """
        MCTS Expansion Phase with progressive widening.

        Args:
            node: Node to expand
            action_generator: Function to generate actions
            state_transition: Function to compute state transitions

        Returns:
            Newly expanded child or original node if cannot expand
        """
        if node.terminal:
            return node

        # Generate available actions if not yet done
        if not node.available_actions:
            node.available_actions = action_generator(node.state)

        if not node.available_actions:
            node.terminal = True
            return node

        # Check progressive widening
        if not self.should_expand(node):
            return node

        # Get unexpanded action
        action = node.get_unexpanded_action()
        if action is None:
            return node

        # Create child
        child_state = state_transition(node.state, action)
        child = RAVENode(
            state=child_state,
            parent=node,
            action=action,
            rng=self.rng,
        )
        node.children.append(child)
        node.expanded_actions.add(action)

        return child

    async def simulate_with_tracking(
        self,
        node: RAVENode,
        rollout_policy: RolloutPolicy,
        max_depth: int = 10,
    ) -> tuple[float, list[str]]:
        """
        Simulation phase that tracks actions for RAVE.

        Args:
            node: Node to simulate from
            rollout_policy: Rollout policy
            max_depth: Maximum simulation depth

        Returns:
            Tuple of (value, actions_taken)
        """
        # Standard simulation using consistent keyword argument style
        value = await rollout_policy.evaluate(
            state=node.state,
            rng=self.rng,
            max_depth=max_depth,
        )

        # For RAVE, we need to track actions taken
        # This is a simplified version - full implementation would track
        # actual actions from the rollout
        actions_taken = []

        # If rollout policy supports action tracking, get them
        if hasattr(rollout_policy, "last_actions"):
            actions_taken = rollout_policy.last_actions

        return value, actions_taken

    def backpropagate_with_rave(
        self,
        node: RAVENode,
        value: float,
        simulation_actions: list[str],
    ) -> None:
        """
        Backpropagation with RAVE updates.

        Updates both standard UCB statistics and RAVE/AMAF statistics
        for all actions that appeared in the simulation.

        Args:
            node: Leaf node to start backpropagation
            value: Value to propagate
            simulation_actions: Actions taken during simulation (for RAVE)
        """
        current = node

        while current is not None:
            # Standard UCB update
            current.visits += 1
            current.value_sum += value

            # RAVE updates: update all actions from simulation
            if self.rave_config.enable_rave:
                for action in simulation_actions:
                    # Update RAVE statistics for this action
                    current.update_rave(action, value)

            current = current.parent
            value = -value  # Flip for opponent

    async def run_iteration(
        self,
        root: RAVENode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int = 10,
    ) -> None:
        """
        Run single MCTS iteration with PW and RAVE.

        Args:
            root: Root node
            action_generator: Action generation function
            state_transition: State transition function
            rollout_policy: Rollout policy
            max_rollout_depth: Maximum rollout depth
        """
        # 1. Selection
        leaf = self.select(root)

        # 2. Expansion
        if not leaf.terminal and leaf.visits > 0:
            leaf = self.expand(leaf, action_generator, state_transition)

        # 3. Simulation with action tracking
        value, actions = await self.simulate_with_tracking(leaf, rollout_policy, max_rollout_depth)

        # 4. Backpropagation with RAVE
        self.backpropagate_with_rave(leaf, value, actions)

        # 5. Adaptive progressive widening
        if self.pw_config.adaptive:
            self._adapt_progressive_widening(root)

    def _adapt_progressive_widening(self, root: RAVENode) -> None:
        """
        Adapt progressive widening k parameter based on value variance.

        Args:
            root: Root node for variance computation
        """
        if not root.children:
            return

        # Compute value variance across children
        values = [child.value for child in root.children if child.visits > 0]

        if len(values) < 2:
            return

        variance = float(np.var(values))
        self.value_variance_history.append(variance)

        if len(self.value_variance_history) > 100:
            self.value_variance_history.pop(0)

        # Adjust k based on variance
        avg_variance = sum(self.value_variance_history) / len(self.value_variance_history)

        # High variance → reduce k (more exploration)
        if avg_variance > 0.3:
            self.pw_config.k = max(self.pw_config.k_min, self.pw_config.k * 0.95)
        # Low variance → increase k (more exploitation)
        elif avg_variance < 0.1:
            self.pw_config.k = min(self.pw_config.k_max, self.pw_config.k * 1.05)

    async def search(
        self,
        root: RAVENode,
        num_iterations: int,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int = 10,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Run MCTS search with progressive widening and RAVE.

        Args:
            root: Root node
            num_iterations: Number of iterations
            action_generator: Action generation function
            state_transition: State transition function
            rollout_policy: Rollout policy
            max_rollout_depth: Maximum rollout depth

        Returns:
            Tuple of (best_action, statistics)
        """
        # Initialize root
        if not root.available_actions:
            root.available_actions = action_generator(root.state)

        # Run iterations
        for _ in range(num_iterations):
            await self.run_iteration(
                root=root,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
                max_rollout_depth=max_rollout_depth,
            )

        # Select best action
        best_action = self._select_best_action(root)

        # Compute statistics
        stats = self._compute_statistics(root, num_iterations)

        return best_action, stats

    def _select_best_action(self, root: RAVENode) -> str | None:
        """
        Select best action from root.

        Args:
            root: Root node

        Returns:
            Best action or None
        """
        if not root.children:
            return None

        # Select child with most visits (most robust)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _compute_statistics(
        self,
        root: RAVENode,
        num_iterations: int,
    ) -> dict[str, Any]:
        """
        Compute comprehensive statistics.

        Args:
            root: Root node
            num_iterations: Number of iterations run

        Returns:
            Statistics dictionary
        """
        # Action statistics with RAVE info
        action_stats = {}
        for child in root.children:
            child_action = child.action or ""
            rave_visits = root.get_rave_visits(child_action)
            rave_value = root.get_rave_value(child_action)
            beta = self.rave_config.compute_beta(child.visits, rave_visits)

            action_stats[child_action] = {
                "visits": child.visits,
                "value": child.value,
                "rave_visits": rave_visits,
                "rave_value": rave_value,
                "beta": beta,
                "num_children": len(child.children),
            }

        # Best child
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None

        return {
            "iterations": num_iterations,
            "root_visits": root.visits,
            "root_value": root.value,
            "num_children": len(root.children),
            "max_children": len(root.available_actions),
            "expansion_rate": len(root.children) / max(1, len(root.available_actions)),
            "best_action": best_child.action if best_child else None,
            "best_action_visits": best_child.visits if best_child else 0,
            "best_action_value": best_child.value if best_child else 0.0,
            "action_stats": action_stats,
            "pw_k": self.pw_config.k,
            "pw_alpha": self.pw_config.alpha,
            "rave_enabled": self.rave_config.enable_rave,
            "rave_constant": self.rave_config.rave_constant,
        }


# Utility functions
def create_pw_config(
    action_space_size: int,
    adaptive: bool = False,
) -> ProgressiveWideningConfig:
    """
    Create progressive widening config appropriate for action space size.

    Args:
        action_space_size: Number of possible actions
        adaptive: Whether to use adaptive PW

    Returns:
        ProgressiveWideningConfig instance
    """
    if action_space_size < 10:
        # Small action space: aggressive expansion
        return ProgressiveWideningConfig(k=0.5, alpha=0.5, adaptive=adaptive)
    elif action_space_size < 50:
        # Medium action space: balanced
        return ProgressiveWideningConfig(k=1.0, alpha=0.5, adaptive=adaptive)
    elif action_space_size < 200:
        # Large action space: conservative
        return ProgressiveWideningConfig(k=2.0, alpha=0.6, adaptive=adaptive)
    else:
        # Very large/continuous: very conservative
        return ProgressiveWideningConfig(k=5.0, alpha=0.7, adaptive=adaptive)


def create_rave_config(
    domain_has_move_ordering: bool = True,
    domain_complexity: str = "medium",
) -> RAVEConfig:
    """
    Create RAVE config appropriate for domain characteristics.

    Args:
        domain_has_move_ordering: Whether move ordering matters
        domain_complexity: Domain complexity ("low", "medium", "high")

    Returns:
        RAVEConfig instance
    """
    if not domain_has_move_ordering:
        # RAVE doesn't help if move order is critical
        return RAVEConfig(enable_rave=False)

    # Adjust RAVE constant based on complexity
    rave_constants = {
        "low": 100.0,  # Quick RAVE decay
        "medium": 300.0,  # Balanced
        "high": 1000.0,  # Longer RAVE influence
    }

    rave_constant = rave_constants.get(domain_complexity, 300.0)

    return RAVEConfig(
        enable_rave=True,
        rave_constant=rave_constant,
        min_visits_for_rave=5,
    )
