"""
MCTS Core Module - Deterministic, testable Monte Carlo Tree Search implementation.

Features:
- Seeded RNG for deterministic behavior
- Progressive widening to control branching factor
- Simulation result caching with hashable state keys
- Clear separation of MCTS phases: select, expand, simulate, backpropagate
- Support for parallel rollouts with asyncio.Semaphore
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .policies import RolloutPolicy, SelectionPolicy, ucb1

logger = logging.getLogger(__name__)


@dataclass
class MCTSState:
    """Hashable state representation for caching."""

    state_id: str
    features: dict[str, Any] = field(default_factory=dict)

    def to_hash_key(self) -> str:
        """Generate a hashable key for this state."""
        # Sort features for deterministic hashing
        feature_str = str(sorted(self.features.items()))
        combined = f"{self.state_id}:{feature_str}"
        return hashlib.sha256(combined.encode()).hexdigest()


class MCTSNode:
    """
    Monte Carlo Tree Search node with proper state management.

    Attributes:
        state: The state this node represents
        parent: Parent node (None for root)
        action: Action taken to reach this node from parent
        children: List of child nodes
        visits: Number of times this node has been visited
        value_sum: Total accumulated value from simulations
        rng: Seeded random number generator for deterministic behavior
    """

    def __init__(
        self,
        state: MCTSState,
        parent: MCTSNode | None = None,
        action: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.terminal: bool = False
        self.expanded_actions: set = set()
        self.available_actions: list[str] = []

        # Track depth for O(1) tree statistics
        self.depth: int = 0 if parent is None else parent.depth + 1

        # Use provided RNG or create default
        self._rng = rng or np.random.default_rng()

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all available actions have been expanded."""
        return len(self.expanded_actions) >= len(self.available_actions)

    def select_child(self, exploration_weight: float = 1.414) -> MCTSNode:
        """
        Select best child using UCB1 policy.

        Args:
            exploration_weight: Exploration constant (c in UCB1)

        Returns:
            Best child node according to UCB1
        """
        if not self.children:
            raise ValueError("No children to select from")

        best_child = None
        best_score = float("-inf")

        for child in self.children:
            score = ucb1(
                value_sum=child.value_sum,
                visits=child.visits,
                parent_visits=self.visits,
                c=exploration_weight,
            )
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None, "No best child found despite non-empty children"
        return best_child

    def add_child(self, action: str, child_state: MCTSState) -> MCTSNode:
        """
        Add a child node for the given action.

        Args:
            action: Action taken to reach child state
            child_state: State of the child node

        Returns:
            Newly created child node
        """
        child = MCTSNode(
            state=child_state,
            parent=self,
            action=action,
            rng=self._rng,
        )
        self.children.append(child)
        self.expanded_actions.add(action)
        return child

    def get_unexpanded_action(self) -> str | None:
        """Get a random unexpanded action."""
        unexpanded = [a for a in self.available_actions if a not in self.expanded_actions]
        if not unexpanded:
            return None
        return str(self._rng.choice(unexpanded))

    def __repr__(self) -> str:
        return (
            f"MCTSNode(state={self.state.state_id}, "
            f"visits={self.visits}, value={self.value:.3f}, "
            f"children={len(self.children)})"
        )


class MCTSEngine:
    """
    Main MCTS engine with deterministic behavior and advanced features.

    Features:
    - Seeded RNG for reproducibility
    - Progressive widening to control branching
    - Simulation result caching
    - Parallel rollout support with semaphore
    """

    def __init__(
        self,
        seed: int = 42,
        exploration_weight: float = 1.414,
        progressive_widening_k: float = 1.0,
        progressive_widening_alpha: float = 0.5,
        max_parallel_rollouts: int = 4,
        cache_size_limit: int = 10000,
    ):
        """
        Initialize MCTS engine.

        Args:
            seed: Random seed for deterministic behavior
            exploration_weight: UCB1 exploration constant
            progressive_widening_k: Progressive widening coefficient
            progressive_widening_alpha: Progressive widening exponent
            max_parallel_rollouts: Maximum concurrent rollouts
            cache_size_limit: Maximum number of cached simulation results
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.exploration_weight = exploration_weight
        self.progressive_widening_k = progressive_widening_k
        self.progressive_widening_alpha = progressive_widening_alpha
        logger.debug(
            "MCTSEngine initialized: seed=%d, c=%.3f, pw_k=%.2f, pw_alpha=%.2f",
            seed,
            exploration_weight,
            progressive_widening_k,
            progressive_widening_alpha,
        )

        # Parallel rollout control
        self.max_parallel_rollouts = max_parallel_rollouts
        self._semaphore: asyncio.Semaphore | None = None

        # Simulation cache: state_hash -> (value, visit_count)
        # Using OrderedDict for LRU eviction
        self._simulation_cache: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self.cache_size_limit = cache_size_limit

        # Statistics
        self.total_simulations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        # Cached tree statistics for O(1) retrieval
        self._cached_tree_depth: int = 0
        self._cached_node_count: int = 0

    def reset_seed(self, seed: int) -> None:
        """Reset the random seed for new experiment."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def clear_cache(self) -> None:
        """Clear simulation result cache."""
        self._simulation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def should_expand(self, node: MCTSNode) -> bool:
        """
        Check if node should expand based on progressive widening.

        Progressive widening formula: expand when visits > k * n^alpha
        where n is the number of children.

        This prevents excessive branching and focuses search on promising areas.
        """
        if node.terminal or node.is_fully_expanded:
            return False

        num_children = len(node.children)
        threshold = self.progressive_widening_k * (num_children**self.progressive_widening_alpha)

        return bool(node.visits > threshold)

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        MCTS Selection Phase: traverse tree to find leaf node.

        Uses UCB1 to balance exploration and exploitation.
        """
        while node.children and not node.terminal:
            # Check if we should expand instead of selecting
            if self.should_expand(node):
                break
            node = node.select_child(self.exploration_weight)
        return node

    def expand(
        self,
        node: MCTSNode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
    ) -> MCTSNode:
        """
        MCTS Expansion Phase: add a new child node.

        Args:
            node: Node to expand
            action_generator: Function to generate available actions
            state_transition: Function to compute next state from action

        Returns:
            Newly expanded child node, or original node if cannot expand
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

        # Create child state
        child_state = state_transition(node.state, action)
        child = node.add_child(action, child_state)

        # Update cached node count for O(1) retrieval
        self._cached_node_count += 1

        return child

    async def simulate(
        self,
        node: MCTSNode,
        rollout_policy: RolloutPolicy,
        max_depth: int = 10,
    ) -> float:
        """
        MCTS Simulation Phase: evaluate node value through rollout.

        Uses caching to avoid redundant simulations.

        Args:
            node: Node to simulate from
            rollout_policy: Policy for rollout evaluation
            max_depth: Maximum rollout depth

        Returns:
            Estimated value from simulation
        """
        # Check cache first
        state_hash = node.state.to_hash_key()
        if state_hash in self._simulation_cache:
            cached_value, cached_count = self._simulation_cache[state_hash]
            # Move to end for LRU (most recently used)
            self._simulation_cache.move_to_end(state_hash)
            self.cache_hits += 1
            # Return cached average with small noise for exploration
            noise = self.rng.normal(0, 0.01)
            return float(cached_value + noise)

        self.cache_misses += 1

        # Acquire semaphore for parallel control
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_parallel_rollouts)

        async with self._semaphore:
            # Perform rollout
            value = await rollout_policy.evaluate(
                state=node.state,
                rng=self.rng,
                max_depth=max_depth,
            )

        self.total_simulations += 1

        # Update cache with LRU eviction
        if state_hash in self._simulation_cache:
            # Update existing cache entry with running average
            old_value, old_count = self._simulation_cache[state_hash]
            new_count = old_count + 1
            new_value = (old_value * old_count + value) / new_count
            self._simulation_cache[state_hash] = (new_value, new_count)
            # Move to end for LRU (most recently used)
            self._simulation_cache.move_to_end(state_hash)
        else:
            # Evict oldest entry if cache is full
            if len(self._simulation_cache) >= self.cache_size_limit:
                # Remove the first item (least recently used)
                self._simulation_cache.popitem(last=False)
                self.cache_evictions += 1
            # Add new entry at the end (most recently used)
            self._simulation_cache[state_hash] = (value, 1)

        return value

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        MCTS Backpropagation Phase: update ancestor statistics.

        Args:
            node: Leaf node to start backpropagation
            value: Value to propagate up the tree
        """
        # Update cached tree depth if this node is deeper than current max
        if node.depth > self._cached_tree_depth:
            self._cached_tree_depth = node.depth

        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    async def run_iteration(
        self,
        root: MCTSNode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int = 10,
    ) -> None:
        """
        Run a single MCTS iteration (select, expand, simulate, backpropagate).

        Args:
            root: Root node of the tree
            action_generator: Function to generate actions
            state_transition: Function to compute state transitions
            rollout_policy: Policy for rollout evaluation
            max_rollout_depth: Maximum depth for rollouts
        """
        # Selection
        leaf = self.select(root)

        # Expansion
        if not leaf.terminal and leaf.visits > 0:
            leaf = self.expand(leaf, action_generator, state_transition)

        # Simulation
        value = await self.simulate(leaf, rollout_policy, max_rollout_depth)

        # Backpropagation
        self.backpropagate(leaf, value)

    async def search(
        self,
        root: MCTSNode,
        num_iterations: int,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int = 10,
        selection_policy: SelectionPolicy = SelectionPolicy.MAX_VISITS,
        early_termination_threshold: float = 0.95,
        min_iterations_before_termination: int = 50,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Run MCTS search for specified number of iterations with early termination.

        Args:
            root: Root node to search from
            num_iterations: Maximum number of MCTS iterations
            action_generator: Function to generate available actions
            state_transition: Function to compute state transitions
            rollout_policy: Policy for rollout simulation
            max_rollout_depth: Maximum rollout depth
            selection_policy: Policy for final action selection
            early_termination_threshold: Stop if best action has this fraction of visits
            min_iterations_before_termination: Minimum iterations before checking termination

        Returns:
            Tuple of (best_action, statistics_dict)
        """
        logger.info(
            "MCTS search started: max_iterations=%d, policy=%s, state=%s",
            num_iterations,
            selection_policy.value,
            root.state.state_id,
        )

        # Reset cached tree statistics for new search
        self._cached_tree_depth = 0
        self._cached_node_count = 1  # Start with root node

        # Initialize root's available actions
        if not root.available_actions:
            root.available_actions = action_generator(root.state)

        # Track actual iterations run
        iterations_run = 0
        early_terminated = False

        # Run iterations with early termination check
        for iteration in range(num_iterations):
            await self.run_iteration(
                root=root,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
                max_rollout_depth=max_rollout_depth,
            )
            iterations_run = iteration + 1

            # Check for early termination after minimum iterations
            if iterations_run >= min_iterations_before_termination and self._should_terminate_early(
                root,
                early_termination_threshold,
            ):
                early_terminated = True
                break

        # Select best action based on policy
        best_action = self._select_best_action(root, selection_policy)

        # Compute statistics
        stats = self._compute_statistics(root, iterations_run)

        # Add early termination info to stats
        stats["early_terminated"] = early_terminated
        stats["iterations_run"] = iterations_run
        stats["max_iterations"] = num_iterations
        if early_terminated:
            stats["termination_reason"] = "visit_threshold_reached"
            logger.info("MCTS search early-terminated at iteration %d/%d", iterations_run, num_iterations)

        logger.info(
            "MCTS search complete: best_action=%s, iterations=%d, cache_hit_rate=%.2f",
            best_action,
            iterations_run,
            stats.get("cache_hit_rate", 0.0),
        )

        return best_action, stats

    def _should_terminate_early(
        self,
        root: MCTSNode,
        threshold: float,
    ) -> bool:
        """
        Check if early termination conditions are met.

        Early termination occurs when the best action has accumulated
        a fraction of visits above the threshold, indicating convergence.

        Args:
            root: Root node with children
            threshold: Fraction of visits for best action to trigger termination

        Returns:
            True if early termination should occur
        """
        if not root.children or root.visits == 0:
            return False

        # Find the child with most visits
        max_child_visits = max(child.visits for child in root.children)

        # Calculate the fraction of total visits on best child
        visit_fraction = max_child_visits / root.visits

        return visit_fraction >= threshold

    def _select_best_action(
        self,
        root: MCTSNode,
        policy: SelectionPolicy,
    ) -> str | None:
        """
        Select the best action from root based on selection policy.

        Args:
            root: Root node with children
            policy: Selection policy to use

        Returns:
            Best action string or None if no children
        """
        if not root.children:
            return None

        if policy == SelectionPolicy.MAX_VISITS:
            # Most robust: select action with most visits
            best_child = max(root.children, key=lambda c: c.visits)
        elif policy == SelectionPolicy.MAX_VALUE:
            # Greedy: select action with highest average value
            best_child = max(root.children, key=lambda c: c.value)
        elif policy == SelectionPolicy.ROBUST_CHILD:
            # Robust: require both high visits and high value
            # Normalize both metrics and combine
            max_visits = max(c.visits for c in root.children)
            max_value = max(c.value for c in root.children) or 1.0

            def robust_score(child):
                visit_score = child.visits / max_visits if max_visits > 0 else 0
                value_score = child.value / max_value if max_value > 0 else 0
                return 0.5 * visit_score + 0.5 * value_score

            best_child = max(root.children, key=robust_score)
        else:
            # Default to max visits
            best_child = max(root.children, key=lambda c: c.visits)

        return best_child.action

    def _compute_statistics(
        self,
        root: MCTSNode,
        num_iterations: int,
    ) -> dict[str, Any]:
        """
        Compute comprehensive MCTS statistics.

        Args:
            root: Root node
            num_iterations: Number of iterations run

        Returns:
            Dictionary of statistics
        """
        # Best child info
        best_child = None
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)

        # Action statistics
        action_stats = {}
        for child in root.children:
            action_stats[child.action] = {
                "visits": child.visits,
                "value": child.value,
                "value_sum": child.value_sum,
                "num_children": len(child.children),
            }

        return {
            "iterations": num_iterations,
            "root_visits": root.visits,
            "root_value": root.value,
            "num_children": len(root.children),
            "best_action": best_child.action if best_child else None,
            "best_action_visits": best_child.visits if best_child else 0,
            "best_action_value": best_child.value if best_child else 0.0,
            "action_stats": action_stats,
            "total_simulations": self.total_simulations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "cache_size": len(self._simulation_cache),
            "seed": self.seed,
        }

    def get_tree_depth(self, node: MCTSNode) -> int:
        """Get maximum depth of the tree from given node.

        Uses iterative BFS to avoid stack overflow for large trees (5000+ nodes).
        Each level of the tree is processed iteratively, tracking depth as we go.
        """
        if not node.children:
            return 0

        from collections import deque

        max_depth = 0
        # Queue contains tuples of (node, depth)
        queue = deque([(node, 0)])

        while queue:
            current_node, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            for child in current_node.children:
                queue.append((child, depth + 1))

        return max_depth

    def count_nodes(self, node: MCTSNode) -> int:
        """Count total number of nodes in tree.

        Uses iterative BFS to avoid stack overflow for large trees (5000+ nodes).
        Traverses all nodes in the tree using a queue-based approach.
        """
        from collections import deque

        count = 0
        queue = deque([node])

        while queue:
            current_node = queue.popleft()
            count += 1

            for child in current_node.children:
                queue.append(child)

        return count

    def get_cached_tree_depth(self) -> int:
        """
        Get cached maximum tree depth in O(1) time.

        Returns:
            Maximum depth of tree from last search
        """
        return self._cached_tree_depth

    def get_cached_node_count(self) -> int:
        """
        Get cached total node count in O(1) time.

        Returns:
            Total number of nodes in tree from last search
        """
        return self._cached_node_count
