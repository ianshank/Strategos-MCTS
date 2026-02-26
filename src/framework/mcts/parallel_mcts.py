"""
Parallel Monte Carlo Tree Search with Virtual Loss.

This module implements parallel MCTS using virtual loss to prevent thread collisions
and maximize search efficiency. Supports multiple parallelization strategies:

1. Tree Parallelization: Multiple threads traverse same tree with virtual loss
2. Root Parallelization: Independent searches merged at the end
3. Leaf Parallelization: Parallel rollouts from single leaf node

Features:
- Asyncio-based concurrency for Python efficiency
- Virtual loss mechanism to reduce thread collisions
- Lock-free operations where possible
- Adaptive virtual loss tuning
- Performance monitoring and metrics

Based on:
- "Parallel Monte-Carlo Tree Search" (Chaslot et al., 2008)
- "A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm" (Enzenberger & Müller, 2010)
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .core import MCTSNode, MCTSState
from .policies import RolloutPolicy


@dataclass
class ParallelMCTSStats:
    """Statistics for parallel MCTS search."""

    total_simulations: int = 0
    total_duration: float = 0.0
    thread_simulations: dict[int, int] = field(default_factory=dict)
    collision_count: int = 0
    lock_wait_time: float = 0.0
    avg_tree_depth: float = 0.0
    effective_parallelism: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        return {
            "total_simulations": self.total_simulations,
            "total_duration": self.total_duration,
            "thread_simulations": dict(self.thread_simulations),
            "collision_count": self.collision_count,
            "lock_wait_time": self.lock_wait_time,
            "avg_tree_depth": self.avg_tree_depth,
            "effective_parallelism": self.effective_parallelism,
        }


@dataclass
class ParallelMCTSConfig:
    """
    Configuration for Parallel MCTS with Virtual Loss.

    Centralizes all tunable parameters to avoid hardcoded values.
    All parameters can be adjusted via settings or dependency injection.
    """

    # Worker configuration
    num_workers: int = 4
    """Number of parallel worker threads."""

    # Virtual loss configuration
    virtual_loss_value: float = 3.0
    """Initial virtual loss value applied during selection."""

    adaptive_virtual_loss: bool = True
    """Whether to adapt VL based on collision rate."""

    virtual_loss_min: float = 1.0
    """Minimum virtual loss value when adapting."""

    virtual_loss_max: float = 10.0
    """Maximum virtual loss value when adapting."""

    virtual_loss_increase_rate: float = 1.1
    """Multiplier for increasing virtual loss (on high collision)."""

    virtual_loss_decrease_rate: float = 0.9
    """Multiplier for decreasing virtual loss (on low collision)."""

    # Collision detection thresholds
    collision_history_size: int = 100
    """Number of recent operations to track for collision rate."""

    collision_rate_high_threshold: float = 0.3
    """Collision rate above which to increase virtual loss."""

    collision_rate_low_threshold: float = 0.1
    """Collision rate below which to decrease virtual loss."""

    # Search parameters
    exploration_weight: float = 1.414
    """UCB1 exploration constant (c). Higher = more exploration."""

    seed: int = 42
    """Random seed for deterministic behavior."""

    # Lock configuration
    lock_timeout_seconds: float | None = None
    """Timeout for acquiring tree lock. None means no timeout."""

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if self.num_workers < 1:
            errors.append("num_workers must be >= 1")
        if self.virtual_loss_value < 0:
            errors.append("virtual_loss_value must be >= 0")
        if self.virtual_loss_min < 0:
            errors.append("virtual_loss_min must be >= 0")
        if self.virtual_loss_max < self.virtual_loss_min:
            errors.append("virtual_loss_max must be >= virtual_loss_min")
        if not 0 <= self.collision_rate_low_threshold < self.collision_rate_high_threshold <= 1:
            errors.append("collision thresholds must be: 0 <= low < high <= 1")
        if self.collision_history_size < 1:
            errors.append("collision_history_size must be >= 1")
        if self.exploration_weight < 0:
            errors.append("exploration_weight must be >= 0")
        if self.lock_timeout_seconds is not None and self.lock_timeout_seconds <= 0:
            errors.append("lock_timeout_seconds must be None or > 0")

        if errors:
            raise ValueError("Invalid ParallelMCTSConfig:\n" + "\n".join(f"  - {e}" for e in errors))


class VirtualLossNode(MCTSNode):
    """
    MCTS node extended with virtual loss for parallel search.

    Virtual loss temporarily reduces node attractiveness to prevent
    multiple threads from exploring the same path simultaneously.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_loss: float = 0.0
        self.virtual_loss_count: int = 0

    @property
    def effective_visits(self) -> int:
        """Visit count including virtual losses."""
        return self.visits + self.virtual_loss_count

    @property
    def effective_value(self) -> float:
        """Average value accounting for virtual losses."""
        total_visits = self.effective_visits
        if total_visits == 0:
            return 0.0
        # Virtual losses reduce apparent value
        return (self.value_sum - self.virtual_loss) / total_visits

    def add_virtual_loss(self, loss_value: float = 3.0) -> None:
        """
        Add virtual loss before starting search down this path.

        Args:
            loss_value: Virtual loss value to add (typically 1-10)
        """
        self.virtual_loss += loss_value
        self.virtual_loss_count += 1

    def revert_virtual_loss(self, loss_value: float = 3.0) -> None:
        """
        Remove virtual loss after completing search.

        Args:
            loss_value: Virtual loss value to remove (must match add_virtual_loss)
        """
        self.virtual_loss -= loss_value
        self.virtual_loss_count -= 1

    def add_child(self, action: str, child_state: MCTSState) -> VirtualLossNode:
        """
        Add a child node for the given action.

        Args:
            action: Action taken to reach child state
            child_state: State of the child node

        Returns:
            Newly created child node
        """
        child = VirtualLossNode(
            state=child_state,
            parent=self,
            action=action,
            rng=self._rng,
        )
        self.children.append(child)
        self.expanded_actions.add(action)
        return child

    def select_child_with_vl(self, exploration_weight: float = 1.414) -> VirtualLossNode:
        """
        Select best child using UCB1 with virtual loss.

        Args:
            exploration_weight: Exploration constant (c in UCB1)

        Returns:
            Best child node according to UCB1 with virtual loss
        """
        if not self.children:
            raise ValueError("No children to select from")

        best_child = None
        best_score = float("-inf")

        for child in self.children:
            vl_child = cast(VirtualLossNode, child)
            # Use effective values for virtual loss
            if vl_child.effective_visits == 0:
                return vl_child

            exploitation = vl_child.effective_value
            exploration = exploration_weight * math.sqrt(math.log(self.effective_visits) / vl_child.effective_visits)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = vl_child

        assert best_child is not None, "No best child found despite non-empty children"
        return best_child


class ParallelMCTSEngine:
    """
    Parallel MCTS engine using virtual loss for thread coordination.

    Implements tree parallelization where multiple workers traverse the same
    search tree concurrently, using virtual loss to reduce collisions.
    """

    def __init__(
        self,
        config: ParallelMCTSConfig | None = None,
        *,
        # Backwards-compatible individual parameters (deprecated)
        num_workers: int | None = None,
        virtual_loss_value: float | None = None,
        adaptive_virtual_loss: bool | None = None,
        exploration_weight: float | None = None,
        seed: int | None = None,
    ):
        """
        Initialize parallel MCTS engine.

        Args:
            config: Configuration object (preferred).
            num_workers: Number of parallel worker threads (deprecated, use config).
            virtual_loss_value: Initial virtual loss value (deprecated, use config).
            adaptive_virtual_loss: Whether to adapt VL based on collisions (deprecated).
            exploration_weight: UCB1 exploration constant (deprecated, use config).
            seed: Random seed for deterministic behavior (deprecated, use config).

        Note:
            Prefer using the `config` parameter. Individual parameters are
            deprecated and will be removed in a future version.
        """
        # Use config or create from individual parameters for backwards compatibility
        if config is None:
            config = ParallelMCTSConfig(
                num_workers=num_workers if num_workers is not None else 4,
                virtual_loss_value=virtual_loss_value if virtual_loss_value is not None else 3.0,
                adaptive_virtual_loss=adaptive_virtual_loss if adaptive_virtual_loss is not None else True,
                exploration_weight=exploration_weight if exploration_weight is not None else 1.414,
                seed=seed if seed is not None else 42,
            )

        # Validate config to catch invalid values early
        config.validate()

        self._config = config
        self.num_workers = config.num_workers
        self.virtual_loss_value = config.virtual_loss_value
        self.adaptive_virtual_loss = config.adaptive_virtual_loss
        self.exploration_weight = config.exploration_weight
        self.seed = config.seed

        # Shared lock for tree modifications
        self._tree_lock = asyncio.Lock()

        # Statistics tracking
        self.stats = ParallelMCTSStats()
        self._collision_history: list[bool] = []

        # Worker-specific RNGs for deterministic parallel behavior
        self._worker_rngs = {i: np.random.default_rng(self.seed + i) for i in range(self.num_workers)}

    async def parallel_search(
        self,
        root: VirtualLossNode,
        num_simulations: int,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int = 10,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Run parallel MCTS search with virtual loss.

        Args:
            root: Root node to search from
            num_simulations: Total number of simulations across all workers
            action_generator: Function to generate available actions
            state_transition: Function to compute state transitions
            rollout_policy: Policy for rollout simulation
            max_rollout_depth: Maximum rollout depth

        Returns:
            Tuple of (best_action, statistics_dict)
        """
        start_time = time.time()

        # Reset statistics
        self.stats = ParallelMCTSStats()

        # Initialize root
        if not root.available_actions:
            root.available_actions = action_generator(root.state)

        # Distribute simulations across workers
        simulations_per_worker = num_simulations // self.num_workers
        remainder = num_simulations % self.num_workers

        # Create worker tasks
        tasks = []
        for worker_id in range(self.num_workers):
            worker_sims = simulations_per_worker + (1 if worker_id < remainder else 0)
            task = self._worker_thread(
                worker_id=worker_id,
                root=root,
                num_simulations=worker_sims,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
                max_rollout_depth=max_rollout_depth,
            )
            tasks.append(task)

        # Run all workers concurrently
        await asyncio.gather(*tasks)

        # Compute final statistics
        duration = time.time() - start_time
        self.stats.total_duration = duration
        self.stats.total_simulations = num_simulations
        self.stats.effective_parallelism = num_simulations / (duration * 1000) if duration > 0 else 0.0

        # Select best action
        best_action = self._select_best_action(root)

        # Build statistics dictionary
        stats_dict = self._build_stats_dict(root)

        return best_action, stats_dict

    async def _worker_thread(
        self,
        worker_id: int,
        root: VirtualLossNode,
        num_simulations: int,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int,
    ) -> None:
        """
        Single worker thread executing simulations.

        Args:
            worker_id: Unique worker identifier
            root: Root node
            num_simulations: Number of simulations for this worker
            action_generator: Action generation function
            state_transition: State transition function
            rollout_policy: Rollout policy
            max_rollout_depth: Maximum rollout depth
        """
        for _ in range(num_simulations):
            await self._run_simulation(
                worker_id=worker_id,
                root=root,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
                max_rollout_depth=max_rollout_depth,
            )

            # Update worker statistics
            self.stats.thread_simulations[worker_id] = self.stats.thread_simulations.get(worker_id, 0) + 1

    async def _run_simulation(
        self,
        worker_id: int,
        root: VirtualLossNode,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int,
    ) -> None:
        """
        Run single MCTS simulation with virtual loss.

        Args:
            worker_id: Worker identifier
            root: Root node
            action_generator: Action generation function
            state_transition: State transition function
            rollout_policy: Rollout policy
            max_rollout_depth: Maximum rollout depth
        """
        path: list[VirtualLossNode] = []
        collision_detected = False

        # 1. SELECTION with virtual loss
        current = root
        lock_start = time.time()
        async with self._tree_lock:
            self.stats.lock_wait_time += time.time() - lock_start

            while current.children and not current.terminal:
                # Check for collision (another thread selected this node)
                if current.virtual_loss_count > 0:
                    collision_detected = True
                    self.stats.collision_count += 1

                current.add_virtual_loss(self.virtual_loss_value)
                path.append(current)

                current = current.select_child_with_vl(self.exploration_weight)

            # Add leaf to path
            current.add_virtual_loss(self.virtual_loss_value)
            path.append(current)

        # Track collision for adaptive VL
        if self.adaptive_virtual_loss:
            self._collision_history.append(collision_detected)
            if len(self._collision_history) > self._config.collision_history_size:
                self._collision_history.pop(0)
                self._adapt_virtual_loss()

        # 2. EXPANSION (may need lock if creating new nodes)
        if not current.terminal and current.visits > 0:
            lock_start = time.time()
            async with self._tree_lock:
                self.stats.lock_wait_time += time.time() - lock_start

                # Double-check after acquiring lock
                if not current.available_actions:
                    current.available_actions = action_generator(current.state)

                if not current.is_fully_expanded and current.available_actions:
                    action = current.get_unexpanded_action()
                    if action is not None:
                        child_state = state_transition(current.state, action)
                        current.add_child(action, child_state)

        # 3. SIMULATION (lock-free, fully parallel)
        rng = self._worker_rngs[worker_id]
        value = await rollout_policy.evaluate(
            state=current.state,
            rng=rng,
            max_depth=max_rollout_depth,
        )

        # 4. BACKPROPAGATION (requires lock)
        lock_start = time.time()
        async with self._tree_lock:
            self.stats.lock_wait_time += time.time() - lock_start

            for node in reversed(path):
                node.visits += 1
                node.value_sum += value
                node.revert_virtual_loss(self.virtual_loss_value)
                value = -value  # Flip for opponent perspective

    def _adapt_virtual_loss(self) -> None:
        """Adapt virtual loss value based on collision rate using config thresholds."""
        if not self._collision_history:
            return

        collision_rate = sum(self._collision_history) / len(self._collision_history)

        # Increase VL if high collision rate
        if collision_rate > self._config.collision_rate_high_threshold:
            self.virtual_loss_value = min(
                self._config.virtual_loss_max,
                self.virtual_loss_value * self._config.virtual_loss_increase_rate,
            )
        # Decrease VL if low collision rate
        elif collision_rate < self._config.collision_rate_low_threshold:
            self.virtual_loss_value = max(
                self._config.virtual_loss_min,
                self.virtual_loss_value * self._config.virtual_loss_decrease_rate,
            )

    def _select_best_action(self, root: VirtualLossNode) -> str | None:
        """
        Select best action from root based on visit counts.

        Args:
            root: Root node

        Returns:
            Best action or None if no children
        """
        if not root.children:
            return None

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _build_stats_dict(self, root: VirtualLossNode) -> dict[str, Any]:
        """
        Build comprehensive statistics dictionary.

        Args:
            root: Root node

        Returns:
            Statistics dictionary
        """
        # Action statistics
        action_stats = {}
        for child in root.children:
            vl_child = cast(VirtualLossNode, child)
            action_stats[vl_child.action] = {
                "visits": vl_child.visits,
                "value": vl_child.value,
                "effective_visits": vl_child.effective_visits,
            }

        # Compute tree depth
        max_depth = self._compute_tree_depth(root)

        return {
            "root_visits": root.visits,
            "root_value": root.value,
            "num_children": len(root.children),
            "tree_depth": max_depth,
            "action_stats": action_stats,
            "parallel_stats": self.stats.to_dict(),
            "virtual_loss_value": self.virtual_loss_value,
        }

    def _compute_tree_depth(self, node: VirtualLossNode) -> int:
        """Compute maximum tree depth from node."""
        if not node.children:
            return 0

        max_depth = 0
        for child in node.children:
            depth = 1 + self._compute_tree_depth(cast(VirtualLossNode, child))
            max_depth = max(max_depth, depth)

        return max_depth


class RootParallelMCTSEngine:
    """
    Root parallelization: Each worker maintains independent search tree.

    Results are merged at the end. This approach has better parallelization
    efficiency but may waste work on duplicated search paths.

    Best for: Expensive rollouts, GPU-based evaluation
    """

    def __init__(
        self,
        num_workers: int = 4,
        exploration_weight: float = 1.414,
        seed: int = 42,
    ):
        """
        Initialize root parallel MCTS engine.

        Args:
            num_workers: Number of parallel workers
            exploration_weight: UCB1 exploration constant
            seed: Random seed base
        """
        self.num_workers = num_workers
        self.exploration_weight = exploration_weight
        self.seed = seed

    async def parallel_search(
        self,
        initial_state: MCTSState,
        num_simulations: int,
        action_generator: Callable[[MCTSState], list[str]],
        state_transition: Callable[[MCTSState, str], MCTSState],
        rollout_policy: RolloutPolicy,
        max_rollout_depth: int = 10,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Run root-parallel MCTS search.

        Args:
            initial_state: Initial state
            num_simulations: Total simulations (divided among workers)
            action_generator: Action generation function
            state_transition: State transition function
            rollout_policy: Rollout policy
            max_rollout_depth: Maximum rollout depth

        Returns:
            Tuple of (best_action, merged_statistics)
        """
        from .core import MCTSEngine

        # Create independent engines for each worker
        simulations_per_worker = num_simulations // self.num_workers

        # Create worker tasks
        tasks = []
        for worker_id in range(self.num_workers):
            # Each worker gets its own engine and root
            engine = MCTSEngine(
                seed=self.seed + worker_id,
                exploration_weight=self.exploration_weight,
            )
            root = MCTSNode(
                state=initial_state,
                rng=np.random.default_rng(self.seed + worker_id),
            )

            task = engine.search(
                root=root,
                num_iterations=simulations_per_worker,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
                max_rollout_depth=max_rollout_depth,
            )
            tasks.append(task)

        # Run all workers in parallel
        results = await asyncio.gather(*tasks)

        # Merge results
        merged_action, merged_stats = self._merge_results(results)

        return merged_action, merged_stats

    def _merge_results(
        self,
        results: list[tuple[str | None, dict[str, Any]]],
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Merge results from multiple independent searches.

        Args:
            results: List of (action, stats) tuples from workers

        Returns:
            Merged (best_action, statistics)
        """
        # Aggregate action statistics
        action_stats: dict[str, dict] = defaultdict(
            lambda: {
                "visits": 0,
                "value_sum": 0.0,
                "num_workers": 0,
            }
        )

        for action, stats in results:
            if action and "action_stats" in stats:
                for act, act_stats in stats["action_stats"].items():
                    action_stats[act]["visits"] += act_stats["visits"]
                    action_stats[act]["value_sum"] += act_stats["value_sum"]
                    action_stats[act]["num_workers"] += 1

        # Compute average values
        for act in action_stats:
            visits = action_stats[act]["visits"]
            action_stats[act]["value"] = action_stats[act]["value_sum"] / visits if visits > 0 else 0.0

        # Select best action (most total visits)
        best_action = max(action_stats.keys(), key=lambda a: action_stats[a]["visits"]) if action_stats else None

        # Merge statistics
        total_simulations = sum(stats["iterations"] for _, stats in results)

        merged_stats = {
            "total_simulations": total_simulations,
            "num_workers": self.num_workers,
            "action_stats": dict(action_stats),
            "best_action": best_action,
            "parallelization": "root",
        }

        return best_action, merged_stats


class LeafParallelMCTSEngine:
    """
    Leaf parallelization: Run multiple rollouts from each leaf node.

    Best for: Cheap rollouts, high variance in rollout results
    """

    def __init__(
        self,
        num_parallel_rollouts: int = 4,
        exploration_weight: float = 1.414,
        seed: int = 42,
    ):
        """
        Initialize leaf parallel MCTS engine.

        Args:
            num_parallel_rollouts: Number of parallel rollouts per leaf
            exploration_weight: UCB1 exploration constant
            seed: Random seed
        """
        self.num_parallel_rollouts = num_parallel_rollouts
        self.exploration_weight = exploration_weight
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    async def parallel_simulate(
        self,
        node: MCTSNode,
        rollout_policy: RolloutPolicy,
        max_depth: int = 10,
    ) -> float:
        """
        Run multiple parallel rollouts from node.

        Args:
            node: Leaf node to simulate from
            rollout_policy: Rollout policy
            max_depth: Maximum rollout depth

        Returns:
            Average value from parallel rollouts
        """
        # Create rollout tasks
        rollout_rngs = [np.random.default_rng(self.seed + i) for i in range(self.num_parallel_rollouts)]

        rollout_tasks = [
            rollout_policy.evaluate(state=node.state, rng=rng, max_depth=max_depth) for rng in rollout_rngs
        ]

        # Run rollouts concurrently
        values = await asyncio.gather(*rollout_tasks)

        # Return average value
        return sum(values) / len(values)


# Helper function to create appropriate parallel MCTS engine
def create_parallel_mcts(
    strategy: str = "tree",
    num_workers: int = 4,
    **kwargs,
) -> ParallelMCTSEngine | RootParallelMCTSEngine | LeafParallelMCTSEngine:
    """
    Factory function to create parallel MCTS engine.

    Args:
        strategy: Parallelization strategy ("tree", "root", "leaf")
        num_workers: Number of parallel workers
        **kwargs: Additional parameters for specific engine

    Returns:
        Parallel MCTS engine instance

    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == "tree":
        return ParallelMCTSEngine(num_workers=num_workers, **kwargs)
    elif strategy == "root":
        return RootParallelMCTSEngine(num_workers=num_workers, **kwargs)
    elif strategy == "leaf":
        return LeafParallelMCTSEngine(num_parallel_rollouts=num_workers, **kwargs)
    else:
        raise ValueError(f"Unknown parallelization strategy: {strategy}. Choose from: tree, root, leaf")
