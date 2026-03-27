"""
Extended unit tests for parallel_mcts module to improve coverage.

Covers missed lines including:
- ParallelMCTSEngine: parallel_search, _worker_thread, _run_simulation,
  _adapt_virtual_loss, _select_best_action, _build_stats_dict, _compute_tree_depth
- RootParallelMCTSEngine: parallel_search, _merge_results
- LeafParallelMCTSEngine: __init__, parallel_simulate
- create_parallel_mcts factory
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.framework.mcts.core import MCTSNode, MCTSState
from src.framework.mcts.parallel_mcts import (
    LeafParallelMCTSEngine,
    ParallelMCTSConfig,
    ParallelMCTSEngine,
    ParallelMCTSStats,
    RootParallelMCTSEngine,
    VirtualLossNode,
    create_parallel_mcts,
)

pytestmark = pytest.mark.unit


def _make_state(state_id: str = "s1") -> MCTSState:
    return MCTSState(state_id=state_id, features={"v": 1})


def _action_generator(state: MCTSState) -> list[str]:
    return ["action_a", "action_b"]


def _state_transition(state: MCTSState, action: str) -> MCTSState:
    return MCTSState(state_id=f"{state.state_id}_{action}", features=state.features)


class MockRolloutPolicy:
    """Mock rollout policy for testing."""

    def __init__(self, return_value: float = 0.5):
        self._return_value = return_value

    async def evaluate(self, state: MCTSState, rng: np.random.Generator, max_depth: int = 10) -> float:
        return self._return_value


# ============================================================================
# ParallelMCTSEngine tests
# ============================================================================


class TestParallelMCTSEngineParallelSearch:
    """Test ParallelMCTSEngine.parallel_search (lines 331-374)."""

    @pytest.mark.asyncio
    async def test_parallel_search_returns_action_and_stats(self):
        """parallel_search should return best action and stats dict."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=2, seed=42, adaptive_virtual_loss=False),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))

        action, stats = await engine.parallel_search(
            root=root,
            num_simulations=8,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.6),
            max_rollout_depth=5,
        )

        assert stats is not None
        assert "root_visits" in stats
        assert "parallel_stats" in stats
        assert "action_stats" in stats
        assert "tree_depth" in stats
        assert stats["root_visits"] > 0

    @pytest.mark.asyncio
    async def test_parallel_search_no_children_returns_none(self):
        """parallel_search with terminal root returns None."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=1, seed=42, adaptive_virtual_loss=False),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))
        root.terminal = True

        action, stats = await engine.parallel_search(
            root=root,
            num_simulations=4,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.5),
        )
        # Terminal root, no children expanded
        assert action is None

    @pytest.mark.asyncio
    async def test_parallel_search_distributes_simulations(self):
        """Simulations should be distributed across workers."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=3, seed=42, adaptive_virtual_loss=False),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))

        await engine.parallel_search(
            root=root,
            num_simulations=10,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.5),
        )

        # Check all workers ran
        total = sum(engine.stats.thread_simulations.values())
        assert total == 10

    @pytest.mark.asyncio
    async def test_parallel_search_initializes_root_actions(self):
        """Root actions should be initialized if not already set."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=1, seed=42, adaptive_virtual_loss=False),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))
        assert root.available_actions == []

        await engine.parallel_search(
            root=root,
            num_simulations=2,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.5),
        )

        assert root.available_actions == ["action_a", "action_b"]

    @pytest.mark.asyncio
    async def test_parallel_search_stats_duration(self):
        """Stats should record duration."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=1, seed=42, adaptive_virtual_loss=False),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))

        _, stats = await engine.parallel_search(
            root=root,
            num_simulations=4,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.5),
        )

        assert stats["parallel_stats"]["total_duration"] > 0
        assert stats["parallel_stats"]["total_simulations"] == 4


class TestParallelMCTSEngineAdaptVL:
    """Test _adapt_virtual_loss (lines 499-512)."""

    def test_adapt_increase_on_high_collision(self):
        """VL should increase when collision rate is high."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(
                num_workers=1,
                virtual_loss_value=3.0,
                adaptive_virtual_loss=True,
                collision_rate_high_threshold=0.3,
                collision_rate_low_threshold=0.1,
                virtual_loss_increase_rate=1.5,
                virtual_loss_max=10.0,
                seed=42,
            ),
        )
        # Simulate high collision rate (> 0.3)
        engine._collision_history = [True] * 50 + [False] * 50  # 50% collision rate
        engine._adapt_virtual_loss()
        assert engine.virtual_loss_value == pytest.approx(3.0 * 1.5)

    def test_adapt_decrease_on_low_collision(self):
        """VL should decrease when collision rate is low."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(
                num_workers=1,
                virtual_loss_value=3.0,
                adaptive_virtual_loss=True,
                collision_rate_high_threshold=0.3,
                collision_rate_low_threshold=0.1,
                virtual_loss_decrease_rate=0.8,
                virtual_loss_min=1.0,
                seed=42,
            ),
        )
        # Simulate low collision rate (< 0.1)
        engine._collision_history = [True] * 5 + [False] * 95  # 5% collision rate
        engine._adapt_virtual_loss()
        assert engine.virtual_loss_value == pytest.approx(3.0 * 0.8)

    def test_adapt_no_change_in_middle(self):
        """VL should stay same when collision rate is between thresholds."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(
                num_workers=1,
                virtual_loss_value=3.0,
                adaptive_virtual_loss=True,
                collision_rate_high_threshold=0.3,
                collision_rate_low_threshold=0.1,
                seed=42,
            ),
        )
        # 20% collision rate - between thresholds
        engine._collision_history = [True] * 20 + [False] * 80
        engine._adapt_virtual_loss()
        assert engine.virtual_loss_value == 3.0

    def test_adapt_respects_max(self):
        """VL should not exceed max."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(
                num_workers=1,
                virtual_loss_value=9.5,
                virtual_loss_max=10.0,
                virtual_loss_increase_rate=1.5,
                collision_rate_high_threshold=0.3,
                seed=42,
            ),
        )
        engine._collision_history = [True] * 100  # 100% collision
        engine._adapt_virtual_loss()
        assert engine.virtual_loss_value == 10.0

    def test_adapt_respects_min(self):
        """VL should not go below min."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(
                num_workers=1,
                virtual_loss_value=1.1,
                virtual_loss_min=1.0,
                virtual_loss_decrease_rate=0.5,
                collision_rate_low_threshold=0.1,
                seed=42,
            ),
        )
        engine._collision_history = [False] * 100  # 0% collision
        engine._adapt_virtual_loss()
        assert engine.virtual_loss_value == 1.0

    def test_adapt_empty_history(self):
        """Empty collision history should not change VL."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=1, virtual_loss_value=3.0, seed=42),
        )
        engine._collision_history = []
        engine._adapt_virtual_loss()
        assert engine.virtual_loss_value == 3.0


class TestParallelMCTSEngineSelectBestAction:
    """Test _select_best_action (lines 527-531)."""

    def test_select_best_action_no_children(self):
        """Returns None when root has no children."""
        engine = ParallelMCTSEngine(config=ParallelMCTSConfig(num_workers=1, seed=42))
        root = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        assert engine._select_best_action(root) is None

    def test_select_best_action_most_visited(self):
        """Selects the child with most visits."""
        engine = ParallelMCTSEngine(config=ParallelMCTSConfig(num_workers=1, seed=42))
        root = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))

        child1 = root.add_child("a1", _make_state("c1"))
        child1.visits = 10

        child2 = root.add_child("a2", _make_state("c2"))
        child2.visits = 20

        assert engine._select_best_action(root) == "a2"


class TestParallelMCTSEngineBuildStats:
    """Test _build_stats_dict and _compute_tree_depth (lines 544-576)."""

    def test_build_stats_dict(self):
        """Stats dict should contain all expected fields."""
        engine = ParallelMCTSEngine(config=ParallelMCTSConfig(num_workers=1, seed=42))
        root = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        root.visits = 10
        root.value_sum = 5.0

        child = root.add_child("a1", _make_state("c1"))
        child.visits = 8
        child.value_sum = 4.0

        stats = engine._build_stats_dict(root)
        assert stats["root_visits"] == 10
        assert stats["root_value"] == pytest.approx(0.5)
        assert stats["num_children"] == 1
        assert "a1" in stats["action_stats"]
        assert stats["action_stats"]["a1"]["visits"] == 8
        assert "tree_depth" in stats
        assert "parallel_stats" in stats
        assert "virtual_loss_value" in stats

    def test_compute_tree_depth_no_children(self):
        """Depth of leaf node is 0."""
        engine = ParallelMCTSEngine(config=ParallelMCTSConfig(num_workers=1, seed=42))
        root = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        assert engine._compute_tree_depth(root) == 0

    def test_compute_tree_depth_nested(self):
        """Depth should reflect deepest branch."""
        engine = ParallelMCTSEngine(config=ParallelMCTSConfig(num_workers=1, seed=42))
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))

        child1 = root.add_child("a1", _make_state("c1"))
        child1_1 = child1.add_child("a1_1", _make_state("c1_1"))
        child1_1.add_child("a1_1_1", _make_state("c1_1_1"))  # depth 3

        root.add_child("a2", _make_state("c2"))  # depth 1

        assert engine._compute_tree_depth(root) == 3


class TestParallelMCTSEngineRunSimulation:
    """Test _run_simulation (lines 431-495)."""

    @pytest.mark.asyncio
    async def test_run_simulation_expands_and_backpropagates(self):
        """A simulation should expand nodes and backpropagate values."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=1, seed=42, adaptive_virtual_loss=False),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))
        root.available_actions = ["a1", "a2"]

        await engine._run_simulation(
            worker_id=0,
            root=root,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.7),
            max_rollout_depth=5,
        )

        # Root should have been visited
        assert root.visits >= 1
        # Virtual loss should be reverted (back to 0)
        assert root.virtual_loss == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_run_simulation_adaptive_vl(self):
        """Simulation with adaptive VL should track collision history."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(
                num_workers=1,
                seed=42,
                adaptive_virtual_loss=True,
                collision_history_size=5,
            ),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))
        root.available_actions = ["a1"]

        for _ in range(3):
            await engine._run_simulation(
                worker_id=0,
                root=root,
                action_generator=_action_generator,
                state_transition=_state_transition,
                rollout_policy=MockRolloutPolicy(0.5),
                max_rollout_depth=5,
            )

        assert len(engine._collision_history) > 0


class TestParallelMCTSEngineWorkerThread:
    """Test _worker_thread (lines 398-409)."""

    @pytest.mark.asyncio
    async def test_worker_thread_runs_correct_number_of_sims(self):
        """Worker should run exactly num_simulations simulations."""
        engine = ParallelMCTSEngine(
            config=ParallelMCTSConfig(num_workers=1, seed=42, adaptive_virtual_loss=False),
        )
        root = VirtualLossNode(state=_make_state("root"), rng=np.random.default_rng(42))
        root.available_actions = ["a1", "a2"]

        await engine._worker_thread(
            worker_id=0,
            root=root,
            num_simulations=5,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.5),
            max_rollout_depth=5,
        )

        assert engine.stats.thread_simulations[0] == 5


# ============================================================================
# RootParallelMCTSEngine tests
# ============================================================================


class TestRootParallelMCTSEngine:
    """Test RootParallelMCTSEngine (lines 603-714)."""

    def test_init(self):
        """Test initialization."""
        engine = RootParallelMCTSEngine(num_workers=3, exploration_weight=2.0, seed=99)
        assert engine.num_workers == 3
        assert engine.exploration_weight == 2.0
        assert engine.seed == 99

    @pytest.mark.asyncio
    async def test_parallel_search(self):
        """Root parallel search should run and merge results."""
        engine = RootParallelMCTSEngine(num_workers=2, seed=42)

        action, stats = await engine.parallel_search(
            initial_state=_make_state("root"),
            num_simulations=8,
            action_generator=_action_generator,
            state_transition=_state_transition,
            rollout_policy=MockRolloutPolicy(0.5),
            max_rollout_depth=5,
        )

        assert stats is not None
        assert "total_simulations" in stats
        assert "num_workers" in stats
        assert stats["num_workers"] == 2
        assert stats["parallelization"] == "root"

    def test_merge_results_aggregates(self):
        """_merge_results should aggregate across workers."""
        engine = RootParallelMCTSEngine(num_workers=2)

        results = [
            (
                "action_a",
                {
                    "iterations": 10,
                    "action_stats": {
                        "action_a": {"visits": 7, "value_sum": 3.5},
                        "action_b": {"visits": 3, "value_sum": 1.5},
                    },
                },
            ),
            (
                "action_a",
                {
                    "iterations": 10,
                    "action_stats": {
                        "action_a": {"visits": 8, "value_sum": 4.0},
                        "action_b": {"visits": 2, "value_sum": 1.0},
                    },
                },
            ),
        ]

        best_action, merged = engine._merge_results(results)

        assert best_action == "action_a"
        assert merged["total_simulations"] == 20
        assert merged["action_stats"]["action_a"]["visits"] == 15
        assert merged["action_stats"]["action_b"]["visits"] == 5
        assert merged["action_stats"]["action_a"]["value"] == pytest.approx(7.5 / 15)

    def test_merge_results_no_actions(self):
        """_merge_results handles case with no action stats."""
        engine = RootParallelMCTSEngine(num_workers=2)

        results = [
            (None, {"iterations": 5}),
            (None, {"iterations": 5}),
        ]

        best_action, merged = engine._merge_results(results)

        assert best_action is None
        assert merged["total_simulations"] == 10

    def test_merge_results_computes_value(self):
        """Merged results should compute average value per action."""
        engine = RootParallelMCTSEngine(num_workers=1)

        results = [
            (
                "a1",
                {
                    "iterations": 5,
                    "action_stats": {
                        "a1": {"visits": 0, "value_sum": 0.0},
                    },
                },
            ),
        ]

        _, merged = engine._merge_results(results)
        # visits=0, value should be 0
        assert merged["action_stats"]["a1"]["value"] == 0.0


# ============================================================================
# LeafParallelMCTSEngine tests
# ============================================================================


class TestLeafParallelMCTSEngine:
    """Test LeafParallelMCTSEngine (lines 738-771)."""

    def test_init(self):
        """Test initialization."""
        engine = LeafParallelMCTSEngine(num_parallel_rollouts=6, exploration_weight=2.0, seed=99)
        assert engine.num_parallel_rollouts == 6
        assert engine.exploration_weight == 2.0
        assert engine.seed == 99

    def test_init_defaults(self):
        """Test default initialization."""
        engine = LeafParallelMCTSEngine()
        assert engine.num_parallel_rollouts == 4
        assert engine.exploration_weight == 1.414
        assert engine.seed == 42

    @pytest.mark.asyncio
    async def test_parallel_simulate(self):
        """parallel_simulate should return average of rollout values."""
        engine = LeafParallelMCTSEngine(num_parallel_rollouts=4, seed=42)
        node = MCTSNode(state=_make_state("leaf"), rng=np.random.default_rng(42))

        result = await engine.parallel_simulate(
            node=node,
            rollout_policy=MockRolloutPolicy(0.6),
            max_depth=5,
        )

        # All rollouts return 0.6, so average = 0.6
        assert result == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_parallel_simulate_varied_values(self):
        """Test with a policy that returns different values per call."""
        call_count = 0

        class VaryingPolicy:
            async def evaluate(self, state, rng, max_depth=10):
                nonlocal call_count
                call_count += 1
                return 0.2 * call_count  # 0.2, 0.4, 0.6, 0.8

        engine = LeafParallelMCTSEngine(num_parallel_rollouts=4, seed=42)
        node = MCTSNode(state=_make_state("leaf"), rng=np.random.default_rng(42))

        result = await engine.parallel_simulate(
            node=node,
            rollout_policy=VaryingPolicy(),
            max_depth=5,
        )

        # Average of 0.2, 0.4, 0.6, 0.8 = 0.5
        assert result == pytest.approx(0.5)


# ============================================================================
# create_parallel_mcts factory tests
# ============================================================================


class TestCreateParallelMCTS:
    """Test create_parallel_mcts factory (lines 794-801)."""

    def test_create_tree_strategy(self):
        """'tree' strategy creates ParallelMCTSEngine."""
        engine = create_parallel_mcts(strategy="tree", num_workers=2, seed=99)
        assert isinstance(engine, ParallelMCTSEngine)
        assert engine.num_workers == 2

    def test_create_root_strategy(self):
        """'root' strategy creates RootParallelMCTSEngine."""
        engine = create_parallel_mcts(strategy="root", num_workers=3, seed=99)
        assert isinstance(engine, RootParallelMCTSEngine)
        assert engine.num_workers == 3

    def test_create_leaf_strategy(self):
        """'leaf' strategy creates LeafParallelMCTSEngine."""
        engine = create_parallel_mcts(strategy="leaf", num_workers=5, seed=99)
        assert isinstance(engine, LeafParallelMCTSEngine)
        assert engine.num_parallel_rollouts == 5

    def test_create_unknown_strategy_raises(self):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown parallelization strategy"):
            create_parallel_mcts(strategy="invalid")

    def test_create_default_strategy(self):
        """Default strategy should be 'tree'."""
        engine = create_parallel_mcts()
        assert isinstance(engine, ParallelMCTSEngine)

    def test_create_with_kwargs(self):
        """Extra kwargs should be passed through."""
        engine = create_parallel_mcts(
            strategy="tree",
            num_workers=2,
            virtual_loss_value=5.0,
            seed=123,
        )
        assert isinstance(engine, ParallelMCTSEngine)
        assert engine.virtual_loss_value == 5.0


# ============================================================================
# VirtualLossNode edge case tests (for stats dict coverage)
# ============================================================================


class TestVirtualLossNodeEffectiveValues:
    """Additional VirtualLossNode tests for stats coverage."""

    def test_effective_visits_property(self):
        """effective_visits = visits + virtual_loss_count."""
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        node.visits = 5
        node.add_virtual_loss(3.0)
        node.add_virtual_loss(3.0)
        assert node.effective_visits == 7

    def test_effective_value_with_virtual_loss(self):
        """effective_value accounts for virtual loss."""
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        node.visits = 10
        node.value_sum = 8.0
        node.add_virtual_loss(2.0)
        # effective_value = (8.0 - 2.0) / (10 + 1) = 6.0/11 ~ 0.545
        assert node.effective_value == pytest.approx(6.0 / 11.0)

    def test_build_stats_includes_effective_visits(self):
        """Build stats should include effective_visits for each child."""
        engine = ParallelMCTSEngine(config=ParallelMCTSConfig(num_workers=1, seed=42))
        root = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        root.visits = 20
        root.value_sum = 10.0

        child = root.add_child("a1", _make_state("c1"))
        child.visits = 15
        child.value_sum = 7.5
        child.add_virtual_loss(1.0)

        stats = engine._build_stats_dict(root)
        assert stats["action_stats"]["a1"]["effective_visits"] == 16  # 15 + 1
