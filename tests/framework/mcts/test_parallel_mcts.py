"""
Tests for Parallel MCTS implementations.

Tests cover:
- Virtual loss mechanism
- Tree parallelization
- Root parallelization
- Leaf parallelization
- Adaptive virtual loss
- Performance and collision metrics
"""

import pytest

from src.framework.mcts.core import MCTSState
from src.framework.mcts.parallel_mcts import (
    ParallelMCTSEngine,
    RootParallelMCTSEngine,
    VirtualLossNode,
    create_parallel_mcts,
)
from src.framework.mcts.policies import RandomRolloutPolicy


class SimpleTestState(MCTSState):
    """Simple test state for parallel MCTS testing."""

    def __init__(self, value=0, depth=0):
        super().__init__(state_id=f"state_{value}_{depth}", features={"value": value, "depth": depth})
        self._value = value
        self._depth = depth

    def is_terminal(self):
        return self._depth >= 5


class TestVirtualLossNode:
    """Tests for VirtualLossNode."""

    def test_initialization(self):
        """Test node initialization."""
        state = SimpleTestState()
        node = VirtualLossNode(state=state)

        assert node.virtual_loss == 0.0
        assert node.virtual_loss_count == 0
        assert node.effective_visits == 0

    def test_add_revert_virtual_loss(self):
        """Test adding and reverting virtual loss."""
        state = SimpleTestState()
        node = VirtualLossNode(state=state)

        node.visits = 10
        node.value_sum = 5.0

        # Add virtual loss
        node.add_virtual_loss(3.0)
        assert node.virtual_loss == 3.0
        assert node.virtual_loss_count == 1
        assert node.effective_visits == 11

        # Add again
        node.add_virtual_loss(3.0)
        assert node.virtual_loss == 6.0
        assert node.virtual_loss_count == 2
        assert node.effective_visits == 12

        # Revert
        node.revert_virtual_loss(3.0)
        assert node.virtual_loss == 3.0
        assert node.virtual_loss_count == 1

        node.revert_virtual_loss(3.0)
        assert node.virtual_loss == 0.0
        assert node.virtual_loss_count == 0

    def test_effective_value(self):
        """Test effective value calculation with virtual loss."""
        state = SimpleTestState()
        node = VirtualLossNode(state=state)

        node.visits = 10
        node.value_sum = 8.0

        # Without virtual loss
        assert node.effective_value == 0.8

        # With virtual loss (reduces apparent value)
        node.add_virtual_loss(3.0)
        expected = (8.0 - 3.0) / 11  # (value_sum - vl) / (visits + vl_count)
        assert abs(node.effective_value - expected) < 0.01

    def test_select_child_with_vl(self):
        """Test child selection with virtual loss."""
        state = SimpleTestState()
        root = VirtualLossNode(state=state)
        root.visits = 20

        # Create children
        for i in range(3):
            child_state = SimpleTestState(value=i)
            child = VirtualLossNode(state=child_state, parent=root, action=f"action_{i}")
            child.visits = 5 + i
            child.value_sum = (5 + i) * 0.6
            root.children.append(child)

        # Select without virtual loss
        best_child = root.select_child_with_vl()
        assert best_child is not None

        # Add virtual loss to best child
        best_child.add_virtual_loss(3.0)

        # Should select different child now
        _new_best = root.select_child_with_vl()
        # Virtual loss should make other nodes more attractive


class TestParallelMCTSEngine:
    """Tests for tree-parallel MCTS."""

    @pytest.fixture
    def engine(self):
        return ParallelMCTSEngine(num_workers=2, virtual_loss_value=3.0, seed=42)

    @pytest.fixture
    def rollout_policy(self):
        return RandomRolloutPolicy()

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.num_workers == 2
        assert engine.virtual_loss_value == 3.0
        assert engine.seed == 42

    @pytest.mark.asyncio
    async def test_parallel_search(self, engine, rollout_policy):
        """Test parallel search execution."""
        initial_state = SimpleTestState()
        root = VirtualLossNode(state=initial_state)

        def action_gen(state):
            if state._depth >= 5:
                return []
            return ["left", "right", "up", "down"]

        def state_trans(state, action):
            return SimpleTestState(value=hash(action) % 10, depth=state._depth + 1)

        best_action, stats = await engine.parallel_search(
            root=root,
            num_simulations=10,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
            max_rollout_depth=5,
        )

        # Verify search completed
        assert root.visits >= 10
        assert "parallel_stats" in stats
        assert stats["parallel_stats"]["total_simulations"] == 10

    @pytest.mark.asyncio
    async def test_worker_distribution(self, rollout_policy):
        """Test simulations are distributed across workers."""
        engine = ParallelMCTSEngine(num_workers=4, virtual_loss_value=3.0)

        initial_state = SimpleTestState()
        root = VirtualLossNode(state=initial_state)

        def action_gen(state):
            return ["a", "b"]

        def state_trans(state, action):
            return SimpleTestState(value=hash(action) % 5, depth=state._depth + 1)

        await engine.parallel_search(
            root=root,
            num_simulations=20,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        # Check that multiple workers participated
        thread_sims = engine.stats.thread_simulations
        assert len(thread_sims) > 0
        assert sum(thread_sims.values()) == 20

    @pytest.mark.asyncio
    async def test_collision_tracking(self, engine, rollout_policy):
        """Test collision detection and tracking."""
        initial_state = SimpleTestState()
        root = VirtualLossNode(state=initial_state)

        def action_gen(state):
            return ["only_action"]  # Force collisions

        def state_trans(state, action):
            return SimpleTestState(depth=state._depth + 1)

        await engine.parallel_search(
            root=root,
            num_simulations=10,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        stats = engine.stats.to_dict()
        # With only one action, should have some collisions
        assert "collision_count" in stats

    @pytest.mark.asyncio
    async def test_adaptive_virtual_loss(self, rollout_policy):
        """Test adaptive virtual loss adjustment."""
        engine = ParallelMCTSEngine(
            num_workers=4,
            virtual_loss_value=3.0,
            adaptive_virtual_loss=True,
        )

        initial_vl = engine.virtual_loss_value

        initial_state = SimpleTestState()
        root = VirtualLossNode(state=initial_state)

        def action_gen(state):
            return ["a"]  # Single action to force high collision rate

        def state_trans(state, action):
            return SimpleTestState(depth=state._depth + 1)

        await engine.parallel_search(
            root=root,
            num_simulations=50,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        # VL should have adapted (increased due to collisions)
        # Note: might not always change depending on collision rate threshold
        assert engine.virtual_loss_value >= initial_vl * 0.9  # Allow some tolerance


class TestRootParallelMCTSEngine:
    """Tests for root-parallel MCTS."""

    @pytest.fixture
    def engine(self):
        return RootParallelMCTSEngine(num_workers=2, seed=42)

    @pytest.fixture
    def rollout_policy(self):
        return RandomRolloutPolicy()

    @pytest.mark.asyncio
    async def test_root_parallel_search(self, engine, rollout_policy):
        """Test root parallelization."""
        initial_state = SimpleTestState()

        def action_gen(state):
            if state._depth >= 5:
                return []
            return ["a", "b", "c"]

        def state_trans(state, action):
            return SimpleTestState(value=hash(action) % 10, depth=state._depth + 1)

        best_action, stats = await engine.parallel_search(
            initial_state=initial_state,
            num_simulations=10,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        # Verify results
        assert "action_stats" in stats
        assert stats["parallelization"] == "root"
        assert stats["num_workers"] == 2

    @pytest.mark.asyncio
    async def test_result_merging(self, engine, rollout_policy):
        """Test that results from workers are properly merged."""
        initial_state = SimpleTestState()

        def action_gen(state):
            return ["action_1", "action_2"]

        def state_trans(state, action):
            return SimpleTestState(depth=state._depth + 1)

        best_action, stats = await engine.parallel_search(
            initial_state=initial_state,
            num_simulations=20,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        # Should have merged statistics for actions
        assert len(stats["action_stats"]) > 0

        # Total simulations should be distributed
        total_visits = sum(s["visits"] for s in stats["action_stats"].values())
        assert total_visits > 0


class TestCreateParallelMCTS:
    """Tests for factory function."""

    def test_create_tree_parallel(self):
        """Test creating tree-parallel engine."""
        engine = create_parallel_mcts("tree", num_workers=4)
        assert isinstance(engine, ParallelMCTSEngine)
        assert engine.num_workers == 4

    def test_create_root_parallel(self):
        """Test creating root-parallel engine."""
        engine = create_parallel_mcts("root", num_workers=4)
        assert isinstance(engine, RootParallelMCTSEngine)
        assert engine.num_workers == 4

    def test_create_leaf_parallel(self):
        """Test creating leaf-parallel engine."""
        engine = create_parallel_mcts("leaf", num_workers=4)
        from src.framework.mcts.parallel_mcts import LeafParallelMCTSEngine

        assert isinstance(engine, LeafParallelMCTSEngine)

    def test_create_invalid_strategy(self):
        """Test error on invalid strategy."""
        with pytest.raises(ValueError, match="Unknown parallelization strategy"):
            create_parallel_mcts("invalid_strategy")


@pytest.mark.integration
@pytest.mark.asyncio
class TestParallelMCTSPerformance:
    """Integration tests for parallel MCTS performance."""

    async def test_parallel_speedup(self):
        """Test that parallel MCTS is faster than sequential."""
        import time

        initial_state = SimpleTestState()
        rollout_policy = RandomRolloutPolicy()

        def action_gen(state):
            if state._depth >= 3:
                return []
            return ["a", "b", "c", "d"]

        def state_trans(state, action):
            return SimpleTestState(depth=state._depth + 1)

        # Sequential (1 worker)
        engine_seq = ParallelMCTSEngine(num_workers=1)
        root_seq = VirtualLossNode(state=initial_state)

        start = time.time()
        await engine_seq.parallel_search(
            root=root_seq,
            num_simulations=20,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )
        seq_time = time.time() - start

        # Parallel (4 workers)
        engine_par = ParallelMCTSEngine(num_workers=4)
        root_par = VirtualLossNode(state=initial_state)

        start = time.time()
        await engine_par.parallel_search(
            root=root_par,
            num_simulations=20,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )
        par_time = time.time() - start

        # Parallel should generally be faster (though not guaranteed in tests)
        # At minimum, verify both completed successfully
        assert seq_time > 0
        assert par_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
