"""
Tests for parallel MCTS module.

Tests ParallelMCTSStats, ParallelMCTSConfig, VirtualLossNode,
and ParallelMCTSEngine initialization.
"""

import numpy as np
import pytest

from src.framework.mcts.core import MCTSState
from src.framework.mcts.parallel_mcts import (
    ParallelMCTSConfig,
    ParallelMCTSEngine,
    ParallelMCTSStats,
    VirtualLossNode,
)


def _make_state(state_id="s1"):
    return MCTSState(state_id=state_id, features={"v": 1})


@pytest.mark.unit
class TestParallelMCTSStats:
    """Tests for ParallelMCTSStats."""

    def test_defaults(self):
        stats = ParallelMCTSStats()
        assert stats.total_simulations == 0
        assert stats.collision_count == 0
        assert stats.effective_parallelism == 0.0

    def test_to_dict(self):
        stats = ParallelMCTSStats(
            total_simulations=100,
            total_duration=5.0,
            thread_simulations={0: 50, 1: 50},
            collision_count=5,
        )
        d = stats.to_dict()
        assert d["total_simulations"] == 100
        assert d["collision_count"] == 5
        assert isinstance(d["thread_simulations"], dict)


@pytest.mark.unit
class TestParallelMCTSConfig:
    """Tests for ParallelMCTSConfig."""

    def test_defaults(self):
        cfg = ParallelMCTSConfig()
        assert cfg.num_workers == 4
        assert cfg.virtual_loss_value == 3.0
        assert cfg.adaptive_virtual_loss is True
        assert cfg.exploration_weight == 1.414
        assert cfg.seed == 42

    def test_custom_config(self):
        cfg = ParallelMCTSConfig(
            num_workers=8,
            virtual_loss_value=5.0,
            seed=99,
        )
        assert cfg.num_workers == 8
        assert cfg.virtual_loss_value == 5.0
        assert cfg.seed == 99

    def test_validate_valid(self):
        cfg = ParallelMCTSConfig()
        cfg.validate()  # Should not raise

    def test_validate_invalid_workers(self):
        cfg = ParallelMCTSConfig(num_workers=0)
        with pytest.raises(ValueError, match="num_workers"):
            cfg.validate()

    def test_validate_negative_vl(self):
        cfg = ParallelMCTSConfig(virtual_loss_value=-1)
        with pytest.raises(ValueError, match="virtual_loss_value"):
            cfg.validate()

    def test_validate_invalid_vl_range(self):
        cfg = ParallelMCTSConfig(virtual_loss_min=5.0, virtual_loss_max=2.0)
        with pytest.raises(ValueError, match="virtual_loss_max"):
            cfg.validate()

    def test_validate_invalid_collision_thresholds(self):
        cfg = ParallelMCTSConfig(
            collision_rate_low_threshold=0.5,
            collision_rate_high_threshold=0.3,
        )
        with pytest.raises(ValueError, match="collision thresholds"):
            cfg.validate()

    def test_validate_invalid_history_size(self):
        cfg = ParallelMCTSConfig(collision_history_size=0)
        with pytest.raises(ValueError, match="collision_history_size"):
            cfg.validate()

    def test_validate_negative_exploration(self):
        cfg = ParallelMCTSConfig(exploration_weight=-1)
        with pytest.raises(ValueError, match="exploration_weight"):
            cfg.validate()

    def test_validate_invalid_lock_timeout(self):
        cfg = ParallelMCTSConfig(lock_timeout_seconds=-1)
        with pytest.raises(ValueError, match="lock_timeout_seconds"):
            cfg.validate()

    def test_validate_none_lock_timeout_ok(self):
        cfg = ParallelMCTSConfig(lock_timeout_seconds=None)
        cfg.validate()  # Should not raise


@pytest.mark.unit
class TestVirtualLossNode:
    """Tests for VirtualLossNode."""

    def test_init(self):
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        assert node.virtual_loss == 0.0
        assert node.virtual_loss_count == 0

    def test_effective_visits(self):
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        node.visits = 10
        node.virtual_loss_count = 3
        assert node.effective_visits == 13

    def test_effective_value_no_visits(self):
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        assert node.effective_value == 0.0

    def test_effective_value_with_visits(self):
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        node.visits = 10
        node.value_sum = 7.0
        node.virtual_loss = 1.0
        node.virtual_loss_count = 2
        # effective_value = (7.0 - 1.0) / (10 + 2) = 6.0/12 = 0.5
        assert node.effective_value == pytest.approx(0.5)

    def test_add_virtual_loss(self):
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        node.add_virtual_loss(3.0)
        assert node.virtual_loss == 3.0
        assert node.virtual_loss_count == 1

    def test_revert_virtual_loss(self):
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        node.add_virtual_loss(3.0)
        node.revert_virtual_loss(3.0)
        assert node.virtual_loss == 0.0
        assert node.virtual_loss_count == 0

    def test_add_child(self):
        rng = np.random.default_rng(42)
        parent = VirtualLossNode(state=_make_state("p"), rng=rng)
        child = parent.add_child("action_1", _make_state("c"))
        assert isinstance(child, VirtualLossNode)
        assert child.action == "action_1"
        assert child.parent is parent
        assert len(parent.children) == 1
        assert "action_1" in parent.expanded_actions

    def test_select_child_with_vl_no_children_raises(self):
        node = VirtualLossNode(state=_make_state(), rng=np.random.default_rng(42))
        with pytest.raises(ValueError, match="No children"):
            node.select_child_with_vl()

    def test_select_child_with_vl_unvisited_first(self):
        rng = np.random.default_rng(42)
        parent = VirtualLossNode(state=_make_state(), rng=rng)
        parent.visits = 10

        child1 = parent.add_child("a1", _make_state("c1"))
        child1.visits = 5
        child1.value_sum = 3.0

        child2 = parent.add_child("a2", _make_state("c2"))
        # child2 unvisited

        selected = parent.select_child_with_vl()
        assert selected is child2

    def test_select_child_with_vl_best_score(self):
        rng = np.random.default_rng(42)
        parent = VirtualLossNode(state=_make_state(), rng=rng)
        parent.visits = 20

        child1 = parent.add_child("a1", _make_state("c1"))
        child1.visits = 10
        child1.value_sum = 8.0  # value = 0.8

        child2 = parent.add_child("a2", _make_state("c2"))
        child2.visits = 10
        child2.value_sum = 2.0  # value = 0.2

        selected = parent.select_child_with_vl()
        assert selected is child1

    def test_virtual_loss_reduces_attractiveness(self):
        """Adding VL should make a node less attractive."""
        rng = np.random.default_rng(42)
        parent = VirtualLossNode(state=_make_state(), rng=rng)
        parent.visits = 20

        child1 = parent.add_child("a1", _make_state("c1"))
        child1.visits = 10
        child1.value_sum = 8.0  # high value

        child2 = parent.add_child("a2", _make_state("c2"))
        child2.visits = 10
        child2.value_sum = 6.0  # lower value

        # Without VL, child1 should be selected
        assert parent.select_child_with_vl() is child1

        # Add heavy VL to child1
        child1.add_virtual_loss(10.0)
        child1.add_virtual_loss(10.0)

        # Now child2 should be selected
        assert parent.select_child_with_vl() is child2


@pytest.mark.unit
class TestParallelMCTSEngine:
    """Tests for ParallelMCTSEngine initialization."""

    def test_init_with_config(self):
        cfg = ParallelMCTSConfig(num_workers=8, seed=99)
        engine = ParallelMCTSEngine(config=cfg)
        assert engine._config.num_workers == 8
        assert engine._config.seed == 99

    def test_init_default_config(self):
        engine = ParallelMCTSEngine()
        assert engine._config.num_workers == 4
        assert engine._config.seed == 42

    def test_init_backwards_compat_params(self):
        engine = ParallelMCTSEngine(
            num_workers=2,
            virtual_loss_value=5.0,
            seed=77,
        )
        assert engine._config.num_workers == 2
        assert engine._config.virtual_loss_value == 5.0
        assert engine._config.seed == 77
