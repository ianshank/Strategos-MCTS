"""
Tests for progressive widening and RAVE MCTS module.

Tests ProgressiveWideningConfig, RAVEConfig, RAVENode,
ProgressiveWideningEngine, and utility functions.
"""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.framework.mcts.core import MCTSState
from src.framework.mcts.progressive_widening import (
    ProgressiveWideningConfig,
    ProgressiveWideningEngine,
    RAVEConfig,
    RAVENode,
    create_pw_config,
    create_rave_config,
)


def _make_state(state_id="s1", features=None):
    return MCTSState(state_id=state_id, features=features or {"v": 1})


@pytest.mark.unit
class TestProgressiveWideningConfig:
    """Tests for ProgressiveWideningConfig."""

    def test_defaults(self):
        cfg = ProgressiveWideningConfig()
        assert cfg.k == 1.0
        assert cfg.alpha == 0.5
        assert cfg.adaptive is False

    def test_should_expand_first_child(self):
        """Always expand first child (num_children=0)."""
        cfg = ProgressiveWideningConfig(k=1.0, alpha=0.5)
        assert cfg.should_expand(visits=0, num_children=0) is True

    def test_should_expand_above_threshold(self):
        cfg = ProgressiveWideningConfig(k=1.0, alpha=0.5)
        # threshold = 1.0 * 2^0.5 ≈ 1.414, visits=2 > 1.414
        assert cfg.should_expand(visits=2, num_children=2) is True

    def test_should_not_expand_below_threshold(self):
        cfg = ProgressiveWideningConfig(k=5.0, alpha=0.5)
        # threshold = 5.0 * 2^0.5 ≈ 7.07, visits=3 < 7.07
        assert cfg.should_expand(visits=3, num_children=2) is False

    def test_min_visits_for_next_child(self):
        cfg = ProgressiveWideningConfig(k=1.0, alpha=0.5)
        result = cfg.min_visits_for_next_child(4)
        # threshold = 1.0 * 4^0.5 = 2.0, ceil(2.0) + 1 = 3
        assert result == 3


@pytest.mark.unit
class TestRAVEConfig:
    """Tests for RAVEConfig."""

    def test_defaults(self):
        cfg = RAVEConfig()
        assert cfg.rave_constant == 300.0
        assert cfg.enable_rave is True
        assert cfg.min_visits_for_rave == 5

    def test_compute_beta_disabled(self):
        cfg = RAVEConfig(enable_rave=False)
        assert cfg.compute_beta(100, 50) == 0.0

    def test_compute_beta_below_min_visits(self):
        cfg = RAVEConfig(min_visits_for_rave=10)
        assert cfg.compute_beta(100, 5) == 0.0

    def test_compute_beta_returns_value(self):
        cfg = RAVEConfig(rave_constant=300.0, min_visits_for_rave=5)
        beta = cfg.compute_beta(node_visits=50, rave_visits=20)
        assert 0.0 <= beta <= 1.0

    def test_compute_beta_high_visits_approaches_zero(self):
        cfg = RAVEConfig(rave_constant=300.0, min_visits_for_rave=5)
        beta = cfg.compute_beta(node_visits=100000, rave_visits=10)
        assert beta < 0.01

    def test_compute_beta_zero_denominator(self):
        cfg = RAVEConfig(rave_constant=0.0, min_visits_for_rave=0)
        beta = cfg.compute_beta(0, 0)
        assert beta == 0.0


@pytest.mark.unit
class TestRAVENode:
    """Tests for RAVENode."""

    def test_init_has_rave_stats(self):
        state = _make_state()
        node = RAVENode(state=state, rng=np.random.default_rng(42))
        assert node.rave_visits == {}
        assert node.rave_value_sum == {}

    def test_update_rave(self):
        state = _make_state()
        node = RAVENode(state=state, rng=np.random.default_rng(42))
        node.update_rave("action_a", 0.7)
        node.update_rave("action_a", 0.3)
        assert node.rave_visits["action_a"] == 2
        assert node.rave_value_sum["action_a"] == pytest.approx(1.0)

    def test_get_rave_value_no_data(self):
        node = RAVENode(state=_make_state(), rng=np.random.default_rng(42))
        assert node.get_rave_value("unknown") == 0.5

    def test_get_rave_value_with_data(self):
        node = RAVENode(state=_make_state(), rng=np.random.default_rng(42))
        node.update_rave("a", 0.8)
        node.update_rave("a", 0.6)
        assert node.get_rave_value("a") == pytest.approx(0.7)

    def test_get_rave_visits(self):
        node = RAVENode(state=_make_state(), rng=np.random.default_rng(42))
        assert node.get_rave_visits("x") == 0
        node.update_rave("x", 1.0)
        assert node.get_rave_visits("x") == 1

    def test_select_child_rave_no_children_raises(self):
        node = RAVENode(state=_make_state(), rng=np.random.default_rng(42))
        with pytest.raises(ValueError, match="No children"):
            node.select_child_rave(RAVEConfig())

    def test_select_child_rave_unvisited_priority(self):
        """Unvisited children should be selected first."""
        rng = np.random.default_rng(42)
        parent = RAVENode(state=_make_state(), rng=rng)
        parent.visits = 10

        child1 = RAVENode(state=_make_state("c1"), parent=parent, action="a1", rng=rng)
        child1.visits = 5
        child1.value_sum = 2.5

        child2 = RAVENode(state=_make_state("c2"), parent=parent, action="a2", rng=rng)
        # child2 has 0 visits

        parent.children = [child1, child2]
        selected = parent.select_child_rave(RAVEConfig())
        assert selected is child2

    def test_select_child_rave_scores(self):
        """With all visited children, select highest hybrid score."""
        rng = np.random.default_rng(42)
        parent = RAVENode(state=_make_state(), rng=rng)
        parent.visits = 20

        child1 = RAVENode(state=_make_state("c1"), parent=parent, action="a1", rng=rng)
        child1.visits = 10
        child1.value_sum = 8.0  # value = 0.8

        child2 = RAVENode(state=_make_state("c2"), parent=parent, action="a2", rng=rng)
        child2.visits = 10
        child2.value_sum = 2.0  # value = 0.2

        parent.children = [child1, child2]
        selected = parent.select_child_rave(RAVEConfig(enable_rave=False))
        assert selected is child1


@pytest.mark.unit
class TestProgressiveWideningEngine:
    """Tests for ProgressiveWideningEngine."""

    def test_init_defaults(self):
        engine = ProgressiveWideningEngine()
        assert engine.pw_config.k == 1.0
        assert engine.rave_config.enable_rave is True
        assert engine.exploration_weight == 1.414

    def test_init_custom(self):
        pw = ProgressiveWideningConfig(k=2.0, alpha=0.6)
        rave = RAVEConfig(enable_rave=False)
        engine = ProgressiveWideningEngine(pw_config=pw, rave_config=rave, seed=99)
        assert engine.pw_config.k == 2.0
        assert engine.rave_config.enable_rave is False

    def test_should_expand_terminal(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)
        node.terminal = True
        assert engine.should_expand(node) is False

    def test_should_expand_no_actions(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)
        node.available_actions = []
        assert engine.should_expand(node) is False

    def test_should_expand_fully_expanded(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)
        node.available_actions = ["a1"]
        child = RAVENode(state=_make_state("c"), parent=node, action="a1", rng=engine.rng)
        node.children = [child]
        assert engine.should_expand(node) is False

    def test_select_leaf(self):
        engine = ProgressiveWideningEngine()
        root = RAVENode(state=_make_state(), rng=engine.rng)
        # No children, should return root
        result = engine.select(root)
        assert result is root

    def test_expand_terminal_noop(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)
        node.terminal = True
        result = engine.expand(node, lambda s: [], lambda s, a: s)
        assert result is node

    def test_expand_generates_actions(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)
        node.visits = 1  # Need visits for expansion

        def action_gen(state):
            return ["a1", "a2", "a3"]

        def state_trans(state, action):
            return MCTSState(state_id=f"new_{action}", features={"v": 1})

        result = engine.expand(node, action_gen, state_trans)
        assert len(node.children) == 1
        assert result.action in ["a1", "a2", "a3"]

    def test_expand_no_available_actions_marks_terminal(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)
        node.visits = 1

        engine.expand(node, lambda s: [], lambda s, a: s)
        assert node.terminal is True

    @pytest.mark.asyncio
    async def test_simulate_with_tracking(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)

        policy = AsyncMock(spec=[])  # spec=[] prevents auto-creating attrs
        policy.evaluate = AsyncMock(return_value=0.7)

        value, actions = await engine.simulate_with_tracking(node, policy)
        assert value == 0.7
        assert actions == []  # No last_actions attr

    @pytest.mark.asyncio
    async def test_simulate_with_tracking_has_last_actions(self):
        engine = ProgressiveWideningEngine()
        node = RAVENode(state=_make_state(), rng=engine.rng)

        policy = AsyncMock()
        policy.evaluate = AsyncMock(return_value=0.5)
        policy.last_actions = ["a1", "a2"]

        value, actions = await engine.simulate_with_tracking(node, policy)
        assert value == 0.5
        assert actions == ["a1", "a2"]

    def test_backpropagate_with_rave(self):
        engine = ProgressiveWideningEngine()
        rng = engine.rng

        root = RAVENode(state=_make_state("root"), rng=rng)
        child = RAVENode(state=_make_state("child"), parent=root, action="a1", rng=rng)

        engine.backpropagate_with_rave(child, 0.8, ["a1", "a2"])

        assert child.visits == 1
        assert child.value_sum == 0.8
        assert root.visits == 1
        assert root.value_sum == -0.8  # flipped
        assert root.rave_visits.get("a1", 0) == 1
        assert root.rave_visits.get("a2", 0) == 1

    def test_backpropagate_rave_disabled(self):
        engine = ProgressiveWideningEngine(rave_config=RAVEConfig(enable_rave=False))
        rng = engine.rng

        root = RAVENode(state=_make_state("root"), rng=rng)
        child = RAVENode(state=_make_state("child"), parent=root, action="a1", rng=rng)

        engine.backpropagate_with_rave(child, 0.5, ["a1"])
        assert root.rave_visits == {}

    def test_adapt_progressive_widening_high_variance(self):
        engine = ProgressiveWideningEngine(pw_config=ProgressiveWideningConfig(k=1.0, adaptive=True, k_min=0.5))
        rng = engine.rng
        root = RAVENode(state=_make_state(), rng=rng)

        # Create children with high value variance (var > 0.3)
        # Need 3+ children with spread values: e.g., [-0.5, 0.5, 1.5] -> var > 0.3
        for i, val in enumerate([-0.5, 0.5, 1.5]):
            child = RAVENode(state=_make_state(f"c{i}"), parent=root, action=f"a{i}", rng=rng)
            child.visits = 10
            child.value_sum = val * 10
            root.children.append(child)

        initial_k = engine.pw_config.k
        engine._adapt_progressive_widening(root)
        assert engine.pw_config.k < initial_k  # k should decrease (high variance -> explore more)

    def test_adapt_progressive_widening_low_variance(self):
        engine = ProgressiveWideningEngine(pw_config=ProgressiveWideningConfig(k=1.0, adaptive=True, k_max=3.0))
        rng = engine.rng
        root = RAVENode(state=_make_state(), rng=rng)

        # Create children with low value variance
        for i, val in enumerate([0.5, 0.51]):
            child = RAVENode(state=_make_state(f"c{i}"), parent=root, action=f"a{i}", rng=rng)
            child.visits = 5
            child.value_sum = val * 5
            root.children.append(child)

        initial_k = engine.pw_config.k
        engine._adapt_progressive_widening(root)
        assert engine.pw_config.k > initial_k  # k should increase

    def test_adapt_no_children_noop(self):
        engine = ProgressiveWideningEngine(pw_config=ProgressiveWideningConfig(adaptive=True))
        root = RAVENode(state=_make_state(), rng=engine.rng)
        engine._adapt_progressive_widening(root)  # Should not raise

    def test_adapt_single_child_noop(self):
        engine = ProgressiveWideningEngine(pw_config=ProgressiveWideningConfig(adaptive=True))
        rng = engine.rng
        root = RAVENode(state=_make_state(), rng=rng)
        child = RAVENode(state=_make_state("c"), parent=root, action="a1", rng=rng)
        child.visits = 5
        child.value_sum = 2.5
        root.children = [child]
        initial_k = engine.pw_config.k
        engine._adapt_progressive_widening(root)
        assert engine.pw_config.k == initial_k

    def test_select_best_action_no_children(self):
        engine = ProgressiveWideningEngine()
        root = RAVENode(state=_make_state(), rng=engine.rng)
        assert engine._select_best_action(root) is None

    def test_select_best_action(self):
        engine = ProgressiveWideningEngine()
        rng = engine.rng
        root = RAVENode(state=_make_state(), rng=rng)

        child1 = RAVENode(state=_make_state("c1"), parent=root, action="a1", rng=rng)
        child1.visits = 10
        child2 = RAVENode(state=_make_state("c2"), parent=root, action="a2", rng=rng)
        child2.visits = 20
        root.children = [child1, child2]

        assert engine._select_best_action(root) == "a2"

    def test_compute_statistics(self):
        engine = ProgressiveWideningEngine()
        rng = engine.rng
        root = RAVENode(state=_make_state(), rng=rng)
        root.visits = 30
        root.value_sum = 15.0
        root.available_actions = ["a1", "a2", "a3"]

        child = RAVENode(state=_make_state("c1"), parent=root, action="a1", rng=rng)
        child.visits = 20
        child.value_sum = 12.0
        root.children = [child]

        stats = engine._compute_statistics(root, 30)
        assert stats["iterations"] == 30
        assert stats["root_visits"] == 30
        assert stats["num_children"] == 1
        assert stats["max_children"] == 3
        assert stats["best_action"] == "a1"
        assert "action_stats" in stats
        assert "a1" in stats["action_stats"]

    @pytest.mark.asyncio
    async def test_search(self):
        engine = ProgressiveWideningEngine()
        rng = engine.rng
        state = _make_state()
        root = RAVENode(state=state, rng=rng)

        actions = ["a1", "a2"]

        def action_gen(s):
            return actions

        def state_trans(s, a):
            return MCTSState(state_id=f"new_{a}", features={"v": 1})

        policy = AsyncMock()
        policy.evaluate = AsyncMock(return_value=0.5)

        best_action, stats = await engine.search(
            root=root,
            num_iterations=10,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=policy,
            max_rollout_depth=5,
        )

        assert best_action in actions or best_action is None
        assert stats["iterations"] == 10
        assert stats["root_visits"] > 0

    @pytest.mark.asyncio
    async def test_run_iteration_with_adaptive(self):
        engine = ProgressiveWideningEngine(pw_config=ProgressiveWideningConfig(adaptive=True))
        rng = engine.rng
        root = RAVENode(state=_make_state(), rng=rng)
        root.available_actions = ["a1", "a2"]

        policy = AsyncMock()
        policy.evaluate = AsyncMock(return_value=0.6)

        def action_gen(s):
            return ["a1", "a2"]

        def state_trans(s, a):
            return MCTSState(state_id=f"new_{a}", features={"v": 1})

        await engine.run_iteration(root, action_gen, state_trans, policy)
        assert root.visits > 0


@pytest.mark.unit
class TestUtilityFunctions:
    """Tests for create_pw_config and create_rave_config."""

    def test_create_pw_config_small_action_space(self):
        cfg = create_pw_config(action_space_size=5)
        assert cfg.k == 0.5
        assert cfg.alpha == 0.5

    def test_create_pw_config_medium_action_space(self):
        cfg = create_pw_config(action_space_size=30)
        assert cfg.k == 1.0

    def test_create_pw_config_large_action_space(self):
        cfg = create_pw_config(action_space_size=100)
        assert cfg.k == 2.0
        assert cfg.alpha == 0.6

    def test_create_pw_config_very_large_action_space(self):
        cfg = create_pw_config(action_space_size=500)
        assert cfg.k == 5.0
        assert cfg.alpha == 0.7

    def test_create_pw_config_adaptive(self):
        cfg = create_pw_config(action_space_size=5, adaptive=True)
        assert cfg.adaptive is True

    def test_create_rave_config_no_move_ordering(self):
        cfg = create_rave_config(domain_has_move_ordering=False)
        assert cfg.enable_rave is False

    def test_create_rave_config_low_complexity(self):
        cfg = create_rave_config(domain_complexity="low")
        assert cfg.rave_constant == 100.0

    def test_create_rave_config_medium_complexity(self):
        cfg = create_rave_config(domain_complexity="medium")
        assert cfg.rave_constant == 300.0

    def test_create_rave_config_high_complexity(self):
        cfg = create_rave_config(domain_complexity="high")
        assert cfg.rave_constant == 1000.0

    def test_create_rave_config_unknown_complexity(self):
        cfg = create_rave_config(domain_complexity="unknown")
        assert cfg.rave_constant == 300.0  # default
