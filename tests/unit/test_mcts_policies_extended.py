"""
Tests for MCTS policies module - covers uncovered policy classes.

Tests GreedyRolloutPolicy, HybridRolloutPolicy, LLMRolloutPolicy,
and ProgressiveWideningConfig from policies.py.
"""

import math
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.framework.mcts.core import MCTSState
from src.framework.mcts.policies import (
    GreedyRolloutPolicy,
    HybridRolloutPolicy,
    LLMRolloutPolicy,
    ProgressiveWideningConfig,
)


def _make_state(state_id="s1", features=None):
    return MCTSState(state_id=state_id, features=features or {"score": 0.5})


@pytest.mark.unit
class TestGreedyRolloutPolicy:
    """Tests for GreedyRolloutPolicy."""

    @pytest.mark.asyncio
    async def test_evaluate_uses_heuristic(self):
        def heuristic(state):
            return 0.7

        policy = GreedyRolloutPolicy(heuristic_fn=heuristic, noise_scale=0.0)
        rng = np.random.default_rng(42)
        value = await policy.evaluate(_make_state(), rng)
        assert value == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_evaluate_adds_noise(self):
        def heuristic(state):
            return 0.5

        policy = GreedyRolloutPolicy(heuristic_fn=heuristic, noise_scale=0.1)
        rng = np.random.default_rng(42)
        values = [await policy.evaluate(_make_state(), rng) for _ in range(20)]
        # With noise, not all values should be exactly 0.5
        assert not all(v == 0.5 for v in values)
        # All values should be in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in values)

    @pytest.mark.asyncio
    async def test_evaluate_clamps_to_bounds(self):
        def heuristic(state):
            return 0.99

        policy = GreedyRolloutPolicy(heuristic_fn=heuristic, noise_scale=0.05)
        rng = np.random.default_rng(42)
        for _ in range(20):
            value = await policy.evaluate(_make_state(), rng)
            assert 0.0 <= value <= 1.0


@pytest.mark.unit
class TestHybridRolloutPolicy:
    """Tests for HybridRolloutPolicy."""

    @pytest.mark.asyncio
    async def test_evaluate_with_heuristic(self):
        def heuristic(state):
            return 0.8

        policy = HybridRolloutPolicy(heuristic_fn=heuristic, heuristic_weight=0.7, random_weight=0.3)
        rng = np.random.default_rng(42)
        value = await policy.evaluate(_make_state(), rng)
        assert 0.0 <= value <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_without_heuristic(self):
        policy = HybridRolloutPolicy(heuristic_fn=None)
        rng = np.random.default_rng(42)
        value = await policy.evaluate(_make_state(), rng)
        assert 0.0 <= value <= 1.0

    def test_weight_normalization(self):
        policy = HybridRolloutPolicy(heuristic_weight=3.0, random_weight=1.0)
        assert policy.heuristic_weight == pytest.approx(0.75)
        assert policy.random_weight == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_evaluate_deterministic_seed(self):
        def heuristic(state):
            return 0.6

        policy = HybridRolloutPolicy(heuristic_fn=heuristic, noise_scale=0.1)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        v1 = await policy.evaluate(_make_state(), rng1)
        v2 = await policy.evaluate(_make_state(), rng2)
        assert v1 == v2


@pytest.mark.unit
class TestLLMRolloutPolicy:
    """Tests for LLMRolloutPolicy."""

    @pytest.mark.asyncio
    async def test_evaluate_calls_fn(self):
        eval_fn = AsyncMock(return_value=0.75)
        policy = LLMRolloutPolicy(evaluate_fn=eval_fn, cache_results=False)
        rng = np.random.default_rng(42)
        value = await policy.evaluate(_make_state(), rng)
        assert value == 0.75
        eval_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_caches_results(self):
        eval_fn = AsyncMock(return_value=0.6)
        policy = LLMRolloutPolicy(evaluate_fn=eval_fn, cache_results=True)
        rng = np.random.default_rng(42)
        state = _make_state()
        v1 = await policy.evaluate(state, rng)
        v2 = await policy.evaluate(state, rng)
        assert v1 == v2 == 0.6
        assert eval_fn.call_count == 1  # Cached on second call

    @pytest.mark.asyncio
    async def test_evaluate_no_cache(self):
        eval_fn = AsyncMock(return_value=0.6)
        policy = LLMRolloutPolicy(evaluate_fn=eval_fn, cache_results=False)
        rng = np.random.default_rng(42)
        state = _make_state()
        await policy.evaluate(state, rng)
        await policy.evaluate(state, rng)
        assert eval_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_clamps_value(self):
        eval_fn = AsyncMock(return_value=1.5)
        policy = LLMRolloutPolicy(evaluate_fn=eval_fn)
        rng = np.random.default_rng(42)
        value = await policy.evaluate(_make_state(), rng)
        assert value == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_clamps_negative(self):
        eval_fn = AsyncMock(return_value=-0.5)
        policy = LLMRolloutPolicy(evaluate_fn=eval_fn)
        rng = np.random.default_rng(42)
        value = await policy.evaluate(_make_state(), rng)
        assert value == 0.0


@pytest.mark.unit
class TestPoliciesProgressiveWideningConfig:
    """Tests for ProgressiveWideningConfig in policies.py."""

    def test_defaults(self):
        cfg = ProgressiveWideningConfig()
        assert cfg.k == 1.0
        assert cfg.alpha == 0.5

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            ProgressiveWideningConfig(k=0)

    def test_invalid_alpha_low(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            ProgressiveWideningConfig(alpha=0)

    def test_invalid_alpha_high(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            ProgressiveWideningConfig(alpha=1.0)

    def test_should_expand(self):
        cfg = ProgressiveWideningConfig(k=1.0, alpha=0.5)
        # threshold = 1.0 * 4^0.5 = 2.0
        assert cfg.should_expand(visits=3, num_children=4) is True
        assert cfg.should_expand(visits=1, num_children=4) is False

    def test_min_visits_for_expansion(self):
        cfg = ProgressiveWideningConfig(k=1.0, alpha=0.5)
        min_v = cfg.min_visits_for_expansion(4)
        assert min_v == math.ceil(1.0 * 4**0.5)

    def test_repr(self):
        cfg = ProgressiveWideningConfig(k=2.0, alpha=0.6)
        assert "k=2.0" in repr(cfg)
        assert "alpha=0.6" in repr(cfg)
