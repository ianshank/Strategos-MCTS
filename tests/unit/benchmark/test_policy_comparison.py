"""Tests for the policy-comparison benchmark (Phase 5.4 / M5).

Covers the domain-type-aware lift metric: the relative-lift math (including the
small-baseline floor), the single-agent (mean-reward) branch with its difference CI,
the adversarial (win-rate) branch with its Wilson CI, the fail-closed CI-lower-bound
gate, and a real end-to-end single-agent comparison through NeuralMCTS.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.benchmark import policy_comparison as pc
from src.framework.domain_registry import METRIC_WIN_RATE, DomainRegistry, register_domain
from src.framework.mcts.single_agent_domains import make_reasoning_state
from src.training.system_config import MCTSConfig
from src.utils.stats import wilson_score_interval

pytestmark = [pytest.mark.unit]


class _TinyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.policy = nn.Linear(in_dim, n_actions)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.log_softmax(self.policy(x), dim=1), torch.tanh(self.value(x))


def test_relative_lift_math():
    assert pc._relative_lift(0.5, 0.6) == pytest.approx(20.0)
    assert pc._relative_lift(0.5, 0.5) == pytest.approx(0.0)
    # Zero baseline falls back to an absolute-point delta (avoids divide-by-zero).
    assert pc._relative_lift(0.0, 0.3) == pytest.approx(30.0)


def test_relative_lift_small_baseline_floor():
    """Baselines below the floor use the absolute-points fallback, not division."""
    # Naive division would report (0.3 - 0.01) / 0.01 * 100 = 2900%.
    assert pc._relative_lift(0.01, 0.3) == pytest.approx(29.0)
    # At/above the floor, the relative formula applies.
    assert pc._relative_lift(0.05, 0.06) == pytest.approx(20.0)
    # The floor is configurable.
    assert pc._relative_lift(0.01, 0.3, min_baseline=0.001) == pytest.approx(2900.0)


def test_single_agent_branch_uses_mean_reward(monkeypatch):
    """compare_policies computes relative lift + CI from per-game terminal rewards."""
    reward_lists = iter([[0.5, 0.5, 0.5, 0.5], [0.6, 0.7, 0.6, 0.7]])  # baseline, trained

    async def _fake_rewards(*_args, **_kwargs):
        return next(reward_lists)

    monkeypatch.setattr(pc, "_terminal_rewards", _fake_rewards)

    result = asyncio.run(pc.compare_policies("reasoning", _TinyNet(1, 1), _TinyNet(1, 1), num_games=4))
    assert result.metric == "mean_reward"
    assert result.baseline_score == pytest.approx(0.5)
    assert result.trained_score == pytest.approx(0.65)
    assert result.lift_pct == pytest.approx(30.0)
    assert result.absolute_delta == pytest.approx(0.15)
    assert result.lift_is_absolute_fallback is False
    # CI populated and centered on the lift (baseline variance is zero here).
    assert result.lift_ci_lower_pct is not None
    assert result.lift_ci_upper_pct is not None
    assert result.lift_ci_lower_pct < 30.0 < result.lift_ci_upper_pct
    assert result.point_meets_target is True  # 30% >= 20%
    # The gate uses the CI lower bound: n=4 with per-game spread does not clear 20%.
    assert result.meets_target is (result.lift_ci_lower_pct >= 20.0)


def test_single_agent_small_baseline_uses_absolute_fallback(monkeypatch):
    """Below the baseline floor the lift and its CI are absolute points, not ratios."""
    reward_lists = iter([[0.0, 0.02], [0.28, 0.32]])

    async def _fake_rewards(*_args, **_kwargs):
        return next(reward_lists)

    monkeypatch.setattr(pc, "_terminal_rewards", _fake_rewards)

    result = asyncio.run(pc.compare_policies("reasoning", _TinyNet(1, 1), _TinyNet(1, 1), num_games=2))
    assert result.lift_is_absolute_fallback is True
    assert result.lift_pct == pytest.approx(29.0)  # (0.30 - 0.01) * 100 points
    assert result.lift_ci_lower_pct is not None
    assert result.lift_ci_lower_pct < 29.0


def test_adversarial_branch_uses_win_rate(monkeypatch):
    """Adversarial domains use head-to-head win-rate with a Wilson CI for lift."""
    register_domain(
        "fake_adversarial",
        make_reasoning_state,  # state factory irrelevant here (evaluator is patched)
        action_space_size=8,
        single_agent=False,
        metric=METRIC_WIN_RATE,
    )

    class _FakeEvaluator:
        def __init__(self, *_a, **_k): ...

        async def evaluate(self, *_a, **_k):
            # Key names mirror the real SelfPlayEvaluator.evaluate return dict.
            return {"win_rate": 0.75, "wins": 3, "draws": 0, "losses": 1, "eval_games": 4}

    monkeypatch.setattr("src.training.agent_trainer.SelfPlayEvaluator", _FakeEvaluator)

    result = asyncio.run(pc.compare_policies("fake_adversarial", _TinyNet(1, 1), _TinyNet(1, 1), num_games=4))
    assert result.metric == METRIC_WIN_RATE
    assert result.trained_score == pytest.approx(0.75)
    assert result.lift_pct == pytest.approx(50.0)  # (0.75 - 0.5) * 2 * 100
    assert result.absolute_delta == pytest.approx(0.25)
    # The lift CI is the Wilson interval mapped through the affine lift transform.
    p_lower, p_upper = wilson_score_interval(3, 4)
    assert result.lift_ci_lower_pct == pytest.approx((p_lower - 0.5) * 200.0)
    assert result.lift_ci_upper_pct == pytest.approx((p_upper - 0.5) * 200.0)
    # n=4 is far too small for the Wilson lower bound to clear 20%: fail-closed.
    assert result.point_meets_target is True
    assert result.meets_target is False


def test_adversarial_branch_handles_draws_and_missing_counts(monkeypatch):
    """Draw-adjusted counts feed the CI; rate-only evaluators fall back gracefully."""
    register_domain(
        "fake_adversarial_draws",
        make_reasoning_state,
        action_space_size=8,
        single_agent=False,
        metric=METRIC_WIN_RATE,
    )

    class _RateOnlyEvaluator:
        def __init__(self, *_a, **_k): ...

        async def evaluate(self, *_a, **_k):
            return {"win_rate": 0.8}

    monkeypatch.setattr("src.training.agent_trainer.SelfPlayEvaluator", _RateOnlyEvaluator)

    result = asyncio.run(pc.compare_policies("fake_adversarial_draws", _TinyNet(1, 1), _TinyNet(1, 1), num_games=10))
    p_lower, p_upper = wilson_score_interval(8.0, 10)  # 0.8 * 10 reconstructed successes
    assert result.lift_ci_lower_pct == pytest.approx((p_lower - 0.5) * 200.0)
    assert result.lift_ci_upper_pct == pytest.approx((p_upper - 0.5) * 200.0)


def test_num_games_defaults_per_metric(monkeypatch):
    """Omitting num_games resolves to the per-metric recommended minimum."""
    captured: dict[str, int] = {}

    async def _fake_rewards(_net, _domain, num_games, *_args, **_kwargs):
        captured["num_games"] = num_games
        return [0.5] * num_games

    monkeypatch.setattr(pc, "_terminal_rewards", _fake_rewards)

    result = asyncio.run(pc.compare_policies("reasoning", _TinyNet(1, 1), _TinyNet(1, 1)))
    assert captured["num_games"] == pc.DEFAULT_GAMES_MEAN_REWARD
    assert result.num_games == pc.DEFAULT_GAMES_MEAN_REWARD


class TestMeetsTargetGate:
    """meets_target is the CI-lower-bound gate; point_meets_target keeps old semantics."""

    def _result(self, **overrides):
        base = {
            "domain": "reasoning",
            "metric": "mean_reward",
            "baseline_score": 0.5,
            "trained_score": 0.65,
            "lift_pct": 30.0,
            "num_games": 100,
        }
        base.update(overrides)
        return pc.PolicyComparisonResult(**base)

    def test_point_above_but_ci_below_target_fails(self):
        result = self._result(lift_ci_lower_pct=12.0, lift_ci_upper_pct=48.0)
        assert result.point_meets_target is True
        assert result.meets_target is False

    def test_ci_lower_bound_at_target_passes(self):
        result = self._result(lift_ci_lower_pct=20.0, lift_ci_upper_pct=40.0)
        assert result.meets_target is True

    def test_missing_ci_fails_closed(self):
        result = self._result(lift_ci_lower_pct=None, lift_ci_upper_pct=None)
        assert result.point_meets_target is True
        assert result.meets_target is False

    def test_custom_target(self):
        result = self._result(lift_ci_lower_pct=15.0, lift_ci_upper_pct=45.0, target_lift_pct=10.0)
        assert result.meets_target is True


def test_end_to_end_single_agent_runs():
    """A real (tiny) single-agent comparison executes and returns a structured result."""
    sample = make_reasoning_state()
    in_dim = sample.to_tensor().shape[0]
    size = DomainRegistry.action_space_size("reasoning")
    cfg = MCTSConfig()
    cfg.num_simulations = 4

    result = asyncio.run(
        pc.compare_policies(
            "reasoning",
            _TinyNet(in_dim, size),
            _TinyNet(in_dim, size),
            num_games=2,
            mcts_config=cfg,
            max_moves=10,
        )
    )
    assert result.domain == "reasoning"
    assert result.metric == "mean_reward"
    assert 0.0 <= result.baseline_score <= 1.0
    assert 0.0 <= result.trained_score <= 1.0
    assert isinstance(result.lift_pct, float)
    assert isinstance(result.lift_ci_lower_pct, float)
    assert isinstance(result.lift_ci_upper_pct, float)
    assert result.lift_ci_lower_pct <= result.lift_ci_upper_pct
