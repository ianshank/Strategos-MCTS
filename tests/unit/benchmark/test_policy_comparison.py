"""Tests for the policy-comparison benchmark (Phase 5.4).

Covers the domain-type-aware lift metric: the relative-lift math, the single-agent
(mean-reward) branch, the adversarial (win-rate) branch, and a real end-to-end
single-agent comparison through NeuralMCTS.
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


def test_single_agent_branch_uses_mean_reward(monkeypatch):
    """compare_policies computes relative lift from mean terminal reward."""
    scores = iter([0.5, 0.65])  # baseline, trained

    async def _fake_mean(*_args, **_kwargs):
        return next(scores)

    monkeypatch.setattr(pc, "_mean_terminal_reward", _fake_mean)

    result = asyncio.run(pc.compare_policies("reasoning", _TinyNet(1, 1), _TinyNet(1, 1), num_games=3))
    assert result.metric == "mean_reward"
    assert result.baseline_score == pytest.approx(0.5)
    assert result.trained_score == pytest.approx(0.65)
    assert result.lift_pct == pytest.approx(30.0)
    assert result.meets_target is True  # 30% >= 20%


def test_adversarial_branch_uses_win_rate(monkeypatch):
    """Adversarial domains use head-to-head win-rate for lift."""
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
            return {"win_rate": 0.75}

    monkeypatch.setattr("src.training.agent_trainer.SelfPlayEvaluator", _FakeEvaluator)

    result = asyncio.run(pc.compare_policies("fake_adversarial", _TinyNet(1, 1), _TinyNet(1, 1), num_games=4))
    assert result.metric == METRIC_WIN_RATE
    assert result.trained_score == pytest.approx(0.75)
    assert result.lift_pct == pytest.approx(50.0)  # (0.75 - 0.5) * 2 * 100


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
