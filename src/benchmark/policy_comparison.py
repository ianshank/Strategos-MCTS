"""Policy-comparison benchmark for measuring decision-quality lift (Phase 5.4).

Compares a baseline (e.g. untrained) policy/value network against a trained one on a
registered self-play :mod:`~src.framework.domain_registry` domain, and reports a
**domain-type-aware** lift metric:

- single-agent domains (reasoning/planning): mean terminal reward over greedy MCTS
  rollouts — arena win-rate is meaningless because the reward is non-negative; lift is
  the relative improvement of that mean.
- adversarial domains (e.g. chess): win-rate of trained vs baseline (head-to-head),
  lift = (win_rate - 0.5) * 2.

This is the harness used to verify the M5 ">=20% decision-quality lift" acceptance
criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import nn

from src.framework.domain_registry import METRIC_MEAN_REWARD, METRIC_WIN_RATE, DomainRegistry
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.observability.logging import get_logger
from src.training.system_config import MCTSConfig

logger = get_logger(__name__)


@dataclass
class PolicyComparisonResult:
    """Outcome of a baseline-vs-trained comparison on one domain."""

    domain: str
    metric: str
    baseline_score: float
    trained_score: float
    lift_pct: float
    num_games: int

    @property
    def meets_target(self) -> bool:
        """Whether the lift meets the M5 >=20% target."""
        return self.lift_pct >= 20.0


async def _mean_terminal_reward(
    network: nn.Module,
    domain: str,
    num_games: int,
    mcts_config: MCTSConfig,
    *,
    max_moves: int,
    device: str,
) -> float:
    """Average terminal reward over greedy single-agent rollouts."""
    mcts = NeuralMCTS(network, mcts_config, device=device, single_agent=True)
    rewards: list[float] = []
    for _ in range(num_games):
        state = DomainRegistry.get_initial_state(domain)
        moves = 0
        while not state.is_terminal() and moves < max_moves:
            action_probs, _ = await mcts.search(state, temperature=0.0, add_root_noise=False)
            action = mcts.select_action(action_probs, deterministic=True)
            if action is None:
                break
            state = state.apply_action(action)
            moves += 1
        rewards.append(float(state.get_reward()))
    return float(np.mean(rewards)) if rewards else 0.0


async def compare_policies(
    domain: str,
    baseline_network: nn.Module,
    trained_network: nn.Module,
    *,
    num_games: int = 20,
    mcts_config: MCTSConfig | None = None,
    max_moves: int = 50,
    device: str = "cpu",
) -> PolicyComparisonResult:
    """Compare two networks on ``domain`` and return the lift result.

    The metric is selected from the domain registration (single-agent vs adversarial).
    """
    spec = DomainRegistry.get(domain)
    mcts_config = mcts_config or MCTSConfig()

    if spec.metric == METRIC_MEAN_REWARD:
        baseline = await _mean_terminal_reward(
            baseline_network, domain, num_games, mcts_config, max_moves=max_moves, device=device
        )
        trained = await _mean_terminal_reward(
            trained_network, domain, num_games, mcts_config, max_moves=max_moves, device=device
        )
        lift_pct = _relative_lift(baseline, trained)
    elif spec.metric == METRIC_WIN_RATE:
        # Adversarial head-to-head is delegated to the existing arena evaluator.
        from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator

        mcts = NeuralMCTS(trained_network, mcts_config, device=device, single_agent=False)
        evaluator = SelfPlayEvaluator(
            mcts,
            spec.initial_state_fn,
            EvaluationConfig(num_games=num_games),
            device=device,
        )
        metrics = await evaluator.evaluate(trained_network, baseline_network)
        win_rate = float(metrics.get("win_rate", 0.0))
        baseline, trained = 0.5, win_rate
        lift_pct = (win_rate - 0.5) * 2.0 * 100.0
    else:  # pragma: no cover - guarded by registry validation
        raise ValueError(f"Unsupported metric '{spec.metric}' for domain '{domain}'")

    result = PolicyComparisonResult(
        domain=domain,
        metric=spec.metric,
        baseline_score=baseline,
        trained_score=trained,
        lift_pct=lift_pct,
        num_games=num_games,
    )
    logger.info(
        "Policy comparison complete",
        extra={
            "domain": domain,
            "metric": spec.metric,
            "baseline": baseline,
            "trained": trained,
            "lift_pct": lift_pct,
            "meets_target": result.meets_target,
        },
    )
    return result


def _relative_lift(baseline: float, trained: float) -> float:
    """Relative improvement (%) of ``trained`` over ``baseline``.

    Falls back to an absolute-point delta (×100) when the baseline is ~0 to avoid
    divide-by-zero blow-ups.
    """
    if abs(baseline) < 1e-9:
        return (trained - baseline) * 100.0
    return (trained - baseline) / abs(baseline) * 100.0


__all__ = ["PolicyComparisonResult", "compare_policies"]
