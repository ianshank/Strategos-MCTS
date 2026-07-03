"""Policy-comparison benchmark for measuring decision-quality lift (Phase 5.4 / M5).

Compares a baseline (e.g. untrained) policy/value network against a trained one on a
registered self-play :mod:`~src.framework.domain_registry` domain, and reports a
**domain-type-aware** lift metric with a confidence interval:

- single-agent domains (reasoning/planning): mean terminal reward over greedy MCTS
  rollouts — arena win-rate is meaningless because the reward is non-negative; lift is
  the relative improvement of that mean, with a CI derived from the per-game rewards.
- adversarial domains (e.g. chess): win-rate of trained vs baseline (head-to-head),
  lift = (win_rate - 0.5) * 2, with a Wilson score CI on the win-rate.

**Gate semantics (M5 ">=20% decision-quality lift").** ``meets_target`` is the
*confidence-interval lower bound* clearing the target, not the point estimate — a run
with too few games or too much variance fails the gate even when the point lift looks
good (fail-closed). The point-estimate check is still available as ``point_meets_target``.

**Smoke-test domains.** The built-in reasoning/planning domains have synthetic,
trivially exploitable rewards (see ``PlanningState.get_reward``); lifts measured on
them validate the plumbing, not decision quality. The M5 acceptance claim must come
from an adversarial domain with an external notion of success (e.g. chess win-rate).

Run from the command line via ``python -m src.benchmark.policy_lift`` (or the
``policy-lift`` console script).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from src.framework.domain_registry import METRIC_MEAN_REWARD, METRIC_WIN_RATE, DomainRegistry
from src.framework.mcts.neural_mcts import NeuralMCTS
from src.observability.logging import get_logger
from src.training.system_config import MCTSConfig
from src.utils.stats import difference_confidence_interval, wilson_score_interval

logger = get_logger(__name__)

# Default target from the M5 acceptance criterion.
DEFAULT_TARGET_LIFT_PCT = 20.0
# Relative lift is meaningless against a near-zero baseline; below this floor the
# benchmark reports an absolute-points delta instead (see _relative_lift).
DEFAULT_MIN_BASELINE = 0.05
# Per-metric game counts: enough that the CI lower bound can clear the target when the
# effect is real (Wilson at n=100 resolves ~±0.10 on win-rate).
DEFAULT_GAMES_WIN_RATE = 100
DEFAULT_GAMES_MEAN_REWARD = 30
MIN_RECOMMENDED_GAMES = {
    METRIC_WIN_RATE: DEFAULT_GAMES_WIN_RATE,
    METRIC_MEAN_REWARD: DEFAULT_GAMES_MEAN_REWARD,
}


@dataclass
class PolicyComparisonResult:
    """Outcome of a baseline-vs-trained comparison on one domain."""

    domain: str
    metric: str
    baseline_score: float
    trained_score: float
    lift_pct: float
    num_games: int
    # Measurement-validity fields (defaulted for backward compatibility).
    lift_ci_lower_pct: float | None = None
    lift_ci_upper_pct: float | None = None
    confidence: float = 0.95
    absolute_delta: float = 0.0
    lift_is_absolute_fallback: bool = False
    target_lift_pct: float = DEFAULT_TARGET_LIFT_PCT

    @property
    def meets_target(self) -> bool:
        """CI-lower-bound gate: the lift is credibly >= target, not just on average.

        Fail-closed: a result without a confidence interval never meets the target.
        """
        if self.lift_ci_lower_pct is None:
            return False
        return self.lift_ci_lower_pct >= self.target_lift_pct

    @property
    def point_meets_target(self) -> bool:
        """Legacy point-estimate check (reporting only — not the acceptance gate)."""
        return self.lift_pct >= self.target_lift_pct


async def _terminal_rewards(
    network: nn.Module,
    domain: str,
    num_games: int,
    mcts_config: MCTSConfig,
    *,
    max_moves: int,
    device: str,
) -> list[float]:
    """Per-game terminal rewards over greedy single-agent rollouts."""
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
    return rewards


async def compare_policies(
    domain: str,
    baseline_network: nn.Module,
    trained_network: nn.Module,
    *,
    num_games: int | None = None,
    mcts_config: MCTSConfig | None = None,
    max_moves: int = 50,
    device: str = "cpu",
    confidence: float = 0.95,
    min_baseline: float = DEFAULT_MIN_BASELINE,
    target_lift_pct: float = DEFAULT_TARGET_LIFT_PCT,
) -> PolicyComparisonResult:
    """Compare two networks on ``domain`` and return the lift result with a CI.

    The metric is selected from the domain registration (single-agent vs adversarial).
    ``num_games`` defaults per metric (win-rate: 100, mean-reward: 30); explicitly
    passing fewer logs a warning because the CI lower bound — which is what
    ``meets_target`` gates on — will rarely clear the target at small n.

    The mean-reward lift CI uses a fixed denominator ``max(|baseline|, min_baseline)``
    on a difference-of-means interval; this ignores the (second-order) sampling
    variance of the denominator itself. A bootstrap over the lift ratio is a possible
    future refinement, deliberately out of scope here.
    """
    spec = DomainRegistry.get(domain)
    mcts_config = mcts_config or MCTSConfig()

    if num_games is None:
        num_games = MIN_RECOMMENDED_GAMES.get(spec.metric, DEFAULT_GAMES_MEAN_REWARD)
    elif num_games < MIN_RECOMMENDED_GAMES.get(spec.metric, 0):
        logger.warning(
            "num_games below the recommended minimum; the CI lower bound will be wide",
            extra={
                "domain": domain,
                "num_games": num_games,
                "recommended": MIN_RECOMMENDED_GAMES.get(spec.metric),
            },
        )

    lift_is_absolute_fallback = False
    if spec.metric == METRIC_MEAN_REWARD:
        baseline_rewards = await _terminal_rewards(
            baseline_network, domain, num_games, mcts_config, max_moves=max_moves, device=device
        )
        trained_rewards = await _terminal_rewards(
            trained_network, domain, num_games, mcts_config, max_moves=max_moves, device=device
        )
        baseline = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.0
        trained = sum(trained_rewards) / len(trained_rewards) if trained_rewards else 0.0
        lift_pct = _relative_lift(baseline, trained, min_baseline=min_baseline)
        delta, delta_lower, delta_upper = difference_confidence_interval(baseline_rewards, trained_rewards, confidence)
        if abs(baseline) < min_baseline:
            # Absolute-points fallback (matches _relative_lift below the floor).
            lift_is_absolute_fallback = True
            lift_ci_lower, lift_ci_upper = delta_lower * 100.0, delta_upper * 100.0
        else:
            denom = abs(baseline)
            lift_ci_lower, lift_ci_upper = delta_lower / denom * 100.0, delta_upper / denom * 100.0
        absolute_delta = delta
    elif spec.metric == METRIC_WIN_RATE:
        # Adversarial head-to-head is delegated to the existing arena evaluator.
        from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator

        mcts = NeuralMCTS(trained_network, mcts_config, device=device, single_agent=False)
        evaluator = SelfPlayEvaluator(
            mcts,
            spec.initial_state_fn,
            # play_game() reads simulations from EvaluationConfig.mcts_iterations, not
            # from the NeuralMCTS config — wire them together so callers control both.
            EvaluationConfig(num_games=num_games, mcts_iterations=mcts_config.num_simulations),
            device=device,
        )
        metrics = await evaluator.evaluate(trained_network, baseline_network)
        win_rate = float(metrics.get("win_rate", 0.0))
        # SelfPlayEvaluator reports "eval_games"; tolerate variants and empty runs.
        games_played = int(metrics.get("eval_games") or metrics.get("games_played") or num_games)
        if "wins" in metrics:
            successes = float(metrics["wins"]) + 0.5 * float(metrics.get("draws", 0))
        else:  # evaluator variants that only report the rate
            successes = win_rate * games_played
        p_lower, p_upper = wilson_score_interval(successes, games_played, confidence)
        baseline, trained = 0.5, win_rate
        # lift = (p - 0.5) * 200 is affine and monotone, so the CI maps through directly.
        lift_pct = (win_rate - 0.5) * 2.0 * 100.0
        lift_ci_lower = (p_lower - 0.5) * 2.0 * 100.0
        lift_ci_upper = (p_upper - 0.5) * 2.0 * 100.0
        absolute_delta = win_rate - 0.5
    else:  # pragma: no cover - guarded by registry validation
        raise ValueError(f"Unsupported metric '{spec.metric}' for domain '{domain}'")

    result = PolicyComparisonResult(
        domain=domain,
        metric=spec.metric,
        baseline_score=baseline,
        trained_score=trained,
        lift_pct=lift_pct,
        num_games=num_games,
        lift_ci_lower_pct=lift_ci_lower,
        lift_ci_upper_pct=lift_ci_upper,
        confidence=confidence,
        absolute_delta=absolute_delta,
        lift_is_absolute_fallback=lift_is_absolute_fallback,
        target_lift_pct=target_lift_pct,
    )
    logger.info(
        "Policy comparison complete",
        extra={
            "domain": domain,
            "metric": spec.metric,
            "baseline": baseline,
            "trained": trained,
            "lift_pct": lift_pct,
            "lift_ci_lower_pct": lift_ci_lower,
            "lift_ci_upper_pct": lift_ci_upper,
            "meets_target": result.meets_target,
        },
    )
    return result


def _relative_lift(baseline: float, trained: float, *, min_baseline: float = DEFAULT_MIN_BASELINE) -> float:
    """Relative improvement (%) of ``trained`` over ``baseline``.

    Falls back to an absolute-point delta (×100) when ``|baseline| < min_baseline`` —
    dividing by a near-zero baseline would let a trivial absolute gain masquerade as a
    huge relative lift.
    """
    if abs(baseline) < min_baseline:
        return (trained - baseline) * 100.0
    return (trained - baseline) / abs(baseline) * 100.0


__all__ = [
    "PolicyComparisonResult",
    "compare_policies",
    "DEFAULT_TARGET_LIFT_PCT",
    "DEFAULT_MIN_BASELINE",
    "DEFAULT_GAMES_WIN_RATE",
    "DEFAULT_GAMES_MEAN_REWARD",
]
