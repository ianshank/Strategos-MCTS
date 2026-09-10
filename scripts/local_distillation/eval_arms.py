"""Search vs no-search arms and a promotion decision that can reject."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import nn

from scripts.local_distillation.settings import DistillationSettings
from src.config.constants import EVIDENCE_PROVENANCE_RANDOM_WEIGHTS, EVIDENCE_PROVENANCES
from src.framework.mcts.neural_mcts import GameState, NeuralMCTS

# Declared before any holdout run. A later experiment PR may write this path; this
# package does not treat artifacts/ or self-play-convergence plumbing as evidence.
PRIMARY_ENDPOINT = "search_minus_no_search"
COMMITTED_RESULTS_RELATIVE_PATH = "benchmarks/results/local_distillation_c4.json"


@dataclass
class ArmResult:
    """One arm's play: greedy net (0 expansions) or NeuralMCTS."""

    name: str
    action: Any
    expansions: int
    wall_clock_s: float
    provenance: str


@dataclass
class SearchNoSearchComparison:
    """Equal-expansion arms plus a wall-clock-matched no-search repeat count."""

    no_search: ArmResult
    search: ArmResult
    no_search_repeats_in_search_budget: int
    primary_endpoint: str = PRIMARY_ENDPOINT


@dataclass
class PromotionDecision:
    """Gated comparison: a criterion written after seeing the number is not a gate."""

    promote: bool
    reason: str
    candidate: float
    incumbent: float
    min_delta: float


def greedy_network_action(network: nn.Module, state: GameState, *, device: str) -> Any:
    """No-search arm: argmax of the network policy over legal actions."""
    legal = state.get_legal_actions()
    if not legal:
        raise ValueError("no legal actions")
    was_training = bool(network.training)
    network.eval()
    try:
        with torch.no_grad():
            log_probs, _value = network(state.to_tensor().unsqueeze(0).to(device))
    finally:
        network.train(was_training)
    probs = torch.exp(log_probs.squeeze(0)).detach().cpu().numpy()
    best = None
    best_p = -1.0
    for action in legal:
        idx = state.action_to_index(action)
        if 0 <= idx < len(probs) and float(probs[idx]) > best_p:
            best_p = float(probs[idx])
            best = action
    if best is None:
        raise ValueError("no legal action mapped into the policy head")
    return best


async def run_search_arm(
    mcts: NeuralMCTS,
    state: GameState,
    *,
    num_simulations: int,
) -> ArmResult:
    start = time.perf_counter()
    action_probs, _root = await mcts.search(
        state, num_simulations=num_simulations, temperature=0.0, add_root_noise=False
    )
    elapsed = time.perf_counter() - start
    action = mcts.select_action(action_probs, temperature=0.0)
    return ArmResult(
        name="search",
        action=action,
        expansions=num_simulations,
        wall_clock_s=elapsed,
        provenance=EVIDENCE_PROVENANCE_RANDOM_WEIGHTS,
    )


def run_no_search_arm(network: nn.Module, state: GameState, *, device: str) -> ArmResult:
    start = time.perf_counter()
    action = greedy_network_action(network, state, device=device)
    elapsed = time.perf_counter() - start
    return ArmResult(
        name="no_search",
        action=action,
        expansions=0,
        wall_clock_s=elapsed,
        provenance=EVIDENCE_PROVENANCE_RANDOM_WEIGHTS,
    )


async def compare_search_vs_no_search(
    mcts: NeuralMCTS,
    state: GameState,
    *,
    num_simulations: int,
    device: str,
    repeat_cap: int | None = None,
) -> SearchNoSearchComparison:
    """Equal-expansion arms. Wall-clock is recorded; no-search is repeated to match search time."""
    cap = DistillationSettings().wall_clock_repeat_cap if repeat_cap is None else repeat_cap
    search = await run_search_arm(mcts, state, num_simulations=num_simulations)
    no_search = run_no_search_arm(mcts.network, state, device=device)
    repeats = repeat_no_search_until(
        mcts.network,
        state,
        device=device,
        budget_s=search.wall_clock_s,
        repeat_cap=cap,
    )
    if no_search.provenance not in EVIDENCE_PROVENANCES or search.provenance not in EVIDENCE_PROVENANCES:
        raise ValueError("arm provenance must be a member of EVIDENCE_PROVENANCES")
    return SearchNoSearchComparison(
        no_search=no_search,
        search=search,
        no_search_repeats_in_search_budget=repeats,
    )


def repeat_no_search_until(
    network: nn.Module,
    state: GameState,
    *,
    device: str,
    budget_s: float,
    repeat_cap: int,
) -> int:
    """Equal wall-clock helper: greedy rollouts until ``budget_s`` elapses. Returns count."""
    if budget_s < 0:
        raise ValueError("budget_s must be >= 0")
    if repeat_cap < 1:
        raise ValueError("repeat_cap must be >= 1")
    deadline = time.perf_counter() + budget_s
    n = 0
    while time.perf_counter() < deadline:
        greedy_network_action(network, state, device=device)
        n += 1
        if n >= repeat_cap:
            break
    return n


def decide_promotion(
    candidate: float,
    incumbent: float,
    *,
    min_delta: float,
) -> PromotionDecision:
    """Reject a degraded (or insufficiently improved) checkpoint."""
    if candidate < incumbent + min_delta:
        return PromotionDecision(
            promote=False,
            reason="rejected: candidate does not clear incumbent + min_delta",
            candidate=candidate,
            incumbent=incumbent,
            min_delta=min_delta,
        )
    return PromotionDecision(
        promote=True,
        reason="accepted: candidate clears declared gate",
        candidate=candidate,
        incumbent=incumbent,
        min_delta=min_delta,
    )
