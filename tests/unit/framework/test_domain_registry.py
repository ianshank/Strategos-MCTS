"""Tests for the self-play domain registry and single-agent domain wrappers (Phase 5.2).

Validates:
- the GameState contract for every registered domain (interface completeness + a
  one-step transition sanity check),
- registry behavior (lookup, metadata, error handling),
- that each registered single-agent domain runs a real self-play iteration end-to-end
  through SelfPlayTrainer (proving the dict-action -> hashable-action wrapper works with
  NeuralMCTS, which keys children by action).
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.framework.domain_registry import (
    METRIC_MEAN_REWARD,
    DomainRegistry,
    DomainSpec,
    register_domain,
)
from src.framework.mcts.single_agent_domains import make_reasoning_state
from src.training.self_play_trainer import SelfPlayConfig, SelfPlayTrainer
from src.training.system_config import MCTSConfig

pytestmark = [pytest.mark.unit]

_REGISTERED = ["reasoning", "planning"]


class _TinyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.policy = nn.Linear(in_dim, n_actions)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.log_softmax(self.policy(x), dim=1), torch.tanh(self.value(x))


@pytest.mark.parametrize("domain", _REGISTERED)
def test_registered_domains_satisfy_gamestate_contract(domain):
    state = DomainRegistry.get_initial_state(domain)

    # Interface completeness.
    actions = state.get_legal_actions()
    assert isinstance(actions, list) and actions, "expected non-empty legal actions"
    assert all(isinstance(a, str) for a in actions), "actions must be hashable strings for NeuralMCTS"
    assert isinstance(state.is_terminal(), bool)
    assert isinstance(state.get_reward(), float)
    tensor = state.to_tensor()
    assert isinstance(tensor, torch.Tensor) and tensor.ndim == 1
    assert isinstance(state.get_hash(), str)

    # Action indices fall within the declared policy-head width.
    size = DomainRegistry.action_space_size(domain)
    for action in actions:
        idx = state.action_to_index(action)
        assert 0 <= idx <= size  # <= size allows the documented "unknown" bucket

    # One-step transition sanity: applying a legal action yields a new state.
    nxt = state.apply_action(actions[0])
    assert nxt is not state
    assert isinstance(nxt.to_tensor(), torch.Tensor)


def test_registry_metadata_and_errors():
    assert set(_REGISTERED).issubset(set(DomainRegistry.list_domains()))
    for domain in _REGISTERED:
        assert DomainRegistry.is_single_agent(domain) is True
        assert DomainRegistry.metric(domain) == METRIC_MEAN_REWARD
    with pytest.raises(KeyError):
        DomainRegistry.get("does_not_exist")


def test_register_rejects_invalid_metric_and_size():
    with pytest.raises(ValueError):
        register_domain("bad_metric", make_reasoning_state, 8, single_agent=True, metric="nonsense")
    with pytest.raises(ValueError):
        DomainRegistry.register(DomainSpec("bad_size", make_reasoning_state, 0, True, METRIC_MEAN_REWARD))


@pytest.mark.parametrize("domain", _REGISTERED)
def test_self_play_iteration_runs_for_registered_domain(domain):
    """Each single-agent domain runs a real self-play iteration through NeuralMCTS."""
    spec = DomainRegistry.get(domain)
    sample = spec.initial_state_fn()
    in_dim = sample.to_tensor().shape[0]

    mcts_config = MCTSConfig()
    mcts_config.num_simulations = 4  # tiny for speed

    trainer = SelfPlayTrainer(
        network=_TinyNet(in_dim, spec.action_space_size),
        initial_state_fn=spec.initial_state_fn,
        action_space_size=spec.action_space_size,
        mcts_config=mcts_config,
        config=SelfPlayConfig(num_games_per_iteration=1, batch_size=4, buffer_capacity=50),
        single_agent=spec.single_agent,
        seed=7,
    )

    metrics = asyncio.run(trainer.train_iteration())

    assert metrics.games_played == 1
    assert metrics.examples_collected > 0
    assert metrics.train_steps >= 1
