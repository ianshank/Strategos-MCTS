"""Unit tests for the generalized SelfPlayTrainer (Phase 5.1).

Validates:
- single-agent self-play does NOT negate value targets or alternate players,
- two-player (default) behavior is preserved (negamax sign flipping),
- a full train iteration runs, updates the network, and grows the buffer,
- torch-safe checkpoint round-trip.

Uses a tiny deterministic single-agent GameState and a tiny policy/value network so
the test is fast and CPU-only (no chess / heavy deps).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.framework.mcts.neural_mcts import GameState, NeuralMCTS, SelfPlayCollector
from src.training.self_play_trainer import SelfPlayConfig, SelfPlayTrainer
from src.training.system_config import MCTSConfig

pytestmark = [pytest.mark.unit]

_ACTIONS = ["up", "stay"]
_ACTION_SPACE = 2


class _ClimbState(GameState):
    """Single-agent toy domain: climb from 0 to ``target`` within ``max_steps``."""

    def __init__(self, pos: int = 0, steps: int = 0, target: int = 2, max_steps: int = 3):
        self.pos = pos
        self.steps = steps
        self.target = target
        self.max_steps = max_steps

    def get_legal_actions(self) -> list[str]:
        return list(_ACTIONS)

    def apply_action(self, action: str) -> _ClimbState:
        new_pos = self.pos + 1 if action == "up" else self.pos
        return _ClimbState(new_pos, self.steps + 1, self.target, self.max_steps)

    def is_terminal(self) -> bool:
        return self.pos >= self.target or self.steps >= self.max_steps

    def get_reward(self, player: int = 1) -> float:
        return min(self.pos / self.target, 1.0)  # absolute, non-negative [0, 1]

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.pos / self.target, self.steps / self.max_steps], dtype=torch.float32)

    def get_hash(self) -> str:
        return f"{self.pos}_{self.steps}"

    def action_to_index(self, action: str) -> int:
        return _ACTIONS.index(action)


class _TwoPlyState(_ClimbState):
    """Deterministic two-step game returning a fixed terminal reward of 1.0."""

    def apply_action(self, action: str) -> _TwoPlyState:
        return _TwoPlyState(self.pos + 1, self.steps + 1, self.target, self.max_steps)

    def is_terminal(self) -> bool:
        return self.steps >= 2

    def get_reward(self, player: int = 1) -> float:
        return 1.0 if self.is_terminal() else 0.0


class _TinyNet(nn.Module):
    def __init__(self, in_dim: int = 2, n_actions: int = _ACTION_SPACE):
        super().__init__()
        self.policy = nn.Linear(in_dim, n_actions)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.log_softmax(self.policy(x), dim=1), torch.tanh(self.value(x))


def _mcts_config() -> MCTSConfig:
    cfg = MCTSConfig()
    cfg.num_simulations = 4  # tiny for speed/determinism
    return cfg


def test_single_agent_collector_does_not_negate_value_targets():
    np.random.seed(0)
    torch.manual_seed(0)
    mcts = NeuralMCTS(_TinyNet(), _mcts_config(), single_agent=True)
    collector = SelfPlayCollector(mcts, _mcts_config(), action_space_size=_ACTION_SPACE)

    examples = asyncio.run(collector.play_game(_ClimbState()))

    assert examples, "expected at least one example"
    assert all(ex.player == 1 for ex in examples)  # no player alternation
    outcome = examples[0].value_target
    assert all(ex.value_target == outcome for ex in examples)  # shared absolute outcome
    assert outcome >= 0.0
    assert all(ex.policy_target.shape == (_ACTION_SPACE,) for ex in examples)  # aligned targets


def test_two_player_value_targets_alternate_sign():
    np.random.seed(0)
    torch.manual_seed(0)
    mcts = NeuralMCTS(_TinyNet(), _mcts_config(), single_agent=False)
    collector = SelfPlayCollector(mcts, _mcts_config(), action_space_size=_ACTION_SPACE)

    examples = asyncio.run(collector.play_game(_TwoPlyState()))

    assert len(examples) == 2
    assert [ex.player for ex in examples] == [1, -1]  # alternation preserved
    assert examples[0].value_target == 1.0
    assert examples[1].value_target == -1.0  # negamax sign flip preserved


def test_train_iteration_runs_and_updates_network():
    config = SelfPlayConfig(num_games_per_iteration=2, batch_size=4, buffer_capacity=100)
    trainer = SelfPlayTrainer(
        network=_TinyNet(),
        initial_state_fn=_ClimbState,
        action_space_size=_ACTION_SPACE,
        mcts_config=_mcts_config(),
        config=config,
        single_agent=True,
        seed=123,
    )
    before = [p.detach().clone() for p in trainer.network.parameters()]

    metrics = asyncio.run(trainer.train_iteration())

    assert metrics.games_played == 2
    assert metrics.examples_collected > 0
    assert metrics.buffer_size == metrics.examples_collected
    assert metrics.train_steps >= 1
    assert np.isfinite(metrics.total_loss)
    after = list(trainer.network.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "expected a weight update"


def test_checkpoint_roundtrip(tmp_path):
    trainer = SelfPlayTrainer(
        network=_TinyNet(),
        initial_state_fn=_ClimbState,
        action_space_size=_ACTION_SPACE,
        mcts_config=_mcts_config(),
        single_agent=True,
        seed=1,
    )
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    fresh = SelfPlayTrainer(
        network=_TinyNet(),
        initial_state_fn=_ClimbState,
        action_space_size=_ACTION_SPACE,
        mcts_config=_mcts_config(),
        single_agent=True,
        seed=2,  # different seed -> different init, then overwritten by load
    )
    fresh.load_checkpoint(path)

    for p1, p2 in zip(trainer.network.parameters(), fresh.network.parameters()):
        assert torch.equal(p1, p2)
