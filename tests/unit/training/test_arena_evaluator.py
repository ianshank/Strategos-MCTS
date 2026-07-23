"""Real-state arena (SelfPlayEvaluator) tests — no mocked MCTS.

Covers the win/loss attribution (F1), the per-ply cache isolation (1a), and deterministic
evaluation (1b) of the two-player arena used by the M5 win-rate gate. Uses a deterministic
1-ply domain whose outcome is fixed (the first mover always wins), so the result depends only
on the code under test, not on the untrained networks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.framework.mcts.neural_mcts import GameState, NeuralMCTS
from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator
from src.training.system_config import MCTSConfig

pytestmark = [pytest.mark.unit]


class _FirstMoverWinsState(GameState):
    """Two-player 1-ply game: whoever moves first (white / player 1) always wins."""

    def __init__(self, moved: bool = False):
        self._moved = moved

    def get_legal_actions(self) -> list[str]:
        return [] if self._moved else ["move"]

    def apply_action(self, action: str) -> _FirstMoverWinsState:
        return _FirstMoverWinsState(moved=True)

    def is_terminal(self) -> bool:
        return self._moved

    def get_reward(self, player: int = 1) -> float:
        if not self._moved:
            return 0.0
        # White (the first mover) won: +1 from white's perspective (player==1), -1 otherwise
        # (mirrors ChessGameState.get_reward's white-vs-black convention).
        return 1.0 if player == 1 else -1.0

    def to_tensor(self) -> torch.Tensor:
        return torch.zeros(4, dtype=torch.float32)

    def get_hash(self) -> str:
        return f"moved={self._moved}"

    def action_to_index(self, action: str) -> int:
        return 0


class _TinyNet(nn.Module):
    def __init__(self, in_dim: int = 4, n_actions: int = 1):
        super().__init__()
        self.policy = nn.Linear(in_dim, n_actions)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.log_softmax(self.policy(x), dim=1), torch.tanh(self.value(x))


class _ConstValueNet(nn.Module):
    """Returns a fixed value for any state — lets a test observe cache contamination."""

    def __init__(self, value: float, n_actions: int = 1):
        super().__init__()
        self._value = float(value)
        self._n = n_actions
        self._unused = nn.Parameter(torch.zeros(1))  # a real parameter so .to(device) works

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        policy = torch.zeros(batch, self._n)
        value = torch.full((batch, 1), self._value)
        return F.log_softmax(policy, dim=1), value


def _evaluator() -> SelfPlayEvaluator:
    config = MCTSConfig()
    config.num_simulations = 1
    mcts = NeuralMCTS(_TinyNet(), config, single_agent=False)
    return SelfPlayEvaluator(
        mcts, _FirstMoverWinsState, EvaluationConfig(num_games=1, mcts_iterations=1, temperature=0.0)
    )


def test_arena_attributes_win_to_the_side_that_actually_won():
    """F1: the first mover (white) wins, so the result must follow model1_starts, not invert it."""
    evaluator = _evaluator()
    net = _TinyNet()

    # model1 starts -> model1 is white -> model1 (the first mover) wins -> result 1.
    result_model1_first, _ = asyncio.run(evaluator.play_game(net, net, model1_starts=True))
    assert result_model1_first == 1

    # model2 starts -> model2 is white -> model2 wins -> result -1 (model2 wins).
    result_model2_first, _ = asyncio.run(evaluator.play_game(net, net, model1_starts=False))
    assert result_model2_first == -1


def test_play_game_clears_cache_and_disables_root_noise_per_swap():
    """1a + 1b: each ply clears the eval cache (network swap) and searches without root noise."""
    captured: dict = {"add_root_noise": None}

    async def fake_search(state, num_simulations, temperature, add_root_noise=True):
        captured["add_root_noise"] = add_root_noise
        return {"move": 1.0}, MagicMock(value=0.0)

    mock_mcts = MagicMock()
    mock_mcts.search = fake_search  # real async fn — must accept add_root_noise
    mock_mcts.network = MagicMock()

    evaluator = SelfPlayEvaluator(
        mock_mcts, _FirstMoverWinsState, EvaluationConfig(num_games=1, mcts_iterations=1, temperature=0.0)
    )
    asyncio.run(evaluator.play_game(MagicMock(), MagicMock(), model1_starts=True))

    assert captured["add_root_noise"] is False  # 1b: deterministic eval, noise off
    mock_mcts.clear_cache.assert_called()  # 1a: cache cleared on the network swap


def test_eval_cache_is_network_isolated_after_clear():
    """1a mechanism: the FEN-keyed cache is network-blind, so a swap without clear_cache leaks the
    previous network's evaluation; clear_cache() (what play_game now calls each ply) restores it."""
    config = MCTSConfig()
    config.num_simulations = 1
    mcts = NeuralMCTS(_ConstValueNet(0.9), config, single_agent=False)
    state = _FirstMoverWinsState()

    _, value_a = asyncio.run(mcts.evaluate_state(state))
    assert value_a == pytest.approx(0.9)

    # Swap the network WITHOUT clearing: the network-blind cache returns net A's stale value.
    mcts.network = _ConstValueNet(-0.9)
    _, value_stale = asyncio.run(mcts.evaluate_state(state))
    assert value_stale == pytest.approx(0.9)  # contamination the arena fix prevents

    # clear_cache() restores correct per-network evaluation.
    mcts.clear_cache()
    _, value_b = asyncio.run(mcts.evaluate_state(state))
    assert value_b == pytest.approx(-0.9)
