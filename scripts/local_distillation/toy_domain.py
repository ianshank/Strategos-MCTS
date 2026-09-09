"""Tiny two-player GameState and counting net for distillation tests and CLI smoke."""

from __future__ import annotations

from typing import SupportsIndex

import torch
from torch import nn
import torch.nn.functional as F

from src.framework.mcts.neural_mcts import GameState

ACTIONS = (0, 1)
ACTION_SPACE = 2


class TwoPlyState(GameState):
    """Two-ply game: player 1 always wins at termination. Supports midgame P2-to-move roots."""

    def __init__(self, ply: int = 0, current_player: int = 1, max_ply: int = 2) -> None:
        self.ply = ply
        self._current_player = current_player
        self.max_ply = max_ply

    @property
    def current_player(self) -> int:
        return self._current_player

    def get_legal_actions(self) -> list[int]:
        return [] if self.is_terminal() else list(ACTIONS)

    def apply_action(self, action: object) -> TwoPlyState:
        del action
        return TwoPlyState(self.ply + 1, -self._current_player, self.max_ply)

    def is_terminal(self) -> bool:
        return self.ply >= self.max_ply

    def get_reward(self, player: int = 1) -> float:
        if not self.is_terminal():
            return 0.0
        return 1.0 if player == 1 else -1.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.ply / self.max_ply, float(self._current_player)], dtype=torch.float32)

    def get_hash(self) -> str:
        return f"{self.ply}:{self._current_player}"

    def action_to_index(self, action: SupportsIndex) -> int:
        return int(action)


class CountingNet(nn.Module):
    """Records train/eval mode and forward count. Returns log-probs + tanh value."""

    def __init__(self, n_actions: int = ACTION_SPACE, in_dim: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_actions)
        self.val = nn.Linear(in_dim, 1)
        self.forwards = 0
        self.modes: list[bool] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forwards += 1
        self.modes.append(bool(self.training))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        log_probs = F.log_softmax(self.fc(x), dim=1)
        value = torch.tanh(self.val(x))
        return log_probs, value


class BiasedNet(nn.Module):
    """Strongly prefers action 0 so visit counts are skewed (π_train vs π_play)."""

    def __init__(
        self,
        n_actions: int = ACTION_SPACE,
        in_dim: int = 2,
        *,
        bias: tuple[float, float] = (8.0, -8.0),
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_actions)
        self.val = nn.Linear(in_dim, 1)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.copy_(torch.tensor(list(bias)))
            self.val.weight.zero_()
            self.val.bias.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        log_probs = F.log_softmax(self.fc(x), dim=1)
        value = torch.tanh(self.val(x))
        return log_probs, value
