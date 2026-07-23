"""Negamax value-sign convention tests for NeuralMCTS.

The invariant under test: every node's stored value is from the perspective of the side to
move AT THAT NODE — terminal leaves via ``get_reward(player=state.current_player)``, network
leaves via the side-to-move value head — with ``select_child`` negating the child Q for
two-player search and the backup negating per ply. Before this convention was reconciled,
terminal leaves used a fixed player-1 perspective while selection never negated, which was
coherent for player-1-to-move roots but made search AVOID winning moves for player -1
(~half of all positions in an adversarial game).

Single-agent search stores absolute values: no negation anywhere (guarded by the engine's
``single_agent`` flag), asserted below so the guard cannot silently regress.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from torch import nn

from src.framework.mcts.neural_mcts import GameState, NeuralMCTS
from src.games.chess.registration import chess_available
from src.training.system_config import MCTSConfig

pytestmark = [pytest.mark.unit]


class _UniformStubNet(nn.Module):
    """Uniform policy, zero value — search signal then comes only from terminal rewards."""

    def __init__(self, action_size: int):
        super().__init__()
        self._action_size = action_size
        self._unused = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        return torch.zeros(batch, self._action_size), torch.zeros(batch, 1)


class _TwoPlayerWinNow(GameState):
    """Alternating two-player game: the side to move may win immediately or play on.

    ``"win"`` ends the game with the mover as winner; ``"meh"`` passes the turn (draw after
    4 plies). ``get_reward(player)`` is +1/-1 from ``player``'s perspective, mirroring chess.
    """

    _ACTIONS = ("win", "meh")

    def __init__(self, to_move: int = 1, winner: int | None = None, plies: int = 0):
        self.to_move = to_move
        self.winner = winner
        self.plies = plies

    @property
    def current_player(self) -> int:
        return self.to_move

    def get_legal_actions(self) -> list[str]:
        return [] if self.is_terminal() else list(self._ACTIONS)

    def apply_action(self, action: str) -> _TwoPlayerWinNow:
        winner = self.to_move if action == "win" else None
        return _TwoPlayerWinNow(to_move=-self.to_move, winner=winner, plies=self.plies + 1)

    def is_terminal(self) -> bool:
        return self.winner is not None or self.plies >= 4

    def get_reward(self, player: int = 1) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([float(self.to_move), self.plies / 4.0], dtype=torch.float32)

    def get_hash(self) -> str:
        return f"{self.to_move}:{self.winner}:{self.plies}"

    def action_to_index(self, action: str) -> int:
        return self._ACTIONS.index(action)


class _SingleAgentPick(GameState):
    """One-step single-agent problem: 'good' scores 1.0, 'bad' scores 0.2."""

    _ACTIONS = ("good", "bad")

    def __init__(self, picked: str | None = None):
        self.picked = picked

    def get_legal_actions(self) -> list[str]:
        return [] if self.picked else list(self._ACTIONS)

    def apply_action(self, action: str) -> _SingleAgentPick:
        return _SingleAgentPick(picked=action)

    def is_terminal(self) -> bool:
        return self.picked is not None

    def get_reward(self, player: int = 1) -> float:
        return {"good": 1.0, "bad": 0.2}.get(self.picked or "", 0.0)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([1.0 if self.picked else 0.0], dtype=torch.float32)

    def get_hash(self) -> str:
        return f"picked={self.picked}"

    def action_to_index(self, action: str) -> int:
        return self._ACTIONS.index(action)


def _config(num_simulations: int) -> MCTSConfig:
    config = MCTSConfig()
    config.num_simulations = num_simulations
    return config


def _best_action(action_probs: dict) -> object:
    return max(action_probs, key=lambda a: action_probs[a])


@pytest.mark.parametrize("to_move", [1, -1])
def test_two_player_search_finds_the_winning_move_for_both_sides(to_move):
    """Both parities must prefer the immediate win — player -1 was the broken half."""
    mcts = NeuralMCTS(_UniformStubNet(action_size=2), _config(16), single_agent=False)
    root_state = _TwoPlayerWinNow(to_move=to_move)

    action_probs, root = asyncio.run(mcts.search(root_state, temperature=0.0, add_root_noise=False))

    assert _best_action(action_probs) == "win"
    # The side to move can force a win, so the root's own-perspective value is positive.
    assert root.value > 0.0


def test_single_agent_search_keeps_absolute_values():
    """The single_agent guard: no negation — the higher-reward action must win, not lose."""
    mcts = NeuralMCTS(_UniformStubNet(action_size=2), _config(16), single_agent=True)

    action_probs, root = asyncio.run(mcts.search(_SingleAgentPick(), temperature=0.0, add_root_noise=False))

    assert _best_action(action_probs) == "good"
    assert root.value > 0.0


@pytest.mark.skipif(not chess_available(), reason="chess extra (python-chess) not installed")
@pytest.mark.parametrize(
    ("fen", "mating_move"),
    [
        # Fool's mate: black to move, Qd8-h4#. This is the parity the old convention broke.
        ("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2", "d8h4"),
        # Mirrored fool's mate: white to move, Qd1-h5#.
        ("rnbqkbnr/ppppp2p/5p2/6p1/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq g6 0 3", "d1h5"),
    ],
)
def test_chess_search_finds_mate_in_one_for_both_colors(fen, mating_move):
    """Real-chess validation of the side-to-move terminal convention, both colors."""
    from src.games.chess.config import ChessActionSpaceConfig
    from src.games.chess.state import ChessGameState

    state = ChessGameState.from_fen(fen)
    assert mating_move in state.get_legal_actions()
    mated = state.apply_action(mating_move)
    assert mated.is_terminal()
    # The mated side (to move in the terminal state) sees -1: side-to-move perspective.
    assert mated.get_reward(player=mated.current_player) == -1.0

    mcts = NeuralMCTS(
        _UniformStubNet(action_size=ChessActionSpaceConfig().total_actions),
        _config(96),
        single_agent=False,
    )
    action_probs, root = asyncio.run(mcts.search(state, temperature=0.0, add_root_noise=False))

    assert _best_action(action_probs) == mating_move
    assert root.value > 0.0
