"""Bounded CPU smoke for the chess (win-rate) gate path — local/manual (@slow, no CI home).

Validates on the REAL chess domain (not the toy TTT of the unit tests): the driver's resnet
architecture builds and forwards on a (22, 8, 8) chess tensor, and the arena attributes a real
checkmate to the correct model (F1) using ``ChessGameState.get_reward``'s white/black convention.

Bounded to dodge the uncapped-chess-game / 300s-timeout hazard: the forward is a single position,
and the arena starts from a terminal checkmate FEN (so no search loop runs). Requires the chess
extra; skips the whole module otherwise.
"""

from __future__ import annotations

import asyncio

import pytest

from src.games.chess.registration import chess_available

pytestmark = [pytest.mark.integration, pytest.mark.slow]

if not chess_available():  # pragma: no cover - depends on the optional chess extra
    pytest.skip("chess extra (python-chess) not installed", allow_module_level=True)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.benchmark.policy_lift import build_network, chess_default_architecture  # noqa: E402
from src.framework.domain_registry import DomainRegistry  # noqa: E402
from src.framework.mcts.neural_mcts import NeuralMCTS  # noqa: E402
from src.games.chess.state import ChessGameState  # noqa: E402
from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator  # noqa: E402
from src.training.system_config import MCTSConfig  # noqa: E402

# Fool's mate: White (to move) is checkmated by the black queen on h4 -> BLACK won.
_FOOLS_MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


class _StubNet(nn.Module):
    """Never actually called from a terminal position; present only to satisfy NeuralMCTS."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        batch = x.shape[0]
        return torch.zeros(batch, 4672), torch.zeros(batch, 1)


def test_chess_resnet_builds_and_forwards_on_real_state():
    spec = DomainRegistry.get("chess")
    net = build_network(chess_default_architecture(spec), spec, "cpu")
    net.eval()
    state = spec.initial_state_fn()  # standard opening
    with torch.no_grad():
        policy, value = net(state.to_tensor().unsqueeze(0))
    assert policy.shape == (1, spec.action_space_size)  # (1, 4672)
    assert value.shape == (1, 1)


def test_chess_arena_attributes_real_checkmate():
    """F1 on real chess: fool's-mate (black won, white to move) — attribution must follow model1."""
    terminal = ChessGameState.from_fen(_FOOLS_MATE_FEN)
    assert terminal.is_terminal()

    config = MCTSConfig()
    config.num_simulations = 1
    mcts = NeuralMCTS(_StubNet(), config, single_agent=False)
    evaluator = SelfPlayEvaluator(
        mcts,
        lambda: ChessGameState.from_fen(_FOOLS_MATE_FEN),
        EvaluationConfig(num_games=1, mcts_iterations=1, temperature=0.0),
    )

    # White (to move) is mated -> black won. model1 starts (white) -> model1 lost -> result -1.
    result_model1_white, _ = asyncio.run(evaluator.play_game(_StubNet(), _StubNet(), model1_starts=True))
    assert result_model1_white == -1

    # model1 does not start (black) -> model1 (black) delivered mate -> result 1.
    result_model1_black, _ = asyncio.run(evaluator.play_game(_StubNet(), _StubNet(), model1_starts=False))
    assert result_model1_black == 1
