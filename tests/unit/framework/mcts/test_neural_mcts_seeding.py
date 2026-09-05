"""NeuralMCTS injected-RNG / Dirichlet reproducibility (hygiene_determinism AC-3).

Covers hygiene_determinism AC-3
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.framework.mcts.neural_mcts import GameState, NeuralMCTS
from src.training.system_config import MCTSConfig
from src.utils.seeding import new_rng


class _TinyState(GameState):
    """Minimal GameState with a fixed action space for reproducibility tests."""

    def __init__(self, step: int = 0) -> None:
        self.step = step

    def get_legal_actions(self) -> list[int]:
        return [0, 1, 2]

    def apply_action(self, action: int) -> GameState:  # noqa: ARG002
        return _TinyState(self.step + 1)

    def is_terminal(self) -> bool:
        return self.step >= 2

    def get_reward(self, player: int = 1) -> float:  # noqa: ARG002
        return 0.0

    def to_tensor(self) -> torch.Tensor:
        return torch.zeros(4)

    def action_to_index(self, action: int) -> int:
        return int(action)

    def get_hash(self) -> str:
        return f"tiny-{self.step}"


class _UniformNet(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
        # 3-logit uniform policy + zero value; deterministic network.
        batch = x.shape[0]
        policy = torch.zeros(batch, 3)
        value = torch.zeros(batch, 1)
        return policy, value


def _config() -> MCTSConfig:
    return MCTSConfig(
        num_simulations=8,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        c_puct=1.25,
        virtual_loss=0.0,
    )


@pytest.mark.neural
def test_dirichlet_noise_reproducible_with_same_rng_seed() -> None:
    """Same seed → identical Dirichlet draws from NeuralMCTS.add_dirichlet_noise."""
    policy = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    mcts_a = NeuralMCTS(_UniformNet(), _config(), seed=123)
    mcts_b = NeuralMCTS(_UniformNet(), _config(), seed=123)
    noise_a = mcts_a.add_dirichlet_noise(policy.copy())
    noise_b = mcts_b.add_dirichlet_noise(policy.copy())
    np.testing.assert_array_equal(noise_a, noise_b)


@pytest.mark.neural
def test_dirichlet_noise_diverges_with_different_seeds() -> None:
    policy = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    mcts_a = NeuralMCTS(_UniformNet(), _config(), seed=1)
    mcts_b = NeuralMCTS(_UniformNet(), _config(), seed=2)
    assert not np.array_equal(
        mcts_a.add_dirichlet_noise(policy.copy()),
        mcts_b.add_dirichlet_noise(policy.copy()),
    )


@pytest.mark.neural
def test_injected_rng_is_used_for_noise_and_choice() -> None:
    rng = new_rng(77)
    # seed kwarg must be ignored when rng is injected — self.seed stays None
    mcts = NeuralMCTS(_UniformNet(), _config(), rng=rng, seed=77)
    assert mcts.rng is rng
    assert mcts.seed is None
    policy = np.ones(3) / 3.0
    noised = mcts.add_dirichlet_noise(policy)
    assert noised.shape == (3,)
    assert abs(noised.sum() - 1.0) < 1e-6
    action = mcts.select_action({0: 0.1, 1: 0.2, 2: 0.7}, temperature=1.0)
    assert action in (0, 1, 2)


@pytest.mark.neural
def test_owned_rng_stores_resolved_seed() -> None:
    mcts = NeuralMCTS(_UniformNet(), _config(), seed=123)
    assert mcts.seed == 123
    assert mcts.rng is not None


@pytest.mark.neural
def test_injected_rng_does_not_resolve_settings_seed() -> None:
    """Injected rng must not pull a misleading seed from Settings."""
    from unittest.mock import MagicMock, patch

    mock_settings = MagicMock()
    mock_settings.SEED = 999
    rng = new_rng(1)
    with patch("src.config.settings.get_settings", return_value=mock_settings):
        mcts = NeuralMCTS(_UniformNet(), _config(), rng=rng)
    assert mcts.seed is None
    assert mcts.rng is rng


@pytest.mark.neural
def test_search_visit_counts_reproducible_with_seeded_rng() -> None:
    """Same-process double-run with identical seeds yields identical visit counts.

    Fresh-process half of AC-3 is covered by the e2e self-play golden path; this
    unit test locks the in-process injected-Generator contract.
    """

    async def _run(seed: int) -> dict[int, int]:
        mcts = NeuralMCTS(_UniformNet(), _config(), seed=seed, single_agent=True)
        _probs, root = await mcts.search(_TinyState(), num_simulations=8, add_root_noise=True)
        return {action: child.visit_count for action, child in root.children.items()}

    visits_a = asyncio.run(_run(42))
    visits_b = asyncio.run(_run(42))
    assert visits_a == visits_b
    assert sum(visits_a.values()) >= 8
