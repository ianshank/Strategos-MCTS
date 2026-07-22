"""Integration tests for new game domains (Connect Four & Othello) with self-play convergence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.self_play_convergence import run as run_convergence


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connect_four_self_play_convergence(tmp_path: Path) -> None:
    class Args:
        domain = "connect_four"
        iterations = 1
        checkpoint_dir = tmp_path
        seed = 42
        device = "cpu"
        resume = False
        num_simulations = 4
        games_per_iteration = 2
        profile = None
        mixed_precision = False
        compile = False

    exit_code = await run_convergence(Args())
    assert exit_code == 0
    assert (tmp_path / "ckpt_iter_1.pt").exists()
    assert (tmp_path / "ckpt_iter_1.pt.meta.json").exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_othello_self_play_convergence(tmp_path: Path) -> None:
    class Args:
        domain = "othello"
        iterations = 1
        checkpoint_dir = tmp_path
        seed = 42
        device = "cpu"
        resume = False
        num_simulations = 4
        games_per_iteration = 2
        profile = None
        mixed_precision = False
        compile = False

    exit_code = await run_convergence(Args())
    assert exit_code == 0
    assert (tmp_path / "ckpt_iter_1.pt").exists()
    assert (tmp_path / "ckpt_iter_1.pt.meta.json").exists()
