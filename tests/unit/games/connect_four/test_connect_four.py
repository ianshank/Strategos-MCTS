"""Unit tests for Connect Four game state and domain rules."""

from __future__ import annotations

import pytest
import torch

from src.framework.domain_registry import DomainRegistry
from src.games.connect_four import ConnectFourState, register_connect_four_domain


@pytest.mark.unit
def test_connect_four_initial_state() -> None:
    state = ConnectFourState.create_initial_state()
    assert state.current_player == 1
    assert state.last_move is None
    assert len(state.get_legal_actions()) == 7
    assert not state.is_terminal()
    assert state.get_reward(1) == 0.0


@pytest.mark.unit
def test_connect_four_apply_action() -> None:
    state = ConnectFourState.create_initial_state()
    next_state = state.apply_action(3)
    assert next_state.current_player == -1
    assert next_state.last_move == 3
    assert next_state.board[5, 3] == 1  # Placed at bottom row (index 5)
    assert len(next_state.get_legal_actions()) == 7


@pytest.mark.unit
def test_connect_four_vertical_win() -> None:
    state = ConnectFourState.create_initial_state()
    # Player 1 plays col 0, Player 2 plays col 1 (4 times each)
    moves = [0, 1, 0, 1, 0, 1, 0]
    for move in moves:
        state = state.apply_action(move)

    assert state.is_terminal()
    assert state.get_reward(1) == 1.0
    assert state.get_reward(-1) == -1.0


@pytest.mark.unit
def test_connect_four_horizontal_win() -> None:
    state = ConnectFourState.create_initial_state()
    # Player 1 plays cols 0, 1, 2, 3; Player 2 plays col 6
    moves = [0, 6, 1, 6, 2, 6, 3]
    for move in moves:
        state = state.apply_action(move)

    assert state.is_terminal()
    assert state.get_reward(1) == 1.0


@pytest.mark.unit
def test_connect_four_tensor_encoding() -> None:
    state = ConnectFourState.create_initial_state()
    state = state.apply_action(2)
    tensor = state.to_tensor()

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 6, 7)
    assert tensor.dtype == torch.float32


@pytest.mark.unit
def test_connect_four_domain_registration() -> None:
    register_connect_four_domain()
    spec = DomainRegistry.get("connect_four")
    assert spec.name == "connect_four"
    assert spec.metric == "win_rate"
    assert spec.single_agent is False
    assert spec.action_space_size == 7
