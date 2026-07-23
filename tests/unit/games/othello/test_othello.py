"""Unit tests for Othello game state and domain rules."""

from __future__ import annotations

import pytest
import torch

from src.framework.domain_registry import DomainRegistry
from src.games.othello import OthelloState, register_othello_domain


@pytest.mark.unit
def test_othello_initial_state() -> None:
    state = OthelloState.create_initial_state()
    assert state.current_player == 1
    assert state.consecutive_passes == 0
    # Standard initial board setup
    assert state.board[3, 3] == -1
    assert state.board[3, 4] == 1
    assert state.board[4, 3] == 1
    assert state.board[4, 4] == -1
    assert not state.is_terminal()


@pytest.mark.unit
def test_othello_legal_moves_and_flips() -> None:
    state = OthelloState.create_initial_state()
    legal_moves = state.get_legal_actions()
    # In initial Othello position, Black (player 1) has 4 legal moves: (2,3), (3,2), (4,5), (5,4)
    # Corresponding to flat action IDs: 19, 26, 37, 44
    assert len(legal_moves) == 4
    assert set(legal_moves) == {19, 26, 37, 44}


@pytest.mark.unit
def test_othello_apply_action_flips_pieces() -> None:
    state = OthelloState.create_initial_state()
    # Play move at (2,3) -> action ID 19
    next_state = state.apply_action(19)
    assert next_state.current_player == -1
    assert next_state.board[2, 3] == 1
    assert next_state.board[3, 3] == 1  # Was -1, now flipped to 1


@pytest.mark.unit
def test_othello_consecutive_passes_terminal() -> None:
    board = OthelloState._make_initial_board()
    state1 = OthelloState(board=board, current_player=1, consecutive_passes=1)
    assert not state1.is_terminal()
    state2 = OthelloState(board=board, current_player=-1, consecutive_passes=2)
    assert state2.is_terminal()


@pytest.mark.unit
def test_othello_to_tensor() -> None:
    state = OthelloState.create_initial_state()
    tensor = state.to_tensor()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == torch.float32


@pytest.mark.unit
def test_othello_domain_registration() -> None:
    register_othello_domain()
    spec = DomainRegistry.get("othello")
    assert spec.name == "othello"
    assert spec.metric == "win_rate"
    assert spec.single_agent is False
    assert spec.action_space_size == 65
