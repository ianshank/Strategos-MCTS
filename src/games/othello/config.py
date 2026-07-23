"""Othello / Reversi domain configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OthelloConfig:
    """Configuration constants for Othello domain."""

    board_size: int = 8
    board_cells: int = 64
    pass_action: int = 64
    action_space_size: int = 65  # 64 positions + 1 pass action
    input_channels: int = 3
    num_players: int = 2
