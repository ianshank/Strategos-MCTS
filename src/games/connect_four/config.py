"""Connect Four domain configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectFourConfig:
    """Configuration constants for Connect Four domain."""

    board_rows: int = 6
    board_cols: int = 7
    action_space_size: int = 7
    in_a_row: int = 4
    input_channels: int = 3
    num_players: int = 2
