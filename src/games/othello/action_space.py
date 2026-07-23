"""Othello action space mapping."""

from __future__ import annotations

from src.games.othello.config import OthelloConfig

CONFIG = OthelloConfig()


class OthelloActionSpace:
    """Action space for Othello mapping (row, col) to action ID (0..64)."""

    def __init__(self) -> None:
        self.action_size = CONFIG.action_space_size

    def coord_to_action(self, r: int, c: int) -> int:
        """Map (row, col) to flat action index."""
        if not (0 <= r < CONFIG.board_size and 0 <= c < CONFIG.board_size):
            raise ValueError(f"Coordinates ({r}, {c}) outside valid board 0..7")
        return r * CONFIG.board_size + c

    def action_to_coord(self, action_id: int) -> tuple[int, int] | None:
        """Map flat action index back to (row, col), or None if pass."""
        if action_id == CONFIG.pass_action:
            return None
        if not (0 <= action_id < CONFIG.board_cells):
            raise ValueError(f"Action ID {action_id} outside valid range 0..64")
        return action_id // CONFIG.board_size, action_id % CONFIG.board_size
