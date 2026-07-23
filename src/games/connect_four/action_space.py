"""Connect Four action space utilities."""

from __future__ import annotations

from src.games.connect_four.config import ConnectFourConfig

CONFIG = ConnectFourConfig()


class ConnectFourActionSpace:
    """Action space for Connect Four mapping column index (0..6) to action ID."""

    def __init__(self) -> None:
        self.action_size = CONFIG.action_space_size

    def action_to_index(self, action: int) -> int:
        """Map column index to action index."""
        if not (0 <= action < self.action_size):
            raise ValueError(f"Action {action} outside valid column range 0..{self.action_size - 1}")
        return action

    def index_to_action(self, index: int) -> int:
        """Map action index back to column index."""
        if not (0 <= index < self.action_size):
            raise ValueError(f"Index {index} outside valid column range 0..{self.action_size - 1}")
        return index
