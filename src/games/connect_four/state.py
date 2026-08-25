"""Connect Four GameState implementation for Neural MCTS."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
import torch

from src.framework.mcts.neural_mcts import GameState
from src.games.connect_four.config import ConnectFourConfig

CONFIG = ConnectFourConfig()


@dataclass
class ConnectFourState(GameState):
    """Connect Four board state.

    Attributes:
        board: 6x7 numpy array where 0 is empty, 1 is Player 1 (Red), -1 is Player 2 (Yellow).
        current_player: Active player (1 or -1).
        last_move: Column index (0..6) of the last move, or None for initial state.
    """

    board: np.ndarray = field(default_factory=lambda: np.zeros((CONFIG.board_rows, CONFIG.board_cols), dtype=np.int8))
    current_player: int = 1
    last_move: int | None = None

    @classmethod
    def create_initial_state(cls) -> ConnectFourState:
        """Factory method creating a fresh Connect Four initial board state."""
        return cls(
            board=np.zeros((CONFIG.board_rows, CONFIG.board_cols), dtype=np.int8),
            current_player=1,
            last_move=None,
        )

    def get_legal_actions(self) -> list[int]:
        """Return list of legal column indices (0..6) that are not full."""
        if self.is_terminal():
            return []
        return [col for col in range(CONFIG.board_cols) if self.board[0, col] == 0]

    def apply_action(self, action: Any) -> ConnectFourState:
        """Apply drop action into column `action` and return new ConnectFourState."""
        col = int(action)
        if col not in self.get_legal_actions():
            raise ValueError(f"Illegal action {col} for state")

        new_board = self.board.copy()
        # Find lowest empty row in column
        for r in reversed(range(CONFIG.board_rows)):
            if new_board[r, col] == 0:
                new_board[r, col] = self.current_player
                break

        return ConnectFourState(
            board=new_board,
            current_player=-self.current_player,
            last_move=col,
        )

    def get_hash(self) -> str:
        """Return deterministic hash representation of board and current_player."""
        state_bytes = self.board.tobytes() + bytes([self.current_player & 0xFF])
        return hashlib.sha256(state_bytes).hexdigest()

    def action_to_index(self, action: Any) -> int:
        """Map action to policy index."""
        return int(action)

    def _check_winner(self) -> int | None:
        """Internal helper to check if player 1 or -1 has N in a row. Returns 1, -1, or None."""
        board = self.board
        rows, cols = CONFIG.board_rows, CONFIG.board_cols
        k = CONFIG.in_a_row

        # Horizontal check
        for r in range(rows):
            for c in range(cols - k + 1):
                val = board[r, c]
                if val != 0 and all(board[r, c + i] == val for i in range(1, k)):
                    return int(val)

        # Vertical check
        for r in range(rows - k + 1):
            for c in range(cols):
                val = board[r, c]
                if val != 0 and all(board[r + i, c] == val for i in range(1, k)):
                    return int(val)

        # Positive diagonal check (bottom-left to top-right)
        for r in range(k - 1, rows):
            for c in range(cols - k + 1):
                val = board[r, c]
                if val != 0 and all(board[r - i, c + i] == val for i in range(1, k)):
                    return int(val)

        # Negative diagonal check (top-left to bottom-right)
        for r in range(rows - k + 1):
            for c in range(cols - k + 1):
                val = board[r, c]
                if val != 0 and all(board[r + i, c + i] == val for i in range(1, k)):
                    return int(val)

        return None

    def is_terminal(self) -> bool:
        """Check if game is over (win or draw)."""
        winner = self._check_winner()
        if winner is not None:
            return True
        # Draw if board is completely full
        return not np.any(self.board == 0)

    def get_reward(self, player: int = 1) -> float:
        """Get reward from perspective of `player` (1 or -1).

        Returns 1.0 for win, -1.0 for loss, 0.0 for draw or in-progress game.
        """
        winner = self._check_winner()
        if winner is not None:
            return 1.0 if winner == player else -1.0
        return 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert Connect Four board state to (3, 6, 7) FloatTensor.

        Channel 0: Current player's pieces (1.0 where board == current_player)
        Channel 1: Opponent's pieces (1.0 where board == -current_player)
        Channel 2: Turn indicator (1.0 if current_player == 1 else 0.0)
        """
        c0 = (self.board == self.current_player).astype(np.float32)
        c1 = (self.board == -self.current_player).astype(np.float32)
        c2 = np.full((CONFIG.board_rows, CONFIG.board_cols), 1.0 if self.current_player == 1 else 0.0, dtype=np.float32)
        tensor_data = np.stack([c0, c1, c2], axis=0)
        return torch.from_numpy(tensor_data)
