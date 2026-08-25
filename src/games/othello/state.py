"""Othello / Reversi GameState implementation for Neural MCTS."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
import torch

from src.framework.mcts.neural_mcts import GameState
from src.games.othello.config import OthelloConfig

CONFIG = OthelloConfig()

# 8 directions for piece flips
DIRECTIONS: list[tuple[int, int]] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


@dataclass
class OthelloState(GameState):
    """Othello board state.

    Attributes:
        board: 8x8 numpy array where 0 is empty, 1 is Black, -1 is White.
        current_player: Active player (1 or -1).
        consecutive_passes: Number of consecutive pass actions taken (terminal when == 2).
    """

    board: np.ndarray = field(default_factory=lambda: OthelloState._make_initial_board())
    current_player: int = 1
    consecutive_passes: int = 0

    @classmethod
    def _make_initial_board(cls) -> np.ndarray:
        board = np.zeros((CONFIG.board_size, CONFIG.board_size), dtype=np.int8)
        # Standard Othello starting configuration in center 2x2
        mid = CONFIG.board_size // 2
        board[mid - 1, mid - 1] = -1
        board[mid - 1, mid] = 1
        board[mid, mid - 1] = 1
        board[mid, mid] = -1
        return board

    @classmethod
    def create_initial_state(cls) -> OthelloState:
        """Factory method creating a fresh Othello initial board state."""
        return cls(
            board=cls._make_initial_board(),
            current_player=1,
            consecutive_passes=0,
        )

    def _get_flips_in_direction(self, r: int, c: int, dr: int, dc: int) -> list[tuple[int, int]]:
        flips: list[tuple[int, int]] = []
        curr_r, curr_c = r + dr, c + dc
        while 0 <= curr_r < CONFIG.board_size and 0 <= curr_c < CONFIG.board_size:
            val = self.board[curr_r, curr_c]
            if val == -self.current_player:
                flips.append((curr_r, curr_c))
            elif val == self.current_player:
                return flips
            else:  # Empty cell
                return []
            curr_r += dr
            curr_c += dc
        return []

    def get_flips_for_move(self, r: int, c: int) -> list[tuple[int, int]]:
        """Return list of opponent piece coordinates flipped by placing piece at (r, c)."""
        if self.board[r, c] != 0:
            return []

        all_flips: list[tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            flips = self._get_flips_in_direction(r, c, dr, dc)
            all_flips.extend(flips)
        return all_flips

    def get_legal_actions(self) -> list[int]:
        """Return list of legal action IDs (0..63 for board moves, or 64 for pass)."""
        if self.is_terminal():
            return []

        legal_moves: list[int] = []
        for r in range(CONFIG.board_size):
            for c in range(CONFIG.board_size):
                if self.get_flips_for_move(r, c):
                    legal_moves.append(r * CONFIG.board_size + c)

        if not legal_moves:
            return [CONFIG.pass_action]

        return legal_moves

    def apply_action(self, action: Any) -> OthelloState:
        """Apply action (0..64) and return new OthelloState."""
        act_id = int(action)
        legal_actions = self.get_legal_actions()
        if act_id not in legal_actions:
            raise ValueError(f"Illegal action {act_id} for state")

        if act_id == CONFIG.pass_action:
            return OthelloState(
                board=self.board.copy(),
                current_player=-self.current_player,
                consecutive_passes=self.consecutive_passes + 1,
            )

        r, c = act_id // CONFIG.board_size, act_id % CONFIG.board_size
        flips = self.get_flips_for_move(r, c)
        new_board = self.board.copy()
        new_board[r, c] = self.current_player
        for fr, fc in flips:
            new_board[fr, fc] = self.current_player

        return OthelloState(
            board=new_board,
            current_player=-self.current_player,
            consecutive_passes=0,
        )

    def get_hash(self) -> str:
        """Return deterministic hash representation of board, current_player, and passes."""
        state_bytes = self.board.tobytes() + bytes([self.current_player & 0xFF, self.consecutive_passes & 0xFF])
        return hashlib.sha256(state_bytes).hexdigest()

    def action_to_index(self, action: Any) -> int:
        """Map action to policy index."""
        return int(action)

    def is_terminal(self) -> bool:
        """Game ends if 2 consecutive passes occur or board is full."""
        if self.consecutive_passes >= 2:
            return True
        if not np.any(self.board == 0):
            return True

        # Check if both players have zero legal board moves
        p1_moves = False
        p2_moves = False
        for r in range(CONFIG.board_size):
            for c in range(CONFIG.board_size):
                if self.board[r, c] == 0:
                    if not p1_moves and self._flips_exist_for_player(r, c, 1):
                        p1_moves = True
                    if not p2_moves and self._flips_exist_for_player(r, c, -1):
                        p2_moves = True
                if p1_moves and p2_moves:
                    break

        return not (p1_moves or p2_moves)

    def _flips_exist_for_player(self, r: int, c: int, player: int) -> bool:
        for dr, dc in DIRECTIONS:
            curr_r, curr_c = r + dr, c + dc
            found_opp = False
            while 0 <= curr_r < CONFIG.board_size and 0 <= curr_c < CONFIG.board_size:
                val = self.board[curr_r, curr_c]
                if val == -player:
                    found_opp = True
                elif val == player:
                    if found_opp:
                        return True
                    break
                else:
                    break
                curr_r += dr
                curr_c += dc
        return False

    def get_reward(self, player: int = 1) -> float:
        """Get reward from perspective of `player` (1 or -1).

        Returns 1.0 for win, -1.0 for loss, 0.0 for draw or in-progress game.
        """
        p_count = float(np.sum(self.board == player))
        opp_count = float(np.sum(self.board == -player))

        if not self.is_terminal():
            return 0.0

        if p_count > opp_count:
            return 1.0
        elif p_count < opp_count:
            return -1.0
        return 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert Othello board state to (3, 8, 8) FloatTensor."""
        c0 = (self.board == self.current_player).astype(np.float32)
        c1 = (self.board == -self.current_player).astype(np.float32)
        c2 = np.full(
            (CONFIG.board_size, CONFIG.board_size),
            1.0 if self.current_player == 1 else 0.0,
            dtype=np.float32,
        )
        tensor_data = np.stack([c0, c1, c2], axis=0)
        return torch.from_numpy(tensor_data)
