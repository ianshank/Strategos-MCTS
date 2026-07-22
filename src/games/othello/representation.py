"""Othello board tensor representation."""

from __future__ import annotations

import torch

from src.games.othello.config import OthelloConfig
from src.games.othello.state import OthelloState

CONFIG = OthelloConfig()


class OthelloRepresentation:
    """Encoder for OthelloState to network input tensor."""

    @staticmethod
    def encode(state: OthelloState) -> torch.Tensor:
        """Encode OthelloState to a (3, 8, 8) FloatTensor."""
        return state.to_tensor()

    @staticmethod
    def get_input_shape() -> tuple[int, int, int]:
        """Return input tensor shape (channels, rows, cols)."""
        return (CONFIG.input_channels, CONFIG.board_size, CONFIG.board_size)
