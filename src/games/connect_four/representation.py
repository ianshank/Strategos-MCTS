"""Connect Four board tensor representation."""

from __future__ import annotations

import torch

from src.games.connect_four.config import ConnectFourConfig
from src.games.connect_four.state import ConnectFourState

CONFIG = ConnectFourConfig()


class ConnectFourRepresentation:
    """Encoder for Connect Four state to network input tensor."""

    @staticmethod
    def encode(state: ConnectFourState) -> torch.Tensor:
        """Encode ConnectFourState to a (3, 6, 7) torch.FloatTensor."""
        return state.to_tensor()

    @staticmethod
    def get_input_shape() -> tuple[int, int, int]:
        """Return input tensor shape (channels, rows, cols)."""
        return (CONFIG.input_channels, CONFIG.board_rows, CONFIG.board_cols)
