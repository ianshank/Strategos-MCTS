"""
Chess Board Representation Module.

Implements the tensor representation of chess positions for neural network input.
Uses AlphaZero-style 19+ plane encoding with piece positions, game state, and history.

Plane layout (22 planes total):
- Planes 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- Planes 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- Plane 12: Side to move (1 if white, 0 if black)
- Plane 13: White kingside castling rights
- Plane 14: White queenside castling rights
- Plane 15: Black kingside castling rights
- Plane 16: Black queenside castling rights
- Plane 17: En passant square
- Plane 18: Halfmove clock (normalized)
- Plane 19: Fullmove number (normalized)
- Plane 20: 1-fold repetition
- Plane 21: 2-fold repetition
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import chess

from src.games.chess.config import ChessBoardConfig

# Piece type to plane index mapping
PIECE_TO_PLANE: dict[int, int] = {
    1: 0,  # Pawn
    2: 1,  # Knight
    3: 2,  # Bishop
    4: 3,  # Rook
    5: 4,  # Queen
    6: 5,  # King
}


class ChessBoardRepresentation:
    """Converts chess positions to tensor representation for neural networks.

    Implements AlphaZero-style board encoding with configurable planes.
    """

    def __init__(self, config: ChessBoardConfig | None = None) -> None:
        """Initialize the board representation encoder.

        Args:
            config: Board configuration. Uses defaults if None.
        """
        self.config = config or ChessBoardConfig()
        self._history_boards: list[str] = []

    @property
    def num_planes(self) -> int:
        """Total number of input planes."""
        return self.config.total_planes

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Expected input tensor shape (C, H, W)."""
        return self.config.input_shape

    def encode(
        self,
        board: chess.Board,
        from_perspective: int = 1,
        history: list[chess.Board] | None = None,
    ) -> torch.Tensor:
        """Encode a chess position to tensor representation.

        Args:
            board: Chess board position to encode
            from_perspective: 1 for white's perspective, -1 for black's
            history: Optional list of previous board positions for history planes

        Returns:
            Tensor of shape (num_planes, 8, 8)
        """
        import chess

        tensor = torch.zeros(self.num_planes, 8, 8, dtype=torch.float32)

        # Determine if we need to flip the board
        flip = from_perspective == -1

        # Encode piece positions (planes 0-11)
        self._encode_pieces(tensor, board, flip)

        # Encode side to move (plane 12)
        plane_idx = self.config.piece_planes
        if board.turn == chess.WHITE:
            tensor[plane_idx].fill_(1.0)
        else:
            tensor[plane_idx].fill_(0.0)
        plane_idx += 1

        # Encode castling rights (planes 13-16)
        castling_rights = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]
        if flip:
            # Swap white and black castling rights when flipped
            castling_rights = [castling_rights[2], castling_rights[3], castling_rights[0], castling_rights[1]]

        for i, has_rights in enumerate(castling_rights):
            if has_rights:
                tensor[plane_idx + i].fill_(1.0)
        plane_idx += self.config.castling_planes

        # Encode en passant (plane 17)
        if board.ep_square is not None:
            file = board.ep_square % 8
            rank = board.ep_square // 8
            if flip:
                file = 7 - file
                rank = 7 - rank
            tensor[plane_idx, rank, file] = 1.0
        plane_idx += 1

        # Encode halfmove clock (plane 18) - normalized to [0, 1]
        halfmove_normalized = min(board.halfmove_clock / 100.0, 1.0)
        tensor[plane_idx].fill_(halfmove_normalized)
        plane_idx += 1

        # Encode fullmove number (plane 19) - normalized
        fullmove_normalized = min(board.fullmove_number / 200.0, 1.0)
        tensor[plane_idx].fill_(fullmove_normalized)
        plane_idx += 1

        # Encode repetition count (planes 20-21)
        if hasattr(board, "is_repetition"):
            if board.is_repetition(1):
                tensor[plane_idx].fill_(1.0)
            if board.is_repetition(2):
                tensor[plane_idx + 1].fill_(1.0)
        plane_idx += self.config.repetition_planes

        # Encode history if enabled
        if self.config.include_history and history:
            self._encode_history(tensor, history, plane_idx, flip)

        return tensor

    def _encode_pieces(
        self,
        tensor: torch.Tensor,
        board: chess.Board,
        flip: bool,
    ) -> None:
        """Encode piece positions into tensor planes.

        Args:
            tensor: Output tensor to fill
            board: Chess board
            flip: Whether to flip for black's perspective
        """
        import chess

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Get file and rank
            file = square % 8
            rank = square // 8

            # Flip if from black's perspective
            if flip:
                file = 7 - file
                rank = 7 - rank

            # Calculate plane index
            piece_type = piece.piece_type
            plane_offset = PIECE_TO_PLANE[piece_type]

            if piece.color == chess.WHITE:
                plane_idx = plane_offset if not flip else plane_offset + 6
            else:
                plane_idx = plane_offset + 6 if not flip else plane_offset

            tensor[plane_idx, rank, file] = 1.0

    def _encode_history(
        self,
        tensor: torch.Tensor,
        history: list[chess.Board],
        start_plane: int,
        flip: bool,
    ) -> None:
        """Encode historical positions into tensor.

        Args:
            tensor: Output tensor to fill
            history: List of previous board positions
            start_plane: Starting plane index for history
            flip: Whether to flip for black's perspective
        """
        for i, hist_board in enumerate(history[-self.config.history_length :]):
            plane_offset = start_plane + i * self.config.piece_planes
            self._encode_pieces_at_offset(tensor, hist_board, plane_offset, flip)

    def _encode_pieces_at_offset(
        self,
        tensor: torch.Tensor,
        board: chess.Board,
        plane_offset: int,
        flip: bool,
    ) -> None:
        """Encode piece positions at a specific plane offset.

        Args:
            tensor: Output tensor to fill
            board: Chess board
            plane_offset: Starting plane index
            flip: Whether to flip for black's perspective
        """
        import chess

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            file = square % 8
            rank = square // 8
            if flip:
                file = 7 - file
                rank = 7 - rank

            piece_type = piece.piece_type
            base_plane = PIECE_TO_PLANE[piece_type]

            if piece.color == chess.WHITE:
                plane_idx = plane_offset + (base_plane if not flip else base_plane + 6)
            else:
                plane_idx = plane_offset + (base_plane + 6 if not flip else base_plane)

            if plane_idx < tensor.shape[0]:
                tensor[plane_idx, rank, file] = 1.0

    def encode_batch(
        self,
        boards: list[chess.Board],
        from_perspective: int | list[int] = 1,
    ) -> torch.Tensor:
        """Encode a batch of chess positions.

        Args:
            boards: List of chess boards
            from_perspective: Perspective (1 or -1) or list of perspectives

        Returns:
            Tensor of shape (batch_size, num_planes, 8, 8)
        """
        batch_size = len(boards)

        if isinstance(from_perspective, int):
            perspectives = [from_perspective] * batch_size
        else:
            perspectives = from_perspective

        tensors = [self.encode(board, persp) for board, persp in zip(boards, perspectives, strict=False)]

        return torch.stack(tensors)

    def decode_piece_planes(self, tensor: torch.Tensor) -> dict[str, list[tuple[int, int]]]:
        """Decode piece positions from tensor (for debugging/visualization).

        Args:
            tensor: Input tensor of shape (num_planes, 8, 8)

        Returns:
            Dictionary mapping piece names to list of (file, rank) positions
        """
        piece_names = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        result = {}

        for i, name in enumerate(piece_names):
            positions = []
            plane = tensor[i]
            for rank in range(8):
                for file in range(8):
                    if plane[rank, file] > 0.5:
                        positions.append((file, rank))
            if positions:
                result[name] = positions

        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChessBoardRepresentation(num_planes={self.num_planes}, "
            f"history={self.config.include_history}, "
            f"history_length={self.config.history_length})"
        )


def board_to_tensor(
    board: chess.Board,
    config: ChessBoardConfig | None = None,
    from_perspective: int = 1,
) -> torch.Tensor:
    """Convenience function to convert a chess board to tensor.

    Args:
        board: Chess board position
        config: Optional configuration
        from_perspective: 1 for white, -1 for black

    Returns:
        Tensor of shape (num_planes, 8, 8)
    """
    encoder = ChessBoardRepresentation(config)
    return encoder.encode(board, from_perspective)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()
