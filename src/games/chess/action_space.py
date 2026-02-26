"""
Chess Action Space Encoder/Decoder.

Implements AlphaZero-style action encoding for chess moves.
Maps between chess moves and neural network action indices.

Action space structure (73 planes x 64 squares = 4672 actions):
- Planes 0-55: Queen-like moves (8 directions x 7 distances)
- Planes 56-63: Knight moves (8 possible L-shaped moves)
- Planes 64-72: Underpromotions (3 piece types x 3 directions)

The source square determines the position in the 64-square grid,
and the move type determines the plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import chess

from src.games.chess.config import ChessActionSpaceConfig


class MoveDirection(IntEnum):
    """Queen-like move directions (clockwise from North)."""

    N = 0  # North (up)
    NE = 1  # Northeast
    E = 2  # East (right)
    SE = 3  # Southeast
    S = 4  # South (down)
    SW = 5  # Southwest
    W = 6  # West (left)
    NW = 7  # Northwest


class KnightMove(IntEnum):
    """Knight move directions (clockwise from NNE)."""

    NNE = 0  # 2 up, 1 right
    ENE = 1  # 1 up, 2 right
    ESE = 2  # 1 down, 2 right
    SSE = 3  # 2 down, 1 right
    SSW = 4  # 2 down, 1 left
    WSW = 5  # 1 down, 2 left
    WNW = 6  # 1 up, 2 left
    NNW = 7  # 2 up, 1 left


class PromotionDirection(IntEnum):
    """Promotion move directions."""

    LEFT = 0  # Capture left and promote
    STRAIGHT = 1  # Push forward and promote
    RIGHT = 2  # Capture right and promote


class PromotionPiece(IntEnum):
    """Underpromotion piece types (Queen is default, not encoded)."""

    KNIGHT = 0
    BISHOP = 1
    ROOK = 2


# Direction vectors for queen-like moves (file_delta, rank_delta)
QUEEN_DIRECTION_VECTORS: dict[MoveDirection, tuple[int, int]] = {
    MoveDirection.N: (0, 1),
    MoveDirection.NE: (1, 1),
    MoveDirection.E: (1, 0),
    MoveDirection.SE: (1, -1),
    MoveDirection.S: (0, -1),
    MoveDirection.SW: (-1, -1),
    MoveDirection.W: (-1, 0),
    MoveDirection.NW: (-1, 1),
}

# Direction vectors for knight moves (file_delta, rank_delta)
KNIGHT_DIRECTION_VECTORS: dict[KnightMove, tuple[int, int]] = {
    KnightMove.NNE: (1, 2),
    KnightMove.ENE: (2, 1),
    KnightMove.ESE: (2, -1),
    KnightMove.SSE: (1, -2),
    KnightMove.SSW: (-1, -2),
    KnightMove.WSW: (-2, -1),
    KnightMove.WNW: (-2, 1),
    KnightMove.NNW: (-1, 2),
}


@dataclass
class EncodedMove:
    """Represents an encoded chess move."""

    action_index: int
    source_square: int
    move_plane: int
    move_type: str  # "queen", "knight", "underpromotion"
    is_valid: bool = True


class ChessActionEncoder:
    """Encodes and decodes chess moves for neural network action space.

    Uses AlphaZero-style encoding:
    - 56 queen-like move planes (8 directions x 7 distances)
    - 8 knight move planes
    - 9 underpromotion planes (3 pieces x 3 directions)

    Total: 73 planes x 64 squares = 4672 actions
    """

    def __init__(self, config: ChessActionSpaceConfig | None = None) -> None:
        """Initialize the action encoder.

        Args:
            config: Action space configuration. Uses defaults if None.
        """
        self.config = config or ChessActionSpaceConfig()

        # Precompute plane offsets
        self._queen_plane_offset = 0
        self._knight_plane_offset = self.config.queen_moves  # 56
        self._promotion_plane_offset = self._knight_plane_offset + self.config.knight_move_types  # 64

        # Build lookup tables for fast encoding/decoding
        self._build_lookup_tables()

    def _build_lookup_tables(self) -> None:
        """Build lookup tables for efficient move encoding/decoding."""
        # Index to move info mapping
        self._index_to_move: dict[int, dict] = {}

        # Build queen move lookups
        for direction in MoveDirection:
            for distance in range(1, 8):  # 1-7 squares
                plane = direction * 7 + (distance - 1)
                for square in range(64):
                    action_idx = plane * 64 + square
                    file_from = square % 8
                    rank_from = square // 8
                    delta = QUEEN_DIRECTION_VECTORS[direction]
                    file_to = file_from + delta[0] * distance
                    rank_to = rank_from + delta[1] * distance

                    if 0 <= file_to < 8 and 0 <= rank_to < 8:
                        to_square = rank_to * 8 + file_to
                        self._index_to_move[action_idx] = {
                            "from": square,
                            "to": to_square,
                            "type": "queen",
                            "direction": direction,
                            "distance": distance,
                            "promotion": None,
                        }

        # Build knight move lookups
        for knight_dir in KnightMove:
            plane = self._knight_plane_offset + knight_dir
            for square in range(64):
                action_idx = plane * 64 + square
                file_from = square % 8
                rank_from = square // 8
                delta = KNIGHT_DIRECTION_VECTORS[knight_dir]
                file_to = file_from + delta[0]
                rank_to = rank_from + delta[1]

                if 0 <= file_to < 8 and 0 <= rank_to < 8:
                    to_square = rank_to * 8 + file_to
                    self._index_to_move[action_idx] = {
                        "from": square,
                        "to": to_square,
                        "type": "knight",
                        "direction": knight_dir,
                        "distance": None,
                        "promotion": None,
                    }

        # Build underpromotion lookups
        for piece in PromotionPiece:
            for promo_direction in PromotionDirection:
                plane = self._promotion_plane_offset + piece * 3 + promo_direction
                for square in range(64):
                    action_idx = plane * 64 + square
                    file_from = square % 8
                    rank_from = square // 8

                    # Promotions only from 7th rank (for white) or 2nd rank (for black)
                    # We encode from white's perspective, board flip handles black
                    if rank_from != 6:  # Only 7th rank (index 6) for white pawns
                        continue

                    # Calculate destination
                    file_delta = promo_direction - 1  # LEFT=-1, STRAIGHT=0, RIGHT=1
                    file_to = file_from + file_delta
                    rank_to = 7  # Promotion always to 8th rank

                    if 0 <= file_to < 8:
                        to_square = rank_to * 8 + file_to
                        # Map piece index to chess piece type
                        promo_piece = ["n", "b", "r"][piece]  # knight, bishop, rook
                        self._index_to_move[action_idx] = {
                            "from": square,
                            "to": to_square,
                            "type": "underpromotion",
                            "direction": promo_direction,
                            "distance": None,
                            "promotion": promo_piece,
                        }

    @property
    def action_size(self) -> int:
        """Total size of action space."""
        return self.config.total_actions

    @lru_cache(maxsize=8192)  # noqa: B019 - intentional: class used as singleton
    def encode_move(self, move_uci: str, from_black_perspective: bool = False) -> int:
        """Encode a UCI move string to action index.

        Args:
            move_uci: Move in UCI format (e.g., "e2e4", "e7e8q")
            from_black_perspective: If True, flip the board for black's perspective

        Returns:
            Action index in range [0, 4671]

        Raises:
            ValueError: If move cannot be encoded
        """
        # Parse UCI move
        from_square_str = move_uci[:2]
        to_square_str = move_uci[2:4]
        promotion = move_uci[4:5] if len(move_uci) > 4 else None

        from_file = ord(from_square_str[0]) - ord("a")
        from_rank = int(from_square_str[1]) - 1
        to_file = ord(to_square_str[0]) - ord("a")
        to_rank = int(to_square_str[1]) - 1

        # Flip for black's perspective if needed
        if from_black_perspective:
            from_file = 7 - from_file
            from_rank = 7 - from_rank
            to_file = 7 - to_file
            to_rank = 7 - to_rank

        from_square = from_rank * 8 + from_file
        from_square = from_rank * 8 + from_file
        # to_square unused

        # Calculate move delta
        file_delta = to_file - from_file
        rank_delta = to_rank - from_rank

        # Check for underpromotion
        if promotion and promotion.lower() in ["n", "b", "r"]:
            piece_idx = {"n": 0, "b": 1, "r": 2}[promotion.lower()]

            # Determine direction
            if file_delta == -1:
                promo_dir = PromotionDirection.LEFT
            elif file_delta == 0:
                promo_dir = PromotionDirection.STRAIGHT
            else:
                promo_dir = PromotionDirection.RIGHT

            plane = self._promotion_plane_offset + piece_idx * 3 + promo_dir
            return plane * 64 + from_square

        # Check for knight move
        if (abs(file_delta), abs(rank_delta)) in [(1, 2), (2, 1)]:
            # Find matching knight direction
            for knight_dir, delta in KNIGHT_DIRECTION_VECTORS.items():
                if delta == (file_delta, rank_delta):
                    plane = self._knight_plane_offset + knight_dir
                    return plane * 64 + from_square
            raise ValueError(f"Could not encode knight move: {move_uci}")

        # Queen-like move (includes queen promotions which are default)
        # Determine direction
        direction = self._get_queen_direction(file_delta, rank_delta)
        if direction is None:
            raise ValueError(f"Invalid move direction: {move_uci}")

        # Calculate distance
        distance = max(abs(file_delta), abs(rank_delta))
        if distance < 1 or distance > 7:
            raise ValueError(f"Invalid move distance: {move_uci}")

        plane = direction * 7 + (distance - 1)
        return plane * 64 + from_square

    def _get_queen_direction(self, file_delta: int, rank_delta: int) -> MoveDirection | None:
        """Get queen direction from file and rank deltas."""
        if file_delta == 0 and rank_delta > 0:
            return MoveDirection.N
        elif file_delta > 0 and rank_delta > 0 and file_delta == rank_delta:
            return MoveDirection.NE
        elif file_delta > 0 and rank_delta == 0:
            return MoveDirection.E
        elif file_delta > 0 and rank_delta < 0 and file_delta == -rank_delta:
            return MoveDirection.SE
        elif file_delta == 0 and rank_delta < 0:
            return MoveDirection.S
        elif file_delta < 0 and rank_delta < 0 and file_delta == rank_delta:
            return MoveDirection.SW
        elif file_delta < 0 and rank_delta == 0:
            return MoveDirection.W
        elif file_delta < 0 and rank_delta > 0 and -file_delta == rank_delta:
            return MoveDirection.NW
        return None

    def decode_move(self, action_index: int, from_black_perspective: bool = False) -> str:
        """Decode action index to UCI move string.

        Args:
            action_index: Action index in range [0, 4671]
            from_black_perspective: If True, flip the board for black's perspective

        Returns:
            Move in UCI format (e.g., "e2e4")

        Raises:
            ValueError: If action index is invalid or doesn't map to a valid move
        """
        if action_index < 0 or action_index >= self.action_size:
            raise ValueError(f"Action index {action_index} out of range [0, {self.action_size - 1}]")

        move_info = self._index_to_move.get(action_index)
        if move_info is None:
            raise ValueError(f"Action index {action_index} does not map to a valid move")

        from_square = move_info["from"]
        to_square = move_info["to"]
        promotion = move_info["promotion"]

        from_file = from_square % 8
        from_rank = from_square // 8
        to_file = to_square % 8
        to_rank = to_square // 8

        # Flip for black's perspective if needed
        if from_black_perspective:
            from_file = 7 - from_file
            from_rank = 7 - from_rank
            to_file = 7 - to_file
            to_rank = 7 - to_rank

        # Build UCI string
        from_str = chr(ord("a") + from_file) + str(from_rank + 1)
        to_str = chr(ord("a") + to_file) + str(to_rank + 1)
        uci = from_str + to_str

        if promotion:
            uci += promotion

        return uci

    def get_legal_action_mask(
        self,
        board: chess.Board,
        from_black_perspective: bool = False,
    ) -> np.ndarray:
        """Get mask of legal actions for current board position.

        Args:
            board: Chess board position
            from_black_perspective: If True, encode from black's perspective

        Returns:
            Boolean numpy array of shape (action_size,) where True = legal
        """
        mask = np.zeros(self.action_size, dtype=np.bool_)

        for move in board.legal_moves:
            try:
                action_idx = self.encode_move(move.uci(), from_black_perspective)
                mask[action_idx] = True
            except ValueError:
                # Some moves may not be encodable (shouldn't happen with proper encoding)
                continue

        return mask

    def filter_policy_to_legal(
        self,
        policy: np.ndarray,
        board: chess.Board,
        from_black_perspective: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Filter policy logits to legal moves and return probabilities.

        Args:
            policy: Policy logits/probabilities of shape (action_size,)
            board: Chess board position
            from_black_perspective: If True, encode from black's perspective
            temperature: Temperature for softmax (lower = more deterministic)

        Returns:
            Dictionary mapping UCI moves to probabilities
        """
        legal_mask = self.get_legal_action_mask(board, from_black_perspective)

        # Mask illegal moves with very negative value
        masked_policy = np.where(legal_mask, policy, -1e9)

        # Apply temperature and softmax
        if temperature != 1.0:
            masked_policy = masked_policy / temperature

        # Softmax over legal moves only
        exp_policy = np.exp(masked_policy - np.max(masked_policy))
        exp_policy = np.where(legal_mask, exp_policy, 0)
        sum_exp = np.sum(exp_policy)

        if sum_exp == 0:
            # Fallback to uniform over legal moves
            num_legal = np.sum(legal_mask)
            probs = np.where(legal_mask, 1.0 / num_legal, 0)
        else:
            probs = exp_policy / sum_exp

        # Build move -> probability mapping
        result = {}
        for action_idx in np.where(legal_mask)[0]:
            move_uci = self.decode_move(int(action_idx), from_black_perspective)
            result[move_uci] = float(probs[action_idx])

        return result

    def encode_moves_batch(
        self,
        moves: list[str],
        from_black_perspective: bool = False,
    ) -> np.ndarray:
        """Encode a batch of UCI moves to action indices.

        Args:
            moves: List of UCI move strings
            from_black_perspective: If True, flip for black's perspective

        Returns:
            Numpy array of action indices
        """
        return np.array([self.encode_move(m, from_black_perspective) for m in moves])

    def get_action_info(self, action_index: int) -> dict | None:
        """Get detailed information about an action.

        Args:
            action_index: Action index

        Returns:
            Dictionary with move details or None if invalid
        """
        return self._index_to_move.get(action_index)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChessActionEncoder(action_size={self.action_size}, "
            f"queen_planes={self.config.queen_moves}, "
            f"knight_planes={self.config.knight_move_types}, "
            f"promo_planes={self.config.promotion_piece_types * self.config.promotion_directions})"
        )
