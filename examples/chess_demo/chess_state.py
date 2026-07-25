"""
Chess Game State for MCTS Integration.

Implements the GameState interface for neural MCTS with python-chess.
Uses chessboard.js for frontend visualization.

Best Practices 2025:
- Protocol-based interfaces
- Efficient tensor representations
- No hardcoded values
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Optional chess import
try:
    import chess
    import chess.pgn
    import chess.svg

    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False
    chess = None  # type: ignore[assignment]

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


# GameState interface - use local Protocol to avoid torch dependency at import time
@runtime_checkable
class GameState(Protocol):
    """Protocol for game state compatible with MCTS."""

    def get_legal_actions(self) -> list[str]:
        """Return list of legal actions."""
        ...

    def apply_action(self, action: str) -> GameState:
        """Apply action and return new state."""
        ...

    def is_terminal(self) -> bool:
        """Check if state is terminal."""
        ...

    def get_reward(self, player: int) -> float:
        """Get reward for player."""
        ...

    def to_tensor(self) -> Any:
        """Convert to tensor representation."""
        ...

    def get_hash(self) -> str:
        """Get unique state hash."""
        ...


# Piece values for evaluation heuristics
# Use integer keys when chess module not available for type safety
if CHESS_AVAILABLE:
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }
else:
    # Fallback with integer piece type constants (1=PAWN, 2=KNIGHT, etc.)
    PIECE_VALUES = {
        1: 100,  # PAWN
        2: 320,  # KNIGHT
        3: 330,  # BISHOP
        4: 500,  # ROOK
        5: 900,  # QUEEN
        6: 20000,  # KING
    }

# Position tables for piece-square evaluation (simplified)
PAWN_TABLE = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    10,
    10,
    20,
    30,
    30,
    20,
    10,
    10,
    5,
    5,
    10,
    25,
    25,
    10,
    5,
    5,
    0,
    0,
    0,
    20,
    20,
    0,
    0,
    0,
    5,
    -5,
    -10,
    0,
    0,
    -10,
    -5,
    5,
    5,
    10,
    10,
    -20,
    -20,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]


@dataclass
class ChessConfig:
    """Configuration for chess game state."""

    # Tensor representation
    num_piece_types: int = 6  # P, N, B, R, Q, K
    num_players: int = 2
    board_size: int = 8

    # Evaluation parameters
    material_weight: float = field(default_factory=lambda: float(os.getenv("CHESS_MATERIAL_WEIGHT", "0.6")))
    position_weight: float = field(default_factory=lambda: float(os.getenv("CHESS_POSITION_WEIGHT", "0.2")))
    mobility_weight: float = field(default_factory=lambda: float(os.getenv("CHESS_MOBILITY_WEIGHT", "0.2")))

    # MCTS parameters
    max_moves: int = field(default_factory=lambda: int(os.getenv("CHESS_MAX_MOVES", "500")))

    def __post_init__(self):
        """Normalize weights."""
        total = self.material_weight + self.position_weight + self.mobility_weight
        self.material_weight /= total
        self.position_weight /= total
        self.mobility_weight /= total


class ChessState(GameState):
    """
    Chess game state implementing MCTS GameState interface.

    Supports:
    - Full chess rules via python-chess
    - Tensor conversion for neural networks
    - FEN serialization for UI
    - Move history tracking
    """

    def __init__(
        self,
        board: Any = None,
        move_history: list[str] | None = None,
        config: ChessConfig | None = None,
    ):
        if not CHESS_AVAILABLE:
            raise RuntimeError("python-chess is required: pip install chess")

        self.board = board if board is not None else chess.Board()
        self.move_history = move_history or []
        self.config = config or ChessConfig()
        self._hash_cache: str | None = None
        self._legal_moves_cache: list[str] | None = None

    def get_legal_actions(self) -> list[str]:
        """Return list of legal moves in UCI format."""
        if self._legal_moves_cache is None:
            self._legal_moves_cache = [move.uci() for move in self.board.legal_moves]
        return self._legal_moves_cache

    def apply_action(self, action: str) -> ChessState:
        """Apply move and return new state."""
        new_board = self.board.copy()
        move = chess.Move.from_uci(action)
        new_board.push(move)

        return ChessState(
            board=new_board,
            move_history=self.move_history + [action],
            config=self.config,
        )

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over() or len(self.move_history) >= self.config.max_moves

    def get_reward(self, player: int = 1) -> float:
        """
        Get reward for the given player.

        Args:
            player: 1 for white, -1 for black

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw/ongoing
        """
        if not self.board.is_game_over():
            return 0.0

        result = self.board.result()

        if result == "1-0":  # White wins
            return 1.0 if player == 1 else -1.0
        elif result == "0-1":  # Black wins
            return -1.0 if player == 1 else 1.0
        else:  # Draw
            return 0.0

    def to_tensor(self) -> Any:
        """
        Convert board to tensor representation.

        Returns:
            Tensor of shape (14, 8, 8):
            - 6 channels for white pieces
            - 6 channels for black pieces
            - 1 channel for current player
            - 1 channel for castling rights
        """
        if not TORCH_AVAILABLE:
            return self._to_numpy()

        # 14 channels: 6 white pieces + 6 black pieces + turn + castling
        tensor = torch.zeros(14, 8, 8)

        piece_map = self.board.piece_map()

        for square, piece in piece_map.items():
            row = 7 - (square // 8)  # Flip for standard view
            col = square % 8

            # Channel index: piece_type - 1 + (6 if black)
            channel = (piece.piece_type - 1) + (0 if piece.color else 6)
            tensor[channel, row, col] = 1.0

        # Turn channel (all 1s if white to move)
        if self.board.turn:
            tensor[12, :, :] = 1.0

        # Castling rights encoded in channel 13
        if self.board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, 7, 7] = 1.0
        if self.board.has_queenside_castling_rights(chess.WHITE):
            tensor[13, 7, 0] = 1.0
        if self.board.has_kingside_castling_rights(chess.BLACK):
            tensor[13, 0, 7] = 1.0
        if self.board.has_queenside_castling_rights(chess.BLACK):
            tensor[13, 0, 0] = 1.0

        return tensor

    def _to_numpy(self) -> np.ndarray:
        """Numpy fallback for tensor conversion."""
        tensor = np.zeros((14, 8, 8), dtype=np.float32)

        piece_map = self.board.piece_map()

        for square, piece in piece_map.items():
            row = 7 - (square // 8)
            col = square % 8
            channel = (piece.piece_type - 1) + (0 if piece.color else 6)
            tensor[channel, row, col] = 1.0

        if self.board.turn:
            tensor[12, :, :] = 1.0

        return tensor

    def get_hash(self) -> str:
        """Get unique state hash for caching."""
        if self._hash_cache is None:
            # Use FEN for hashing (includes position, turn, castling, en passant)
            fen = self.board.fen()
            self._hash_cache = hashlib.sha256(fen.encode()).hexdigest()[:16]
        return self._hash_cache

    def get_fen(self) -> str:
        """Get FEN string for UI."""
        return self.board.fen()

    def get_pgn(self) -> str:
        """Get PGN representation of the game."""
        game = chess.pgn.Game()
        node = game

        temp_board = chess.Board()
        for move_uci in self.move_history:
            move = chess.Move.from_uci(move_uci)
            node = node.add_variation(move)
            temp_board.push(move)

        return str(game)

    def evaluate(self) -> float:
        """
        Evaluate position heuristically.

        Returns:
            Score from white's perspective in [-1, 1]
        """
        if self.board.is_checkmate():
            return -1.0 if self.board.turn else 1.0

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0

        # Material evaluation
        material_score = self._evaluate_material()

        # Mobility evaluation
        mobility_score = self._evaluate_mobility()

        # Position evaluation (simplified)
        position_score = self._evaluate_position()

        # Combine with weights
        total = (
            self.config.material_weight * material_score
            + self.config.mobility_weight * mobility_score
            + self.config.position_weight * position_score
        )

        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, total / 10.0))

    def _evaluate_material(self) -> float:
        """Evaluate material balance."""
        score = 0.0

        for piece_type in chess.PIECE_TYPES:
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            black_count = len(self.board.pieces(piece_type, chess.BLACK))

            value = PIECE_VALUES.get(piece_type, 0)
            score += (white_count - black_count) * value / 100.0

        return score

    def _evaluate_mobility(self) -> float:
        """Evaluate piece mobility."""
        # Count legal moves for current player
        current_moves = len(list(self.board.legal_moves))

        # Approximate opponent moves
        self.board.push(chess.Move.null())
        opponent_moves = len(list(self.board.legal_moves))
        self.board.pop()

        mobility_diff = current_moves - opponent_moves

        # Adjust for whose turn it is
        if not self.board.turn:  # Black to move
            mobility_diff = -mobility_diff

        return mobility_diff / 20.0  # Normalize

    def _evaluate_position(self) -> float:
        """Evaluate piece positions using simple tables."""
        score = 0.0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue

            # Only use pawn table for simplicity
            if piece.piece_type == chess.PAWN:
                table_idx = square if piece.color else (63 - square)
                pos_value = PAWN_TABLE[table_idx] / 100.0
                score += pos_value if piece.color else -pos_value

        return score

    def get_phase(self) -> str:
        """Determine game phase: opening, middlegame, or endgame."""
        piece_count = len(self.board.piece_map())

        if len(self.move_history) < 10 and piece_count > 28:
            return "opening"
        elif piece_count <= 12:
            return "endgame"
        else:
            return "middlegame"

    def get_threats(self) -> list[str]:
        """Get list of threats in current position."""
        threats = []

        # Check for checks
        if self.board.is_check():
            threats.append("check")

        # Check for attacked pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue

            if piece.color == self.board.turn:
                # Our piece - check if attacked
                attackers = self.board.attackers(not self.board.turn, square)
                if attackers:
                    threats.append(f"{piece.symbol()}@{chess.square_name(square)}_attacked")

        return threats

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for API/dashboard."""
        return {
            "fen": self.get_fen(),
            "move_history": self.move_history,
            "turn": "white" if self.board.turn else "black",
            "phase": self.get_phase(),
            "evaluation": self.evaluate(),
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "legal_moves": self.get_legal_actions(),
            "threats": self.get_threats(),
            "fullmove_number": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
        }

    @classmethod
    def from_fen(cls, fen: str, config: ChessConfig | None = None) -> ChessState:
        """Create state from FEN string."""
        board = chess.Board(fen)
        return cls(board=board, config=config)

    @classmethod
    def from_pgn(cls, pgn_str: str, config: ChessConfig | None = None) -> ChessState:
        """Create state from PGN string."""
        import io

        pgn = io.StringIO(pgn_str)
        game = chess.pgn.read_game(pgn)

        if game is None:
            return cls(config=config)

        board = game.board()
        move_history = []

        for move in game.mainline_moves():
            move_history.append(move.uci())
            board.push(move)

        return cls(board=board, move_history=move_history, config=config)

    def get_svg(self, size: int = 400, last_move: bool = True) -> str:
        """Generate SVG representation of the board."""
        last_move_obj = None
        if last_move and self.move_history:
            last_move_obj = chess.Move.from_uci(self.move_history[-1])

        return chess.svg.board(
            self.board,
            size=size,
            lastmove=last_move_obj,
        )

    def __str__(self) -> str:
        """String representation."""
        return str(self.board)

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"ChessState(fen='{self.get_fen()}', moves={len(self.move_history)})"


# Utility functions for move encoding/decoding


def uci_to_index(uci: str) -> int:
    """
    Convert UCI move to action index.

    Chess has ~4672 possible moves (including promotions).
    This uses a simple encoding: from_square * 73 + move_type
    where move_type encodes direction, distance, and promotion.
    """
    if len(uci) < 4:
        return 0

    from_sq = chess.parse_square(uci[:2])
    to_sq = chess.parse_square(uci[2:4])

    # Simple encoding
    return from_sq * 64 + to_sq


def index_to_uci(index: int, board: Any) -> str:
    """Convert action index back to UCI move."""
    from_sq = index // 64
    to_sq = index % 64

    # Validate squares are in bounds (0-63)
    if from_sq >= 64 or from_sq < 0 or to_sq >= 64 or to_sq < 0:
        return ""

    move = chess.Move(from_sq, to_sq)

    # Check if legal
    if move in board.legal_moves:
        return move.uci()

    # Try with promotions
    for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        move = chess.Move(from_sq, to_sq, promotion=promotion)
        if move in board.legal_moves:
            return move.uci()

    return ""
