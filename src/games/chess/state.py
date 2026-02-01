"""
Chess Game State Module.

Implements the GameState interface for chess, providing all methods required
by the Neural MCTS framework for self-play and training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import numpy as np

from src.framework.mcts.neural_mcts import GameState
from src.games.chess.action_space import ChessActionEncoder
from src.games.chess.config import ChessActionSpaceConfig, ChessBoardConfig, GamePhase
from src.games.chess.representation import ChessBoardRepresentation


@dataclass(frozen=False)
class ChessGameState(GameState):
    """Chess game state implementing the GameState interface.

    This class wraps a python-chess Board and provides all methods required
    by the Neural MCTS framework for self-play training and inference.

    The state is immutable - apply_action returns a new state instance.
    """

    # Board state (use FEN string internally for hashability)
    _fen: str = field(default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # Configuration
    _board_config: ChessBoardConfig = field(default_factory=ChessBoardConfig)
    _action_config: ChessActionSpaceConfig = field(default_factory=ChessActionSpaceConfig)

    # Cached objects (not included in hash/equality)
    _board: Any = field(default=None, repr=False, compare=False)
    _encoder: Any = field(default=None, repr=False, compare=False)
    _representation: Any = field(default=None, repr=False, compare=False)
    _legal_actions_cache: list[str] | None = field(default=None, repr=False, compare=False)
    _history: list[str] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize internal board representation."""
        import chess

        if self._board is None:
            object.__setattr__(self, "_board", chess.Board(self._fen))
        if self._encoder is None:
            object.__setattr__(self, "_encoder", ChessActionEncoder(self._action_config))
        if self._representation is None:
            object.__setattr__(self, "_representation", ChessBoardRepresentation(self._board_config))

    @classmethod
    def from_fen(
        cls,
        fen: str,
        board_config: ChessBoardConfig | None = None,
        action_config: ChessActionSpaceConfig | None = None,
    ) -> ChessGameState:
        """Create a game state from a FEN string.

        Args:
            fen: FEN string representation of the position
            board_config: Optional board configuration
            action_config: Optional action space configuration

        Returns:
            New ChessGameState instance
        """
        return cls(
            _fen=fen,
            _board_config=board_config or ChessBoardConfig(),
            _action_config=action_config or ChessActionSpaceConfig(),
        )

    @classmethod
    def initial(
        cls,
        board_config: ChessBoardConfig | None = None,
        action_config: ChessActionSpaceConfig | None = None,
    ) -> ChessGameState:
        """Create the initial chess position.

        Args:
            board_config: Optional board configuration
            action_config: Optional action space configuration

        Returns:
            New ChessGameState at starting position
        """
        return cls(
            _board_config=board_config or ChessBoardConfig(),
            _action_config=action_config or ChessActionSpaceConfig(),
        )

    @property
    def board(self) -> Any:
        """Get the underlying python-chess Board object."""
        return self._board

    @property
    def fen(self) -> str:
        """Get the FEN string for this position."""
        return self._fen

    @property
    def current_player(self) -> int:
        """Get current player (1 for white, -1 for black)."""
        import chess

        return 1 if self._board.turn == chess.WHITE else -1

    @property
    def move_number(self) -> int:
        """Get the current full move number."""
        return self._board.fullmove_number

    @property
    def halfmove_clock(self) -> int:
        """Get the halfmove clock (for 50-move rule)."""
        return self._board.halfmove_clock

    def get_legal_actions(self) -> list[str]:
        """Return list of legal moves in UCI format.

        Returns:
            List of legal moves as UCI strings (e.g., ['e2e4', 'd2d4', ...])
        """
        if self._legal_actions_cache is None:
            object.__setattr__(
                self,
                "_legal_actions_cache",
                [move.uci() for move in self._board.legal_moves],
            )
        return self._legal_actions_cache

    def apply_action(self, action: str) -> ChessGameState:
        """Apply a move and return a new state.

        Args:
            action: Move in UCI format (e.g., 'e2e4')

        Returns:
            New ChessGameState after the move

        Raises:
            ValueError: If the move is illegal
        """
        import chess

        # Validate move is legal
        try:
            move = chess.Move.from_uci(action)
        except ValueError as e:
            raise ValueError(f"Invalid UCI move: {action}") from e

        if move not in self._board.legal_moves:
            raise ValueError(f"Illegal move: {action}")

        # Create new board with move applied
        new_board = self._board.copy()
        new_board.push(move)

        # Track history for repetition detection
        new_history = self._history.copy()
        new_history.append(self._fen)

        # Create new state
        return ChessGameState(
            _fen=new_board.fen(),
            _board_config=self._board_config,
            _action_config=self._action_config,
            _board=new_board,
            _encoder=self._encoder,
            _representation=self._representation,
            _history=new_history,
        )

    def is_terminal(self) -> bool:
        """Check if the game is over.

        Returns:
            True if game is over (checkmate, stalemate, draw)
        """
        return self._board.is_game_over()

    def get_reward(self, player: int = 1) -> float:
        """Get the reward for the specified player.

        Args:
            player: Player to get reward for (1 for white, -1 for black)

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw or non-terminal
        """
        if not self.is_terminal():
            return 0.0

        outcome = self._board.outcome()
        if outcome is None:
            return 0.0

        # Determine winner
        if outcome.winner is None:
            return 0.0  # Draw

        import chess

        if outcome.winner == chess.WHITE:
            return 1.0 if player == 1 else -1.0
        else:
            return -1.0 if player == 1 else 1.0

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input.

        Returns:
            Tensor of shape (num_planes, 8, 8)
        """
        return self._representation.encode(
            self._board,
            from_perspective=self.current_player,
        )

    def get_canonical_form(self, player: int) -> ChessGameState:
        """Get state from the perspective of the specified player.

        For chess, this flips the board when viewing from black's perspective.

        Args:
            player: Player perspective (1 for white, -1 for black)

        Returns:
            State from the player's perspective (may be same object if white)
        """
        if player == 1:
            return self
        # For black's perspective, return same state but tensor encoding will flip
        return self

    def get_hash(self) -> str:
        """Get unique hash for this state.

        Used for MCTS node caching and transposition detection.

        Returns:
            Hash string (FEN representation)
        """
        return self._fen

    def action_to_index(self, action: str) -> int:
        """Map a move to its index in the action space.

        Args:
            action: Move in UCI format

        Returns:
            Index in the neural network output space [0, action_size)
        """
        from_black = self.current_player == -1
        return self._encoder.encode_move(action, from_black_perspective=from_black)

    def index_to_action(self, index: int) -> str:
        """Map an action index to a move.

        Args:
            index: Index in the neural network output space

        Returns:
            Move in UCI format
        """
        from_black = self.current_player == -1
        return self._encoder.decode_move(index, from_black_perspective=from_black)

    def get_action_mask(self) -> np.ndarray:
        """Get mask of legal actions.

        Returns:
            Boolean numpy array where True indicates legal action
        """
        from_black = self.current_player == -1
        return self._encoder.get_legal_action_mask(self._board, from_black_perspective=from_black)

    def get_game_phase(self) -> GamePhase:
        """Determine the current game phase.

        Uses material count and move number to classify phase.

        Returns:
            GamePhase enum value
        """
        import chess

        # Count material
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        total_material = 0
        queens = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_pieces = len(self._board.pieces(piece_type, chess.WHITE))
            black_pieces = len(self._board.pieces(piece_type, chess.BLACK))
            total_material += (white_pieces + black_pieces) * piece_values[piece_type]
            if piece_type == chess.QUEEN:
                queens = white_pieces + black_pieces

        # Opening: first 10 moves or high material
        if self.move_number <= 10:
            return GamePhase.OPENING

        # Endgame: low material (roughly less than R+R+minor each side)
        # or no queens and low material
        if total_material < 26 or (queens == 0 and total_material < 32):
            return GamePhase.ENDGAME

        return GamePhase.MIDDLEGAME

    def get_material_balance(self) -> int:
        """Get material balance from white's perspective.

        Returns:
            Positive = white advantage, negative = black advantage
        """
        import chess

        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }

        balance = 0
        for piece_type, value in piece_values.items():
            white_count = len(self._board.pieces(piece_type, chess.WHITE))
            black_count = len(self._board.pieces(piece_type, chess.BLACK))
            balance += (white_count - black_count) * value

        return balance

    def is_check(self) -> bool:
        """Check if the current player is in check."""
        return self._board.is_check()

    def is_checkmate(self) -> bool:
        """Check if the current player is checkmated."""
        return self._board.is_checkmate()

    def is_stalemate(self) -> bool:
        """Check if the position is stalemate."""
        return self._board.is_stalemate()

    def is_insufficient_material(self) -> bool:
        """Check if there's insufficient material to mate."""
        return self._board.is_insufficient_material()

    def is_fifty_moves(self) -> bool:
        """Check if fifty-move rule can be claimed."""
        return self._board.is_fifty_moves()

    def is_repetition(self, count: int = 3) -> bool:
        """Check if position has repeated.

        Args:
            count: Number of repetitions to check for

        Returns:
            True if position has repeated 'count' times
        """
        return self._board.is_repetition(count)

    def copy(self) -> ChessGameState:
        """Create a copy of this state.

        Returns:
            New ChessGameState instance with same position
        """
        return ChessGameState.from_fen(
            self._fen,
            board_config=self._board_config,
            action_config=self._action_config,
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return str(self._board)

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ChessGameState(fen='{self._fen}', phase={self.get_game_phase().value})"

    def __hash__(self) -> int:
        """Hash based on FEN string."""
        return hash(self._fen)

    def __eq__(self, other: object) -> bool:
        """Equality based on FEN string."""
        if not isinstance(other, ChessGameState):
            return NotImplemented
        return self._fen == other._fen


def create_initial_state(
    board_config: ChessBoardConfig | None = None,
    action_config: ChessActionSpaceConfig | None = None,
) -> ChessGameState:
    """Factory function to create initial chess state.

    This function is suitable for passing to UnifiedTrainingOrchestrator.

    Args:
        board_config: Optional board configuration
        action_config: Optional action space configuration

    Returns:
        Initial chess position
    """
    return ChessGameState.initial(board_config, action_config)


def create_state_from_fen(
    fen: str,
    board_config: ChessBoardConfig | None = None,
    action_config: ChessActionSpaceConfig | None = None,
) -> ChessGameState:
    """Factory function to create chess state from FEN.

    Args:
        fen: FEN string
        board_config: Optional board configuration
        action_config: Optional action space configuration

    Returns:
        Chess state at specified position
    """
    return ChessGameState.from_fen(fen, board_config, action_config)
