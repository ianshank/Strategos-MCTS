"""
Chess Test Builders.

Provides builder classes for creating chess-specific test fixtures
in a fluent, readable way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.games.chess.config import ChessActionSpaceConfig, ChessBoardConfig, GamePhase
from src.games.chess.state import ChessGameState
from src.games.chess.verification.types import (
    GameResult,
)


@dataclass
class ChessPositionBuilder:
    """Builder for chess position test fixtures.

    Example:
        >>> state = (
        ...     ChessPositionBuilder()
        ...     .with_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        ...     .with_phase(GamePhase.OPENING)
        ...     .build()
        ... )
    """

    _fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    _phase: GamePhase | None = None
    _board_config: ChessBoardConfig = field(default_factory=ChessBoardConfig)
    _action_config: ChessActionSpaceConfig = field(default_factory=ChessActionSpaceConfig)

    def with_fen(self, fen: str) -> ChessPositionBuilder:
        """Set the FEN string.

        Args:
            fen: FEN position string

        Returns:
            Self for chaining
        """
        self._fen = fen
        return self

    def with_phase(self, phase: GamePhase) -> ChessPositionBuilder:
        """Set the expected game phase.

        Args:
            phase: Expected game phase

        Returns:
            Self for chaining
        """
        self._phase = phase
        return self

    def with_board_config(self, config: ChessBoardConfig) -> ChessPositionBuilder:
        """Set the board configuration.

        Args:
            config: Board configuration

        Returns:
            Self for chaining
        """
        self._board_config = config
        return self

    def with_action_config(self, config: ChessActionSpaceConfig) -> ChessPositionBuilder:
        """Set the action space configuration.

        Args:
            config: Action space configuration

        Returns:
            Self for chaining
        """
        self._action_config = config
        return self

    def with_initial_position(self) -> ChessPositionBuilder:
        """Set to initial chess position.

        Returns:
            Self for chaining
        """
        self._fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        return self

    def with_check(self) -> ChessPositionBuilder:
        """Set a position where the side to move is in check.

        Returns:
            Self for chaining
        """
        # White to move, in check from black queen
        self._fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        return self

    def with_mate_threat(self) -> ChessPositionBuilder:
        """Set a position with a mate threat.

        Returns:
            Self for chaining
        """
        # Scholar's mate position
        self._fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        return self

    def with_checkmate(self) -> ChessPositionBuilder:
        """Set a checkmate position.

        Returns:
            Self for chaining
        """
        # Back rank mate
        self._fen = "6k1/5ppp/8/8/8/8/8/R3K3 b - - 0 1"
        return self

    def with_stalemate(self) -> ChessPositionBuilder:
        """Set a stalemate position.

        Returns:
            Self for chaining
        """
        self._fen = "k7/8/1K6/8/8/8/8/8 b - - 0 1"
        return self

    def with_endgame(self) -> ChessPositionBuilder:
        """Set an endgame position.

        Returns:
            Self for chaining
        """
        # King and pawn endgame
        self._fen = "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1"
        self._phase = GamePhase.ENDGAME
        return self

    def with_middlegame(self) -> ChessPositionBuilder:
        """Set a middlegame position.

        Returns:
            Self for chaining
        """
        # Typical middlegame
        self._fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        self._phase = GamePhase.MIDDLEGAME
        return self

    def build(self) -> ChessGameState:
        """Build the ChessGameState.

        Returns:
            Configured ChessGameState instance
        """
        return ChessGameState.from_fen(
            self._fen,
            board_config=self._board_config,
            action_config=self._action_config,
        )


@dataclass
class ChessGameSequenceBuilder:
    """Builder for chess game sequence fixtures.

    Example:
        >>> states, result = (
        ...     ChessGameSequenceBuilder()
        ...     .with_opening("italian_game")
        ...     .with_expected_outcome(GameResult.IN_PROGRESS)
        ...     .build()
        ... )
    """

    _initial_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    _moves: list[str] = field(default_factory=list)
    _expected_outcome: GameResult = GameResult.IN_PROGRESS
    _opening_name: str | None = None

    # Common opening sequences
    OPENINGS: dict[str, list[str]] = field(
        default_factory=lambda: {
            "italian_game": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
            "sicilian": ["e2e4", "c7c5"],
            "french": ["e2e4", "e7e6"],
            "caro_kann": ["e2e4", "c7c6"],
            "queens_gambit": ["d2d4", "d7d5", "c2c4"],
            "kings_indian": ["d2d4", "g8f6", "c2c4", "g7g6"],
            "ruy_lopez": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
            "scholars_mate": ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
            "fools_mate": ["f2f3", "e7e5", "g2g4", "d8h4"],
        }
    )

    def with_initial_fen(self, fen: str) -> ChessGameSequenceBuilder:
        """Set the initial FEN.

        Args:
            fen: Initial FEN string

        Returns:
            Self for chaining
        """
        self._initial_fen = fen
        return self

    def with_moves(self, moves: list[str]) -> ChessGameSequenceBuilder:
        """Set the move sequence.

        Args:
            moves: List of UCI moves

        Returns:
            Self for chaining
        """
        self._moves = moves
        return self

    def with_opening(self, opening_name: str) -> ChessGameSequenceBuilder:
        """Use a named opening.

        Args:
            opening_name: Name of the opening

        Returns:
            Self for chaining

        Raises:
            ValueError: If opening name is not recognized
        """
        if opening_name not in self.OPENINGS:
            raise ValueError(f"Unknown opening: {opening_name}")
        self._moves = self.OPENINGS[opening_name].copy()
        self._opening_name = opening_name
        return self

    def with_expected_outcome(self, outcome: GameResult) -> ChessGameSequenceBuilder:
        """Set the expected game outcome.

        Args:
            outcome: Expected game result

        Returns:
            Self for chaining
        """
        self._expected_outcome = outcome
        return self

    def add_move(self, move: str) -> ChessGameSequenceBuilder:
        """Add a single move.

        Args:
            move: UCI move string

        Returns:
            Self for chaining
        """
        self._moves.append(move)
        return self

    def build(self) -> tuple[list[ChessGameState], GameResult]:
        """Build the game sequence.

        Returns:
            Tuple of (list of states, expected result)
        """
        states: list[ChessGameState] = []
        current_state = ChessGameState.from_fen(self._initial_fen)
        states.append(current_state)

        for move in self._moves:
            try:
                current_state = current_state.apply_action(move)
                states.append(current_state)
            except ValueError:
                break

        return states, self._expected_outcome


@dataclass
class TacticalPositionBuilder:
    """Builder for tactical test scenarios.

    Example:
        >>> scenario = (
        ...     TacticalPositionBuilder()
        ...     .with_type("fork")
        ...     .with_best_move("e4c5")
        ...     .with_evaluation(300)
        ...     .build()
        ... )
    """

    _id: str = "tactical_001"
    _name: str = "Tactical Position"
    _fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    _tactic_type: str = "unknown"
    _best_move: str = ""
    _evaluation_cp: int = 0
    _expected_agent: str = "mcts"
    _difficulty: str = "medium"
    _extra_info: dict[str, Any] = field(default_factory=dict)

    # Common tactical positions
    POSITIONS: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "fork": {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "best_move": "h5f7",
                "evaluation": 500,
                "name": "Scholar's Mate Fork",
            },
            "pin": {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
                "best_move": "f8c5",
                "evaluation": -50,
                "name": "Bishop Pin",
            },
            "skewer": {
                "fen": "8/8/4k3/8/8/4R3/8/4K3 w - - 0 1",
                "best_move": "e3e6",
                "evaluation": 300,
                "name": "Rook Skewer",
            },
            "discovered_attack": {
                "fen": "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
                "best_move": "d2d4",
                "evaluation": 30,
                "name": "Discovered Attack Setup",
            },
            "mate_in_one": {
                "fen": "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
                "best_move": None,  # Black is mated
                "evaluation": 10000,
                "name": "Scholar's Mate",
            },
        }
    )

    def with_id(self, id: str) -> TacticalPositionBuilder:
        """Set the scenario ID.

        Args:
            id: Scenario identifier

        Returns:
            Self for chaining
        """
        self._id = id
        return self

    def with_name(self, name: str) -> TacticalPositionBuilder:
        """Set the scenario name.

        Args:
            name: Scenario name

        Returns:
            Self for chaining
        """
        self._name = name
        return self

    def with_fen(self, fen: str) -> TacticalPositionBuilder:
        """Set the FEN string.

        Args:
            fen: FEN position

        Returns:
            Self for chaining
        """
        self._fen = fen
        return self

    def with_type(self, tactic: str) -> TacticalPositionBuilder:
        """Set the tactic type from presets.

        Args:
            tactic: Type of tactic (fork, pin, skewer, etc.)

        Returns:
            Self for chaining
        """
        if tactic in self.POSITIONS:
            preset = self.POSITIONS[tactic]
            self._fen = preset["fen"]
            self._best_move = preset.get("best_move", "")
            self._evaluation_cp = preset.get("evaluation", 0)
            self._name = preset.get("name", tactic)
        self._tactic_type = tactic
        return self

    def with_best_move(self, move: str) -> TacticalPositionBuilder:
        """Set the best move.

        Args:
            move: Best move in UCI format

        Returns:
            Self for chaining
        """
        self._best_move = move
        return self

    def with_evaluation(self, eval_cp: int) -> TacticalPositionBuilder:
        """Set the evaluation in centipawns.

        Args:
            eval_cp: Evaluation in centipawns

        Returns:
            Self for chaining
        """
        self._evaluation_cp = eval_cp
        return self

    def with_expected_agent(self, agent: str) -> TacticalPositionBuilder:
        """Set the expected best agent.

        Args:
            agent: Expected agent name (hrm, trm, mcts)

        Returns:
            Self for chaining
        """
        self._expected_agent = agent
        return self

    def with_difficulty(self, difficulty: str) -> TacticalPositionBuilder:
        """Set the difficulty level.

        Args:
            difficulty: Difficulty (easy, medium, hard)

        Returns:
            Self for chaining
        """
        self._difficulty = difficulty
        return self

    def build(self) -> dict[str, Any]:
        """Build the tactical scenario.

        Returns:
            Dictionary with scenario data
        """
        return {
            "id": self._id,
            "name": self._name,
            "fen": self._fen,
            "tactic_type": self._tactic_type,
            "best_move": self._best_move,
            "evaluation_cp": self._evaluation_cp,
            "expected_agent": self._expected_agent,
            "difficulty": self._difficulty,
            "extra_info": self._extra_info,
        }


# Convenience functions for common test scenarios
def initial_position() -> ChessGameState:
    """Create initial chess position."""
    return ChessPositionBuilder().with_initial_position().build()


def check_position() -> ChessGameState:
    """Create a position with check."""
    return ChessPositionBuilder().with_check().build()


def checkmate_position() -> ChessGameState:
    """Create a checkmate position."""
    return ChessPositionBuilder().with_checkmate().build()


def stalemate_position() -> ChessGameState:
    """Create a stalemate position."""
    return ChessPositionBuilder().with_stalemate().build()


def endgame_position() -> ChessGameState:
    """Create an endgame position."""
    return ChessPositionBuilder().with_endgame().build()


def italian_game_sequence() -> tuple[list[ChessGameState], GameResult]:
    """Create Italian Game opening sequence."""
    return ChessGameSequenceBuilder().with_opening("italian_game").build()


def scholars_mate_sequence() -> tuple[list[ChessGameState], GameResult]:
    """Create Scholar's Mate sequence."""
    return ChessGameSequenceBuilder().with_opening("scholars_mate").with_expected_outcome(GameResult.WHITE_WINS).build()


def fools_mate_sequence() -> tuple[list[ChessGameState], GameResult]:
    """Create Fool's Mate sequence."""
    return ChessGameSequenceBuilder().with_opening("fools_mate").with_expected_outcome(GameResult.BLACK_WINS).build()


def fork_tactical() -> dict[str, Any]:
    """Create fork tactical scenario."""
    return TacticalPositionBuilder().with_type("fork").build()


def pin_tactical() -> dict[str, Any]:
    """Create pin tactical scenario."""
    return TacticalPositionBuilder().with_type("pin").build()
