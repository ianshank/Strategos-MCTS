"""
Chess Test Fixtures.

Provides reusable test data for chess verification tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.games.chess.config import AgentType, GamePhase


@dataclass
class TacticalScenario:
    """A tactical chess scenario for testing."""

    id: str
    name: str
    fen: str
    best_move: str | None
    expected_agent: AgentType
    evaluation_cp: int = 0
    tactic_type: str = "unknown"
    difficulty: str = "medium"


# Tactical scenarios for sub-agent testing
CHESS_TACTICAL_SCENARIOS: list[TacticalScenario] = [
    TacticalScenario(
        id="fork_001",
        name="Knight Fork",
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        best_move="h5f7",
        expected_agent=AgentType.MCTS,
        evaluation_cp=500,
        tactic_type="fork",
        difficulty="easy",
    ),
    TacticalScenario(
        id="endgame_001",
        name="King and Pawn Endgame",
        fen="8/8/4k3/8/4P3/8/4K3/8 w - - 0 1",
        best_move="e2e3",
        expected_agent=AgentType.TRM,
        evaluation_cp=100,
        tactic_type="endgame",
        difficulty="medium",
    ),
    TacticalScenario(
        id="opening_001",
        name="Italian Game Development",
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        best_move="f8c5",
        expected_agent=AgentType.HRM,
        evaluation_cp=-30,
        tactic_type="development",
        difficulty="easy",
    ),
    TacticalScenario(
        id="tactical_001",
        name="Back Rank Mate Threat",
        fen="6k1/5ppp/8/8/8/8/5PPP/R3K3 w - - 0 1",
        best_move="a1a8",
        expected_agent=AgentType.MCTS,
        evaluation_cp=10000,
        tactic_type="mate",
        difficulty="medium",
    ),
    TacticalScenario(
        id="complex_001",
        name="Complex Middlegame",
        fen="r2qkb1r/pp2pppp/2n2n2/2pp4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
        best_move="c4d5",
        expected_agent=AgentType.MCTS,
        evaluation_cp=40,
        tactic_type="positional",
        difficulty="hard",
    ),
]


# Castling edge cases
CASTLING_EDGE_CASES: list[dict[str, Any]] = [
    {
        "name": "Kingside castling available",
        "fen": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "move": "e1g1",
        "should_be_legal": True,
        "castling_type": "kingside",
    },
    {
        "name": "Queenside castling available",
        "fen": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "move": "e1c1",
        "should_be_legal": True,
        "castling_type": "queenside",
    },
    {
        "name": "Castling through check (illegal)",
        "fen": "r3k2r/pppppppp/8/4r3/8/8/PPPP1PPP/R3K2R w KQkq - 0 1",
        "move": "e1g1",
        "should_be_legal": False,
        "reason": "Cannot castle through check",
    },
    {
        "name": "Castling out of check (illegal)",
        "fen": "r3k2r/pppppppp/8/8/4r3/8/PPPP1PPP/R3K2R w KQkq - 0 1",
        "move": "e1g1",
        "should_be_legal": False,
        "reason": "Cannot castle while in check",
    },
    {
        "name": "Castling rights lost (king moved)",
        "fen": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1",
        "move": "e1g1",
        "should_be_legal": False,
        "reason": "Castling rights lost",
    },
    {
        "name": "Castling blocked by piece",
        "fen": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R2BK2R w KQkq - 0 1",
        "move": "e1c1",
        "should_be_legal": False,
        "reason": "Piece blocking castling path",
    },
]


# En passant edge cases
EN_PASSANT_EDGE_CASES: list[dict[str, Any]] = [
    {
        "name": "En passant available",
        "fen": "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "move": "e5d6",
        "should_be_legal": True,
        "is_en_passant": True,
    },
    {
        "name": "En passant not available (no ep square)",
        "fen": "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
        "move": "e5d6",
        "should_be_legal": False,
        "is_en_passant": False,
    },
    {
        "name": "En passant would expose king to check (illegal)",
        "fen": "8/8/8/k2pP2R/8/8/8/K7 w - d6 0 1",
        "move": "e5d6",
        "should_be_legal": False,
        "reason": "Would expose king to check",
    },
    {
        "name": "Black en passant",
        "fen": "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
        "move": "e4d3",
        "should_be_legal": True,
        "is_en_passant": True,
    },
]


# Promotion edge cases
PROMOTION_EDGE_CASES: list[dict[str, Any]] = [
    {
        "name": "Queen promotion",
        "fen": "8/4P3/8/8/8/8/8/4K2k w - - 0 1",
        "move": "e7e8q",
        "should_be_legal": True,
        "promotion_piece": "queen",
    },
    {
        "name": "Knight promotion",
        "fen": "8/4P3/8/8/8/8/8/4K2k w - - 0 1",
        "move": "e7e8n",
        "should_be_legal": True,
        "promotion_piece": "knight",
    },
    {
        "name": "Rook promotion",
        "fen": "8/4P3/8/8/8/8/8/4K2k w - - 0 1",
        "move": "e7e8r",
        "should_be_legal": True,
        "promotion_piece": "rook",
    },
    {
        "name": "Bishop promotion",
        "fen": "8/4P3/8/8/8/8/8/4K2k w - - 0 1",
        "move": "e7e8b",
        "should_be_legal": True,
        "promotion_piece": "bishop",
    },
    {
        "name": "Promotion with capture",
        "fen": "3n4/4P3/8/8/8/8/8/4K2k w - - 0 1",
        "move": "e7d8q",
        "should_be_legal": True,
        "promotion_piece": "queen",
        "is_capture": True,
    },
    {
        "name": "Black queen promotion",
        "fen": "4K2k/8/8/8/8/8/4p3/8 b - - 0 1",
        "move": "e2e1q",
        "should_be_legal": True,
        "promotion_piece": "queen",
    },
]


# Famous games for full playthrough testing
FAMOUS_GAMES: list[dict[str, Any]] = [
    {
        "name": "Scholar's Mate",
        "moves": ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
        "result": "white_wins",
        "description": "Classic beginner trap",
    },
    {
        "name": "Fool's Mate",
        "moves": ["f2f3", "e7e5", "g2g4", "d8h4"],
        "result": "black_wins",
        "description": "Fastest possible checkmate",
    },
    {
        "name": "Opera Game (partial)",
        "moves": [
            "e2e4",
            "e7e5",
            "g1f3",
            "d7d6",
            "d2d4",
            "c8g4",
            "d4e5",
            "g4f3",
            "d1f3",
            "d6e5",
            "f1c4",
            "g8f6",
        ],
        "result": "in_progress",
        "description": "Paul Morphy's famous game opening",
    },
    {
        "name": "Italian Game Main Line",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
        "result": "in_progress",
        "description": "Classic opening",
    },
]


# Phase-specific positions
PHASE_POSITIONS: dict[GamePhase, list[dict[str, Any]]] = {
    GamePhase.OPENING: [
        {
            "name": "Starting position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move_number": 1,
        },
        {
            "name": "After 1.e4",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "move_number": 1,
        },
        {
            "name": "Italian Game",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "move_number": 3,
        },
    ],
    GamePhase.MIDDLEGAME: [
        {
            "name": "Complex middlegame",
            "fen": "r2qkb1r/pp2pppp/2n2n2/2pp4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
            "move_number": 6,
        },
        {
            "name": "Tactical middlegame",
            "fen": "r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQ - 0 7",
            "move_number": 7,
        },
    ],
    GamePhase.ENDGAME: [
        {
            "name": "King and pawn",
            "fen": "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1",
            "material": "K+P vs K",
        },
        {
            "name": "Rook endgame",
            "fen": "8/8/4k3/8/4P3/8/4K3/4R3 w - - 0 1",
            "material": "R+K+P vs K",
        },
        {
            "name": "Queen endgame",
            "fen": "8/8/4k3/8/4Q3/8/4K3/8 w - - 0 1",
            "material": "Q+K vs K",
        },
    ],
}


# Error injection scenarios
ERROR_INJECTION_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "Invalid FEN",
        "fen": "invalid_fen_string",
        "error_type": "invalid_fen",
    },
    {
        "name": "Invalid move format",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move": "invalid_move",
        "error_type": "invalid_uci",
    },
    {
        "name": "Illegal move",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move": "e1e2",
        "error_type": "illegal_move",
    },
    {
        "name": "Move to occupied square (same color)",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move": "a1a2",
        "error_type": "illegal_move",
    },
]


def get_tactical_scenarios() -> list[TacticalScenario]:
    """Get all tactical test scenarios."""
    return CHESS_TACTICAL_SCENARIOS.copy()


def get_castling_edge_cases() -> list[dict[str, Any]]:
    """Get all castling edge case scenarios."""
    return CASTLING_EDGE_CASES.copy()


def get_en_passant_edge_cases() -> list[dict[str, Any]]:
    """Get all en passant edge case scenarios."""
    return EN_PASSANT_EDGE_CASES.copy()


def get_promotion_edge_cases() -> list[dict[str, Any]]:
    """Get all promotion edge case scenarios."""
    return PROMOTION_EDGE_CASES.copy()


def get_famous_games() -> list[dict[str, Any]]:
    """Get all famous game sequences."""
    return FAMOUS_GAMES.copy()


def get_phase_positions(phase: GamePhase) -> list[dict[str, Any]]:
    """Get positions for a specific game phase."""
    return PHASE_POSITIONS.get(phase, []).copy()


def get_error_injection_scenarios() -> list[dict[str, Any]]:
    """Get all error injection scenarios."""
    return ERROR_INJECTION_SCENARIOS.copy()
