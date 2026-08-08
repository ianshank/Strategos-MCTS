"""
Chess Constants and Shared Values.

Centralizes all chess-related constants to avoid duplication and ensure
consistency across the codebase. All values are configurable via settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.config.settings import Settings

# Standard starting position FEN
STARTING_FEN: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Default FEN truncation length for logging (overridable via settings)
DEFAULT_FEN_LOG_TRUNCATE_LENGTH: int = 40

# Invalid pawn ranks (0-indexed: rank 0 = 1st rank, rank 7 = 8th rank)
INVALID_PAWN_RANKS: frozenset[int] = frozenset({0, 7})

# Stockfish executable names to search for (platform-independent)
DEFAULT_STOCKFISH_EXECUTABLES: tuple[str, ...] = (
    "stockfish",
    "stockfish.exe",
    "stockfish-ubuntu-x86-64-avx2",
    "stockfish-ubuntu-x86-64",
    "stockfish-macos-m1-apple-silicon",
    "stockfish-windows-x86-64-avx2.exe",
)

# Castling move UCI strings by (color, is_kingside)
# These are determined by standard chess rules and square positions
CASTLING_MOVES: dict[tuple[bool, bool], str] = {
    (True, True): "e1g1",  # White kingside (O-O)
    (True, False): "e1c1",  # White queenside (O-O-O)
    (False, True): "e8g8",  # Black kingside (O-O)
    (False, False): "e8c8",  # Black queenside (O-O-O)
}

# En passant capture ranks (0-indexed)
# These are the ranks where en passant capture squares appear
EN_PASSANT_RANKS: dict[bool, int] = {
    True: 5,  # White: rank 6 (0-indexed as 5) - where Black pawn was captured
    False: 2,  # Black: rank 3 (0-indexed as 2) - where White pawn was captured
}

# UUID short length for game IDs
# Using 8 characters provides ~4 billion unique IDs before 50% collision probability
UUID_SHORT_LENGTH: int = 8

# Default verification thresholds (used when settings unavailable)
DEFAULT_AGREEMENT_THRESHOLD: float = 0.6
DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD: float = 0.3
DEFAULT_ROUTING_THRESHOLD: float = 0.5
DEFAULT_VALUE_DIVERGENCE_THRESHOLD: float = 0.2
DEFAULT_CONFIDENCE: float = 0.5


def get_fen_truncate_length(settings: Settings | None = None) -> int:
    """Get the FEN truncation length from settings or default.

    Args:
        settings: Optional settings instance. If not provided, uses get_settings().

    Returns:
        FEN truncation length for logging
    """
    if settings is None:
        try:
            from src.config.settings import get_settings

            settings = get_settings()
        except Exception:
            return DEFAULT_FEN_LOG_TRUNCATE_LENGTH

    return getattr(settings, "CHESS_FEN_LOG_TRUNCATE_LENGTH", DEFAULT_FEN_LOG_TRUNCATE_LENGTH)


def truncate_fen(fen: str, length: int | None = None, settings: Settings | None = None) -> str:
    """Truncate FEN string for logging purposes.

    Args:
        fen: FEN string to truncate
        length: Optional explicit length. If not provided, uses settings.
        settings: Optional settings instance for length lookup.

    Returns:
        Truncated FEN with "..." suffix if truncated
    """
    if length is None:
        length = get_fen_truncate_length(settings)

    if len(fen) > length:
        return fen[:length] + "..."
    return fen


def get_piece_values(settings: Settings | None = None) -> dict[int, int]:
    """Get piece values from settings for material evaluation.

    Args:
        settings: Optional settings instance. If not provided, uses get_settings().

    Returns:
        Dictionary mapping chess piece types to centipawn values
    """
    import chess

    if settings is None:
        try:
            from src.config.settings import get_settings

            settings = get_settings()
        except Exception:
            # Return defaults if settings unavailable
            return {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
            }

    return {
        chess.PAWN: getattr(settings, "CHESS_PIECE_VALUE_PAWN", 100),
        chess.KNIGHT: getattr(settings, "CHESS_PIECE_VALUE_KNIGHT", 320),
        chess.BISHOP: getattr(settings, "CHESS_PIECE_VALUE_BISHOP", 330),
        chess.ROOK: getattr(settings, "CHESS_PIECE_VALUE_ROOK", 500),
        chess.QUEEN: getattr(settings, "CHESS_PIECE_VALUE_QUEEN", 900),
    }


def get_routing_scores(settings: Settings | None = None) -> dict[str, float]:
    """Get routing consistency scores from settings.

    Args:
        settings: Optional settings instance. If not provided, uses get_settings().

    Returns:
        Dictionary of routing score keys to values
    """
    if settings is None:
        try:
            from src.config.settings import get_settings

            settings = get_settings()
        except Exception:
            # Return defaults if settings unavailable
            return {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }

    return {
        "match": getattr(settings, "CHESS_ROUTING_SCORE_MATCH", 1.0),
        "middlegame_fallback": getattr(settings, "CHESS_ROUTING_SCORE_MIDDLEGAME_FALLBACK", 0.7),
        "phase_appropriate": getattr(settings, "CHESS_ROUTING_SCORE_PHASE_APPROPRIATE", 0.8),
        "phase_mismatch": getattr(settings, "CHESS_ROUTING_SCORE_PHASE_MISMATCH", 0.4),
        "default": getattr(settings, "CHESS_ROUTING_SCORE_DEFAULT", 0.5),
    }


def get_stockfish_executables(settings: Settings | None = None) -> tuple[str, ...]:
    """Get Stockfish executable names to search for.

    Args:
        settings: Optional settings instance. If not provided, uses get_settings().

    Returns:
        Tuple of executable names to search for
    """
    if settings is None:
        try:
            from src.config.settings import get_settings

            settings = get_settings()
        except (ImportError, RuntimeError, OSError):
            return DEFAULT_STOCKFISH_EXECUTABLES

    # Allow custom executables from settings (comma-separated string)
    custom_executables = getattr(settings, "STOCKFISH_EXECUTABLES", None)
    if custom_executables:
        return tuple(name.strip() for name in custom_executables.split(","))

    return DEFAULT_STOCKFISH_EXECUTABLES


__all__ = [
    "STARTING_FEN",
    "DEFAULT_FEN_LOG_TRUNCATE_LENGTH",
    "INVALID_PAWN_RANKS",
    "DEFAULT_STOCKFISH_EXECUTABLES",
    "CASTLING_MOVES",
    "EN_PASSANT_RANKS",
    "UUID_SHORT_LENGTH",
    "DEFAULT_AGREEMENT_THRESHOLD",
    "DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD",
    "DEFAULT_ROUTING_THRESHOLD",
    "DEFAULT_VALUE_DIVERGENCE_THRESHOLD",
    "DEFAULT_CONFIDENCE",
    "get_fen_truncate_length",
    "truncate_fen",
    "get_piece_values",
    "get_routing_scores",
    "get_stockfish_executables",
    "AI_WIN_MARKER",
    "PLAYER_WIN_MARKER",
    "LEARNING_REFRESH_SECONDS",
]

# ---------------------------------------------------------------------------
# UI result markers and timing
# ---------------------------------------------------------------------------

# Lowercase markers used to attribute a finished game to a winner. Matched
# against the lowercased result string, so they must be lowercase here.
# "checkmate" is deliberately NOT a marker: it appears in both win strings and
# identifies the ending, not the winner.
AI_WIN_MARKER: Final[str] = "ai wins"
PLAYER_WIN_MARKER: Final[str] = "you win"

# Auto-refresh interval for the continuous-learning panel, in seconds.
LEARNING_REFRESH_SECONDS: Final[float] = 1.0
