"""Othello domain package."""

from __future__ import annotations

from src.games.othello.config import OthelloConfig
from src.games.othello.registration import register_othello_domain
from src.games.othello.state import OthelloState

__all__ = [
    "OthelloConfig",
    "OthelloState",
    "register_othello_domain",
]
