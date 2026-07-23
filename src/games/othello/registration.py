"""Othello domain registration helper."""

from __future__ import annotations

from src.framework.domain_registry import DomainRegistry, DomainSpec
from src.games.othello.config import OthelloConfig
from src.games.othello.state import OthelloState

CONFIG = OthelloConfig()


def register_othello_domain() -> None:
    """Register the Othello domain in the DomainRegistry."""
    DomainRegistry.register(
        DomainSpec(
            name="othello",
            metric="win_rate",
            single_agent=False,
            initial_state_fn=OthelloState.create_initial_state,
            action_space_size=CONFIG.action_space_size,
        )
    )
