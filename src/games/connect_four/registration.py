"""Connect Four domain registration helper."""

from __future__ import annotations

from src.framework.domain_registry import DomainRegistry, DomainSpec
from src.games.connect_four.config import ConnectFourConfig
from src.games.connect_four.state import ConnectFourState

CONFIG = ConnectFourConfig()


def register_connect_four_domain() -> None:
    """Register the Connect Four domain in the DomainRegistry."""
    DomainRegistry.register(
        DomainSpec(
            name="connect_four",
            metric="win_rate",
            single_agent=False,
            initial_state_fn=ConnectFourState.create_initial_state,
            action_space_size=CONFIG.action_space_size,
        )
    )
