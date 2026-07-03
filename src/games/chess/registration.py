"""Optional DomainRegistry registration for the chess domain.

Chess is the adversarial (win-rate) M5 domain, but depends on the optional
``python-chess`` package (the ``chess`` extra) and torch. Registration therefore
degrades to a no-op when either dependency is missing, so importing this module is
always safe. :meth:`~src.framework.domain_registry.DomainRegistry.get` consults this
module lazily on a registry miss — ``DomainRegistry.get("chess")`` works without any
explicit import when the extras are installed.
"""

from __future__ import annotations

from src.framework.domain_registry import METRIC_WIN_RATE, DomainRegistry, register_domain
from src.observability.logging import get_logger

logger = get_logger(__name__)

CHESS_DOMAIN = "chess"


def chess_available() -> bool:
    """Whether the chess domain's optional dependencies (python-chess, torch) import."""
    try:
        import chess  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def register_chess_domain() -> bool:
    """Register the chess domain; returns False (no-op) when optional deps are missing.

    Idempotent: when the domain is already registered this is a no-op (the existing
    entry is kept and no duplicate registration log is emitted) and True is returned.
    """
    if not chess_available():
        logger.debug("Chess domain not registered (python-chess/torch unavailable)")
        return False

    from src.games.chess.config import ChessActionSpaceConfig
    from src.games.chess.state import create_initial_state

    if CHESS_DOMAIN not in DomainRegistry.list_domains():
        register_domain(
            CHESS_DOMAIN,
            create_initial_state,
            ChessActionSpaceConfig().total_actions,
            single_agent=False,
            metric=METRIC_WIN_RATE,
        )
        logger.info("Chess domain registered", extra={"domain": CHESS_DOMAIN})
    return True


__all__ = ["CHESS_DOMAIN", "chess_available", "register_chess_domain"]
