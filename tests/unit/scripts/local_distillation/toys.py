"""Tiny two-player GameState and a counting policy/value net for distillation tests."""

from __future__ import annotations

from scripts.local_distillation.toy_domain import (
    ACTION_SPACE,
    ACTIONS,
    BiasedNet,
    CountingNet,
    TwoPlyState,
)

__all__ = ["ACTION_SPACE", "ACTIONS", "BiasedNet", "CountingNet", "TwoPlyState"]
