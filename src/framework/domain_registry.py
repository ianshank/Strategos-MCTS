"""Dynamic domain registry for neural self-play (Phase 5.2).

Provides config-driven selection of self-play domains (no hard-wired dispatch). Each
domain registers a factory for its initial :class:`~src.framework.mcts.neural_mcts.GameState`,
its policy-head action-space size, whether it is single-agent (non-adversarial), and the
quality metric used to measure decision-quality lift:

- ``"win_rate"`` for adversarial two-player domains (e.g. chess),
- ``"mean_reward"`` for single-agent domains (reasoning/planning), whose non-negative
  reward makes arena win-rate meaningless.

The built-in reasoning and planning domains are registered out of the box. **They are
synthetic smoke-test domains**: their rewards are hand-crafted and trivially exploitable
(see :meth:`~src.framework.mcts.game_states.PlanningState.get_reward`), so lifts measured
on them validate plumbing, not decision quality. Domains with optional dependencies
(chess) are registered lazily on first :meth:`DomainRegistry.get` via
``_LAZY_LOADERS`` — a no-op when the extra isn't installed. Any other domain can
register itself via :func:`register_domain`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib

from src.framework.mcts.neural_mcts import GameState
from src.framework.mcts.single_agent_domains import (
    PLANNING_ACTION_SPACE,
    REASONING_ACTION_SPACE,
    make_planning_state,
    make_reasoning_state,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Supported decision-quality metrics (see module docstring).
METRIC_WIN_RATE = "win_rate"
METRIC_MEAN_REWARD = "mean_reward"
_VALID_METRICS = frozenset({METRIC_WIN_RATE, METRIC_MEAN_REWARD})

# Domains with optional dependencies register lazily on first lookup. Values are
# "module.path:function" strings (imported at call time) so this module never imports
# optional-dependency packages — the loader itself no-ops when its extra is missing.
_LAZY_LOADERS: dict[str, str] = {
    "chess": "src.games.chess.registration:register_chess_domain",
    "connect_four": "src.games.connect_four.registration:register_connect_four_domain",
    "othello": "src.games.othello.registration:register_othello_domain",
}


@dataclass(frozen=True)
class DomainSpec:
    """Registration record for a self-play domain."""

    name: str
    initial_state_fn: Callable[[], GameState]
    action_space_size: int
    single_agent: bool
    metric: str


class DomainRegistry:
    """Process-wide registry of self-play domains (string name -> DomainSpec)."""

    _registry: dict[str, DomainSpec] = {}

    @classmethod
    def register(cls, spec: DomainSpec) -> None:
        if spec.metric not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {sorted(_VALID_METRICS)}, got '{spec.metric}'")
        if spec.action_space_size < 1:
            raise ValueError("action_space_size must be >= 1")
        cls._registry[spec.name] = spec
        logger.debug("Registered self-play domain", extra={"domain": spec.name, "metric": spec.metric})

    @classmethod
    def get(cls, name: str) -> DomainSpec:
        if name not in cls._registry and name in _LAZY_LOADERS:
            module_path, function_name = _LAZY_LOADERS[name].split(":")
            loader: Callable[[], bool] = getattr(importlib.import_module(module_path), function_name)
            loader()  # no-op (returns False) when the domain's optional deps are missing
        if name not in cls._registry:
            optional = sorted(set(_LAZY_LOADERS) - set(cls._registry))
            raise KeyError(
                f"Unknown domain '{name}'. Registered: {sorted(cls._registry)}"
                + (f" (optional, need extras installed: {optional})" if optional else "")
            )
        return cls._registry[name]

    @classmethod
    def get_initial_state(cls, name: str) -> GameState:
        return cls.get(name).initial_state_fn()

    @classmethod
    def action_space_size(cls, name: str) -> int:
        return cls.get(name).action_space_size

    @classmethod
    def is_single_agent(cls, name: str) -> bool:
        return cls.get(name).single_agent

    @classmethod
    def metric(cls, name: str) -> str:
        return cls.get(name).metric

    @classmethod
    def list_domains(cls) -> list[str]:
        """Currently registered domains (lazy domains appear only after a successful ``get``)."""
        return sorted(cls._registry)


def register_domain(
    name: str,
    initial_state_fn: Callable[[], GameState],
    action_space_size: int,
    *,
    single_agent: bool,
    metric: str,
) -> None:
    """Convenience wrapper around :meth:`DomainRegistry.register`."""
    DomainRegistry.register(
        DomainSpec(
            name=name,
            initial_state_fn=initial_state_fn,
            action_space_size=action_space_size,
            single_agent=single_agent,
            metric=metric,
        )
    )


# Register the built-in single-agent domains (the two non-chess M5 domains).
register_domain(
    "reasoning",
    make_reasoning_state,
    REASONING_ACTION_SPACE,
    single_agent=True,
    metric=METRIC_MEAN_REWARD,
)
register_domain(
    "planning",
    make_planning_state,
    PLANNING_ACTION_SPACE,
    single_agent=True,
    metric=METRIC_MEAN_REWARD,
)


__all__ = [
    "DomainSpec",
    "DomainRegistry",
    "register_domain",
    "METRIC_WIN_RATE",
    "METRIC_MEAN_REWARD",
]
