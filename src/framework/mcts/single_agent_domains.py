"""Single-agent domain wrappers for neural self-play (Phase 5.2).

The reusable single-agent ``GameState`` implementations in :mod:`game_states`
(``ReasoningState``/``PlanningState``) expose **dict** actions, but
:class:`~src.framework.mcts.neural_mcts.NeuralMCTSNode` keys its children by action and
therefore requires **hashable** actions. :class:`StringActionGameState` adapts any such
dict-action state to hashable string actions so these domains can be driven by
:class:`~src.framework.mcts.neural_mcts.NeuralMCTS` and the
:class:`~src.training.self_play_trainer.SelfPlayTrainer`.

The wrapper is schema-agnostic: it derives a stable string identifier for each legal
action (preferring common id keys, falling back to a canonical repr) and remembers the
mapping back to the original action dict, so ``apply_action``/``action_to_index`` keep
working regardless of the underlying action schema (``ReasoningState`` uses ``"type"``,
``PlanningState`` uses ``"name"``).
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import torch

from src.framework.mcts.game_states import ReasoningState
from src.framework.mcts.neural_mcts import GameState

# Default action identifier keys, in priority order, used to derive a hashable id.
_ID_KEYS = ("type", "name", "action", "id")

# Planning action vocabulary (mirrors PlanningState._get_action_cost). Kept as a named
# constant so the wrapper need not construct a PlanningState at import time (its
# construction pulls settings). "wait"/"finish" are auto-added by the domain and fall in
# the unknown bucket, hence the +1 in the action-space size below.
_PLANNING_ACTIONS = ("analyze", "execute", "verify", "optimize")


class StringActionGameState(GameState):
    """Adapt a dict-action single-agent GameState to hashable string actions."""

    def __init__(self, inner: Any, id_keys: tuple[str, ...] = _ID_KEYS) -> None:
        self._inner = inner
        self._id_keys = id_keys
        self._action_map: dict[str, Any] = {}

    @property
    def inner(self) -> Any:
        """The wrapped domain state (for inspection/metrics)."""
        return self._inner

    def _identifier(self, action: Any) -> str:
        if isinstance(action, dict):
            for key in self._id_keys:
                if key in action:
                    return str(action[key])
            return repr(sorted((k, str(v)) for k, v in action.items()))
        return str(action)

    def get_legal_actions(self) -> list[str]:
        self._action_map = {}
        ids: list[str] = []
        for action in self._inner.get_legal_actions():
            ident = self._identifier(action)
            if ident not in self._action_map:
                self._action_map[ident] = action
                ids.append(ident)
        return ids

    def _original(self, action: str) -> Any:
        if not self._action_map:
            self.get_legal_actions()
        # Fall back to a minimal dict carrying the id under all candidate keys so the
        # wrapped state can still resolve it if the cached map missed.
        return self._action_map.get(action, dict.fromkeys(self._id_keys, action))

    def apply_action(self, action: str) -> StringActionGameState:
        return StringActionGameState(self._inner.apply_action(self._original(action)), self._id_keys)

    def is_terminal(self) -> bool:
        return bool(self._inner.is_terminal())

    def get_reward(self, player: int = 1) -> float:
        return float(self._inner.get_reward(player))

    def to_tensor(self) -> torch.Tensor:
        return self._inner.to_tensor()

    def get_hash(self) -> str:
        return str(self._inner.get_hash())

    def action_to_index(self, action: str) -> int:
        return int(self._inner.action_to_index(self._original(action)))


# Action-space sizes (policy-head width) for the wrapped domains.
# ReasoningState's action types are read from the dataclass field's default_factory WITHOUT
# instantiating it — constructing ReasoningState reads Settings (which require OPENAI_API_KEY
# under the default provider), so import-time instantiation could fail in credential-less envs.
# Planning uses the named vocabulary above (+1 unknown bucket).
_reasoning_action_factory = ReasoningState.__dataclass_fields__["_action_types"].default_factory
if _reasoning_action_factory is MISSING:  # pragma: no cover - the field always defines a factory
    raise RuntimeError("ReasoningState._action_types must define a default_factory")
REASONING_ACTION_SPACE = len(_reasoning_action_factory())
PLANNING_ACTION_SPACE = len(_PLANNING_ACTIONS) + 1


def make_reasoning_state(problem: str = "Solve the problem", **kwargs: Any) -> StringActionGameState:
    """Build a hashable-action reasoning domain initial state."""
    return StringActionGameState(ReasoningState(problem=problem, **kwargs))


def make_planning_state(
    goal: str = "Reach the goal",
    current_state: str = "start",
    **kwargs: Any,
) -> StringActionGameState:
    """Build a hashable-action planning domain initial state.

    Seeds the action vocabulary and a resource budget so the search has non-trivial,
    affordable choices (the bare ``PlanningState`` defaults to an empty vocabulary).
    """
    # Imported lazily: PlanningState construction reads settings.
    from src.framework.mcts.game_states import PlanningState

    params: dict[str, Any] = {
        "goal": goal,
        "current_state": current_state,
        "available_actions": list(_PLANNING_ACTIONS),
        "resources": {"time": 10.0, "compute": 5.0},
    }
    params.update(kwargs)
    return StringActionGameState(PlanningState(**params))


__all__ = [
    "StringActionGameState",
    "REASONING_ACTION_SPACE",
    "PLANNING_ACTION_SPACE",
    "make_reasoning_state",
    "make_planning_state",
]
