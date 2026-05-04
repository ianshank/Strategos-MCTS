"""
Model-specific prompt/decoding presets, looked up by name pattern.

Each :class:`ModelPreset` records quirks for a particular family of models
(e.g. chat-template stop tokens, reasoning-mode flag, default temperature).
Presets are matched against a model name using a regex (``name_pattern``),
which keeps adapter code provider-agnostic: agents look up a preset by
model name, never by hardcoded checks like ``if model == "phi-4": ...``.

The registry is process-wide and additive. ``register_preset`` is the only
mutator and rejects duplicate ``name`` values unless ``override=True``.
``clear_registry`` is exposed for tests; production callers should never use
it. To restore built-ins after clearing, ``importlib.reload`` this module.

Patterns are compiled with :data:`re.IGNORECASE` so callers do not need to
add ``(?i)`` themselves, although they MAY for clarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelPreset:
    """
    A set of model-specific request-shaping defaults.

    Attributes:
        name: Human-readable identifier (e.g. ``"phi4-reasoning"``). Must be
            unique within the registry.
        name_pattern: Regex matched against the model name passed to
            :func:`get_preset`. Compiled case-insensitively.
        stop_tokens: Tuple of stop sequences to merge into the request.
        reasoning: ``True`` if the model honors a ``reasoning`` block on
            requests; consumers should only forward ``reasoning_effort`` when
            this is set.
        default_temperature: Optional default temperature applied when the
            caller does not specify one. ``None`` means "no opinion".
        extra_params: Free-form extra parameters merged into the payload at
            *lower* precedence than caller-supplied keys.
    """

    name: str
    name_pattern: str
    stop_tokens: tuple[str, ...] = ()
    reasoning: bool = False
    default_temperature: float | None = None
    extra_params: dict[str, object] = field(default_factory=dict)

    def matches(self, model_name: str | None) -> bool:
        """Return ``True`` if ``model_name`` matches this preset's pattern.

        Empty or ``None`` model names never match. Matching uses
        :func:`re.search` with :data:`re.IGNORECASE`, so the pattern matches
        anywhere within ``model_name``.
        """
        if not model_name:
            return False
        return re.search(self.name_pattern, model_name, re.IGNORECASE) is not None


# Module-level registry. Insertion order is preserved (Python 3.7+ dict),
# which makes "first match wins" semantics deterministic.
_PRESET_REGISTRY: dict[str, ModelPreset] = {}


def register_preset(preset: ModelPreset, *, override: bool = False) -> None:
    """
    Register a preset under its ``name``.

    Args:
        preset: The preset to register.
        override: When ``True``, replace any existing preset with the same
            name. When ``False`` (default), raise ``ValueError`` on collision.

    Raises:
        ValueError: If a preset with ``preset.name`` is already registered
            and ``override`` is ``False``.
    """
    if preset.name in _PRESET_REGISTRY and not override:
        raise ValueError(f"Preset '{preset.name}' is already registered. Use override=True to replace it.")
    _PRESET_REGISTRY[preset.name] = preset


def get_preset(model_name: str | None) -> ModelPreset | None:
    """
    Return the first registered preset whose pattern matches ``model_name``.

    Args:
        model_name: Model identifier (e.g. ``"phi-4-reasoning-q4"``). May be
            ``None`` or empty, in which case ``None`` is returned.

    Returns:
        The first matching preset, or ``None`` if no preset matches.
    """
    if not model_name:
        return None
    for preset in _PRESET_REGISTRY.values():
        if preset.matches(model_name):
            return preset
    return None


def get_preset_by_name(name: str | None) -> ModelPreset | None:
    """
    Look up a preset by its ``name`` (not by pattern).

    Args:
        name: The preset's registered ``name``. ``None`` or empty returns
            ``None``.

    Returns:
        The preset with that exact name, or ``None`` if none is registered.
    """
    if not name:
        return None
    return _PRESET_REGISTRY.get(name)


def list_presets() -> list[str]:
    """Return the names of all registered presets, in insertion order."""
    return list(_PRESET_REGISTRY.keys())


def clear_registry() -> None:
    """
    Drop all registered presets.

    Test-only helper. To restore the module's built-in presets after
    clearing, callers should ``importlib.reload(src.adapters.llm.model_presets)``.
    """
    _PRESET_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

#: Phi-4 reasoning model preset.
#:
#: Matches any model name containing ``phi-4``, ``phi_4``, ``phi 4``, or
#: ``phi4`` (case-insensitive). Phi-4 reasoning models emit ``<|im_end|>`` as
#: a chat-turn terminator and benefit from a low default temperature.
PHI4_REASONING = ModelPreset(
    name="phi4-reasoning",
    name_pattern=r"phi[-_ ]?4",
    stop_tokens=("<|im_end|>",),
    reasoning=True,
    default_temperature=0.2,
)

register_preset(PHI4_REASONING)


__all__ = [
    "ModelPreset",
    "ReasoningEffort",
    "PHI4_REASONING",
    "register_preset",
    "get_preset",
    "get_preset_by_name",
    "list_presets",
    "clear_registry",
]
