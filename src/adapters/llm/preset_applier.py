"""
Apply a :class:`ModelPreset` to an OpenAI-compatible request payload.

The :func:`apply_preset` function is a pure transformation: it takes a
payload dict and a preset (either of which may be ``None``), and returns a
*new* dict with preset-derived fields merged in. Caller-supplied values
always take precedence over preset defaults — the preset never clobbers an
explicit setting.

Merge rules (see :func:`apply_preset` docstring for full detail):

* **stop**: union of ``preset.stop_tokens`` and any existing ``payload["stop"]``,
  preserving order with duplicates removed.
* **temperature**: only applied if the caller did not provide ``user_temperature``
  AND the payload does not already carry a non-``None`` ``temperature``.
* **reasoning**: only applied when the preset has ``reasoning=True`` and a
  ``reasoning_effort`` argument is supplied.
* **extra_params**: merged with *lower* precedence than the input payload.
"""

from __future__ import annotations

import logging
from typing import Any

from src.adapters.llm.model_presets import ModelPreset, ReasoningEffort

__all__ = ["apply_preset"]

_logger = logging.getLogger(__name__)


def _union_stops(
    payload_stops: Any,
    preset_stops: tuple[str, ...],
) -> list[str]:
    """Return ordered union of preset stops and existing payload stops, no duplicates."""
    combined: list[str] = []
    seen: set[str] = set()

    # Preset stops are added first so callers can rely on their presence
    # at the front of the list when ordering matters (most servers do not
    # care, but being deterministic keeps tests predictable).
    for s in preset_stops:
        if s not in seen:
            combined.append(s)
            seen.add(s)

    if payload_stops is None:
        return combined

    # Accept either a list or any iterable of strings; tolerate single string.
    if isinstance(payload_stops, str):
        iterable: list[str] = [payload_stops]
    else:
        iterable = list(payload_stops)

    for s in iterable:
        if s not in seen:
            combined.append(s)
            seen.add(s)

    return combined


def apply_preset(
    payload: dict[str, Any],
    preset: ModelPreset | None,
    *,
    reasoning_effort: ReasoningEffort | None = None,
    user_temperature: float | None = None,
) -> dict[str, Any]:
    """
    Return a new payload dict with preset-derived fields merged in.

    Args:
        payload: The base OpenAI-compatible request dict (e.g. with
            ``model``, ``messages``, ``temperature``, ...). NOT mutated.
        preset: The preset to apply, or ``None`` for a passthrough copy.
        reasoning_effort: Optional ``low`` / ``medium`` / ``high`` hint. Only
            applied when the preset has ``reasoning=True``.
        user_temperature: Temperature explicitly chosen by the caller. When
            not ``None``, the preset's ``default_temperature`` is ignored.

    Returns:
        A *new* dict (shallow copy of ``payload``) with merged values.
    """
    # Always start from a shallow copy so we never mutate the caller's dict.
    result: dict[str, Any] = dict(payload)

    if preset is None:
        _logger.debug("apply_preset: no preset supplied; returning passthrough copy")
        return result

    applied: list[str] = []

    # ------------------------------------------------------------------
    # Extra params: lowest precedence — don't overwrite existing keys.
    # ------------------------------------------------------------------
    for key, value in preset.extra_params.items():
        if key not in result:
            result[key] = value
            applied.append(f"extra:{key}")

    # ------------------------------------------------------------------
    # Stop tokens: union, dedupe, preserve order (preset first).
    # ------------------------------------------------------------------
    if preset.stop_tokens:
        merged_stops = _union_stops(result.get("stop"), preset.stop_tokens)
        if merged_stops:
            result["stop"] = merged_stops
            applied.append("stop")

    # ------------------------------------------------------------------
    # Temperature: only apply preset default when caller did not specify
    # AND payload doesn't already carry a non-None temperature.
    # ------------------------------------------------------------------
    if user_temperature is None and preset.default_temperature is not None and result.get("temperature") is None:
        result["temperature"] = preset.default_temperature
        applied.append("temperature")

    # ------------------------------------------------------------------
    # Reasoning: forward only when both the preset and caller agree.
    # ------------------------------------------------------------------
    if preset.reasoning and reasoning_effort is not None:
        result["reasoning"] = {"effort": reasoning_effort}
        applied.append("reasoning")

    _logger.debug(
        "apply_preset: preset=%s applied=%s user_temperature=%s reasoning_effort=%s",
        preset.name,
        applied or ["<none>"],
        user_temperature,
        reasoning_effort,
    )
    return result
