"""
Tests for the LM Studio preset / reasoning / temperature settings fields.

These tests verify that the new ``LMSTUDIO_PRESET``,
``LMSTUDIO_REASONING_EFFORT``, and ``LMSTUDIO_TEMPERATURE`` fields on
:class:`Settings` are optional, default to ``None``, and validate inputs.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


def _make_settings(**overrides):
    """Build Settings with a minimal valid base configuration."""
    env = {
        "OPENAI_API_KEY": "sk-" + "a" * 50,
        "LLM_PROVIDER": "openai",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    with patch.dict(os.environ, env, clear=False):
        from src.config.settings import Settings

        return Settings()


@pytest.mark.unit
class TestLMStudioPresetField:
    """`LMSTUDIO_PRESET` is optional and defaults to None."""

    def test_lmstudio_preset_field_optional_default_none(self):
        # Don't provide the env var; it should default to None.
        env = {
            "OPENAI_API_KEY": "sk-" + "a" * 50,
            "LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings()
            assert s.LMSTUDIO_PRESET is None

    def test_lmstudio_preset_accepts_string_value(self):
        s = _make_settings(LMSTUDIO_PRESET="phi4-reasoning")
        assert s.LMSTUDIO_PRESET == "phi4-reasoning"


@pytest.mark.unit
class TestLMStudioReasoningEffortField:
    """`LMSTUDIO_REASONING_EFFORT` accepts only the documented literals."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_lmstudio_reasoning_effort_accepts_valid_literal(self, effort):
        s = _make_settings(LMSTUDIO_REASONING_EFFORT=effort)
        assert effort == s.LMSTUDIO_REASONING_EFFORT

    def test_lmstudio_reasoning_effort_validates_literal(self):
        with pytest.raises(ValidationError):
            _make_settings(LMSTUDIO_REASONING_EFFORT="extreme")

    def test_lmstudio_reasoning_effort_default_none(self):
        env = {
            "OPENAI_API_KEY": "sk-" + "a" * 50,
            "LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings()
            assert s.LMSTUDIO_REASONING_EFFORT is None


@pytest.mark.unit
class TestLMStudioTemperatureField:
    """`LMSTUDIO_TEMPERATURE` is optional, validated to [0, 2]."""

    def test_lmstudio_temperature_default_none(self):
        env = {
            "OPENAI_API_KEY": "sk-" + "a" * 50,
            "LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings()
            assert s.LMSTUDIO_TEMPERATURE is None

    def test_lmstudio_temperature_accepts_in_range(self):
        s = _make_settings(LMSTUDIO_TEMPERATURE=0.4)
        assert pytest.approx(0.4) == s.LMSTUDIO_TEMPERATURE

    def test_lmstudio_temperature_validates_range_above(self):
        with pytest.raises(ValidationError):
            _make_settings(LMSTUDIO_TEMPERATURE=3.0)

    def test_lmstudio_temperature_validates_range_below(self):
        with pytest.raises(ValidationError):
            _make_settings(LMSTUDIO_TEMPERATURE=-0.5)
