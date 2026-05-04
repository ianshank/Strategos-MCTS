"""
Tests for ``LLMClientFactory`` preset auto-attachment.

When the factory creates an LM Studio client, it should resolve a preset
either from explicit kwargs, from a settings override, or from regex
matching against the model name. For other providers the factory must be
unaffected by preset-related settings (no regression).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.adapters.llm.lmstudio_client import LMStudioClient
from src.adapters.llm.model_presets import PHI4_REASONING, ModelPreset
from src.framework.factories import LLMClientFactory


def _mock_settings(**overrides):
    """Return a MagicMock with the attributes the factory needs."""
    s = MagicMock()
    s.LLM_PROVIDER = "lmstudio"
    s.HTTP_TIMEOUT_SECONDS = 60.0
    s.HTTP_MAX_RETRIES = 3
    s.LMSTUDIO_PRESET = None
    s.LMSTUDIO_REASONING_EFFORT = None
    s.LMSTUDIO_MODEL = None
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


@pytest.mark.unit
class TestLMStudioFactoryPreset:
    """When provider == 'lmstudio', the factory should auto-attach a preset."""

    def test_factory_attaches_preset_when_model_matches_phi4_pattern(self):
        settings = _mock_settings(
            LLM_PROVIDER="lmstudio",
            LMSTUDIO_MODEL="phi-4-reasoning-q4",
        )
        factory = LLMClientFactory(settings=settings)

        client = factory.create(model="phi-4-reasoning-q4")

        assert isinstance(client, LMStudioClient)
        assert client._preset is not None
        assert client._preset.name == "phi4-reasoning"

    def test_factory_uses_explicit_preset_name_override(self):
        # Settings override takes precedence over auto-detection.
        settings = _mock_settings(
            LLM_PROVIDER="lmstudio",
            LMSTUDIO_PRESET="phi4-reasoning",
            LMSTUDIO_MODEL="totally-unrelated-model",
        )
        factory = LLMClientFactory(settings=settings)

        client = factory.create(model="totally-unrelated-model")

        assert isinstance(client, LMStudioClient)
        # Even though the model name does not match phi-4 pattern, the
        # explicit override pulls in the phi4-reasoning preset.
        assert client._preset is PHI4_REASONING

    def test_explicit_preset_kwarg_beats_settings_and_auto_detect(self):
        custom = ModelPreset(name="custom", name_pattern=r"custom", stop_tokens=("X",))
        settings = _mock_settings(
            LLM_PROVIDER="lmstudio",
            LMSTUDIO_PRESET="phi4-reasoning",
            LMSTUDIO_MODEL="phi-4",
        )
        factory = LLMClientFactory(settings=settings)

        client = factory.create(model="phi-4", preset=custom)

        assert isinstance(client, LMStudioClient)
        assert client._preset is custom

    def test_factory_attaches_reasoning_effort_from_settings(self):
        settings = _mock_settings(
            LLM_PROVIDER="lmstudio",
            LMSTUDIO_MODEL="phi-4",
            LMSTUDIO_REASONING_EFFORT="high",
        )
        factory = LLMClientFactory(settings=settings)

        client = factory.create(model="phi-4")

        assert isinstance(client, LMStudioClient)
        assert client._reasoning_effort == "high"

    def test_factory_no_preset_when_model_does_not_match(self):
        settings = _mock_settings(
            LLM_PROVIDER="lmstudio",
            LMSTUDIO_MODEL="random-model",
        )
        factory = LLMClientFactory(settings=settings)

        client = factory.create(model="random-model")

        assert isinstance(client, LMStudioClient)
        assert client._preset is None

    def test_unknown_preset_name_override_falls_back_to_auto_detect(self):
        # Settings override pointing to a non-existent name must NOT raise;
        # the factory should silently fall through to auto-detection.
        settings = _mock_settings(
            LLM_PROVIDER="lmstudio",
            LMSTUDIO_PRESET="does-not-exist",
            LMSTUDIO_MODEL="phi-4",
        )
        factory = LLMClientFactory(settings=settings)

        client = factory.create(model="phi-4")

        assert client._preset is PHI4_REASONING


@pytest.mark.unit
class TestNonLMStudioProviderUnaffected:
    """Other providers must not see preset / reasoning_effort kwargs."""

    @patch("src.adapters.llm.create_client")
    def test_non_lmstudio_provider_unaffected_by_preset_settings(self, mock_create):
        mock_create.return_value = MagicMock()
        settings = _mock_settings(
            LLM_PROVIDER="openai",
            LMSTUDIO_PRESET="phi4-reasoning",
            LMSTUDIO_REASONING_EFFORT="high",
            LMSTUDIO_MODEL="phi-4",
        )
        factory = LLMClientFactory(settings=settings)

        factory.create(provider="openai", model="gpt-4")

        # Inspect the kwargs forwarded to create_client.
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert "preset" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs

    @patch("src.adapters.llm.create_client")
    def test_anthropic_provider_unaffected(self, mock_create):
        mock_create.return_value = MagicMock()
        settings = _mock_settings(
            LLM_PROVIDER="anthropic",
            LMSTUDIO_PRESET="phi4-reasoning",
        )
        factory = LLMClientFactory(settings=settings)

        factory.create(provider="anthropic", model="claude-3-opus-20240229")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["provider"] == "anthropic"
        assert "preset" not in call_kwargs
        assert "reasoning_effort" not in call_kwargs


# ---------------------------------------------------------------------------
# Logging contract: factory must emit INFO when a preset is auto-attached.
# ---------------------------------------------------------------------------


def test_factory_logs_info_when_preset_auto_detected(caplog: pytest.LogCaptureFixture) -> None:
    import logging as _logging

    settings = MagicMock()
    settings.LLM_PROVIDER = "lmstudio"
    settings.LMSTUDIO_PRESET = None
    settings.LMSTUDIO_REASONING_EFFORT = None
    settings.HTTP_TIMEOUT_SECONDS = 30
    settings.HTTP_MAX_RETRIES = 3

    factory = LLMClientFactory(settings=settings)
    with patch("src.adapters.llm.create_client") as mock_create:
        mock_create.return_value = MagicMock(spec=LMStudioClient)
        with caplog.at_level(_logging.INFO, logger="src.framework.factories"):
            factory.create(model="phi-4-reasoning-q4")

    messages = [r.getMessage() for r in caplog.records if r.name == "src.framework.factories"]
    assert any("preset attached" in m and "phi4" in m for m in messages)


def test_factory_logs_debug_when_no_preset_match(caplog: pytest.LogCaptureFixture) -> None:
    import logging as _logging

    settings = MagicMock()
    settings.LLM_PROVIDER = "lmstudio"
    settings.LMSTUDIO_PRESET = None
    settings.LMSTUDIO_REASONING_EFFORT = None
    settings.HTTP_TIMEOUT_SECONDS = 30
    settings.HTTP_MAX_RETRIES = 3

    factory = LLMClientFactory(settings=settings)
    with patch("src.adapters.llm.create_client") as mock_create:
        mock_create.return_value = MagicMock(spec=LMStudioClient)
        with caplog.at_level(_logging.DEBUG, logger="src.framework.factories"):
            factory.create(model="some-unknown-model")

    messages = [r.getMessage() for r in caplog.records if r.name == "src.framework.factories"]
    assert any("preset not attached" in m for m in messages)
