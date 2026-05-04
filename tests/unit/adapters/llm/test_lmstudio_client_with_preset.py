"""
Tests verifying the LMStudioClient honors :class:`ModelPreset` configuration.

The tests use a small mock for the underlying ``httpx.AsyncClient`` so we
can capture the JSON payload sent to the server and assert that
preset-derived fields (stop tokens, reasoning hint, default temperature)
appear correctly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.adapters.llm.lmstudio_client import LMStudioClient
from src.adapters.llm.model_presets import ModelPreset
from src.config.constants import DEFAULT_LMSTUDIO_TEMPERATURE


def _wire_post_mock(client: LMStudioClient, response_body: dict) -> AsyncMock:
    """Attach a stub ``httpx.AsyncClient`` to ``client`` and return its post mock."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_body

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    mock_http.is_closed = False
    client._client = mock_http
    return mock_http.post


@pytest.fixture
def mock_success_response() -> dict:
    """Standard 200 chat-completion body."""
    return {
        "id": "chatcmpl-local-123",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.unit
class TestBackwardsCompatibility:
    """The new preset / reasoning_effort kwargs must not break old callers."""

    def test_init_signature_unchanged_when_preset_omitted(self):
        client = LMStudioClient()
        # New attributes default to None.
        assert client._preset is None
        assert client._reasoning_effort is None
        # Pre-existing attributes still work.
        assert client.model == "local-model"
        assert client.api_key == "not-required"

    @pytest.mark.asyncio
    async def test_generate_without_temperature_kwarg_still_works(self, mock_success_response):
        """Regression: callers may invoke ``generate(prompt='...')`` with no temp."""
        client = LMStudioClient(model="any-model")
        post_mock = _wire_post_mock(client, mock_success_response)

        result = await client.generate(prompt="hi")
        assert result.text == "ok"
        assert post_mock.call_count == 1

        payload = post_mock.call_args.kwargs["json"]
        # The wire payload must always carry a numeric temperature.
        assert isinstance(payload["temperature"], float)


@pytest.mark.unit
class TestPresetAppliedToPayload:
    """Verify that a preset attached at __init__ flows through to the request."""

    @pytest.mark.asyncio
    async def test_payload_includes_preset_stops(self, mock_success_response):
        preset = ModelPreset(
            name="custom",
            name_pattern=r"custom",
            stop_tokens=("<|im_end|>",),
        )
        client = LMStudioClient(preset=preset, model="some-model")
        post_mock = _wire_post_mock(client, mock_success_response)

        await client.generate(prompt="hi")
        payload = post_mock.call_args.kwargs["json"]

        assert "<|im_end|>" in payload["stop"]

    @pytest.mark.asyncio
    async def test_payload_includes_reasoning_param_when_preset_reasoning_true_and_effort_set(
        self, mock_success_response
    ):
        preset = ModelPreset(name="r", name_pattern=r"r", reasoning=True)
        client = LMStudioClient(preset=preset, reasoning_effort="medium")
        post_mock = _wire_post_mock(client, mock_success_response)

        await client.generate(prompt="hi")
        payload = post_mock.call_args.kwargs["json"]

        assert payload["reasoning"] == {"effort": "medium"}

    @pytest.mark.asyncio
    async def test_payload_omits_reasoning_when_preset_reasoning_false(self, mock_success_response):
        preset = ModelPreset(name="r", name_pattern=r"r", reasoning=False)
        client = LMStudioClient(preset=preset, reasoning_effort="medium")
        post_mock = _wire_post_mock(client, mock_success_response)

        await client.generate(prompt="hi")
        payload = post_mock.call_args.kwargs["json"]

        assert "reasoning" not in payload


@pytest.mark.unit
class TestTemperatureFallback:
    """Default temperature resolution rules."""

    @pytest.mark.asyncio
    async def test_default_temperature_falls_back_to_constant_when_no_preset_or_setting(self, mock_success_response):
        client = LMStudioClient()  # no preset
        post_mock = _wire_post_mock(client, mock_success_response)

        await client.generate(prompt="hi")
        payload = post_mock.call_args.kwargs["json"]

        assert payload["temperature"] == pytest.approx(DEFAULT_LMSTUDIO_TEMPERATURE)

    @pytest.mark.asyncio
    async def test_explicit_user_temperature_is_respected_over_preset(self, mock_success_response):
        preset = ModelPreset(
            name="t",
            name_pattern=r"t",
            default_temperature=0.2,
        )
        client = LMStudioClient(preset=preset)
        post_mock = _wire_post_mock(client, mock_success_response)

        await client.generate(prompt="hi", temperature=0.9)
        payload = post_mock.call_args.kwargs["json"]

        assert payload["temperature"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_preset_temperature_used_when_caller_omits_it(self, mock_success_response):
        preset = ModelPreset(
            name="t",
            name_pattern=r"t",
            default_temperature=0.25,
        )
        client = LMStudioClient(preset=preset)
        post_mock = _wire_post_mock(client, mock_success_response)

        await client.generate(prompt="hi")  # no temperature kwarg
        payload = post_mock.call_args.kwargs["json"]

        assert payload["temperature"] == pytest.approx(0.25)


@pytest.mark.unit
class TestStreamingPresetIntegration:
    """Streaming generation must apply the preset just like non-streaming."""

    @pytest.mark.asyncio
    async def test_stream_payload_includes_preset_stops_and_reasoning(self):
        # Build a streaming mock that yields a single [DONE].
        from unittest.mock import AsyncMock as _AsyncMock

        import httpx

        preset = ModelPreset(
            name="t",
            name_pattern=r"t",
            stop_tokens=("<|im_end|>",),
            reasoning=True,
        )
        client = LMStudioClient(preset=preset, reasoning_effort="low")

        mock_response = _AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            for line in ["data: [DONE]"]:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        async def mock_aread():
            return b""

        mock_response.aread = mock_aread

        mock_http = _AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False

        stream_cm = _AsyncMock()
        stream_cm.__aenter__ = _AsyncMock(return_value=mock_response)
        stream_cm.__aexit__ = _AsyncMock(return_value=False)
        mock_http.stream.return_value = stream_cm
        client._client = mock_http

        result = await client.generate(prompt="hi", stream=True)
        # Drain so that ``stream()`` is actually called.
        _ = [chunk async for chunk in result]

        payload = mock_http.stream.call_args.kwargs["json"]
        assert "<|im_end|>" in payload["stop"]
        assert payload["reasoning"] == {"effort": "low"}
        # Always numeric — never None on the wire.
        assert isinstance(payload["temperature"], float)
