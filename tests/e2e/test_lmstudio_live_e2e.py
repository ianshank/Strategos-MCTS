"""Live LM Studio adapter QA. Skipped unless REQUIRE_LMSTUDIO=1.

Not a distillation teacher. Distillation stays NeuralMCTS-only.
"""

from __future__ import annotations

import os

import pytest

from src.adapters.llm.lmstudio_client import LMStudioClient

pytestmark = [pytest.mark.e2e, pytest.mark.live]


def _live_client() -> LMStudioClient:
    return LMStudioClient(
        model=os.environ.get("LMSTUDIO_MODEL", "nvidia/nemotron-3-nano-omni"),
        base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        timeout=float(os.environ.get("LMSTUDIO_TIMEOUT", "300")),
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_lmstudio_health_and_model_id_present() -> None:
    client = _live_client()
    try:
        healthy = await client.check_health()
        assert healthy, "LM Studio /v1/models did not return HTTP 200"
        models = await client.list_models()
        ids = [str(item.get("id", "")) for item in models if item.get("id")]
        assert ids, "LM Studio listed no model ids"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_lmstudio_generate_twice_at_temperature_zero() -> None:
    client = _live_client()
    try:
        first = await client.generate(
            prompt="Reply with the single word ping.",
            temperature=0.0,
            max_tokens=128,
        )
        second = await client.generate(
            prompt="Reply with the single word ping.",
            temperature=0.0,
            max_tokens=128,
        )
        assert first.text.strip(), "first generate returned empty text"
        assert second.text.strip(), "second generate returned empty text"
    finally:
        await client.close()
