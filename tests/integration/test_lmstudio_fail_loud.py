"""LM Studio adapter fails loud on a closed port (no mock fallback)."""

from __future__ import annotations

import pytest

from src.adapters.llm.exceptions import LLMConnectionError, LLMTimeoutError
from src.adapters.llm.lmstudio_client import LMStudioClient

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


@pytest.mark.asyncio
async def test_closed_port_raises_connection_error_without_mock(unused_tcp_port: int) -> None:
    client = LMStudioClient(
        model="unused-model",
        base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
        timeout=1.0,
        max_retries=1,
    )
    try:
        with pytest.raises((LLMConnectionError, LLMTimeoutError)):
            await client.generate(prompt="ping", max_tokens=1)
    finally:
        await client.close()
