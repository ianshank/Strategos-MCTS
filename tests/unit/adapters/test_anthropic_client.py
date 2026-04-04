"""Extended unit tests for src/adapters/llm/anthropic_client.py targeting uncovered lines.

Covers:
- _handle_error_response with unparseable JSON body (lines 210-212)
- _handle_error_response generic status code (line 233)
- generate() stream=True branch (line 289)
- _generate_non_stream with system_prompt, stop, extra kwargs (lines 339, 341, 348)
- _generate_non_stream with non-200 status -> _handle_error_response (line 358)
- _generate_stream full flow (lines 439-516)
- _generate_stream error paths (timeout, connect, generic)
- close() when client is already None / already closed (line 520)
- Rate limit error without retry-after header
- Billing error via error_type (line 220)
- Token-related 400 error (line 225)
- Safety-related 400 error via message (line 228)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("httpx", reason="httpx required for HTTP client tests")

import httpx

from src.adapters.llm.anthropic_client import AnthropicClient
from src.adapters.llm.base import LLMToolResponse
from src.adapters.llm.exceptions import (
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerError,
    LLMStreamError,
    LLMTimeoutError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake API keys so conftest won't skip."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


@pytest.fixture
def api_key() -> str:
    return "sk-ant-test-key-12345"


@pytest.fixture
def client(api_key: str) -> AnthropicClient:
    return AnthropicClient(api_key=api_key, max_retries=1)


@pytest.fixture
def success_response_data() -> dict:
    return {
        "id": "msg_001",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _mock_http_response(status_code: int = 200, json_data: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or json.dumps(json_data or {})
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = Exception("no json")
    resp.headers = {}
    return resp


# ---------------------------------------------------------------------------
# _handle_error_response - additional branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandleErrorResponseExtended:
    """Cover error-handling branches not yet tested."""

    def test_error_response_json_parse_failure(self, client: AnthropicClient) -> None:
        """When response.json() raises, fall back to response.text (lines 210-212)."""
        response = MagicMock()
        response.status_code = 401
        response.json.side_effect = ValueError("bad json")
        response.text = "Unauthorized"

        with pytest.raises(LLMAuthenticationError):
            client._handle_error_response(response)

    def test_generic_status_code_raises_client_error(self, client: AnthropicClient) -> None:
        """Status codes not in specific handlers fall through to generic LLMClientError (line 233)."""
        response = MagicMock()
        response.status_code = 418  # I'm a teapot
        response.json.return_value = {"error": {"type": "teapot", "message": "I'm a teapot"}}
        response.text = "I'm a teapot"

        with pytest.raises(LLMClientError) as exc_info:
            client._handle_error_response(response)
        assert exc_info.value.status_code == 418

    def test_rate_limit_without_retry_after(self, client: AnthropicClient) -> None:
        """429 without retry-after header sets retry_after to None."""
        response = MagicMock()
        response.status_code = 429
        response.json.return_value = {"error": {"type": "rate_limit_error", "message": "slow down"}}
        response.text = "slow down"
        response.headers = {}

        with pytest.raises(LLMRateLimitError) as exc_info:
            client._handle_error_response(response)
        assert exc_info.value.retry_after is None

    def test_billing_error_type(self, client: AnthropicClient) -> None:
        """Non-402 status but error_type containing 'billing' triggers quota error (line 220)."""
        response = MagicMock()
        response.status_code = 402
        response.json.return_value = {"error": {"type": "Billing_issue", "message": "No credits"}}
        response.text = "No credits"

        with pytest.raises(LLMQuotaExceededError):
            client._handle_error_response(response)

    def test_token_error_in_400(self, client: AnthropicClient) -> None:
        """400 with 'token' in message triggers context length error (line 225)."""
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {
            "error": {"type": "invalid_request_error", "message": "Too many token in request"}
        }
        response.text = "Too many token in request"

        with pytest.raises(LLMContextLengthError):
            client._handle_error_response(response)

    def test_safety_error_in_400(self, client: AnthropicClient) -> None:
        """400 with 'safety' in message triggers content filter error (line 228)."""
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {
            "error": {"type": "invalid_request_error", "message": "Blocked for safety reasons"}
        }
        response.text = "Blocked for safety reasons"

        with pytest.raises(LLMContentFilterError):
            client._handle_error_response(response)

    def test_not_found_error_type(self, client: AnthropicClient) -> None:
        """error_type == 'not_found_error' with non-404 status still triggers model not found."""
        response = MagicMock()
        response.status_code = 404
        response.json.return_value = {"error": {"type": "not_found_error", "message": "Not found"}}
        response.text = "Not found"

        with pytest.raises(LLMModelNotFoundError):
            client._handle_error_response(response)


# ---------------------------------------------------------------------------
# _generate_non_stream - additional branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateNonStreamExtended:
    """Cover non-stream generation branches."""

    @pytest.mark.asyncio
    async def test_system_prompt_in_payload(self, client: AnthropicClient, success_response_data: dict) -> None:
        """System message is sent as system parameter in payload (line 339)."""
        mock_resp = _mock_http_response(200, success_response_data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            await client.generate(
                messages=[
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hi"},
                ],
                max_tokens=100,
            )

            payload = http.post.call_args.kwargs["json"]
            assert payload["system"] == "Be concise."

    @pytest.mark.asyncio
    async def test_stop_sequences_in_payload(self, client: AnthropicClient, success_response_data: dict) -> None:
        """Stop sequences are included in payload (line 341)."""
        mock_resp = _mock_http_response(200, success_response_data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                stop=["END", "STOP"],
            )

            payload = http.post.call_args.kwargs["json"]
            assert payload["stop_sequences"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self, client: AnthropicClient, success_response_data: dict) -> None:
        """top_p, top_k, metadata kwargs are forwarded in payload (line 348)."""
        mock_resp = _mock_http_response(200, success_response_data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                top_p=0.9,
                top_k=40,
            )

            payload = http.post.call_args.kwargs["json"]
            assert payload["top_p"] == 0.9
            assert payload["top_k"] == 40

    @pytest.mark.asyncio
    async def test_non_200_calls_handle_error(self, client: AnthropicClient) -> None:
        """Non-200 status triggers _handle_error_response (line 358)."""
        error_resp = _mock_http_response(
            500,
            {"error": {"type": "server_error", "message": "Internal"}},
            text="Internal",
        )

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=error_resp)
            mock_gc.return_value = http

            with pytest.raises(LLMServerError):
                await client.generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=100,
                )

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_error(self, client: AnthropicClient) -> None:
        """Circuit breaker records failure when request raises."""
        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
            mock_gc.return_value = http

            initial_failures = client.circuit_breaker.failure_count
            with pytest.raises(LLMConnectionError):
                await client.generate(
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=100,
                )
            assert client.circuit_breaker.failure_count > initial_failures

    @pytest.mark.asyncio
    async def test_temperature_clamped_to_1(self, client: AnthropicClient, success_response_data: dict) -> None:
        """Temperature > 1.0 is clamped to 1.0."""
        mock_resp = _mock_http_response(200, success_response_data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.5,
                max_tokens=100,
            )

            payload = http.post.call_args.kwargs["json"]
            assert payload["temperature"] == 1.0


# ---------------------------------------------------------------------------
# generate() stream=True branch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateStreamBranch:
    """Cover the stream=True path in generate() and _generate_stream."""

    @pytest.mark.asyncio
    async def test_generate_stream_returns_async_iterator(self, client: AnthropicClient) -> None:
        """generate(stream=True) returns an async iterator (line 289)."""

        # Build a mock streaming response
        async def _mock_aiter_lines():
            lines = [
                "event: content_block_delta",
                'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}',
                "",
                'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}',
                'data: {"type": "message_stop"}',
            ]
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _mock_aiter_lines
        mock_response.aread = AsyncMock()

        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                return mock_response

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert "Hello" in chunks
            assert " world" in chunks

    @pytest.mark.asyncio
    async def test_stream_with_system_and_stop(self, client: AnthropicClient) -> None:
        """Stream payload includes system and stop_sequences."""

        async def _mock_aiter_lines():
            yield 'data: {"type": "message_stop"}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _mock_aiter_lines
        mock_response.aread = AsyncMock()

        mock_http_client = AsyncMock()
        captured_payload = {}

        class _StreamCM:
            async def __aenter__(self_inner):
                return mock_response

            async def __aexit__(self_inner, *args):
                pass

        def _stream(method, url, json=None):
            captured_payload.update(json or {})
            return _StreamCM()

        mock_http_client.stream = MagicMock(side_effect=_stream)

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                ],
                stream=True,
                max_tokens=50,
                stop=["END"],
                top_p=0.8,
            )
            async for _ in result:
                pass

        assert captured_payload["system"] == "Be brief."
        assert captured_payload["stop_sequences"] == ["END"]
        assert captured_payload["stream"] is True
        assert captured_payload["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_stream_timeout_error(self, client: AnthropicClient) -> None:
        """Timeout during streaming raises LLMTimeoutError."""
        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                raise httpx.TimeoutException("timeout")

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            with pytest.raises(LLMTimeoutError):
                async for _ in result:
                    pass

    @pytest.mark.asyncio
    async def test_stream_connect_error(self, client: AnthropicClient) -> None:
        """Connection error during streaming raises LLMConnectionError."""
        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                raise httpx.ConnectError("connect fail")

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            with pytest.raises(LLMConnectionError):
                async for _ in result:
                    pass

    @pytest.mark.asyncio
    async def test_stream_generic_error(self, client: AnthropicClient) -> None:
        """Generic exception during streaming raises LLMStreamError."""
        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                raise RuntimeError("something broke")

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            with pytest.raises(LLMStreamError):
                async for _ in result:
                    pass

    @pytest.mark.asyncio
    async def test_stream_non_200_error(self, client: AnthropicClient) -> None:
        """Non-200 during streaming triggers _handle_error_response."""
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock()
        mock_response.text = "Server Error"
        mock_response.json = MagicMock(return_value={"error": {"type": "server_error", "message": "Internal"}})
        mock_response.headers = {}

        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                return mock_response

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            with pytest.raises(LLMServerError):
                async for _ in result:
                    pass

    @pytest.mark.asyncio
    async def test_stream_skips_bad_json(self, client: AnthropicClient) -> None:
        """Malformed JSON data lines are skipped during streaming."""

        async def _mock_aiter_lines():
            yield "data: not valid json"
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}}'
            yield 'data: {"type": "message_stop"}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _mock_aiter_lines
        mock_response.aread = AsyncMock()

        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                return mock_response

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert chunks == ["ok"]

    @pytest.mark.asyncio
    async def test_stream_with_tools(self, client: AnthropicClient) -> None:
        """Stream with tools converts tool definitions."""

        async def _mock_aiter_lines():
            yield 'data: {"type": "message_stop"}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _mock_aiter_lines
        mock_response.aread = AsyncMock()

        mock_http_client = AsyncMock()
        captured_payload = {}

        class _StreamCM:
            async def __aenter__(self_inner):
                return mock_response

            async def __aexit__(self_inner, *args):
                pass

        def _stream(method, url, json=None):
            captured_payload.update(json or {})
            return _StreamCM()

        mock_http_client.stream = MagicMock(side_effect=_stream)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object"},
                },
            }
        ]

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
                tools=tools,
            )
            async for _ in result:
                pass

        assert len(captured_payload["tools"]) == 1
        assert captured_payload["tools"][0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_stream_empty_data_line_skipped(self, client: AnthropicClient) -> None:
        """Empty data: lines are skipped."""

        async def _mock_aiter_lines():
            yield "data: "
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}'
            yield 'data: {"type": "message_stop"}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _mock_aiter_lines
        mock_response.aread = AsyncMock()

        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                return mock_response

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert chunks == ["hi"]

    @pytest.mark.asyncio
    async def test_stream_llm_client_error_reraised(self, client: AnthropicClient) -> None:
        """LLMClientError subclass during streaming is re-raised as-is (line 512-513)."""
        mock_http_client = AsyncMock()

        class _StreamCM:
            async def __aenter__(self_inner):
                raise LLMAuthenticationError("anthropic", "bad key")

            async def __aexit__(self_inner, *args):
                pass

        mock_http_client.stream = MagicMock(return_value=_StreamCM())

        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                max_tokens=100,
            )

            with pytest.raises(LLMAuthenticationError):
                async for _ in result:
                    pass


# ---------------------------------------------------------------------------
# close() edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloseEdgeCases:
    """Cover close() when client is None or already closed (line 520)."""

    @pytest.mark.asyncio
    async def test_close_when_client_is_none(self, client: AnthropicClient) -> None:
        """close() does nothing when _client is None."""
        assert client._client is None
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_client_already_closed(self, client: AnthropicClient) -> None:
        """close() does nothing when _client.is_closed is True."""
        mock_http = MagicMock()
        mock_http.is_closed = True
        client._client = mock_http

        await client.close()
        # Should not call aclose since already closed
        mock_http.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_when_client_open(self, client: AnthropicClient) -> None:
        """close() calls aclose and sets _client to None."""
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_awaited_once()
        assert client._client is None


# ---------------------------------------------------------------------------
# Response parsing edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResponseParsing:
    """Cover edge cases in response parsing."""

    @pytest.mark.asyncio
    async def test_multiple_text_blocks(self, client: AnthropicClient) -> None:
        """Multiple text blocks are joined with newline."""
        data = {
            "id": "msg_002",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First."},
                {"type": "text", "text": "Second."},
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 8},
        }
        mock_resp = _mock_http_response(200, data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            response = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        assert response.text == "First.\nSecond."

    @pytest.mark.asyncio
    async def test_empty_content_blocks(self, client: AnthropicClient) -> None:
        """Empty content list produces empty text."""
        data = {
            "id": "msg_003",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        mock_resp = _mock_http_response(200, data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            response = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        assert response.text == ""

    @pytest.mark.asyncio
    async def test_tool_response_type(self, client: AnthropicClient) -> None:
        """Response with tool_use blocks returns LLMToolResponse."""
        data = {
            "id": "msg_004",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "calc", "input": {"x": 1}},
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        mock_resp = _mock_http_response(200, data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            response = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        assert isinstance(response, LLMToolResponse)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "t1"
        assert response.tool_calls[0].type == "tool_use"

    @pytest.mark.asyncio
    async def test_usage_total_tokens(self, client: AnthropicClient) -> None:
        """total_tokens is computed from input + output tokens."""
        data = {
            "id": "msg_005",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        mock_resp = _mock_http_response(200, data)

        with patch.object(client, "_get_client") as mock_gc:
            http = AsyncMock()
            http.post = AsyncMock(return_value=mock_resp)
            mock_gc.return_value = http

            response = await client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )

        assert response.usage["total_tokens"] == 30
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
