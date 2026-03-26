"""
Comprehensive tests for the OpenAI LLM client adapter.

Covers CircuitBreaker state machine, OpenAIClient constructor validation,
HTTP error-to-exception mapping, generate (non-streaming and streaming),
tool call parsing, and circuit breaker integration during generation.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.adapters.llm.base import LLMResponse, LLMToolResponse, ToolCall
from src.adapters.llm.exceptions import (
    CircuitBreakerOpenError,
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
    LLMTimeoutError,
)
from src.adapters.llm.openai_client import CircuitBreaker, OpenAIClient

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(
    content: str = "Hello!",
    model: str = "gpt-4-turbo-preview",
    finish_reason: str = "stop",
    tool_calls: list | None = None,
    usage: dict | None = None,
) -> dict:
    """Build a realistic OpenAI chat completion response dict."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _make_httpx_response(
    status_code: int = 200,
    json_body: dict | None = None,
    text: str = "",
    headers: dict | None = None,
) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text or json.dumps(json_body or {})
    resp.headers = httpx.Headers(headers or {})
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = json.JSONDecodeError("", "", 0)
    return resp


def _make_client(monkeypatch, **kwargs) -> OpenAIClient:
    """Create an OpenAIClient with env cleaned and sensible test defaults."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    defaults = {"api_key": "test-key-xxx", "max_retries": 1}
    defaults.update(kwargs)
    return OpenAIClient(**defaults)


def _inject_mock_http(client: OpenAIClient, mock_response: MagicMock) -> AsyncMock:
    """Inject a mock httpx.AsyncClient that returns *mock_response* on post()."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.return_value = mock_response
    mock_http.is_closed = False
    client._client = mock_http
    return mock_http


# ---------------------------------------------------------------------------
# 1. CircuitBreaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for the CircuitBreaker state machine."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_can_execute_when_closed(self):
        cb = CircuitBreaker()
        assert cb.can_execute() is True

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.can_execute() is True

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_open_blocks_execution(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
        cb.record_failure()
        assert cb.state == "open"
        # reset_timeout=0.0 means the timeout has already elapsed
        assert cb.can_execute() is True
        assert cb.state == "half-open"

    def test_half_open_limits_calls(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0, half_open_max_calls=1)
        cb.record_failure()
        assert cb.can_execute() is True  # transitions to half-open
        cb.half_open_calls = 1
        assert cb.can_execute() is False

    def test_success_resets_to_closed_from_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
        cb.record_failure()
        cb.can_execute()  # half-open
        assert cb.state == "half-open"
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_success_resets_failure_count_when_closed(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        cb.record_success()
        assert cb.failure_count == 0

    def test_failure_in_half_open_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
        cb.record_failure()
        cb.can_execute()  # half-open
        assert cb.state == "half-open"
        cb.record_failure()
        assert cb.state == "open"

    def test_get_reset_time_when_closed_is_zero(self):
        cb = CircuitBreaker()
        assert cb.get_reset_time() == 0.0

    def test_get_reset_time_when_half_open_is_zero(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
        cb.record_failure()
        cb.can_execute()  # half-open
        assert cb.get_reset_time() == 0.0

    def test_get_reset_time_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
        cb.record_failure()
        reset_time = cb.get_reset_time()
        assert 59.0 <= reset_time <= 60.0

    def test_half_open_transition_uses_real_timeout(self):
        """Circuit stays open when timeout has not elapsed."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=300.0)
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_execute() is False  # timeout not elapsed


# ---------------------------------------------------------------------------
# 2. Constructor tests
# ---------------------------------------------------------------------------


class TestOpenAIClientConstructor:
    """Tests for OpenAIClient initialization."""

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMAuthenticationError, match="API key not provided"):
            OpenAIClient()

    def test_accepts_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAIClient(api_key="test-key-xxx")
        assert client.api_key == "test-key-xxx"

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-env-key-xxx")
        client = OpenAIClient()
        assert client.api_key == "test-env-key-xxx"

    def test_explicit_key_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-env-key-xxx")
        client = OpenAIClient(api_key="test-explicit-key-xxx")
        assert client.api_key == "test-explicit-key-xxx"

    def test_default_model(self, monkeypatch):
        client = _make_client(monkeypatch)
        assert client.model == "gpt-4-turbo-preview"

    def test_custom_model(self, monkeypatch):
        client = _make_client(monkeypatch, model="gpt-3.5-turbo")
        assert client.model == "gpt-3.5-turbo"

    def test_default_base_url(self, monkeypatch):
        client = _make_client(monkeypatch)
        assert client.base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self, monkeypatch):
        client = _make_client(monkeypatch, base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_organization_stored(self, monkeypatch):
        client = _make_client(monkeypatch, organization="org-test123")
        assert client.organization == "org-test123"

    def test_circuit_breaker_settings(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            circuit_breaker_threshold=10,
            circuit_breaker_reset=120.0,
        )
        assert client.circuit_breaker.failure_threshold == 10
        assert client.circuit_breaker.reset_timeout == 120.0

    def test_default_timeout(self, monkeypatch):
        client = _make_client(monkeypatch)
        assert client.timeout == 60.0

    def test_custom_timeout(self, monkeypatch):
        client = _make_client(monkeypatch, timeout=30.0)
        assert client.timeout == 30.0


# ---------------------------------------------------------------------------
# 3. Error mapping tests
# ---------------------------------------------------------------------------


class TestHandleErrorResponse:
    """Tests for _handle_error_response mapping HTTP status codes to exceptions."""

    @pytest.fixture()
    def client(self, monkeypatch):
        return _make_client(monkeypatch)

    @pytest.mark.parametrize(
        "status_code, error_msg, expected_exc",
        [
            (401, "Invalid API key", LLMAuthenticationError),
            (429, "Rate limit exceeded", LLMRateLimitError),
            (402, "Quota exceeded", LLMQuotaExceededError),
            (404, "Model not found", LLMModelNotFoundError),
            (400, "Bad request", LLMInvalidRequestError),
            (500, "Internal server error", LLMServerError),
            (502, "Bad gateway", LLMServerError),
            (503, "Service unavailable", LLMServerError),
        ],
    )
    def test_status_code_mapping(self, client, status_code, error_msg, expected_exc):
        resp = _make_httpx_response(
            status_code=status_code,
            json_body={"error": {"message": error_msg}},
        )
        with pytest.raises(expected_exc):
            client._handle_error_response(resp)

    def test_400_with_context_length_raises_context_length_error(self, client):
        resp = _make_httpx_response(
            status_code=400,
            json_body={"error": {"message": "maximum context_length is 4096"}},
        )
        with pytest.raises(LLMContextLengthError):
            client._handle_error_response(resp)

    def test_429_with_retry_after_header(self, client):
        resp = _make_httpx_response(
            status_code=429,
            json_body={"error": {"message": "Rate limit exceeded"}},
            headers={"Retry-After": "30"},
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            client._handle_error_response(resp)
        assert exc_info.value.retry_after == 30.0

    def test_429_without_retry_after_header(self, client):
        resp = _make_httpx_response(
            status_code=429,
            json_body={"error": {"message": "Rate limit exceeded"}},
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            client._handle_error_response(resp)
        assert exc_info.value.retry_after is None

    def test_unknown_4xx_raises_generic_client_error(self, client):
        resp = _make_httpx_response(
            status_code=418,
            json_body={"error": {"message": "I'm a teapot"}},
        )
        with pytest.raises(LLMClientError):
            client._handle_error_response(resp)

    def test_unparseable_json_falls_back_to_text(self, client):
        resp = _make_httpx_response(status_code=401, text="Unauthorized")
        with pytest.raises(LLMAuthenticationError):
            client._handle_error_response(resp)


# ---------------------------------------------------------------------------
# 4. generate() non-streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateNonStream:
    """Tests for OpenAIClient.generate() in non-streaming mode."""

    @pytest.fixture()
    def client(self, monkeypatch):
        return _make_client(monkeypatch)

    async def test_basic_prompt_generation(self, client):
        response_data = _make_openai_response(content="Hi there!")
        mock_resp = _make_httpx_response(status_code=200, json_body=response_data)
        mock_http = _inject_mock_http(client, mock_resp)

        result = await client.generate(prompt="Hello")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hi there!"
        assert result.model == "gpt-4-turbo-preview"
        assert result.total_tokens == 15

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert payload["model"] == "gpt-4-turbo-preview"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["temperature"] == 0.7

    async def test_messages_passed_through(self, client):
        mock_resp = _make_httpx_response(status_code=200, json_body=_make_openai_response())
        mock_http = _inject_mock_http(client, mock_resp)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        await client.generate(messages=messages)

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert payload["messages"] == messages

    async def test_max_tokens_and_stop_in_payload(self, client):
        mock_resp = _make_httpx_response(status_code=200, json_body=_make_openai_response())
        mock_http = _inject_mock_http(client, mock_resp)

        await client.generate(prompt="test", max_tokens=100, stop=["END"])

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert payload["max_tokens"] == 100
        assert payload["stop"] == ["END"]

    async def test_custom_temperature_in_payload(self, client):
        mock_resp = _make_httpx_response(status_code=200, json_body=_make_openai_response())
        mock_http = _inject_mock_http(client, mock_resp)

        await client.generate(prompt="test", temperature=0.0)

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert payload["temperature"] == 0.0

    async def test_max_tokens_omitted_when_none(self, client):
        mock_resp = _make_httpx_response(status_code=200, json_body=_make_openai_response())
        mock_http = _inject_mock_http(client, mock_resp)

        await client.generate(prompt="test")

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert "max_tokens" not in payload

    async def test_finish_reason_propagated(self, client):
        data = _make_openai_response(finish_reason="length")
        mock_resp = _make_httpx_response(status_code=200, json_body=data)
        _inject_mock_http(client, mock_resp)

        result = await client.generate(prompt="test")
        assert result.finish_reason == "length"

    async def test_usage_stats_propagated(self, client):
        usage = {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
        data = _make_openai_response(usage=usage)
        mock_resp = _make_httpx_response(status_code=200, json_body=data)
        _inject_mock_http(client, mock_resp)

        result = await client.generate(prompt="test")
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 30
        assert result.total_tokens == 50

    async def test_http_error_triggers_circuit_breaker_failure(self, client):
        mock_resp = _make_httpx_response(
            status_code=500,
            json_body={"error": {"message": "Server error"}},
        )
        _inject_mock_http(client, mock_resp)

        with pytest.raises(LLMServerError):
            await client.generate(prompt="test")

        assert client.circuit_breaker.failure_count > 0

    async def test_success_resets_circuit_breaker(self, client):
        client.circuit_breaker.record_failure()
        assert client.circuit_breaker.failure_count == 1

        mock_resp = _make_httpx_response(status_code=200, json_body=_make_openai_response())
        _inject_mock_http(client, mock_resp)

        await client.generate(prompt="test")
        assert client.circuit_breaker.failure_count == 0

    async def test_timeout_raises_llm_timeout_error(self, client):
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post.side_effect = httpx.TimeoutException("timed out")
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(LLMTimeoutError):
            await client.generate(prompt="test")

    async def test_connect_error_raises_llm_connection_error(self, client):
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post.side_effect = httpx.ConnectError("connection refused")
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(LLMConnectionError):
            await client.generate(prompt="test")

    async def test_malformed_response_raises_parse_error(self, client):
        bad_data = {"id": "chatcmpl-bad", "object": "chat.completion"}
        mock_resp = _make_httpx_response(status_code=200, json_body=bad_data)
        _inject_mock_http(client, mock_resp)

        with pytest.raises(LLMResponseParseError):
            await client.generate(prompt="test")

    async def test_extra_kwargs_passed_to_payload(self, client):
        mock_resp = _make_httpx_response(status_code=200, json_body=_make_openai_response())
        mock_http = _inject_mock_http(client, mock_resp)

        await client.generate(prompt="test", top_p=0.9, presence_penalty=0.5)

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert payload["top_p"] == 0.9
        assert payload["presence_penalty"] == 0.5


# ---------------------------------------------------------------------------
# 5. Tool call parsing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestToolCallParsing:
    """Tests for parsing tool/function calls from OpenAI responses."""

    @pytest.fixture()
    def client(self, monkeypatch):
        return _make_client(monkeypatch)

    async def test_single_tool_call(self, client):
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "London"}),
                },
            }
        ]
        data = _make_openai_response(content="", tool_calls=tool_calls, finish_reason="tool_calls")
        mock_resp = _make_httpx_response(status_code=200, json_body=data)
        _inject_mock_http(client, mock_resp)

        result = await client.generate(
            prompt="Weather?",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        assert isinstance(result, LLMToolResponse)
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "call_abc123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "London"}

    async def test_multiple_tool_calls(self, client):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "London"}),
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": json.dumps({"timezone": "UTC"}),
                },
            },
        ]
        data = _make_openai_response(content="", tool_calls=tool_calls)
        mock_resp = _make_httpx_response(status_code=200, json_body=data)
        _inject_mock_http(client, mock_resp)

        result = await client.generate(prompt="test")
        assert isinstance(result, LLMToolResponse)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[1].name == "get_time"
        assert result.tool_calls[1].arguments == {"timezone": "UTC"}

    async def test_tools_and_tool_choice_in_payload(self, client):
        data = _make_openai_response()
        mock_resp = _make_httpx_response(status_code=200, json_body=data)
        mock_http = _inject_mock_http(client, mock_resp)

        tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
        await client.generate(prompt="test", tools=tools)

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"

    async def test_custom_tool_choice(self, client):
        data = _make_openai_response()
        mock_resp = _make_httpx_response(status_code=200, json_body=data)
        mock_http = _inject_mock_http(client, mock_resp)

        tools = [{"type": "function", "function": {"name": "my_tool"}}]
        await client.generate(prompt="test", tools=tools, tool_choice="none")

        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1]["json"]
        assert payload["tool_choice"] == "none"

    async def test_response_without_tool_calls_returns_llm_response(self, client):
        data = _make_openai_response(content="Just text, no tools.")
        mock_resp = _make_httpx_response(status_code=200, json_body=data)
        _inject_mock_http(client, mock_resp)

        result = await client.generate(prompt="test")
        assert isinstance(result, LLMResponse)
        assert not isinstance(result, LLMToolResponse)


# ---------------------------------------------------------------------------
# 6. Circuit breaker integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCircuitBreakerIntegration:
    """Tests for circuit breaker behaviour during generate() calls."""

    @pytest.fixture()
    def client(self, monkeypatch):
        return _make_client(
            monkeypatch,
            circuit_breaker_threshold=2,
            circuit_breaker_reset=300.0,
        )

    async def test_generate_fails_fast_when_circuit_open(self, client):
        client.circuit_breaker.state = "open"
        client.circuit_breaker.failure_count = 5
        client.circuit_breaker.last_failure_time = time.time()

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await client.generate(prompt="test")
        assert exc_info.value.failure_count == 5

    async def test_circuit_opens_after_repeated_failures(self, client):
        mock_resp = _make_httpx_response(
            status_code=500,
            json_body={"error": {"message": "Server error"}},
        )
        _inject_mock_http(client, mock_resp)

        # Threshold is 2
        for _ in range(2):
            with pytest.raises(LLMServerError):
                await client.generate(prompt="test")

        assert client.circuit_breaker.state == "open"

        # Subsequent call should be blocked immediately
        with pytest.raises(CircuitBreakerOpenError):
            await client.generate(prompt="test")

    async def test_circuit_recovers_after_success_in_half_open(self, client):
        client.circuit_breaker.state = "open"
        client.circuit_breaker.failure_count = 2
        client.circuit_breaker.last_failure_time = time.time()
        client.circuit_breaker.reset_timeout = 0.0  # instant transition to half-open

        mock_resp = _make_httpx_response(status_code=200, json_body=_make_openai_response())
        _inject_mock_http(client, mock_resp)

        result = await client.generate(prompt="test")
        assert isinstance(result, LLMResponse)
        assert client.circuit_breaker.state == "closed"

    async def test_stream_blocked_when_circuit_open(self, client):
        client.circuit_breaker.state = "open"
        client.circuit_breaker.failure_count = 5
        client.circuit_breaker.last_failure_time = time.time()

        with pytest.raises(CircuitBreakerOpenError):
            await client.generate(prompt="test", stream=True)


# ---------------------------------------------------------------------------
# 7. Streaming tests
# ---------------------------------------------------------------------------


def _build_stream_mocks(client: OpenAIClient, sse_lines: list[str], status_code: int = 200) -> None:
    """Wire up mock httpx client for streaming with the given SSE lines."""
    mock_response = AsyncMock()
    mock_response.status_code = status_code

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    # For error path: provide aread and the error fields
    async def mock_aread():
        return b""

    mock_response.aread = mock_aread
    mock_response.text = ""
    mock_response.headers = httpx.Headers({})
    mock_response.json = MagicMock(side_effect=json.JSONDecodeError("", "", 0))

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = False

    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    mock_http.stream.return_value = stream_cm

    client._client = mock_http


@pytest.mark.asyncio
class TestGenerateStream:
    """Tests for OpenAIClient.generate() in streaming mode."""

    @pytest.fixture()
    def client(self, monkeypatch):
        return _make_client(monkeypatch)

    async def test_streaming_returns_async_iterator(self, client):
        _build_stream_mocks(
            client,
            [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                "data: [DONE]",
            ],
        )

        result = await client.generate(prompt="Hello", stream=True)
        assert isinstance(result, AsyncIterator)

        chunks = [chunk async for chunk in result]
        assert chunks == ["Hello", " world"]

    async def test_streaming_skips_empty_content(self, client):
        _build_stream_mocks(
            client,
            [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                'data: {"choices":[{"delta":{"content":""}}]}',
                'data: {"choices":[{"delta":{"content":"OK"}}]}',
                "data: [DONE]",
            ],
        )

        result = await client.generate(prompt="test", stream=True)
        chunks = [chunk async for chunk in result]
        assert chunks == ["OK"]

    async def test_streaming_ignores_non_data_lines(self, client):
        _build_stream_mocks(
            client,
            [
                ": keep-alive",
                "",
                'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                "data: [DONE]",
            ],
        )

        result = await client.generate(prompt="test", stream=True)
        chunks = [chunk async for chunk in result]
        assert chunks == ["Hi"]

    async def test_streaming_payload_includes_stream_flag(self, client):
        _build_stream_mocks(client, ["data: [DONE]"])

        result = await client.generate(prompt="test", stream=True)
        _ = [chunk async for chunk in result]  # consume

        mock_http = client._client
        call_args = mock_http.stream.call_args
        payload = call_args.kwargs.get("json") or call_args[1]["json"]
        assert payload["stream"] is True
        assert payload["model"] == "gpt-4-turbo-preview"

    async def test_streaming_handles_malformed_json_gracefully(self, client):
        _build_stream_mocks(
            client,
            [
                "data: {invalid-json}",
                'data: {"choices":[{"delta":{"content":"OK"}}]}',
                "data: [DONE]",
            ],
        )

        result = await client.generate(prompt="test", stream=True)
        chunks = [chunk async for chunk in result]
        # Malformed line is skipped, only valid chunk yielded
        assert chunks == ["OK"]

    async def test_streaming_multiple_chunks_concatenated(self, client):
        _build_stream_mocks(
            client,
            [
                'data: {"choices":[{"delta":{"content":"The "}}]}',
                'data: {"choices":[{"delta":{"content":"quick "}}]}',
                'data: {"choices":[{"delta":{"content":"brown "}}]}',
                'data: {"choices":[{"delta":{"content":"fox"}}]}',
                "data: [DONE]",
            ],
        )

        result = await client.generate(prompt="test", stream=True)
        chunks = [chunk async for chunk in result]
        assert "".join(chunks) == "The quick brown fox"

    async def test_streaming_records_success_on_completion(self, client):
        client.circuit_breaker.record_failure()
        assert client.circuit_breaker.failure_count == 1

        _build_stream_mocks(
            client,
            [
                'data: {"choices":[{"delta":{"content":"done"}}]}',
                "data: [DONE]",
            ],
        )

        result = await client.generate(prompt="test", stream=True)
        _ = [chunk async for chunk in result]

        # Circuit breaker should have recorded success
        assert client.circuit_breaker.failure_count == 0
