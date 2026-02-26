"""
Comprehensive unit tests for LLM adapter clients.

Tests:
- Response parsing from different providers
- Retry logic with exponential backoff
- Timeout and error handling
- Rate limiting compliance
- Mock-based isolation
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("httpx", reason="httpx required for HTTP client tests")

import httpx

# Import adapters (with fallback if not available)
try:
    from src.adapters.llm.anthropic_client import AnthropicClient
    from src.adapters.llm.base import LLMResponse
    from src.adapters.llm.exceptions import (
        LLMConnectionError,
        LLMInvalidResponseError,
        LLMTimeoutError,
    )
    from src.adapters.llm.lmstudio_client import LMStudioClient
    from src.adapters.llm.openai_client import OpenAIClient

    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="LLM adapters not available")
class TestLLMResponseParsing:
    """Test response parsing from different LLM providers."""

    def test_parse_openai_chat_completion(self):
        """Parse standard OpenAI chat completion response."""
        raw_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }

        response = LLMResponse.from_openai(raw_response)

        assert response.text == "Hello! How can I help you today?"
        assert response.tokens_used == 21
        assert response.finish_reason == "stop"

    def test_parse_anthropic_message(self):
        """Parse Anthropic message response."""
        raw_response = {
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "I'm Claude, an AI assistant."}],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 25},
        }

        response = LLMResponse.from_anthropic(raw_response)

        assert response.text == "I'm Claude, an AI assistant."
        assert response.tokens_used == 35
        assert response.finish_reason == "end_turn"

    def test_parse_lmstudio_response(self):
        """Parse LM Studio (OpenAI-compatible) response."""
        raw_response = {
            "id": "lmstudio-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "local-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Local model response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        response = LLMResponse.from_openai(raw_response)
        assert response.text == "Local model response"

    def test_parse_empty_response_raises_error(self):
        """Empty response should raise an error."""
        with pytest.raises(LLMInvalidResponseError):
            LLMResponse.from_openai({})

    def test_parse_malformed_response(self):
        """Malformed response should raise descriptive error."""
        malformed = {"choices": []}  # No choices

        with pytest.raises(LLMInvalidResponseError):
            LLMResponse.from_openai(malformed)

    def test_parse_response_with_tool_calls(self):
        """Response with tool/function calls should be parsed."""
        raw_response = {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 50},
        }

        response = LLMResponse.from_openai(raw_response)
        assert response.finish_reason == "tool_calls"
        assert response.tool_calls is not None


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="LLM adapters not available")
class TestRetryLogic:
    """Test retry behavior with exponential backoff."""

    @pytest.fixture
    def mock_httpx_client(self):
        """Create a mock httpx async client."""
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, mock_httpx_client):
        """Should retry on connection errors."""
        mock_httpx_client.post = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                httpx.ConnectError("Connection refused"),
                # Third attempt succeeds
                Mock(
                    json=Mock(
                        return_value={
                            "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}],
                            "usage": {"total_tokens": 10},
                        }
                    ),
                    status_code=200,
                ),
            ]
        )

        client = OpenAIClient(api_key="test-key", http_client=mock_httpx_client, max_retries=3)

        response = await client.generate("Test prompt")
        assert response.text == "Success"
        assert mock_httpx_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, mock_httpx_client):
        """Backoff should increase exponentially."""
        delays = []

        async def mock_sleep(seconds):
            delays.append(seconds)
            # Don't actually sleep in tests

        mock_httpx_client.post = AsyncMock(
            side_effect=[
                httpx.ConnectError("Error 1"),
                httpx.ConnectError("Error 2"),
                httpx.ConnectError("Error 3"),
                Mock(
                    json=Mock(
                        return_value={
                            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                            "usage": {"total_tokens": 5},
                        }
                    ),
                    status_code=200,
                ),
            ]
        )

        with patch("asyncio.sleep", mock_sleep):
            client = OpenAIClient(api_key="key", http_client=mock_httpx_client, max_retries=4, base_delay=1.0)
            await client.generate("prompt")

        # Verify exponential backoff: 1, 2, 4 seconds (approximately)
        assert len(delays) == 3
        assert delays[0] >= 1.0
        assert delays[1] >= 2.0
        assert delays[2] >= 4.0

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises_error(self, mock_httpx_client):
        """Should raise error after max retries exceeded."""
        mock_httpx_client.post = AsyncMock(side_effect=httpx.ConnectError("Always fails"))

        client = OpenAIClient(api_key="key", http_client=mock_httpx_client, max_retries=3)

        with pytest.raises(LLMConnectionError):
            await client.generate("prompt")

        assert mock_httpx_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_invalid_api_key(self, mock_httpx_client):
        """Should not retry on authentication errors."""
        mock_httpx_client.post = AsyncMock(
            return_value=Mock(status_code=401, json=Mock(return_value={"error": {"message": "Invalid API key"}}))
        )

        client = OpenAIClient(api_key="invalid-key", http_client=mock_httpx_client, max_retries=3)

        with pytest.raises(LLMInvalidResponseError):  # Auth error
            await client.generate("prompt")

        # Should not retry auth errors
        assert mock_httpx_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_with_backoff(self, mock_httpx_client):
        """Should retry rate limit errors with longer backoff."""
        mock_httpx_client.post = AsyncMock(
            side_effect=[
                Mock(
                    status_code=429,
                    headers={"Retry-After": "5"},
                    json=Mock(return_value={"error": {"message": "Rate limited"}}),
                ),
                Mock(
                    json=Mock(
                        return_value={
                            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                            "usage": {"total_tokens": 5},
                        }
                    ),
                    status_code=200,
                ),
            ]
        )

        delays = []

        async def mock_sleep(seconds):
            delays.append(seconds)

        with patch("asyncio.sleep", mock_sleep):
            client = OpenAIClient(api_key="key", http_client=mock_httpx_client, max_retries=2)
            response = await client.generate("prompt")

        assert response.text == "OK"
        # Should respect Retry-After header
        assert delays[0] >= 5


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="LLM adapters not available")
class TestTimeoutHandling:
    """Test timeout behavior."""

    @pytest.fixture
    def mock_httpx_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_request_timeout_raises_error(self, mock_httpx_client):
        """Timeout should raise specific error."""
        mock_httpx_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))

        client = OpenAIClient(api_key="key", http_client=mock_httpx_client, timeout_seconds=10)

        with pytest.raises(LLMTimeoutError):
            await client.generate("Long prompt")

    @pytest.mark.asyncio
    async def test_timeout_is_configurable(self, mock_httpx_client):
        """Timeout should be passed to HTTP client."""
        mock_httpx_client.post = AsyncMock(
            return_value=Mock(
                json=Mock(
                    return_value={
                        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                        "usage": {"total_tokens": 5},
                    }
                ),
                status_code=200,
            )
        )

        client = OpenAIClient(api_key="key", http_client=mock_httpx_client, timeout_seconds=60)

        await client.generate("prompt")

        # Verify timeout was set
        call_kwargs = mock_httpx_client.post.call_args
        assert call_kwargs.kwargs.get("timeout") == 60

    @pytest.mark.asyncio
    async def test_streaming_timeout_handling(self, mock_httpx_client):
        """Streaming responses should handle timeouts gracefully."""
        # This tests streaming-specific timeout behavior
        pass  # Implementation depends on streaming API


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="LLM adapters not available")
class TestErrorHandling:
    """Test error handling for various failure modes."""

    @pytest.fixture
    def mock_httpx_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_500_error_is_retried(self, mock_httpx_client):
        """Server errors (5xx) should be retried."""
        mock_httpx_client.post = AsyncMock(
            side_effect=[
                Mock(status_code=500, json=Mock(return_value={"error": "Internal error"})),
                Mock(status_code=503, json=Mock(return_value={"error": "Service unavailable"})),
                Mock(
                    json=Mock(
                        return_value={
                            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                            "usage": {"total_tokens": 5},
                        }
                    ),
                    status_code=200,
                ),
            ]
        )

        with patch("asyncio.sleep", AsyncMock()):
            client = OpenAIClient(api_key="key", http_client=mock_httpx_client, max_retries=3)
            response = await client.generate("prompt")

        assert response.text == "OK"
        assert mock_httpx_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_400_error_not_retried(self, mock_httpx_client):
        """Client errors (4xx except 429) should not be retried."""
        mock_httpx_client.post = AsyncMock(
            return_value=Mock(status_code=400, json=Mock(return_value={"error": {"message": "Bad request"}}))
        )

        client = OpenAIClient(api_key="key", http_client=mock_httpx_client, max_retries=3)

        with pytest.raises(LLMInvalidResponseError):
            await client.generate("invalid prompt")

        # Should not retry client errors
        assert mock_httpx_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_network_error_handling(self, mock_httpx_client):
        """Network errors should be handled gracefully."""
        mock_httpx_client.post = AsyncMock(side_effect=httpx.NetworkError("DNS resolution failed"))

        client = OpenAIClient(api_key="key", http_client=mock_httpx_client, max_retries=1)

        with pytest.raises(LLMConnectionError) as exc_info:
            await client.generate("prompt")

        assert "DNS" in str(exc_info.value) or "network" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self, mock_httpx_client):
        """Invalid JSON response should be handled."""
        mock_httpx_client.post = AsyncMock(
            return_value=Mock(status_code=200, json=Mock(side_effect=ValueError("Invalid JSON")))
        )

        client = OpenAIClient(api_key="key", http_client=mock_httpx_client, max_retries=1)

        with pytest.raises(LLMInvalidResponseError):
            await client.generate("prompt")


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason="LLM adapters not available")
class TestProviderSpecificBehavior:
    """Test provider-specific behaviors."""

    @pytest.mark.asyncio
    async def test_anthropic_system_prompt_handling(self):
        """Anthropic requires system prompt as separate parameter."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            return_value=Mock(
                json=Mock(
                    return_value={
                        "id": "msg_123",
                        "content": [{"type": "text", "text": "Response"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }
                ),
                status_code=200,
            )
        )

        client = AnthropicClient(api_key="key", http_client=mock_client)

        await client.generate(prompt="User message", system_prompt="You are a helpful assistant")

        # Verify system prompt is passed separately
        call_kwargs = mock_client.post.call_args
        request_body = call_kwargs.kwargs.get("json", {})
        assert "system" in request_body

    @pytest.mark.asyncio
    async def test_lmstudio_local_connection(self):
        """LM Studio should connect to local endpoint."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            return_value=Mock(
                json=Mock(
                    return_value={
                        "choices": [{"message": {"content": "Local"}, "finish_reason": "stop"}],
                        "usage": {"total_tokens": 5},
                    }
                ),
                status_code=200,
            )
        )

        client = LMStudioClient(base_url="http://localhost:1234/v1", http_client=mock_client)

        await client.generate("prompt")

        call_args = mock_client.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert "localhost" in url or "127.0.0.1" in url


class TestMockClientFactory:
    """Test utilities for creating mock LLM clients."""

    @pytest.fixture
    def mock_llm_client(self):
        """Factory for creating mock LLM clients."""

        def _create_mock(responses=None, errors=None):
            client = AsyncMock()

            if responses:
                client.generate = AsyncMock(
                    side_effect=[Mock(text=r, tokens_used=10, finish_reason="stop") for r in responses]
                )
            elif errors:
                client.generate = AsyncMock(side_effect=errors)
            else:
                client.generate = AsyncMock(return_value=Mock(text="Default response", tokens_used=10))

            return client

        return _create_mock

    @pytest.mark.asyncio
    async def test_mock_client_returns_responses(self, mock_llm_client):
        """Mock client should return configured responses."""
        client = mock_llm_client(responses=["Response 1", "Response 2"])

        r1 = await client.generate("prompt 1")
        r2 = await client.generate("prompt 2")

        assert r1.text == "Response 1"
        assert r2.text == "Response 2"

    @pytest.mark.asyncio
    async def test_mock_client_raises_errors(self, mock_llm_client):
        """Mock client should raise configured errors."""
        client = mock_llm_client(errors=[Exception("First error"), Exception("Second error")])

        with pytest.raises(Exception, match="First error"):
            await client.generate("prompt")
