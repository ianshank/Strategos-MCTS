"""
Comprehensive tests for Anthropic Claude LLM client adapter.

Tests AnthropicClient class methods, message conversion, error handling,
streaming, and circuit breaker functionality.

Note: These are unit tests with mocks, not integration tests requiring actual API access.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("httpx", reason="httpx required for HTTP client tests")

import httpx

from src.adapters.llm.anthropic_client import ANTHROPIC_MODELS, AnthropicClient
from src.adapters.llm.exceptions import (
    CircuitBreakerOpenError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
    LLMTimeoutError,
)


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake API keys to prevent conftest from skipping tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-unit-tests")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests")


@pytest.fixture
def api_key() -> str:
    """Provide test API key."""
    return "sk-ant-test-key-12345"


@pytest.fixture
def client(api_key: str) -> AnthropicClient:
    """Create Anthropic client for testing."""
    return AnthropicClient(api_key=api_key)


@pytest.fixture
def mock_response() -> dict:
    """Create mock successful response."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello, world!"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


# ============================================================================
# Model Mapping Tests
# ============================================================================


class TestModelMappings:
    """Tests for model alias mappings."""

    def test_opus_alias(self) -> None:
        """Test opus alias resolves correctly."""
        assert ANTHROPIC_MODELS["opus"] == "claude-3-opus-20240229"

    def test_sonnet_alias(self) -> None:
        """Test sonnet alias resolves correctly."""
        assert ANTHROPIC_MODELS["sonnet"] == "claude-3-5-sonnet-20241022"

    def test_haiku_alias(self) -> None:
        """Test haiku alias resolves correctly."""
        assert ANTHROPIC_MODELS["haiku"] == "claude-3-haiku-20240307"

    def test_full_model_names(self) -> None:
        """Test full model names are available."""
        assert "claude-3-opus" in ANTHROPIC_MODELS
        assert "claude-3-sonnet" in ANTHROPIC_MODELS
        assert "claude-3.5-sonnet" in ANTHROPIC_MODELS


# ============================================================================
# Initialization Tests
# ============================================================================


class TestAnthropicClientInit:
    """Tests for AnthropicClient initialization."""

    def test_init_with_api_key(self, api_key: str) -> None:
        """Test initialization with API key."""
        client = AnthropicClient(api_key=api_key)

        assert client.api_key == api_key
        assert client.PROVIDER_NAME == "anthropic"

    def test_init_without_api_key_raises(self) -> None:
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMAuthenticationError):
                AnthropicClient(api_key=None)

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            client = AnthropicClient()
            assert client.api_key == "sk-ant-env-key"

    def test_init_with_model_alias(self, api_key: str) -> None:
        """Test model alias is resolved."""
        client = AnthropicClient(api_key=api_key, model="sonnet")
        assert client.model == "claude-3-5-sonnet-20241022"

    def test_init_with_full_model_name(self, api_key: str) -> None:
        """Test full model name is preserved."""
        client = AnthropicClient(api_key=api_key, model="claude-3-opus-20240229")
        assert client.model == "claude-3-opus-20240229"

    def test_init_with_custom_base_url(self, api_key: str) -> None:
        """Test custom base URL."""
        client = AnthropicClient(api_key=api_key, base_url="https://custom.api.com")
        assert client.base_url == "https://custom.api.com"

    def test_init_with_custom_timeout(self, api_key: str) -> None:
        """Test custom timeout."""
        client = AnthropicClient(api_key=api_key, timeout=300.0)
        assert client.timeout == 300.0

    def test_default_model(self, api_key: str) -> None:
        """Test default model is set."""
        client = AnthropicClient(api_key=api_key)
        assert client.model == AnthropicClient.DEFAULT_MODEL


# ============================================================================
# Message Conversion Tests
# ============================================================================


class TestMessageConversion:
    """Tests for message conversion methods."""

    def test_convert_user_message(self, client: AnthropicClient) -> None:
        """Test converting user message."""
        messages = [{"role": "user", "content": "Hello"}]

        system, converted = client._convert_messages_to_anthropic(messages)

        assert system is None
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"

    def test_convert_assistant_message(self, client: AnthropicClient) -> None:
        """Test converting assistant message."""
        messages = [{"role": "assistant", "content": "Hi there!"}]

        system, converted = client._convert_messages_to_anthropic(messages)

        assert system is None
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "Hi there!"

    def test_convert_system_message(self, client: AnthropicClient) -> None:
        """Test converting system message to separate parameter."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        system, converted = client._convert_messages_to_anthropic(messages)

        assert system == "You are a helpful assistant."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_convert_tool_result_message(self, client: AnthropicClient) -> None:
        """Test converting tool result message."""
        messages = [{"role": "tool", "tool_call_id": "call_123", "content": "Result: 42"}]

        system, converted = client._convert_messages_to_anthropic(messages)

        assert system is None
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"][0]["type"] == "tool_result"
        assert converted[0]["content"][0]["tool_use_id"] == "call_123"

    def test_convert_multiple_messages(self, client: AnthropicClient) -> None:
        """Test converting conversation with multiple messages."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "Thanks!"},
        ]

        system, converted = client._convert_messages_to_anthropic(messages)

        assert system == "Be helpful."
        assert len(converted) == 3
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[2]["role"] == "user"


# ============================================================================
# Tool Conversion Tests
# ============================================================================


class TestToolConversion:
    """Tests for tool definition conversion."""

    def test_convert_openai_style_tool(self, client: AnthropicClient) -> None:
        """Test converting OpenAI-style tool definition."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        converted = client._convert_tools_to_anthropic(tools)

        assert len(converted) == 1
        assert converted[0]["name"] == "get_weather"
        assert converted[0]["description"] == "Get current weather"
        assert converted[0]["input_schema"]["type"] == "object"

    def test_convert_anthropic_style_tool(self, client: AnthropicClient) -> None:
        """Test Anthropic-style tool passes through."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
            }
        ]

        converted = client._convert_tools_to_anthropic(tools)

        assert len(converted) == 1
        assert converted[0] == tools[0]

    def test_convert_tool_without_description(self, client: AnthropicClient) -> None:
        """Test tool conversion without description."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "parameters": {"type": "object"},
                },
            }
        ]

        converted = client._convert_tools_to_anthropic(tools)

        assert converted[0]["description"] == ""


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error response handling."""

    def test_authentication_error(self, client: AnthropicClient) -> None:
        """Test 401 error raises authentication error."""
        response = MagicMock()
        response.status_code = 401
        response.json.return_value = {"error": {"type": "authentication_error", "message": "Invalid API key"}}
        response.text = "Invalid API key"

        with pytest.raises(LLMAuthenticationError):
            client._handle_error_response(response)

    def test_rate_limit_error(self, client: AnthropicClient) -> None:
        """Test 429 error raises rate limit error."""
        response = MagicMock()
        response.status_code = 429
        response.json.return_value = {"error": {"type": "rate_limit_error", "message": "Rate limited"}}
        response.text = "Rate limited"
        response.headers = {"retry-after": "60"}

        with pytest.raises(LLMRateLimitError) as exc_info:
            client._handle_error_response(response)

        assert exc_info.value.retry_after == 60.0

    def test_quota_exceeded_error(self, client: AnthropicClient) -> None:
        """Test 402 error raises quota exceeded error."""
        response = MagicMock()
        response.status_code = 402
        response.json.return_value = {"error": {"type": "billing_error", "message": "Quota exceeded"}}
        response.text = "Quota exceeded"

        with pytest.raises(LLMQuotaExceededError):
            client._handle_error_response(response)

    def test_model_not_found_error(self, client: AnthropicClient) -> None:
        """Test 404 error raises model not found error."""
        response = MagicMock()
        response.status_code = 404
        response.json.return_value = {"error": {"type": "not_found_error", "message": "Model not found"}}
        response.text = "Model not found"

        with pytest.raises(LLMModelNotFoundError):
            client._handle_error_response(response)

    def test_context_length_error(self, client: AnthropicClient) -> None:
        """Test context length error from 400."""
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {"error": {"type": "invalid_request_error", "message": "context length exceeded"}}
        response.text = "context length exceeded"

        with pytest.raises(LLMContextLengthError):
            client._handle_error_response(response)

    def test_content_filter_error(self, client: AnthropicClient) -> None:
        """Test content policy error from 400."""
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {"error": {"type": "content_policy", "message": "Content filtered"}}
        response.text = "Content filtered"

        with pytest.raises(LLMContentFilterError):
            client._handle_error_response(response)

    def test_invalid_request_error(self, client: AnthropicClient) -> None:
        """Test generic 400 error raises invalid request error."""
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {"error": {"type": "invalid_request_error", "message": "Bad request"}}
        response.text = "Bad request"

        with pytest.raises(LLMInvalidRequestError):
            client._handle_error_response(response)

    def test_server_error(self, client: AnthropicClient) -> None:
        """Test 500+ error raises server error."""
        response = MagicMock()
        response.status_code = 503
        response.json.return_value = {"error": {"type": "server_error", "message": "Service unavailable"}}
        response.text = "Service unavailable"

        with pytest.raises(LLMServerError):
            client._handle_error_response(response)


# ============================================================================
# HTTP Client Tests
# ============================================================================


class TestHTTPClient:
    """Tests for HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, client: AnthropicClient) -> None:
        """Test _get_client creates httpx client."""
        http_client = await client._get_client()

        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)

        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, client: AnthropicClient) -> None:
        """Test _get_client reuses existing client."""
        http_client1 = await client._get_client()
        http_client2 = await client._get_client()

        assert http_client1 is http_client2

        await client.close()

    @pytest.mark.asyncio
    async def test_close_client(self, client: AnthropicClient) -> None:
        """Test closing HTTP client."""
        await client._get_client()
        await client.close()

        assert client._client is None


# ============================================================================
# Generation Tests
# ============================================================================


class TestGeneration:
    """Tests for generation methods."""

    @pytest.mark.asyncio
    async def test_generate_non_stream(self, client: AnthropicClient, mock_response: dict) -> None:
        """Test non-streaming generation."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_http_response)
            mock_get_client.return_value = mock_http_client

            response = await client.generate(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )

            assert response.text == "Hello, world!"
            assert response.model == "claude-3-5-sonnet-20241022"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_generate_with_prompt(self, client: AnthropicClient, mock_response: dict) -> None:
        """Test generation with simple prompt."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_http_response)
            mock_get_client.return_value = mock_http_client

            response = await client.generate(
                prompt="Hello",
                max_tokens=100,
            )

            assert response.text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_generate_default_max_tokens(self, client: AnthropicClient, mock_response: dict) -> None:
        """Test generation sets default max_tokens."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_http_response)
            mock_get_client.return_value = mock_http_client

            await client.generate(
                messages=[{"role": "user", "content": "Hello"}],
            )

            # Verify request payload included max_tokens
            call_kwargs = mock_http_client.post.call_args.kwargs
            assert call_kwargs["json"]["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, client: AnthropicClient) -> None:
        """Test generation with tool response."""
        tool_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "NYC"},
                },
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }

        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = tool_response

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_http_response)
            mock_get_client.return_value = mock_http_client

            response = await client.generate(
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "get_weather"
            assert response.tool_calls[0].arguments == {"location": "NYC"}

    @pytest.mark.asyncio
    async def test_generate_circuit_breaker_open(self, client: AnthropicClient) -> None:
        """Test circuit breaker blocks requests when open."""
        # Open the circuit breaker
        for _ in range(5):
            client.circuit_breaker.record_failure()

        with pytest.raises(CircuitBreakerOpenError):
            await client.generate(messages=[{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self, client: AnthropicClient) -> None:
        """Test timeout raises appropriate error."""
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_get_client.return_value = mock_http_client

            with pytest.raises(LLMTimeoutError):
                await client.generate(
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=100,
                )

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, client: AnthropicClient) -> None:
        """Test connection error raises appropriate error."""
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_get_client.return_value = mock_http_client

            with pytest.raises(LLMConnectionError):
                await client.generate(
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=100,
                )

    @pytest.mark.asyncio
    async def test_generate_response_parse_error(self, client: AnthropicClient) -> None:
        """Test malformed response raises parse error."""
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.side_effect = json.JSONDecodeError("", "", 0)
        mock_http_response.text = "invalid json"

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_http_response)
            mock_get_client.return_value = mock_http_client

            with pytest.raises(LLMResponseParseError):
                await client.generate(
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=100,
                )


# ============================================================================
# Circuit Breaker Integration Tests
# ============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_records_success(self, client: AnthropicClient) -> None:
        """Test circuit breaker records success."""
        client.circuit_breaker.record_success()

        assert client.circuit_breaker.failure_count == 0

    def test_circuit_breaker_records_failure(self, client: AnthropicClient) -> None:
        """Test circuit breaker records failure."""
        client.circuit_breaker.record_failure()

        assert client.circuit_breaker.failure_count == 1

    def test_circuit_breaker_opens_after_threshold(self, client: AnthropicClient) -> None:
        """Test circuit breaker opens after threshold failures."""
        for _ in range(5):
            client.circuit_breaker.record_failure()

        assert client.circuit_breaker.can_execute() is False


# ============================================================================
# Retry Logic Tests
# ============================================================================


class TestRetryLogic:
    """Tests for retry decorator creation."""

    def test_make_retry_decorator(self, client: AnthropicClient) -> None:
        """Test retry decorator is created properly."""
        decorator = client._make_retry_decorator()
        assert decorator is not None


# ============================================================================
# API Version Tests
# ============================================================================


class TestAPIVersion:
    """Tests for API version handling."""

    @pytest.mark.asyncio
    async def test_api_version_header(self, client: AnthropicClient) -> None:
        """Test API version header is set."""
        http_client = await client._get_client()

        assert "anthropic-version" in http_client.headers
        assert http_client.headers["anthropic-version"] == client.API_VERSION

        await client.close()

    @pytest.mark.asyncio
    async def test_api_key_header(self, client: AnthropicClient) -> None:
        """Test API key header is set."""
        http_client = await client._get_client()

        assert "x-api-key" in http_client.headers
        assert http_client.headers["x-api-key"] == client.api_key

        await client.close()
