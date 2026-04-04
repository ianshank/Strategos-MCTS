"""
Anthropic Claude LLM client adapter.

Implements the LLMClient protocol for Anthropic's Messages API.
Supports Claude 3 models with proper content block handling.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.constants import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ANTHROPIC_TIMEOUT,
    DEFAULT_MAX_TOKENS,
)

from .base import BaseLLMClient, LLMResponse, LLMToolResponse, ToolCall
from .exceptions import (
    CircuitBreakerOpenError,
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
    LLMStreamError,
    LLMTimeoutError,
)
from .openai_client import CircuitBreaker

logger = logging.getLogger(__name__)


# Model mappings for convenience
ANTHROPIC_MODELS = {
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3.5-sonnet-v2": "claude-3-5-sonnet-20241022",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    # Add latest models
    "opus": "claude-3-opus-20240229",
    "sonnet": "claude-3-5-sonnet-20241022",
    "haiku": "claude-3-haiku-20240307",
}


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude API client.

    Features:
    - Messages API support (not legacy completion API)
    - Content block handling (text, tool_use)
    - Streaming with proper SSE parsing
    - Model alias mapping
    - System prompt support
    - Tool/function calling (beta)
    """

    PROVIDER_NAME = "anthropic"
    DEFAULT_BASE_URL = DEFAULT_ANTHROPIC_BASE_URL
    DEFAULT_MODEL = DEFAULT_ANTHROPIC_MODEL
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_ANTHROPIC_TIMEOUT,  # Claude can be slower
        max_retries: int = 3,
        # Circuit breaker settings
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset: float = 60.0,
        # Rate limiting
        rate_limit_per_minute: int | None = None,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use (supports aliases like 'sonnet', 'opus')
            base_url: API base URL
            timeout: Request timeout in seconds (default longer for Claude)
            max_retries: Max retry attempts
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_reset: Seconds before circuit resets
            rate_limit_per_minute: Rate limit for requests per minute (None to disable)
        """
        import os

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMAuthenticationError(self.PROVIDER_NAME, "API key not provided and ANTHROPIC_API_KEY not set")

        # Resolve model alias
        model_name = model or self.DEFAULT_MODEL
        resolved_model = ANTHROPIC_MODELS.get(model_name, model_name)

        super().__init__(
            api_key=api_key,
            model=resolved_model,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            rate_limit_per_minute=rate_limit_per_minute,
        )

        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset,
        )
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.API_VERSION,
                "Content-Type": "application/json",
            }

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def _convert_messages_to_anthropic(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """
        Convert OpenAI-style messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic uses separate system parameter
                system_prompt = content
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "tool":
                # Tool result message
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": content,
                            }
                        ],
                    }
                )

        return system_prompt, anthropic_messages

    def _convert_tools_to_anthropic(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tool definitions to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object"}),
                    }
                )
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)

        return anthropic_tools

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Convert HTTP error responses to appropriate exceptions."""
        status_code = response.status_code

        try:
            error_data = response.json()
            error_type = error_data.get("error", {}).get("type", "")
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_type = ""
            error_message = response.text

        if status_code == 401:
            raise LLMAuthenticationError(self.PROVIDER_NAME, error_message)
        elif status_code == 429:
            retry_after = response.headers.get("retry-after")
            retry_after_float = float(retry_after) if retry_after else None
            raise LLMRateLimitError(self.PROVIDER_NAME, retry_after=retry_after_float, message=error_message)
        elif status_code == 402 or "billing" in error_type.lower():
            raise LLMQuotaExceededError(self.PROVIDER_NAME, error_message)
        elif status_code == 404 or error_type == "not_found_error":
            raise LLMModelNotFoundError(self.PROVIDER_NAME, self.model)
        elif status_code == 400:
            if "context" in error_message.lower() or "token" in error_message.lower():
                raise LLMContextLengthError(self.PROVIDER_NAME)
            if "content_policy" in error_type or "safety" in error_message.lower():
                raise LLMContentFilterError(self.PROVIDER_NAME, error_message)
            raise LLMInvalidRequestError(self.PROVIDER_NAME, error_message)
        elif status_code >= 500:
            raise LLMServerError(self.PROVIDER_NAME, status_code, error_message)
        else:
            raise LLMClientError(error_message, self.PROVIDER_NAME, status_code=status_code)

    def _make_retry_decorator(self):
        """Create retry decorator with exponential backoff."""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=120),
            retry=retry_if_exception_type((LLMRateLimitError, LLMServerError, LLMConnectionError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """
        Generate a response from Anthropic Claude.

        Args:
            messages: Chat messages (will be converted to Anthropic format)
            prompt: Simple string prompt
            temperature: Sampling temperature (0.0 to 1.0 for Claude)
            max_tokens: Maximum tokens to generate (required for Anthropic)
            tools: Tool definitions (will be converted to Anthropic format)
            stream: If True, returns AsyncIterator
            stop: Stop sequences
            **kwargs: Additional parameters (top_p, top_k, etc.)

        Returns:
            LLMResponse or AsyncIterator[str] for streaming
        """
        # Apply rate limiting before proceeding
        await self._apply_rate_limit()

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(
                self.PROVIDER_NAME,
                self.circuit_breaker.failure_count,
                self.circuit_breaker.get_reset_time(),
            )

        # Anthropic requires max_tokens
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        if stream:
            return await self._generate_stream(
                messages=messages,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stop=stop,
                **kwargs,
            )
        else:
            return await self._generate_non_stream(
                messages=messages,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stop=stop,
                **kwargs,
            )

    async def _generate_non_stream(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming generation with retry logic."""

        @self._make_retry_decorator()
        async def _request():
            client = await self._get_client()

            # Convert messages
            built_messages = self._build_messages(messages, prompt)
            system_prompt, anthropic_messages = self._convert_messages_to_anthropic(built_messages)

            # Build request payload
            payload = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "temperature": min(temperature, 1.0),  # Anthropic max is 1.0
            }

            if system_prompt:
                payload["system"] = system_prompt
            if stop:
                payload["stop_sequences"] = stop
            if tools:
                payload["tools"] = self._convert_tools_to_anthropic(tools)

            # Add any additional kwargs (top_p, top_k, etc.)
            for key in ["top_p", "top_k", "metadata"]:
                if key in kwargs:
                    payload[key] = kwargs[key]

            try:
                response = await client.post("/v1/messages", json=payload)
            except httpx.TimeoutException as e:
                raise LLMTimeoutError(self.PROVIDER_NAME, self.timeout) from e
            except httpx.ConnectError as e:
                raise LLMConnectionError(self.PROVIDER_NAME, self.base_url) from e

            if response.status_code != 200:
                self._handle_error_response(response)

            return response

        try:
            response = await _request()
            self.circuit_breaker.record_success()
        except Exception:
            self.circuit_breaker.record_failure()
            raise

        # Parse response
        try:
            data = response.json()

            # Extract text from content blocks
            text_parts = []
            tool_calls = []

            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input", {}),
                            type="tool_use",
                        )
                    )

            text = "\n".join(text_parts)

            # Build usage dict
            usage = {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            finish_reason = data.get("stop_reason", "stop")

            llm_response: LLMResponse | LLMToolResponse
            if tool_calls:
                llm_response = LLMToolResponse(
                    text=text,
                    usage=usage,
                    model=data.get("model", self.model),
                    raw_response=data,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )
            else:
                llm_response = LLMResponse(
                    text=text,
                    usage=usage,
                    model=data.get("model", self.model),
                    raw_response=data,
                    finish_reason=finish_reason,
                )

            self._update_stats(llm_response)
            return llm_response

        except (KeyError, json.JSONDecodeError) as e:
            raise LLMResponseParseError(self.PROVIDER_NAME, response.text) from e

    async def _generate_stream(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming generation with Server-Sent Events."""

        client = await self._get_client()

        # Convert messages
        built_messages = self._build_messages(messages, prompt)
        system_prompt, anthropic_messages = self._convert_messages_to_anthropic(built_messages)

        # Build request payload
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": min(temperature, 1.0),
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt
        if stop:
            payload["stop_sequences"] = stop
        if tools:
            payload["tools"] = self._convert_tools_to_anthropic(tools)

        for key in ["top_p", "top_k"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        async def stream_generator() -> AsyncIterator[str]:
            try:
                async with client.stream("POST", "/v1/messages", json=payload) as response:
                    if response.status_code != 200:
                        await response.aread()
                        self._handle_error_response(response)

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            continue

                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if not data_str:
                                continue

                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type", "")

                                if event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            yield text

                                elif event_type == "message_stop":
                                    break

                            except json.JSONDecodeError:
                                continue

                self.circuit_breaker.record_success()

            except httpx.TimeoutException as e:
                self.circuit_breaker.record_failure()
                raise LLMTimeoutError(self.PROVIDER_NAME, self.timeout) from e
            except httpx.ConnectError as e:
                self.circuit_breaker.record_failure()
                raise LLMConnectionError(self.PROVIDER_NAME, self.base_url) from e
            except Exception as e:
                self.circuit_breaker.record_failure()
                if isinstance(e, LLMClientError):
                    raise
                raise LLMStreamError(self.PROVIDER_NAME, str(e)) from e

        return stream_generator()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
