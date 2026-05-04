"""
LM Studio local LLM client adapter.

Implements the LLMClient protocol for LM Studio's OpenAI-compatible API.
Designed for running local models with configurable endpoint.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.config.constants import (
    DEFAULT_LMSTUDIO_TEMPERATURE,
    DEFAULT_LMSTUDIO_TIMEOUT,
    DEFAULT_LMSTUDIO_URL,
)
from src.observability.logging import get_logger

from .base import BaseLLMClient, LLMResponse
from .exceptions import (
    LLMClientError,
    LLMConnectionError,
    LLMResponseParseError,
    LLMServerError,
    LLMStreamError,
    LLMTimeoutError,
)
from .model_presets import ModelPreset, ReasoningEffort
from .preset_applier import apply_preset

logger = get_logger(__name__)


class LMStudioClient(BaseLLMClient):
    """
    LM Studio local server client.

    LM Studio provides an OpenAI-compatible API for running local models.
    This client is optimized for local deployment with:
    - No authentication required (local)
    - Configurable base URL
    - No circuit breaker (local server expected to be stable)
    - Longer timeouts for large models
    """

    PROVIDER_NAME = "lmstudio"
    DEFAULT_BASE_URL = DEFAULT_LMSTUDIO_URL
    DEFAULT_MODEL = "local-model"  # LM Studio uses the loaded model

    def __init__(
        self,
        api_key: str | None = None,  # Not required for local
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_LMSTUDIO_TIMEOUT,  # Long timeout for local inference
        max_retries: int = 2,  # Fewer retries for local
        # Rate limiting
        rate_limit_per_minute: int | None = None,
        *,
        preset: ModelPreset | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ):
        """
        Initialize LM Studio client.

        Args:
            api_key: Not required for local server (ignored)
            model: Model identifier (often ignored by LM Studio, uses loaded model)
            base_url: Local server URL (default: http://localhost:1234/v1)
            timeout: Request timeout in seconds (default longer for local models)
            max_retries: Max retry attempts (fewer for local)
            rate_limit_per_minute: Rate limit for requests per minute (None to disable)
            preset: Optional :class:`ModelPreset` whose stops, reasoning flag,
                and default temperature are merged into every request payload.
                When ``None`` (default), no preset shaping is applied.
            reasoning_effort: Hint forwarded as ``reasoning.effort`` in the
                payload, only when ``preset.reasoning`` is ``True``.
        """
        import os

        # Allow overriding via environment variable
        base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", self.DEFAULT_BASE_URL)

        super().__init__(
            api_key=api_key or "not-required",  # Placeholder
            model=model or self.DEFAULT_MODEL,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            rate_limit_per_minute=rate_limit_per_minute,
        )

        self._client: httpx.AsyncClient | None = None
        self._preset: ModelPreset | None = preset
        self._reasoning_effort: ReasoningEffort | None = reasoning_effort

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}

            # Add auth header if provided (some local servers may require it)
            if self.api_key and self.api_key != "not-required":
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def check_health(self) -> bool:
        """
        Check if LM Studio server is running.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return bool(response.status_code == 200)
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available models on the LM Studio server.

        Returns:
            List of model information dicts
        """
        try:
            client = await self._get_client()
            response = await client.get("/models")
            if response.status_code == 200:
                data = response.json()
                return list(data.get("data", []))
            return []
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses from LM Studio server."""
        status_code = response.status_code

        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text

        if status_code >= 500:
            raise LLMServerError(self.PROVIDER_NAME, status_code, error_message)
        else:
            raise LLMClientError(error_message, self.PROVIDER_NAME, status_code=status_code)

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """
        Generate a response from LM Studio local model.

        Args:
            messages: Chat messages in OpenAI format
            prompt: Simple string prompt
            temperature: Sampling temperature. ``None`` means "use the active
                preset's default, or :data:`DEFAULT_LMSTUDIO_TEMPERATURE` if
                no preset is configured".
            max_tokens: Maximum tokens to generate
            tools: Tool definitions (limited support in local models)
            stream: If True, returns AsyncIterator
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            LLMResponse or AsyncIterator[str] for streaming
        """
        # Apply rate limiting before proceeding
        await self._apply_rate_limit()

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
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming generation."""
        client = await self._get_client()

        # Build request payload (OpenAI-compatible). We start with the
        # caller's explicit temperature (which may be None) so that the
        # preset applier can decide whether to fill it in.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(messages, prompt),
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        # Note: most local models don't support tools well
        if tools:
            logger.warning("Tool calling may not be fully supported by local models")
            payload["tools"] = tools

        # Add additional kwargs (e.g., top_p, repeat_penalty)
        for key in ["top_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        # Merge preset-derived fields. The applier honors the caller's
        # explicit temperature (``user_temperature=temperature``) and only
        # falls back to the preset default when ``temperature is None``.
        payload = apply_preset(
            payload,
            self._preset,
            reasoning_effort=self._reasoning_effort,
            user_temperature=temperature,
        )

        # Final guard: LM Studio requires a numeric temperature on the wire.
        if payload.get("temperature") is None:
            payload["temperature"] = DEFAULT_LMSTUDIO_TEMPERATURE

        # Retry logic for local server
        last_error: LLMClientError | None = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post("/chat/completions", json=payload)

                if response.status_code != 200:
                    self._handle_error_response(response)

                # Parse response
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    message = choice["message"]

                    usage = data.get("usage", {})
                    finish_reason = choice.get("finish_reason", "stop")

                    llm_response = LLMResponse(
                        text=message.get("content", ""),
                        usage=usage,
                        model=data.get("model", self.model),
                        raw_response=data,
                        finish_reason=finish_reason,
                    )

                    self._update_stats(llm_response)
                    return llm_response

                except (KeyError, json.JSONDecodeError) as e:
                    raise LLMResponseParseError(self.PROVIDER_NAME, response.text) from e

            except httpx.TimeoutException:
                last_error = LLMTimeoutError(self.PROVIDER_NAME, self.timeout)
                logger.warning(f"Attempt {attempt + 1} timed out, retrying...")
            except httpx.ConnectError:
                last_error = LLMConnectionError(self.PROVIDER_NAME, self.base_url)
                logger.warning(f"Attempt {attempt + 1} connection failed, retrying...")
            except LLMClientError:
                raise  # Don't retry client errors

        # All retries exhausted
        if last_error:
            raise last_error
        raise LLMConnectionError(self.PROVIDER_NAME, self.base_url)

    async def _generate_stream(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,  # noqa: ARG002
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        client = await self._get_client()

        # Build request payload. ``temperature`` flows through as the
        # caller's explicit value (possibly None) so the preset applier
        # can decide whether to fill it in from the active preset.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(messages, prompt),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        for key in ["top_p", "top_k", "repeat_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        # Merge preset-derived fields (stops, reasoning, default temp).
        payload = apply_preset(
            payload,
            self._preset,
            reasoning_effort=self._reasoning_effort,
            user_temperature=temperature,
        )

        # Final guard: ensure a numeric temperature for the server.
        if payload.get("temperature") is None:
            payload["temperature"] = DEFAULT_LMSTUDIO_TEMPERATURE

        async def stream_generator() -> AsyncIterator[str]:
            try:
                async with client.stream("POST", "/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        await response.aread()
                        self._handle_error_response(response)

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except (json.JSONDecodeError, KeyError):
                                continue

            except httpx.TimeoutException as e:
                raise LLMTimeoutError(self.PROVIDER_NAME, self.timeout) from e
            except httpx.ConnectError as e:
                raise LLMConnectionError(self.PROVIDER_NAME, self.base_url) from e
            except Exception as e:
                if isinstance(e, LLMClientError):
                    raise
                raise LLMStreamError(self.PROVIDER_NAME, str(e)) from e

        return stream_generator()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
