"""Record / replay LLM client wrapper.

``ReplayLLMClient`` wraps any underlying :class:`LLMClient` and operates in
one of three modes (chosen at construction):

* ``record`` — every request is forwarded to the inner client and persisted
  to a cassette file alongside the response.
* ``replay`` — requests are answered solely from the cassette; if no entry
  matches, :class:`CassetteMiss` is raised.
* ``passthrough`` — neither records nor replays; identical to using the
  inner client directly. Useful in tests as the default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Literal

from src.adapters.llm.base import LLMClient, LLMResponse, LLMToolResponse, ToolCall
from src.framework.harness.replay.cassette import Cassette, hash_request
from src.observability.logging import get_logger

ReplayMode = Literal["record", "replay", "passthrough"]


class CassetteMiss(RuntimeError):
    """Raised in ``replay`` mode when a request has no recorded entry."""


@dataclass
class ReplayLLMClient:
    """Wrap an :class:`LLMClient` with record/replay semantics.

    Attributes:
        inner: The underlying real client (used in ``record`` and ``passthrough``).
        cassette: On-disk store for request/response pairs.
        mode: Operating mode; see module docstring.
    """

    inner: LLMClient | None
    cassette: Cassette
    mode: ReplayMode = "passthrough"
    logger: logging.Logger = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = get_logger(__name__)
        if self.mode in ("record", "passthrough") and self.inner is None:
            raise ValueError(f"ReplayLLMClient mode={self.mode!r} requires an inner client")

    def _build_request_payload(
        self,
        *,
        messages: list[dict] | None,
        prompt: str | None,
        temperature: float,
        max_tokens: int | None,
        tools: list[dict] | None,
        stream: bool,
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Project the call signature into a stable, hashable dict."""
        return {
            "messages": messages,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "stream": stream,
            "stop": stop,
            "kwargs": dict(sorted(kwargs.items())),
        }

    @staticmethod
    def _serialize_response(response: LLMResponse) -> dict[str, Any]:
        """Project an :class:`LLMResponse` (or its tool-call subclass) into JSON."""
        payload: dict[str, Any] = {
            "type": "tool_response" if isinstance(response, LLMToolResponse) else "response",
            "text": response.text,
            "usage": response.usage,
            "model": response.model,
            "finish_reason": response.finish_reason,
        }
        if isinstance(response, LLMToolResponse):
            payload["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments, "type": tc.type} for tc in response.tool_calls
            ]
        return payload

    @staticmethod
    def _deserialize_response(payload: dict[str, Any]) -> LLMResponse:
        """Inverse of :meth:`_serialize_response`."""
        if payload.get("type") == "tool_response":
            return LLMToolResponse(
                text=payload.get("text", ""),
                usage=payload.get("usage", {}) or {},
                model=payload.get("model", ""),
                finish_reason=payload.get("finish_reason", "stop"),
                tool_calls=[
                    ToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                        type=tc.get("type", "function"),
                    )
                    for tc in payload.get("tool_calls", [])
                ],
            )
        return LLMResponse(
            text=payload.get("text", ""),
            usage=payload.get("usage", {}) or {},
            model=payload.get("model", ""),
            finish_reason=payload.get("finish_reason", "stop"),
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
        """Mirrors the :class:`LLMClient` protocol with record/replay routing."""
        if stream:
            # Streaming responses are not currently recorded — fall through to inner.
            if self.inner is None:
                raise CassetteMiss("Streaming responses cannot be replayed without an inner client")
            return await self.inner.generate(
                messages=messages,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                stop=stop,
                **kwargs,
            )

        request_payload = self._build_request_payload(
            messages=messages,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=stream,
            stop=stop,
            kwargs=kwargs,
        )
        request_hash = hash_request(request_payload)

        if self.mode == "replay":
            entry = self.cassette.lookup(request_hash)
            if entry is None:
                raise CassetteMiss(f"No cassette entry for request_hash={request_hash}")
            self.logger.debug("replay hit hash=%s cassette=%s", request_hash, self.cassette.path)
            return self._deserialize_response(entry.response)

        assert self.inner is not None  # narrowed by __post_init__ for non-replay modes
        response = await self.inner.generate(
            messages=messages,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=False,
            stop=stop,
            **kwargs,
        )
        if isinstance(response, LLMResponse):
            if self.mode == "record":
                self.cassette.append(request_payload, self._serialize_response(response))
                self.logger.debug("recorded hash=%s cassette=%s", request_hash, self.cassette.path)
            return response

        # Non-streaming generate must return LLMResponse — anything else is a contract bug.
        raise TypeError(f"Inner client returned unsupported type {type(response).__name__}")


def make_replay_client(
    inner: LLMClient | None,
    cassette_dir: Path,
    *,
    mode: ReplayMode = "passthrough",
    cassette_name: str = "cassette.jsonl",
    logger: logging.Logger | None = None,
) -> ReplayLLMClient:
    """Convenience factory that constructs the cassette path under ``cassette_dir``."""
    cassette = Cassette(path=cassette_dir / cassette_name)
    return ReplayLLMClient(inner=inner, cassette=cassette, mode=mode, logger=logger or get_logger(__name__))


__all__ = ["CassetteMiss", "ReplayLLMClient", "ReplayMode", "make_replay_client"]
