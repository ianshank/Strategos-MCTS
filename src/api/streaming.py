"""
Streaming service for Server-Sent Events (SSE) over LangGraph execution.

Wraps an integrated-framework-like object (anything exposing the
``astream_events(...)`` async-iterator from
:class:`src.framework.graph.integrated.IntegratedFramework`) and adapts its
event dicts into SSE-ready payloads.

Design constraints:
- No FastAPI import. All logic lives here so it is coverage-bearing and the thin
  REST adapter (``src/api/rest_server.py``, coverage-omitted) stays trivial.
- Dependency-injection friendly: the framework is injected, never constructed.
- Configuration-driven: gated on ``settings.ENABLE_STREAMING``; no hardcoded values.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any, Protocol

from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger

logger = get_logger(__name__)

# SSE wire-format tokens. Centralized so the framing is not duplicated as
# magic strings throughout the module.
SSE_DATA_PREFIX = "data: "
SSE_EVENT_PREFIX = "event: "
SSE_RECORD_SEPARATOR = "\n\n"
# Terminal sentinel event emitted once the underlying stream is exhausted so SSE
# clients can deterministically detect completion.
SSE_DONE_PAYLOAD = "[DONE]"


class StreamingFramework(Protocol):
    """Structural protocol for the framework object StreamingService consumes."""

    def astream_events(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict[str, Any] | None = None,
        include_types: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]: ...


class StreamingDisabledError(RuntimeError):
    """Raised when streaming is requested while ``ENABLE_STREAMING`` is False."""


class StreamingService:
    """
    Adapt a framework's ``astream_events`` stream into SSE-ready event dicts.

    The service is intentionally thin: it delegates the actual streaming to the
    injected framework and is responsible only for gating, logging, and SSE
    formatting.
    """

    def __init__(
        self,
        framework: StreamingFramework,
        settings: Settings | None = None,
    ) -> None:
        self._framework = framework
        self._settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        """Whether streaming is enabled by configuration."""
        return bool(self._settings.ENABLE_STREAMING)

    @staticmethod
    def format_sse(event: dict[str, Any]) -> str:
        """
        Format an event dict into an SSE record string.

        Produces an optional ``event:`` line (when the event carries an
        ``event_type``) followed by a ``data:`` line carrying the JSON-encoded
        event, terminated by the SSE record separator.
        """
        payload = json.dumps(event, default=str)
        lines = []
        event_type = event.get("event_type")
        if event_type:
            lines.append(f"{SSE_EVENT_PREFIX}{event_type}")
        lines.append(f"{SSE_DATA_PREFIX}{payload}")
        return "\n".join(lines) + SSE_RECORD_SEPARATOR

    @staticmethod
    def format_sse_done() -> str:
        """Format the terminal SSE sentinel record signalling stream completion."""
        return f"{SSE_DATA_PREFIX}{SSE_DONE_PAYLOAD}{SSE_RECORD_SEPARATOR}"

    async def stream_events(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict[str, Any] | None = None,
        include_types: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yield standardized event dicts from the framework's event stream.

        Raises:
            StreamingDisabledError: When streaming is disabled by configuration.
        """
        if not self.enabled:
            logger.warning("Streaming requested but ENABLE_STREAMING is False")
            raise StreamingDisabledError("Streaming is disabled (ENABLE_STREAMING=False)")

        logger.info(
            "Starting event stream: query_len=%d, use_rag=%s, use_mcts=%s",
            len(query),
            use_rag,
            use_mcts,
        )

        event_count = 0
        async for event in self._framework.astream_events(
            query,
            use_rag=use_rag,
            use_mcts=use_mcts,
            config=config,
            include_types=include_types,
        ):
            event_count += 1
            yield event

        logger.info("Event stream complete: events_emitted=%d", event_count)

    async def stream_sse(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict[str, Any] | None = None,
        include_types: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Yield SSE-formatted record strings ready to write to an HTTP response.

        Emits one ``data:`` record per framework event, followed by a terminal
        ``[DONE]`` sentinel record.
        """
        async for event in self.stream_events(
            query,
            use_rag=use_rag,
            use_mcts=use_mcts,
            config=config,
            include_types=include_types,
        ):
            yield self.format_sse(event)
        yield self.format_sse_done()


__all__ = [
    "StreamingService",
    "StreamingFramework",
    "StreamingDisabledError",
    "SSE_DATA_PREFIX",
    "SSE_EVENT_PREFIX",
    "SSE_DONE_PAYLOAD",
]
