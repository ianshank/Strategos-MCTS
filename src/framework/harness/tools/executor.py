"""Async tool executor: dispatches a :class:`ToolInvocation` against a registry.

The executor handles cross-cutting concerns — timeout enforcement, output
truncation, structured logging, and metric emission — so individual tool
handlers can stay tiny and focused.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time

from src.framework.harness.settings import HarnessSettings
from src.framework.harness.state import Observation, ToolInvocation
from src.framework.harness.tools.registry import ToolRegistry
from src.framework.harness.tools.truncation import truncate_with_spillover
from src.observability.logging import get_logger


class AsyncToolExecutor:
    """Run tool invocations against a :class:`ToolRegistry`."""

    def __init__(
        self,
        registry: ToolRegistry,
        settings: HarnessSettings,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._logger = logger or get_logger(__name__)

    def list_tools(self) -> list[str]:
        return self._registry.list_names()

    def tool_schemas(self) -> list[dict[str, object]]:
        """Return tool schemas in the OpenAI/Anthropic ``tools=`` shape."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters or {"type": "object", "properties": {}},
                },
            }
            for s in self._registry.schemas()
        ]

    async def execute(self, call: ToolInvocation, *, correlation_id: str) -> Observation:
        """Run a single tool invocation."""
        if not self._registry.has(call.tool_name):
            self._logger.warning("unknown tool name=%s call=%s", call.tool_name, call.id)
            return Observation(
                invocation_id=call.id,
                tool_name=call.tool_name,
                success=False,
                payload=f"unknown tool: {call.tool_name}",
                duration_ms=0.0,
                metadata={"reason": "unknown_tool"},
            )

        handler = self._registry.get_handler(call.tool_name)
        start = time.perf_counter()
        try:
            payload = await asyncio.wait_for(
                handler(dict(call.arguments)),
                timeout=self._settings.TOOL_DEFAULT_TIMEOUT_SECONDS,
            )
            success = True
            error_meta: dict[str, str] = {}
        except TimeoutError:
            payload = f"tool '{call.tool_name}' timed out after " f"{self._settings.TOOL_DEFAULT_TIMEOUT_SECONDS}s"
            success = False
            error_meta = {"reason": "timeout"}
            self._logger.warning(
                "tool timeout name=%s call=%s timeout=%.1fs",
                call.tool_name,
                call.id,
                self._settings.TOOL_DEFAULT_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            payload = f"tool '{call.tool_name}' raised {type(exc).__name__}: {exc}"
            success = False
            error_meta = {"reason": "exception", "exception": type(exc).__name__}
            self._logger.warning(
                "tool exception name=%s call=%s err=%s",
                call.tool_name,
                call.id,
                type(exc).__name__,
            )
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0

        truncated, spillover = truncate_with_spillover(
            payload,
            head_chars=self._settings.TOOL_OUTPUT_HEAD_CHARS,
            tail_chars=self._settings.TOOL_OUTPUT_TAIL_CHARS,
            spillover_dir=Path(self._settings.TOOL_OUTPUT_SPILLOVER_DIR),
            correlation_id=correlation_id,
            step_id=call.id,
            marker_template=self._settings.TOOL_OUTPUT_TRUNCATION_MARKER,
        )

        observation = Observation(
            invocation_id=call.id,
            tool_name=call.tool_name,
            success=success,
            payload=truncated,
            spillover_path=spillover,
            duration_ms=duration_ms,
            metadata=error_meta,
        )
        self._logger.info(
            "tool executed name=%s call=%s success=%s duration_ms=%.2f truncated=%s",
            call.tool_name,
            call.id,
            success,
            duration_ms,
            spillover is not None,
        )
        return observation


__all__ = ["AsyncToolExecutor"]
