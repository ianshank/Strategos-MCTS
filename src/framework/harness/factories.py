"""Factory wiring for the harness.

Composes settings + LLM client + tools + memory + hooks + verifier into a
ready-to-run :class:`HarnessRunner` (and optional :class:`RalphLoop`). The
factory keeps the CLI and tests honest: nobody hand-instantiates internal
collaborators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.adapters.llm.base import LLMClient
from src.config.settings import Settings, get_settings
from src.framework.factories import LLMClientFactory
from src.framework.harness.context import DefaultContextInjector
from src.framework.harness.hooks import HookChain
from src.framework.harness.intent import DefaultIntentNormalizer
from src.framework.harness.loop.runner import HarnessRunner
from src.framework.harness.memory.markdown import MarkdownMemoryStore
from src.framework.harness.memory.tools import register_memory_tools
from src.framework.harness.planner import HeuristicPlanner, LLMPlanner
from src.framework.harness.ralph import RalphLoop
from src.framework.harness.replay import (
    SystemClock,
    make_replay_client,
)
from src.framework.harness.replay.clock import DeterministicClock, RecordingClock
from src.framework.harness.settings import HarnessPermissions, HarnessSettings, get_harness_settings
from src.framework.harness.tools import AsyncToolExecutor, ToolRegistry
from src.framework.harness.tools.builtins import register_builtin_tools
from src.framework.harness.verifier import AcceptanceCriteriaVerifier
from src.observability.logging import get_logger


@dataclass
class HarnessFactory:
    """Construct fully-wired harness runners from settings."""

    settings: Settings | None = None
    harness_settings: HarnessSettings | None = None
    perms: HarnessPermissions | None = None
    logger: logging.Logger | None = None

    def _resolve(self) -> tuple[Settings, HarnessSettings, HarnessPermissions, logging.Logger]:
        return (
            self.settings or get_settings(),
            self.harness_settings or get_harness_settings(),
            self.perms or HarnessPermissions(),
            self.logger or get_logger(__name__),
        )

    def create_llm(self) -> LLMClient:
        """Construct the underlying LLM client (record/replay-aware)."""
        s, hs, _, log = self._resolve()
        if hs.REPLAY_DIR is not None:
            log.info("harness LLM in replay mode dir=%s", hs.REPLAY_DIR)
            return make_replay_client(inner=None, cassette_dir=hs.REPLAY_DIR, mode="replay", logger=log)
        inner = LLMClientFactory(settings=s).create_from_settings()
        if hs.RECORD_DIR is not None:
            log.info("harness LLM in record mode dir=%s", hs.RECORD_DIR)
            return make_replay_client(inner=inner, cassette_dir=hs.RECORD_DIR, mode="record", logger=log)
        return inner

    def create_clock(self) -> RecordingClock:
        """Pick a clock based on ``HARNESS_DETERMINISTIC_CLOCK``."""
        _, hs, _, _ = self._resolve()
        if hs.DETERMINISTIC_CLOCK:
            seed = hs.SEED if hs.SEED is not None else 0
            return DeterministicClock(seed=seed)
        return SystemClock()

    def create_memory_store(self) -> MarkdownMemoryStore:
        _, hs, _, log = self._resolve()
        return MarkdownMemoryStore(settings=hs, logger=log.getChild("memory"))

    def create_tool_registry(
        self,
        *,
        memory_store: MarkdownMemoryStore | None,
        correlation_id: str | None = None,
        shell_allowlist: list[str] | None = None,
    ) -> ToolRegistry:
        """Register builtin + memory tools onto a fresh registry."""
        _, hs, perms, _ = self._resolve()
        registry = ToolRegistry()
        register_builtin_tools(
            registry,
            root=Path.cwd(),
            perms=perms,
            correlation_id=correlation_id,
            shell_allowlist=shell_allowlist,
        )
        if memory_store is not None:
            register_memory_tools(registry, memory_store)
        del hs  # unused but documents the dependency
        return registry

    def create_runner(
        self,
        *,
        memory_store: MarkdownMemoryStore | None = None,
        hooks: HookChain | None = None,
        shell_allowlist: list[str] | None = None,
    ) -> HarnessRunner:
        """Construct a full :class:`HarnessRunner`."""
        _, hs, _, log = self._resolve()
        memory_store = memory_store or self.create_memory_store()
        llm = self.create_llm()
        registry = self.create_tool_registry(memory_store=memory_store, shell_allowlist=shell_allowlist)
        executor = AsyncToolExecutor(registry, hs, logger=log.getChild("tools"))
        intent = DefaultIntentNormalizer(logger=log.getChild("intent"))
        planner = (
            LLMPlanner(llm, max_tokens=hs.PLANNER_MAX_TOKENS, logger=log.getChild("planner"))
            if hs.PLANNER_ENABLED
            else HeuristicPlanner(logger=log.getChild("planner"))
        )
        injector = DefaultContextInjector(memory=memory_store, logger=log.getChild("context"))
        verifier = AcceptanceCriteriaVerifier(logger=log.getChild("verifier"))

        async def persist(event: dict[str, object]) -> None:
            await memory_store.append_event(event)

        return HarnessRunner(
            settings=hs,
            llm=llm,
            intent=intent,
            planner=planner,
            context_injector=injector,
            tool_executor=executor,
            verifier=verifier,
            hooks=hooks or HookChain(),
            clock=self.create_clock(),
            persist=persist,
            logger=log.getChild("runner"),
        )

    def create_ralph(self, runner: HarnessRunner, spec_path: Path | None = None) -> RalphLoop:
        _, hs, _, log = self._resolve()
        return RalphLoop(runner=runner, settings=hs, spec_path=spec_path, logger=log.getChild("ralph"))


__all__ = ["HarnessFactory"]
