"""Unit tests for :class:`HarnessFactory` wiring.

These cover the factory's construction paths without real LLM I/O: the
record/replay-aware ``create_llm``, the fully-wired ``create_runner``, and
``create_ralph``. The LLM client is stubbed so no network/keys are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.framework.harness.factories import HarnessFactory
from src.framework.harness.loop.runner import HarnessRunner
from src.framework.harness.ralph import RalphLoop
from src.framework.harness.settings import HarnessSettings
from src.framework.harness.tools import ToolRegistry

pytestmark = pytest.mark.unit


def _stub_llm_factory(monkeypatch: pytest.MonkeyPatch, inner: object) -> None:
    """Patch factories.LLMClientFactory so create_from_settings returns ``inner``."""
    fake = MagicMock()
    fake.create_from_settings.return_value = inner
    monkeypatch.setattr(
        "src.framework.harness.factories.LLMClientFactory",
        MagicMock(return_value=fake),
    )


def test_create_llm_replay_mode(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With REPLAY_DIR set, create_llm returns a replay-mode client (no inner)."""
    # create_llm prioritizes replay when both dirs are set; clear the opposite so
    # this test unambiguously exercises the replay branch.
    monkeypatch.delenv("HARNESS_RECORD_DIR", raising=False)
    monkeypatch.setenv("HARNESS_REPLAY_DIR", str(tmp_path / "cass"))
    factory = HarnessFactory(harness_settings=HarnessSettings())
    client = factory.create_llm()
    assert client.mode == "replay"
    assert client.inner is None


def test_create_llm_record_mode(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With RECORD_DIR set (and no REPLAY_DIR), create_llm wraps the inner client."""
    monkeypatch.delenv("HARNESS_REPLAY_DIR", raising=False)
    monkeypatch.setenv("HARNESS_RECORD_DIR", str(tmp_path / "rec"))
    inner = MagicMock(name="inner-llm")
    _stub_llm_factory(monkeypatch, inner)
    factory = HarnessFactory(harness_settings=HarnessSettings())
    client = factory.create_llm()
    assert client.mode == "record"
    assert client.inner is inner


def test_create_llm_plain(monkeypatch: pytest.MonkeyPatch, harness_settings: HarnessSettings) -> None:
    """Without record/replay dirs, create_llm returns the inner client directly."""
    # Guard against record/replay env leaking from other tests in the suite.
    monkeypatch.delenv("HARNESS_REPLAY_DIR", raising=False)
    monkeypatch.delenv("HARNESS_RECORD_DIR", raising=False)
    sentinel = MagicMock(name="inner-llm")
    _stub_llm_factory(monkeypatch, sentinel)
    factory = HarnessFactory(harness_settings=HarnessSettings())
    assert factory.create_llm() is sentinel


def test_create_tool_registry_without_memory(harness_settings: HarnessSettings) -> None:
    """A registry can be built with no memory store (memory tools skipped)."""
    factory = HarnessFactory(harness_settings=harness_settings)
    registry = factory.create_tool_registry(memory_store=None)
    assert isinstance(registry, ToolRegistry)


def test_create_runner_builds_wired_runner(monkeypatch: pytest.MonkeyPatch, harness_settings: HarnessSettings) -> None:
    """create_runner composes a fully-wired HarnessRunner (LLM stubbed)."""
    monkeypatch.setattr(HarnessFactory, "create_llm", lambda self: MagicMock(name="llm"))
    factory = HarnessFactory(harness_settings=harness_settings)
    runner = factory.create_runner()
    assert isinstance(runner, HarnessRunner)


def test_create_ralph_wraps_runner(harness_settings: HarnessSettings) -> None:
    """create_ralph returns a RalphLoop around the given runner."""
    factory = HarnessFactory(harness_settings=harness_settings)
    loop = factory.create_ralph(MagicMock(name="runner"))
    assert isinstance(loop, RalphLoop)
