"""Unit tests for StreamingService (SSE adaptation over a framework event stream)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.api.streaming import (
    SSE_DATA_PREFIX,
    SSE_DONE_PAYLOAD,
    StreamingDisabledError,
    StreamingService,
)


def _settings(enable_streaming: bool = True) -> SimpleNamespace:
    return SimpleNamespace(ENABLE_STREAMING=enable_streaming)


class _FakeFramework:
    """Fake framework exposing astream_events as an async generator."""

    def __init__(self, events):
        self._events = events
        self.calls: list[dict] = []

    async def astream_events(self, query, use_rag=True, use_mcts=False, config=None, include_types=None):
        self.calls.append(
            {
                "query": query,
                "use_rag": use_rag,
                "use_mcts": use_mcts,
                "config": config,
                "include_types": include_types,
            }
        )
        for event in self._events:
            yield event


@pytest.fixture
def sample_events():
    return [
        {"event_type": "on_chain_start", "name": "entry", "data": {}, "metadata": {}},
        {"event_type": "on_llm_stream", "name": "hrm", "token": "hello", "data": {}, "metadata": {}},
        {"event_type": "on_chain_end", "name": "synthesize", "data": {"output": "done"}, "metadata": {}},
    ]


def test_enabled_reflects_settings():
    svc = StreamingService(framework=MagicMock(), settings=_settings(True))
    assert svc.enabled is True
    svc_off = StreamingService(framework=MagicMock(), settings=_settings(False))
    assert svc_off.enabled is False


def test_format_sse_includes_event_and_data_lines():
    event = {"event_type": "on_chain_start", "name": "entry", "data": {"k": "v"}}
    record = StreamingService.format_sse(event)
    lines = record.rstrip("\n").split("\n")
    assert lines[0] == "event: on_chain_start"
    assert lines[1].startswith(SSE_DATA_PREFIX)
    assert record.endswith("\n\n")
    payload = json.loads(lines[1][len(SSE_DATA_PREFIX) :])
    assert payload["name"] == "entry"
    assert payload["data"] == {"k": "v"}


def test_format_sse_without_event_type_omits_event_line():
    record = StreamingService.format_sse({"name": "x", "data": {}})
    lines = record.rstrip("\n").split("\n")
    assert len(lines) == 1
    assert lines[0].startswith(SSE_DATA_PREFIX)


def test_format_sse_serializes_non_json_values():
    # default=str path: an object that is not natively JSON serializable
    record = StreamingService.format_sse({"event_type": "x", "data": {"obj": object()}})
    assert SSE_DATA_PREFIX in record


def test_format_sse_done_sentinel():
    done = StreamingService.format_sse_done()
    assert SSE_DONE_PAYLOAD in done
    assert done.endswith("\n\n")


@pytest.mark.asyncio
async def test_stream_events_yields_all_and_passes_flags(sample_events):
    fw = _FakeFramework(sample_events)
    svc = StreamingService(framework=fw, settings=_settings(True))

    collected = [e async for e in svc.stream_events("q", use_rag=False, use_mcts=True, include_types=["on_llm_stream"])]

    assert collected == sample_events
    assert fw.calls[0]["use_rag"] is False
    assert fw.calls[0]["use_mcts"] is True
    assert fw.calls[0]["include_types"] == ["on_llm_stream"]


@pytest.mark.asyncio
async def test_stream_events_raises_when_disabled():
    svc = StreamingService(framework=_FakeFramework([]), settings=_settings(False))
    with pytest.raises(StreamingDisabledError):
        async for _ in svc.stream_events("q"):
            pass


@pytest.mark.asyncio
async def test_stream_sse_emits_records_and_done(sample_events):
    fw = _FakeFramework(sample_events)
    svc = StreamingService(framework=fw, settings=_settings(True))

    records = [r async for r in svc.stream_sse("q")]

    # one record per event, plus a terminal [DONE]
    assert len(records) == len(sample_events) + 1
    assert all(r.endswith("\n\n") for r in records)
    assert SSE_DONE_PAYLOAD in records[-1]
    first_payload = json.loads(records[0].rstrip("\n").split("\n")[-1][len(SSE_DATA_PREFIX) :])
    assert first_payload["event_type"] == "on_chain_start"


@pytest.mark.asyncio
async def test_stream_sse_disabled_raises():
    svc = StreamingService(framework=_FakeFramework([]), settings=_settings(False))
    with pytest.raises(StreamingDisabledError):
        async for _ in svc.stream_sse("q"):
            pass


def test_default_settings_used_when_not_injected(monkeypatch):
    monkeypatch.setattr("src.api.streaming.get_settings", lambda: _settings(True))
    svc = StreamingService(framework=MagicMock())
    assert svc.enabled is True
