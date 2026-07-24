"""Unit tests for structured execution-trace logging.

Covers ``src/framework/graph/tracing.py`` — digests, the per-run recorder/sink, node-wrapper
transition recording (sync + async, ok + error, attempt propagation), and trace reconstruction.
Maps to spec ``strategos_langgraph_hardening`` AC-3.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

pytest.importorskip("numpy", reason="graph package import chain requires numpy")

from src.framework.graph.retry import NodeRetryPolicy, with_node_retry  # noqa: E402
from src.framework.graph.tracing import (  # noqa: E402
    GraphTraceRecorder,
    JsonlTraceSink,
    NodeTraceEvent,
    event_from_dict,
    load_trace,
    make_traced_node,
    reconstruct_path,
    set_trace_context,
    state_digest,
)


class ListSink:
    """In-memory sink capturing appended records for assertions."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(self, run_id: str, record: dict) -> None:
        self.records.append(dict(record))


def _fixed_clock():
    return datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)


def _recorder(sink=None, metrics=None):
    return GraphTraceRecorder(sink=sink, clock=_fixed_clock, metrics=metrics, slog=MagicMock())


class TestStateDigest:
    def test_deterministic_and_order_invariant(self):
        a = {"x": 1, "y": [1, 2], "z": "v"}
        b = {"z": "v", "y": [1, 2], "x": 1}
        assert state_digest(a) == state_digest(b)

    def test_length_is_configured(self):
        assert len(state_digest({"a": 1})) == 16

    def test_non_serializable_values_safe(self):
        # An arbitrary object is coerced via default=str rather than raising.
        digest = state_digest({"node": object(), "n": 3})
        assert isinstance(digest, str) and len(digest) == 16

    def test_different_states_differ(self):
        assert state_digest({"a": 1}) != state_digest({"a": 2})


class TestEventRoundTrip:
    def test_to_dict_from_dict(self):
        event = NodeTraceEvent(
            run_id="r",
            thread_id="t",
            node="entry",
            seq=0,
            ts="2026-07-23T12:00:00+00:00",
            duration_ms=1.5,
            status="ok",
            attempts=1,
            input_digest="abc",
            output_digest="def",
            error=None,
        )
        assert event_from_dict(event.to_dict()) == event


class TestRecorder:
    def test_records_to_sink_metrics_and_logger(self):
        sink = ListSink()
        metrics = MagicMock()
        rec = _recorder(sink=sink, metrics=metrics)
        rec.begin_run("run1")
        rec.record(
            run_id="run1",
            thread_id="t",
            node="entry",
            duration_ms=2.0,
            status="ok",
            attempts=1,
            input_digest="i",
            output_digest="o",
            error=None,
        )
        assert len(sink.records) == 1
        assert sink.records[0]["node"] == "entry"
        metrics.record_node_timing.assert_called_once_with("entry", 2.0)
        rec._slog.info.assert_called_once()

    def test_sequence_increments_per_run(self):
        rec = _recorder()
        rec.begin_run("run1")
        seqs = [
            rec.record(
                run_id="run1",
                thread_id="t",
                node=f"n{i}",
                duration_ms=1.0,
                status="ok",
                attempts=1,
                input_digest="i",
                output_digest="o",
                error=None,
            ).seq
            for i in range(3)
        ]
        assert seqs == [0, 1, 2]

    def test_runs_have_independent_sequences(self):
        rec = _recorder()
        rec.begin_run("a")
        rec.begin_run("b")
        first = rec.record(
            run_id="a",
            thread_id="t",
            node="x",
            duration_ms=1.0,
            status="ok",
            attempts=1,
            input_digest="i",
            output_digest="o",
            error=None,
        ).seq
        second = rec.record(
            run_id="b",
            thread_id="t",
            node="y",
            duration_ms=1.0,
            status="ok",
            attempts=1,
            input_digest="i",
            output_digest="o",
            error=None,
        ).seq
        assert first == 0 and second == 0


class TestMakeTracedNode:
    async def test_async_node_ok(self):
        sink = ListSink()
        rec = _recorder(sink=sink)
        rec.begin_run("run1")
        set_trace_context("run1", "th1")

        async def handler(state):
            return {"out": 1}

        result = await make_traced_node(rec, handler, "hrm_agent")({"q": "hi"})
        assert result == {"out": 1}
        assert sink.records[0]["status"] == "ok"
        assert sink.records[0]["node"] == "hrm_agent"
        assert sink.records[0]["output_digest"] is not None

    def test_sync_node_ok(self):
        sink = ListSink()
        rec = _recorder(sink=sink)
        rec.begin_run("run1")
        set_trace_context("run1", "th1")

        def handler(state):
            return {"out": 2}

        result = make_traced_node(rec, handler, "entry")({"q": "hi"})
        assert result == {"out": 2}
        assert sink.records[0]["status"] == "ok"

    async def test_error_status_recorded_and_reraised(self):
        sink = ListSink()
        rec = _recorder(sink=sink)
        rec.begin_run("run1")
        set_trace_context("run1", "th1")

        async def handler(state):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await make_traced_node(rec, handler, "synthesize")({"q": "hi"})
        assert sink.records[0]["status"] == "error"
        assert sink.records[0]["output_digest"] is None
        assert "ValueError" in sink.records[0]["error"]

    async def test_attempts_reflect_inner_retry(self):
        sink = ListSink()
        rec = _recorder(sink=sink)
        rec.begin_run("run1")
        set_trace_context("run1", "th1")
        policy = NodeRetryPolicy(
            enabled=True,
            max_attempts=3,
            initial_delay_seconds=0.0,
            backoff_factor=1.0,
            retryable_exceptions=(ConnectionError,),
        )

        calls = {"n": 0}

        async def handler(state):
            async def io():
                calls["n"] += 1
                if calls["n"] < 3:
                    raise ConnectionError("transient")
                return "ok"

            return await with_node_retry(policy, "hrm_agent", io)()

        await make_traced_node(rec, handler, "hrm_agent")({"q": "hi"})
        assert sink.records[0]["attempts"] == 3


class TestLoadTraceAndPath:
    def test_load_orders_by_seq_and_reconstructs_path(self, tmp_path):
        sink = JsonlTraceSink(tmp_path)
        rec = GraphTraceRecorder(sink=sink, clock=_fixed_clock, metrics=None, slog=MagicMock())
        rec.begin_run("run1")
        for node in ["entry", "retrieve_context", "synthesize"]:
            rec.record(
                run_id="run1",
                thread_id="t",
                node=node,
                duration_ms=1.0,
                status="ok",
                attempts=1,
                input_digest="i",
                output_digest="o",
                error=None,
            )
        events = load_trace(tmp_path, "run1")
        assert [e.seq for e in events] == [0, 1, 2]
        assert reconstruct_path(events) == ["entry", "retrieve_context", "synthesize"]

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_trace(tmp_path, "nope") == []


class TestEdgeCases:
    def test_current_run_id_reflects_context(self):
        from src.framework.graph.tracing import current_run_id

        set_trace_context("run-xyz", "th")
        assert current_run_id() == "run-xyz"

    def test_record_without_begin_run_uses_fallback_sequence(self):
        rec = _recorder()
        # No begin_run("solo") first; the recorder lazily creates a counter.
        event = rec.record(
            run_id="solo",
            thread_id="t",
            node="entry",
            duration_ms=1.0,
            status="ok",
            attempts=1,
            input_digest="i",
            output_digest="o",
            error=None,
        )
        assert event.seq == 0

    def test_sync_node_error_recorded_and_reraised(self):
        sink = ListSink()
        rec = _recorder(sink=sink)
        rec.begin_run("run1")
        set_trace_context("run1", "th1")

        def handler(state):
            raise RuntimeError("sync boom")

        with pytest.raises(RuntimeError):
            make_traced_node(rec, handler, "aggregate_results")({"q": "hi"})
        assert sink.records[0]["status"] == "error"
        assert "RuntimeError" in sink.records[0]["error"]

    def test_torn_trailing_line_tolerated(self, tmp_path):
        sink = JsonlTraceSink(tmp_path)
        rec = GraphTraceRecorder(sink=sink, clock=_fixed_clock, metrics=None, slog=MagicMock())
        rec.begin_run("run1")
        rec.record(
            run_id="run1",
            thread_id="t",
            node="entry",
            duration_ms=1.0,
            status="ok",
            attempts=1,
            input_digest="i",
            output_digest="o",
            error=None,
        )
        # Simulate a hard kill mid-write: append a partial JSON line.
        with (tmp_path / "run1.jsonl").open("a") as handle:
            handle.write("{partial-record")
        events = load_trace(tmp_path, "run1")
        assert len(events) == 1
        assert events[0].node == "entry"
