"""Unit tests for the pluggable graph checkpointer and per-job thread id.

Covers ``IntegratedFramework`` checkpoint-backend selection, the sqlite-absent construction
error, and per-job config/thread-id resolution. Maps to spec ``strategos_langgraph_hardening``
AC-4. The helpers are exercised on a ``__new__`` instance to avoid constructing the full
framework (which needs a model adapter).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="graph package import chain requires numpy")

from src.framework.graph.integrated import IntegratedFramework  # noqa: E402
from src.framework.graph.schema import GraphConstructionError  # noqa: E402


@pytest.fixture
def framework() -> IntegratedFramework:
    # Bypass __init__: these helpers do not depend on constructed state.
    return IntegratedFramework.__new__(IntegratedFramework)


class TestResolveCheckpointer:
    def test_injected_checkpointer_wins(self, framework):
        sentinel = object()
        settings = SimpleNamespace(GRAPH_CHECKPOINT_BACKEND="sqlite")
        assert framework._resolve_checkpointer(settings, sentinel) is sentinel

    def test_memory_backend_returns_saver(self, framework):
        settings = SimpleNamespace(GRAPH_CHECKPOINT_BACKEND="memory")
        saver = framework._resolve_checkpointer(settings, None)
        assert saver is not None  # langgraph MemorySaver/InMemorySaver

    def test_none_settings_defaults_to_memory(self, framework):
        saver = framework._resolve_checkpointer(None, None)
        assert saver is not None

    def test_sqlite_without_extra_raises_at_construction(self, framework):
        settings = SimpleNamespace(GRAPH_CHECKPOINT_BACKEND="sqlite", GRAPH_CHECKPOINT_SQLITE_PATH=None)
        # The optional dependency is not installed in the test environment, so selecting it
        # must raise rather than silently fall back to an ephemeral saver.
        with pytest.raises(GraphConstructionError, match="langgraph-checkpoint-sqlite"):
            framework._resolve_checkpointer(settings, None)


class TestResolveConfig:
    def test_default_thread_id(self):
        assert IntegratedFramework._resolve_config(None, None) == {"configurable": {"thread_id": "default"}}

    def test_per_job_thread_id(self):
        cfg = IntegratedFramework._resolve_config(None, "run1:taskA:0")
        assert cfg == {"configurable": {"thread_id": "run1:taskA:0"}}

    def test_explicit_config_respected(self):
        explicit = {"configurable": {"thread_id": "explicit"}, "extra": 1}
        assert IntegratedFramework._resolve_config(explicit, "ignored") is explicit

    def test_resolve_thread_id_reads_config(self):
        assert IntegratedFramework._resolve_thread_id({"configurable": {"thread_id": "abc"}}) == "abc"
        assert IntegratedFramework._resolve_thread_id({}) == "default"


class TestTraceScopeRestore:
    def test_start_end_trace_restores_prior_context(self, framework):
        from src.framework.graph.tracing import set_trace_context, snapshot_trace_context
        from src.observability.logging import peek_correlation_id, set_correlation_id

        framework.trace_recorder = None
        set_correlation_id("outer-corr")
        set_trace_context("outer-run", "outer-thread")

        scope = framework._start_trace({"configurable": {"thread_id": "job1"}})
        # During the run, the correlation id and trace context reflect the new run.
        assert peek_correlation_id() == scope.run_id
        assert snapshot_trace_context() == (scope.run_id, "job1")

        framework._end_trace(scope)
        # After the run, the prior context is restored (no leakage onto the same task).
        assert peek_correlation_id() == "outer-corr"
        assert snapshot_trace_context() == ("outer-run", "outer-thread")
