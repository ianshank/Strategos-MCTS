"""Integration test: retry-with-backoff fires through the compiled LangGraph, and the
execution trace records the resulting attempt count.

Maps to spec ``strategos_langgraph_hardening`` AC-2 / AC-3. Drives a real compiled graph with
an injected HRM agent that raises a transient error twice before succeeding.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("numpy", reason="graph package import chain requires numpy")
pytest.importorskip("langgraph", reason="langgraph required to compile the graph")

from src.framework.graph.builder import GraphBuilder
from src.framework.graph.retry import NodeRetryPolicy
from src.framework.graph.tracing import (
    GraphTraceRecorder,
    JsonlTraceSink,
    load_trace,
    set_trace_context,
)

pytestmark = [pytest.mark.integration]


def _build_app(tmp_path, hrm_side_effect, retryable=(ConnectionError,)):
    hrm = AsyncMock()
    hrm.process.side_effect = hrm_side_effect
    trm = AsyncMock()
    trm.process.return_value = {"response": "trm", "metadata": {"final_quality_score": 0.7}}
    model = AsyncMock()
    model.generate.return_value = type("Resp", (), {"text": "final synthesized response"})()

    recorder = GraphTraceRecorder(sink=JsonlTraceSink(tmp_path), metrics=None)
    policy = NodeRetryPolicy(
        enabled=True,
        max_attempts=3,
        initial_delay_seconds=0.0,
        backoff_factor=1.0,
        retryable_exceptions=retryable,
    )
    builder = GraphBuilder(
        hrm_agent=hrm,
        trm_agent=trm,
        model_adapter=model,
        logger=logging.getLogger("test.retry.integration"),
        vector_store=None,
        max_iterations=1,
        consensus_threshold=0.99,  # force the max-iterations -> synthesize path
        enable_parallel_agents=False,
        retry_policy=policy,
        trace_recorder=recorder,
    )
    app = builder.build_graph().compile()
    return app, hrm, recorder


def _initial_state():
    return {
        "query": "explain hierarchical reasoning",
        "use_rag": False,
        "use_mcts": False,
        "iteration": 0,
        "max_iterations": 1,
        "agent_outputs": [],
    }


@pytest.mark.asyncio
async def test_transient_error_retried_then_succeeds(tmp_path):
    ok = {"response": "hrm answer", "metadata": {"decomposition_quality_score": 0.8}}
    app, hrm, recorder = _build_app(
        tmp_path,
        hrm_side_effect=[ConnectionError("flaky 1"), ConnectionError("flaky 2"), ok],
    )
    run_id = "faultrun"
    set_trace_context(run_id, "t1")
    recorder.begin_run(run_id)

    result = await app.ainvoke(_initial_state())

    # HRM was retried to a third, successful attempt.
    assert hrm.process.call_count == 3
    assert result.get("final_response") == "final synthesized response"

    # The trace attributes 3 attempts to the hrm_agent transition.
    events = load_trace(tmp_path, run_id)
    hrm_events = [e for e in events if e.node == "hrm_agent"]
    assert hrm_events and hrm_events[0].attempts == 3
    assert hrm_events[0].status == "ok"


@pytest.mark.asyncio
async def test_non_transient_error_not_retried(tmp_path):
    app, hrm, recorder = _build_app(
        tmp_path,
        hrm_side_effect=[ValueError("permanent")],  # not in the allowlist
    )
    set_trace_context("faultrun2", "t1")
    recorder.begin_run("faultrun2")

    with pytest.raises(ValueError):
        await app.ainvoke(_initial_state())

    # A non-allowlisted error propagates on the first attempt, no retry.
    assert hrm.process.call_count == 1
