"""Unit tests for configurable node retry-with-backoff.

Covers ``src/framework/graph/retry.py`` — exception-allowlist resolution, policy
construction, and the retry wrapper's succeed-after-N / non-allowlisted / exhaustion
behaviour at a node's I/O boundary. Maps to spec ``strategos_langgraph_hardening`` AC-2.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="graph package import chain requires numpy")

from src.framework.graph.retry import (  # noqa: E402
    NodeRetryPolicy,
    get_node_attempts,
    is_retryable_node,
    policy_from_settings,
    reset_node_attempts,
    resolve_exception_types,
    with_node_retry,
)
from src.framework.graph.schema import GraphConstructionError  # noqa: E402


class TestResolveExceptionTypes:
    def test_builtin_names(self):
        assert resolve_exception_types(["TimeoutError", "ConnectionError"]) == (TimeoutError, ConnectionError)

    def test_dotted_paths(self):
        from src.adapters.llm.exceptions import LLMTimeoutError

        assert resolve_exception_types(["src.adapters.llm.exceptions.LLMTimeoutError"]) == (LLMTimeoutError,)

    def test_unknown_builtin_rejected(self):
        with pytest.raises(GraphConstructionError, match="Unknown builtin"):
            resolve_exception_types(["NotARealBuiltinError"])

    def test_bad_module_rejected(self):
        with pytest.raises(GraphConstructionError, match="Cannot import module"):
            resolve_exception_types(["nonexistent.module.SomeError"])

    def test_missing_attribute_rejected(self):
        with pytest.raises(GraphConstructionError, match="has no attribute"):
            resolve_exception_types(["src.adapters.llm.exceptions.NopeError"])

    def test_non_exception_rejected(self):
        with pytest.raises(GraphConstructionError, match="not an Exception subclass"):
            resolve_exception_types(["os.getcwd"])

    def test_empty_path_rejected(self):
        with pytest.raises(GraphConstructionError, match="Empty retry exception path"):
            resolve_exception_types(["  "])


class TestIsRetryableNode:
    def test_retryable_members(self):
        assert is_retryable_node("hrm_agent")
        assert is_retryable_node("synthesize")

    def test_adk_prefix_retryable(self):
        assert is_retryable_node("adk_deep_search")

    def test_deterministic_nodes_excluded(self):
        for node in ("entry", "route_decision", "aggregate_results", "evaluate_consensus"):
            assert not is_retryable_node(node)


class TestPolicyFromSettings:
    def _settings(self, **overrides):
        base = {
            "GRAPH_NODE_RETRY_ENABLED": True,
            "GRAPH_NODE_RETRY_MAX_ATTEMPTS": 3,
            "GRAPH_NODE_RETRY_INITIAL_DELAY_SECONDS": 0.5,
            "GRAPH_NODE_RETRY_BACKOFF_FACTOR": 2.0,
            "GRAPH_NODE_RETRY_EXCEPTIONS": ["TimeoutError"],
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_enabled_policy(self):
        policy = policy_from_settings(self._settings())
        assert policy.enabled
        assert policy.max_attempts == 3
        assert policy.retryable_exceptions == (TimeoutError,)

    def test_disabled_policy(self):
        policy = policy_from_settings(self._settings(GRAPH_NODE_RETRY_ENABLED=False))
        assert not policy.enabled
        assert policy.retryable_exceptions == ()

    def test_bad_allowlist_raises_at_construction(self):
        with pytest.raises(GraphConstructionError):
            policy_from_settings(self._settings(GRAPH_NODE_RETRY_EXCEPTIONS=["NotAThing"]))


def _policy(**overrides) -> NodeRetryPolicy:
    base = {
        "enabled": True,
        "max_attempts": 3,
        "initial_delay_seconds": 0.0,
        "backoff_factor": 1.0,
        "retryable_exceptions": (ConnectionError,),
    }
    base.update(overrides)
    return NodeRetryPolicy(**base)


class TestWithNodeRetryAsync:
    async def test_succeeds_after_retries(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        reset_node_attempts()
        result = await with_node_retry(_policy(), "hrm_agent", flaky)()
        assert result == "ok"
        assert calls["n"] == 3
        assert get_node_attempts() == 3

    async def test_non_allowlisted_not_retried(self):
        calls = {"n": 0}

        async def bad():
            calls["n"] += 1
            raise ValueError("not transient")

        with pytest.raises(ValueError):
            await with_node_retry(_policy(), "hrm_agent", bad)()
        assert calls["n"] == 1

    async def test_exhaustion_propagates(self):
        calls = {"n": 0}

        async def always():
            calls["n"] += 1
            raise ConnectionError("always")

        with pytest.raises(ConnectionError):
            await with_node_retry(_policy(max_attempts=2), "hrm_agent", always)()
        assert calls["n"] == 2

    async def test_disabled_policy_no_retry(self):
        calls = {"n": 0}

        async def bad():
            calls["n"] += 1
            raise ConnectionError("x")

        with pytest.raises(ConnectionError):
            await with_node_retry(_policy(enabled=False), "hrm_agent", bad)()
        assert calls["n"] == 1

    async def test_non_retryable_node_no_retry(self):
        calls = {"n": 0}

        async def bad():
            calls["n"] += 1
            raise ConnectionError("x")

        with pytest.raises(ConnectionError):
            await with_node_retry(_policy(), "entry", bad)()
        assert calls["n"] == 1

    async def test_empty_allowlist_no_retry(self):
        calls = {"n": 0}

        async def bad():
            calls["n"] += 1
            raise ConnectionError("x")

        with pytest.raises(ConnectionError):
            await with_node_retry(_policy(retryable_exceptions=()), "hrm_agent", bad)()
        assert calls["n"] == 1


class TestWithNodeRetrySync:
    def test_sync_callable_retried(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return "done"

        result = with_node_retry(_policy(), "retrieve_context", flaky)()
        assert result == "done"
        assert calls["n"] == 2
