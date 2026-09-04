"""The REST service, started through its own lifespan and answered through its own graph.

``docs/CLAIM_LEDGER.md`` grades the LangGraph-introspection and observability claims
(CL-2, CL-10) ``PARTIAL`` because the endpoints are unit-tested against mocks and no test
drives them through a started application. This module starts the real app: the lifespan
builds the authenticator, the ``FrameworkService`` and the integrated graph, and the
requests below are answered by that graph rather than by an ``AsyncMock``.

**The network guard.** The framework is deliberately pushed onto its documented degraded
path by making LLM-client creation fail, which is the only supported way to get a
``MockLLMClient`` (``src/api/framework_service.py``). The test asserts
``framework_degraded`` is true on ``/ready`` *before* issuing any query, so a regression
that silently constructed a live client would fail here rather than after reaching a
provider. That is ``CHARTER.md`` NG-8 applied to an end-to-end test, and it is also what
makes the degraded-mode reporting itself covered: nothing in the tree asserted the
``framework_degraded`` flag before.

Written synchronously on purpose: ``TestClient`` as a context manager is what runs the
lifespan, and driving its blocking portal from inside a running event loop (this suite
sets ``asyncio_mode = auto``) would deadlock.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.api]

# Imported normally, not via `pytest.importorskip`. The `api` extra is guarded for this
# module by `_require_or_ignore` in tests/conftest.py, which under STRICT_OPTIONAL_DEPS
# (as the CI test job sets) aborts collection with an actionable message instead of
# skipping. importorskip would skip silently even in strict mode, quietly shrinking the
# PR-gating e2e suite — a suite that can shrink without going red gates nothing.
from fastapi.testclient import TestClient

from src.api.rest_server import app
from src.config.settings import reset_settings
from src.framework.factories import LLMClientFactory

#: Injected through ``API_KEYS``, which the lifespan reads to build the authenticator.
E2E_API_KEY = "e2e-rest-key"

#: Why the LLM client refuses to build. The message is asserted nowhere; it exists so a
#: human reading the captured log knows the failure was deliberate.
REFUSAL_MESSAGE = "LLM client creation refused by the end-to-end network guard"


@pytest.fixture
def rest_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The real app, started through its lifespan, with the provider client refused."""
    monkeypatch.setenv("API_KEYS", E2E_API_KEY)
    # Accepted under the default development DEPLOYMENT_ENV; Settings refuses this
    # combination outright in the production environments, which is the point of the flag.
    monkeypatch.setenv("ALLOW_MOCK_LLM_FALLBACK", "true")

    # get_settings() is lru-cached and thousands of earlier tests may have warmed it.
    reset_settings()
    try:
        with patch.object(LLMClientFactory, "create_from_settings", side_effect=RuntimeError(REFUSAL_MESSAGE)):
            with TestClient(app) as client:
                yield client
    finally:
        # The lifespan's shutdown resets the FrameworkService singleton; the settings
        # cache is this test's to clean up.
        reset_settings()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": E2E_API_KEY}


def test_service_reports_itself_degraded_before_serving(rest_client: Any) -> None:
    """``/health`` and ``/ready`` describe a started, deliberately degraded service.

    The degraded assertion is the network guard: it must hold before any query is issued.
    """
    health = rest_client.get("/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] in {"healthy", "degraded", "initializing"}
    assert health.json()["version"], "the health response carries no version"

    ready = rest_client.get("/ready")
    assert ready.status_code == 200, ready.text
    checks = ready.json()["checks"]
    assert checks["framework_ready"] is True, f"the framework did not start: {checks}"
    assert checks["framework_degraded"] is True, (
        "the service reports a non-degraded framework, which means a real LLM client was "
        f"constructed despite the refusal patch; a query would reach a provider. checks={checks}"
    )


def test_graph_structure_is_introspectable_at_runtime(rest_client: Any, auth_headers: dict[str, str]) -> None:
    """CL-2: the built graph describes itself over HTTP, as JSON and as mermaid."""
    structure = rest_client.get("/graph/structure", headers=auth_headers)
    assert structure.status_code == 200, structure.text
    payload = structure.json()
    assert payload["nodes"], "the served graph reports no nodes"
    node_ids = {node["id"] for node in payload["nodes"]}
    assert "entry" in node_ids, f"the graph has no entry node; got {sorted(node_ids)}"

    mermaid = rest_client.get("/graph/mermaid", headers=auth_headers)
    assert mermaid.status_code == 200, mermaid.text
    diagram = mermaid.json()["mermaid"]
    assert diagram.startswith("flowchart"), f"not a mermaid flowchart: {diagram[:80]!r}"
    # The rendering must describe the same graph, not a static picture.
    assert any(node_id in diagram for node_id in node_ids)


def test_metrics_endpoint_answers_honestly(rest_client: Any) -> None:
    """``/metrics`` either serves Prometheus text or says the dependency is absent.

    ``prometheus-client`` lives in its own extra, so both outcomes are correct; what would
    be wrong is a 404 (route lost) or a 500 (route broken).
    """
    response = rest_client.get("/metrics")
    assert response.status_code in (200, 501), response.text
    if response.status_code == 200:
        assert "text/plain" in response.headers.get("content-type", "")


def test_query_is_answered_through_the_real_graph(rest_client: Any, auth_headers: dict[str, str]) -> None:
    """A query traverses the started graph and returns a structured answer."""
    response = rest_client.post(
        "/query",
        json={"query": "What is Monte Carlo tree search?"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["response"], "the service returned an empty response body"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["agents_used"], "no agent was recorded as having run"


def test_query_requires_authentication(rest_client: Any) -> None:
    """The authenticator the lifespan built is actually enforced on the query path."""
    response = rest_client.post("/query", json={"query": "unauthenticated"})
    assert response.status_code in (401, 403, 422), response.text
