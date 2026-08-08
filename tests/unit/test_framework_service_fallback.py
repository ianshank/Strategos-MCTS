"""
Regression tests for the framework-initialization fallback path.

This branch is the one that failed silently in production. It previously caught
only ``(ImportError, NotImplementedError)``, so a ``TypeError`` raised while
constructing the integrated framework escaped it entirely and left
``FrameworkService.framework`` as ``None`` — every /query, /query-stream and
/graph route then returned 503 with nothing in the logs naming the cause.

It now catches broadly but distinguishes an expected missing dependency from an
unexpected defect, logging the latter at ``exception`` level with a traceback.
That distinction is made with ``isinstance(e, ImportError | NotImplementedError)``
— a PEP 604 union, valid as the second argument to ``isinstance`` since Python
3.10, and this project declares ``requires-python = ">=3.10"``.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi", reason="requires the [api] extra")

import src.api.framework_service as framework_service  # noqa: E402
import src.framework.graph as framework_graph  # noqa: E402

pytestmark = pytest.mark.unit


class _ForcedFailure(Exception):
    """An exception type the fallback does not expect."""


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch):
    """A fresh FrameworkService, isolated from the module-level singleton."""
    monkeypatch.setattr(framework_service.FrameworkService, "_instance", None, raising=False)
    settings = framework_service.get_settings()
    return framework_service.FrameworkService(
        config=framework_service.FrameworkConfig.from_settings(settings),
        settings=settings,
    )


class TestUnexpectedExceptionDuringInit:
    """A defect must degrade visibly, not take the whole product surface down."""

    @pytest.mark.asyncio
    async def test_unexpected_exception_falls_back_instead_of_escaping(
        self, service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args, **kwargs):
            raise _ForcedFailure("forced construction failure")

        monkeypatch.setattr(framework_graph, "IntegratedFramework", explode)

        await service.initialize()

        # The pre-fix behaviour was framework is None + every route 503ing.
        assert service.framework is not None
        assert service._framework_mode == "lightweight"

    @pytest.mark.asyncio
    async def test_isinstance_union_does_not_raise_in_the_handler(
        self, service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Guards the PEP 604 union in the handler's expected-vs-unexpected test.

        If `isinstance(e, A | B)` were invalid at runtime it would raise inside
        the except block, so reaching a completed initialize() at all proves it
        evaluates. Asserted explicitly because a TypeError there would silently
        reintroduce the original outage.
        """
        monkeypatch.setattr(
            framework_graph,
            "IntegratedFramework",
            lambda *a, **k: (_ for _ in ()).throw(_ForcedFailure("boom")),
        )

        await service.initialize()  # must not raise TypeError

        assert service.is_ready

    @pytest.mark.asyncio
    async def test_unexpected_failure_is_logged_with_a_traceback(
        self, service, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A defect must be loud. Silent degradation is what hid the original bug."""

        def explode(*args, **kwargs):
            raise _ForcedFailure("forced construction failure")

        monkeypatch.setattr(framework_graph, "IntegratedFramework", explode)

        with caplog.at_level(logging.ERROR):
            await service.initialize()

        assert any(
            record.levelno >= logging.ERROR and record.exc_info for record in caplog.records
        ), "an unexpected initialization failure must be logged at ERROR with a traceback"


class TestExpectedExceptionDuringInit:
    """A missing optional dependency is not a defect and must stay at warning."""

    @pytest.mark.asyncio
    async def test_import_error_still_falls_back(self, service, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing_dep(*args, **kwargs):
            raise ImportError("optional dependency absent")

        monkeypatch.setattr(framework_graph, "IntegratedFramework", missing_dep)

        await service.initialize()

        assert service._framework_mode == "lightweight"
