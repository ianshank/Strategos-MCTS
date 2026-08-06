"""
Contract tests for :class:`src.observability.logging.StructuredLogger`.

These guard a defect class rather than a single call site. ``StructuredLogger``
is *injected* into components that were written against a stdlib ``Logger`` —
``GraphBuilder`` takes a ``logger`` parameter and calls it printf-style, and
``FrameworkService`` passes a ``StructuredLogger`` into it. When the structured
methods only accepted ``(message, **extra)``, that injection raised ``TypeError``
and took out every reasoning, streaming and graph endpoint in the REST API while
115 mocked tests stayed green.

The contract asserted here is: a ``StructuredLogger`` is a drop-in for a stdlib
``Logger``, and adding structured fields never crashes the call site.
"""

from __future__ import annotations

import logging

import pytest

from src.observability.logging import (
    StructuredLogger,
    ensure_structured_logger,
    get_structured_logger,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def emitted(request: pytest.FixtureRequest) -> list[logging.LogRecord]:
    """Capture records from a uniquely-named logger, isolated per test."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    name = f"contract.{request.node.name}"
    logger = logging.getLogger(name)
    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    request.addfinalizer(lambda: logger.removeHandler(handler))
    return records


@pytest.fixture
def structured(request: pytest.FixtureRequest, emitted: list[logging.LogRecord]) -> StructuredLogger:
    return get_structured_logger(f"contract.{request.node.name}")


class TestPrintfStyleCalls:
    """The stdlib calling convention must work on an injected StructuredLogger."""

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "critical"])
    def test_positional_args_do_not_raise(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord], level: str
    ) -> None:
        getattr(structured, level)("value=%s count=%d", "alpha", 7)

        assert len(emitted) == 1
        assert emitted[0].getMessage() == "value=alpha count=7"

    def test_reproduces_the_graphbuilder_call_shape(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        """The exact four-positional-arg shape at src/framework/graph/builder.py:213."""
        structured.debug(
            "GraphBuilder initialized: max_iterations=%d, consensus_threshold=%.2f, "
            "parallel_agents=%s, mcts_seed=%d",
            10,
            0.75,
            True,
            42,
        )

        assert emitted[0].getMessage() == (
            "GraphBuilder initialized: max_iterations=10, consensus_threshold=0.75, "
            "parallel_agents=True, mcts_seed=42"
        )

    def test_interpolation_stays_lazy(self, structured: StructuredLogger, emitted: list[logging.LogRecord]) -> None:
        """Args reach the record unformatted so a filtered-out record costs nothing."""
        structured.info("count=%d", 3)

        assert emitted[0].args == (3,)
        assert "%d" in emitted[0].msg


class TestStructuredCalls:
    """The pre-existing keyword convention must keep working unchanged."""

    def test_keyword_fields_land_on_the_record(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        structured.info("Graph built", nodes=12, edges=7)

        record = emitted[0]
        assert record.getMessage() == "Graph built"
        assert record.nodes == 12
        assert record.edges == 7

    def test_correlation_id_is_always_attached(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        structured.info("no explicit fields")

        assert getattr(emitted[0], "correlation_id", None)

    def test_positional_and_keyword_combine(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        structured.warning("retry %d of %d", 2, 5, component="mcts")

        record = emitted[0]
        assert record.getMessage() == "retry 2 of 5"
        assert record.component == "mcts"

    def test_secrets_are_redacted_in_fields(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        structured.info("auth", api_key="sk-should-not-appear")

        assert emitted[0].api_key == "***REDACTED***"


class TestStdlibKeywordPassthrough:
    """exc_info/stack_info/stacklevel are stdlib parameters, not record fields."""

    def test_exc_info_produces_a_traceback(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            structured.error("failed", exc_info=True)

        assert emitted[0].exc_info is not None
        assert emitted[0].exc_info[0] is ValueError

    def test_exception_helper_captures_the_active_exception(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        try:
            raise KeyError("missing")
        except KeyError:
            structured.exception("while handling")

        record = emitted[0]
        assert record.exc_info is not None
        assert record.exc_info[0] is KeyError
        assert "KeyError" in record.traceback


class TestReservedAttributeShielding:
    """Structured fields must never collide with reserved LogRecord slots."""

    @pytest.mark.parametrize("reserved", ["module", "name", "filename", "message", "args", "lineno"])
    def test_reserved_field_names_do_not_raise(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord], reserved: str
    ) -> None:
        structured.info("collision", **{reserved: "user-value"})

        assert len(emitted) == 1
        assert getattr(emitted[0], f"{reserved}_") == "user-value"

    def test_non_reserved_fields_are_left_alone(
        self, structured: StructuredLogger, emitted: list[logging.LogRecord]
    ) -> None:
        structured.info("fine", nodes=1, module="chess")

        record = emitted[0]
        assert record.nodes == 1
        assert record.module_ == "chess"


class TestEnsureStructuredLogger:
    """Normalization for functions that accept an injected logger."""

    def test_none_yields_a_logger_on_the_default_name(self) -> None:
        assert isinstance(ensure_structured_logger(None, "fallback.name"), StructuredLogger)

    def test_structured_logger_passes_through_unchanged(self) -> None:
        original = get_structured_logger("already.structured")

        assert ensure_structured_logger(original, "unused") is original

    def test_stdlib_logger_wrapper_preserves_target(self, request: pytest.FixtureRequest) -> None:
        """A stdlib Logger raises TypeError on these kwargs; the wrapper must not."""
        name = f"contract.{request.node.name}"
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target = logging.getLogger(name)
        target.addHandler(_Capture())
        target.setLevel(logging.DEBUG)
        target.propagate = False

        wrapped = ensure_structured_logger(target, "unused")
        wrapped.warning("degraded", checkpoint_status="lfs_pointer")

        assert records[0].name == name
        assert records[0].checkpoint_status == "lfs_pointer"
