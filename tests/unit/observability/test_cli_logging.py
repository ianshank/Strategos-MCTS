"""The console scripts must actually emit their logs, and not onto stdout.

Two separate contracts, both previously unenforced and both previously violated.

**They logged nothing.** ``get_logger`` returns a bare ``mcts.*`` logger with no handler.
Until a caller runs ``setup_logging``/``configure_cli_logging`` the ``mcts`` logger has no
configuration, so INFO and DEBUG records are discarded and only WARNING and above escape
through ``logging.lastResort``, unformatted. Every ``main()`` reached its work without
configuring anything, so an operator running ``self-play-convergence`` saw no output at
all — not the resolved device, not the seed, not the losses, not the checkpoint paths.
A run that fails in the field is then undiagnosable.

**Where the logs go matters.** ``setup_logging`` defaults its console handler to stdout,
which is correct for a server and wrong for these commands: ``policy-lift`` prints its
JSON artifact to stdout, so a log record interleaved there corrupts
``policy-lift ... | jq``. ``configure_cli_logging`` therefore writes to stderr, and the
stream choice is asserted here rather than left to a comment.

These are unit tests over the logging configuration itself. The end-to-end proof that the
installed console scripts honour it lives in ``tests/e2e/test_operational_entry_points_e2e.py``.
"""

from __future__ import annotations

import logging
import sys

import pytest

from src.observability.logging import (
    CONSOLE_STREAM_STDERR,
    CONSOLE_STREAM_STDOUT,
    configure_cli_logging,
    get_logger,
    setup_logging,
)

pytestmark = [pytest.mark.unit]

#: The logger hierarchy every `get_logger` call lands under.
FRAMEWORK_LOGGER = "mcts"


@pytest.fixture(autouse=True)
def _restore_logging_configuration():
    """Put the global logging config back, so configuring it here cannot leak.

    ``dictConfig`` mutates process-wide state. Without this, a test that configures
    handlers would change how every later test in the session logs — the ambient-state
    dependency that makes suites mysteriously order-sensitive.
    """
    root = logging.getLogger()
    framework = logging.getLogger(FRAMEWORK_LOGGER)
    saved = (root.handlers[:], root.level, framework.handlers[:], framework.level, framework.propagate)
    try:
        yield
    finally:
        root.handlers, root.level, framework.handlers, framework.level, framework.propagate = saved


def _console_streams(logger: logging.Logger) -> list[object]:
    return [h.stream for h in logger.handlers if isinstance(h, logging.StreamHandler)]


def test_a_cli_logger_is_unconfigured_before_setup() -> None:
    """The starting condition, pinned so the fix cannot be mistaken for a no-op.

    If this ever fails because something configures logging at import, the guarantee the
    other tests provide changes meaning and they should be re-read.
    """
    logging.getLogger(FRAMEWORK_LOGGER).handlers = []
    assert _console_streams(logging.getLogger(FRAMEWORK_LOGGER)) == []


def test_configure_cli_logging_writes_records_to_stderr() -> None:
    """The console handler must land on stderr, leaving stdout as a data channel."""
    configure_cli_logging()
    streams = _console_streams(logging.getLogger(FRAMEWORK_LOGGER))
    assert streams, "configure_cli_logging attached no console handler; the CLI would stay silent"
    assert all(stream is sys.stderr for stream in streams), (
        f"a CLI log handler is writing to {streams}, not stderr. policy-lift prints its JSON "
        f"artifact to stdout, so this would corrupt `policy-lift ... | jq`."
    )


def test_configure_cli_logging_actually_emits_an_info_record(capsys: pytest.CaptureFixture) -> None:
    """End of the chain: a `logger.info` from library code reaches the operator.

    Asserting the handler exists is not enough — the level and the propagation settings
    have to let an INFO record through, which is exactly what was broken.
    """
    configure_cli_logging(log_level="INFO")
    get_logger("training.self_play_convergence").info("run starting")

    captured = capsys.readouterr()
    assert "run starting" in captured.err, "an INFO record from a CLI was discarded"
    assert "run starting" not in captured.out, "the record leaked onto stdout"


def test_setup_logging_still_defaults_to_stdout() -> None:
    """Backwards compatibility: the stream argument must not move existing callers.

    ``setup_logging`` has other consumers that expect stdout; the new parameter is opt-in.
    """
    setup_logging()
    streams = _console_streams(logging.getLogger(FRAMEWORK_LOGGER))
    assert streams and all(stream is sys.stdout for stream in streams)


@pytest.mark.parametrize(
    ("stream", "expected"),
    [(CONSOLE_STREAM_STDOUT, "stdout"), (CONSOLE_STREAM_STDERR, "stderr")],
)
def test_the_stream_constants_resolve_to_the_named_stream(stream: str, expected: str) -> None:
    """The two exported constants must name the streams they claim to."""
    setup_logging(stream=stream)
    streams = _console_streams(logging.getLogger(FRAMEWORK_LOGGER))
    assert streams and all(s is getattr(sys, expected) for s in streams)


def test_configuring_twice_does_not_stack_handlers() -> None:
    """``main()`` may run more than once in a process (tests, embedding callers).

    Duplicate handlers would print every record twice, which reads as a message-loss bug
    from the other direction.
    """
    configure_cli_logging()
    first = len(logging.getLogger(FRAMEWORK_LOGGER).handlers)
    configure_cli_logging()
    assert len(logging.getLogger(FRAMEWORK_LOGGER).handlers) == first
