"""
Tests for :mod:`src.ui.gradio_compat`.

``pyproject.toml`` declares ``gradio>=4.0.0,<6.0.0``, a range that spans the
removal of ``Blocks.load(every=...)`` in favour of ``gr.Timer``. Both branches
are exercised here against injected stand-ins, so the compatibility claim is
tested on every run rather than only against whichever major happens to be
installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.ui.gradio_compat import detect_capabilities, schedule_refresh

pytestmark = pytest.mark.unit


class _Timer:
    """Stand-in for ``gr.Timer`` (Gradio 5+)."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.ticks: list[dict[str, Any]] = []

    def tick(self, **kwargs: Any) -> None:
        self.ticks.append(kwargs)


class _Gradio5:
    __version__ = "5.50.0"
    Timer = _Timer


class _Gradio4:
    """Gradio 4 exposes no Timer; periodic work goes through Blocks.load."""

    __version__ = "4.44.0"


class _Blocks:
    def __init__(self) -> None:
        self.load_calls: list[dict[str, Any]] = []

    def load(self, **kwargs: Any) -> str:
        self.load_calls.append(kwargs)
        return "load-result"


def _noop() -> tuple[str, str]:
    return ("status", "board")


class TestCapabilityDetection:
    def test_gradio5_reports_timer(self) -> None:
        caps = detect_capabilities(_Gradio5())

        assert caps.has_timer
        assert caps.refresh_api == "gr.Timer"
        assert caps.version == "5.50.0"

    def test_gradio4_reports_load_every(self) -> None:
        caps = detect_capabilities(_Gradio4())

        assert not caps.has_timer
        assert caps.refresh_api == "Blocks.load(every=...)"

    def test_unknown_version_does_not_raise(self) -> None:
        class _Bare:
            pass

        assert detect_capabilities(_Bare()).version == "unknown"


class TestScheduleRefreshOnGradio5:
    def test_uses_timer_and_never_calls_load(self) -> None:
        """`Blocks.load(every=...)` raises TypeError on Gradio 5 — it must not be used."""
        demo = _Blocks()

        timer = schedule_refresh(_Gradio5(), demo, fn=_noop, outputs=["a", "b"], every_seconds=1.0)

        assert isinstance(timer, _Timer)
        assert demo.load_calls == []

    def test_timer_carries_the_interval_and_outputs(self) -> None:
        timer = schedule_refresh(_Gradio5(), _Blocks(), fn=_noop, outputs=["a", "b"], every_seconds=2.5)

        assert timer.interval == 2.5
        assert timer.ticks[0]["outputs"] == ["a", "b"]
        assert timer.ticks[0]["fn"] is _noop


class TestScheduleRefreshOnGradio4:
    def test_falls_back_to_load_with_every(self) -> None:
        demo = _Blocks()

        result = schedule_refresh(_Gradio4(), demo, fn=_noop, outputs=["a"], every_seconds=1.0)

        assert result == "load-result"
        assert demo.load_calls[0]["every"] == 1.0
        assert demo.load_calls[0]["outputs"] == ["a"]

    def test_inputs_are_forwarded(self) -> None:
        demo = _Blocks()

        schedule_refresh(_Gradio4(), demo, fn=_noop, outputs=["a"], every_seconds=1.0, inputs=["in"])

        assert demo.load_calls[0]["inputs"] == ["in"]


class TestAgainstInstalledGradio:
    """Guards the real dependency, not just the stand-ins."""

    def test_installed_gradio_is_supported(self) -> None:
        gr = pytest.importorskip("gradio")

        caps = detect_capabilities(gr)

        assert caps.has_timer or hasattr(gr.Blocks, "load"), (
            f"gradio {caps.version} provides neither gr.Timer nor Blocks.load; "
            "the declared >=4,<6 range no longer holds"
        )
