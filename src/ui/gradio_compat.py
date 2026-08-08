"""
Version-tolerant helpers for the Gradio APIs this project depends on.

``pyproject.toml`` declares ``gradio>=4.0.0,<6.0.0``. That range spans a breaking
change: Gradio 4 scheduled periodic work with ``Blocks.load(..., every=N)``,
while Gradio 5 removed the ``every`` keyword in favour of ``gr.Timer``. Code
written against either one crashes on the other — and because the installed
version resolves to 5.x, the chess UI raised

    TypeError: event_trigger() got an unexpected keyword argument 'every'

at import of its Blocks graph, before a socket was ever opened.

Detecting the capability at runtime keeps the declared range honest, so the
package works across the whole span instead of forcing a narrower pin.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["GradioCapabilities", "detect_capabilities", "schedule_refresh"]


class GradioCapabilities:
    """Which periodic-refresh API the installed Gradio provides."""

    def __init__(self, *, has_timer: bool, version: str) -> None:
        self.has_timer = has_timer
        self.version = version

    @property
    def refresh_api(self) -> str:
        """Name of the API :func:`schedule_refresh` will use."""
        return "gr.Timer" if self.has_timer else "Blocks.load(every=...)"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GradioCapabilities(version={self.version!r}, refresh_api={self.refresh_api!r})"


def detect_capabilities(gradio_module: Any) -> GradioCapabilities:
    """
    Inspect an imported ``gradio`` module for the APIs this project uses.

    Args:
        gradio_module: The imported ``gradio`` module (injected so this is
            testable against stand-ins for either major version).

    Returns:
        A :class:`GradioCapabilities` describing what is available.
    """
    return GradioCapabilities(
        has_timer=getattr(gradio_module, "Timer", None) is not None,
        version=str(getattr(gradio_module, "__version__", "unknown")),
    )


def schedule_refresh(
    gradio_module: Any,
    demo: Any,
    fn: Callable[..., Any],
    outputs: Sequence[Any],
    *,
    every_seconds: float,
    inputs: Sequence[Any] | None = None,
) -> Any:
    """
    Schedule ``fn`` to run every ``every_seconds`` and update ``outputs``.

    Uses ``gr.Timer`` when present (Gradio 5+) and falls back to
    ``Blocks.load(..., every=...)`` on Gradio 4. Must be called inside an active
    ``gr.Blocks`` context, since a Timer registers itself on the enclosing graph.

    Args:
        gradio_module: The imported ``gradio`` module.
        demo: The enclosing ``gr.Blocks`` instance.
        fn: Callable producing the refreshed values.
        outputs: Components to update on each tick.
        every_seconds: Refresh interval in seconds.
        inputs: Optional inputs forwarded to ``fn``.

    Returns:
        The ``gr.Timer`` on Gradio 5+, otherwise the value returned by
        ``Blocks.load``.
    """
    capabilities = detect_capabilities(gradio_module)

    if capabilities.has_timer:
        timer = gradio_module.Timer(every_seconds)
        timer.tick(fn=fn, inputs=inputs, outputs=list(outputs))
        return timer

    return demo.load(fn=fn, inputs=inputs, outputs=list(outputs), every=every_seconds)
