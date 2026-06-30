"""
Graph introspection / visualization service.

Wraps the visualization surface of an integrated-framework-like object
(:class:`src.framework.graph.integrated.IntegratedFramework`):
``get_graph_structure()``, ``get_graph_mermaid(theme=...)`` and
``draw_mermaid(...)``.

Design constraints:
- No FastAPI import. All logic is coverage-bearing here; the REST adapter stays thin.
- Dependency-injection friendly: the framework is injected.
- Configuration-driven: gated on ``settings.ENABLE_GRAPH_VISUALIZATION``.
- The render path (Kroki HTTP via the framework's ``draw_mermaid``) is delegated
  to the framework so it can be mocked in tests without real network access.
"""

from __future__ import annotations

import base64
import zlib
from typing import Any, Protocol

from src.config.constants import DEFAULT_KROKI_BASE_URL
from src.config.settings import Settings, get_settings
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Supported render output formats. Centralized so the validation list is not a
# magic literal scattered through the module.
SUPPORTED_RENDER_FORMATS = ("png", "svg")
DEFAULT_MERMAID_THEME = "default"


class GraphFramework(Protocol):
    """Structural protocol for the framework object GraphService consumes."""

    def get_graph_structure(self) -> dict[str, Any]: ...

    def get_graph_mermaid(self, include_descriptions: bool = True, theme: str = "default") -> str: ...

    def draw_mermaid(
        self,
        output_file: str | None = None,
        format: str = "png",
        include_descriptions: bool = True,
    ) -> str: ...


class GraphVisualizationDisabledError(RuntimeError):
    """Raised when graph visualization is requested while disabled by config."""


class GraphService:
    """Expose graph structure and Mermaid rendering of an injected framework."""

    def __init__(
        self,
        framework: GraphFramework,
        settings: Settings | None = None,
    ) -> None:
        self._framework = framework
        self._settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        """Whether graph visualization is enabled by configuration."""
        return bool(self._settings.ENABLE_GRAPH_VISUALIZATION)

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            logger.warning("Graph visualization requested but ENABLE_GRAPH_VISUALIZATION is False")
            raise GraphVisualizationDisabledError("Graph visualization is disabled (ENABLE_GRAPH_VISUALIZATION=False)")

    def get_structure(self) -> dict[str, Any]:
        """Return the graph structure (nodes, edges, conditional routing)."""
        self._ensure_enabled()
        structure = self._framework.get_graph_structure()
        logger.info(
            "Graph structure produced: nodes=%d, edges=%d",
            len(structure.get("nodes", [])),
            len(structure.get("edges", [])),
        )
        return structure

    def get_mermaid(
        self,
        include_descriptions: bool = True,
        theme: str = DEFAULT_MERMAID_THEME,
    ) -> str:
        """Return Mermaid flowchart source for the graph."""
        self._ensure_enabled()
        mermaid = self._framework.get_graph_mermaid(include_descriptions=include_descriptions, theme=theme)
        logger.info("Graph Mermaid produced: theme=%s, length=%d", theme, len(mermaid))
        return mermaid

    @staticmethod
    def kroki_url(mermaid_code: str, fmt: str = "png", base_url: str = DEFAULT_KROKI_BASE_URL) -> str:
        """
        Build the Kroki URL that renders the given Mermaid source.

        Mirrors the encoding used by ``IntegratedFramework.draw_mermaid`` so the
        endpoint can hand a client a direct render URL without itself fetching.
        """
        compressed = zlib.compress(mermaid_code.encode("utf-8"), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        return f"{base_url}/mermaid/{fmt}/{encoded}"

    def render(
        self,
        output_file: str | None = None,
        fmt: str = "png",
        include_descriptions: bool = True,
    ) -> dict[str, Any]:
        """
        Render the graph diagram, optionally writing a PNG/SVG via Kroki.

        Delegates the actual HTTP render to the framework's ``draw_mermaid`` so
        the network call is fully mockable in tests. Returns a structured result
        containing the Mermaid source, a direct Kroki URL, and the output path.

        Raises:
            ValueError: For an unsupported ``fmt``.
            RuntimeError: Propagated from the framework if rendering fails.
        """
        self._ensure_enabled()

        if fmt not in SUPPORTED_RENDER_FORMATS:
            raise ValueError(f"Unsupported render format '{fmt}'. Supported: {', '.join(SUPPORTED_RENDER_FORMATS)}.")

        logger.info("Rendering graph: format=%s, output_file=%s", fmt, output_file)
        mermaid_code = self._framework.draw_mermaid(
            output_file=output_file,
            format=fmt,
            include_descriptions=include_descriptions,
        )

        return {
            "mermaid": mermaid_code,
            "format": fmt,
            "kroki_url": self.kroki_url(mermaid_code, fmt=fmt),
            "output_file": output_file,
            "rendered": output_file is not None,
        }


__all__ = [
    "GraphService",
    "GraphFramework",
    "GraphVisualizationDisabledError",
    "SUPPORTED_RENDER_FORMATS",
]
