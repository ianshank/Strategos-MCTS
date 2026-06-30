"""Unit tests for GraphService (structure/mermaid passthrough + mocked Kroki render)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.api.graph_service import (
    SUPPORTED_RENDER_FORMATS,
    GraphService,
    GraphVisualizationDisabledError,
)
from src.config.constants import DEFAULT_KROKI_BASE_URL


def _settings(enable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(ENABLE_GRAPH_VISUALIZATION=enable)


@pytest.fixture
def fake_framework():
    fw = MagicMock()
    fw.get_graph_structure.return_value = {
        "nodes": [{"id": "entry"}, {"id": "synthesize"}],
        "edges": [{"source": "entry", "target": "synthesize"}],
        "conditional_edges": {},
    }
    fw.get_graph_mermaid.return_value = "flowchart TD\n  entry --> synthesize"
    fw.draw_mermaid.return_value = "flowchart TD\n  entry --> synthesize"
    return fw


def test_enabled_reflects_settings(fake_framework):
    assert GraphService(fake_framework, settings=_settings(True)).enabled is True
    assert GraphService(fake_framework, settings=_settings(False)).enabled is False


def test_get_structure_passthrough(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(True))
    result = svc.get_structure()
    assert result == fake_framework.get_graph_structure.return_value
    fake_framework.get_graph_structure.assert_called_once()


def test_get_structure_disabled_raises(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(False))
    with pytest.raises(GraphVisualizationDisabledError):
        svc.get_structure()


def test_get_mermaid_passthrough_with_theme(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(True))
    mermaid = svc.get_mermaid(theme="dark")
    assert mermaid == fake_framework.get_graph_mermaid.return_value
    fake_framework.get_graph_mermaid.assert_called_once_with(include_descriptions=True, theme="dark")


def test_get_mermaid_disabled_raises(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(False))
    with pytest.raises(GraphVisualizationDisabledError):
        svc.get_mermaid()


def test_kroki_url_uses_default_base_and_encodes():
    url = GraphService.kroki_url("flowchart TD\n a-->b", fmt="svg")
    assert url.startswith(f"{DEFAULT_KROKI_BASE_URL}/mermaid/svg/")
    # encoded segment present and URL-safe base64 (no '+' or '/')
    encoded = url.rsplit("/", 1)[-1]
    assert encoded
    assert "+" not in encoded


def test_kroki_url_custom_base():
    url = GraphService.kroki_url("x", fmt="png", base_url="https://kroki.local")
    assert url.startswith("https://kroki.local/mermaid/png/")


def test_render_returns_structured_result_no_output_file(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(True))
    result = svc.render(fmt="png")
    assert result["mermaid"] == fake_framework.draw_mermaid.return_value
    assert result["format"] == "png"
    assert result["kroki_url"].startswith(f"{DEFAULT_KROKI_BASE_URL}/mermaid/png/")
    assert result["output_file"] is None
    assert result["rendered"] is False
    fake_framework.draw_mermaid.assert_called_once_with(output_file=None, format="png", include_descriptions=True)


def test_render_with_output_file_marks_rendered(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(True))
    result = svc.render(output_file="/tmp/out.png", fmt="png")
    assert result["output_file"] == "/tmp/out.png"
    assert result["rendered"] is True


def test_render_invalid_format_raises(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(True))
    with pytest.raises(ValueError):
        svc.render(fmt="gif")


def test_render_propagates_framework_runtime_error(fake_framework):
    fake_framework.draw_mermaid.side_effect = RuntimeError("kroki down")
    svc = GraphService(fake_framework, settings=_settings(True))
    with pytest.raises(RuntimeError, match="kroki down"):
        svc.render(fmt="png")


def test_render_disabled_raises(fake_framework):
    svc = GraphService(fake_framework, settings=_settings(False))
    with pytest.raises(GraphVisualizationDisabledError):
        svc.render()


def test_supported_formats_constant():
    assert set(SUPPORTED_RENDER_FORMATS) == {"png", "svg"}


def test_default_settings_used_when_not_injected(monkeypatch, fake_framework):
    monkeypatch.setattr("src.api.graph_service.get_settings", lambda: _settings(True))
    svc = GraphService(fake_framework)
    assert svc.enabled is True
