"""Regression test: the LangGraph example module must import without optional deps.

``examples/langgraph_multi_agent_mcts.py`` previously hard-imported
``langchain_openai`` at module top, so importing its public
``LangGraphMultiAgentFramework`` class failed in a ``[dev]``-only environment.
The optional imports are now guarded; this test forces the dependency to be
absent and asserts the module still imports.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def test_example_imports_without_langchain_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force ``from langchain_openai import ...`` to raise ImportError.
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    monkeypatch.delitem(sys.modules, "examples.langgraph_multi_agent_mcts", raising=False)

    module = importlib.import_module("examples.langgraph_multi_agent_mcts")

    assert hasattr(module, "LangGraphMultiAgentFramework")
    assert isinstance(module.LangGraphMultiAgentFramework, type)
    # The guard collapses the missing optional symbol to ``None``.
    assert module.OpenAIEmbeddings is None
