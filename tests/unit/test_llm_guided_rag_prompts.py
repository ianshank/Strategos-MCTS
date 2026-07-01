"""Unit tests for RAG-enhanced prompt builders (llm_guided.rag.prompts).

Covers the context-section rendering (solutions/patterns/API docs), brief mode,
test-results rendering, max-length truncation, and the convenience wrappers.
"""

from __future__ import annotations

import pytest

from src.framework.mcts.llm_guided.rag.context import RAGContext
from src.framework.mcts.llm_guided.rag.prompts import (
    RAGPromptBuilder,
    build_generator_prompt_with_rag,
    build_reflector_prompt_with_rag,
)

pytestmark = pytest.mark.unit


def _rich_context() -> RAGContext:
    return RAGContext(
        similar_solutions=[
            {"code": "def a():\n    return 1", "description": "returns one"},
            {"code": "def b():\n    return 2", "description": ""},
        ],
        code_patterns=[{"name": "Memoize", "code": "@cache\ndef f(): ..."}],
        api_docs=[{"name": "itertools", "content": "x" * 600}],
    )


def test_generator_prompt_includes_all_sections() -> None:
    """A populated context, code, and feedback render every section."""
    builder = RAGPromptBuilder()
    prompt = builder.build_generator_prompt(
        problem="Sum a list",
        current_code="def s(xs): pass",
        rag_context=_rich_context(),
        num_variants=2,
        iteration=1,
        feedback="handle empty list",
    )
    assert "## Relevant Context" in prompt
    assert "### Similar Solutions" in prompt
    assert "### Useful Patterns" in prompt
    assert "### API Reference" in prompt
    assert "## Current Code" in prompt
    assert "## Feedback from Previous Attempts" in prompt
    # iteration > 0 branch in the instructions
    assert "iteration 1" in prompt
    # API doc content is truncated at 500 chars + ellipsis
    assert "..." in prompt


def test_generator_prompt_without_context_or_code() -> None:
    """Empty context and no code omit the optional sections."""
    builder = RAGPromptBuilder()
    prompt = builder.build_generator_prompt("P", None, RAGContext(), num_variants=1, iteration=0)
    assert "## Relevant Context" not in prompt
    assert "## Current Code" not in prompt
    assert "## Problem" in prompt


def test_reflector_prompt_brief_context_and_test_results() -> None:
    """Reflector uses brief context and renders failing test results."""
    builder = RAGPromptBuilder()
    long_solution = {"code": "\n".join(f"line{i}" for i in range(20)), "description": "big"}
    ctx = RAGContext(similar_solutions=[long_solution])
    prompt = builder.build_reflector_prompt(
        problem="P",
        code="def x(): ...",
        test_results={"passed": False, "num_passed": 1, "num_total": 3, "errors": ["boom"], "stdout": "trace"},
        rag_context=ctx,
    )
    assert "## Relevant Context" in prompt
    assert "# ..." in prompt  # brief-mode truncation marker
    assert "1/3 tests passed" in prompt
    assert "boom" in prompt
    assert "trace" in prompt


def test_reflector_prompt_all_passed() -> None:
    """The all-passed branch renders the success status."""
    builder = RAGPromptBuilder()
    prompt = builder.build_reflector_prompt("P", "code", {"passed": True}, None)
    assert "All tests passed!" in prompt


def test_context_section_respects_max_length() -> None:
    """A tiny max_context_length truncates the rendered context section."""
    builder = RAGPromptBuilder(max_context_length=50)
    ctx = RAGContext(similar_solutions=[{"code": "x" * 500, "description": "d"}])
    section = builder._rag_context_section(ctx)
    assert "[Context truncated...]" in section


def test_convenience_wrappers() -> None:
    """The module-level convenience functions delegate to the builder."""
    gen = build_generator_prompt_with_rag("P", current_code="c", rag_context=_rich_context())
    ref = build_reflector_prompt_with_rag("P", "code", test_results={"passed": True})
    assert "## Problem" in gen
    assert "All tests passed!" in ref
