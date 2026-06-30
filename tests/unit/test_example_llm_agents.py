"""Unit tests for the reference LLM-orchestration agents in the example framework.

The ``examples/langgraph_multi_agent_mcts.py`` HRM/TRM agents are LLM-backed
reference implementations (not the neural ``src.agents`` modules). They are
exercised indirectly by the chaos/load suites; these deterministic tests pin
the ``process()`` contract and its graceful-degradation behavior directly.

The example module lives in ``examples/`` and imports cleanly without the
optional ``langchain``/``langgraph`` extras (it guards them), so these tests run
in a standard ``[dev]`` environment.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = [pytest.mark.unit]

_EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from langgraph_multi_agent_mcts import (  # noqa: E402  (path injected above)
    DEFAULT_DECOMPOSITION_QUALITY_SCORE,
    DEFAULT_FINAL_QUALITY_SCORE,
    HRMAgent,
    TRMAgent,
)


@pytest.mark.asyncio
async def test_hrm_process_success_returns_text_and_metadata():
    adapter = AsyncMock()
    adapter.generate = AsyncMock(return_value=Mock(text="decomposed answer"))
    agent = HRMAgent(model_adapter=adapter, logger=Mock())

    result = await agent.process(query="solve X", rag_context="ctx")

    assert result["response"] == "decomposed answer"
    assert result["metadata"]["agent"] == "hrm"
    assert result["metadata"]["decomposition_quality_score"] == DEFAULT_DECOMPOSITION_QUALITY_SCORE
    # rag_context is woven into the prompt passed to the adapter.
    _, kwargs = adapter.generate.call_args
    assert "ctx" in kwargs["prompt"]


@pytest.mark.asyncio
async def test_trm_process_degrades_to_empty_on_llm_failure():
    logger = Mock()
    adapter = AsyncMock()
    adapter.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
    agent = TRMAgent(model_adapter=adapter, logger=logger)

    result = await agent.process(query="refine Y")

    # Graceful degradation: empty response, metadata preserved, error logged.
    assert result["response"] == ""
    assert result["metadata"]["agent"] == "trm"
    assert result["metadata"]["final_quality_score"] == DEFAULT_FINAL_QUALITY_SCORE
    assert logger.error.called


@pytest.mark.asyncio
async def test_temperature_override_via_kwargs():
    adapter = AsyncMock()
    adapter.generate = AsyncMock(return_value=Mock(text="ok"))
    agent = HRMAgent(model_adapter=adapter, logger=Mock(), temperature=0.99)

    await agent.process(query="q")

    _, kwargs = adapter.generate.call_args
    assert kwargs["temperature"] == 0.99
