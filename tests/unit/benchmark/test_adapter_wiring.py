"""
Tests for adapter wiring to real framework components.

Validates the LangGraph adapter's framework integration path,
including IntegratedFramework.process() calls and response normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.adapters.langgraph_adapter import LangGraphBenchmarkAdapter
from src.benchmark.config.benchmark_settings import BenchmarkSettings, reset_benchmark_settings
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory


def _make_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="T1",
        category=TaskCategory.QE,
        description="Test task",
        input_data="Test input data",
    )


@dataclass
class MockLLMResponse:
    text: str
    usage: dict


@pytest.mark.unit
class TestLangGraphAdapterFrameworkWiring:
    """Test adapter wired to IntegratedFramework."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    @pytest.mark.asyncio
    async def test_framework_takes_precedence(self) -> None:
        """Framework path should be used when framework is provided."""
        mock_framework = AsyncMock()
        mock_framework.process.return_value = {
            "response": "Framework response with agents",
            "metadata": {"tool_calls": 3, "input_tokens": 200, "output_tokens": 150},
            "state": {
                "agent_outputs": [
                    {"agent": "hrm", "output": "analysis"},
                    {"agent": "trm", "output": "refinement"},
                ],
            },
        }

        mock_client = AsyncMock()

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            llm_client=mock_client,
            framework=mock_framework,
        )

        result = await adapter.execute(_make_task())

        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "T1"
        assert result.raw_response == "Framework response with agents"
        assert result.num_agent_calls == 2
        assert result.num_tool_calls == 3
        assert result.input_tokens == 200
        assert result.output_tokens == 150
        assert len(result.agent_trace) == 2

        # Verify framework was called, not LLM client
        mock_framework.process.assert_called_once()
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_framework_process_params(self) -> None:
        """Verify correct parameters are passed to framework.process()."""
        mock_framework = AsyncMock()
        mock_framework.process.return_value = {
            "response": "ok",
            "metadata": {},
            "state": {"agent_outputs": []},
        }

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            framework=mock_framework,
        )

        await adapter.execute(_make_task())

        mock_framework.process.assert_called_once_with(
            query="Test input data",
            use_rag=False,
            use_mcts=True,
        )

    @pytest.mark.asyncio
    async def test_framework_error_captured(self) -> None:
        """Framework errors should be captured in result."""
        mock_framework = AsyncMock()
        mock_framework.process.side_effect = RuntimeError("Graph compilation failed")

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            framework=mock_framework,
        )

        result = await adapter.execute(_make_task())

        assert result.has_error
        assert "RuntimeError" in result.raw_response
        assert "Graph compilation failed" in result.raw_response
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_framework_empty_metadata(self) -> None:
        """Handle framework responses with minimal metadata."""
        mock_framework = AsyncMock()
        mock_framework.process.return_value = {
            "response": "minimal response",
            "metadata": {},
            "state": {},
        }

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            framework=mock_framework,
        )

        result = await adapter.execute(_make_task())

        assert result.raw_response == "minimal response"
        assert result.num_agent_calls == 0
        assert result.num_tool_calls == 0

    @pytest.mark.asyncio
    async def test_fallback_to_llm_when_no_framework(self) -> None:
        """Without framework, should fall back to LLM client."""
        mock_client = AsyncMock()
        mock_client.generate.return_value = MockLLMResponse(
            text="Direct LLM response",
            usage={"prompt_tokens": 50, "completion_tokens": 30},
        )

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            llm_client=mock_client,
        )

        result = await adapter.execute(_make_task())

        assert "Direct LLM response" in result.raw_response
        mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_framework_no_client(self) -> None:
        """Without framework or client, should return not-configured message."""
        adapter = LangGraphBenchmarkAdapter(settings=self.settings)
        result = await adapter.execute(_make_task())

        assert "not configured" in result.raw_response.lower()

    def test_framework_param_stored(self) -> None:
        """Verify framework parameter is stored."""
        mock_framework = MagicMock()
        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            framework=mock_framework,
        )
        assert adapter._framework is mock_framework

    @pytest.mark.asyncio
    async def test_framework_mcts_enabled_by_default(self) -> None:
        """Default MCTS iterations > 0, so use_mcts should be True."""
        mock_framework = AsyncMock()
        mock_framework.process.return_value = {
            "response": "with mcts",
            "metadata": {},
            "state": {"agent_outputs": []},
        }

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            framework=mock_framework,
        )

        await adapter.execute(_make_task())

        # Default mcts_iterations >= 1, so use_mcts=True
        mock_framework.process.assert_called_once_with(
            query="Test input data",
            use_rag=False,
            use_mcts=True,
        )
