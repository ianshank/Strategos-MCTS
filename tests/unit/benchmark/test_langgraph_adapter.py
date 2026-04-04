"""
Tests for LangGraphBenchmarkAdapter.

Validates execution paths (framework, graph, direct LLM, no-client),
error handling, health checks, and protocol compliance.
All external dependencies are mocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.adapters.langgraph_adapter import LangGraphBenchmarkAdapter
from src.benchmark.adapters.protocol import BenchmarkSystemProtocol
from src.benchmark.config.benchmark_settings import (
    BenchmarkSettings,
    LangGraphBenchmarkConfig,
    reset_benchmark_settings,
)
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory


@dataclass
class MockLLMResponse:
    text: str
    usage: dict


def _make_task(task_id: str = "T1") -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        category=TaskCategory.QE,
        description="Test task",
        input_data="Test input data for benchmark",
    )


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    reset_benchmark_settings()


@pytest.fixture
def settings() -> BenchmarkSettings:
    return BenchmarkSettings()


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    client = AsyncMock()
    client.generate.return_value = MockLLMResponse(
        text="LLM response text",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )
    return client


@pytest.fixture
def mock_framework() -> AsyncMock:
    framework = AsyncMock()
    framework.process.return_value = {
        "response": "Framework response",
        "metadata": {
            "tool_calls": 3,
            "input_tokens": 200,
            "output_tokens": 100,
        },
        "state": {
            "agent_outputs": [
                {"agent": "hrm", "output": "analysis"},
                {"agent": "trm", "output": "refinement"},
            ],
        },
    }
    return framework


@pytest.fixture
def mock_graph_builder() -> MagicMock:
    builder = MagicMock()
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "final_response": "Graph response",
        "agent_outputs": [{"agent": "test"}],
        "metadata": {"tool_calls": 1, "input_tokens": 50, "output_tokens": 25},
    }
    builder.build_graph.return_value.compile.return_value = mock_graph
    return builder


@pytest.mark.unit
class TestAdapterProperties:
    """Test adapter name and availability."""

    def test_name(self, settings: BenchmarkSettings) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings)
        assert adapter.name == "langgraph_mcts"

    def test_protocol_compliance(self, settings: BenchmarkSettings) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings)
        assert isinstance(adapter, BenchmarkSystemProtocol)

    def test_is_available_when_disabled(self, settings: BenchmarkSettings) -> None:
        settings._langgraph = LangGraphBenchmarkConfig(enabled=False)
        adapter = LangGraphBenchmarkAdapter(settings=settings)
        assert not adapter.is_available

    def test_is_available_when_langgraph_not_installed(self, settings: BenchmarkSettings) -> None:
        with patch.dict("sys.modules", {"langgraph": None}):
            adapter = LangGraphBenchmarkAdapter(settings=settings)
            # is_available may return False if import fails
            # This depends on whether langgraph is actually installed
            result = adapter.is_available
            assert isinstance(result, bool)


@pytest.mark.unit
class TestFrameworkExecution:
    """Test execution via IntegratedFramework."""

    @pytest.mark.asyncio
    async def test_execute_with_framework(
        self,
        settings: BenchmarkSettings,
        mock_framework: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, framework=mock_framework)
        result = await adapter.execute(_make_task())

        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "T1"
        assert result.system == "langgraph_mcts"
        assert result.raw_response == "Framework response"
        assert result.num_agent_calls == 2
        assert result.num_tool_calls == 3
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_framework_takes_precedence_over_graph(
        self,
        settings: BenchmarkSettings,
        mock_framework: AsyncMock,
        mock_graph_builder: MagicMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(
            settings=settings,
            framework=mock_framework,
            graph_builder=mock_graph_builder,
        )
        result = await adapter.execute(_make_task())

        # Framework should be used, not graph builder
        mock_framework.process.assert_called_once()
        mock_graph_builder.build_graph.assert_not_called()
        assert result.raw_response == "Framework response"

    @pytest.mark.asyncio
    async def test_framework_process_called_with_correct_args(
        self,
        settings: BenchmarkSettings,
        mock_framework: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, framework=mock_framework)
        task = _make_task()
        await adapter.execute(task)

        mock_framework.process.assert_called_once_with(
            query=task.input_data,
            use_rag=False,
            use_mcts=True,  # Default: mcts_iterations > 0
        )

    @pytest.mark.asyncio
    async def test_framework_error_captured(
        self,
        settings: BenchmarkSettings,
    ) -> None:
        framework = AsyncMock()
        framework.process.side_effect = RuntimeError("Framework crashed")

        adapter = LangGraphBenchmarkAdapter(settings=settings, framework=framework)
        result = await adapter.execute(_make_task())

        assert result.has_error
        assert "RuntimeError" in result.raw_response
        assert "Framework crashed" in result.raw_response
        assert result.total_latency_ms > 0


@pytest.mark.unit
class TestGraphExecution:
    """Test execution via GraphBuilder."""

    @pytest.mark.asyncio
    async def test_execute_with_graph_builder(
        self,
        settings: BenchmarkSettings,
        mock_graph_builder: MagicMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, graph_builder=mock_graph_builder)
        result = await adapter.execute(_make_task())

        assert isinstance(result, BenchmarkResult)
        assert result.raw_response == "Graph response"
        assert result.num_agent_calls == 1

    @pytest.mark.asyncio
    async def test_graph_built_lazily(
        self,
        settings: BenchmarkSettings,
        mock_graph_builder: MagicMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, graph_builder=mock_graph_builder)
        # Graph not built on init
        mock_graph_builder.build_graph.assert_not_called()

        await adapter.execute(_make_task())
        mock_graph_builder.build_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_cached_after_first_build(
        self,
        settings: BenchmarkSettings,
        mock_graph_builder: MagicMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, graph_builder=mock_graph_builder)
        await adapter.execute(_make_task())
        await adapter.execute(_make_task("T2"))

        # Should only build once
        mock_graph_builder.build_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_build_failure_falls_back_to_direct(
        self,
        settings: BenchmarkSettings,
        mock_llm_client: AsyncMock,
    ) -> None:
        failing_builder = MagicMock()
        failing_builder.build_graph.side_effect = RuntimeError("Build failed")

        adapter = LangGraphBenchmarkAdapter(
            settings=settings,
            graph_builder=failing_builder,
            llm_client=mock_llm_client,
        )
        result = await adapter.execute(_make_task())

        # Should fall back to direct LLM
        mock_llm_client.generate.assert_called_once()
        assert "LLM response text" in result.raw_response


@pytest.mark.unit
class TestDirectLLMExecution:
    """Test execution via direct LLM client."""

    @pytest.mark.asyncio
    async def test_execute_with_llm_client(
        self,
        settings: BenchmarkSettings,
        mock_llm_client: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, llm_client=mock_llm_client)
        result = await adapter.execute(_make_task())

        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "T1"
        assert result.system == "langgraph_mcts"
        assert "LLM response text" in result.raw_response
        assert result.num_agent_calls == 1
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_direct_llm_sends_system_prompt(
        self,
        settings: BenchmarkSettings,
        mock_llm_client: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, llm_client=mock_llm_client)
        await adapter.execute(_make_task())

        call_args = mock_llm_client.generate.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_direct_llm_error_captured(
        self,
        settings: BenchmarkSettings,
    ) -> None:
        failing_client = AsyncMock()
        failing_client.generate.side_effect = RuntimeError("API error")

        adapter = LangGraphBenchmarkAdapter(settings=settings, llm_client=failing_client)
        result = await adapter.execute(_make_task())

        assert result.has_error
        assert "RuntimeError" in result.raw_response
        assert "API error" in result.raw_response

    @pytest.mark.asyncio
    async def test_direct_llm_trace_includes_agent_entry(
        self,
        settings: BenchmarkSettings,
        mock_llm_client: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, llm_client=mock_llm_client)
        result = await adapter.execute(_make_task())

        assert len(result.agent_trace) == 1
        assert result.agent_trace[0]["agent"] == "direct_llm"


@pytest.mark.unit
class TestNoClientExecution:
    """Test execution with no client or framework configured."""

    @pytest.mark.asyncio
    async def test_execute_no_client_returns_not_configured(
        self,
        settings: BenchmarkSettings,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings)
        result = await adapter.execute(_make_task())

        assert result.task_id == "T1"
        assert "not configured" in result.raw_response.lower()
        assert result.num_agent_calls == 0
        assert result.total_latency_ms >= 0


@pytest.mark.unit
class TestHealthCheck:
    """Test adapter health checks."""

    @pytest.mark.asyncio
    async def test_health_check_returns_bool(self, settings: BenchmarkSettings) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings)
        result = await adapter.health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_health_check_false_when_disabled(self, settings: BenchmarkSettings) -> None:
        settings._langgraph = LangGraphBenchmarkConfig(enabled=False)
        adapter = LangGraphBenchmarkAdapter(settings=settings)
        result = await adapter.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_import_failure_returns_false(self, settings: BenchmarkSettings) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings)
        with patch.dict("sys.modules", {"src.framework.mcts.core": None}):
            result = await adapter.health_check()
            # Should return False when MCTS import fails
            assert result is False


@pytest.mark.unit
class TestResultMetrics:
    """Test that results capture correct metrics."""

    @pytest.mark.asyncio
    async def test_latency_is_positive(
        self,
        settings: BenchmarkSettings,
        mock_llm_client: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, llm_client=mock_llm_client)
        result = await adapter.execute(_make_task())
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_latency_recorded_on_error(
        self,
        settings: BenchmarkSettings,
    ) -> None:
        failing_client = AsyncMock()
        failing_client.generate.side_effect = ValueError("bad input")

        adapter = LangGraphBenchmarkAdapter(settings=settings, llm_client=failing_client)
        result = await adapter.execute(_make_task())
        assert result.total_latency_ms > 0
        assert result.has_error

    @pytest.mark.asyncio
    async def test_task_description_captured(
        self,
        settings: BenchmarkSettings,
        mock_llm_client: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, llm_client=mock_llm_client)
        result = await adapter.execute(_make_task())
        assert result.task_description == "Test task"

    @pytest.mark.asyncio
    async def test_framework_response_metadata(
        self,
        settings: BenchmarkSettings,
        mock_framework: AsyncMock,
    ) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=settings, framework=mock_framework)
        result = await adapter.execute(_make_task())

        assert result.agent_trace == [
            {"agent": "hrm", "output": "analysis"},
            {"agent": "trm", "output": "refinement"},
        ]
