"""
Tests for benchmark system adapters and factory.

Validates adapter protocol compliance, factory creation,
and mock-based execution behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.adapters.factory import BenchmarkAdapterFactory
from src.benchmark.adapters.langgraph_adapter import LangGraphBenchmarkAdapter
from src.benchmark.adapters.protocol import BenchmarkSystemProtocol
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
class TestLangGraphBenchmarkAdapter:
    """Test LangGraph MCTS benchmark adapter."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    def test_name(self) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=self.settings)
        assert adapter.name == "langgraph_mcts"

    def test_protocol_compliance(self) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=self.settings)
        assert isinstance(adapter, BenchmarkSystemProtocol)

    @pytest.mark.asyncio
    async def test_execute_with_llm_client(self) -> None:
        mock_client = AsyncMock()
        mock_client.generate.return_value = MockLLMResponse(
            text="Analysis: Found 2 bugs",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            llm_client=mock_client,
        )
        result = await adapter.execute(_make_task())

        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "T1"
        assert result.system == "langgraph_mcts"
        assert result.total_latency_ms > 0
        assert "Found 2 bugs" in result.raw_response

    @pytest.mark.asyncio
    async def test_execute_no_client_no_graph(self) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=self.settings)
        result = await adapter.execute(_make_task())

        assert result.task_id == "T1"
        assert result.total_latency_ms >= 0
        assert "not configured" in result.raw_response.lower()

    @pytest.mark.asyncio
    async def test_execute_error_handling(self) -> None:
        mock_client = AsyncMock()
        mock_client.generate.side_effect = RuntimeError("API error")

        adapter = LangGraphBenchmarkAdapter(
            settings=self.settings,
            llm_client=mock_client,
        )
        result = await adapter.execute(_make_task())

        assert result.has_error
        assert "RuntimeError" in result.raw_response

    @pytest.mark.asyncio
    async def test_health_check_available(self) -> None:
        adapter = LangGraphBenchmarkAdapter(settings=self.settings)
        # Health check should succeed if langgraph and MCTS are importable
        health = await adapter.health_check()
        assert isinstance(health, bool)

    def test_is_available_disabled(self) -> None:
        settings = BenchmarkSettings()
        with patch.dict("os.environ", {"BENCHMARK_LG_ENABLED": "false"}):
            from src.benchmark.config.benchmark_settings import LangGraphBenchmarkConfig

            disabled_config = LangGraphBenchmarkConfig(enabled=False)
            settings._langgraph = disabled_config
            adapter = LangGraphBenchmarkAdapter(settings=settings)
            assert not adapter.is_available


@pytest.mark.unit
class TestBenchmarkAdapterFactory:
    """Test adapter factory creation."""

    def setup_method(self) -> None:
        reset_benchmark_settings()

    def test_create_langgraph(self) -> None:
        factory = BenchmarkAdapterFactory()
        adapter = factory.create("langgraph_mcts")
        assert adapter.name == "langgraph_mcts"
        assert isinstance(adapter, BenchmarkSystemProtocol)

    def test_create_unknown_raises(self) -> None:
        factory = BenchmarkAdapterFactory()
        with pytest.raises(ValueError, match="Unknown system"):
            factory.create("nonexistent_system")

    def test_get_available_systems(self) -> None:
        factory = BenchmarkAdapterFactory()
        systems = factory.get_available_systems()
        assert "langgraph_mcts" in systems
        assert "vertex_adk" in systems

    def test_register_custom_adapter(self) -> None:
        class CustomAdapter:
            def __init__(self, settings: BenchmarkSettings | None = None, **kwargs):  # type: ignore[no-untyped-def]
                pass

            @property
            def name(self) -> str:
                return "custom"

            @property
            def is_available(self) -> bool:
                return True

            async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
                return BenchmarkResult(task_id=task.task_id, system="custom")

            async def health_check(self) -> bool:
                return True

        factory = BenchmarkAdapterFactory()
        factory.register_adapter("custom", CustomAdapter)
        adapter = factory.create("custom")
        assert adapter.name == "custom"

    def test_create_all_available(self) -> None:
        factory = BenchmarkAdapterFactory()
        adapters = factory.create_all_available()
        # Returns list of adapters that are available
        assert isinstance(adapters, list)
        # All returned adapters should implement the protocol
        for adapter in adapters:
            assert isinstance(adapter, BenchmarkSystemProtocol)
            assert adapter.is_available

    def test_create_all_available_empty_when_none_available(self) -> None:
        settings = BenchmarkSettings()
        from src.benchmark.config.benchmark_settings import LangGraphBenchmarkConfig

        settings._langgraph = LangGraphBenchmarkConfig(enabled=False)
        factory = BenchmarkAdapterFactory(settings=settings)
        adapters = factory.create_all_available()
        # LangGraph disabled, ADK not installed — should have no available adapters
        langgraph_adapters = [a for a in adapters if a.name == "langgraph_mcts"]
        assert len(langgraph_adapters) == 0

    def test_create_with_kwargs(self) -> None:
        from unittest.mock import MagicMock

        factory = BenchmarkAdapterFactory()
        mock_client = MagicMock()
        adapter = factory.create("langgraph_mcts", llm_client=mock_client)
        assert adapter.name == "langgraph_mcts"
        assert adapter._llm_client is mock_client
