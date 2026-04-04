"""
Tests for BenchmarkSystemProtocol.

Validates protocol compliance via runtime_checkable isinstance checks,
structural subtyping behavior, and contract verification for
conforming and non-conforming classes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.adapters.protocol import BenchmarkSystemProtocol
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory


def _make_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="T1",
        category=TaskCategory.QE,
        description="Test task",
        input_data="Test input data",
    )


class ConformingAdapter:
    """A class that structurally conforms to BenchmarkSystemProtocol."""

    @property
    def name(self) -> str:
        return "conforming"

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        return BenchmarkResult(task_id=task.task_id, system=self.name, raw_response="done")

    async def health_check(self) -> bool:
        return True


class MinimalConformingAdapter:
    """Minimal conforming adapter with different implementations."""

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def is_available(self) -> bool:
        return False

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        return BenchmarkResult(task_id=task.task_id, system="minimal")

    async def health_check(self) -> bool:
        return False


class MissingNameAdapter:
    """Adapter missing the 'name' property."""

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        return BenchmarkResult()

    async def health_check(self) -> bool:
        return True


class MissingExecuteAdapter:
    """Adapter missing the 'execute' method."""

    @property
    def name(self) -> str:
        return "no_execute"

    @property
    def is_available(self) -> bool:
        return True

    async def health_check(self) -> bool:
        return True


class MissingHealthCheckAdapter:
    """Adapter missing the 'health_check' method."""

    @property
    def name(self) -> str:
        return "no_health"

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        return BenchmarkResult()


class MissingIsAvailableAdapter:
    """Adapter missing the 'is_available' property."""

    @property
    def name(self) -> str:
        return "no_available"

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        return BenchmarkResult()

    async def health_check(self) -> bool:
        return True


class NameAsMethodAdapter:
    """Adapter with 'name' as a method instead of a property."""

    def name(self) -> str:
        return "method_name"

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        return BenchmarkResult()

    async def health_check(self) -> bool:
        return True


class EmptyClass:
    """Completely empty class."""

    pass


@pytest.mark.unit
class TestProtocolConformance:
    """Test that conforming classes pass isinstance checks."""

    def test_conforming_adapter_is_instance(self) -> None:
        adapter = ConformingAdapter()
        assert isinstance(adapter, BenchmarkSystemProtocol)

    def test_minimal_conforming_adapter_is_instance(self) -> None:
        adapter = MinimalConformingAdapter()
        assert isinstance(adapter, BenchmarkSystemProtocol)


@pytest.mark.unit
class TestProtocolNonConformance:
    """Test that non-conforming classes fail isinstance checks."""

    def test_missing_name_not_instance(self) -> None:
        adapter = MissingNameAdapter()
        assert not isinstance(adapter, BenchmarkSystemProtocol)

    def test_missing_execute_not_instance(self) -> None:
        adapter = MissingExecuteAdapter()
        assert not isinstance(adapter, BenchmarkSystemProtocol)

    def test_missing_health_check_not_instance(self) -> None:
        adapter = MissingHealthCheckAdapter()
        assert not isinstance(adapter, BenchmarkSystemProtocol)

    def test_missing_is_available_not_instance(self) -> None:
        adapter = MissingIsAvailableAdapter()
        assert not isinstance(adapter, BenchmarkSystemProtocol)

    def test_empty_class_not_instance(self) -> None:
        assert not isinstance(EmptyClass(), BenchmarkSystemProtocol)

    def test_string_not_instance(self) -> None:
        assert not isinstance("not an adapter", BenchmarkSystemProtocol)

    def test_none_not_instance(self) -> None:
        assert not isinstance(None, BenchmarkSystemProtocol)


@pytest.mark.unit
class TestProtocolExecution:
    """Test that conforming adapters can be used polymorphically."""

    @pytest.mark.asyncio
    async def test_execute_via_protocol_reference(self) -> None:
        adapter: BenchmarkSystemProtocol = ConformingAdapter()
        task = _make_task()
        result = await adapter.execute(task)

        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "T1"
        assert result.system == "conforming"
        assert result.raw_response == "done"

    @pytest.mark.asyncio
    async def test_health_check_via_protocol_reference(self) -> None:
        adapter: BenchmarkSystemProtocol = ConformingAdapter()
        assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_unavailable_adapter_health_check(self) -> None:
        adapter: BenchmarkSystemProtocol = MinimalConformingAdapter()
        assert adapter.is_available is False
        assert await adapter.health_check() is False

    @pytest.mark.asyncio
    async def test_multiple_adapters_polymorphic(self) -> None:
        adapters: list[BenchmarkSystemProtocol] = [
            ConformingAdapter(),
            MinimalConformingAdapter(),
        ]
        task = _make_task()

        names = []
        for adapter in adapters:
            result = await adapter.execute(task)
            names.append(result.system)

        assert names == ["conforming", "minimal"]

    def test_name_property_accessible(self) -> None:
        adapter: BenchmarkSystemProtocol = ConformingAdapter()
        assert adapter.name == "conforming"

    def test_is_available_property_accessible(self) -> None:
        adapter: BenchmarkSystemProtocol = ConformingAdapter()
        assert adapter.is_available is True


@pytest.mark.unit
class TestLangGraphAdapterProtocolCompliance:
    """Verify the real LangGraphBenchmarkAdapter conforms to the protocol."""

    def test_langgraph_adapter_is_instance(self) -> None:
        from src.benchmark.adapters.langgraph_adapter import LangGraphBenchmarkAdapter
        from src.benchmark.config.benchmark_settings import BenchmarkSettings, reset_benchmark_settings

        reset_benchmark_settings()
        adapter = LangGraphBenchmarkAdapter(settings=BenchmarkSettings())
        assert isinstance(adapter, BenchmarkSystemProtocol)

    def test_langgraph_adapter_has_name(self) -> None:
        from src.benchmark.adapters.langgraph_adapter import LangGraphBenchmarkAdapter
        from src.benchmark.config.benchmark_settings import BenchmarkSettings, reset_benchmark_settings

        reset_benchmark_settings()
        adapter = LangGraphBenchmarkAdapter(settings=BenchmarkSettings())
        assert adapter.name == "langgraph_mcts"
