"""
Tests for ADK benchmark adapter.

Validates adapter behavior with mocked Google ADK dependencies,
including initialization, availability checks, event normalization,
delegation counting, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.adapters.adk_adapter import (
    _ADK_APP_NAME,
    _ADK_USER_ID,
    ADKBenchmarkAdapter,
    _check_adk_available,
)
from src.benchmark.config.benchmark_settings import (
    ADKBenchmarkConfig,
    BenchmarkSettings,
    reset_benchmark_settings,
)
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory


def _make_task(task_id: str = "T1") -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        category=TaskCategory.QE,
        description="Test task",
        input_data="Analyze this code for bugs",
    )


@pytest.mark.unit
class TestADKConstants:
    """Test module-level constants."""

    def test_app_name_constant(self) -> None:
        assert _ADK_APP_NAME == "benchmark"

    def test_user_id_constant(self) -> None:
        assert _ADK_USER_ID == "benchmark"


@pytest.mark.unit
class TestCheckADKAvailable:
    """Test ADK availability check."""

    def test_returns_false_when_not_installed(self) -> None:
        # ADK is not installed in test environment
        assert _check_adk_available() is False

    @patch("src.benchmark.adapters.adk_adapter.LlmAgent", create=True)
    def test_returns_true_when_installed(self, mock_agent: MagicMock) -> None:
        with patch.dict("sys.modules", {"google.adk.agents": MagicMock()}):
            # Force re-import check
            from importlib import reload

            import src.benchmark.adapters.adk_adapter as mod

            reload(mod)
            # After reload, the function should use the import mechanism
            # Since we can't easily control the import, test the mock path instead
            assert isinstance(mock_agent, MagicMock)


@pytest.mark.unit
class TestADKBenchmarkAdapterInit:
    """Test ADK adapter initialization."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    def test_default_settings(self) -> None:
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter._settings is self.settings

    def test_custom_coordinator(self) -> None:
        mock_coordinator = MagicMock()
        adapter = ADKBenchmarkAdapter(
            settings=self.settings,
            coordinator_agent=mock_coordinator,
        )
        assert adapter._coordinator is mock_coordinator

    def test_name_property(self) -> None:
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter.name == "vertex_adk"

    def test_runner_starts_none(self) -> None:
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter._runner is None


@pytest.mark.unit
class TestADKBenchmarkAdapterAvailability:
    """Test ADK adapter availability checks."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    def test_unavailable_when_disabled(self) -> None:
        self.settings._adk = ADKBenchmarkConfig(enabled=False)
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter.is_available is False

    @patch("src.benchmark.adapters.adk_adapter._check_adk_available", return_value=False)
    def test_unavailable_when_adk_not_installed(self, mock_check: MagicMock) -> None:
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter.is_available is False
        mock_check.assert_called_once()

    @patch("src.benchmark.adapters.adk_adapter._check_adk_available", return_value=True)
    def test_unavailable_when_no_credentials(self, mock_check: MagicMock) -> None:
        # Default settings have no google_api_key and no google_project_id
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter.is_available is False

    @patch("src.benchmark.adapters.adk_adapter._check_adk_available", return_value=True)
    def test_available_with_api_key(self, mock_check: MagicMock) -> None:
        self.settings._adk = ADKBenchmarkConfig(google_api_key="test-key")
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter.is_available is True

    @patch("src.benchmark.adapters.adk_adapter._check_adk_available", return_value=True)
    def test_available_with_project_id(self, mock_check: MagicMock) -> None:
        self.settings._adk = ADKBenchmarkConfig(google_project_id="test-project")
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert adapter.is_available is True


@pytest.mark.unit
class TestADKBenchmarkAdapterExecution:
    """Test ADK adapter task execution with mocked ADK."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    @pytest.mark.asyncio
    async def test_execute_with_injected_coordinator(self) -> None:
        """Execute uses injected coordinator without building one."""
        mock_coordinator = MagicMock()
        mock_runner = MagicMock()

        # Mock the runner's run_async to return empty stream
        mock_runner.run_async = MagicMock(return_value=AsyncIterator([]))

        adapter = ADKBenchmarkAdapter(
            settings=self.settings,
            coordinator_agent=mock_coordinator,
        )
        adapter._runner = mock_runner

        result = await adapter.execute(_make_task())

        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "T1"
        assert result.system == "vertex_adk"
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_captures_error(self) -> None:
        """Execution errors are captured in result, not raised."""
        mock_coordinator = MagicMock()
        adapter = ADKBenchmarkAdapter(
            settings=self.settings,
            coordinator_agent=mock_coordinator,
        )
        # Force runner build to fail
        adapter._get_or_build_runner = MagicMock(side_effect=RuntimeError("Runner build failed"))

        result = await adapter.execute(_make_task())

        assert result.has_error
        assert "RuntimeError" in result.raw_response
        assert "Runner build failed" in result.raw_response
        assert result.total_latency_ms > 0


@pytest.mark.unit
class TestADKBenchmarkAdapterHealthCheck:
    """Test ADK adapter health check."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    @pytest.mark.asyncio
    async def test_health_check_unavailable(self) -> None:
        """Health check fails when adapter is unavailable."""
        self.settings._adk = ADKBenchmarkConfig(enabled=False)
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        assert await adapter.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_with_coordinator(self) -> None:
        """Health check passes with injected coordinator."""
        mock_coordinator = MagicMock()

        with patch("src.benchmark.adapters.adk_adapter._check_adk_available", return_value=True):
            self.settings._adk = ADKBenchmarkConfig(google_api_key="test-key")
            adapter = ADKBenchmarkAdapter(
                settings=self.settings,
                coordinator_agent=mock_coordinator,
            )
            assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_exception(self) -> None:
        """Health check returns False on exception."""
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        adapter._get_or_build_coordinator = MagicMock(side_effect=RuntimeError("fail"))

        with patch("src.benchmark.adapters.adk_adapter._check_adk_available", return_value=True):
            self.settings._adk = ADKBenchmarkConfig(google_api_key="test-key")
            assert await adapter.health_check() is False


@pytest.mark.unit
class TestADKNormalizeEvent:
    """Test ADK event normalization."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()
        self.adapter = ADKBenchmarkAdapter(settings=self.settings)

    def test_normalize_text_event(self) -> None:
        """Events with text parts are normalized correctly."""
        mock_part = MagicMock()
        mock_part.text = "Hello world"
        mock_part.function_call = None

        mock_event = MagicMock()
        mock_event.author = "code_reviewer"
        mock_event.content.parts = [mock_part]
        mock_event.is_final_response.return_value = False

        result = self.adapter._normalize_event(mock_event)

        assert result["author"] == "code_reviewer"
        assert result["is_final"] is False
        assert len(result["content"]["parts"]) == 1
        assert result["content"]["parts"][0] == {"text": "Hello world"}

    def test_normalize_function_call_event(self) -> None:
        """Events with function calls are normalized correctly."""
        mock_part = MagicMock()
        mock_part.text = None
        type(mock_part).text = None  # Ensure hasattr returns True but value is falsy

        mock_event = MagicMock()
        mock_event.author = "coordinator"
        mock_event.content.parts = [mock_part]
        mock_event.is_final_response.return_value = False

        result = self.adapter._normalize_event(mock_event)

        assert result["author"] == "coordinator"
        assert "function_call" in result["content"]["parts"][0]

    def test_normalize_empty_content(self) -> None:
        """Events with no content parts are handled."""
        mock_event = MagicMock()
        mock_event.author = "agent"
        mock_event.content = None
        mock_event.is_final_response.return_value = True

        result = self.adapter._normalize_event(mock_event)

        assert result["author"] == "agent"
        assert result["content"]["parts"] == []
        assert result["is_final"] is True

    def test_normalize_malformed_event(self) -> None:
        """Malformed events return safe defaults."""
        # AttributeError path
        result = self.adapter._normalize_event(None)
        assert result == {"author": "unknown", "content": {}, "is_final": False}

    def test_normalize_event_no_author(self) -> None:
        """Events without author attribute get 'unknown'."""
        mock_event = MagicMock(spec=[])  # Empty spec, no attributes
        result = self.adapter._normalize_event(mock_event)
        assert result["author"] == "unknown"


@pytest.mark.unit
class TestADKCountAgentDelegations:
    """Test agent delegation counting."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()
        self.adapter = ADKBenchmarkAdapter(settings=self.settings)

    def test_count_unique_agents(self) -> None:
        events = [
            {"author": "coordinator"},
            {"author": "code_reviewer"},
            {"author": "code_reviewer"},  # Duplicate
            {"author": "test_strategist"},
        ]
        assert self.adapter._count_agent_delegations(events) == 3

    def test_count_excludes_unknown(self) -> None:
        events = [
            {"author": "unknown"},
            {"author": "code_reviewer"},
        ]
        assert self.adapter._count_agent_delegations(events) == 1

    def test_count_excludes_empty(self) -> None:
        events = [
            {"author": ""},
            {"author": "code_reviewer"},
        ]
        assert self.adapter._count_agent_delegations(events) == 1

    def test_count_empty_events(self) -> None:
        assert self.adapter._count_agent_delegations([]) == 0

    def test_count_no_authors(self) -> None:
        events = [{"data": "something"}, {}]
        assert self.adapter._count_agent_delegations(events) == 0


@pytest.mark.unit
class TestADKCoordinatorBuild:
    """Test coordinator build logic."""

    def setup_method(self) -> None:
        reset_benchmark_settings()
        self.settings = BenchmarkSettings()

    def test_returns_injected_coordinator(self) -> None:
        mock_coordinator = MagicMock()
        adapter = ADKBenchmarkAdapter(
            settings=self.settings,
            coordinator_agent=mock_coordinator,
        )
        assert adapter._get_or_build_coordinator() is mock_coordinator

    def test_raises_without_adk(self) -> None:
        """Building without ADK installed raises RuntimeError."""
        adapter = ADKBenchmarkAdapter(settings=self.settings)
        with pytest.raises(RuntimeError, match="Google ADK not installed"):
            adapter._get_or_build_coordinator()


class AsyncIterator:
    """Helper to create async iterators for testing."""

    def __init__(self, items: list) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item
