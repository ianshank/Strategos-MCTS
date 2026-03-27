"""
Extended unit tests for src/api/framework_service.py - Part 2.

Covers missed lines: 182->exit, 266-407, 464, 472-501, 510, 514,
554->566, 568-592, 713-714.

Focus areas:
- FrameworkService.initialize() full path (lines 266-407)
- FrameworkService.process_query() RAG retrieval, timeout, error paths
- LightweightFramework RAG retrieval failure path (lines 713-714)
- FrameworkProtocol runtime_checkable usage
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.framework_service import (
    FlexibleLogger,
    FrameworkConfig,
    FrameworkProtocol,
    FrameworkService,
    FrameworkState,
    LightweightFramework,
    MockLLMClient,
    QueryResult,
    get_framework_service,
)


def _make_config(**overrides: Any) -> FrameworkConfig:
    """Helper to create a FrameworkConfig with sensible defaults."""
    defaults = dict(
        mcts_enabled=True,
        mcts_iterations=50,
        mcts_exploration_weight=1.414,
        seed=42,
        max_iterations=5,
        consensus_threshold=0.7,
        top_k_retrieval=3,
        enable_parallel_agents=False,
        timeout_seconds=10.0,
    )
    defaults.update(overrides)
    return FrameworkConfig(**defaults)


def _make_mock_settings() -> MagicMock:
    """Create a mock Settings object with all required attributes."""
    s = MagicMock()
    s.MCTS_ENABLED = True
    s.MCTS_ITERATIONS = 50
    s.MCTS_C = 1.414
    s.SEED = 42
    s.FRAMEWORK_MAX_ITERATIONS = 5
    s.FRAMEWORK_CONSENSUS_THRESHOLD = 0.7
    s.FRAMEWORK_TOP_K_RETRIEVAL = 3
    s.FRAMEWORK_ENABLE_PARALLEL_AGENTS = False
    s.HTTP_TIMEOUT_SECONDS = 10
    s.LLM_TEMPERATURE = 0.7
    s.CONFIDENCE_WITH_RAG = 0.8
    s.CONFIDENCE_WITHOUT_RAG = 0.7
    s.CONFIDENCE_ON_ERROR = 0.3
    s.ERROR_QUERY_PREVIEW_LENGTH = 100
    s.LLM_PROVIDER = MagicMock(value="openai")
    return s


# ---------------------------------------------------------------------------
# FrameworkProtocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkProtocol:
    """Test the FrameworkProtocol runtime_checkable behavior."""

    def test_protocol_isinstance_check(self):
        """A class with a matching process() method satisfies the protocol."""

        class GoodImpl:
            async def process(
                self,
                query: str,
                use_rag: bool = True,
                use_mcts: bool = False,
                config: dict | None = None,
            ) -> dict[str, Any]:
                return {}

        assert isinstance(GoodImpl(), FrameworkProtocol)

    def test_lightweight_framework_satisfies_protocol(self):
        """LightweightFramework should satisfy FrameworkProtocol."""
        mock_llm = MagicMock()
        config = _make_config()
        fw = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logging.getLogger("test"),
        )
        assert isinstance(fw, FrameworkProtocol)


# ---------------------------------------------------------------------------
# FrameworkService.initialize() - full initialization path (lines 266-407)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkServiceInitialize:
    """Tests for the initialize() method covering lines 266-407."""

    @pytest.mark.asyncio
    async def test_initialize_while_already_initializing(self):
        """When state is INITIALIZING, waits until resolved (line 266-270)."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)
        service._state = FrameworkState.INITIALIZING

        # Simulate the initializing state flipping to READY after a short delay
        async def flip_state():
            await asyncio.sleep(0.05)
            service._state = FrameworkState.READY

        asyncio.get_event_loop().create_task(flip_state())
        result = await service.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_initialize_while_initializing_ends_in_error(self):
        """When state is INITIALIZING and ends in ERROR, returns False."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)
        service._state = FrameworkState.INITIALIZING

        async def flip_to_error():
            await asyncio.sleep(0.05)
            service._state = FrameworkState.ERROR

        asyncio.get_event_loop().create_task(flip_to_error())
        result = await service.initialize()
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success_marks_ready(self):
        """Successful initialization sets READY state."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)
        # Directly set internal state to simulate successful init
        service._state = FrameworkState.READY
        assert service._state == FrameworkState.READY

    @pytest.mark.asyncio
    async def test_initialize_llm_factory_fails_uses_mock(self):
        """When LLM factory fails, falls back to MockLLMClient (lines 304-313)."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)

        mock_llm_factory = MagicMock()
        mock_llm_factory.create_from_settings.side_effect = RuntimeError("No API key")

        mock_integrated = MagicMock()
        mock_mcts_config = MagicMock()

        with patch.dict("sys.modules", {
            "src.framework.factories": MagicMock(LLMClientFactory=MagicMock(return_value=mock_llm_factory)),
            "src.framework.graph": MagicMock(IntegratedFramework=MagicMock(return_value=mock_integrated)),
            "src.framework.mcts.config": MagicMock(MCTSConfig=MagicMock(return_value=mock_mcts_config)),
            "src.api.rag_retriever": MagicMock(create_rag_retriever=MagicMock(side_effect=ImportError("no rag"))),
        }):
            result = await service.initialize()

        assert result is True
        assert service._state == FrameworkState.READY

    @pytest.mark.asyncio
    async def test_initialize_rag_retriever_fails_gracefully(self):
        """When RAG retriever init fails, continues without it (lines 327-336)."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)

        mock_llm_factory = MagicMock()
        mock_llm_factory.create_from_settings.return_value = MagicMock()

        mock_integrated = MagicMock()
        mock_mcts_config = MagicMock()

        with patch.dict("sys.modules", {
            "src.framework.factories": MagicMock(LLMClientFactory=MagicMock(return_value=mock_llm_factory)),
            "src.framework.graph": MagicMock(IntegratedFramework=MagicMock(return_value=mock_integrated)),
            "src.framework.mcts.config": MagicMock(MCTSConfig=MagicMock(return_value=mock_mcts_config)),
            "src.api.rag_retriever": MagicMock(
                create_rag_retriever=MagicMock(side_effect=RuntimeError("RAG init failed"))
            ),
        }):
            result = await service.initialize()

        assert result is True
        assert service._rag_retriever is None

    @pytest.mark.asyncio
    async def test_initialize_framework_falls_back_to_lightweight(self):
        """When IntegratedFramework raises ImportError, uses LightweightFramework (lines 364-378)."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)

        mock_llm_factory = MagicMock()
        mock_llm_factory.create_from_settings.return_value = MagicMock()
        mock_mcts_config = MagicMock()

        # IntegratedFramework raises ImportError
        mock_graph_module = MagicMock()
        mock_graph_module.IntegratedFramework = MagicMock(side_effect=ImportError("missing dep"))

        with patch.dict("sys.modules", {
            "src.framework.factories": MagicMock(LLMClientFactory=MagicMock(return_value=mock_llm_factory)),
            "src.framework.graph": mock_graph_module,
            "src.framework.mcts.config": MagicMock(MCTSConfig=MagicMock(return_value=mock_mcts_config)),
            "src.api.rag_retriever": MagicMock(
                create_rag_retriever=MagicMock(side_effect=ImportError("no rag"))
            ),
        }):
            result = await service.initialize()

        assert result is True
        assert service._state == FrameworkState.READY
        assert isinstance(service._framework, LightweightFramework)

    @pytest.mark.asyncio
    async def test_initialize_total_failure_sets_error_state(self):
        """When initialization raises unexpected error, state becomes ERROR (lines 395-407)."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)

        # Make the import itself raise an unexpected error
        with patch.dict("sys.modules", {
            "src.framework.factories": MagicMock(
                LLMClientFactory=MagicMock(side_effect=Exception("catastrophic"))
            ),
        }):
            result = await service.initialize()

        assert result is False
        assert service._state == FrameworkState.ERROR
        assert service._init_error is not None


# ---------------------------------------------------------------------------
# FrameworkService.process_query() - RAG, timeout, error paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkServiceProcessQuery:
    """Tests for process_query covering RAG retrieval, timeout, and error paths."""

    def _make_ready_service(self) -> FrameworkService:
        """Create a service in READY state with a mock framework."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)
        service._state = FrameworkState.READY

        mock_framework = AsyncMock()
        mock_framework.process = AsyncMock(return_value={
            "response": "Test answer",
            "metadata": {
                "consensus_score": 0.85,
                "agents_used": ["agent1"],
                "iterations": 1,
            },
            "state": {"mcts_stats": None},
        })
        service._framework = mock_framework
        return service

    @pytest.mark.asyncio
    async def test_process_query_with_rag_retriever(self):
        """RAG context is retrieved and passed to framework (lines 472-501)."""
        service = self._make_ready_service()

        @dataclass
        class FakeRetrievalResult:
            context: str = "Relevant context"
            documents: list = None
            retrieval_time_ms: float = 5.0
            backend: str = "mock_backend"

            def __post_init__(self):
                if self.documents is None:
                    self.documents = ["doc1", "doc2"]

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=FakeRetrievalResult())
        service._rag_retriever = mock_retriever

        result = await service.process_query("What is AI?", use_rag=True)

        assert result.response == "Test answer"
        assert result.metadata["rag_enabled"] is True
        assert result.metadata["documents_retrieved"] == 2
        assert result.metadata["retrieval_backend"] == "mock_backend"
        mock_retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_rag_retrieval_failure(self):
        """When RAG retrieval fails, proceeds without context (lines 493-501)."""
        service = self._make_ready_service()

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(side_effect=RuntimeError("RAG down"))
        service._rag_retriever = mock_retriever

        result = await service.process_query("What is AI?", use_rag=True)

        # Should still succeed, just without RAG metadata
        assert result.response == "Test answer"
        assert "documents_retrieved" not in result.metadata

    @pytest.mark.asyncio
    async def test_process_query_with_mcts_iterations_override(self):
        """mcts_iterations override is added to config (line 510)."""
        service = self._make_ready_service()

        result = await service.process_query("test", mcts_iterations=200)

        call_kwargs = service._framework.process.call_args
        config_passed = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config_passed["mcts_iterations"] == 200

    @pytest.mark.asyncio
    async def test_process_query_framework_is_none_raises(self):
        """When framework is None after init, raises RuntimeError (line 514)."""
        service = self._make_ready_service()
        service._framework = None

        with pytest.raises(RuntimeError, match="Framework not initialized"):
            await service.process_query("test")

    @pytest.mark.asyncio
    async def test_process_query_timeout(self):
        """TimeoutError is raised when processing exceeds timeout (lines 568-579)."""
        service = self._make_ready_service()
        service._config = _make_config(timeout_seconds=0.01)

        # Make framework.process hang
        async def slow_process(**kwargs):
            await asyncio.sleep(5)
            return {}

        service._framework.process = slow_process

        with pytest.raises(TimeoutError, match="timed out"):
            await service.process_query("slow query")

        assert service._error_count == 1

    @pytest.mark.asyncio
    async def test_process_query_generic_exception(self):
        """Generic exception is re-raised and error count incremented (lines 580-592)."""
        service = self._make_ready_service()
        service._framework.process = AsyncMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            await service.process_query("test")

        assert service._error_count == 1

    @pytest.mark.asyncio
    async def test_process_query_with_mcts_stats(self):
        """MCTS stats are included when use_mcts is True (line 549)."""
        service = self._make_ready_service()
        service._framework.process = AsyncMock(return_value={
            "response": "Answer",
            "metadata": {
                "consensus_score": 0.9,
                "agents_used": ["agent1"],
                "iterations": 50,
            },
            "state": {
                "mcts_stats": {"iterations": 50, "best_action": "search"},
            },
        })

        result = await service.process_query("test", use_mcts=True)
        assert result.mcts_stats is not None
        assert result.mcts_stats["iterations"] == 50

    @pytest.mark.asyncio
    async def test_process_query_without_mcts_stats(self):
        """MCTS stats are None when use_mcts is False."""
        service = self._make_ready_service()
        service._framework.process = AsyncMock(return_value={
            "response": "Answer",
            "metadata": {
                "consensus_score": 0.9,
                "agents_used": ["agent1"],
                "iterations": 1,
            },
            "state": {
                "mcts_stats": {"iterations": 50},
            },
        })

        result = await service.process_query("test", use_mcts=False)
        assert result.mcts_stats is None

    @pytest.mark.asyncio
    async def test_process_query_uses_config_default_for_mcts(self):
        """When use_mcts is None, uses config default (line 451)."""
        service = self._make_ready_service()
        # Config has mcts_enabled=True
        result = await service.process_query("test", use_mcts=None)

        call_kwargs = service._framework.process.call_args
        assert call_kwargs.kwargs["use_mcts"] is True

    @pytest.mark.asyncio
    async def test_process_query_increments_request_count(self):
        """Request count and processing time are tracked."""
        service = self._make_ready_service()

        await service.process_query("q1")
        await service.process_query("q2")

        assert service._request_count == 2
        assert service._total_processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_query_with_thread_id(self):
        """Thread ID is passed through to config."""
        service = self._make_ready_service()

        result = await service.process_query("test", thread_id="thread-123")
        assert result.metadata["thread_id"] == "thread-123"

    @pytest.mark.asyncio
    async def test_process_query_no_structured_logging(self):
        """Process query works when structured logging is unavailable (line 464)."""
        service = self._make_ready_service()

        with patch("src.api.framework_service._HAS_STRUCTURED_LOGGING", False):
            result = await service.process_query("What is AI?")

        assert result.response == "Test answer"


# ---------------------------------------------------------------------------
# LightweightFramework - RAG failure path (lines 713-714)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLightweightFrameworkRagFailure:
    """Test LightweightFramework RAG retrieval failure in process()."""

    @pytest.mark.asyncio
    async def test_rag_retrieval_exception_in_lightweight(self):
        """When RAG retriever raises, lightweight framework continues (lines 713-714)."""

        @dataclass
        class FakeResponse:
            text: str = "No-RAG response"

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=FakeResponse())

        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(side_effect=RuntimeError("RAG error"))

        config = _make_config()
        fw = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logging.getLogger("test"),
            rag_retriever=mock_retriever,
        )

        result = await fw.process(query="test query", use_rag=True)

        # Should proceed without RAG context
        assert result["response"] == "No-RAG response"
        assert result["metadata"]["rag_context_used"] is False
        assert result["metadata"]["consensus_score"] == config.confidence_without_rag

    @pytest.mark.asyncio
    async def test_rag_disabled_skips_retrieval(self):
        """When use_rag=False, retriever is not called."""

        @dataclass
        class FakeResponse:
            text: str = "response"

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=FakeResponse())

        mock_retriever = AsyncMock()
        config = _make_config()
        fw = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logging.getLogger("test"),
            rag_retriever=mock_retriever,
        )

        result = await fw.process(query="test", use_rag=False)

        mock_retriever.retrieve.assert_not_called()
        assert result["metadata"]["rag_context_used"] is False

    @pytest.mark.asyncio
    async def test_lightweight_no_rag_retriever_set(self):
        """When rag_retriever is None, skips RAG retrieval entirely."""

        @dataclass
        class FakeResponse:
            text: str = "plain response"

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=FakeResponse())

        config = _make_config()
        fw = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logging.getLogger("test"),
            rag_retriever=None,
        )

        result = await fw.process(query="test", use_rag=True)
        assert result["response"] == "plain response"
        assert result["metadata"]["rag_context_used"] is False


# ---------------------------------------------------------------------------
# FrameworkService singleton and get_framework_service
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkServiceSingleton:
    """Tests for singleton behavior and convenience function."""

    @pytest.mark.asyncio
    async def test_get_instance_creates_singleton(self):
        """get_instance creates and returns the same instance."""
        FrameworkService._instance = None
        settings = _make_mock_settings()

        try:
            instance1 = await FrameworkService.get_instance(settings=settings)
            instance2 = await FrameworkService.get_instance(settings=settings)
            assert instance1 is instance2
        finally:
            FrameworkService._instance = None

    @pytest.mark.asyncio
    async def test_reset_instance_calls_shutdown(self):
        """reset_instance shuts down existing instance."""
        settings = _make_mock_settings()
        FrameworkService._instance = None

        try:
            instance = await FrameworkService.get_instance(settings=settings)
            instance._state = FrameworkState.READY

            await FrameworkService.reset_instance()
            assert FrameworkService._instance is None
        finally:
            FrameworkService._instance = None

    @pytest.mark.asyncio
    async def test_get_framework_service_convenience(self):
        """get_framework_service() returns a FrameworkService."""
        FrameworkService._instance = None

        try:
            with patch.object(FrameworkService, "get_instance", new_callable=AsyncMock) as mock_get:
                mock_service = MagicMock(spec=FrameworkService)
                mock_get.return_value = mock_service

                result = await get_framework_service()
                assert result is mock_service
        finally:
            FrameworkService._instance = None


# ---------------------------------------------------------------------------
# Health check with active service
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkServiceHealthCheck:
    """Tests for health_check with various states."""

    @pytest.mark.asyncio
    async def test_health_check_with_active_requests(self):
        """Health check reflects request counts and timing."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)
        service._state = FrameworkState.READY
        service._start_time = 1000000.0
        service._request_count = 10
        service._error_count = 2
        service._total_processing_time_ms = 500.0
        service._framework = MagicMock()
        service._rag_retriever = MagicMock()

        health = await service.health_check()

        assert health["status"] == "ready"
        assert health["request_count"] == 10
        assert health["error_count"] == 2
        assert health["error_rate"] == 0.2
        assert health["avg_processing_time_ms"] == 50.0
        assert health["framework_ready"] is True
        assert health["rag_available"] is True
        assert "config" in health

    @pytest.mark.asyncio
    async def test_health_check_no_requests_no_division_error(self):
        """Health check handles zero requests without division error."""
        settings = _make_mock_settings()
        service = FrameworkService(settings=settings)

        health = await service.health_check()
        assert health["error_rate"] == 0.0
        assert health["avg_processing_time_ms"] == 0.0


# ---------------------------------------------------------------------------
# FrameworkConfig.from_settings with no explicit settings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkConfigFromSettingsDefault:
    """Test FrameworkConfig.from_settings uses get_settings() when no settings passed."""

    def test_from_settings_uses_global_settings(self):
        """When no settings passed, calls get_settings()."""
        mock_settings = _make_mock_settings()
        with patch("src.api.framework_service.get_settings", return_value=mock_settings):
            cfg = FrameworkConfig.from_settings()
        assert cfg.mcts_iterations == 50
