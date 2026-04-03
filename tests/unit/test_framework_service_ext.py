"""
Unit tests for src/api/framework_service.py.

Tests FrameworkConfig, QueryResult, FrameworkState, FlexibleLogger,
MockLLMClient, LightweightFramework, and FrameworkService helper methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.framework_service import (
    FlexibleLogger,
    FrameworkConfig,
    FrameworkService,
    FrameworkState,
    LightweightFramework,
    MockLLMClient,
    QueryResult,
)

# ---------------------------------------------------------------------------
# FrameworkState
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkState:
    def test_values(self):
        assert FrameworkState.UNINITIALIZED.value == "uninitialized"
        assert FrameworkState.INITIALIZING.value == "initializing"
        assert FrameworkState.READY.value == "ready"
        assert FrameworkState.ERROR.value == "error"
        assert FrameworkState.SHUTTING_DOWN.value == "shutting_down"


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueryResult:
    def test_creation_and_to_dict(self):
        qr = QueryResult(
            response="Hello",
            confidence=0.95,
            agents_used=["agent1", "agent2"],
            mcts_stats={"iterations": 100},
            processing_time_ms=42.0,
            metadata={"key": "val"},
        )
        d = qr.to_dict()
        assert d["response"] == "Hello"
        assert d["confidence"] == 0.95
        assert d["agents_used"] == ["agent1", "agent2"]
        assert d["mcts_stats"] == {"iterations": 100}
        assert d["processing_time_ms"] == 42.0
        assert d["metadata"] == {"key": "val"}

    def test_default_metadata(self):
        qr = QueryResult(
            response="test",
            confidence=0.5,
            agents_used=[],
            mcts_stats=None,
            processing_time_ms=0.0,
        )
        assert qr.metadata == {}

    def test_to_dict_none_mcts(self):
        qr = QueryResult(
            response="r",
            confidence=0.5,
            agents_used=[],
            mcts_stats=None,
            processing_time_ms=1.0,
        )
        d = qr.to_dict()
        assert d["mcts_stats"] is None


# ---------------------------------------------------------------------------
# FrameworkConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkConfig:
    def test_defaults(self):
        cfg = FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=100,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=5,
            consensus_threshold=0.7,
            top_k_retrieval=3,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )
        assert cfg.llm_temperature == 0.7
        assert cfg.confidence_with_rag == 0.8
        assert cfg.confidence_without_rag == 0.7
        assert cfg.confidence_on_error == 0.3
        assert cfg.error_query_preview_length == 100

    def test_frozen(self):
        cfg = FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=100,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=5,
            consensus_threshold=0.7,
            top_k_retrieval=3,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )
        with pytest.raises(AttributeError):
            cfg.mcts_enabled = False  # type: ignore

    def test_from_settings(self):
        mock_settings = MagicMock()
        mock_settings.MCTS_ENABLED = True
        mock_settings.MCTS_ITERATIONS = 200
        mock_settings.MCTS_C = 2.0
        mock_settings.SEED = 99
        mock_settings.FRAMEWORK_MAX_ITERATIONS = 10
        mock_settings.FRAMEWORK_CONSENSUS_THRESHOLD = 0.8
        mock_settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
        mock_settings.FRAMEWORK_ENABLE_PARALLEL_AGENTS = False
        mock_settings.HTTP_TIMEOUT_SECONDS = 60
        mock_settings.LLM_TEMPERATURE = 0.5
        mock_settings.CONFIDENCE_WITH_RAG = 0.9
        mock_settings.CONFIDENCE_WITHOUT_RAG = 0.6
        mock_settings.CONFIDENCE_ON_ERROR = 0.2
        mock_settings.ERROR_QUERY_PREVIEW_LENGTH = 50

        cfg = FrameworkConfig.from_settings(mock_settings)
        assert cfg.mcts_enabled is True
        assert cfg.mcts_iterations == 200
        assert cfg.mcts_exploration_weight == 2.0
        assert cfg.seed == 99
        assert cfg.max_iterations == 10
        assert cfg.timeout_seconds == 60.0
        assert cfg.llm_temperature == 0.5
        assert cfg.confidence_with_rag == 0.9


# ---------------------------------------------------------------------------
# FlexibleLogger
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlexibleLogger:
    def test_info(self):
        mock_logger = MagicMock(spec=logging.Logger)
        fl = FlexibleLogger(mock_logger)
        fl.info("test message")
        mock_logger.log.assert_called_once_with(logging.INFO, "test message")

    def test_debug(self):
        mock_logger = MagicMock(spec=logging.Logger)
        fl = FlexibleLogger(mock_logger)
        fl.debug("debug msg")
        mock_logger.log.assert_called_once_with(logging.DEBUG, "debug msg")

    def test_warning(self):
        mock_logger = MagicMock(spec=logging.Logger)
        fl = FlexibleLogger(mock_logger)
        fl.warning("warn msg")
        mock_logger.log.assert_called_once_with(logging.WARNING, "warn msg")

    def test_error(self):
        mock_logger = MagicMock(spec=logging.Logger)
        fl = FlexibleLogger(mock_logger)
        fl.error("error msg")
        mock_logger.log.assert_called_once_with(logging.ERROR, "error msg")

    def test_fallback_on_type_error(self):
        """If logger doesn't support kwargs, should fall back to simple message."""
        mock_logger = MagicMock(spec=logging.Logger)
        mock_logger.log.side_effect = [TypeError("bad kwarg"), None]
        fl = FlexibleLogger(mock_logger)
        fl.info("structured msg", fallback_msg="simple msg", extra_key="val")
        assert mock_logger.log.call_count == 2
        # Second call should be the fallback
        mock_logger.log.assert_called_with(logging.INFO, "simple msg")


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMockLLMClient:
    @pytest.mark.asyncio
    async def test_generate(self):
        client = MockLLMClient()
        response = await client.generate("test prompt")
        assert hasattr(response, "text")
        assert "mock response" in response.text.lower()

    @pytest.mark.asyncio
    async def test_generate_with_temperature(self):
        client = MockLLMClient()
        response = await client.generate("prompt", temperature=0.9)
        assert response.text is not None


# ---------------------------------------------------------------------------
# LightweightFramework
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLightweightFramework:
    def _make_config(self) -> FrameworkConfig:
        return FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=50,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=5,
            consensus_threshold=0.7,
            top_k_retrieval=3,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

    @pytest.mark.asyncio
    async def test_process_basic(self):
        @dataclass
        class FakeResponse:
            text: str = "LLM response"

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=FakeResponse())
        config = self._make_config()
        logger = logging.getLogger("test")

        fw = LightweightFramework(llm_client=mock_llm, config=config, logger=logger)
        result = await fw.process(query="What is AI?", use_rag=False, use_mcts=False)

        assert result["response"] == "LLM response"
        assert result["metadata"]["agents_used"] == ["lightweight"]
        assert result["metadata"]["consensus_score"] == config.confidence_without_rag
        assert result["state"]["mcts_stats"] is None

    @pytest.mark.asyncio
    async def test_process_with_mcts(self):
        @dataclass
        class FakeResponse:
            text: str = "response"

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=FakeResponse())
        config = self._make_config()
        logger = logging.getLogger("test")

        fw = LightweightFramework(llm_client=mock_llm, config=config, logger=logger)
        result = await fw.process(query="test", use_mcts=True)

        assert result["state"]["mcts_stats"] is not None
        assert result["state"]["mcts_stats"]["iterations"] == 50

    @pytest.mark.asyncio
    async def test_process_llm_failure(self):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        config = self._make_config()
        logger = logging.getLogger("test")

        fw = LightweightFramework(llm_client=mock_llm, config=config, logger=logger)
        result = await fw.process(query="test query", use_rag=False)

        assert "Unable to process query" in result["response"]
        assert result["metadata"]["consensus_score"] == config.confidence_on_error

    @pytest.mark.asyncio
    async def test_process_with_rag_retriever(self):
        @dataclass
        class FakeResponse:
            text: str = "RAG response"

        @dataclass
        class FakeRetrieval:
            context: str = "Some context"
            documents: list = None
            retrieval_time_ms: float = 5.0
            backend: str = "mock"

            def __post_init__(self):
                if self.documents is None:
                    self.documents = ["doc1"]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=FakeResponse())
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(return_value=FakeRetrieval())
        config = self._make_config()
        logger = logging.getLogger("test")

        fw = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logger,
            rag_retriever=mock_retriever,
        )
        result = await fw.process(query="test", use_rag=True)

        assert result["response"] == "RAG response"
        assert result["metadata"]["consensus_score"] == config.confidence_with_rag
        assert result["metadata"]["rag_context_used"] is True


# ---------------------------------------------------------------------------
# FrameworkService
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrameworkService:
    def _make_service(self) -> FrameworkService:
        mock_settings = MagicMock()
        mock_settings.MCTS_ENABLED = False
        mock_settings.MCTS_ITERATIONS = 10
        mock_settings.MCTS_C = 1.414
        mock_settings.SEED = 42
        mock_settings.FRAMEWORK_MAX_ITERATIONS = 3
        mock_settings.FRAMEWORK_CONSENSUS_THRESHOLD = 0.7
        mock_settings.FRAMEWORK_TOP_K_RETRIEVAL = 3
        mock_settings.FRAMEWORK_ENABLE_PARALLEL_AGENTS = False
        mock_settings.HTTP_TIMEOUT_SECONDS = 10
        mock_settings.LLM_TEMPERATURE = 0.7
        mock_settings.CONFIDENCE_WITH_RAG = 0.8
        mock_settings.CONFIDENCE_WITHOUT_RAG = 0.7
        mock_settings.CONFIDENCE_ON_ERROR = 0.3
        mock_settings.ERROR_QUERY_PREVIEW_LENGTH = 100
        return FrameworkService(settings=mock_settings)

    def test_initial_state(self):
        service = self._make_service()
        assert service.state == FrameworkState.UNINITIALIZED
        assert service.is_ready is False
        assert service._request_count == 0
        assert service._error_count == 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        service = self._make_service()
        health = await service.health_check()
        assert health["status"] == "uninitialized"
        assert health["request_count"] == 0
        assert health["framework_ready"] is False

    @pytest.mark.asyncio
    async def test_shutdown(self):
        service = self._make_service()
        service._state = FrameworkState.READY
        await service.shutdown()
        assert service.state == FrameworkState.UNINITIALIZED
        assert service._framework is None

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        service = self._make_service()
        service._state = FrameworkState.SHUTTING_DOWN
        await service.shutdown()
        # Should return immediately
        assert service.state == FrameworkState.SHUTTING_DOWN

    @pytest.mark.asyncio
    async def test_process_query_not_initialized(self):
        service = self._make_service()
        # Make initialize fail
        with patch.object(service, "initialize", return_value=False):
            with pytest.raises(RuntimeError, match="Framework not available"):
                await service.process_query("test")

    @pytest.mark.asyncio
    async def test_process_query_empty_query(self):
        service = self._make_service()
        service._state = FrameworkState.READY
        service._framework = MagicMock()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service.process_query("")

    @pytest.mark.asyncio
    async def test_process_query_whitespace_query(self):
        service = self._make_service()
        service._state = FrameworkState.READY
        service._framework = MagicMock()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service.process_query("   ")

    @pytest.mark.asyncio
    async def test_process_query_success(self):
        service = self._make_service()
        service._state = FrameworkState.READY

        mock_framework = AsyncMock()
        mock_framework.process = AsyncMock(return_value={
            "response": "Answer",
            "metadata": {
                "consensus_score": 0.9,
                "agents_used": ["agent1"],
                "iterations": 1,
            },
            "state": {},
        })
        service._framework = mock_framework

        result = await service.process_query("What is AI?")
        assert result.response == "Answer"
        assert result.confidence == 0.9
        assert service._request_count == 1

    @pytest.mark.asyncio
    async def test_initialize_already_ready(self):
        service = self._make_service()
        service._state = FrameworkState.READY
        result = await service.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_reset_instance(self):
        """Test that reset_instance cleans up."""
        FrameworkService._instance = None  # ensure clean state
        await FrameworkService.reset_instance()
        assert FrameworkService._instance is None
