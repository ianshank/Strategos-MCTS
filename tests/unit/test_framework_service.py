"""
Unit tests for Framework Service module.

Tests:
- FrameworkService initialization
- Query processing
- Health checks
- Configuration handling
- Error handling
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip if pydantic_settings not available (required for Settings)
pydantic_settings = pytest.importorskip("pydantic_settings")

from src.api.framework_service import (  # noqa: E402
    FrameworkConfig,
    FrameworkService,
    FrameworkState,
    LightweightFramework,
    MockLLMClient,
    QueryResult,
    get_framework_service,
)


class TestFrameworkConfig:
    """Tests for FrameworkConfig dataclass."""

    def test_default_values_from_settings(self):
        """Test creating config from default settings."""
        with patch("src.api.framework_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                MCTS_ENABLED=True,
                MCTS_ITERATIONS=100,
                MCTS_C=1.414,
                SEED=42,
                HTTP_TIMEOUT_SECONDS=30,
                FRAMEWORK_MAX_ITERATIONS=3,
                FRAMEWORK_CONSENSUS_THRESHOLD=0.75,
                FRAMEWORK_TOP_K_RETRIEVAL=5,
                FRAMEWORK_ENABLE_PARALLEL_AGENTS=True,
            )

            config = FrameworkConfig.from_settings()

            assert config.mcts_enabled is True
            assert config.mcts_iterations == 100
            assert config.mcts_exploration_weight == 1.414
            assert config.seed == 42
            assert config.timeout_seconds == 30.0

    def test_config_immutable(self):
        """Test that config is frozen (immutable)."""
        config = FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=100,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=3,
            consensus_threshold=0.75,
            top_k_retrieval=5,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

        with pytest.raises((AttributeError, TypeError)):  # Frozen dataclass raises on modification
            config.mcts_enabled = False  # type: ignore


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = QueryResult(
            response="Test response",
            confidence=0.85,
            agents_used=["hrm", "trm"],
            mcts_stats={"iterations": 100},
            processing_time_ms=150.5,
            metadata={"thread_id": "123"},
        )

        result_dict = result.to_dict()

        assert result_dict["response"] == "Test response"
        assert result_dict["confidence"] == 0.85
        assert result_dict["agents_used"] == ["hrm", "trm"]
        assert result_dict["mcts_stats"] == {"iterations": 100}
        assert result_dict["processing_time_ms"] == 150.5
        assert result_dict["metadata"]["thread_id"] == "123"

    def test_default_metadata(self):
        """Test default empty metadata."""
        result = QueryResult(
            response="Test",
            confidence=0.5,
            agents_used=[],
            mcts_stats=None,
            processing_time_ms=50.0,
        )

        assert result.metadata == {}


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    @pytest.mark.asyncio
    async def test_generate_returns_response(self):
        """Test that mock client generates a response."""
        client = MockLLMClient()
        response = await client.generate("Test prompt")

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text) > 0


class TestLightweightFramework:
    """Tests for LightweightFramework."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()

        async def mock_generate(*args, **kwargs):
            response = MagicMock()
            response.text = "Generated response"
            return response

        client.generate = mock_generate
        return client

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=100,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=3,
            consensus_threshold=0.75,
            top_k_retrieval=5,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

    @pytest.mark.asyncio
    async def test_process_returns_result(self, mock_llm_client, mock_config):
        """Test that process returns a valid result."""
        import logging

        framework = LightweightFramework(
            llm_client=mock_llm_client,
            config=mock_config,
            logger=logging.getLogger(__name__),
        )

        result = await framework.process(
            query="Test query",
            use_rag=True,
            use_mcts=True,
        )

        assert "response" in result
        assert "metadata" in result
        assert result["metadata"]["agents_used"] == ["lightweight"]

    @pytest.mark.asyncio
    async def test_process_includes_mcts_stats_when_enabled(self, mock_llm_client, mock_config):
        """Test that MCTS stats are included when MCTS is enabled."""
        import logging

        framework = LightweightFramework(
            llm_client=mock_llm_client,
            config=mock_config,
            logger=logging.getLogger(__name__),
        )

        result = await framework.process(
            query="Test query",
            use_mcts=True,
        )

        assert result["state"]["mcts_stats"] is not None
        assert result["state"]["mcts_stats"]["iterations"] == 100

    @pytest.mark.asyncio
    async def test_process_no_mcts_stats_when_disabled(self, mock_llm_client, mock_config):
        """Test that MCTS stats are None when MCTS is disabled."""
        import logging

        framework = LightweightFramework(
            llm_client=mock_llm_client,
            config=mock_config,
            logger=logging.getLogger(__name__),
        )

        result = await framework.process(
            query="Test query",
            use_mcts=False,
        )

        assert result["state"]["mcts_stats"] is None


class TestFrameworkService:
    """Tests for FrameworkService."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.MCTS_ENABLED = True
        settings.MCTS_ITERATIONS = 100
        settings.MCTS_C = 1.414
        settings.SEED = 42
        settings.HTTP_TIMEOUT_SECONDS = 30
        return settings

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=100,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=3,
            consensus_threshold=0.75,
            top_k_retrieval=5,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

    def test_initial_state_uninitialized(self, mock_config, mock_settings):
        """Test that service starts in uninitialized state."""
        service = FrameworkService(config=mock_config, settings=mock_settings)
        assert service.state == FrameworkState.UNINITIALIZED
        assert service.is_ready is False

    @pytest.mark.asyncio
    async def test_singleton_pattern(self, mock_config, mock_settings):
        """Test that get_instance returns singleton."""
        # Reset any existing instance
        await FrameworkService.reset_instance()

        service1 = await FrameworkService.get_instance(config=mock_config, settings=mock_settings)
        service2 = await FrameworkService.get_instance()

        assert service1 is service2

        # Cleanup
        await FrameworkService.reset_instance()

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, mock_config, mock_settings):
        """Test health check returns proper status."""
        service = FrameworkService(config=mock_config, settings=mock_settings)

        health = await service.health_check()

        assert "status" in health
        assert "uptime_seconds" in health
        assert "request_count" in health
        assert "error_count" in health
        assert health["status"] == FrameworkState.UNINITIALIZED.value

    @pytest.mark.asyncio
    async def test_process_query_requires_initialization(self, mock_config, mock_settings):
        """Test that process_query initializes if needed."""
        service = FrameworkService(config=mock_config, settings=mock_settings)

        # Mock the initialize to set up a lightweight framework
        async def mock_init():
            import logging

            service._state = FrameworkState.READY
            service._framework = LightweightFramework(
                llm_client=MockLLMClient(),
                config=mock_config,
                logger=logging.getLogger(__name__),
            )
            return True

        service.initialize = mock_init

        result = await service.process_query("Test query")

        assert isinstance(result, QueryResult)
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_process_query_validates_empty_query(self, mock_config, mock_settings):
        """Test that empty query raises ValueError."""
        service = FrameworkService(config=mock_config, settings=mock_settings)

        # Mock initialization
        service._state = FrameworkState.READY
        service._framework = MagicMock()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service.process_query("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service.process_query("   ")

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(self, mock_config, mock_settings):
        """Test that shutdown resets service state."""
        service = FrameworkService(config=mock_config, settings=mock_settings)
        service._state = FrameworkState.READY
        service._framework = MagicMock()

        await service.shutdown()

        assert service.state == FrameworkState.UNINITIALIZED
        assert service._framework is None


class TestFrameworkState:
    """Tests for FrameworkState enum."""

    def test_all_states_have_values(self):
        """Test that all states have string values."""
        assert FrameworkState.UNINITIALIZED.value == "uninitialized"
        assert FrameworkState.INITIALIZING.value == "initializing"
        assert FrameworkState.READY.value == "ready"
        assert FrameworkState.ERROR.value == "error"
        assert FrameworkState.SHUTTING_DOWN.value == "shutting_down"


class TestGetFrameworkService:
    """Tests for get_framework_service convenience function."""

    @pytest.mark.asyncio
    async def test_returns_service_instance(self):
        """Test that function returns a service instance."""
        # Reset any existing instance
        await FrameworkService.reset_instance()

        with patch("src.api.framework_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                MCTS_ENABLED=True,
                MCTS_ITERATIONS=100,
                MCTS_C=1.414,
                SEED=42,
                HTTP_TIMEOUT_SECONDS=30,
                FRAMEWORK_MAX_ITERATIONS=3,
                FRAMEWORK_CONSENSUS_THRESHOLD=0.75,
                FRAMEWORK_TOP_K_RETRIEVAL=5,
                FRAMEWORK_ENABLE_PARALLEL_AGENTS=True,
            )

            service = await get_framework_service()
            assert isinstance(service, FrameworkService)

        # Cleanup
        await FrameworkService.reset_instance()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
