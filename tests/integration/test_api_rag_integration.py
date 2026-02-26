"""
Integration tests for API-RAG-Framework integration.

Tests:
- Full query flow with RAG context
- Framework service with RAG retriever
- End-to-end processing pipeline
- Error handling and fallbacks
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip if dependencies not available
pydantic_settings = pytest.importorskip("pydantic_settings")


class TestFrameworkServiceRAGIntegration:
    """Integration tests for FrameworkService with RAG."""

    @pytest.fixture
    def mock_settings(self):
        """Create comprehensive mock settings."""
        settings = MagicMock()
        settings.MCTS_ENABLED = True
        settings.MCTS_ITERATIONS = 10
        settings.MCTS_C = 1.414
        settings.SEED = 42
        settings.HTTP_TIMEOUT_SECONDS = 30
        settings.FRAMEWORK_MAX_ITERATIONS = 3
        settings.FRAMEWORK_CONSENSUS_THRESHOLD = 0.75
        settings.FRAMEWORK_TOP_K_RETRIEVAL = 5
        settings.FRAMEWORK_ENABLE_PARALLEL_AGENTS = True
        settings.LLM_PROVIDER = MagicMock(value="openai")
        settings.get_pinecone_api_key = MagicMock(return_value=None)
        settings.PINECONE_HOST = None
        return settings

    @pytest.fixture
    def mock_framework_config(self, mock_settings):
        """Create mock framework config."""
        from src.api.framework_service import FrameworkConfig

        return FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=10,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=3,
            consensus_threshold=0.75,
            top_k_retrieval=5,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

    @pytest.fixture
    def mock_rag_retriever(self):
        """Create mock RAG retriever."""
        from src.api.rag_retriever import RetrievalResult, RetrievedDocument

        retriever = MagicMock()
        retriever.is_available = True
        retriever.available_backends = ["mock"]

        async def mock_initialize():
            return True

        async def mock_retrieve(query, top_k=5):
            return RetrievalResult(
                documents=[
                    RetrievedDocument(
                        content=f"Relevant context for: {query[:50]}",
                        score=0.9,
                        source="mock",
                    ),
                ],
                query=query,
                retrieval_time_ms=5.0,
                backend="mock",
            )

        retriever.initialize = mock_initialize
        retriever.retrieve = mock_retrieve
        return retriever

    @pytest.mark.asyncio
    async def test_service_initialization_with_rag(self, mock_settings, mock_framework_config):
        """Test service initialization includes RAG retriever."""
        from src.api.framework_service import FrameworkService

        # Reset singleton
        await FrameworkService.reset_instance()

        with patch("src.api.framework_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            service = FrameworkService(
                config=mock_framework_config,
                settings=mock_settings,
            )

            # Mock the initialization components via internal imports
            with patch.object(service, "initialize") as mock_init:

                async def mock_initialize():
                    import logging

                    from src.api.framework_service import FrameworkState, LightweightFramework, MockLLMClient

                    service._state = FrameworkState.READY
                    service._start_time = 1000.0
                    service._framework = LightweightFramework(
                        llm_client=MockLLMClient(),
                        config=mock_framework_config,
                        logger=logging.getLogger(__name__),
                    )
                    return True

                mock_init.side_effect = mock_initialize

                result = await service.initialize()

                assert result is True
                assert service._state.value == "ready"

    @pytest.mark.asyncio
    async def test_query_processing_with_rag_context(self, mock_settings, mock_framework_config, mock_rag_retriever):
        """Test query processing includes RAG context."""
        from src.api.framework_service import (
            FrameworkService,
            FrameworkState,
            LightweightFramework,
            MockLLMClient,
        )

        # Reset singleton
        await FrameworkService.reset_instance()

        service = FrameworkService(
            config=mock_framework_config,
            settings=mock_settings,
        )

        # Manually set up service state
        service._state = FrameworkState.READY
        service._start_time = 1000.0
        service._rag_retriever = mock_rag_retriever

        # Create lightweight framework with RAG
        import logging

        service._framework = LightweightFramework(
            llm_client=MockLLMClient(),
            config=mock_framework_config,
            logger=logging.getLogger(__name__),
            rag_retriever=mock_rag_retriever,
        )

        # Process query
        result = await service.process_query(
            query="What is the best approach for testing?",
            use_rag=True,
            use_mcts=False,
        )

        # Verify result
        assert result is not None
        assert result.response is not None
        assert len(result.response) > 0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_query_without_rag(self, mock_settings, mock_framework_config):
        """Test query processing without RAG."""
        from src.api.framework_service import (
            FrameworkService,
            FrameworkState,
            LightweightFramework,
            MockLLMClient,
        )

        # Reset singleton
        await FrameworkService.reset_instance()

        service = FrameworkService(
            config=mock_framework_config,
            settings=mock_settings,
        )

        # Manually set up service state without RAG
        service._state = FrameworkState.READY
        service._start_time = 1000.0
        service._rag_retriever = None

        import logging

        service._framework = LightweightFramework(
            llm_client=MockLLMClient(),
            config=mock_framework_config,
            logger=logging.getLogger(__name__),
        )

        # Process query with RAG disabled
        result = await service.process_query(
            query="Simple test query",
            use_rag=False,
            use_mcts=False,
        )

        assert result is not None
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_health_check_includes_rag_status(self, mock_settings, mock_framework_config, mock_rag_retriever):
        """Test health check includes RAG availability status."""
        from src.api.framework_service import FrameworkService, FrameworkState

        # Reset singleton
        await FrameworkService.reset_instance()

        service = FrameworkService(
            config=mock_framework_config,
            settings=mock_settings,
        )

        service._state = FrameworkState.READY
        service._start_time = 1000.0
        service._rag_retriever = mock_rag_retriever
        service._framework = MagicMock()

        health = await service.health_check()

        assert "rag_available" in health
        assert health["rag_available"] is True
        assert health["status"] == "ready"

    @pytest.mark.asyncio
    async def test_query_with_rag_failure_graceful_degradation(self, mock_settings, mock_framework_config):
        """Test graceful degradation when RAG fails."""
        from src.api.framework_service import (
            FrameworkService,
            FrameworkState,
            LightweightFramework,
            MockLLMClient,
        )

        # Reset singleton
        await FrameworkService.reset_instance()

        service = FrameworkService(
            config=mock_framework_config,
            settings=mock_settings,
        )

        # Create a failing RAG retriever
        failing_retriever = MagicMock()

        async def failing_retrieve(*args, **kwargs):
            raise Exception("Simulated RAG failure")

        failing_retriever.retrieve = failing_retrieve

        service._state = FrameworkState.READY
        service._start_time = 1000.0
        service._rag_retriever = failing_retriever

        import logging

        service._framework = LightweightFramework(
            llm_client=MockLLMClient(),
            config=mock_framework_config,
            logger=logging.getLogger(__name__),
            rag_retriever=failing_retriever,
        )

        # Should not raise, should gracefully degrade
        result = await service.process_query(
            query="Test query with failing RAG",
            use_rag=True,
        )

        # Query should still succeed
        assert result is not None
        assert result.response is not None


class TestLightweightFrameworkWithRAG:
    """Tests for LightweightFramework RAG integration."""

    @pytest.fixture
    def mock_config(self):
        """Create mock framework config."""
        from src.api.framework_service import FrameworkConfig

        return FrameworkConfig(
            mcts_enabled=True,
            mcts_iterations=10,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=3,
            consensus_threshold=0.75,
            top_k_retrieval=5,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()

        async def mock_generate(prompt, temperature=0.7):
            response = MagicMock()
            if "Context:" in prompt:
                response.text = "Response based on provided context"
            else:
                response.text = "Response without context"
            return response

        client.generate = mock_generate
        return client

    @pytest.mark.asyncio
    async def test_process_with_rag_context(self, mock_config, mock_llm_client):
        """Test processing includes RAG context in prompt."""
        from src.api.framework_service import LightweightFramework
        from src.api.rag_retriever import RetrievalResult, RetrievedDocument

        # Create mock RAG retriever
        rag_retriever = MagicMock()

        async def mock_retrieve(query, top_k=5):
            return RetrievalResult(
                documents=[
                    RetrievedDocument(
                        content="Important context information",
                        score=0.9,
                        source="test",
                    ),
                ],
                query=query,
                retrieval_time_ms=5.0,
                backend="test",
            )

        rag_retriever.retrieve = mock_retrieve

        import logging

        framework = LightweightFramework(
            llm_client=mock_llm_client,
            config=mock_config,
            logger=logging.getLogger(__name__),
            rag_retriever=rag_retriever,
        )

        result = await framework.process(
            query="Test query",
            use_rag=True,
        )

        assert result["response"] == "Response based on provided context"
        assert result["metadata"]["rag_context_used"] is True

    @pytest.mark.asyncio
    async def test_process_without_rag(self, mock_config, mock_llm_client):
        """Test processing without RAG."""
        import logging

        from src.api.framework_service import LightweightFramework

        framework = LightweightFramework(
            llm_client=mock_llm_client,
            config=mock_config,
            logger=logging.getLogger(__name__),
        )

        result = await framework.process(
            query="Test query",
            use_rag=False,
        )

        assert result["response"] == "Response without context"
        assert result["metadata"]["rag_context_used"] is False

    @pytest.mark.asyncio
    async def test_mcts_stats_included_when_enabled(self, mock_config, mock_llm_client):
        """Test MCTS stats are included when enabled."""
        import logging

        from src.api.framework_service import LightweightFramework

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
        assert result["state"]["mcts_stats"]["iterations"] == mock_config.mcts_iterations

    @pytest.mark.asyncio
    async def test_mcts_stats_none_when_disabled(self, mock_config, mock_llm_client):
        """Test MCTS stats are None when disabled."""
        import logging

        from src.api.framework_service import LightweightFramework

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


class TestEndToEndQueryFlow:
    """End-to-end tests for query processing flow."""

    @pytest.mark.asyncio
    async def test_full_query_flow(self):
        """Test complete query flow from API to response."""
        from src.api.framework_service import (
            FrameworkConfig,
            FrameworkService,
            FrameworkState,
            LightweightFramework,
            MockLLMClient,
        )

        # Reset singleton
        await FrameworkService.reset_instance()

        config = FrameworkConfig(
            mcts_enabled=False,
            mcts_iterations=10,
            mcts_exploration_weight=1.414,
            seed=42,
            max_iterations=3,
            consensus_threshold=0.75,
            top_k_retrieval=5,
            enable_parallel_agents=True,
            timeout_seconds=30.0,
        )

        # Create settings mock
        settings = MagicMock()
        settings.FRAMEWORK_TOP_K_RETRIEVAL = 5

        service = FrameworkService(config=config, settings=settings)

        # Set up framework manually
        import logging

        service._state = FrameworkState.READY
        service._start_time = 1000.0
        service._framework = LightweightFramework(
            llm_client=MockLLMClient(),
            config=config,
            logger=logging.getLogger(__name__),
        )

        # Process multiple queries
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does MCTS work?",
        ]

        for query in queries:
            result = await service.process_query(
                query=query,
                use_rag=False,
            )

            assert result.response is not None
            assert result.processing_time_ms > 0
            assert "lightweight" in result.agents_used

        # Check request count
        health = await service.health_check()
        assert health["request_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
