"""
Unit tests for graph.py validation and error handling.

Tests input validation, error recovery, and graceful degradation.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")


class TestEntryNodeValidation:
    """Tests for _entry_node query validation."""

    @pytest.fixture
    def mock_graph_builder(self):
        """Create a minimal mock graph builder for testing."""
        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", False):
            from src.framework.graph import GraphBuilder

            # Create mock dependencies
            mock_adapter = MagicMock()
            mock_logger = MagicMock()

            builder = GraphBuilder.__new__(GraphBuilder)
            builder.model_adapter = mock_adapter
            builder.logger = mock_logger
            builder.vector_store = None
            builder.top_k_retrieval = 5
            builder.mcts_config = MagicMock()
            builder.mcts_config.to_dict.return_value = {}

            return builder

    def test_valid_query_passes(self, mock_graph_builder):
        """Test that valid query passes validation."""
        state = {"query": "What is machine learning?"}
        result = mock_graph_builder._entry_node(state)

        assert result["iteration"] == 0
        assert result["agent_outputs"] == []
        mock_graph_builder.logger.info.assert_called_once()

    def test_empty_query_raises_error(self, mock_graph_builder):
        """Test that empty query raises ValueError."""
        state = {"query": ""}

        with pytest.raises(ValueError, match="must be a non-empty string"):
            mock_graph_builder._entry_node(state)

    def test_whitespace_only_query_raises_error(self, mock_graph_builder):
        """Test that whitespace-only query raises ValueError."""
        state = {"query": "   \n\t  "}

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            mock_graph_builder._entry_node(state)

    def test_none_query_raises_error(self, mock_graph_builder):
        """Test that None query raises ValueError."""
        state = {"query": None}

        with pytest.raises(ValueError, match="must be a non-empty string"):
            mock_graph_builder._entry_node(state)

    def test_missing_query_key_raises_error(self, mock_graph_builder):
        """Test that missing query key raises ValueError."""
        state = {}

        with pytest.raises(ValueError, match="must be a non-empty string"):
            mock_graph_builder._entry_node(state)

    def test_non_string_query_raises_error(self, mock_graph_builder):
        """Test that non-string query raises ValueError."""
        state = {"query": 12345}

        with pytest.raises(ValueError, match="must be a non-empty string"):
            mock_graph_builder._entry_node(state)

    def test_long_query_is_truncated_in_log(self, mock_graph_builder):
        """Test that long queries are truncated in log messages."""
        long_query = "x" * 200
        state = {"query": long_query}

        mock_graph_builder._entry_node(state)

        # Check log was called with truncated message
        log_call = mock_graph_builder.logger.info.call_args[0][0]
        assert "..." in log_call
        assert len(log_call) < len(long_query) + 50  # Account for prefix


class TestRetrieveContextNodeErrorHandling:
    """Tests for _retrieve_context_node error handling."""

    @pytest.fixture
    def mock_graph_builder_with_vector_store(self):
        """Create mock graph builder with vector store."""
        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", False):
            from src.framework.graph import GraphBuilder

            mock_adapter = MagicMock()
            mock_logger = MagicMock()
            mock_vector_store = MagicMock()

            builder = GraphBuilder.__new__(GraphBuilder)
            builder.model_adapter = mock_adapter
            builder.logger = mock_logger
            builder.vector_store = mock_vector_store
            builder.top_k_retrieval = 5
            builder.mcts_config = MagicMock()

            return builder

    def test_successful_retrieval(self, mock_graph_builder_with_vector_store):
        """Test successful document retrieval."""
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test"}

        mock_graph_builder_with_vector_store.vector_store.similarity_search.return_value = [mock_doc]

        state = {"query": "test query", "use_rag": True}
        result = mock_graph_builder_with_vector_store._retrieve_context_node(state)

        assert result["rag_context"] == "Test content"
        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_docs"][0]["content"] == "Test content"

    def test_vector_store_failure_graceful_degradation(self, mock_graph_builder_with_vector_store):
        """Test graceful degradation when vector store fails."""
        mock_graph_builder_with_vector_store.vector_store.similarity_search.side_effect = Exception("Database error")

        state = {"query": "test query", "use_rag": True}
        result = mock_graph_builder_with_vector_store._retrieve_context_node(state)

        # Should return empty context, not raise
        assert result["rag_context"] == ""
        assert result["retrieved_docs"] == []
        # Uses logger.exception for better traceback logging
        mock_graph_builder_with_vector_store.logger.exception.assert_called_once()

    def test_rag_disabled_returns_empty(self, mock_graph_builder_with_vector_store):
        """Test that disabling RAG returns empty context."""
        state = {"query": "test query", "use_rag": False}
        result = mock_graph_builder_with_vector_store._retrieve_context_node(state)

        assert result["rag_context"] == ""
        assert result["retrieved_docs"] == []
        # Vector store should not be called
        mock_graph_builder_with_vector_store.vector_store.similarity_search.assert_not_called()

    def test_no_vector_store_returns_empty(self):
        """Test that missing vector store returns empty context."""
        with patch("src.framework.graph._META_CONTROLLER_AVAILABLE", False):
            from src.framework.graph import GraphBuilder

            builder = GraphBuilder.__new__(GraphBuilder)
            builder.vector_store = None
            builder.logger = MagicMock()

            state = {"query": "test query", "use_rag": True}
            result = builder._retrieve_context_node(state)

            assert result["rag_context"] == ""
            assert result["retrieved_docs"] == []

    def test_empty_query_returns_empty(self, mock_graph_builder_with_vector_store):
        """Test that empty query returns empty context with warning."""
        state = {"query": "", "use_rag": True}
        result = mock_graph_builder_with_vector_store._retrieve_context_node(state)

        assert result["rag_context"] == ""
        assert result["retrieved_docs"] == []
        mock_graph_builder_with_vector_store.logger.warning.assert_called_once()


class TestConfigConstants:
    """Tests for configuration constants module."""

    def test_constants_module_imports(self):
        """Test that constants module can be imported."""
        from src.config import constants as C

        # Check key constants exist
        assert hasattr(C, "DEFAULT_MCTS_ITERATIONS")
        assert hasattr(C, "DEFAULT_MCTS_C")
        assert hasattr(C, "DEFAULT_HTTP_TIMEOUT_SECONDS")
        assert hasattr(C, "API_KEY_PLACEHOLDERS")

    def test_mcts_bounds_are_valid(self):
        """Test that MCTS bounds are logically consistent."""
        from src.config import constants as C

        assert C.MIN_MCTS_ITERATIONS < C.DEFAULT_MCTS_ITERATIONS < C.MAX_MCTS_ITERATIONS
        assert C.MIN_MCTS_C <= C.DEFAULT_MCTS_C <= C.MAX_MCTS_C

    def test_http_bounds_are_valid(self):
        """Test that HTTP bounds are logically consistent."""
        from src.config import constants as C

        assert C.MIN_HTTP_TIMEOUT_SECONDS < C.DEFAULT_HTTP_TIMEOUT_SECONDS < C.MAX_HTTP_TIMEOUT_SECONDS
        assert C.MIN_HTTP_MAX_RETRIES <= C.DEFAULT_HTTP_MAX_RETRIES <= C.MAX_HTTP_MAX_RETRIES

    def test_confidence_thresholds_are_valid(self):
        """Test that confidence thresholds are in valid range."""
        from src.config import constants as C

        assert 0.0 <= C.DEFAULT_CONFIDENCE_WITH_RAG <= 1.0
        assert 0.0 <= C.DEFAULT_CONFIDENCE_WITHOUT_RAG <= 1.0
        assert 0.0 <= C.DEFAULT_CONFIDENCE_ON_ERROR <= 1.0
        # With RAG should be higher than without
        assert C.DEFAULT_CONFIDENCE_WITH_RAG >= C.DEFAULT_CONFIDENCE_WITHOUT_RAG

    def test_api_key_placeholders_are_non_empty(self):
        """Test that API key placeholders tuple is non-empty."""
        from src.config import constants as C

        assert len(C.API_KEY_PLACEHOLDERS) > 0
        assert "" in C.API_KEY_PLACEHOLDERS  # Empty string should be a placeholder
