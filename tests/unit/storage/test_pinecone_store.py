"""
Unit tests for Pinecone vector store.

Tests initialization, store operations, query, buffering, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.meta_controller.base import MetaControllerFeatures, MetaControllerPrediction

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Fixtures
# =============================================================================


def _make_features(**overrides: object) -> MetaControllerFeatures:
    """Create a MetaControllerFeatures with sensible defaults."""
    defaults = dict(
        hrm_confidence=0.8,
        trm_confidence=0.6,
        mcts_value=0.7,
        consensus_score=0.75,
        last_agent="hrm",
        iteration=1,
        query_length=50,
        has_rag_context=False,
    )
    defaults.update(overrides)
    return MetaControllerFeatures(**defaults)  # type: ignore[arg-type]


def _make_prediction(**overrides: object) -> MetaControllerPrediction:
    """Create a MetaControllerPrediction with sensible defaults."""
    defaults = dict(
        agent="hrm",
        confidence=0.9,
        probabilities={"hrm": 0.6, "trm": 0.2, "mcts": 0.2},
    )
    defaults.update(overrides)
    return MetaControllerPrediction(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def mock_pinecone_index() -> MagicMock:
    """Create a mock Pinecone index."""
    index = MagicMock()
    index.upsert = MagicMock()
    index.query = MagicMock(
        return_value={
            "matches": [
                {
                    "id": "vec-1",
                    "score": 0.95,
                    "metadata": {
                        "selected_agent": "hrm",
                        "confidence": 0.9,
                    },
                },
                {
                    "id": "vec-2",
                    "score": 0.85,
                    "metadata": {
                        "selected_agent": "trm",
                        "confidence": 0.7,
                    },
                },
            ]
        }
    )
    index.delete = MagicMock()
    index.describe_index_stats = MagicMock(
        return_value={
            "total_vector_count": 100,
            "namespaces": {"meta_controller": {"vector_count": 50}},
            "dimension": 10,
        }
    )
    return index


@pytest.fixture
def mock_pinecone_client(mock_pinecone_index: MagicMock) -> MagicMock:
    """Create a mock Pinecone client."""
    client = MagicMock()
    client.Index.return_value = mock_pinecone_index
    return client


@pytest.fixture
def available_store(mock_pinecone_client: MagicMock, mock_pinecone_index: MagicMock, monkeypatch: pytest.MonkeyPatch):
    """Create a PineconeVectorStore that is initialized and available."""
    import src.storage.pinecone_store as ps
    from src.storage.pinecone_store import PineconeVectorStore

    monkeypatch.setattr(ps, "PINECONE_AVAILABLE", True)
    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store._api_key = "test-key"
    store._host = "test-host"
    store.namespace = "meta_controller"
    store._client = mock_pinecone_client
    store._index = mock_pinecone_index
    store._is_initialized = True
    store._operation_buffer = []
    return store


@pytest.fixture
def unavailable_store() -> "PineconeVectorStore":
    """Create a PineconeVectorStore that is not available."""
    from src.storage.pinecone_store import PineconeVectorStore

    with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", False):
        store = PineconeVectorStore(api_key="test-key", host="test-host", auto_init=False)
    return store


# =============================================================================
# Module Import Tests
# =============================================================================


class TestPineconeModuleImport:
    """Tests for Pinecone module availability detection."""

    def test_pinecone_availability_flag_exists(self) -> None:
        """Test module has Pinecone availability flag."""
        from src.storage import pinecone_store

        assert hasattr(pinecone_store, "PINECONE_AVAILABLE")
        assert isinstance(pinecone_store.PINECONE_AVAILABLE, bool)

    def test_module_exports(self) -> None:
        """Test module __all__ exports."""
        from src.storage.pinecone_store import __all__

        assert "PineconeVectorStore" in __all__
        assert "PINECONE_AVAILABLE" in __all__


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPineconeVectorStoreInit:
    """Tests for PineconeVectorStore initialization."""

    def test_init_without_api_key(self) -> None:
        """Test initialization without API key does not initialize."""
        from src.storage.pinecone_store import PineconeVectorStore

        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            with patch.dict("os.environ", {}, clear=True):
                store = PineconeVectorStore(api_key=None, host=None, auto_init=True)

        assert store._is_initialized is False
        assert store.is_available is False

    def test_init_with_auto_init_false(self) -> None:
        """Test initialization with auto_init=False skips client setup."""
        from src.storage.pinecone_store import PineconeVectorStore

        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            store = PineconeVectorStore(api_key="key", host="host", auto_init=False)

        assert store._is_initialized is False
        assert store._api_key == "key"
        assert store._host == "host"

    def test_init_with_custom_namespace(self) -> None:
        """Test initialization with custom namespace."""
        from src.storage.pinecone_store import PineconeVectorStore

        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            store = PineconeVectorStore(namespace="custom_ns", auto_init=False)

        assert store.namespace == "custom_ns"

    def test_init_default_namespace(self) -> None:
        """Test default namespace is meta_controller."""
        from src.storage.pinecone_store import PineconeVectorStore

        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            store = PineconeVectorStore(auto_init=False)

        assert store.namespace == "meta_controller"

    def test_init_when_pinecone_not_installed(self) -> None:
        """Test initialization when pinecone package is not installed."""
        from src.storage.pinecone_store import PineconeVectorStore

        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", False):
            store = PineconeVectorStore(api_key="key", host="host", auto_init=True)

        assert store._is_initialized is False

    def test_init_reads_env_vars(self) -> None:
        """Test initialization reads from environment variables."""
        from src.storage.pinecone_store import PineconeVectorStore

        env = {"PINECONE_API_KEY": "env-key", "PINECONE_HOST": "env-host"}
        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            with patch.dict("os.environ", env, clear=False):
                store = PineconeVectorStore(auto_init=False)

        assert store._api_key == "env-key"
        assert store._host == "env-host"

    def test_explicit_params_override_env(self) -> None:
        """Test explicit parameters override environment variables."""
        from src.storage.pinecone_store import PineconeVectorStore

        env = {"PINECONE_API_KEY": "env-key", "PINECONE_HOST": "env-host"}
        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            with patch.dict("os.environ", env, clear=False):
                store = PineconeVectorStore(api_key="explicit-key", host="explicit-host", auto_init=False)

        assert store._api_key == "explicit-key"
        assert store._host == "explicit-host"

    def test_auto_init_calls_initialize(self, mock_pinecone_client: MagicMock) -> None:
        """Test auto_init=True triggers initialization."""
        from src.storage.pinecone_store import PineconeVectorStore

        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            with patch("src.storage.pinecone_store.Pinecone", return_value=mock_pinecone_client):
                store = PineconeVectorStore(api_key="key", host="host", auto_init=True)

        assert store._is_initialized is True

    def test_initialize_handles_connection_error(self) -> None:
        """Test _initialize handles connection errors gracefully."""
        from src.storage.pinecone_store import PineconeVectorStore

        with patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True):
            with patch("src.storage.pinecone_store.Pinecone", side_effect=ConnectionError("fail")):
                store = PineconeVectorStore(api_key="key", host="host", auto_init=True)

        assert store._is_initialized is False

    def test_vector_dimension_constant(self) -> None:
        """Test VECTOR_DIMENSION is set."""
        from src.storage.pinecone_store import PineconeVectorStore

        assert PineconeVectorStore.VECTOR_DIMENSION == 10


# =============================================================================
# is_available Property Tests
# =============================================================================


class TestPineconeIsAvailable:
    """Tests for is_available property."""

    def test_available_when_fully_initialized(self, available_store: "PineconeVectorStore") -> None:
        """Test is_available returns True when fully initialized."""
        assert available_store.is_available is True

    def test_not_available_when_not_initialized(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test is_available returns False when not initialized."""
        assert unavailable_store.is_available is False

    def test_not_available_without_api_key(self, available_store: "PineconeVectorStore") -> None:
        """Test is_available returns False without API key."""
        available_store._api_key = None
        assert available_store.is_available is False

    def test_not_available_without_host(self, available_store: "PineconeVectorStore") -> None:
        """Test is_available returns False without host."""
        available_store._host = None
        assert available_store.is_available is False


# =============================================================================
# store_prediction Tests
# =============================================================================


class TestStorePrediction:
    """Tests for store_prediction method."""

    def test_store_prediction_success(self, available_store: "PineconeVectorStore") -> None:
        """Test successful prediction storage returns vector ID."""
        features = _make_features()
        prediction = _make_prediction()

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            result = available_store.store_prediction(features, prediction)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        available_store._index.upsert.assert_called_once()

    def test_store_prediction_with_metadata(self, available_store: "PineconeVectorStore") -> None:
        """Test prediction storage with additional metadata."""
        features = _make_features()
        prediction = _make_prediction()
        extra = {"experiment": "test-run-1"}

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            result = available_store.store_prediction(features, prediction, metadata=extra)

        assert result is not None
        call_args = available_store._index.upsert.call_args
        vector_metadata = call_args[1]["vectors"][0]["metadata"]
        assert vector_metadata["experiment"] == "test-run-1"

    def test_store_prediction_buffers_when_unavailable(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test prediction is buffered when Pinecone is unavailable."""
        features = _make_features()
        prediction = _make_prediction()

        result = unavailable_store.store_prediction(features, prediction)

        assert result is None
        assert len(unavailable_store._operation_buffer) == 1
        assert unavailable_store._operation_buffer[0]["type"] == "store_prediction"

    def test_store_prediction_handles_connection_error(self, available_store: "PineconeVectorStore") -> None:
        """Test store_prediction handles connection errors."""
        features = _make_features()
        prediction = _make_prediction()
        available_store._index.upsert.side_effect = ConnectionError("network error")

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            result = available_store.store_prediction(features, prediction)

        assert result is None

    def test_store_prediction_handles_timeout(self, available_store: "PineconeVectorStore") -> None:
        """Test store_prediction handles timeout errors."""
        features = _make_features()
        prediction = _make_prediction()
        available_store._index.upsert.side_effect = TimeoutError("timeout")

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            result = available_store.store_prediction(features, prediction)

        assert result is None

    def test_store_prediction_includes_feature_metadata(self, available_store: "PineconeVectorStore") -> None:
        """Test stored vector metadata includes feature information."""
        features = _make_features(iteration=5, query_length=100, last_agent="trm", has_rag_context=True)
        prediction = _make_prediction(agent="mcts", confidence=0.85)

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            available_store.store_prediction(features, prediction)

        call_args = available_store._index.upsert.call_args
        metadata = call_args[1]["vectors"][0]["metadata"]
        assert metadata["selected_agent"] == "mcts"
        assert metadata["confidence"] == 0.85
        assert metadata["iteration"] == 5
        assert metadata["query_length"] == 100
        assert metadata["last_agent"] == "trm"
        assert metadata["has_rag_context"] is True


# =============================================================================
# find_similar_decisions Tests
# =============================================================================


class TestFindSimilarDecisions:
    """Tests for find_similar_decisions method."""

    def test_find_similar_returns_results(self, available_store: "PineconeVectorStore") -> None:
        """Test find_similar_decisions returns formatted results."""
        features = _make_features()

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            results = available_store.find_similar_decisions(features, top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == "vec-1"
        assert results[0]["score"] == 0.95
        assert "metadata" in results[0]

    def test_find_similar_returns_empty_when_unavailable(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test returns empty list when Pinecone is unavailable."""
        features = _make_features()
        results = unavailable_store.find_similar_decisions(features)
        assert results == []

    def test_find_similar_with_custom_top_k(self, available_store: "PineconeVectorStore") -> None:
        """Test custom top_k is passed to query."""
        features = _make_features()

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            available_store.find_similar_decisions(features, top_k=3)

        call_args = available_store._index.query.call_args
        assert call_args[1]["top_k"] == 3

    def test_find_similar_without_metadata(self, available_store: "PineconeVectorStore") -> None:
        """Test find_similar_decisions without metadata."""
        features = _make_features()
        available_store._index.query.return_value = {
            "matches": [{"id": "vec-1", "score": 0.9}]
        }

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            results = available_store.find_similar_decisions(features, include_metadata=False)

        assert len(results) == 1
        assert "metadata" not in results[0]

    def test_find_similar_handles_connection_error(self, available_store: "PineconeVectorStore") -> None:
        """Test handles connection errors gracefully."""
        features = _make_features()
        available_store._index.query.side_effect = ConnectionError("fail")

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            results = available_store.find_similar_decisions(features)

        assert results == []

    def test_find_similar_handles_empty_matches(self, available_store: "PineconeVectorStore") -> None:
        """Test handles empty matches list."""
        features = _make_features()
        available_store._index.query.return_value = {"matches": []}

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            results = available_store.find_similar_decisions(features)

        assert results == []


# =============================================================================
# get_agent_distribution Tests
# =============================================================================


class TestGetAgentDistribution:
    """Tests for get_agent_distribution method."""

    def test_distribution_from_similar_decisions(self, available_store: "PineconeVectorStore") -> None:
        """Test agent distribution calculation."""
        features = _make_features()

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            dist = available_store.get_agent_distribution(features)

        assert "hrm" in dist
        assert "trm" in dist
        assert "mcts" in dist
        assert dist["hrm"] == 0.5
        assert dist["trm"] == 0.5
        assert dist["mcts"] == 0.0

    def test_distribution_returns_zeros_when_unavailable(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test returns zero distribution when unavailable."""
        features = _make_features()
        dist = unavailable_store.get_agent_distribution(features)

        assert dist == {"hrm": 0.0, "trm": 0.0, "mcts": 0.0}

    def test_distribution_returns_zeros_for_no_matches(self, available_store: "PineconeVectorStore") -> None:
        """Test returns zero distribution when no similar decisions found."""
        features = _make_features()
        available_store._index.query.return_value = {"matches": []}

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            dist = available_store.get_agent_distribution(features)

        assert dist == {"hrm": 0.0, "trm": 0.0, "mcts": 0.0}

    def test_distribution_ignores_unknown_agents(self, available_store: "PineconeVectorStore") -> None:
        """Test distribution ignores unknown agent names."""
        features = _make_features()
        available_store._index.query.return_value = {
            "matches": [
                {"id": "1", "score": 0.9, "metadata": {"selected_agent": "unknown_agent"}},
                {"id": "2", "score": 0.8, "metadata": {"selected_agent": "hrm"}},
            ]
        }

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            dist = available_store.get_agent_distribution(features)

        assert dist["hrm"] == 1.0
        assert dist["trm"] == 0.0
        assert dist["mcts"] == 0.0


# =============================================================================
# store_batch Tests
# =============================================================================


class TestStoreBatch:
    """Tests for store_batch method."""

    def test_store_batch_success(self, available_store: "PineconeVectorStore") -> None:
        """Test successful batch storage."""
        features_list = [_make_features(), _make_features(iteration=2)]
        predictions_list = [_make_prediction(), _make_prediction(agent="trm")]

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            count = available_store.store_batch(features_list, predictions_list)

        assert count == 2
        available_store._index.upsert.assert_called_once()

    def test_store_batch_buffers_when_unavailable(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test batch is buffered when unavailable."""
        features_list = [_make_features()]
        predictions_list = [_make_prediction()]

        count = unavailable_store.store_batch(features_list, predictions_list)

        assert count == 0
        assert len(unavailable_store._operation_buffer) == 1
        assert unavailable_store._operation_buffer[0]["type"] == "store_batch"

    def test_store_batch_length_mismatch_raises(self, available_store: "PineconeVectorStore") -> None:
        """Test mismatched list lengths raise ValueError."""
        features_list = [_make_features(), _make_features()]
        predictions_list = [_make_prediction()]

        with pytest.raises(ValueError, match="same length"):
            available_store.store_batch(features_list, predictions_list)

    def test_store_batch_with_metadata(self, available_store: "PineconeVectorStore") -> None:
        """Test batch storage with batch metadata."""
        features_list = [_make_features()]
        predictions_list = [_make_prediction()]
        batch_meta = {"batch_id": "batch-001"}

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            count = available_store.store_batch(features_list, predictions_list, batch_metadata=batch_meta)

        assert count == 1
        call_args = available_store._index.upsert.call_args
        vector_metadata = call_args[1]["vectors"][0]["metadata"]
        assert vector_metadata["batch_id"] == "batch-001"

    def test_store_batch_handles_error(self, available_store: "PineconeVectorStore") -> None:
        """Test batch storage handles errors gracefully."""
        features_list = [_make_features()]
        predictions_list = [_make_prediction()]
        available_store._index.upsert.side_effect = RuntimeError("batch fail")

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            count = available_store.store_batch(features_list, predictions_list)

        assert count == 0


# =============================================================================
# delete_namespace Tests
# =============================================================================


class TestDeleteNamespace:
    """Tests for delete_namespace method."""

    def test_delete_namespace_success(self, available_store: "PineconeVectorStore") -> None:
        """Test successful namespace deletion."""
        result = available_store.delete_namespace()

        assert result is True
        available_store._index.delete.assert_called_once_with(
            delete_all=True, namespace="meta_controller"
        )

    def test_delete_namespace_when_unavailable(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test returns False when unavailable."""
        result = unavailable_store.delete_namespace()
        assert result is False

    def test_delete_namespace_handles_error(self, available_store: "PineconeVectorStore") -> None:
        """Test handles errors gracefully."""
        available_store._index.delete.side_effect = RuntimeError("delete fail")
        result = available_store.delete_namespace()
        assert result is False


# =============================================================================
# get_stats Tests
# =============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_when_available(self, available_store: "PineconeVectorStore") -> None:
        """Test stats retrieval when available."""
        stats = available_store.get_stats()

        assert stats["available"] is True
        assert stats["total_vectors"] == 100
        assert stats["dimension"] == 10
        assert stats["buffered_operations"] == 0

    def test_get_stats_when_unavailable(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test stats when unavailable."""
        stats = unavailable_store.get_stats()

        assert stats["available"] is False
        assert stats["buffered_operations"] == 0

    def test_get_stats_handles_error(self, available_store: "PineconeVectorStore") -> None:
        """Test stats handles errors gracefully."""
        available_store._index.describe_index_stats.side_effect = RuntimeError("stats fail")
        stats = available_store.get_stats()

        assert stats["available"] is True
        assert "error" in stats

    def test_get_stats_includes_buffered_count(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test stats includes buffered operations count."""
        unavailable_store._operation_buffer = [{"type": "test"}]
        stats = unavailable_store.get_stats()
        assert stats["buffered_operations"] == 1


# =============================================================================
# Buffer Operations Tests
# =============================================================================


class TestBufferOperations:
    """Tests for buffer management methods."""

    def test_get_buffered_operations_returns_copy(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test get_buffered_operations returns a copy."""
        unavailable_store._operation_buffer = [{"type": "test"}]
        buf = unavailable_store.get_buffered_operations()

        assert buf == [{"type": "test"}]
        buf.append({"type": "new"})
        assert len(unavailable_store._operation_buffer) == 1

    def test_clear_buffer(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test clear_buffer empties the buffer."""
        unavailable_store._operation_buffer = [{"type": "test"}]
        unavailable_store.clear_buffer()
        assert len(unavailable_store._operation_buffer) == 0

    def test_flush_buffer_when_unavailable(self, unavailable_store: "PineconeVectorStore") -> None:
        """Test flush_buffer returns 0 when unavailable."""
        unavailable_store._operation_buffer = [{"type": "test"}]
        result = unavailable_store.flush_buffer()
        assert result == 0

    def test_flush_buffer_empty(self, available_store: "PineconeVectorStore") -> None:
        """Test flush_buffer with empty buffer."""
        result = available_store.flush_buffer()
        assert result == 0

    def test_flush_buffer_processes_store_prediction(self, available_store: "PineconeVectorStore") -> None:
        """Test flush_buffer processes buffered store_prediction operations."""
        features = _make_features()
        prediction = _make_prediction()
        available_store._operation_buffer = [
            {
                "type": "store_prediction",
                "features": features,
                "prediction": prediction,
                "metadata": None,
                "timestamp": "2024-01-01T00:00:00",
            }
        ]

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            flushed = available_store.flush_buffer()

        assert flushed == 1
        assert len(available_store._operation_buffer) == 0

    def test_flush_buffer_processes_store_batch(self, available_store: "PineconeVectorStore") -> None:
        """Test flush_buffer processes buffered store_batch operations."""
        features = _make_features()
        prediction = _make_prediction()
        available_store._operation_buffer = [
            {
                "type": "store_batch",
                "features_list": [features],
                "predictions_list": [prediction],
                "batch_metadata": None,
                "timestamp": "2024-01-01T00:00:00",
            }
        ]

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            flushed = available_store.flush_buffer()

        assert flushed == 1
        assert len(available_store._operation_buffer) == 0

    def test_flush_buffer_keeps_failed_operations(self, available_store: "PineconeVectorStore") -> None:
        """Test failed flush operations remain in buffer."""
        features = _make_features()
        prediction = _make_prediction()
        available_store._index.upsert.side_effect = ConnectionError("fail")
        available_store._operation_buffer = [
            {
                "type": "store_prediction",
                "features": features,
                "prediction": prediction,
                "metadata": None,
                "timestamp": "2024-01-01T00:00:00",
            }
        ]

        with patch("src.storage.pinecone_store.normalize_features", return_value=[0.1] * 10):
            flushed = available_store.flush_buffer()

        assert flushed == 0
        # The operation returns None (not raising), so it goes to remaining_buffer
        assert len(available_store._operation_buffer) == 1
