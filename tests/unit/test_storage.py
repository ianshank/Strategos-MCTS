"""
Comprehensive unit tests for storage modules.

Tests:
- S3StorageClient: Configuration, key generation, compression, retry logic
- PineconeVectorStore: Vector operations, namespace management, buffering
"""

import gzip
import hashlib
import json

# Import storage modules
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Skip entire module if torch is not available (meta_controller package requires it)
pytest.importorskip("torch", reason="PyTorch required for meta_controller module")

sys.path.insert(0, ".")

# Mock the observability module before importing storage modules
# This avoids import issues with opentelemetry instrumentation
sys.modules["src.observability"] = MagicMock()
sys.modules["src.observability.logging"] = MagicMock()
sys.modules["src.observability.tracing"] = MagicMock()

# Create a mock get_logger that returns a Mock logger
mock_logger_module = MagicMock()
mock_logger_module.get_logger = Mock(return_value=Mock())
sys.modules["src.observability.logging"] = mock_logger_module

from src.agents.meta_controller.base import MetaControllerFeatures, MetaControllerPrediction  # noqa: E402
from src.storage.pinecone_store import PineconeVectorStore  # noqa: E402
from src.storage.s3_client import S3Config, S3StorageClient  # noqa: E402


class TestS3Config:
    """Test suite for S3Config dataclass."""

    def test_default_configuration(self):
        """Test that default configuration values are set correctly."""
        with patch.dict("os.environ", {}, clear=True):
            config = S3Config()

            assert config.bucket_name == "mcts-framework-storage"
            assert config.region_name == "us-east-1"
            assert config.endpoint_url is None
            assert config.max_retries == 5
            assert config.initial_wait_seconds == 1.0
            assert config.max_wait_seconds == 60.0
            assert config.exponential_base == 2.0
            assert config.enable_compression is True
            assert config.compression_threshold_bytes == 1024
            assert config.use_content_hash_keys is True

    def test_prefix_defaults(self):
        """Test that storage prefixes have correct defaults."""
        config = S3Config()

        assert config.prefix_configs == "configs/"
        assert config.prefix_mcts_stats == "mcts-stats/"
        assert config.prefix_traces == "traces/"
        assert config.prefix_logs == "logs/"
        assert config.prefix_checkpoints == "checkpoints/"

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        env_vars = {
            "S3_BUCKET_NAME": "custom-bucket",
            "AWS_REGION": "eu-west-1",
            "S3_ENDPOINT_URL": "http://localhost:9000",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = S3Config()

            assert config.bucket_name == "custom-bucket"
            assert config.region_name == "eu-west-1"
            assert config.endpoint_url == "http://localhost:9000"

    def test_custom_retry_configuration(self):
        """Test custom retry configuration values."""
        config = S3Config(
            max_retries=10,
            initial_wait_seconds=2.0,
            max_wait_seconds=120.0,
            exponential_base=3.0,
        )

        assert config.max_retries == 10
        assert config.initial_wait_seconds == 2.0
        assert config.max_wait_seconds == 120.0
        assert config.exponential_base == 3.0

    def test_compression_disabled(self):
        """Test configuration with compression disabled."""
        config = S3Config(enable_compression=False)

        assert config.enable_compression is False

    def test_custom_compression_threshold(self):
        """Test custom compression threshold."""
        config = S3Config(compression_threshold_bytes=2048)

        assert config.compression_threshold_bytes == 2048

    def test_content_hash_keys_disabled(self):
        """Test configuration with content hash keys disabled."""
        config = S3Config(use_content_hash_keys=False)

        assert config.use_content_hash_keys is False

    def test_custom_prefixes(self):
        """Test custom storage prefixes."""
        config = S3Config(
            prefix_configs="custom-configs/",
            prefix_mcts_stats="custom-stats/",
            prefix_traces="custom-traces/",
            prefix_logs="custom-logs/",
        )

        assert config.prefix_configs == "custom-configs/"
        assert config.prefix_mcts_stats == "custom-stats/"
        assert config.prefix_traces == "custom-traces/"
        assert config.prefix_logs == "custom-logs/"


class TestS3StorageClient:
    """Test suite for S3StorageClient class."""

    def test_client_initialization_with_default_config(self):
        """Test client initialization with default configuration."""
        client = S3StorageClient()

        assert client.config is not None
        assert isinstance(client.config, S3Config)
        assert client._initialized is False
        assert client._session is None

    def test_client_initialization_with_custom_config(self):
        """Test client initialization with custom configuration."""
        custom_config = S3Config(bucket_name="test-bucket", region_name="ap-south-1")
        client = S3StorageClient(config=custom_config)

        assert client.config.bucket_name == "test-bucket"
        assert client.config.region_name == "ap-south-1"

    def test_compute_content_hash(self):
        """Test SHA256 content hashing."""
        client = S3StorageClient()

        data = b"test data for hashing"
        expected_hash = hashlib.sha256(data).hexdigest()
        actual_hash = client._compute_content_hash(data)

        assert actual_hash == expected_hash
        assert len(actual_hash) == 64  # SHA256 hex digest length

    def test_compress_data(self):
        """Test gzip compression."""
        client = S3StorageClient()

        original_data = b"test data " * 100  # Repetitive data compresses well
        compressed = client._compress_data(original_data)

        # Compressed data should be smaller
        assert len(compressed) < len(original_data)
        # Should be valid gzip
        decompressed = gzip.decompress(compressed)
        assert decompressed == original_data

    def test_decompress_data(self):
        """Test gzip decompression."""
        client = S3StorageClient()

        original_data = b"test data for decompression"
        compressed = gzip.compress(original_data)
        decompressed = client._decompress_data(compressed)

        assert decompressed == original_data

    def test_should_compress_enabled_above_threshold(self):
        """Test compression decision when enabled and above threshold."""
        config = S3Config(enable_compression=True, compression_threshold_bytes=100)
        client = S3StorageClient(config=config)

        large_data = b"x" * 150
        assert client._should_compress(large_data) is True

    def test_should_compress_enabled_below_threshold(self):
        """Test compression decision when enabled but below threshold."""
        config = S3Config(enable_compression=True, compression_threshold_bytes=100)
        client = S3StorageClient(config=config)

        small_data = b"x" * 50
        assert client._should_compress(small_data) is False

    def test_should_compress_disabled(self):
        """Test compression decision when disabled."""
        config = S3Config(enable_compression=False)
        client = S3StorageClient(config=config)

        large_data = b"x" * 10000
        assert client._should_compress(large_data) is False

    def test_generate_key_with_content_hash(self):
        """Test key generation with content hash."""
        config = S3Config(use_content_hash_keys=True)
        client = S3StorageClient(config=config)

        data = b"test content"
        timestamp = datetime(2024, 6, 15, 10, 30, 45)

        key = client._generate_key(
            prefix="configs/",
            name="test_config",
            data=data,
            timestamp=timestamp,
        )

        # Key should contain date prefix
        assert "2024/06/15" in key
        # Key should contain name
        assert "test_config" in key
        # Key should contain hash (first 12 chars)
        expected_hash = hashlib.sha256(data).hexdigest()[:12]
        assert expected_hash in key
        # Key should start with prefix
        assert key.startswith("configs/")

    def test_generate_key_without_content_hash(self):
        """Test key generation without content hash (timestamp-based)."""
        config = S3Config(use_content_hash_keys=False)
        client = S3StorageClient(config=config)

        timestamp = datetime(2024, 6, 15, 10, 30, 45, 123456)

        key = client._generate_key(
            prefix="logs/",
            name="session_log",
            data=b"some data",
            timestamp=timestamp,
        )

        # Key should contain timestamp format
        assert "2024/06/15" in key
        assert "session_log" in key
        # Should contain timestamp string (HHMMSS_microseconds)
        assert "103045" in key

    def test_generate_key_without_data_uses_timestamp(self):
        """Test key generation when no data provided uses timestamp."""
        config = S3Config(use_content_hash_keys=True)
        client = S3StorageClient(config=config)

        timestamp = datetime(2024, 6, 15, 10, 30, 45, 123456)

        key = client._generate_key(
            prefix="traces/",
            name="trace_data",
            data=None,  # No data
            timestamp=timestamp,
        )

        # Should fall back to timestamp-based key
        assert "2024/06/15" in key
        assert "trace_data" in key
        assert "103045" in key

    def test_generate_key_uses_current_time_when_no_timestamp(self):
        """Test key generation uses current time when no timestamp provided."""
        client = S3StorageClient()

        key = client._generate_key(
            prefix="checkpoints/",
            name="checkpoint",
            data=b"checkpoint data",
        )

        # Key should have date-based structure
        assert key.startswith("checkpoints/")
        assert "/" in key
        # Should have some timestamp or hash
        assert "checkpoint" in key

    def test_get_client_params_without_endpoint(self):
        """Test client parameters without endpoint URL."""
        config = S3Config(region_name="us-west-2", endpoint_url=None)
        client = S3StorageClient(config=config)

        params = client._get_client_params()

        assert params["region_name"] == "us-west-2"
        assert "config" in params
        assert "endpoint_url" not in params

    def test_get_client_params_with_endpoint(self):
        """Test client parameters with endpoint URL (for LocalStack/MinIO)."""
        config = S3Config(
            region_name="us-east-1",
            endpoint_url="http://localhost:4566",
        )
        client = S3StorageClient(config=config)

        params = client._get_client_params()

        assert params["region_name"] == "us-east-1"
        assert params["endpoint_url"] == "http://localhost:4566"
        assert "config" in params

    @pytest.mark.asyncio
    async def test_initialize_creates_session(self):
        """Test that initialize creates aioboto3 session."""
        with patch("src.storage.s3_client.aioboto3.Session") as mock_session:
            client = S3StorageClient()

            await client.initialize()

            mock_session.assert_called_once()
            assert client._initialized is True
            assert client._session is not None

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize is idempotent (only runs once)."""
        with patch("src.storage.s3_client.aioboto3.Session") as mock_session:
            client = S3StorageClient()

            await client.initialize()
            await client.initialize()  # Second call

            # Should only be called once
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_resets_initialized_flag(self):
        """Test that close resets initialized flag."""
        client = S3StorageClient()
        client._initialized = True

        await client.close()

        assert client._initialized is False


class TestPineconeVectorStore:
    """Test suite for PineconeVectorStore class."""

    def create_sample_features(self) -> MetaControllerFeatures:
        """Create sample features for testing."""
        return MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.75,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )

    def create_sample_prediction(self) -> MetaControllerPrediction:
        """Create sample prediction for testing."""
        return MetaControllerPrediction(
            agent="hrm",
            confidence=0.85,
            probabilities={"hrm": 0.85, "trm": 0.10, "mcts": 0.05},
        )

    def test_vector_dimension_constant(self):
        """Test that vector dimension constant is correct."""
        assert PineconeVectorStore.VECTOR_DIMENSION == 10

    def test_initialization_without_api_key(self):
        """Test initialization when API key is not provided."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True),
        ):
            store = PineconeVectorStore(auto_init=False)

            assert store._api_key is None
            assert store._host is None
            assert store.namespace == "meta_controller"
            assert store._is_initialized is False

    def test_initialization_with_custom_namespace(self):
        """Test initialization with custom namespace."""
        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            namespace="custom_namespace",
            auto_init=False,
        )

        assert store.namespace == "custom_namespace"

    def test_initialization_from_environment(self):
        """Test initialization from environment variables."""
        env_vars = {
            "PINECONE_API_KEY": "env-api-key",
            "PINECONE_HOST": "env-host-url",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            store = PineconeVectorStore(auto_init=False)

            assert store._api_key == "env-api-key"
            assert store._host == "env-host-url"

    def test_is_available_when_not_initialized(self):
        """Test is_available returns False when not initialized."""
        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=False,
        )

        # Not initialized yet
        assert store.is_available is False

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", False)
    def test_initialization_when_pinecone_not_available(self):
        """Test initialization when pinecone package is not installed."""
        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        assert store.is_available is False
        assert store._is_initialized is False

    def test_store_prediction_when_not_available_buffers(self):
        """Test that store_prediction buffers when Pinecone not available."""
        store = PineconeVectorStore(auto_init=False)
        store._is_initialized = False

        features = self.create_sample_features()
        prediction = self.create_sample_prediction()

        result = store.store_prediction(features, prediction, {"test": "metadata"})

        assert result is None
        assert len(store._operation_buffer) == 1
        assert store._operation_buffer[0]["type"] == "store_prediction"
        assert store._operation_buffer[0]["features"] is features
        assert store._operation_buffer[0]["prediction"] is prediction

    def test_find_similar_decisions_when_not_available_returns_empty(self):
        """Test that find_similar_decisions returns empty list when not available."""
        store = PineconeVectorStore(auto_init=False)
        features = self.create_sample_features()

        results = store.find_similar_decisions(features)

        assert results == []

    def test_get_agent_distribution_when_not_available(self):
        """Test get_agent_distribution returns zeros when not available."""
        store = PineconeVectorStore(auto_init=False)
        features = self.create_sample_features()

        distribution = store.get_agent_distribution(features)

        assert distribution == {"hrm": 0.0, "trm": 0.0, "mcts": 0.0}

    def test_store_batch_when_not_available_buffers(self):
        """Test that store_batch buffers when not available."""
        store = PineconeVectorStore(auto_init=False)

        features_list = [self.create_sample_features()]
        predictions_list = [self.create_sample_prediction()]

        count = store.store_batch(features_list, predictions_list)

        assert count == 0
        assert len(store._operation_buffer) == 1
        assert store._operation_buffer[0]["type"] == "store_batch"

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    def test_store_batch_validates_list_lengths(self, mock_pinecone_class):
        """Test that store_batch validates equal list lengths."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        features_list = [self.create_sample_features(), self.create_sample_features()]
        predictions_list = [self.create_sample_prediction()]  # Mismatched length

        with pytest.raises(ValueError, match="same length"):
            store.store_batch(features_list, predictions_list)

    def test_get_stats_when_not_available(self):
        """Test get_stats when Pinecone not available."""
        store = PineconeVectorStore(auto_init=False)
        store._operation_buffer = [{"test": "op1"}, {"test": "op2"}]

        stats = store.get_stats()

        assert stats["available"] is False
        assert stats["buffered_operations"] == 2

    def test_delete_namespace_when_not_available(self):
        """Test delete_namespace returns False when not available."""
        store = PineconeVectorStore(auto_init=False)

        result = store.delete_namespace()

        assert result is False

    def test_get_buffered_operations_returns_copy(self):
        """Test that get_buffered_operations returns a copy."""
        store = PineconeVectorStore(auto_init=False)
        store._operation_buffer = [{"op": 1}]

        buffer_copy = store.get_buffered_operations()

        # Should be a different list object
        assert buffer_copy is not store._operation_buffer
        assert buffer_copy == store._operation_buffer

    def test_clear_buffer(self):
        """Test clear_buffer empties the operation buffer."""
        store = PineconeVectorStore(auto_init=False)
        store._operation_buffer = [{"op": 1}, {"op": 2}]

        store.clear_buffer()

        assert store._operation_buffer == []

    def test_flush_buffer_when_not_available(self):
        """Test flush_buffer returns 0 when not available."""
        store = PineconeVectorStore(auto_init=False)
        store._operation_buffer = [{"op": 1}]

        flushed = store.flush_buffer()

        assert flushed == 0

    def test_flush_buffer_when_empty(self):
        """Test flush_buffer returns 0 when buffer is empty."""
        store = PineconeVectorStore(auto_init=False)
        store._operation_buffer = []

        flushed = store.flush_buffer()

        assert flushed == 0

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    def test_initialization_success(self, mock_pinecone_class):
        """Test successful Pinecone initialization."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        store = PineconeVectorStore(
            api_key="test-api-key",
            host="test-host-url",
            auto_init=True,
        )

        assert store._is_initialized is True
        assert store._client is mock_client
        assert store._index is mock_index
        mock_pinecone_class.assert_called_once_with(api_key="test-api-key")
        mock_client.Index.assert_called_once_with(host="test-host-url")

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    def test_initialization_failure_handled_gracefully(self, mock_pinecone_class):
        """Test that initialization failures are handled gracefully."""
        mock_pinecone_class.side_effect = ConnectionError("Connection error")

        store = PineconeVectorStore(
            api_key="test-api-key",
            host="test-host-url",
            auto_init=True,
        )

        assert store._is_initialized is False

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    @patch("src.storage.pinecone_store.normalize_features")
    def test_store_prediction_success(self, mock_normalize, mock_pinecone_class):
        """Test successful prediction storage."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        # Mock normalize_features to return a 10-dimensional vector
        mock_normalize.return_value = [0.8, 0.6, 0.75, 0.7, 1.0, 0.0, 0.0, 0.1, 0.015, 1.0]

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        features = self.create_sample_features()
        prediction = self.create_sample_prediction()

        result = store.store_prediction(features, prediction)

        assert result is not None
        # Check that upsert was called
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args
        assert call_args.kwargs["namespace"] == "meta_controller"
        assert len(call_args.kwargs["vectors"]) == 1

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    @patch("src.storage.pinecone_store.normalize_features")
    def test_find_similar_decisions_success(self, mock_normalize, mock_pinecone_class):
        """Test successful similarity search."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_normalize.return_value = [0.8, 0.6, 0.75, 0.7, 1.0, 0.0, 0.0, 0.1, 0.015, 1.0]

        # Mock query response
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "vec-1",
                    "score": 0.95,
                    "metadata": {"selected_agent": "hrm", "confidence": 0.9},
                },
                {
                    "id": "vec-2",
                    "score": 0.88,
                    "metadata": {"selected_agent": "trm", "confidence": 0.85},
                },
            ]
        }

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        features = self.create_sample_features()
        results = store.find_similar_decisions(features, top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == "vec-1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["selected_agent"] == "hrm"
        assert results[1]["id"] == "vec-2"
        assert results[1]["score"] == 0.88

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    @patch("src.storage.pinecone_store.normalize_features")
    def test_get_agent_distribution_calculates_correctly(self, mock_normalize, mock_pinecone_class):
        """Test agent distribution calculation from similar decisions."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_normalize.return_value = [0.8, 0.6, 0.75, 0.7, 1.0, 0.0, 0.0, 0.1, 0.015, 1.0]

        # Mock query response with mixed agents
        mock_index.query.return_value = {
            "matches": [
                {"id": "1", "score": 0.9, "metadata": {"selected_agent": "hrm"}},
                {"id": "2", "score": 0.85, "metadata": {"selected_agent": "hrm"}},
                {"id": "3", "score": 0.8, "metadata": {"selected_agent": "trm"}},
                {"id": "4", "score": 0.75, "metadata": {"selected_agent": "mcts"}},
            ]
        }

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        features = self.create_sample_features()
        distribution = store.get_agent_distribution(features, top_k=10)

        # 2 hrm, 1 trm, 1 mcts out of 4
        assert distribution["hrm"] == 0.5
        assert distribution["trm"] == 0.25
        assert distribution["mcts"] == 0.25

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    def test_get_stats_success(self, mock_pinecone_class):
        """Test getting index statistics."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_index.describe_index_stats.return_value = {
            "total_vector_count": 1000,
            "namespaces": {"meta_controller": {"vector_count": 500}},
            "dimension": 10,
        }

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        stats = store.get_stats()

        assert stats["available"] is True
        assert stats["total_vectors"] == 1000
        assert stats["dimension"] == 10
        assert "namespace_stats" in stats  # Actual key name in the implementation

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    def test_delete_namespace_success(self, mock_pinecone_class):
        """Test successful namespace deletion."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            namespace="test_namespace",
            auto_init=True,
        )

        result = store.delete_namespace()

        assert result is True
        mock_index.delete.assert_called_once_with(delete_all=True, namespace="test_namespace")

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    def test_delete_namespace_failure_handled(self, mock_pinecone_class):
        """Test delete namespace handles errors gracefully."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index
        mock_index.delete.side_effect = RuntimeError("Delete failed")

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        result = store.delete_namespace()

        assert result is False

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    @patch("src.storage.pinecone_store.normalize_features")
    def test_store_batch_success(self, mock_normalize, mock_pinecone_class):
        """Test successful batch storage."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_normalize.return_value = [0.8, 0.6, 0.75, 0.7, 1.0, 0.0, 0.0, 0.1, 0.015, 1.0]

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        features_list = [self.create_sample_features(), self.create_sample_features()]
        predictions_list = [self.create_sample_prediction(), self.create_sample_prediction()]

        count = store.store_batch(features_list, predictions_list)

        assert count == 2
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args
        assert len(call_args.kwargs["vectors"]) == 2


class TestS3StorageClientIntegration:
    """Integration-style tests for S3StorageClient (with mocked aioboto3)."""

    @pytest.mark.asyncio
    async def test_store_config_below_compression_threshold(self):
        """Test storing config below compression threshold."""
        config = S3Config(
            enable_compression=True,
            compression_threshold_bytes=10000,  # High threshold
        )
        client = S3StorageClient(config=config)

        # Mock the internal method
        with patch.object(client, "_put_object_with_retry", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = {
                "key": "test-key",
                "etag": '"abc123"',
                "size_bytes": 100,
            }

            result = await client.store_config(
                config_name="test_config",
                config_data={"setting": "value"},
            )

            assert result["key"] == "test-key"
            # Should not compress small data - args are (key, body, content_type, metadata)
            call_args = mock_put.call_args
            # Content type is the 3rd positional argument (index 2)
            assert call_args.args[2] == "application/json"

    @pytest.mark.asyncio
    async def test_store_config_above_compression_threshold(self):
        """Test storing config above compression threshold."""
        config = S3Config(
            enable_compression=True,
            compression_threshold_bytes=10,  # Very low threshold
        )
        client = S3StorageClient(config=config)

        with patch.object(client, "_put_object_with_retry", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = {
                "key": "test-key.gz",
                "etag": '"abc123"',
                "size_bytes": 50,
            }

            await client.store_config(
                config_name="test_config",
                config_data={"setting": "value", "data": "x" * 100},
            )

            # Should compress and use gzip content type
            call_args = mock_put.call_args
            # Content type is the 3rd positional argument (index 2)
            assert call_args.args[2] == "application/gzip"
            assert ".gz" in call_args.args[0]  # Key should have .gz extension

    @pytest.mark.asyncio
    async def test_store_mcts_stats_with_iteration(self):
        """Test storing MCTS stats with iteration number."""
        client = S3StorageClient()

        with patch.object(client, "_put_object_with_retry", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = {"key": "test-key", "etag": '"abc"', "size_bytes": 100}

            await client.store_mcts_stats(
                session_id="session123",
                stats={"nodes_expanded": 100, "best_value": 0.95},
                iteration=5,
            )

            call_args = mock_put.call_args
            # Metadata is the 4th positional argument (index 3)
            metadata = call_args.args[3]
            assert metadata["iteration"] == "5"
            assert metadata["session_id"] == "session123"

    @pytest.mark.asyncio
    async def test_store_logs_as_ndjson(self):
        """Test storing logs in NDJSON format."""
        config = S3Config(enable_compression=False)  # Disable compression for this test
        client = S3StorageClient(config=config)

        with patch.object(client, "_put_object_with_retry", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = {"key": "test.ndjson", "etag": '"abc"', "size_bytes": 100}

            log_entries = [
                {"level": "INFO", "message": "Test 1"},
                {"level": "ERROR", "message": "Test 2"},
            ]

            await client.store_logs(
                session_id="session123",
                log_entries=log_entries,
            )

            call_args = mock_put.call_args
            # Content type is the 3rd positional argument (index 2)
            assert call_args.args[2] == "application/x-ndjson"
            # Metadata is the 4th positional argument (index 3)
            assert call_args.args[3]["entry_count"] == "2"
            # Key should have .ndjson extension
            assert ".ndjson" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_retrieve_json_decompresses_gzip(self):
        """Test retrieving JSON automatically decompresses gzip."""
        client = S3StorageClient()

        original_data = {"test": "data", "number": 42}
        json_bytes = json.dumps(original_data).encode("utf-8")
        compressed_bytes = gzip.compress(json_bytes)

        with patch.object(client, "_get_object_with_retry", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = compressed_bytes

            result = await client.retrieve_json("configs/test.gz")

            assert result == original_data

    @pytest.mark.asyncio
    async def test_retrieve_object_no_decompression_for_non_gz(self):
        """Test retrieving object does not decompress non-.gz files."""
        client = S3StorageClient()

        raw_data = b"raw bytes data"

        with patch.object(client, "_get_object_with_retry", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = raw_data

            result = await client.retrieve_object("data/file.bin")

            assert result == raw_data


class TestErrorHandling:
    """Test error handling in storage modules."""

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    @patch("src.storage.pinecone_store.normalize_features")
    def test_pinecone_store_prediction_handles_exception(self, mock_normalize, mock_pinecone_class):
        """Test that store_prediction handles exceptions gracefully."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_normalize.return_value = [0.8] * 10
        mock_index.upsert.side_effect = ConnectionError("Network error")

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        features = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.75,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )
        prediction = MetaControllerPrediction(
            agent="hrm",
            confidence=0.85,
            probabilities={"hrm": 0.85, "trm": 0.10, "mcts": 0.05},
        )

        result = store.store_prediction(features, prediction)

        # Should return None on error
        assert result is None

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    @patch("src.storage.pinecone_store.normalize_features")
    def test_pinecone_find_similar_handles_exception(self, mock_normalize, mock_pinecone_class):
        """Test that find_similar_decisions handles exceptions gracefully."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_normalize.return_value = [0.8] * 10
        mock_index.query.side_effect = ConnectionError("Query failed")

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        features = MetaControllerFeatures(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_value=0.75,
            consensus_score=0.7,
            last_agent="hrm",
            iteration=2,
            query_length=150,
            has_rag_context=True,
        )

        results = store.find_similar_decisions(features)

        # Should return empty list on error
        assert results == []

    @patch("src.storage.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("src.storage.pinecone_store.Pinecone")
    def test_pinecone_get_stats_handles_exception(self, mock_pinecone_class):
        """Test that get_stats handles exceptions gracefully."""
        mock_client = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index
        mock_index.describe_index_stats.side_effect = RuntimeError("Stats error")

        store = PineconeVectorStore(
            api_key="test-key",
            host="test-host",
            auto_init=True,
        )

        stats = store.get_stats()

        assert stats["available"] is True
        assert "error" in stats
        assert "Stats error" in stats["error"]
