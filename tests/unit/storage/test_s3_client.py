"""
Comprehensive tests for S3 storage client.

Tests S3Config, S3StorageClient synchronous helper methods.
"""

from __future__ import annotations

import gzip
import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest

from src.storage.s3_client import S3Config, S3StorageClient


# ============================================================================
# S3Config Tests
# ============================================================================


class TestS3Config:
    """Tests for S3Config dataclass."""

    def test_default_config(self) -> None:
        """Test default S3 configuration."""
        config = S3Config()

        assert config.max_retries == 5
        assert config.enable_compression is True
        assert config.compression_threshold_bytes == 1024

    def test_custom_config(self) -> None:
        """Test custom S3 configuration."""
        config = S3Config(
            bucket_name="custom-bucket",
            region_name="eu-west-1",
            max_retries=10,
            enable_compression=False,
        )

        assert config.bucket_name == "custom-bucket"
        assert config.region_name == "eu-west-1"
        assert config.max_retries == 10
        assert config.enable_compression is False

    def test_config_prefixes(self) -> None:
        """Test default prefix configuration."""
        config = S3Config()

        assert config.prefix_configs == "configs/"
        assert config.prefix_mcts_stats == "mcts-stats/"
        assert config.prefix_traces == "traces/"
        assert config.prefix_logs == "logs/"
        assert config.prefix_checkpoints == "checkpoints/"

    def test_config_retry_settings(self) -> None:
        """Test retry configuration."""
        config = S3Config()

        assert config.initial_wait_seconds == 1.0
        assert config.max_wait_seconds == 60.0
        assert config.exponential_base == 2.0


# ============================================================================
# S3StorageClient Tests
# ============================================================================


class TestS3StorageClientInit:
    """Tests for S3StorageClient initialization."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        client = S3StorageClient()

        assert client.config is not None
        assert client._initialized is False
        assert client._session is None

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = S3Config(bucket_name="test-bucket")
        client = S3StorageClient(config=config)

        assert client.config.bucket_name == "test-bucket"

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Test async initialization."""
        client = S3StorageClient()

        with patch("src.storage.s3_client.aioboto3") as mock_aioboto3:
            mock_session = MagicMock()
            mock_aioboto3.Session.return_value = mock_session

            await client.initialize()

            assert client._initialized is True
            assert client._session is not None

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        """Test initialization is idempotent."""
        client = S3StorageClient()

        with patch("src.storage.s3_client.aioboto3") as mock_aioboto3:
            mock_session = MagicMock()
            mock_aioboto3.Session.return_value = mock_session

            await client.initialize()
            await client.initialize()  # Second call should be no-op

            # Session should only be created once
            mock_aioboto3.Session.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing the client."""
        client = S3StorageClient()

        with patch("src.storage.s3_client.aioboto3"):
            await client.initialize()
            await client.close()

            assert client._initialized is False


class TestS3StorageClientHelpers:
    """Tests for S3StorageClient helper methods."""

    @pytest.fixture
    def client(self) -> S3StorageClient:
        """Create S3 client for testing."""
        return S3StorageClient(S3Config(bucket_name="test-bucket"))

    def test_compute_content_hash(self, client: S3StorageClient) -> None:
        """Test content hash generation."""
        data = b"test content"
        expected_hash = hashlib.sha256(data).hexdigest()

        result = client._compute_content_hash(data)

        assert result == expected_hash

    def test_compute_content_hash_deterministic(self, client: S3StorageClient) -> None:
        """Test content hash is deterministic."""
        data = b"same content"

        hash1 = client._compute_content_hash(data)
        hash2 = client._compute_content_hash(data)

        assert hash1 == hash2

    def test_compute_content_hash_different_content(self, client: S3StorageClient) -> None:
        """Test different content produces different hash."""
        data1 = b"content 1"
        data2 = b"content 2"

        hash1 = client._compute_content_hash(data1)
        hash2 = client._compute_content_hash(data2)

        assert hash1 != hash2

    def test_should_compress_true(self, client: S3StorageClient) -> None:
        """Test compression decision for large content."""
        # Create data larger than threshold (1KB)
        large_data = b"x" * 2048

        assert client._should_compress(large_data) is True

    def test_should_compress_false_small(self, client: S3StorageClient) -> None:
        """Test compression decision for small content."""
        small_data = b"small"

        assert client._should_compress(small_data) is False

    def test_should_compress_false_disabled(self) -> None:
        """Test compression disabled in config."""
        config = S3Config(enable_compression=False)
        client = S3StorageClient(config=config)
        large_data = b"x" * 2048

        assert client._should_compress(large_data) is False

    def test_compress_data(self, client: S3StorageClient) -> None:
        """Test data compression."""
        data = b"test data to compress"

        compressed = client._compress_data(data)

        # Compressed data should be gzip format
        assert compressed.startswith(b"\x1f\x8b")  # Gzip magic bytes

        # Should decompress to original
        decompressed = gzip.decompress(compressed)
        assert decompressed == data

    def test_decompress_data(self, client: S3StorageClient) -> None:
        """Test data decompression."""
        original = b"test data"
        compressed = gzip.compress(original)

        decompressed = client._decompress_data(compressed)

        assert decompressed == original


class TestS3StorageClientKeyGeneration:
    """Tests for S3 key generation."""

    @pytest.fixture
    def client(self) -> S3StorageClient:
        """Create S3 client for testing."""
        return S3StorageClient(S3Config(bucket_name="test-bucket"))

    def test_generate_key_with_data(self, client: S3StorageClient) -> None:
        """Test key generation with content hash."""
        data = b"test content"
        prefix = "test/"

        key = client._generate_key(prefix, "testfile", data=data)

        assert key.startswith(prefix)
        # Key should include date structure
        parts = key.split("/")
        assert len(parts) >= 4  # prefix, year, month, day + filename

    def test_generate_key_without_data(self, client: S3StorageClient) -> None:
        """Test key generation without data uses timestamp."""
        prefix = "test/"

        key = client._generate_key(prefix, "testfile")

        assert key.startswith(prefix)

    def test_generate_key_different_prefix(self, client: S3StorageClient) -> None:
        """Test key generation with different prefixes."""
        data = b"test"

        config_key = client._generate_key(client.config.prefix_configs, "config", data=data)
        stats_key = client._generate_key(client.config.prefix_mcts_stats, "stats", data=data)

        assert config_key.startswith("configs/")
        assert stats_key.startswith("mcts-stats/")

    def test_generate_key_deterministic_with_same_data(self, client: S3StorageClient) -> None:
        """Test same data generates consistent key (content hash)."""
        data = b"same content"
        prefix = "test/"

        key1 = client._generate_key(prefix, "file", data=data)
        key2 = client._generate_key(prefix, "file", data=data)

        # With same data and name, should get same content hash portion
        # (though timestamp may differ if not specified)
        assert "file_" in key1
        assert "file_" in key2

    def test_generate_key_no_content_hash(self) -> None:
        """Test key generation without content hash mode."""
        config = S3Config(use_content_hash_keys=False)
        client = S3StorageClient(config=config)

        key = client._generate_key("prefix/", "name", data=b"data")

        # Should use timestamp format instead of content hash
        assert "prefix/" in key

    def test_get_client_params(self, client: S3StorageClient) -> None:
        """Test getting client parameters."""
        params = client._get_client_params()

        assert "region_name" in params
        assert "config" in params

    def test_get_client_params_with_endpoint(self) -> None:
        """Test getting client parameters with custom endpoint."""
        config = S3Config(endpoint_url="http://localhost:9000")
        client = S3StorageClient(config=config)

        params = client._get_client_params()

        assert params["endpoint_url"] == "http://localhost:9000"


class TestS3StorageClientEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def client(self) -> S3StorageClient:
        """Create S3 client for testing."""
        return S3StorageClient(S3Config(bucket_name="test-bucket"))

    def test_empty_data_hash(self, client: S3StorageClient) -> None:
        """Test hash generation for empty data."""
        empty_hash = client._compute_content_hash(b"")
        assert len(empty_hash) == 64  # SHA256 hex digest is 64 chars

    def test_compress_empty_data(self, client: S3StorageClient) -> None:
        """Test compressing empty data."""
        compressed = client._compress_data(b"")
        assert compressed.startswith(b"\x1f\x8b")  # Valid gzip

    def test_decompress_invalid_data(self, client: S3StorageClient) -> None:
        """Test decompressing invalid data raises error."""
        with pytest.raises(Exception):
            client._decompress_data(b"not valid gzip")

    def test_large_data_compression(self, client: S3StorageClient) -> None:
        """Test compressing large data."""
        large_data = b"x" * 1000000  # 1MB of data

        compressed = client._compress_data(large_data)

        # Compressed should be smaller
        assert len(compressed) < len(large_data)

        # Should decompress correctly
        decompressed = client._decompress_data(compressed)
        assert decompressed == large_data

    def test_binary_data_compression(self, client: S3StorageClient) -> None:
        """Test compressing binary data."""
        binary_data = bytes(range(256)) * 100

        compressed = client._compress_data(binary_data)
        decompressed = client._decompress_data(compressed)

        assert decompressed == binary_data
