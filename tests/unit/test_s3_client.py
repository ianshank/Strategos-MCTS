"""Unit tests for src/storage/s3_client.py."""

from __future__ import annotations

import gzip
import hashlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.storage.s3_client import S3Config, S3StorageClient


@pytest.mark.unit
class TestS3Config:
    def test_defaults(self):
        with patch.dict("os.environ", {}, clear=False):
            config = S3Config()
            assert config.max_retries == 5
            assert config.enable_compression is True
            assert config.compression_threshold_bytes == 1024
            assert config.prefix_configs == "configs/"

    def test_custom(self):
        config = S3Config(
            bucket_name="my-bucket",
            region_name="eu-west-1",
            max_retries=3,
            enable_compression=False,
        )
        assert config.bucket_name == "my-bucket"
        assert config.region_name == "eu-west-1"
        assert config.enable_compression is False


@pytest.mark.unit
class TestS3StorageClient:
    def test_init_default(self):
        client = S3StorageClient()
        assert client._initialized is False
        assert client._session is None

    def test_init_custom_config(self):
        config = S3Config(bucket_name="test-bucket")
        client = S3StorageClient(config)
        assert client.config.bucket_name == "test-bucket"

    def test_compute_content_hash(self):
        client = S3StorageClient()
        data = b"test data"
        expected = hashlib.sha256(data).hexdigest()
        assert client._compute_content_hash(data) == expected

    def test_compress_decompress(self):
        client = S3StorageClient()
        data = b"test data" * 100
        compressed = client._compress_data(data)
        decompressed = client._decompress_data(compressed)
        assert decompressed == data
        assert len(compressed) < len(data)

    def test_should_compress_large(self):
        config = S3Config(enable_compression=True, compression_threshold_bytes=10)
        client = S3StorageClient(config)
        assert client._should_compress(b"x" * 20) is True

    def test_should_compress_small(self):
        config = S3Config(enable_compression=True, compression_threshold_bytes=100)
        client = S3StorageClient(config)
        assert client._should_compress(b"x" * 10) is False

    def test_should_compress_disabled(self):
        config = S3Config(enable_compression=False)
        client = S3StorageClient(config)
        assert client._should_compress(b"x" * 10000) is False

    def test_generate_key_without_hash(self):
        config = S3Config(use_content_hash_keys=False)
        client = S3StorageClient(config)
        ts = datetime(2024, 1, 15, 10, 30, 0)
        key = client._generate_key("configs/", "model_config", timestamp=ts)
        assert key.startswith("configs/2024/01/15/")
        assert "model_config" in key

    def test_generate_key_with_hash(self):
        config = S3Config(use_content_hash_keys=True)
        client = S3StorageClient(config)
        data = b"some data"
        ts = datetime(2024, 6, 1, 12, 0, 0)
        key = client._generate_key("logs/", "app_log", data=data, timestamp=ts)
        assert key.startswith("logs/2024/06/01/")
        content_hash = hashlib.sha256(data).hexdigest()[:12]
        assert content_hash in key

    def test_get_client_params_basic(self):
        config = S3Config(region_name="us-west-2", endpoint_url=None)
        client = S3StorageClient(config)
        params = client._get_client_params()
        assert params["region_name"] == "us-west-2"
        assert "endpoint_url" not in params

    def test_get_client_params_with_endpoint(self):
        config = S3Config(endpoint_url="http://localhost:4566")
        client = S3StorageClient(config)
        params = client._get_client_params()
        assert params["endpoint_url"] == "http://localhost:4566"

    @pytest.mark.asyncio
    async def test_initialize(self):
        client = S3StorageClient()
        with patch("src.storage.s3_client.aioboto3") as mock_boto:
            mock_boto.Session.return_value = MagicMock()
            await client.initialize()
            assert client._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        client = S3StorageClient()
        with patch("src.storage.s3_client.aioboto3") as mock_boto:
            mock_boto.Session.return_value = MagicMock()
            await client.initialize()
            await client.initialize()  # Should not reinitialize
            assert mock_boto.Session.call_count == 1

    @pytest.mark.asyncio
    async def test_close(self):
        client = S3StorageClient()
        client._initialized = True
        await client.close()
        assert client._initialized is False
