"""Extended unit tests for src/storage/s3_client.py targeting uncovered lines.

Covers:
- _put_object_with_retry (auto-initialize, metadata, response)
- _get_object_with_retry (auto-initialize, body streaming)
- store_config with session_id
- store_mcts_stats with/without iteration, compression
- store_traces
- store_logs with compression
- store_checkpoint
- list_objects
- delete_object
- health_check (healthy and unhealthy paths)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.storage.s3_client import S3Config, S3StorageClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s3_mock(response: dict | None = None, body_data: bytes = b""):
    """Build a mock S3 client that works with ``async with session.client(...) as s3``."""
    s3 = AsyncMock()

    # put_object
    s3.put_object = AsyncMock(return_value=response or {"ETag": '"abc123"', "VersionId": "v1"})

    # get_object - needs an async body stream
    stream = AsyncMock()
    stream.read = AsyncMock(return_value=body_data)
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=False)
    s3.get_object = AsyncMock(return_value={"Body": stream})

    # list_objects_v2
    s3.list_objects_v2 = AsyncMock(return_value={
        "Contents": [
            {"Key": "test/key1", "Size": 100, "LastModified": "2024-01-01", "ETag": '"e1"'},
            {"Key": "test/key2", "Size": 200, "LastModified": "2024-01-02", "ETag": '"e2"'},
        ]
    })

    # delete_object
    s3.delete_object = AsyncMock(return_value={"VersionId": "v2"})

    # head_bucket
    s3.head_bucket = AsyncMock(return_value={})

    return s3


def _make_session(s3_mock):
    """Return a mock session whose ``.client(...)`` is an async context manager."""
    session = MagicMock()

    @asynccontextmanager
    async def _client(*args, **kwargs):
        yield s3_mock

    session.client = _client
    return session


def _small_config() -> S3Config:
    """Config with compression disabled for simpler assertions."""
    return S3Config(
        bucket_name="test-bucket",
        region_name="us-east-1",
        enable_compression=False,
        use_content_hash_keys=False,
    )


def _compress_config() -> S3Config:
    """Config with compression enabled and a very low threshold."""
    return S3Config(
        bucket_name="test-bucket",
        region_name="us-east-1",
        enable_compression=True,
        compression_threshold_bytes=1,  # compress everything
        use_content_hash_keys=False,
    )


# ---------------------------------------------------------------------------
# Tests: _put_object_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPutObjectWithRetry:
    """Tests for the internal _put_object_with_retry method."""

    @pytest.mark.asyncio
    async def test_put_object_auto_initializes(self):
        """When _session is None, initialize() is called automatically."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        assert client._session is None

        with patch("src.storage.s3_client.aioboto3") as mock_boto:
            mock_boto.Session.return_value = session
            result = await client._put_object_with_retry("test/key", b"data")

        assert result["key"] == "test/key"
        assert result["etag"] == '"abc123"'
        assert result["size_bytes"] == 4
        assert result["version_id"] == "v1"

    @pytest.mark.asyncio
    async def test_put_object_with_metadata(self):
        """Metadata dict is forwarded to S3."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        await client._put_object_with_retry(
            "test/key", b"data", metadata={"foo": "bar"},
        )

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Metadata"] == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_put_object_without_metadata(self):
        """When metadata is None, Metadata key is not sent."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        await client._put_object_with_retry("test/key", b"data")

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert "Metadata" not in call_kwargs


# ---------------------------------------------------------------------------
# Tests: _get_object_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetObjectWithRetry:
    """Tests for the internal _get_object_with_retry method."""

    @pytest.mark.asyncio
    async def test_get_object_auto_initializes(self):
        """When _session is None, initialize() is called automatically."""
        s3_mock = _make_s3_mock(body_data=b"hello world")
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        assert client._session is None

        with patch("src.storage.s3_client.aioboto3") as mock_boto:
            mock_boto.Session.return_value = session
            data = await client._get_object_with_retry("test/key")

        assert data == b"hello world"

    @pytest.mark.asyncio
    async def test_get_object_returns_bytes(self):
        """Downloaded data is returned as bytes."""
        s3_mock = _make_s3_mock(body_data=b"payload")
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        data = await client._get_object_with_retry("some/key")
        assert isinstance(data, bytes)
        assert data == b"payload"


# ---------------------------------------------------------------------------
# Tests: store_config with session_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoreConfigSessionId:
    """Tests for store_config with session_id parameter."""

    @pytest.mark.asyncio
    async def test_store_config_with_session_id(self):
        """session_id is added to metadata when provided."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.store_config("test_cfg", {"key": "val"}, session_id="sess-123")

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Metadata"]["session_id"] == "sess-123"
        assert "key" in result

    @pytest.mark.asyncio
    async def test_store_config_without_session_id(self):
        """No session_id metadata when not provided."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        await client.store_config("test_cfg", {"key": "val"})

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert "session_id" not in call_kwargs["Metadata"]


# ---------------------------------------------------------------------------
# Tests: store_mcts_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoreMctsStats:
    """Tests for store_mcts_stats including iteration and compression branches."""

    @pytest.mark.asyncio
    async def test_store_mcts_stats_without_iteration(self):
        """Stats stored without iteration number."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.store_mcts_stats("sess-1", {"nodes": 100})

        assert "key" in result
        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert "iteration" not in call_kwargs["Metadata"]
        # Key should not contain "iter"
        assert "_iter" not in call_kwargs["Key"]

    @pytest.mark.asyncio
    async def test_store_mcts_stats_with_iteration(self):
        """Stats stored with iteration number appended to name and metadata."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.store_mcts_stats("sess-1", {"nodes": 100}, iteration=5)

        assert "key" in result
        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Metadata"]["iteration"] == "5"
        assert "_iter5" in call_kwargs["Key"]

    @pytest.mark.asyncio
    async def test_store_mcts_stats_with_compression(self):
        """Stats are compressed when above threshold."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_compress_config())
        client._session = session
        client._initialized = True

        big_stats = {"data": "x" * 2000}
        result = await client.store_mcts_stats("sess-1", big_stats)

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Key"].endswith(".gz")
        assert call_kwargs["ContentType"] == "application/gzip"
        assert "key" in result


# ---------------------------------------------------------------------------
# Tests: store_traces
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoreTraces:
    """Tests for store_traces method."""

    @pytest.mark.asyncio
    async def test_store_traces_dict(self):
        """Store a dict of trace data."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.store_traces("sess-1", {"spans": [{"id": "s1"}]})

        assert "key" in result
        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Metadata"]["data_type"] == "traces"
        assert call_kwargs["Metadata"]["session_id"] == "sess-1"
        assert call_kwargs["ContentType"] == "application/json"

    @pytest.mark.asyncio
    async def test_store_traces_list(self):
        """Store a list of trace dicts."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        traces = [{"span_id": "a"}, {"span_id": "b"}]
        result = await client.store_traces("sess-2", traces)

        assert "key" in result

    @pytest.mark.asyncio
    async def test_store_traces_with_compression(self):
        """Traces are compressed when above threshold."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_compress_config())
        client._session = session
        client._initialized = True

        big_traces = {"data": "y" * 2000}
        result = await client.store_traces("sess-1", big_traces)

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Key"].endswith(".gz")
        assert call_kwargs["ContentType"] == "application/gzip"


# ---------------------------------------------------------------------------
# Tests: store_logs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoreLogs:
    """Tests for store_logs method."""

    @pytest.mark.asyncio
    async def test_store_logs_no_compression(self):
        """Logs stored as NDJSON without compression."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        entries = [{"level": "INFO", "msg": "ok"}]
        result = await client.store_logs("sess-1", entries)

        assert "key" in result
        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["ContentType"] == "application/x-ndjson"
        assert call_kwargs["Key"].endswith(".ndjson")
        assert call_kwargs["Metadata"]["entry_count"] == "1"

    @pytest.mark.asyncio
    async def test_store_logs_with_compression(self):
        """Logs are compressed when above threshold."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_compress_config())
        client._session = session
        client._initialized = True

        entries = [{"level": "INFO", "msg": "x" * 500} for _ in range(10)]
        result = await client.store_logs("sess-1", entries)

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Key"].endswith(".gz")
        assert call_kwargs["ContentType"] == "application/gzip"


# ---------------------------------------------------------------------------
# Tests: store_checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoreCheckpoint:
    """Tests for store_checkpoint method."""

    @pytest.mark.asyncio
    async def test_store_checkpoint_default_name(self):
        """Checkpoint stored with default checkpoint_name."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.store_checkpoint("sess-1", {"state": "running"})

        assert "key" in result
        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Metadata"]["checkpoint_name"] == "checkpoint"
        assert call_kwargs["Metadata"]["data_type"] == "checkpoint"
        assert "sess-1_checkpoint" in call_kwargs["Key"]

    @pytest.mark.asyncio
    async def test_store_checkpoint_custom_name(self):
        """Checkpoint stored with custom checkpoint_name."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.store_checkpoint("sess-1", {"state": "done"}, checkpoint_name="final")

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Metadata"]["checkpoint_name"] == "final"
        assert "sess-1_final" in call_kwargs["Key"]

    @pytest.mark.asyncio
    async def test_store_checkpoint_with_compression(self):
        """Checkpoint is compressed when above threshold."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_compress_config())
        client._session = session
        client._initialized = True

        big_data = {"state": "x" * 2000}
        result = await client.store_checkpoint("sess-1", big_data)

        call_kwargs = s3_mock.put_object.call_args.kwargs
        assert call_kwargs["Key"].endswith(".gz")
        assert call_kwargs["ContentType"] == "application/gzip"


# ---------------------------------------------------------------------------
# Tests: retrieve_object / retrieve_json
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrieveObject:
    """Tests for retrieve_object and retrieve_json."""

    @pytest.mark.asyncio
    async def test_retrieve_object_plain(self):
        """Retrieve non-compressed object."""
        s3_mock = _make_s3_mock(body_data=b"plain data")
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        data = await client.retrieve_object("test/key.json")
        assert data == b"plain data"

    @pytest.mark.asyncio
    async def test_retrieve_object_gzipped(self):
        """Retrieve gzip-compressed object (auto-decompressed by key suffix)."""
        import gzip
        original = b"compressed data"
        compressed = gzip.compress(original)

        s3_mock = _make_s3_mock(body_data=compressed)
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        data = await client.retrieve_object("test/key.gz")
        assert data == original

    @pytest.mark.asyncio
    async def test_retrieve_json(self):
        """Retrieve and parse JSON object."""
        payload = {"hello": "world", "count": 42}
        s3_mock = _make_s3_mock(body_data=json.dumps(payload).encode())
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.retrieve_json("test/key.json")
        assert result == payload


# ---------------------------------------------------------------------------
# Tests: list_objects
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListObjects:
    """Tests for list_objects method."""

    @pytest.mark.asyncio
    async def test_list_objects(self):
        """List objects returns structured metadata."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        objects = await client.list_objects("test/")

        assert len(objects) == 2
        assert objects[0]["key"] == "test/key1"
        assert objects[0]["size"] == 100
        assert objects[1]["key"] == "test/key2"

    @pytest.mark.asyncio
    async def test_list_objects_auto_initializes(self):
        """list_objects initializes session if not done."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        assert client._session is None

        with patch("src.storage.s3_client.aioboto3") as mock_boto:
            mock_boto.Session.return_value = session
            objects = await client.list_objects("prefix/")

        assert len(objects) == 2

    @pytest.mark.asyncio
    async def test_list_objects_empty(self):
        """list_objects returns empty list when no contents."""
        s3_mock = _make_s3_mock()
        s3_mock.list_objects_v2 = AsyncMock(return_value={})
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        objects = await client.list_objects("empty/")
        assert objects == []


# ---------------------------------------------------------------------------
# Tests: delete_object
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteObject:
    """Tests for delete_object method."""

    @pytest.mark.asyncio
    async def test_delete_object(self):
        """delete_object returns expected structure."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.delete_object("test/key")

        assert result["key"] == "test/key"
        assert result["deleted"] is True
        assert result["version_id"] == "v2"

    @pytest.mark.asyncio
    async def test_delete_object_auto_initializes(self):
        """delete_object initializes session if not done."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        assert client._session is None

        with patch("src.storage.s3_client.aioboto3") as mock_boto:
            mock_boto.Session.return_value = session
            result = await client.delete_object("key/to/delete")

        assert result["deleted"] is True


# ---------------------------------------------------------------------------
# Tests: health_check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Healthy check returns status healthy."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.health_check()

        assert result["status"] == "healthy"
        assert result["bucket"] == "test-bucket"
        assert result["region"] == "us-east-1"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_health_check_auto_initializes(self):
        """health_check initializes session if not done."""
        s3_mock = _make_s3_mock()
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        assert client._session is None

        with patch("src.storage.s3_client.aioboto3") as mock_boto:
            mock_boto.Session.return_value = session
            result = await client.health_check()

        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Unhealthy check returns error details."""
        from botocore.exceptions import ClientError

        s3_mock = _make_s3_mock()
        s3_mock.head_bucket = AsyncMock(
            side_effect=ClientError(
                {"Error": {"Code": "403", "Message": "Forbidden"}},
                "HeadBucket",
            )
        )
        session = _make_session(s3_mock)

        client = S3StorageClient(_small_config())
        client._session = session
        client._initialized = True

        result = await client.health_check()

        assert result["status"] == "unhealthy"
        assert result["error_code"] == "403"
        assert result["bucket"] == "test-bucket"
        assert "timestamp" in result
