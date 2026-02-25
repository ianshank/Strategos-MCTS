"""
Async S3 storage client for multi-agent MCTS framework.

Provides:
- aioboto3 async client
- Retry strategy with tenacity
- Exponential backoff for failures
- Content-hash based idempotent keys
- Store: configs, MCTS stats, traces, logs
- Compression support
"""

import asyncio
import gzip
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aioboto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.observability.logging import get_structured_logger


@dataclass
class S3Config:
    """Configuration for S3 storage client."""

    bucket_name: str = field(default_factory=lambda: os.environ.get("S3_BUCKET_NAME", "mcts-framework-storage"))
    region_name: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    endpoint_url: str | None = field(default_factory=lambda: os.environ.get("S3_ENDPOINT_URL"))

    # Retry configuration
    max_retries: int = 5
    initial_wait_seconds: float = 1.0
    max_wait_seconds: float = 60.0
    exponential_base: float = 2.0

    # Storage options
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024  # Compress if larger than 1KB
    use_content_hash_keys: bool = True

    # Prefixes for different data types
    prefix_configs: str = "configs/"
    prefix_mcts_stats: str = "mcts-stats/"
    prefix_traces: str = "traces/"
    prefix_logs: str = "logs/"
    prefix_checkpoints: str = "checkpoints/"


class S3StorageClient:
    """
    Async S3 storage client with retry logic and compression.

    Features:
    - Automatic retries with exponential backoff
    - Content-hash based idempotent keys for deduplication
    - Gzip compression for large payloads
    - Organized storage by data type
    """

    def __init__(self, config: S3Config | None = None):
        """
        Initialize S3 storage client.

        Args:
            config: S3 configuration (uses environment variables if not provided)
        """
        self.config = config or S3Config()
        self.logger = get_structured_logger("storage.s3")
        self._session: Any = None
        self._initialized = False

        # boto3 config with retries and timeouts
        self._boto_config = BotoConfig(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
            max_pool_connections=25,
        )

    async def initialize(self) -> None:
        """Initialize the aioboto3 session."""
        if self._initialized:
            return

        self._session = aioboto3.Session()
        self._initialized = True
        self.logger.info("S3 client initialized", bucket_name=self.config.bucket_name)

    async def close(self) -> None:
        """Close the client (cleanup if needed)."""
        self._initialized = False
        self.logger.info("S3 client closed")

    def _get_client_params(self) -> dict[str, Any]:
        """Get parameters for S3 client context manager."""
        params = {
            "region_name": self.config.region_name,
            "config": self._boto_config,
        }
        if self.config.endpoint_url:
            params["endpoint_url"] = self.config.endpoint_url
        return params

    def _compute_content_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of content for idempotent keys."""
        return hashlib.sha256(data).hexdigest()

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        return gzip.decompress(data)

    def _should_compress(self, data: bytes) -> bool:
        """Determine if data should be compressed."""
        return self.config.enable_compression and len(data) >= self.config.compression_threshold_bytes

    def _generate_key(
        self,
        prefix: str,
        name: str,
        data: bytes | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """
        Generate S3 key for object.

        Args:
            prefix: Storage prefix (e.g., configs/, logs/)
            name: Object name
            data: Optional data for content-hash key generation
            timestamp: Optional timestamp for key

        Returns:
            Full S3 key
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        date_prefix = timestamp.strftime("%Y/%m/%d")

        if self.config.use_content_hash_keys and data:
            content_hash = self._compute_content_hash(data)[:12]
            return f"{prefix}{date_prefix}/{name}_{content_hash}"
        else:
            timestamp_str = timestamp.strftime("%H%M%S_%f")
            return f"{prefix}{date_prefix}/{name}_{timestamp_str}"

    @retry(
        retry=retry_if_exception_type((ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    async def _put_object_with_retry(
        self,
        key: str,
        body: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Put object to S3 with retry logic.

        Uses tenacity for exponential backoff retry strategy.
        """
        if not self._session:
            await self.initialize()

        async with self._session.client("s3", **self._get_client_params()) as s3:
            extra_args: dict[str, Any] = {
                "ContentType": content_type,
            }
            if metadata:
                extra_args["Metadata"] = metadata

            response = await s3.put_object(
                Bucket=self.config.bucket_name,
                Key=key,
                Body=body,
                **extra_args,
            )

            self.logger.debug(
                "Uploaded object to S3",
                s3_key=key,
                size_bytes=len(body),
                etag=response.get("ETag"),
            )

            return {
                "key": key,
                "etag": response.get("ETag"),
                "size_bytes": len(body),
                "version_id": response.get("VersionId"),
            }

    @retry(
        retry=retry_if_exception_type((ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    async def _get_object_with_retry(self, key: str) -> bytes:
        """
        Get object from S3 with retry logic.
        """
        if not self._session:
            await self.initialize()

        async with self._session.client("s3", **self._get_client_params()) as s3:
            response = await s3.get_object(
                Bucket=self.config.bucket_name,
                Key=key,
            )

            async with response["Body"] as stream:
                data = await stream.read()

            self.logger.debug(
                "Downloaded object from S3",
                s3_key=key,
                size_bytes=len(data),
            )

            return bytes(data)

    async def store_config(
        self,
        config_name: str,
        config_data: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Store configuration data to S3.

        Args:
            config_name: Name of the configuration
            config_data: Configuration dictionary
            session_id: Optional session identifier

        Returns:
            Upload result with key and metadata
        """
        json_data = json.dumps(config_data, indent=2, default=str).encode("utf-8")

        key = self._generate_key(
            prefix=self.config.prefix_configs,
            name=config_name,
            data=json_data if self.config.use_content_hash_keys else None,
        )

        if self._should_compress(json_data):
            body = self._compress_data(json_data)
            key += ".gz"
            content_type = "application/gzip"
        else:
            body = json_data
            content_type = "application/json"

        metadata = {
            "config_name": config_name,
            "original_size": str(len(json_data)),
            "compressed": str(len(body) != len(json_data)),
        }
        if session_id:
            metadata["session_id"] = session_id

        result = await self._put_object_with_retry(key, body, content_type, metadata)
        self.logger.info(
            "Stored config to S3",
            config_name=config_name,
            s3_key=key,
        )
        return dict(result)

    async def store_mcts_stats(
        self,
        session_id: str,
        stats: dict[str, Any],
        iteration: int | None = None,
    ) -> dict[str, Any]:
        """
        Store MCTS statistics to S3.

        Args:
            session_id: MCTS session identifier
            stats: Statistics dictionary
            iteration: Optional iteration number

        Returns:
            Upload result
        """
        name = f"{session_id}_stats"
        if iteration is not None:
            name += f"_iter{iteration}"

        json_data = json.dumps(stats, indent=2, default=str).encode("utf-8")

        key = self._generate_key(
            prefix=self.config.prefix_mcts_stats,
            name=name,
            data=json_data if self.config.use_content_hash_keys else None,
        )

        if self._should_compress(json_data):
            body = self._compress_data(json_data)
            key += ".gz"
            content_type = "application/gzip"
        else:
            body = json_data
            content_type = "application/json"

        metadata = {
            "session_id": session_id,
            "data_type": "mcts_stats",
            "original_size": str(len(json_data)),
        }
        if iteration is not None:
            metadata["iteration"] = str(iteration)

        result = await self._put_object_with_retry(key, body, content_type, metadata)
        self.logger.info(
            "Stored MCTS stats to S3",
            session_id=session_id,
            s3_key=key,
            iteration=iteration,
        )
        return dict(result)

    async def store_traces(
        self,
        session_id: str,
        trace_data: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Store trace data to S3.

        Args:
            session_id: Session identifier
            trace_data: Trace spans/events

        Returns:
            Upload result
        """
        json_data = json.dumps(trace_data, indent=2, default=str).encode("utf-8")

        key = self._generate_key(
            prefix=self.config.prefix_traces,
            name=f"{session_id}_traces",
            data=json_data if self.config.use_content_hash_keys else None,
        )

        if self._should_compress(json_data):
            body = self._compress_data(json_data)
            key += ".gz"
            content_type = "application/gzip"
        else:
            body = json_data
            content_type = "application/json"

        metadata = {
            "session_id": session_id,
            "data_type": "traces",
            "original_size": str(len(json_data)),
        }

        result = await self._put_object_with_retry(key, body, content_type, metadata)
        self.logger.info(
            "Stored traces to S3",
            session_id=session_id,
            s3_key=key,
        )
        return dict(result)

    async def store_logs(
        self,
        session_id: str,
        log_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Store log entries to S3.

        Args:
            session_id: Session identifier
            log_entries: List of JSON log entries

        Returns:
            Upload result
        """
        # Store as newline-delimited JSON (NDJSON)
        ndjson_data = "\n".join(json.dumps(entry, default=str) for entry in log_entries).encode("utf-8")

        key = self._generate_key(
            prefix=self.config.prefix_logs,
            name=f"{session_id}_logs",
            data=ndjson_data if self.config.use_content_hash_keys else None,
        )

        if self._should_compress(ndjson_data):
            body = self._compress_data(ndjson_data)
            key += ".gz"
            content_type = "application/gzip"
        else:
            body = ndjson_data
            key += ".ndjson"
            content_type = "application/x-ndjson"

        metadata = {
            "session_id": session_id,
            "data_type": "logs",
            "entry_count": str(len(log_entries)),
            "original_size": str(len(ndjson_data)),
        }

        result = await self._put_object_with_retry(key, body, content_type, metadata)
        self.logger.info(
            "Stored log entries to S3",
            session_id=session_id,
            entry_count=len(log_entries),
            s3_key=key,
        )
        return dict(result)

    async def store_checkpoint(
        self,
        session_id: str,
        checkpoint_data: dict[str, Any],
        checkpoint_name: str = "checkpoint",
    ) -> dict[str, Any]:
        """
        Store framework checkpoint/state to S3.

        Args:
            session_id: Session identifier
            checkpoint_data: Checkpoint state
            checkpoint_name: Name for the checkpoint

        Returns:
            Upload result
        """
        json_data = json.dumps(checkpoint_data, indent=2, default=str).encode("utf-8")

        key = self._generate_key(
            prefix=self.config.prefix_checkpoints,
            name=f"{session_id}_{checkpoint_name}",
            data=json_data if self.config.use_content_hash_keys else None,
        )

        if self._should_compress(json_data):
            body = self._compress_data(json_data)
            key += ".gz"
            content_type = "application/gzip"
        else:
            body = json_data
            content_type = "application/json"

        metadata = {
            "session_id": session_id,
            "data_type": "checkpoint",
            "checkpoint_name": checkpoint_name,
            "original_size": str(len(json_data)),
        }

        result = await self._put_object_with_retry(key, body, content_type, metadata)
        self.logger.info(
            "Stored checkpoint to S3",
            checkpoint_name=checkpoint_name,
            session_id=session_id,
            s3_key=key,
        )
        return dict(result)

    async def retrieve_object(self, key: str) -> bytes:
        """
        Retrieve and decompress object from S3.

        Args:
            key: S3 object key

        Returns:
            Decompressed data bytes
        """
        data = await self._get_object_with_retry(key)

        # Auto-decompress if gzip
        if key.endswith(".gz"):
            data = self._decompress_data(data)

        return bytes(data)

    async def retrieve_json(self, key: str) -> Any:
        """
        Retrieve JSON object from S3.

        Args:
            key: S3 object key

        Returns:
            Parsed JSON data
        """
        data = await self.retrieve_object(key)
        return json.loads(data.decode("utf-8"))

    async def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        List objects with given prefix.

        Args:
            prefix: S3 key prefix
            max_keys: Maximum objects to return

        Returns:
            List of object metadata
        """
        if not self._session:
            await self.initialize()

        async with self._session.client("s3", **self._get_client_params()) as s3:
            response = await s3.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys,
            )

            objects = []
            for obj in response.get("Contents", []):
                objects.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                        "etag": obj["ETag"],
                    }
                )

            return objects

    async def delete_object(self, key: str) -> dict[str, Any]:
        """
        Delete object from S3.

        Args:
            key: S3 object key

        Returns:
            Deletion result
        """
        if not self._session:
            await self.initialize()

        async with self._session.client("s3", **self._get_client_params()) as s3:
            response = await s3.delete_object(
                Bucket=self.config.bucket_name,
                Key=key,
            )

            self.logger.info("Deleted object from S3", s3_key=key)

            return {
                "key": key,
                "deleted": True,
                "version_id": response.get("VersionId"),
            }

    async def health_check(self) -> dict[str, Any]:
        """
        Check S3 connectivity and bucket access.

        Returns:
            Health check result
        """
        if not self._session:
            await self.initialize()

        try:
            async with self._session.client("s3", **self._get_client_params()) as s3:
                # Try to head the bucket
                await s3.head_bucket(Bucket=self.config.bucket_name)

            return {
                "status": "healthy",
                "bucket": self.config.bucket_name,
                "region": self.config.region_name,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            return {
                "status": "unhealthy",
                "bucket": self.config.bucket_name,
                "error_code": error_code,
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
