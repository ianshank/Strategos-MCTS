"""
Connection Pool Management for LLM Clients.

Provides configurable HTTP connection pooling with:
- Dynamic pool sizing
- Connection health monitoring
- Automatic retry with backoff
- Metrics collection
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False
    httpx = None

if TYPE_CHECKING:
    from httpx import AsyncClient


logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for HTTP connection pooling."""

    # Pool sizing
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0

    # Timeouts (seconds)
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    write_timeout: float = 60.0
    pool_timeout: float = 5.0

    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)

    # Health monitoring
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    enable_health_monitoring: bool = True

    # HTTP/2 support
    http2: bool = True

    def to_httpx_limits(self) -> Any:
        """Convert to httpx.Limits object."""
        if not _HAS_HTTPX:
            return None

        return httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry,
        )

    def to_httpx_timeout(self) -> Any:
        """Convert to httpx.Timeout object."""
        if not _HAS_HTTPX:
            return None

        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.write_timeout,
            pool=self.pool_timeout,
        )


@dataclass
class PoolMetrics:
    """Metrics for connection pool monitoring."""

    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    retries_total: int = 0

    # Timing (milliseconds)
    latency_samples: list[float] = field(default_factory=list)
    max_latency_samples: int = 1000

    # Connection stats
    connections_created: int = 0
    connections_reused: int = 0
    connections_closed: int = 0

    # Health
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    last_health_check: float = 0.0

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        retries: int = 0,
        connection_reused: bool = False,
    ) -> None:
        """Record request metrics."""
        self.requests_total += 1
        if success:
            self.requests_successful += 1
        else:
            self.requests_failed += 1

        self.retries_total += retries

        # Track latency with bounded list
        if len(self.latency_samples) >= self.max_latency_samples:
            self.latency_samples.pop(0)
        self.latency_samples.append(latency_ms)

        # Connection tracking
        if connection_reused:
            self.connections_reused += 1
        else:
            self.connections_created += 1

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics summary."""
        latency_stats = {}
        if self.latency_samples:
            sorted_latencies = sorted(self.latency_samples)
            n = len(sorted_latencies)
            latency_stats = {
                "min_ms": sorted_latencies[0],
                "max_ms": sorted_latencies[-1],
                "mean_ms": sum(sorted_latencies) / n,
                "p50_ms": sorted_latencies[n // 2],
                "p95_ms": sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1],
                "p99_ms": sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1],
            }

        return {
            "requests": {
                "total": self.requests_total,
                "successful": self.requests_successful,
                "failed": self.requests_failed,
                "success_rate": (self.requests_successful / self.requests_total if self.requests_total > 0 else 0.0),
            },
            "retries_total": self.retries_total,
            "latency": latency_stats,
            "connections": {
                "created": self.connections_created,
                "reused": self.connections_reused,
                "closed": self.connections_closed,
                "reuse_rate": (
                    self.connections_reused / (self.connections_created + self.connections_reused)
                    if (self.connections_created + self.connections_reused) > 0
                    else 0.0
                ),
            },
            "health": {
                "checks_passed": self.health_checks_passed,
                "checks_failed": self.health_checks_failed,
                "last_check": self.last_health_check,
            },
        }


class ConnectionPool:
    """
    Managed HTTP connection pool with health monitoring.

    Features:
    - Configurable pool sizing
    - Automatic connection reuse
    - Health check monitoring
    - Request metrics collection
    - Retry with exponential backoff
    """

    def __init__(self, config: ConnectionPoolConfig | None = None):
        """
        Initialize connection pool.

        Args:
            config: Pool configuration (uses defaults if not provided)
        """
        if not _HAS_HTTPX:
            raise ImportError("httpx is required for connection pooling. Install it with: pip install httpx")

        self.config = config or ConnectionPoolConfig()
        self._client: AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._metrics = PoolMetrics()
        self._closed = False
        self._health_task: asyncio.Task | None = None

    async def _get_client(self) -> AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        limits=self.config.to_httpx_limits(),
                        timeout=self.config.to_httpx_timeout(),
                        http2=self.config.http2,
                    )
                    logger.debug(
                        "Created HTTP connection pool with max_connections=%d",
                        self.config.max_connections,
                    )
        return self._client

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """
        Make an HTTP request with automatic retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Request headers
            json: JSON body
            data: Raw body data
            timeout: Request timeout override

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: On request failure after retries
        """
        client = await self._get_client()
        start_time = time.perf_counter()
        retries = 0
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    content=data,
                    timeout=timeout,
                )

                # Check for retryable status codes
                if response.status_code in self.config.retry_status_codes and attempt < self.config.max_retries:
                    retries += 1
                    backoff = self.config.retry_backoff_factor * (2**attempt)
                    logger.warning(
                        "Retrying request to %s (attempt %d/%d) after %.2fs - status %d",
                        url,
                        attempt + 1,
                        self.config.max_retries,
                        backoff,
                        response.status_code,
                    )
                    await asyncio.sleep(backoff)
                    continue

                latency_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.record_request(
                    success=response.is_success,
                    latency_ms=latency_ms,
                    retries=retries,
                    connection_reused=True,  # httpx manages reuse internally
                )

                return response

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    retries += 1
                    backoff = self.config.retry_backoff_factor * (2**attempt)
                    logger.warning(
                        "Retrying request to %s (attempt %d/%d) after %.2fs - %s",
                        url,
                        attempt + 1,
                        self.config.max_retries,
                        backoff,
                        type(e).__name__,
                    )
                    await asyncio.sleep(backoff)

        # Record failure after all retries exhausted
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._metrics.record_request(
            success=False,
            latency_ms=latency_ms,
            retries=retries,
            connection_reused=True,
        )

        if last_error is not None:
            raise last_error
        raise httpx.HTTPError(f"Request failed after {self.config.max_retries} retries")

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """HTTP GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """HTTP POST request."""
        return await self.request("POST", url, **kwargs)

    async def health_check(self, url: str) -> bool:
        """
        Perform health check on a URL.

        Args:
            url: Health check URL

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.get(url, timeout=self.config.health_check_timeout)
            healthy = response.status_code < 500
            self._metrics.last_health_check = time.time()

            if healthy:
                self._metrics.health_checks_passed += 1
            else:
                self._metrics.health_checks_failed += 1

            return healthy
        except Exception as e:
            logger.warning("Health check failed for %s: %s", url, e)
            self._metrics.health_checks_failed += 1
            self._metrics.last_health_check = time.time()
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return self._metrics.get_stats()

    async def close(self) -> None:
        """Close the connection pool."""
        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task

        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._metrics.connections_closed += 1
            logger.debug("Closed HTTP connection pool")

        self._closed = True

    async def __aenter__(self) -> ConnectionPool:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


def create_connection_pool(
    *,
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
    connect_timeout: float = 10.0,
    read_timeout: float = 60.0,
    http2: bool = True,
    **kwargs: Any,
) -> ConnectionPool | None:
    """
    Factory function to create a connection pool.

    Args:
        max_connections: Maximum total connections
        max_keepalive_connections: Maximum keepalive connections
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        http2: Enable HTTP/2 support
        **kwargs: Additional configuration options

    Returns:
        ConnectionPool instance, or None if httpx is not available
    """
    if not _HAS_HTTPX:
        logger.warning("httpx not available, connection pooling disabled")
        return None

    config = ConnectionPoolConfig(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        http2=http2,
        **kwargs,
    )

    return ConnectionPool(config)
