"""Unit tests for src/performance/connection_pool.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.performance.connection_pool import (
    ConnectionPool,
    ConnectionPoolConfig,
    PoolMetrics,
    create_connection_pool,
)


@pytest.mark.unit
class TestConnectionPoolConfig:
    """Tests for ConnectionPoolConfig dataclass."""

    def test_default_values(self):
        config = ConnectionPoolConfig()
        assert config.max_connections == 100
        assert config.max_keepalive_connections == 20
        assert config.keepalive_expiry == 30.0
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0
        assert config.write_timeout == 60.0
        assert config.pool_timeout == 5.0
        assert config.max_retries == 3
        assert config.retry_backoff_factor == 0.5
        assert config.retry_status_codes == (429, 500, 502, 503, 504)
        assert config.http2 is True

    def test_custom_values(self):
        config = ConnectionPoolConfig(max_connections=50, read_timeout=30.0, max_retries=5)
        assert config.max_connections == 50
        assert config.read_timeout == 30.0
        assert config.max_retries == 5

    def test_to_httpx_limits(self):
        config = ConnectionPoolConfig(max_connections=50, max_keepalive_connections=10, keepalive_expiry=15.0)
        limits = config.to_httpx_limits()
        assert isinstance(limits, httpx.Limits)

    def test_to_httpx_timeout(self):
        config = ConnectionPoolConfig(connect_timeout=5.0, read_timeout=30.0)
        timeout = config.to_httpx_timeout()
        assert isinstance(timeout, httpx.Timeout)


@pytest.mark.unit
class TestPoolMetrics:
    """Tests for PoolMetrics dataclass."""

    def test_default_values(self):
        metrics = PoolMetrics()
        assert metrics.requests_total == 0
        assert metrics.requests_successful == 0
        assert metrics.requests_failed == 0
        assert metrics.latency_samples == []

    def test_record_successful_request(self):
        metrics = PoolMetrics()
        metrics.record_request(success=True, latency_ms=100.0, retries=0, connection_reused=True)
        assert metrics.requests_total == 1
        assert metrics.requests_successful == 1
        assert metrics.requests_failed == 0
        assert metrics.retries_total == 0
        assert metrics.connections_reused == 1
        assert metrics.connections_created == 0
        assert metrics.latency_samples == [100.0]

    def test_record_failed_request(self):
        metrics = PoolMetrics()
        metrics.record_request(success=False, latency_ms=5000.0, retries=3, connection_reused=False)
        assert metrics.requests_total == 1
        assert metrics.requests_failed == 1
        assert metrics.retries_total == 3
        assert metrics.connections_created == 1

    def test_latency_samples_bounded(self):
        metrics = PoolMetrics(max_latency_samples=3)
        for i in range(5):
            metrics.record_request(success=True, latency_ms=float(i))
        assert len(metrics.latency_samples) == 3
        # Oldest samples should be removed
        assert metrics.latency_samples == [2.0, 3.0, 4.0]

    def test_get_stats_empty(self):
        metrics = PoolMetrics()
        stats = metrics.get_stats()
        assert stats["requests"]["total"] == 0
        assert stats["requests"]["success_rate"] == 0.0
        assert stats["latency"] == {}
        assert stats["connections"]["reuse_rate"] == 0.0

    def test_get_stats_with_data(self):
        metrics = PoolMetrics()
        metrics.record_request(success=True, latency_ms=100.0, connection_reused=True)
        metrics.record_request(success=True, latency_ms=200.0, connection_reused=False)
        metrics.record_request(success=False, latency_ms=5000.0, connection_reused=True)

        stats = metrics.get_stats()
        assert stats["requests"]["total"] == 3
        assert stats["requests"]["successful"] == 2
        assert stats["requests"]["failed"] == 1
        assert stats["requests"]["success_rate"] == pytest.approx(2 / 3)
        assert stats["latency"]["min_ms"] == 100.0
        assert stats["latency"]["max_ms"] == 5000.0
        assert stats["connections"]["reuse_rate"] == pytest.approx(2 / 3)


@pytest.mark.unit
class TestConnectionPool:
    """Tests for ConnectionPool class."""

    def test_init_default_config(self):
        pool = ConnectionPool()
        assert pool.config.max_connections == 100
        assert pool._client is None
        assert pool._closed is False

    def test_init_custom_config(self):
        config = ConnectionPoolConfig(max_connections=50)
        pool = ConnectionPool(config)
        assert pool.config.max_connections == 50

    async def test_get_client_creates_client(self):
        pool = ConnectionPool(ConnectionPoolConfig(http2=False))
        client = await pool._get_client()
        assert client is not None
        assert pool._client is not None
        await pool.close()

    async def test_get_client_reuses_client(self):
        pool = ConnectionPool(ConnectionPoolConfig(http2=False))
        client1 = await pool._get_client()
        client2 = await pool._get_client()
        assert client1 is client2
        await pool.close()

    async def test_close(self):
        pool = ConnectionPool(ConnectionPoolConfig(http2=False))
        await pool._get_client()
        await pool.close()
        assert pool._client is None
        assert pool._closed is True

    async def test_context_manager(self):
        async with ConnectionPool() as pool:
            assert pool._closed is False
        assert pool._closed is True

    async def test_request_success(self):
        pool = ConnectionPool()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        pool._client = mock_client

        response = await pool.request("GET", "http://example.com")
        assert response.status_code == 200
        assert pool._metrics.requests_successful == 1
        await pool.close()

    async def test_request_retry_on_status_code(self):
        pool = ConnectionPool(ConnectionPoolConfig(max_retries=2, retry_backoff_factor=0.0))

        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.is_success = False

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.is_success = True

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=[fail_response, success_response])
        pool._client = mock_client

        response = await pool.request("GET", "http://example.com")
        assert response.status_code == 200
        assert pool._metrics.retries_total == 1
        await pool.close()

    async def test_request_retry_on_connect_error(self):
        pool = ConnectionPool(ConnectionPoolConfig(max_retries=2, retry_backoff_factor=0.0))

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.is_success = True

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=[httpx.ConnectError("fail"), success_response])
        pool._client = mock_client

        response = await pool.request("GET", "http://example.com")
        assert response.status_code == 200
        await pool.close()

    async def test_request_exhausts_retries_raises(self):
        pool = ConnectionPool(ConnectionPoolConfig(max_retries=1, retry_backoff_factor=0.0))

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("fail"))
        pool._client = mock_client

        with pytest.raises(httpx.ConnectError):
            await pool.request("GET", "http://example.com")
        assert pool._metrics.requests_failed == 1
        await pool.close()

    async def test_get_shortcut(self):
        pool = ConnectionPool()
        mock_response = MagicMock(status_code=200, is_success=True)
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        pool._client = mock_client

        response = await pool.get("http://example.com")
        assert response.status_code == 200
        mock_client.request.assert_called_once()
        await pool.close()

    async def test_post_shortcut(self):
        pool = ConnectionPool()
        mock_response = MagicMock(status_code=201, is_success=True)
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        pool._client = mock_client

        response = await pool.post("http://example.com", json={"key": "value"})
        assert response.status_code == 201
        await pool.close()

    async def test_health_check_healthy(self):
        pool = ConnectionPool()
        mock_response = MagicMock(status_code=200, is_success=True)
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        pool._client = mock_client

        result = await pool.health_check("http://example.com/health")
        assert result is True
        assert pool._metrics.health_checks_passed == 1
        await pool.close()

    async def test_health_check_unhealthy(self):
        pool = ConnectionPool()
        mock_response = MagicMock(status_code=500, is_success=False)
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        pool._client = mock_client

        result = await pool.health_check("http://example.com/health")
        assert result is False
        assert pool._metrics.health_checks_failed == 1
        await pool.close()

    async def test_health_check_exception(self):
        pool = ConnectionPool()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("fail"))
        pool._client = mock_client

        # Health check with max_retries=0 would still retry on ConnectError
        # Use a config with 0 retries to make the exception propagate to health_check
        pool.config = ConnectionPoolConfig(max_retries=0)
        result = await pool.health_check("http://example.com/health")
        assert result is False
        assert pool._metrics.health_checks_failed == 1
        await pool.close()

    def test_get_stats(self):
        pool = ConnectionPool()
        stats = pool.get_stats()
        assert "requests" in stats
        assert "connections" in stats


@pytest.mark.unit
class TestCreateConnectionPool:
    """Tests for create_connection_pool factory function."""

    def test_creates_pool_with_defaults(self):
        pool = create_connection_pool()
        assert pool is not None
        assert isinstance(pool, ConnectionPool)
        assert pool.config.max_connections == 100

    def test_creates_pool_with_custom_params(self):
        pool = create_connection_pool(max_connections=50, read_timeout=30.0, http2=False)
        assert pool is not None
        assert pool.config.max_connections == 50
        assert pool.config.read_timeout == 30.0
        assert pool.config.http2 is False

    def test_returns_none_without_httpx(self):
        with patch("src.performance.connection_pool._HAS_HTTPX", False):
            pool = create_connection_pool()
            assert pool is None
