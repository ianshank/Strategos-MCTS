"""
Unit tests for health check endpoints.

Tests health checking functionality for all system components.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 10
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Set environment variables before importing modules that depend on settings
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

# Import health modules with graceful fallback
try:
    from src.api.health import (
        ComponentHealth,
        HealthChecker,
        HealthStatus,
        get_health_checker,
        reset_health_checker,
    )

    HEALTH_AVAILABLE = True
except ImportError:
    HEALTH_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not HEALTH_AVAILABLE, reason="Health module not available"),
]


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )

        d = health.to_dict()

        assert d["name"] == "test"
        assert d["status"] == "healthy"
        assert "latency_ms" not in d
        assert "message" not in d
        assert "details" not in d

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.DEGRADED,
            latency_ms=12.345,
            message="Test message",
            details={"key": "value"},
        )

        d = health.to_dict()

        assert d["name"] == "test"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 12.35  # Rounded
        assert d["message"] == "Test message"
        assert d["details"] == {"key": "value"}


class TestHealthChecker:
    """Test HealthChecker functionality."""

    @pytest.fixture
    def checker(self):
        """Create health checker for testing."""
        reset_health_checker()
        return HealthChecker(timeout_seconds=5.0)

    @pytest.mark.asyncio
    async def test_check_liveness(self, checker):
        """Test liveness check."""
        result = await checker.check_liveness()

        assert result["status"] == "alive"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_check_readiness(self, checker):
        """Test readiness check."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            result = await checker.check_readiness()

            assert "ready" in result
            assert "timestamp" in result
            assert "check_duration_ms" in result

    @pytest.mark.asyncio
    async def test_check_all(self, checker):
        """Test full health check."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            result = await checker.check_all()

            assert "status" in result
            assert result["status"] in ["healthy", "degraded", "unhealthy"]
            assert "timestamp" in result
            assert "version" in result
            assert "components" in result
            assert isinstance(result["components"], list)

    @pytest.mark.asyncio
    async def test_check_settings_healthy(self, checker):
        """Test settings health check when healthy."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            result = await checker._check_settings()

            assert result.name == "settings"
            assert result.status == HealthStatus.HEALTHY
            assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_check_memory_healthy(self, checker):
        """Test memory health check."""
        result = await checker._check_memory()

        assert result.name == "memory"
        assert result.latency_ms is not None
        assert result.details is not None
        assert "memory_mb" in result.details

    @pytest.mark.asyncio
    async def test_check_memory_high_usage(self, checker):
        """Test memory health check with high usage."""
        with patch("psutil.Process") as mock_process:
            mock_instance = MagicMock()
            mock_instance.memory_info.return_value = MagicMock(
                rss=8 * 1024 * 1024 * 1024,  # 8GB
                vms=16 * 1024 * 1024 * 1024,
            )
            mock_instance.memory_percent.return_value = 80.0
            mock_process.return_value = mock_instance

            result = await checker._check_memory()

            assert result.status == HealthStatus.DEGRADED
            assert result.message is not None

    @pytest.mark.asyncio
    async def test_check_llm_client(self, checker):
        """Test LLM client health check."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            result = await checker._check_llm_client()

            assert result.name == "llm_client"
            # Should be healthy or degraded (not unhealthy)
            assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @pytest.mark.asyncio
    async def test_check_mcts_engine(self, checker):
        """Test MCTS engine health check."""
        result = await checker._check_mcts_engine()

        assert result.name == "mcts_engine"
        assert result.status == HealthStatus.HEALTHY
        assert result.details is not None
        assert "seed" in result.details

    @pytest.mark.asyncio
    async def test_check_vector_store_not_configured(self, checker):
        """Test vector store check when not configured."""
        # Remove Pinecone API key from settings
        with patch.object(checker.settings, "PINECONE_API_KEY", None):
            result = await checker._check_vector_store()

            assert result.name == "vector_store"
            assert result.status == HealthStatus.HEALTHY
            assert "Not configured" in (result.message or "")

    @pytest.mark.asyncio
    async def test_check_cache(self, checker):
        """Test cache health check."""
        result = await checker._check_cache()

        assert result.name == "cache"
        # Should be healthy or degraded (cache may not be initialized)
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @pytest.mark.asyncio
    async def test_overall_status_unhealthy(self, checker):
        """Test overall status is unhealthy when any component is unhealthy."""

        # Mock _check_mcts_engine to return unhealthy
        async def mock_unhealthy():
            return ComponentHealth(
                name="mcts_engine",
                status=HealthStatus.UNHEALTHY,
                message="Test failure",
            )

        with patch.object(checker, "_check_mcts_engine", mock_unhealthy):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
                result = await checker.check_all()

                assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_overall_status_degraded(self, checker):
        """Test overall status is degraded when any component is degraded."""

        # Mock _check_cache to return degraded
        async def mock_degraded():
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message="Test degradation",
            )

        with patch.object(checker, "_check_cache", mock_degraded):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
                result = await checker.check_all()

                assert result["status"] in ["degraded", "unhealthy"]


class TestGlobalHealthChecker:
    """Test global health checker instance."""

    @pytest.fixture(autouse=True)
    def reset_global_checker(self):
        """Reset global checker before each test."""
        reset_health_checker()
        yield
        reset_health_checker()

    def test_get_health_checker_singleton(self):
        """Test health checker is singleton."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2

    def test_reset_health_checker(self):
        """Test health checker reset."""
        checker1 = get_health_checker()
        reset_health_checker()
        checker2 = get_health_checker()

        assert checker1 is not checker2
