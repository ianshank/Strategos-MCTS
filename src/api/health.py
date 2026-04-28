"""
Comprehensive health check endpoints.

Provides:
- Liveness probe (is the service running)
- Readiness probe (is the service ready to accept traffic)
- Detailed health status (all components)

Designed for Kubernetes health probes and monitoring dashboards.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 4
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.config.settings import get_settings
from src.observability.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels for components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.message is not None:
            result["message"] = self.message
        if self.details is not None:
            result["details"] = self.details
        return result


class HealthChecker:
    """
    Comprehensive health checking for all system components.

    Provides three levels of health checks:
    1. Liveness - Is the process alive?
    2. Readiness - Can the service handle traffic?
    3. Full - Detailed status of all components

    Example:
        >>> checker = HealthChecker()
        >>> health = await checker.check_all()
        >>> print(health["status"])  # "healthy", "degraded", or "unhealthy"
    """

    def __init__(
        self,
        timeout_seconds: float | None = None,
    ):
        """
        Initialize health checker.

        Args:
            timeout_seconds: Timeout for individual component checks
        """
        settings = get_settings()
        self.timeout = timeout_seconds or settings.HTTP_TIMEOUT_SECONDS
        self.settings = settings
        self._logger = logger.getChild("HealthChecker")

    async def check_liveness(self) -> dict[str, Any]:
        """
        Liveness check - is the service running?

        Returns minimal response for Kubernetes liveness probe.
        Should never fail unless the process is dead.

        Returns:
            Simple alive status with timestamp
        """
        return {
            "status": "alive",
            "timestamp": time.time(),
        }

    async def check_readiness(self) -> dict[str, Any]:
        """
        Readiness check - is the service ready for traffic?

        Checks critical dependencies only (fast check).
        Used by Kubernetes readiness probe.

        Returns:
            Readiness status with critical component health
        """
        start_time = time.time()

        # Check only critical components
        checks = await asyncio.gather(
            self._check_settings(),
            self._check_memory(),
            return_exceptions=True,
        )

        # Determine readiness
        all_healthy = all(isinstance(c, ComponentHealth) and c.status == HealthStatus.HEALTHY for c in checks)

        return {
            "ready": all_healthy,
            "timestamp": time.time(),
            "check_duration_ms": round((time.time() - start_time) * 1000, 2),
        }

    async def check_all(self) -> dict[str, Any]:
        """
        Detailed health check of all components.

        Returns comprehensive status for monitoring dashboards.
        Includes all components with individual status and latency.

        Returns:
            Full health report with component details
        """
        start_time = time.time()

        # Check all components in parallel
        checks = await asyncio.gather(
            self._check_settings(),
            self._check_memory(),
            self._check_llm_client(),
            self._check_mcts_engine(),
            self._check_vector_store(),
            self._check_cache(),
            return_exceptions=True,
        )

        components = []
        for check in checks:
            if isinstance(check, ComponentHealth):
                components.append(check.to_dict())
            elif isinstance(check, Exception):
                components.append(
                    {
                        "name": "unknown",
                        "status": HealthStatus.UNHEALTHY.value,
                        "message": str(check),
                    }
                )

        # Determine overall status
        statuses = [c["status"] for c in components]
        if HealthStatus.UNHEALTHY.value in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED.value in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            "status": overall.value,
            "timestamp": time.time(),
            "version": "0.1.0",
            "check_duration_ms": round((time.time() - start_time) * 1000, 2),
            "components": components,
        }

    async def _check_settings(self) -> ComponentHealth:
        """Check settings are loaded correctly."""
        start = time.time()
        try:
            settings = get_settings()
            # Verify critical settings exist
            assert settings.LLM_PROVIDER is not None

            return ComponentHealth(
                name="settings",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={
                    "provider": settings.LLM_PROVIDER.value,
                    "mcts_enabled": settings.MCTS_ENABLED,
                },
            )
        except Exception as e:
            self._logger.warning(f"Settings health check failed: {e}")
            return ComponentHealth(
                name="settings",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_memory(self) -> ComponentHealth:
        """Check memory usage is acceptable."""
        start = time.time()
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()

            # Determine status based on memory usage
            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "Critical memory pressure"
            elif memory_percent > 75:
                status = HealthStatus.DEGRADED
                message = "High memory usage"
            else:
                status = HealthStatus.HEALTHY
                message = None

            return ComponentHealth(
                name="memory",
                status=status,
                latency_ms=(time.time() - start) * 1000,
                message=message,
                details={
                    "memory_mb": round(memory_mb, 2),
                    "memory_percent": round(memory_percent, 2),
                    "virtual_mb": round(memory_info.vms / (1024 * 1024), 2),
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=f"Could not check memory: {e}",
            )

    async def _check_llm_client(self) -> ComponentHealth:
        """Check LLM client can be created."""
        start = time.time()
        try:
            from src.framework.factories import LLMClientFactory

            LLMClientFactory()  # Verify factory can be created
            # Don't make actual API calls in health check

            return ComponentHealth(
                name="llm_client",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={
                    "provider": self.settings.LLM_PROVIDER.value,
                    "timeout": self.settings.HTTP_TIMEOUT_SECONDS,
                },
            )
        except ImportError:
            return ComponentHealth(
                name="llm_client",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message="LLM factory not available",
            )
        except Exception as e:
            return ComponentHealth(
                name="llm_client",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_mcts_engine(self) -> ComponentHealth:
        """Check MCTS engine can be instantiated."""
        start = time.time()
        try:
            from src.framework.mcts.core import MCTSEngine

            # Create engine with test seed
            engine = MCTSEngine(seed=42)

            return ComponentHealth(
                name="mcts_engine",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={
                    "seed": engine.seed,
                    "exploration_weight": engine.exploration_weight,
                    "cache_size_limit": engine.cache_size_limit,
                },
            )
        except ImportError:
            return ComponentHealth(
                name="mcts_engine",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message="MCTS module not available",
            )
        except Exception as e:
            return ComponentHealth(
                name="mcts_engine",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_vector_store(self) -> ComponentHealth:
        """Check vector store connectivity (if configured)."""
        start = time.time()

        # Check if vector store is configured
        if not self.settings.PINECONE_API_KEY:
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message="Not configured (optional)",
            )

        try:
            # Just verify settings are valid, don't make API call
            if not self.settings.PINECONE_HOST:
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.DEGRADED,
                    latency_ms=(time.time() - start) * 1000,
                    message="API key set but host not configured",
                )

            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={
                    "host": self.settings.PINECONE_HOST[:30] + "..." if self.settings.PINECONE_HOST else None,
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_cache(self) -> ComponentHealth:
        """Check cache health."""
        start = time.time()
        try:
            from src.framework.caching import get_query_cache

            cache = get_query_cache()
            stats = cache.stats

            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details=stats,
            )
        except ImportError:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message="Cache module not loaded",
            )
        except Exception as e:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )


# Singleton instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get or create health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def reset_health_checker() -> None:
    """Reset health checker. Useful for testing."""
    global _health_checker
    _health_checker = None


__all__ = [
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "get_health_checker",
    "reset_health_checker",
]
