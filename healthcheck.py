#!/usr/bin/env python3
"""
Comprehensive health check script for LangGraph Multi-Agent MCTS Framework.

This script performs async health checks covering:
- GPU availability (CUDA)
- LLM provider connectivity (OpenAI, Anthropic, LMStudio)
- Pinecone vector database connectivity
- Proper timeout handling
- Graceful degradation (partial health states)
- Structured health check responses
- OpenTelemetry integration (if available)

Exit codes:
- 0: Operational (Docker-healthy) — all critical checks pass. Covers both
  HEALTHY (every check passed) and DEGRADED (optional services such as the GPU,
  LLM provider, Pinecone or OpenTelemetry are absent or unconfigured, but the
  container is ready to serve). The structured JSON report distinguishes the
  two so operators can see which optional integrations are down.
- 1: Unhealthy — one or more critical checks failed; the container cannot serve.

Docker's HEALTHCHECK contract treats only exit 0 as healthy; any non-zero
exit (including the previously-used code 2) is a check failure. A DEGRADED
container is operational, so it must exit 0 or the image can never become
healthy on a CPU-only host or in a smoke-test environment.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
import sys
import time
from typing import Any, Final

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Environment variable names this script honours. Named rather than inlined so each
# contract is stated once.
#
# Deliberately read from os.environ rather than the project's Pydantic Settings:
# this module is a STANDALONE script copied into the image (Dockerfile.train:95,
# wired as the container HEALTHCHECK). It holds no `src` imports at all, and every
# one of its nine config reads — PINECONE_API_KEY, OPENAI_API_KEY, LLM_PROVIDER,
# LMSTUDIO_BASE_URL, OTEL_EXPORTER_OTLP_ENDPOINT and the rest — uses os.environ.
# Importing Settings would add this file's first `src` dependency AND require a
# configured provider key: get_settings() raises ValidationError without one, which
# would crash the healthcheck in precisely the degraded containers it exists to
# probe.
REQUIRE_GPU_ENV: Final[str] = "REQUIRE_GPU"

# Accepted spellings for a boolean environment flag.
TRUTHY_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes"})


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


def exit_code_for_status(status: HealthStatus) -> int:
    """Map an overall health status to a process exit code for the Docker HEALTHCHECK.

    Docker treats only exit 0 as healthy; any non-zero exit is a check failure
    (exit 2 is reserved and *must not* be used — it counts as a failure).
    A DEGRADED container — critical checks pass but optional services are
    absent — is operational and must exit 0, otherwise the image can never
    report healthy on a CPU-only host or in a smoke-test environment where no
    GPU, LLM key, Pinecone host or OTEL endpoint is configured.

    The structured JSON report still carries the DEGRADED status, so operators
    do not lose the signal that optional integrations are down.

    Args:
        status: The overall ``HealthStatus`` returned by ``run_all_checks``.

    Returns:
        0 for HEALTHY and DEGRADED (operational), 1 for UNHEALTHY
        (critical failure).
    """
    return 1 if status is HealthStatus.UNHEALTHY else 0


@dataclass
class CheckResult:
    """Result of an individual health check."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    critical: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
            "critical": self.critical,
            "metadata": self.metadata,
        }


@dataclass
class HealthCheckReport:
    """Overall health check report."""

    status: HealthStatus
    checks: list[CheckResult]
    timestamp: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "checks": {check.name: check.to_dict() for check in self.checks},
            "summary": {
                "total": len(self.checks),
                "healthy": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
            },
        }


class HealthChecker:
    """Comprehensive health checker with async checks and timeout handling."""

    def __init__(self, timeout: float = 5.0):
        """
        Initialize health checker.

        Args:
            timeout: Default timeout in seconds for each check
        """
        self.timeout = timeout
        self.checks: list[CheckResult] = []

    async def run_check(
        self,
        name: str,
        check_fn: callable,
        critical: bool = True,
        timeout: float | None = None,
    ) -> CheckResult:
        """
        Run a single health check with timeout handling.

        Args:
            name: Name of the check
            check_fn: Async function that performs the check
            critical: Whether this check is critical for service health
            timeout: Custom timeout for this check (overrides default)

        Returns:
            CheckResult with status and timing information
        """
        start = time.time()
        check_timeout = timeout or self.timeout

        try:
            # Run check with timeout
            result = await asyncio.wait_for(check_fn(), timeout=check_timeout)
            duration_ms = (time.time() - start) * 1000

            if isinstance(result, CheckResult):
                result.duration_ms = duration_ms
                return result
            else:
                # If check_fn returns a simple dict or bool
                if isinstance(result, dict):
                    return CheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        message="Check passed",
                        duration_ms=duration_ms,
                        critical=critical,
                        metadata=result,
                    )
                elif result is True:
                    return CheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        message="Check passed",
                        duration_ms=duration_ms,
                        critical=critical,
                    )
                else:
                    return CheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message="Check failed",
                        duration_ms=duration_ms,
                        critical=critical,
                    )

        except TimeoutError:
            duration_ms = (time.time() - start) * 1000
            logger.warning(f"Health check '{name}' timed out after {check_timeout}s")
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {check_timeout}s",
                duration_ms=duration_ms,
                critical=critical,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            logger.error(f"Health check '{name}' failed: {e}")
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)[:200]}",
                duration_ms=duration_ms,
                critical=critical,
                metadata={"error_type": type(e).__name__},
            )

    @staticmethod
    def _gpu_required() -> bool:
        """
        Whether a missing GPU should make the container UNHEALTHY.

        The training image runs on both GPU hosts and CPU-only hosts (CI smoke tests
        use `ubuntu-latest`). Treating an absent GPU as unconditionally critical meant
        the container could never report healthy off-GPU, which kept the deployment
        workflow red for a reason unrelated to the image.

        Defaults to False so a CPU host degrades rather than fails. GPU deployments
        set ``REQUIRE_GPU=1`` to keep the check gating.

        Read from the environment rather than Pydantic Settings by design — see
        ``REQUIRE_GPU_ENV`` above.
        """
        return os.environ.get(REQUIRE_GPU_ENV, "").strip().lower() in TRUTHY_ENV_VALUES

    async def check_cuda(self) -> CheckResult:
        """Check CUDA/GPU availability."""
        gpu_required = self._gpu_required()
        # Absent GPU is a hard failure only where a GPU was declared necessary;
        # elsewhere it is a capability the container is running without.
        absent_status = HealthStatus.UNHEALTHY if gpu_required else HealthStatus.DEGRADED
        absent_hint = "" if gpu_required else " (set REQUIRE_GPU=1 to treat this as fatal)"

        try:
            import torch

            if not torch.cuda.is_available():
                return CheckResult(
                    name="cuda",
                    status=absent_status,
                    message=f"CUDA not available{absent_hint}",
                    duration_ms=0,
                    critical=gpu_required,
                )

            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                return CheckResult(
                    name="cuda",
                    status=absent_status,
                    message=f"No GPUs detected{absent_hint}",
                    duration_ms=0,
                    critical=gpu_required,
                )

            # Get GPU info
            gpu_info = []
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_info.append({"id": i, "name": gpu_name})

            return CheckResult(
                name="cuda",
                status=HealthStatus.HEALTHY,
                message=f"{gpu_count} GPU(s) available",
                duration_ms=0,
                critical=True,
                metadata={"gpu_count": gpu_count, "gpus": gpu_info},
            )

        except ImportError:
            return CheckResult(
                name="cuda",
                status=absent_status,
                message=f"PyTorch not installed{absent_hint}",
                duration_ms=0,
                critical=gpu_required,
            )
        except Exception as e:
            return CheckResult(
                name="cuda",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                duration_ms=0,
                critical=True,
            )

    async def check_pinecone(self) -> CheckResult:
        """Check Pinecone vector database connectivity."""
        try:
            # Check if Pinecone is configured
            api_key = os.environ.get("PINECONE_API_KEY")
            host = os.environ.get("PINECONE_HOST")

            if not api_key or not host:
                return CheckResult(
                    name="pinecone",
                    status=HealthStatus.DEGRADED,
                    message="Not configured (optional service)",
                    duration_ms=0,
                    critical=False,
                    metadata={"configured": False},
                )

            # Try to import and initialize Pinecone
            try:
                from pinecone import Pinecone
            except ImportError:
                return CheckResult(
                    name="pinecone",
                    status=HealthStatus.DEGRADED,
                    message="Pinecone package not installed",
                    duration_ms=0,
                    critical=False,
                    metadata={"configured": True, "package_available": False},
                )

            # Initialize client and test connectivity
            start = time.time()
            client = Pinecone(api_key=api_key)
            index = client.Index(host=host)

            # Test with describe_index_stats (lightweight operation)
            stats = index.describe_index_stats()
            duration_ms = (time.time() - start) * 1000

            return CheckResult(
                name="pinecone",
                status=HealthStatus.HEALTHY,
                message="Connected successfully",
                duration_ms=duration_ms,
                critical=False,
                metadata={
                    "configured": True,
                    "connected": True,
                    "total_vectors": stats.get("total_vector_count", 0),
                    "dimension": stats.get("dimension", 0),
                },
            )

        except Exception as e:
            return CheckResult(
                name="pinecone",
                status=HealthStatus.DEGRADED,
                message=f"Connection failed: {str(e)[:100]}",
                duration_ms=0,
                critical=False,
                metadata={"configured": True, "error": type(e).__name__},
            )

    async def check_llm_provider(self, provider: str) -> CheckResult:
        """
        Check LLM provider connectivity.

        Args:
            provider: Provider name (openai, anthropic, lmstudio)

        Returns:
            CheckResult with connectivity status
        """
        try:
            # Import LLM adapter
            try:
                from src.adapters.llm import create_client
                from src.adapters.llm.exceptions import LLMClientError
            except ImportError:
                return CheckResult(
                    name=f"llm_{provider}",
                    status=HealthStatus.UNHEALTHY,
                    message="LLM adapters not available",
                    duration_ms=0,
                    critical=True,
                    metadata={"provider": provider},
                )

            # Check provider-specific configuration
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key or api_key in ("", "your-api-key-here", "sk-xxx"):
                    return CheckResult(
                        name=f"llm_{provider}",
                        status=HealthStatus.DEGRADED,
                        message="Not configured",
                        duration_ms=0,
                        critical=False,
                        metadata={"provider": provider, "configured": False},
                    )
            elif provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key or api_key in ("", "your-api-key-here"):
                    return CheckResult(
                        name=f"llm_{provider}",
                        status=HealthStatus.DEGRADED,
                        message="Not configured",
                        duration_ms=0,
                        critical=False,
                        metadata={"provider": provider, "configured": False},
                    )
            elif provider == "lmstudio":
                base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
                # LMStudio is optional and doesn't require API key
                pass

            # Create client and test with minimal request
            start = time.time()

            try:
                client = create_client(provider=provider, timeout=self.timeout)

                # For OpenAI/Anthropic, test with a minimal completion
                # For LMStudio, we can check the health endpoint if available
                if provider in ("openai", "anthropic"):
                    # Make a minimal test request (1 token)
                    response = await client.generate(
                        prompt="Hi",
                        max_tokens=1,
                        temperature=0.0,
                    )
                    duration_ms = (time.time() - start) * 1000

                    await client.close()

                    return CheckResult(
                        name=f"llm_{provider}",
                        status=HealthStatus.HEALTHY,
                        message="Connected and responding",
                        duration_ms=duration_ms,
                        critical=True,
                        metadata={
                            "provider": provider,
                            "configured": True,
                            "model": response.model if hasattr(response, "model") else "unknown",
                        },
                    )
                elif provider == "lmstudio":
                    # For LMStudio, check if we can create a client
                    # Full test would require server to be running
                    import httpx

                    base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
                    async with httpx.AsyncClient(timeout=self.timeout) as http_client:
                        response = await http_client.get(f"{base_url}/models")
                        duration_ms = (time.time() - start) * 1000

                        if response.status_code == 200:
                            models = response.json().get("data", [])
                            return CheckResult(
                                name=f"llm_{provider}",
                                status=HealthStatus.HEALTHY,
                                message="Connected and responding",
                                duration_ms=duration_ms,
                                critical=False,
                                metadata={
                                    "provider": provider,
                                    "configured": True,
                                    "models_available": len(models),
                                },
                            )
                        else:
                            return CheckResult(
                                name=f"llm_{provider}",
                                status=HealthStatus.DEGRADED,
                                message=f"HTTP {response.status_code}",
                                duration_ms=duration_ms,
                                critical=False,
                                metadata={"provider": provider},
                            )

            except LLMClientError as e:
                duration_ms = (time.time() - start) * 1000
                return CheckResult(
                    name=f"llm_{provider}",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Client error: {str(e)[:100]}",
                    duration_ms=duration_ms,
                    critical=True,
                    metadata={"provider": provider, "error": type(e).__name__},
                )

        except Exception as e:
            return CheckResult(
                name=f"llm_{provider}",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)[:100]}",
                duration_ms=0,
                critical=True,
                metadata={"provider": provider, "error": type(e).__name__},
            )

    async def check_opentelemetry(self) -> CheckResult:
        """Check if OpenTelemetry is configured and available."""
        try:
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

            if not endpoint:
                return CheckResult(
                    name="opentelemetry",
                    status=HealthStatus.DEGRADED,
                    message="Not configured (optional)",
                    duration_ms=0,
                    critical=False,
                    metadata={"configured": False},
                )

            try:
                from opentelemetry import trace
            except ImportError:
                return CheckResult(
                    name="opentelemetry",
                    status=HealthStatus.DEGRADED,
                    message="Package not installed",
                    duration_ms=0,
                    critical=False,
                    metadata={"configured": True, "package_available": False},
                )

            # Check if tracer provider is set up
            tracer_provider = trace.get_tracer_provider()
            has_tracer = hasattr(tracer_provider, "get_tracer")

            return CheckResult(
                name="opentelemetry",
                status=HealthStatus.HEALTHY if has_tracer else HealthStatus.DEGRADED,
                message="Available" if has_tracer else "Not initialized",
                duration_ms=0,
                critical=False,
                metadata={"configured": True, "initialized": has_tracer, "endpoint": endpoint},
            )

        except Exception as e:
            return CheckResult(
                name="opentelemetry",
                status=HealthStatus.DEGRADED,
                message=f"Error: {str(e)[:100]}",
                duration_ms=0,
                critical=False,
                metadata={"error": type(e).__name__},
            )

    async def run_all_checks(self) -> HealthCheckReport:
        """
        Run all health checks concurrently.

        Returns:
            HealthCheckReport with overall status and individual check results
        """
        start = time.time()
        logger.info("Starting health checks...")

        # Determine which LLM provider is configured
        llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()

        # Define checks with their configurations
        checks_to_run = [
            # critical flag follows REQUIRE_GPU, matching the CheckResult itself; it was
            # hard-coded True, so a CPU-only host could never report healthy.
            ("cuda", self.check_cuda(), self._gpu_required(), None),
            ("pinecone", self.run_check("pinecone", self.check_pinecone, critical=False), False, None),
            (
                f"llm_{llm_provider}",
                self.run_check(
                    f"llm_{llm_provider}",
                    lambda: self.check_llm_provider(llm_provider),
                    critical=True,
                    timeout=10.0,
                ),
                True,
                None,
            ),
            (
                "opentelemetry",
                self.run_check("opentelemetry", self.check_opentelemetry, critical=False),
                False,
                None,
            ),
        ]

        # Run all checks sequentially (each awaited in turn)
        check_results = []
        for _name, check_coro, _critical, _timeout in checks_to_run:
            result = await check_coro
            check_results.append(result)
            logger.info(f"Check '{result.name}': {result.status.value} - {result.message} ({result.duration_ms:.2f}ms)")

        # Determine overall status based on check results
        critical_failures = [c for c in check_results if c.critical and c.status == HealthStatus.UNHEALTHY]
        any_failures = [c for c in check_results if c.status == HealthStatus.UNHEALTHY]
        any_degraded = [c for c in check_results if c.status == HealthStatus.DEGRADED]

        if critical_failures:
            overall_status = HealthStatus.UNHEALTHY
        elif any_failures or any_degraded:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        duration_ms = (time.time() - start) * 1000
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        report = HealthCheckReport(
            status=overall_status,
            checks=check_results,
            timestamp=timestamp,
            duration_ms=duration_ms,
        )

        logger.info(f"Health check completed: {overall_status.value} (took {duration_ms:.2f}ms)")

        return report


async def main():
    """Main entry point for health check script."""
    try:
        # Initialize health checker with 5s timeout
        checker = HealthChecker(timeout=5.0)

        # Run all checks
        report = await checker.run_all_checks()

        # Output structured JSON report
        print("\n" + "=" * 80)
        print("HEALTH CHECK REPORT")
        print("=" * 80)
        print(json.dumps(report.to_dict(), indent=2))
        print("=" * 80 + "\n")

        # Determine exit code. DEGRADED maps to 0: the container is operational
        # (only optional services are down), and Docker's HEALTHCHECK treats any
        # non-zero exit as a failure. See ``exit_code_for_status``.
        exit_code = exit_code_for_status(report.status)
        if report.status == HealthStatus.HEALTHY:
            print(f"[OK] Status: {report.status.value.upper()} - All checks passed")
        elif report.status == HealthStatus.DEGRADED:
            print(f"[WARN] Status: {report.status.value.upper()} - Service operational with degraded functionality")
        else:
            print(f"[FAIL] Status: {report.status.value.upper()} - Critical checks failed")
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\nHealth check interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logger.exception("Health check failed with unexpected error")
        print(f"\nERROR: Health check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
