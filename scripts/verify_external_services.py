#!/usr/bin/env python3
"""
External Services Verification Script
======================================

Validates connectivity and authentication for all external services required
by the training pipeline before execution begins.

Services Verified:
- Pinecone (vector database)
- Weights & Biases (experiment tracking)
- GitHub API (data fetching)
- OpenAI API (optional extraction)
- Neo4j (optional knowledge graph)

Usage:
    python scripts/verify_external_services.py [--config CONFIG_PATH] [--verbose]

2025 Best Practices:
- Type hints throughout
- Pydantic for validation
- Async/await for concurrent checks
- Structured logging
- Comprehensive error handling
- Security: No API keys in logs
- Retry logic with exponential backoff
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
import sys
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import yaml

# ============================================================================
# Configuration
# ============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ServiceStatus(str, Enum):
    """Service verification status."""

    SUCCESS = "✓ SUCCESS"
    FAILED = "✗ FAILED"
    WARNING = "⚠ WARNING"
    SKIPPED = "- SKIPPED"


class ServiceType(str, Enum):
    """Type of external service."""

    VECTOR_DB = "vector_database"
    EXPERIMENT_TRACKING = "experiment_tracking"
    API = "api"
    DATABASE = "database"


@dataclass
class VerificationResult:
    """Result of service verification."""

    service_name: str
    status: ServiceStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    is_critical: bool = True


class ServiceConfig(BaseModel):
    """Configuration for a single external service."""

    name: str = Field(..., description="Service name")
    env_var: str = Field(..., description="Environment variable for API key")
    description: str = Field(..., description="Service description")
    verification_endpoint: str | None = Field(None, description="Endpoint for connectivity check")
    required: bool = Field(True, description="Is service required?")
    timeout_seconds: int = Field(10, description="Request timeout")
    service_type: ServiceType = Field(ServiceType.API, description="Type of service")

    @field_validator("env_var")
    @classmethod
    def env_var_uppercase(cls, v: str) -> str:
        """Ensure environment variable is uppercase."""
        return v.upper()


# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up structured logging with Rich handler.

    Args:
        verbose: Enable debug logging

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
                show_time=True,
                show_path=verbose,
            )
        ],
    )

    return logging.getLogger(__name__)


# ============================================================================
# Service Verifiers
# ============================================================================


class ServiceVerifier:
    """Base class for service verification."""

    def __init__(
        self,
        config: ServiceConfig,
        logger: logging.Logger,
        console: Console,
    ):
        """
        Initialize service verifier.

        Args:
            config: Service configuration
            logger: Logger instance
            console: Rich console for output
        """
        self.config = config
        self.logger = logger
        self.console = console
        self.client = httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=True,
        )

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        await self.client.aclose()

    def get_api_key(self) -> str | None:
        """
        Get API key from environment.

        Returns:
            API key or None if not found
        """
        key = os.getenv(self.config.env_var)
        if key:
            self.logger.debug(f"Found API key for {self.config.name} " f"(length: {len(key)})")
        return key

    @retry(
        retry=retry_if_exception_type(httpx.TimeoutException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def verify(self) -> VerificationResult:
        """
        Verify service connectivity and authentication.

        Returns:
            Verification result

        Note:
            Subclasses should override this method for service-specific checks
        """
        raise NotImplementedError("Subclasses must implement verify()")


class PineconeVerifier(ServiceVerifier):
    """Pinecone vector database verifier."""

    async def verify(self) -> VerificationResult:
        """Verify Pinecone connectivity and authentication."""
        api_key = self.get_api_key()

        if not api_key:
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Missing {self.config.env_var} environment variable",
                is_critical=self.config.required,
            )

        try:
            # Verify API key by listing indexes
            import time

            start = time.time()

            response = await self.client.get(
                "https://api.pinecone.io/indexes",
                headers={"Api-Key": api_key},
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                indexes = data.get("indexes", [])

                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.SUCCESS,
                    message=f"Connected successfully ({len(indexes)} indexes)",
                    details={
                        "indexes": [idx.get("name") for idx in indexes],
                        "api_version": response.headers.get("X-Api-Version"),
                    },
                    latency_ms=latency,
                    is_critical=self.config.required,
                )
            elif response.status_code == 401:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.FAILED,
                    message="Invalid API key",
                    details={"status_code": 401},
                    is_critical=self.config.required,
                )
            else:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.WARNING,
                    message=f"Unexpected status code: {response.status_code}",
                    details={"status_code": response.status_code},
                    is_critical=self.config.required,
                )

        except httpx.TimeoutException:
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message="Request timeout - check network connectivity",
                is_critical=self.config.required,
            )
        except Exception as e:
            self.logger.exception(f"Pinecone verification failed: {e}")
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Verification error: {str(e)[:100]}",
                is_critical=self.config.required,
            )


class WandBVerifier(ServiceVerifier):
    """Weights & Biases verifier."""

    async def verify(self) -> VerificationResult:
        """Verify W&B connectivity and authentication."""
        api_key = self.get_api_key()

        if not api_key:
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Missing {self.config.env_var} environment variable",
                is_critical=self.config.required,
            )

        try:
            import time

            start = time.time()

            # Verify using W&B viewer query
            query = """
            query Viewer {
                viewer {
                    id
                    username
                    email
                    teams {
                        edges {
                            node {
                                name
                            }
                        }
                    }
                }
            }
            """

            response = await self.client.post(
                "https://api.wandb.ai/graphql",
                json={"query": query},
                headers={"Authorization": f"Bearer {api_key}"},
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                viewer = data.get("data", {}).get("viewer", {})

                if viewer:
                    return VerificationResult(
                        service_name=self.config.name,
                        status=ServiceStatus.SUCCESS,
                        message=f"Authenticated as {viewer.get('username')}",
                        details={
                            "user_id": viewer.get("id"),
                            "username": viewer.get("username"),
                            "teams": [edge["node"]["name"] for edge in viewer.get("teams", {}).get("edges", [])],
                        },
                        latency_ms=latency,
                        is_critical=self.config.required,
                    )
                else:
                    return VerificationResult(
                        service_name=self.config.name,
                        status=ServiceStatus.FAILED,
                        message="Invalid API key",
                        is_critical=self.config.required,
                    )
            else:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.FAILED,
                    message=f"HTTP {response.status_code}",
                    details={"status_code": response.status_code},
                    is_critical=self.config.required,
                )

        except Exception as e:
            self.logger.exception(f"W&B verification failed: {e}")
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Verification error: {str(e)[:100]}",
                is_critical=self.config.required,
            )


class GitHubVerifier(ServiceVerifier):
    """GitHub API verifier."""

    async def verify(self) -> VerificationResult:
        """Verify GitHub API connectivity and authentication."""
        api_key = self.get_api_key()

        if not api_key:
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Missing {self.config.env_var} environment variable",
                is_critical=self.config.required,
            )

        try:
            import time

            start = time.time()

            response = await self.client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"token {api_key}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()

                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.SUCCESS,
                    message=f"Authenticated as {data.get('login')}",
                    details={
                        "username": data.get("login"),
                        "user_type": data.get("type"),
                        "scopes": response.headers.get("X-OAuth-Scopes", "").split(", "),
                    },
                    latency_ms=latency,
                    is_critical=self.config.required,
                )
            elif response.status_code == 401:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.FAILED,
                    message="Invalid token or insufficient permissions",
                    is_critical=self.config.required,
                )
            else:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.FAILED,
                    message=f"HTTP {response.status_code}",
                    details={"status_code": response.status_code},
                    is_critical=self.config.required,
                )

        except Exception as e:
            self.logger.exception(f"GitHub verification failed: {e}")
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Verification error: {str(e)[:100]}",
                is_critical=self.config.required,
            )


class OpenAIVerifier(ServiceVerifier):
    """OpenAI API verifier."""

    async def verify(self) -> VerificationResult:
        """Verify OpenAI API connectivity and authentication."""
        api_key = self.get_api_key()

        if not api_key:
            if not self.config.required:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.SKIPPED,
                    message="Optional service - API key not provided",
                    is_critical=False,
                )

            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Missing {self.config.env_var} environment variable",
                is_critical=self.config.required,
            )

        try:
            import time

            start = time.time()

            response = await self.client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])

                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.SUCCESS,
                    message=f"Connected ({len(models)} models available)",
                    details={
                        "model_count": len(models),
                        "sample_models": [m.get("id") for m in models[:5]],
                    },
                    latency_ms=latency,
                    is_critical=self.config.required,
                )
            elif response.status_code == 401:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.FAILED,
                    message="Invalid API key",
                    is_critical=self.config.required,
                )
            else:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.WARNING,
                    message=f"HTTP {response.status_code}",
                    details={"status_code": response.status_code},
                    is_critical=self.config.required,
                )

        except Exception as e:
            self.logger.exception(f"OpenAI verification failed: {e}")
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED if self.config.required else ServiceStatus.WARNING,
                message=f"Verification error: {str(e)[:100]}",
                is_critical=self.config.required,
            )


class Neo4jVerifier(ServiceVerifier):
    """Neo4j database verifier."""

    async def verify(self) -> VerificationResult:
        """Verify Neo4j connectivity."""
        password = self.get_api_key()

        if not password:
            if not self.config.required:
                return VerificationResult(
                    service_name=self.config.name,
                    status=ServiceStatus.SKIPPED,
                    message="Optional service - credentials not provided",
                    is_critical=False,
                )

            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.FAILED,
                message=f"Missing {self.config.env_var} environment variable",
                is_critical=self.config.required,
            )

        try:
            # Neo4j requires specialized client
            # For now, just check that credentials are present
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.WARNING,
                message="Credentials found (full connectivity check requires neo4j driver)",
                details={"credential_length": len(password)},
                is_critical=self.config.required,
            )

        except Exception as e:
            self.logger.exception(f"Neo4j verification failed: {e}")
            return VerificationResult(
                service_name=self.config.name,
                status=ServiceStatus.WARNING if not self.config.required else ServiceStatus.FAILED,
                message=f"Verification error: {str(e)[:100]}",
                is_critical=self.config.required,
            )


# ============================================================================
# Verifier Factory
# ============================================================================


VERIFIER_MAP = {
    "pinecone": PineconeVerifier,
    "wandb": WandBVerifier,
    "github": GitHubVerifier,
    "openai": OpenAIVerifier,
    "neo4j": Neo4jVerifier,
}


def create_verifier(
    config: ServiceConfig,
    logger: logging.Logger,
    console: Console,
) -> ServiceVerifier:
    """
    Create appropriate verifier for service.

    Args:
        config: Service configuration
        logger: Logger instance
        console: Rich console

    Returns:
        Service verifier instance
    """
    verifier_class = VERIFIER_MAP.get(config.name.lower())

    if not verifier_class:
        # Return base verifier for unknown services
        return ServiceVerifier(config, logger, console)

    return verifier_class(config, logger, console)


# ============================================================================
# Main Verification Logic
# ============================================================================


async def verify_all_services(
    config_path: Path,
    logger: logging.Logger,
    console: Console,
) -> list[VerificationResult]:
    """
    Verify all external services concurrently.

    Args:
        config_path: Path to training configuration
        logger: Logger instance
        console: Rich console

    Returns:
        List of verification results
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract service configurations
    services: list[ServiceConfig] = []

    # Required services
    for svc in config.get("external_services", {}).get("required", []):
        services.append(ServiceConfig(**svc, required=True))

    # Optional services
    for svc in config.get("external_services", {}).get("optional", []):
        services.append(ServiceConfig(**svc, required=False))

    logger.info(f"Verifying {len(services)} external services...")

    # Run verifications concurrently
    tasks = []
    for svc_config in services:
        verifier = create_verifier(svc_config, logger, console)
        tasks.append(verifier.verify())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.exception(f"Verification failed: {result}")
            final_results.append(
                VerificationResult(
                    service_name=services[i].name,
                    status=ServiceStatus.FAILED,
                    message=f"Exception: {str(result)[:100]}",
                    is_critical=services[i].required,
                )
            )
        else:
            final_results.append(result)

    return final_results


def display_results(
    results: list[VerificationResult],
    console: Console,
) -> None:
    """
    Display verification results in a formatted table.

    Args:
        results: List of verification results
        console: Rich console
    """
    table = Table(title="External Services Verification Results")

    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Message", style="white")
    table.add_column("Latency", justify="right")

    for result in results:
        # Color-code status
        status_style = {
            ServiceStatus.SUCCESS: "green",
            ServiceStatus.FAILED: "red",
            ServiceStatus.WARNING: "yellow",
            ServiceStatus.SKIPPED: "dim",
        }.get(result.status, "white")

        latency_str = f"{result.latency_ms:.0f}ms" if result.latency_ms is not None else "-"

        table.add_row(
            result.service_name,
            f"[{status_style}]{result.status.value}[/{status_style}]",
            result.message,
            latency_str,
        )

    console.print(table)


def check_critical_failures(results: list[VerificationResult]) -> bool:
    """
    Check if any critical services failed.

    Args:
        results: List of verification results

    Returns:
        True if all critical services passed, False otherwise
    """
    critical_failures = [r for r in results if r.is_critical and r.status == ServiceStatus.FAILED]

    return len(critical_failures) == 0


# ============================================================================
# CLI Entry Point
# ============================================================================


async def main():
    """Main entry point for verification script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify external services connectivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training" / "config_local_demo.yaml",
        help="Path to training configuration file",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    # Setup
    console = Console()
    logger = setup_logging(args.verbose)

    # Verify config exists
    if not args.config.exists():
        console.print(f"[red]Error: Configuration file not found: {args.config}[/red]")
        return 1

    # Run verification
    console.print("\n[bold cyan]🔍 External Services Verification[/bold cyan]\n")

    try:
        results = await verify_all_services(args.config, logger, console)
        display_results(results, console)

        # Check for critical failures
        if check_critical_failures(results):
            console.print("\n[bold green]✓ All critical services verified successfully![/bold green]\n")
            return 0
        else:
            console.print("\n[bold red]✗ Critical service verification failed![/bold red]")
            console.print("[yellow]Please ensure all required environment variables are set.[/yellow]\n")
            return 1

    except Exception as e:
        logger.exception("Verification failed with exception")
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
