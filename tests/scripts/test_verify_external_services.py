"""
Unit Tests for External Services Verification Script
=====================================================

Tests all verifiers (Pinecone, W&B, GitHub, OpenAI, Neo4j) with mocked responses.

2025 Best Practices:
- pytest with pytest-asyncio for async tests
- httpx.AsyncClient mocking with respx
- Parameterized tests for multiple scenarios
- Fixtures for reusable test data
- Type hints throughout
- Structured test organization
"""

import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
from scripts.verify_external_services import (  # noqa: E402, I001
    GitHubVerifier,
    Neo4jVerifier,
    OpenAIVerifier,
    PineconeVerifier,
    ServiceConfig,
    ServiceStatus,
    ServiceType,
    VerificationResult,
    WandBVerifier,
    check_critical_failures,
    create_verifier,
    verify_all_services,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_console():
    """Mock Rich console."""
    from rich.console import Console

    return Console()


@pytest.fixture
def mock_logger():
    """Mock logger."""
    return logging.getLogger("test")


@pytest.fixture
def pinecone_config() -> ServiceConfig:
    """Pinecone service configuration."""
    return ServiceConfig(
        name="pinecone",
        env_var="PINECONE_API_KEY",
        description="Vector database",
        verification_endpoint="https://api.pinecone.io/indexes",
        required=True,
        service_type=ServiceType.VECTOR_DB,
    )


@pytest.fixture
def wandb_config() -> ServiceConfig:
    """W&B service configuration."""
    return ServiceConfig(
        name="wandb",
        env_var="WANDB_API_KEY",
        description="Experiment tracking",
        verification_endpoint="https://api.wandb.ai/graphql",
        required=True,
        service_type=ServiceType.EXPERIMENT_TRACKING,
    )


@pytest.fixture
def github_config() -> ServiceConfig:
    """GitHub service configuration."""
    return ServiceConfig(
        name="github",
        env_var="GITHUB_TOKEN",
        description="Repository access",
        verification_endpoint="https://api.github.com/user",
        required=True,
        service_type=ServiceType.API,
    )


@pytest.fixture
def openai_config() -> ServiceConfig:
    """OpenAI service configuration."""
    return ServiceConfig(
        name="openai",
        env_var="OPENAI_API_KEY",
        description="OpenAI API",
        verification_endpoint="https://api.openai.com/v1/models",
        required=False,
        service_type=ServiceType.API,
    )


@pytest.fixture
def neo4j_config() -> ServiceConfig:
    """Neo4j service configuration."""
    return ServiceConfig(
        name="neo4j",
        env_var="NEO4J_PASSWORD",
        description="Knowledge graph",
        verification_endpoint=None,
        required=False,
        service_type=ServiceType.DATABASE,
    )


@pytest.fixture
def demo_config_path(tmp_path: Path) -> Path:
    """Create temporary demo config file."""
    config = {
        "external_services": {
            "required": [
                {
                    "name": "pinecone",
                    "env_var": "PINECONE_API_KEY",
                    "description": "Vector database",
                    "verification_endpoint": "https://api.pinecone.io/indexes",
                },
                {
                    "name": "wandb",
                    "env_var": "WANDB_API_KEY",
                    "description": "Experiment tracking",
                    "verification_endpoint": "https://api.wandb.ai/graphql",
                },
            ],
            "optional": [
                {
                    "name": "openai",
                    "env_var": "OPENAI_API_KEY",
                    "description": "OpenAI API",
                    "verification_endpoint": "https://api.openai.com/v1/models",
                }
            ],
        }
    }

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


# ============================================================================
# ServiceConfig Tests
# ============================================================================


def test_service_config_validation():
    """Test ServiceConfig validation."""
    config = ServiceConfig(
        name="test_service",
        env_var="test_api_key",  # Should be uppercased
        description="Test service",
        verification_endpoint="https://example.com",
    )

    assert config.env_var == "TEST_API_KEY"  # Uppercased
    assert config.required is True  # Default
    assert config.timeout_seconds == 10  # Default


def test_service_config_optional():
    """Test optional service configuration."""
    config = ServiceConfig(
        name="optional_service",
        env_var="OPTIONAL_KEY",
        description="Optional service",
        verification_endpoint=None,
        required=False,
    )

    assert config.required is False
    assert config.verification_endpoint is None


# ============================================================================
# PineconeVerifier Tests
# ============================================================================


@pytest.mark.asyncio
async def test_pinecone_verifier_success(
    pinecone_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test successful Pinecone verification."""
    # Mock environment variable
    monkeypatch.setenv("PINECONE_API_KEY", "test-api-key")

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "indexes": [
            {"name": "test-index-1"},
            {"name": "test-index-2"},
        ]
    }
    mock_response.headers = {"X-Api-Version": "2024-01"}

    async with PineconeVerifier(pinecone_config, mock_logger, mock_console) as verifier:
        # Mock client.get
        verifier.client.get = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == ServiceStatus.SUCCESS
        assert result.service_name == "pinecone"
        assert "2 indexes" in result.message
        assert "test-index-1" in result.details["indexes"]
        assert result.latency_ms is not None


@pytest.mark.asyncio
async def test_pinecone_verifier_missing_api_key(
    pinecone_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test Pinecone verification with missing API key."""
    # Ensure env var is not set
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    async with PineconeVerifier(pinecone_config, mock_logger, mock_console) as verifier:
        result = await verifier.verify()

        assert result.status == ServiceStatus.FAILED
        assert "Missing PINECONE_API_KEY" in result.message
        assert result.is_critical is True


@pytest.mark.asyncio
async def test_pinecone_verifier_invalid_api_key(
    pinecone_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test Pinecone verification with invalid API key."""
    monkeypatch.setenv("PINECONE_API_KEY", "invalid-key")

    mock_response = MagicMock()
    mock_response.status_code = 401

    async with PineconeVerifier(pinecone_config, mock_logger, mock_console) as verifier:
        verifier.client.get = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == ServiceStatus.FAILED
        assert "Invalid API key" in result.message


@pytest.mark.asyncio
async def test_pinecone_verifier_timeout(
    pinecone_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test Pinecone verification timeout."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")

    async with PineconeVerifier(pinecone_config, mock_logger, mock_console) as verifier:
        verifier.client.get = AsyncMock(side_effect=httpx.TimeoutException("Request timeout"))

        # Tenacity will retry, but we want to test the final result
        # Patch tenacity to not retry for this test
        with patch(
            "scripts.verify_external_services.retry",
            lambda **_kwargs: lambda f: f,
        ):
            result = await verifier.verify()

        assert result.status == ServiceStatus.FAILED
        assert "timeout" in result.message.lower()


# ============================================================================
# WandBVerifier Tests
# ============================================================================


@pytest.mark.asyncio
async def test_wandb_verifier_success(
    wandb_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test successful W&B verification."""
    monkeypatch.setenv("WANDB_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "viewer": {
                "id": "user-123",
                "username": "testuser",
                "email": "test@example.com",
                "teams": {
                    "edges": [
                        {"node": {"name": "team1"}},
                        {"node": {"name": "team2"}},
                    ]
                },
            }
        }
    }

    async with WandBVerifier(wandb_config, mock_logger, mock_console) as verifier:
        verifier.client.post = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == ServiceStatus.SUCCESS
        assert "testuser" in result.message
        assert result.details["username"] == "testuser"
        assert "team1" in result.details["teams"]


@pytest.mark.asyncio
async def test_wandb_verifier_invalid_response(
    wandb_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test W&B verification with invalid response."""
    monkeypatch.setenv("WANDB_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {}}  # No viewer

    async with WandBVerifier(wandb_config, mock_logger, mock_console) as verifier:
        verifier.client.post = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == ServiceStatus.FAILED
        assert "Invalid API key" in result.message


# ============================================================================
# GitHubVerifier Tests
# ============================================================================


@pytest.mark.asyncio
async def test_github_verifier_success(
    github_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test successful GitHub verification."""
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "login": "testuser",
        "type": "User",
    }
    mock_response.headers = {"X-OAuth-Scopes": "repo, read:org"}

    async with GitHubVerifier(github_config, mock_logger, mock_console) as verifier:
        verifier.client.get = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == ServiceStatus.SUCCESS
        assert "testuser" in result.message
        assert "repo" in result.details["scopes"]


@pytest.mark.asyncio
async def test_github_verifier_insufficient_permissions(
    github_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test GitHub verification with insufficient permissions."""
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    mock_response = MagicMock()
    mock_response.status_code = 401

    async with GitHubVerifier(github_config, mock_logger, mock_console) as verifier:
        verifier.client.get = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == ServiceStatus.FAILED
        assert "Invalid token" in result.message or "insufficient" in result.message


# ============================================================================
# OpenAIVerifier Tests
# ============================================================================


@pytest.mark.asyncio
async def test_openai_verifier_success(
    openai_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test successful OpenAI verification."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "gpt-4"},
            {"id": "gpt-3.5-turbo"},
            {"id": "text-embedding-ada-002"},
        ]
    }

    async with OpenAIVerifier(openai_config, mock_logger, mock_console) as verifier:
        verifier.client.get = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == ServiceStatus.SUCCESS
        assert "3 models" in result.message
        assert result.details["model_count"] == 3


@pytest.mark.asyncio
async def test_openai_verifier_optional_missing(
    openai_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test OpenAI verification when optional and missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async with OpenAIVerifier(openai_config, mock_logger, mock_console) as verifier:
        result = await verifier.verify()

        assert result.status == ServiceStatus.SKIPPED
        assert "Optional service" in result.message
        assert result.is_critical is False


# ============================================================================
# Neo4jVerifier Tests
# ============================================================================


@pytest.mark.asyncio
async def test_neo4j_verifier_credentials_present(
    neo4j_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test Neo4j verification with credentials present."""
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")

    async with Neo4jVerifier(neo4j_config, mock_logger, mock_console) as verifier:
        result = await verifier.verify()

        assert result.status == ServiceStatus.WARNING
        assert "Credentials found" in result.message


@pytest.mark.asyncio
async def test_neo4j_verifier_optional_missing(
    neo4j_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test Neo4j verification when optional and missing."""
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    async with Neo4jVerifier(neo4j_config, mock_logger, mock_console) as verifier:
        result = await verifier.verify()

        assert result.status == ServiceStatus.SKIPPED
        assert "Optional service" in result.message


# ============================================================================
# Verifier Factory Tests
# ============================================================================


def test_create_verifier_pinecone(pinecone_config: ServiceConfig, mock_logger, mock_console):
    """Test verifier factory creates PineconeVerifier."""
    verifier = create_verifier(pinecone_config, mock_logger, mock_console)
    assert isinstance(verifier, PineconeVerifier)


def test_create_verifier_wandb(wandb_config: ServiceConfig, mock_logger, mock_console):
    """Test verifier factory creates WandBVerifier."""
    verifier = create_verifier(wandb_config, mock_logger, mock_console)
    assert isinstance(verifier, WandBVerifier)


def test_create_verifier_github(github_config: ServiceConfig, mock_logger, mock_console):
    """Test verifier factory creates GitHubVerifier."""
    verifier = create_verifier(github_config, mock_logger, mock_console)
    assert isinstance(verifier, GitHubVerifier)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_verify_all_services(
    demo_config_path: Path,
    mock_logger,
    mock_console,
    monkeypatch,
):
    """Test verifying all services from config."""
    # Set environment variables
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Mock all HTTP calls
    with (
        patch("scripts.verify_external_services.PineconeVerifier.verify") as mock_pinecone,
        patch("scripts.verify_external_services.WandBVerifier.verify") as mock_wandb,
        patch("scripts.verify_external_services.OpenAIVerifier.verify") as mock_openai,
    ):
        mock_pinecone.return_value = VerificationResult(
            service_name="pinecone",
            status=ServiceStatus.SUCCESS,
            message="Connected",
            is_critical=True,
        )

        mock_wandb.return_value = VerificationResult(
            service_name="wandb",
            status=ServiceStatus.SUCCESS,
            message="Authenticated",
            is_critical=True,
        )

        mock_openai.return_value = VerificationResult(
            service_name="openai",
            status=ServiceStatus.SUCCESS,
            message="Connected",
            is_critical=False,
        )

        results = await verify_all_services(demo_config_path, mock_logger, mock_console)

        assert len(results) == 3
        assert all(r.status == ServiceStatus.SUCCESS for r in results)


def test_check_critical_failures_all_pass():
    """Test check_critical_failures with all passing."""
    results = [
        VerificationResult(
            service_name="svc1",
            status=ServiceStatus.SUCCESS,
            message="OK",
            is_critical=True,
        ),
        VerificationResult(
            service_name="svc2",
            status=ServiceStatus.SUCCESS,
            message="OK",
            is_critical=True,
        ),
    ]

    assert check_critical_failures(results) is True


def test_check_critical_failures_optional_fail():
    """Test check_critical_failures with optional service failing."""
    results = [
        VerificationResult(
            service_name="svc1",
            status=ServiceStatus.SUCCESS,
            message="OK",
            is_critical=True,
        ),
        VerificationResult(
            service_name="svc2",
            status=ServiceStatus.FAILED,
            message="Failed",
            is_critical=False,  # Optional
        ),
    ]

    assert check_critical_failures(results) is True  # Still passes


def test_check_critical_failures_critical_fail():
    """Test check_critical_failures with critical service failing."""
    results = [
        VerificationResult(
            service_name="svc1",
            status=ServiceStatus.SUCCESS,
            message="OK",
            is_critical=True,
        ),
        VerificationResult(
            service_name="svc2",
            status=ServiceStatus.FAILED,
            message="Failed",
            is_critical=True,  # Critical
        ),
    ]

    assert check_critical_failures(results) is False  # Fails


# ============================================================================
# Parameterized Tests
# ============================================================================


@pytest.mark.parametrize(
    "status_code,expected_status",
    [
        (200, ServiceStatus.SUCCESS),
        (401, ServiceStatus.FAILED),
        (403, ServiceStatus.WARNING),
        (500, ServiceStatus.WARNING),
    ],
)
@pytest.mark.asyncio
async def test_pinecone_various_status_codes(
    pinecone_config: ServiceConfig,
    mock_logger,
    mock_console,
    monkeypatch,
    status_code: int,
    expected_status: ServiceStatus,
):
    """Test Pinecone verifier with various HTTP status codes."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.status_code = status_code

    if status_code == 200:
        mock_response.json.return_value = {"indexes": []}
        mock_response.headers = {}

    async with PineconeVerifier(pinecone_config, mock_logger, mock_console) as verifier:
        verifier.client.get = AsyncMock(return_value=mock_response)

        result = await verifier.verify()

        assert result.status == expected_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
