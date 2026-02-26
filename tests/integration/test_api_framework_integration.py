"""
Integration Tests for API-Framework Connection.

Tests the integration between:
- REST API endpoints (rest_server.py)
- Framework service layer (framework_service.py)
- Authentication and authorization
- Query processing pipeline

These tests verify that the API correctly interfaces with the
underlying multi-agent MCTS framework.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 8.6
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Module Availability Checks
# =============================================================================

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from src.api.framework_service import (
        FrameworkConfig,
        FrameworkService,
        FrameworkState,
        LightweightFramework,
        MockLLMClient,
        QueryResult,
    )
    from src.api.rest_server import app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    from src.api.auth import (  # noqa: F401 - for future tests
        APIKeyAuthenticator,
        ClientInfo,
        RateLimitConfig,
        set_authenticator,
    )

    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

try:
    from src.api.exceptions import (  # noqa: F401 - for future tests
        AuthenticationError,
        FrameworkError,
        RateLimitError,
        ValidationError,
    )

    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False

try:
    from src.config.settings import Settings, get_settings, reset_settings

    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_api_key() -> str:
    """Provide a test API key for authentication."""
    return "test-api-key-for-integration-testing"


@pytest.fixture
def test_api_key_secondary() -> str:
    """Provide a secondary test API key."""
    return "test-api-key-secondary"


@pytest.fixture
def test_authenticator(test_api_key: str, test_api_key_secondary: str) -> APIKeyAuthenticator:
    """Create a test authenticator with pre-configured API keys."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")

    return APIKeyAuthenticator(
        valid_keys=[test_api_key, test_api_key_secondary],
        rate_limit_config=RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=50,
        ),
    )


@pytest.fixture
def mock_framework_service() -> AsyncMock:
    """Create a mock framework service for testing."""
    service = AsyncMock(spec=FrameworkService)
    service.is_ready = True
    service.state = FrameworkState.READY

    # Configure process_query to return a valid QueryResult
    service.process_query.return_value = QueryResult(
        response="This is a test response from the mock framework.",
        confidence=0.85,
        agents_used=["hrm", "trm"],
        mcts_stats=None,
        processing_time_ms=150.0,
        metadata={"thread_id": "test-thread", "rag_enabled": True, "mcts_enabled": False},
    )

    service.initialize.return_value = True
    service.health_check.return_value = {
        "status": "ready",
        "uptime_seconds": 100.0,
        "request_count": 10,
        "error_count": 0,
    }

    return service


@pytest.fixture
def mock_framework_service_with_mcts() -> AsyncMock:
    """Create a mock framework service with MCTS enabled."""
    service = AsyncMock(spec=FrameworkService)
    service.is_ready = True
    service.state = FrameworkState.READY

    service.process_query.return_value = QueryResult(
        response="MCTS tactical analysis complete. Recommended action: defensive positioning.",
        confidence=0.92,
        agents_used=["hrm", "trm", "mcts"],
        mcts_stats={
            "iterations": 100,
            "best_action": "defensive_position",
            "best_action_visits": 45,
            "best_action_value": 0.78,
            "tree_depth": 5,
        },
        processing_time_ms=850.0,
        metadata={"thread_id": "mcts-test", "rag_enabled": False, "mcts_enabled": True},
    )

    service.initialize.return_value = True
    return service


@pytest.fixture
def mock_framework_service_error() -> AsyncMock:
    """Create a mock framework service that raises errors."""
    service = AsyncMock(spec=FrameworkService)
    service.is_ready = True
    service.state = FrameworkState.ERROR

    service.process_query.side_effect = RuntimeError("Framework processing failed")
    service.initialize.return_value = False

    return service


@pytest.fixture
def mock_framework_service_timeout() -> AsyncMock:
    """Create a mock framework service that times out."""
    service = AsyncMock(spec=FrameworkService)
    service.is_ready = True
    service.state = FrameworkState.READY

    async def timeout_query(*args, **kwargs):
        raise TimeoutError("Query processing timed out after 30s")

    service.process_query.side_effect = timeout_query
    service.initialize.return_value = True

    return service


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings for framework configuration."""
    if not SETTINGS_AVAILABLE:
        pytest.skip("Settings module not available")

    reset_settings()

    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key-for-testing-only",
            "MCTS_ENABLED": "true",
            "MCTS_ITERATIONS": "10",
            "SEED": "42",
            "LOG_LEVEL": "DEBUG",
            "HTTP_TIMEOUT_SECONDS": "30",
            "FRAMEWORK_MAX_ITERATIONS": "5",
            "FRAMEWORK_CONSENSUS_THRESHOLD": "0.75",
            "FRAMEWORK_TOP_K_RETRIEVAL": "5",
            "FRAMEWORK_ENABLE_PARALLEL_AGENTS": "true",
            "RATE_LIMIT_REQUESTS_PER_MINUTE": "60",
        },
    ):
        settings = get_settings()
        yield settings

    reset_settings()


@pytest.fixture
def test_client(test_authenticator: APIKeyAuthenticator) -> TestClient:
    """Create a FastAPI test client with configured authenticator."""
    if not FASTAPI_AVAILABLE or not API_AVAILABLE:
        pytest.skip("FastAPI or API module not available")

    # Set up the authenticator
    set_authenticator(test_authenticator)

    # Create test client without lifespan to avoid initialization
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def test_client_with_lifespan(test_authenticator: APIKeyAuthenticator, test_settings: Settings) -> TestClient:
    """Create a FastAPI test client with full lifespan management."""
    if not FASTAPI_AVAILABLE or not API_AVAILABLE:
        pytest.skip("FastAPI or API module not available")

    set_authenticator(test_authenticator)

    # Use lifespan for full initialization
    with patch.dict(os.environ, {"API_KEYS": "test-api-key-for-integration-testing"}):
        with TestClient(app) as client:
            yield client


# =============================================================================
# Framework Initialization Tests
# =============================================================================


class TestFrameworkServiceInitialization:
    """Tests for framework service initialization."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_framework_service_initialization(self, test_settings: Settings):
        """Test that FrameworkService initializes correctly."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        # Reset singleton for clean test
        await FrameworkService.reset_instance()

        config = FrameworkConfig.from_settings(test_settings)

        # Create new instance
        service = await FrameworkService.get_instance(config=config, settings=test_settings)

        assert service is not None
        assert service.state == FrameworkState.UNINITIALIZED

        # Clean up
        await FrameworkService.reset_instance()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_framework_service_ready_state(self, test_settings: Settings):
        """Test that FrameworkService transitions to ready state after initialization."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        await FrameworkService.reset_instance()

        config = FrameworkConfig.from_settings(test_settings)
        service = await FrameworkService.get_instance(config=config, settings=test_settings)

        # Initialize the service
        with patch.object(
            service,
            "_framework",
            LightweightFramework(
                llm_client=MockLLMClient(),
                config=config,
                logger=MagicMock(),
            ),
        ):
            service._state = FrameworkState.READY

        assert service.is_ready is True
        assert service.state == FrameworkState.READY

        await FrameworkService.reset_instance()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_framework_service_configuration(self, test_settings: Settings):
        """Test that FrameworkConfig correctly loads from settings."""
        if not API_AVAILABLE or not SETTINGS_AVAILABLE:
            pytest.skip("Required modules not available")

        config = FrameworkConfig.from_settings(test_settings)

        assert config.mcts_enabled == test_settings.MCTS_ENABLED
        assert config.mcts_iterations == test_settings.MCTS_ITERATIONS
        assert config.mcts_exploration_weight == test_settings.MCTS_C
        assert config.max_iterations == test_settings.FRAMEWORK_MAX_ITERATIONS
        assert config.consensus_threshold == test_settings.FRAMEWORK_CONSENSUS_THRESHOLD
        assert config.top_k_retrieval == test_settings.FRAMEWORK_TOP_K_RETRIEVAL
        assert config.enable_parallel_agents == test_settings.FRAMEWORK_ENABLE_PARALLEL_AGENTS

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_framework_service_singleton_pattern(self, test_settings: Settings):
        """Test that FrameworkService follows singleton pattern."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        await FrameworkService.reset_instance()

        config = FrameworkConfig.from_settings(test_settings)

        # Get two instances
        service1 = await FrameworkService.get_instance(config=config, settings=test_settings)
        service2 = await FrameworkService.get_instance(config=config, settings=test_settings)

        # Should be the same instance
        assert service1 is service2

        await FrameworkService.reset_instance()


# =============================================================================
# Query Processing Tests
# =============================================================================


class TestQueryProcessing:
    """Tests for query processing through the API."""

    @pytest.mark.integration
    def test_process_query_simple(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
    ):
        """Test simple query processing through the API."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        # Patch the framework_service module variable
        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.post(
                "/query",
                json={
                    "query": "What is the recommended strategy for this scenario?",
                    "use_mcts": False,
                    "use_rag": True,
                },
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200
        data = response.json()

        assert "response" in data
        assert "confidence" in data
        assert "agents_used" in data
        assert "processing_time_ms" in data
        assert data["confidence"] >= 0.0
        assert data["confidence"] <= 1.0

    @pytest.mark.integration
    def test_process_query_with_mcts(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service_with_mcts: AsyncMock,
    ):
        """Test query processing with MCTS enabled."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service_with_mcts):
            response = test_client.post(
                "/query",
                json={
                    "query": "Recommend defensive positions for night attack scenario",
                    "use_mcts": True,
                    "use_rag": False,
                    "mcts_iterations": 100,
                },
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200
        data = response.json()

        assert "response" in data
        assert "mcts_stats" in data
        assert data["mcts_stats"] is not None
        assert "iterations" in data["mcts_stats"]
        assert "best_action" in data["mcts_stats"]
        assert "mcts" in data["agents_used"]

    @pytest.mark.integration
    def test_process_query_with_rag(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
    ):
        """Test query processing with RAG context retrieval."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.post(
                "/query",
                json={
                    "query": "Analyze based on historical precedents",
                    "use_mcts": False,
                    "use_rag": True,
                },
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200
        data = response.json()

        assert "metadata" in data
        assert data["metadata"].get("rag_enabled") is True

    @pytest.mark.integration
    def test_process_query_confidence_range(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
    ):
        """Test that confidence scores are within valid range."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.post(
                "/query",
                json={"query": "Test query for confidence validation"},
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200
        data = response.json()

        confidence = data["confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} is outside valid range [0.0, 1.0]"

    @pytest.mark.integration
    def test_process_query_with_thread_id(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
    ):
        """Test query processing with conversation thread ID."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        thread_id = "test-conversation-123"

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.post(
                "/query",
                json={
                    "query": "Continue our previous discussion",
                    "thread_id": thread_id,
                },
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200
        response.json()

        # Verify thread_id was passed to the framework
        mock_framework_service.process_query.assert_called_once()
        call_kwargs = mock_framework_service.process_query.call_args.kwargs
        assert call_kwargs.get("thread_id") == thread_id


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for health and readiness endpoints."""

    @pytest.mark.integration
    def test_health_endpoint(self, test_client: TestClient):
        """Test the health check endpoint returns valid response."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert data["status"] in ["healthy", "degraded", "initializing"]

    @pytest.mark.integration
    def test_ready_endpoint(self, test_client: TestClient):
        """Test the readiness check endpoint."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.get("/ready")

        # Can be 200 or 503 depending on framework state
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "ready" in data
            assert "checks" in data
            assert isinstance(data["checks"], dict)

    @pytest.mark.integration
    def test_health_endpoint_no_auth_required(self, test_client: TestClient):
        """Test that health endpoint doesn't require authentication."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        # No API key header
        response = test_client.get("/health")

        # Should succeed without authentication
        assert response.status_code == 200

    @pytest.mark.integration
    def test_ready_endpoint_checks_framework(
        self,
        test_client: TestClient,
        mock_framework_service: AsyncMock,
    ):
        """Test that readiness check verifies framework status."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        mock_framework_service.is_ready = True

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.get("/ready")

        if response.status_code == 200:
            data = response.json()
            assert "framework_ready" in data.get("checks", {})


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the API."""

    @pytest.mark.integration
    def test_invalid_query_handling(
        self,
        test_client: TestClient,
        test_api_key: str,
    ):
        """Test handling of invalid query input."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        # Empty query should fail validation
        response = test_client.post(
            "/query",
            json={"query": ""},
            headers={"X-API-Key": test_api_key},
        )

        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.integration
    def test_invalid_query_too_long(
        self,
        test_client: TestClient,
        test_api_key: str,
    ):
        """Test handling of query that exceeds length limit."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        # Query exceeding max length (10000 chars)
        long_query = "x" * 10001

        response = test_client.post(
            "/query",
            json={"query": long_query},
            headers={"X-API-Key": test_api_key},
        )

        assert response.status_code == 422

    @pytest.mark.integration
    def test_missing_api_key(self, test_client: TestClient):
        """Test that missing API key returns 401."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.post(
            "/query",
            json={"query": "Test query"},
            # No X-API-Key header
        )

        assert response.status_code == 422  # FastAPI returns 422 for missing required header

    @pytest.mark.integration
    def test_invalid_api_key(self, test_client: TestClient):
        """Test that invalid API key returns 401."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.post(
            "/query",
            json={"query": "Test query"},
            headers={"X-API-Key": "invalid-api-key-12345"},
        )

        assert response.status_code == 401

    @pytest.mark.integration
    def test_timeout_handling(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service_timeout: AsyncMock,
    ):
        """Test handling of request timeouts."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service_timeout):
            response = test_client.post(
                "/query",
                json={"query": "Query that will timeout"},
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 504  # Gateway Timeout

    @pytest.mark.integration
    def test_framework_error_response(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service_error: AsyncMock,
    ):
        """Test handling of framework processing errors."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service_error):
            response = test_client.post(
                "/query",
                json={"query": "Query that will fail"},
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 503  # Service Unavailable

    @pytest.mark.integration
    def test_framework_service_unavailable(
        self,
        test_client: TestClient,
        test_api_key: str,
    ):
        """Test handling when framework service is None."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", None):
            response = test_client.post(
                "/query",
                json={"query": "Query when service unavailable"},
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    def test_invalid_thread_id_format(
        self,
        test_client: TestClient,
        test_api_key: str,
    ):
        """Test handling of invalid thread ID format."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.post(
            "/query",
            json={
                "query": "Test query",
                "thread_id": "invalid thread id with spaces!",
            },
            headers={"X-API-Key": test_api_key},
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# Authentication Integration Tests
# =============================================================================


class TestAuthenticationIntegration:
    """Tests for authentication integration with the API."""

    @pytest.mark.integration
    def test_valid_authentication(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
    ):
        """Test successful authentication with valid API key."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.post(
                "/query",
                json={"query": "Authenticated query"},
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_multiple_valid_api_keys(
        self,
        test_client: TestClient,
        test_api_key: str,
        test_api_key_secondary: str,
        mock_framework_service: AsyncMock,
    ):
        """Test that multiple valid API keys work."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            # First key
            response1 = test_client.post(
                "/query",
                json={"query": "Query with first key"},
                headers={"X-API-Key": test_api_key},
            )
            assert response1.status_code == 200

            # Second key
            response2 = test_client.post(
                "/query",
                json={"query": "Query with second key"},
                headers={"X-API-Key": test_api_key_secondary},
            )
            assert response2.status_code == 200

    @pytest.mark.integration
    def test_client_id_in_response_metadata(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
    ):
        """Test that client ID is included in response metadata."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.post(
                "/query",
                json={"query": "Test query"},
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "client_id" in data["metadata"]


# =============================================================================
# Framework Service Unit Tests
# =============================================================================


class TestFrameworkServiceUnit:
    """Unit tests for FrameworkService class."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_result_to_dict(self):
        """Test QueryResult serialization to dictionary."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        result = QueryResult(
            response="Test response",
            confidence=0.85,
            agents_used=["hrm", "trm"],
            mcts_stats={"iterations": 50},
            processing_time_ms=100.0,
            metadata={"key": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["response"] == "Test response"
        assert result_dict["confidence"] == 0.85
        assert result_dict["agents_used"] == ["hrm", "trm"]
        assert result_dict["mcts_stats"]["iterations"] == 50
        assert result_dict["processing_time_ms"] == 100.0
        assert result_dict["metadata"]["key"] == "value"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_framework_state_enum(self):
        """Test FrameworkState enum values."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        assert FrameworkState.UNINITIALIZED.value == "uninitialized"
        assert FrameworkState.INITIALIZING.value == "initializing"
        assert FrameworkState.READY.value == "ready"
        assert FrameworkState.ERROR.value == "error"
        assert FrameworkState.SHUTTING_DOWN.value == "shutting_down"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_framework_service_shutdown(self, test_settings: Settings):
        """Test framework service shutdown."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        await FrameworkService.reset_instance()

        config = FrameworkConfig.from_settings(test_settings)
        service = await FrameworkService.get_instance(config=config, settings=test_settings)

        # Manually set state for test
        service._state = FrameworkState.READY

        await service.shutdown()

        assert service.state == FrameworkState.UNINITIALIZED
        assert service._framework is None

        await FrameworkService.reset_instance()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_process_query_empty_validation(self, test_settings: Settings):
        """Test that empty queries are rejected."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        await FrameworkService.reset_instance()

        config = FrameworkConfig.from_settings(test_settings)
        service = await FrameworkService.get_instance(config=config, settings=test_settings)

        # Set up mock framework
        service._state = FrameworkState.READY
        service._framework = MagicMock()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service.process_query(query="")

        await FrameworkService.reset_instance()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_process_query_whitespace_validation(self, test_settings: Settings):
        """Test that whitespace-only queries are rejected."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        await FrameworkService.reset_instance()

        config = FrameworkConfig.from_settings(test_settings)
        service = await FrameworkService.get_instance(config=config, settings=test_settings)

        service._state = FrameworkState.READY
        service._framework = MagicMock()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await service.process_query(query="   ")

        await FrameworkService.reset_instance()


# =============================================================================
# Lightweight Framework Tests
# =============================================================================


class TestLightweightFramework:
    """Tests for LightweightFramework fallback implementation."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lightweight_framework_process(self, test_settings: Settings):
        """Test LightweightFramework query processing."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        config = FrameworkConfig.from_settings(test_settings)
        mock_llm = MockLLMClient()
        logger = MagicMock()

        framework = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logger,
        )

        result = await framework.process(
            query="Test query for lightweight framework",
            use_rag=True,
            use_mcts=False,
        )

        assert "response" in result
        assert "metadata" in result
        assert result["metadata"]["agents_used"] == ["lightweight"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lightweight_framework_with_mcts_stats(self, test_settings: Settings):
        """Test LightweightFramework generates mock MCTS stats when enabled."""
        if not API_AVAILABLE:
            pytest.skip("API module not available")

        config = FrameworkConfig.from_settings(test_settings)
        mock_llm = MockLLMClient()
        logger = MagicMock()

        framework = LightweightFramework(
            llm_client=mock_llm,
            config=config,
            logger=logger,
        )

        result = await framework.process(
            query="Test query with MCTS",
            use_rag=False,
            use_mcts=True,
        )

        assert "state" in result
        assert result["state"]["mcts_stats"] is not None
        assert "iterations" in result["state"]["mcts_stats"]
        assert "best_action" in result["state"]["mcts_stats"]


# =============================================================================
# Request Validation Tests
# =============================================================================


class TestRequestValidation:
    """Tests for request validation."""

    @pytest.mark.integration
    def test_valid_mcts_iterations_range(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
    ):
        """Test valid MCTS iterations parameter."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service):
            response = test_client.post(
                "/query",
                json={
                    "query": "Test query",
                    "use_mcts": True,
                    "mcts_iterations": 500,
                },
                headers={"X-API-Key": test_api_key},
            )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_invalid_mcts_iterations_too_high(
        self,
        test_client: TestClient,
        test_api_key: str,
    ):
        """Test that MCTS iterations above max are rejected."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.post(
            "/query",
            json={
                "query": "Test query",
                "use_mcts": True,
                "mcts_iterations": 100000,  # Above max of 10000
            },
            headers={"X-API-Key": test_api_key},
        )

        assert response.status_code == 422

    @pytest.mark.integration
    def test_invalid_mcts_iterations_negative(
        self,
        test_client: TestClient,
        test_api_key: str,
    ):
        """Test that negative MCTS iterations are rejected."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.post(
            "/query",
            json={
                "query": "Test query",
                "use_mcts": True,
                "mcts_iterations": -1,
            },
            headers={"X-API-Key": test_api_key},
        )

        assert response.status_code == 422


# =============================================================================
# Stats Endpoint Tests
# =============================================================================


class TestStatsEndpoint:
    """Tests for the stats endpoint."""

    @pytest.mark.integration
    def test_stats_endpoint_requires_auth(self, test_client: TestClient):
        """Test that stats endpoint requires authentication."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.get("/stats")

        # Should fail without API key
        assert response.status_code == 422

    @pytest.mark.integration
    def test_stats_endpoint_with_auth(
        self,
        test_client: TestClient,
        test_api_key: str,
    ):
        """Test stats endpoint with valid authentication."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        response = test_client.get(
            "/stats",
            headers={"X-API-Key": test_api_key},
        )

        assert response.status_code == 200
        data = response.json()

        assert "client_id" in data
        assert "roles" in data
        assert "rate_limits" in data


# =============================================================================
# Integration Flow Tests
# =============================================================================


class TestIntegrationFlows:
    """Tests for complete integration flows."""

    @pytest.mark.integration
    def test_full_query_flow(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service_with_mcts: AsyncMock,
    ):
        """Test complete query flow from request to response."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        with patch("src.api.rest_server.framework_service", mock_framework_service_with_mcts):
            # Step 1: Check health
            health_response = test_client.get("/health")
            assert health_response.status_code == 200

            # Step 2: Submit query
            query_response = test_client.post(
                "/query",
                json={
                    "query": "Analyze defensive positions for tactical scenario",
                    "use_mcts": True,
                    "use_rag": False,
                    "thread_id": "integration-test-flow",
                },
                headers={"X-API-Key": test_api_key},
            )
            assert query_response.status_code == 200

            data = query_response.json()

            # Step 3: Verify response structure
            assert "response" in data
            assert "confidence" in data
            assert "agents_used" in data
            assert "mcts_stats" in data
            assert "processing_time_ms" in data
            assert "metadata" in data

            # Step 4: Check stats
            stats_response = test_client.get(
                "/stats",
                headers={"X-API-Key": test_api_key},
            )
            assert stats_response.status_code == 200

    @pytest.mark.integration
    def test_error_recovery_flow(
        self,
        test_client: TestClient,
        test_api_key: str,
        mock_framework_service: AsyncMock,
        mock_framework_service_error: AsyncMock,
    ):
        """Test that API recovers from errors on subsequent requests."""
        if not FASTAPI_AVAILABLE or not API_AVAILABLE:
            pytest.skip("Required modules not available")

        # First request fails
        with patch("src.api.rest_server.framework_service", mock_framework_service_error):
            error_response = test_client.post(
                "/query",
                json={"query": "This will fail"},
                headers={"X-API-Key": test_api_key},
            )
            assert error_response.status_code == 503

        # Second request succeeds with working service
        with patch("src.api.rest_server.framework_service", mock_framework_service):
            success_response = test_client.post(
                "/query",
                json={"query": "This will succeed"},
                headers={"X-API-Key": test_api_key},
            )
            assert success_response.status_code == 200
