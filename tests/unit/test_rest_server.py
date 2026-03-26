"""
Tests for the REST server module.

Tests request/response models, route handlers, authentication dependency,
exception handlers, middleware, and lifespan management.
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

# rest_server.py calls get_settings() at module level for CORS config.
# We must ensure settings can be constructed before the import.
# Set env vars so Settings() won't fail on API key validation.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-unit-tests")

from src.config.settings import reset_settings  # noqa: E402

reset_settings()  # Force fresh settings with the env vars above

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.rest_server import (  # noqa: E402
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    app,
    verify_api_key,
)


# ---------------------------------------------------------------------------
# Request/Response model tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueryRequest:
    """Tests for QueryRequest model."""

    def test_defaults(self):
        req = QueryRequest(query="test query")
        assert req.query == "test query"
        assert req.use_mcts is True
        assert req.use_rag is True
        assert req.mcts_iterations is None
        assert req.thread_id is None

    def test_custom_values(self):
        req = QueryRequest(
            query="custom",
            use_mcts=False,
            use_rag=False,
            mcts_iterations=500,
            thread_id="session-123",
        )
        assert req.use_mcts is False
        assert req.use_rag is False
        assert req.mcts_iterations == 500
        assert req.thread_id == "session-123"

    def test_query_min_length(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_max_length(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 10001)

    def test_mcts_iterations_bounds(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", mcts_iterations=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", mcts_iterations=10001)

    def test_thread_id_pattern(self):
        # Valid patterns
        req = QueryRequest(query="q", thread_id="abc-123_XYZ")
        assert req.thread_id == "abc-123_XYZ"

        # Invalid pattern (spaces not allowed)
        with pytest.raises(ValidationError):
            QueryRequest(query="q", thread_id="has spaces")

    def test_thread_id_max_length(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="q", thread_id="a" * 101)


@pytest.mark.unit
class TestQueryResponse:
    """Tests for QueryResponse model."""

    def test_basic(self):
        resp = QueryResponse(
            response="answer",
            confidence=0.9,
            agents_used=["hrm", "trm"],
            processing_time_ms=150.0,
        )
        assert resp.response == "answer"
        assert resp.confidence == 0.9
        assert resp.mcts_stats is None
        assert resp.metadata == {}

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            QueryResponse(
                response="x", confidence=-0.1, agents_used=[], processing_time_ms=0.0
            )
        with pytest.raises(ValidationError):
            QueryResponse(
                response="x", confidence=1.1, agents_used=[], processing_time_ms=0.0
            )

    def test_with_mcts_stats(self):
        resp = QueryResponse(
            response="r",
            confidence=0.5,
            agents_used=[],
            mcts_stats={"iterations": 100},
            processing_time_ms=50.0,
            metadata={"key": "value"},
        )
        assert resp.mcts_stats == {"iterations": 100}
        assert resp.metadata == {"key": "value"}


@pytest.mark.unit
class TestHealthResponseRest:
    """Tests for REST HealthResponse model."""

    def test_basic(self):
        resp = HealthResponse(
            status="healthy",
            timestamp="2025-01-01T00:00:00Z",
            uptime_seconds=120.0,
        )
        assert resp.status == "healthy"
        assert resp.version == "1.0.0"


@pytest.mark.unit
class TestReadinessResponse:
    """Tests for ReadinessResponse model."""

    def test_basic(self):
        resp = ReadinessResponse(
            ready=True,
            checks={"imports": True, "auth": True},
        )
        assert resp.ready is True
        assert resp.checks["imports"] is True


@pytest.mark.unit
class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_basic(self):
        resp = ErrorResponse(
            error_code="AUTH_ERROR",
            message="Unauthorized",
            timestamp="2025-01-01T00:00:00Z",
        )
        assert resp.error is True
        assert resp.error_code == "AUTH_ERROR"


# ---------------------------------------------------------------------------
# verify_api_key dependency tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerifyApiKey:
    """Tests for the verify_api_key dependency."""

    @pytest.mark.asyncio
    @patch("src.api.rest_server.IMPORTS_AVAILABLE", False)
    async def test_imports_not_available_raises_500(self):
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="some-key")
        assert exc_info.value.status_code == 500
        assert "not available" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.get_authenticator")
    async def test_valid_key_returns_client_info(self, mock_get_auth):
        mock_auth = MagicMock()
        mock_client = MagicMock()
        mock_auth.require_auth.return_value = mock_client
        mock_get_auth.return_value = mock_auth

        result = await verify_api_key(x_api_key="valid-key")
        assert result is mock_client
        mock_auth.require_auth.assert_called_once_with("valid-key")

    @pytest.mark.asyncio
    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.get_authenticator")
    async def test_invalid_key_raises_401(self, mock_get_auth):
        from src.api.exceptions import AuthenticationError

        mock_auth = MagicMock()
        mock_auth.require_auth.side_effect = AuthenticationError(
            user_message="Invalid API key"
        )
        mock_get_auth.return_value = mock_auth

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="bad-key")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.get_settings")
    @patch("src.api.rest_server.get_authenticator")
    async def test_rate_limit_raises_429(self, mock_get_auth, mock_get_settings):
        from src.api.exceptions import RateLimitError

        mock_settings = MagicMock()
        mock_settings.RATE_LIMIT_RETRY_AFTER_SECONDS = 60
        mock_get_settings.return_value = mock_settings

        mock_auth = MagicMock()
        mock_auth.require_auth.side_effect = RateLimitError(
            user_message="Too many requests",
            retry_after_seconds=30,
        )
        mock_get_auth.return_value = mock_auth

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="key")
        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers


# ---------------------------------------------------------------------------
# Route handler tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self):
        with patch("src.api.rest_server.framework_service", None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "uptime_seconds" in data
            assert data["version"] == "1.0.0"

    def test_health_degraded_on_error_state(self):
        from src.api.framework_service import FrameworkState

        mock_service = MagicMock()
        mock_service.state = FrameworkState.ERROR

        with patch("src.api.rest_server.framework_service", mock_service):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "degraded"

    def test_health_initializing_state(self):
        from src.api.framework_service import FrameworkState

        mock_service = MagicMock()
        mock_service.state = FrameworkState.UNINITIALIZED

        with patch("src.api.rest_server.framework_service", mock_service):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "initializing"


@pytest.mark.unit
class TestReadinessEndpoint:
    """Tests for the /ready endpoint."""

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    def test_ready_all_checks_pass(self):
        mock_service = MagicMock()
        mock_service.is_ready = True

        with patch("src.api.rest_server.framework_service", mock_service):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ready"] is True
            assert data["checks"]["framework_ready"] is True

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", False)
    def test_ready_without_framework_service(self):
        with patch("src.api.rest_server.framework_service", None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["checks"]["framework_ready"] is False
            # Still ready because framework readiness is optional
            assert data["ready"] is True

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", False)
    def test_ready_returns_503_when_imports_missing(self):
        with patch("src.api.rest_server.framework_service", None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/ready")
            assert resp.status_code == 503


@pytest.mark.unit
class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    @patch("src.api.rest_server.PROMETHEUS_AVAILABLE", False)
    def test_metrics_unavailable_returns_501(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert resp.status_code == 501

    def test_metrics_endpoint_responds(self):
        """Metrics endpoint returns 501 when prometheus is not installed, or 200 when it is."""
        from src.api.rest_server import PROMETHEUS_AVAILABLE

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        if PROMETHEUS_AVAILABLE:
            assert resp.status_code == 200
        else:
            assert resp.status_code == 501


@pytest.mark.unit
class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def _make_client_with_auth(self, mock_client_info=None):
        """Create a TestClient with auth dependency overridden."""
        if mock_client_info is None:
            mock_client_info = MagicMock()
            mock_client_info.client_id = "test-client"
            mock_client_info.roles = {"user"}

        app.dependency_overrides[verify_api_key] = lambda: mock_client_info
        client = TestClient(app, raise_server_exceptions=False)
        return client

    def teardown_method(self):
        app.dependency_overrides.clear()

    def test_query_missing_api_key_returns_422(self):
        """Without override, missing X-API-Key header returns 422."""
        app.dependency_overrides.clear()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 422

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_success(self, mock_fw_service):
        mock_result = MagicMock()
        mock_result.response = "The answer"
        mock_result.confidence = 0.85
        mock_result.agents_used = ["hrm", "trm"]
        mock_result.mcts_stats = {"iterations": 100}
        mock_result.processing_time_ms = 250.0
        mock_result.metadata = {}
        mock_fw_service.process_query = AsyncMock(return_value=mock_result)

        client = self._make_client_with_auth()
        resp = client.post("/query", json={"query": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "The answer"
        assert data["confidence"] == 0.85

    @patch("src.api.rest_server.framework_service", None)
    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    def test_query_no_framework_returns_503(self):
        client = self._make_client_with_auth()
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 503
        assert "not available" in resp.json()["detail"]

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_timeout_returns_504(self, mock_fw_service):
        mock_fw_service.process_query = AsyncMock(
            side_effect=TimeoutError("timed out")
        )

        client = self._make_client_with_auth()
        resp = client.post("/query", json={"query": "slow query"})
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_value_error_returns_400(self, mock_fw_service):
        mock_fw_service.process_query = AsyncMock(
            side_effect=ValueError("bad input")
        )

        client = self._make_client_with_auth()
        resp = client.post("/query", json={"query": "bad query"})
        assert resp.status_code == 400

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_runtime_error_returns_503(self, mock_fw_service):
        mock_fw_service.process_query = AsyncMock(
            side_effect=RuntimeError("broken")
        )

        client = self._make_client_with_auth()
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 503

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_unexpected_error_returns_500(self, mock_fw_service):
        mock_fw_service.process_query = AsyncMock(
            side_effect=Exception("unexpected")
        )

        client = self._make_client_with_auth()
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 500

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.QueryInput")
    @patch("src.api.rest_server.framework_service", None)
    def test_query_validation_failure_returns_400(self, mock_query_input):
        mock_query_input.side_effect = Exception("validation failed")

        client = self._make_client_with_auth()
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 400
        assert "Validation failed" in resp.json()["detail"]


@pytest.mark.unit
class TestStatsEndpoint:
    """Tests for the /stats endpoint."""

    def teardown_method(self):
        app.dependency_overrides.clear()

    @patch("src.api.rest_server.get_authenticator")
    def test_stats_returns_client_info(self, mock_get_auth):
        mock_auth = MagicMock()
        mock_auth.get_client_stats.return_value = {"total_requests": 10}
        mock_auth.rate_limit_config.requests_per_minute = 60
        mock_auth.rate_limit_config.requests_per_hour = 3600
        mock_auth.rate_limit_config.requests_per_day = 86400
        mock_get_auth.return_value = mock_auth

        mock_client = MagicMock()
        mock_client.client_id = "client-1"
        mock_client.roles = {"user", "admin"}

        app.dependency_overrides[verify_api_key] = lambda: mock_client
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["client_id"] == "client-1"
        assert data["total_requests"] == 10
        assert "rate_limits" in data
        assert data["rate_limits"]["per_minute"] == 60


@pytest.mark.unit
class TestExceptionHandlers:
    """Tests for custom exception handlers."""

    def test_framework_error_handler(self):
        from src.api.exceptions import FrameworkError

        # Register a test route that raises FrameworkError
        @app.get("/test-framework-error")
        async def _raise_framework_error():
            raise FrameworkError(user_message="Something failed", error_code="TEST_ERR")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-framework-error")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] is True
        assert data["error_code"] == "TEST_ERR"
        assert data["message"] == "Something failed"

        # Clean up the test route
        app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-framework-error"]

    def test_validation_error_handler(self):
        from src.api.exceptions import ValidationError as FwValidationError

        @app.get("/test-validation-error")
        async def _raise_validation_error():
            raise FwValidationError(user_message="Invalid field", field_name="query")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-validation-error")
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] is True
        assert data["error_code"] == "VALIDATION_ERROR"

        # Clean up
        app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-validation-error"]


@pytest.mark.unit
class TestMetricsMiddleware:
    """Tests for the metrics middleware."""

    @patch("src.api.rest_server.PROMETHEUS_AVAILABLE", False)
    def test_middleware_works_without_prometheus(self):
        """Middleware should not fail when prometheus is not available."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
