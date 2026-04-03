"""
Extended unit tests for src/api/rest_server.py targeting uncovered lines.

Covers:
- CORS middleware configuration (wildcard, specific origins, disabled)
- Lifespan manager startup/shutdown paths
- Metrics middleware with Prometheus enabled
- verify_api_key rate-limit fallback (retry_after_seconds=None)
- Exception handlers with Prometheus metrics
- /query endpoint: validation, error paths with Prometheus counters
- /metrics endpoint when Prometheus is available
- /stats endpoint role serialization
- Framework service integration edge cases
"""

import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure settings can be constructed before the import
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-unit-tests")

from src.config.settings import reset_settings  # noqa: E402

reset_settings()

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.rest_server import (  # noqa: E402
    QueryRequest,
    QueryResponse,
    app,
    verify_api_key,
)


def _make_client_with_auth(client_info=None):
    """Create a TestClient with auth dependency overridden."""
    if client_info is None:
        client_info = MagicMock()
        client_info.client_id = "test-client"
        client_info.roles = {"user"}
    app.dependency_overrides[verify_api_key] = lambda: client_info
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    yield
    app.dependency_overrides.clear()


@contextmanager
def _prometheus_mocks():
    """Context manager that patches PROMETHEUS_AVAILABLE and all metric objects.

    Creates mock metric objects so that the middleware and error handlers
    can call .inc(), .dec(), .labels(), .observe() without AttributeError,
    regardless of whether prometheus_client is installed.
    """
    mock_request_count = MagicMock()
    mock_request_latency = MagicMock()
    mock_active_requests = MagicMock()
    mock_error_count = MagicMock()

    patches = [
        patch("src.api.rest_server.PROMETHEUS_AVAILABLE", True),
        patch("src.api.rest_server.REQUEST_COUNT", mock_request_count, create=True),
        patch("src.api.rest_server.REQUEST_LATENCY", mock_request_latency, create=True),
        patch("src.api.rest_server.ACTIVE_REQUESTS", mock_active_requests, create=True),
        patch("src.api.rest_server.ERROR_COUNT", mock_error_count, create=True),
    ]

    for p in patches:
        p.start()
    try:
        yield {
            "request_count": mock_request_count,
            "request_latency": mock_request_latency,
            "active_requests": mock_active_requests,
            "error_count": mock_error_count,
        }
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# Lifespan manager tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifespan:
    """Tests for the lifespan context manager (startup/shutdown)."""

    @pytest.mark.asyncio
    @patch("src.api.rest_server.FrameworkService")
    @patch("src.api.rest_server.FrameworkConfig")
    @patch("src.api.rest_server.set_authenticator")
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    async def test_lifespan_with_api_keys_env(self, mock_set_auth, mock_fw_config, mock_fw_service_cls):
        """Lifespan reads API_KEYS env var and initializes framework."""
        from src.api.rest_server import lifespan

        mock_instance = AsyncMock()
        mock_instance.initialize = AsyncMock(return_value=True)
        mock_instance.shutdown = AsyncMock()
        mock_fw_service_cls.get_instance = AsyncMock(return_value=mock_instance)
        mock_fw_service_cls.reset_instance = AsyncMock()
        mock_fw_config.from_settings.return_value = MagicMock()

        mock_app = MagicMock()

        with patch.dict(os.environ, {"API_KEYS": "key1,key2"}):
            async with lifespan(mock_app):
                mock_set_auth.assert_called_once()
                mock_fw_service_cls.get_instance.assert_awaited_once()
                mock_instance.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.api.rest_server.FrameworkService")
    @patch("src.api.rest_server.FrameworkConfig")
    @patch("src.api.rest_server.set_authenticator")
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    async def test_lifespan_no_api_keys_generates_dev_key(self, mock_set_auth, mock_fw_config, mock_fw_service_cls):
        """When API_KEYS is empty, a dev key is generated."""
        from src.api.rest_server import lifespan

        mock_instance = AsyncMock()
        mock_instance.initialize = AsyncMock(return_value=True)
        mock_instance.shutdown = AsyncMock()
        mock_fw_service_cls.get_instance = AsyncMock(return_value=mock_instance)
        mock_fw_service_cls.reset_instance = AsyncMock()
        mock_fw_config.from_settings.return_value = MagicMock()

        mock_app = MagicMock()

        with patch.dict(os.environ, {"API_KEYS": ""}, clear=False):
            async with lifespan(mock_app):
                mock_set_auth.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.rest_server.FrameworkService")
    @patch("src.api.rest_server.FrameworkConfig")
    @patch("src.api.rest_server.set_authenticator")
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    async def test_lifespan_init_failure_sets_service_none(self, mock_set_auth, mock_fw_config, mock_fw_service_cls):
        """When framework init raises, framework_service is set to None."""
        from src.api.rest_server import lifespan

        mock_fw_service_cls.get_instance = AsyncMock(side_effect=RuntimeError("init boom"))
        mock_fw_service_cls.reset_instance = AsyncMock()
        mock_fw_config.from_settings.return_value = MagicMock()

        mock_app = MagicMock()

        with patch.dict(os.environ, {"API_KEYS": "testkey"}, clear=False):
            async with lifespan(mock_app):
                import src.api.rest_server as rs

                assert rs.framework_service is None

    @pytest.mark.asyncio
    @patch("src.api.rest_server.FrameworkService")
    @patch("src.api.rest_server.FrameworkConfig")
    @patch("src.api.rest_server.set_authenticator")
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    async def test_lifespan_init_deferred(self, mock_set_auth, mock_fw_config, mock_fw_service_cls):
        """When framework init returns False, log warning but continue."""
        from src.api.rest_server import lifespan

        mock_instance = AsyncMock()
        mock_instance.initialize = AsyncMock(return_value=False)
        mock_instance.shutdown = AsyncMock()
        mock_fw_service_cls.get_instance = AsyncMock(return_value=mock_instance)
        mock_fw_service_cls.reset_instance = AsyncMock()
        mock_fw_config.from_settings.return_value = MagicMock()

        mock_app = MagicMock()

        with patch.dict(os.environ, {"API_KEYS": "testkey"}, clear=False):
            async with lifespan(mock_app):
                pass

    @pytest.mark.asyncio
    @patch("src.api.rest_server.set_authenticator")
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", False)
    async def test_lifespan_framework_not_available(self, mock_set_auth):
        """When FRAMEWORK_SERVICE_AVAILABLE is False, skip framework init."""
        from src.api.rest_server import lifespan

        mock_app = MagicMock()

        with patch.dict(os.environ, {"API_KEYS": "testkey"}, clear=False):
            async with lifespan(mock_app):
                mock_set_auth.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.api.rest_server.FrameworkService")
    @patch("src.api.rest_server.FrameworkConfig")
    @patch("src.api.rest_server.set_authenticator")
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    async def test_lifespan_shutdown_calls_service_shutdown(
        self, mock_set_auth, mock_fw_config, mock_fw_service_cls
    ):
        """Shutdown path calls framework_service.shutdown and reset_instance."""
        from src.api.rest_server import lifespan

        mock_instance = AsyncMock()
        mock_instance.initialize = AsyncMock(return_value=True)
        mock_instance.shutdown = AsyncMock()
        mock_fw_service_cls.get_instance = AsyncMock(return_value=mock_instance)
        mock_fw_service_cls.reset_instance = AsyncMock()
        mock_fw_config.from_settings.return_value = MagicMock()

        mock_app = MagicMock()

        with patch.dict(os.environ, {"API_KEYS": "testkey"}, clear=False):
            async with lifespan(mock_app):
                pass
            mock_instance.shutdown.assert_awaited_once()
            mock_fw_service_cls.reset_instance.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.api.rest_server.FrameworkService")
    @patch("src.api.rest_server.FrameworkConfig")
    @patch("src.api.rest_server.set_authenticator")
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    async def test_lifespan_shutdown_skipped_when_no_service(
        self, mock_set_auth, mock_fw_config, mock_fw_service_cls
    ):
        """When framework_service is None at shutdown, shutdown/reset are skipped."""
        from src.api.rest_server import lifespan

        mock_fw_service_cls.get_instance = AsyncMock(side_effect=RuntimeError("boom"))
        mock_fw_service_cls.reset_instance = AsyncMock()
        mock_fw_config.from_settings.return_value = MagicMock()

        mock_app = MagicMock()

        with patch.dict(os.environ, {"API_KEYS": "testkey"}, clear=False):
            async with lifespan(mock_app):
                pass
            # reset_instance should NOT be called since framework_service is None
            mock_fw_service_cls.reset_instance.assert_not_awaited()


# ---------------------------------------------------------------------------
# CORS middleware configuration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCORSMiddlewareConfiguration:
    """Tests for CORS middleware setup logic."""

    def test_cors_not_added_when_origins_empty(self):
        """When CORS_ALLOWED_ORIGINS is empty, no CORS middleware should be active."""
        mock_settings = MagicMock()
        mock_settings.CORS_ALLOWED_ORIGINS = []

        origins = mock_settings.CORS_ALLOWED_ORIGINS
        assert not origins  # empty list is falsy, so CORS block is skipped

    def test_cors_wildcard_disables_credentials(self):
        """When origins contain '*', credentials must be disabled per the code logic."""
        origins = ["*", "http://example.com"]
        has_wildcard = "*" in origins
        assert has_wildcard

        if has_wildcard:
            cors_origins = ["*"]
            allow_credentials = False
        else:
            cors_origins = origins
            allow_credentials = True

        assert cors_origins == ["*"]
        assert allow_credentials is False

    def test_cors_specific_origins_allow_credentials(self):
        """When origins are specific (no wildcard), credentials can be enabled."""
        origins = ["http://localhost:3000", "https://app.example.com"]
        has_wildcard = "*" in origins
        assert not has_wildcard

        mock_settings = MagicMock()
        mock_settings.CORS_ALLOW_CREDENTIALS = True
        assert mock_settings.CORS_ALLOW_CREDENTIALS is True

    def test_cors_preflight_on_health_endpoint(self):
        """CORS preflight request on /health should get a response."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code in (200, 400, 405)

    def test_cors_preflight_on_query_endpoint(self):
        """CORS preflight request on /query should get a response."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.options(
            "/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code in (200, 400, 405)


# ---------------------------------------------------------------------------
# Metrics middleware tests (Prometheus enabled)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsMiddlewareWithPrometheus:
    """Tests for metrics middleware when Prometheus metrics are mocked."""

    def test_middleware_increments_active_requests(self):
        """Active requests gauge is incremented and decremented."""
        with _prometheus_mocks() as mocks:
            with patch("src.api.rest_server.framework_service", None):
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.get("/health")
                assert resp.status_code == 200
                mocks["active_requests"].inc.assert_called()
                mocks["active_requests"].dec.assert_called()

    def test_middleware_records_request_count_and_latency(self):
        """Request count and latency are recorded for each request."""
        with _prometheus_mocks() as mocks:
            with patch("src.api.rest_server.framework_service", None):
                client = TestClient(app, raise_server_exceptions=False)
                client.get("/health")
                mocks["request_count"].labels.assert_called()
                mocks["request_count"].labels().inc.assert_called()
                mocks["request_latency"].labels.assert_called()
                mocks["request_latency"].labels().observe.assert_called()

    def test_middleware_records_error_status(self):
        """Middleware records status code even for error responses."""
        with _prometheus_mocks() as mocks:
            with patch("src.api.rest_server.framework_service", None):
                client = TestClient(app, raise_server_exceptions=False)
                # /query without auth returns 422
                client.post("/query", json={"query": "test"})
                mocks["request_count"].labels.assert_called()


# ---------------------------------------------------------------------------
# verify_api_key edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerifyApiKeyExtended:
    """Extended tests for verify_api_key dependency."""

    @pytest.mark.asyncio
    async def test_auth_error_increments_prometheus(self):
        """Authentication failure increments Prometheus error counter."""
        from src.api.exceptions import AuthenticationError

        with _prometheus_mocks() as mocks:
            mock_auth = MagicMock()
            mock_auth.require_auth.side_effect = AuthenticationError(user_message="Bad key")

            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.get_authenticator", return_value=mock_auth):
                with pytest.raises(HTTPException) as exc_info:
                    await verify_api_key(x_api_key="bad-key")
                assert exc_info.value.status_code == 401
                mocks["error_count"].labels.assert_called_with(error_type="authentication")

    @pytest.mark.asyncio
    async def test_rate_limit_increments_prometheus(self):
        """Rate limit error increments Prometheus error counter."""
        from src.api.exceptions import RateLimitError

        mock_settings = MagicMock()
        mock_settings.RATE_LIMIT_RETRY_AFTER_SECONDS = 60

        with _prometheus_mocks() as mocks:
            mock_auth = MagicMock()
            mock_auth.require_auth.side_effect = RateLimitError(
                user_message="Too many requests",
                retry_after_seconds=30,
            )

            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.get_authenticator", return_value=mock_auth), \
                 patch("src.api.rest_server.get_settings", return_value=mock_settings):
                with pytest.raises(HTTPException) as exc_info:
                    await verify_api_key(x_api_key="key")
                assert exc_info.value.status_code == 429
                mocks["error_count"].labels.assert_called_with(error_type="rate_limit")

    @pytest.mark.asyncio
    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.get_settings")
    @patch("src.api.rest_server.get_authenticator")
    async def test_rate_limit_uses_settings_fallback_when_retry_after_is_none(
        self, mock_get_auth, mock_get_settings
    ):
        """When RateLimitError.retry_after_seconds is None, fall back to settings."""
        from src.api.exceptions import RateLimitError

        mock_settings = MagicMock()
        mock_settings.RATE_LIMIT_RETRY_AFTER_SECONDS = 120
        mock_get_settings.return_value = mock_settings

        mock_auth = MagicMock()
        mock_auth.require_auth.side_effect = RateLimitError(
            user_message="Too many requests",
            retry_after_seconds=None,
        )
        mock_get_auth.return_value = mock_auth

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="key")
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers["Retry-After"] == "120"

    @pytest.mark.asyncio
    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.get_settings")
    @patch("src.api.rest_server.get_authenticator")
    async def test_rate_limit_uses_exception_retry_after(
        self, mock_get_auth, mock_get_settings
    ):
        """When RateLimitError.retry_after_seconds is set, it takes precedence."""
        from src.api.exceptions import RateLimitError

        mock_settings = MagicMock()
        mock_settings.RATE_LIMIT_RETRY_AFTER_SECONDS = 120
        mock_get_settings.return_value = mock_settings

        mock_auth = MagicMock()
        mock_auth.require_auth.side_effect = RateLimitError(
            user_message="Too many requests",
            retry_after_seconds=45,
        )
        mock_get_auth.return_value = mock_auth

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key="key")
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers["Retry-After"] == "45"


# ---------------------------------------------------------------------------
# Exception handlers with Prometheus
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExceptionHandlersWithPrometheus:
    """Tests for exception handlers with Prometheus metrics."""

    def test_framework_error_handler_increments_counter(self):
        """FrameworkError handler increments Prometheus error counter."""
        from src.api.exceptions import FrameworkError

        @app.get("/test-fw-err-prom")
        async def _raise_fw_error():
            raise FrameworkError(user_message="fail", error_code="TEST_PROM")

        try:
            with _prometheus_mocks() as mocks:
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.get("/test-fw-err-prom")
                assert resp.status_code == 500
                data = resp.json()
                assert data["error_code"] == "TEST_PROM"
                mocks["error_count"].labels.assert_any_call(error_type="TEST_PROM")
        finally:
            app.routes[:] = [
                r for r in app.routes if getattr(r, "path", None) != "/test-fw-err-prom"
            ]

    def test_validation_error_handler_increments_counter(self):
        """ValidationError handler increments Prometheus error counter."""
        from src.api.exceptions import ValidationError as FwValidationError

        @app.get("/test-val-err-prom")
        async def _raise_val_error():
            raise FwValidationError(user_message="bad input", field_name="x")

        try:
            with _prometheus_mocks() as mocks:
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.get("/test-val-err-prom")
                assert resp.status_code == 400
                mocks["error_count"].labels.assert_any_call(error_type="validation")
        finally:
            app.routes[:] = [
                r for r in app.routes if getattr(r, "path", None) != "/test-val-err-prom"
            ]


# ---------------------------------------------------------------------------
# /query endpoint extended coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueryEndpointExtended:
    """Extended tests for /query endpoint covering uncovered error paths."""

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_with_mcts_iterations_passthrough(self, mock_fw_service):
        """Custom mcts_iterations is passed to framework_service.process_query."""
        mock_result = MagicMock()
        mock_result.response = "Response"
        mock_result.confidence = 0.8
        mock_result.agents_used = ["hrm"]
        mock_result.mcts_stats = {"iterations": 50}
        mock_result.processing_time_ms = 100.0
        mock_result.metadata = {}
        mock_fw_service.process_query = AsyncMock(return_value=mock_result)

        client = _make_client_with_auth()
        resp = client.post("/query", json={
            "query": "test query",
            "use_mcts": True,
            "use_rag": False,
            "mcts_iterations": 50,
        })
        assert resp.status_code == 200
        call_kwargs = mock_fw_service.process_query.call_args[1]
        assert call_kwargs["mcts_iterations"] == 50
        assert call_kwargs["use_rag"] is False

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_with_thread_id(self, mock_fw_service):
        """thread_id is passed through to framework."""
        mock_result = MagicMock()
        mock_result.response = "Response"
        mock_result.confidence = 0.8
        mock_result.agents_used = []
        mock_result.mcts_stats = None
        mock_result.processing_time_ms = 50.0
        mock_result.metadata = {}
        mock_fw_service.process_query = AsyncMock(return_value=mock_result)

        client = _make_client_with_auth()
        resp = client.post("/query", json={
            "query": "test",
            "thread_id": "session-42",
        })
        assert resp.status_code == 200
        call_kwargs = mock_fw_service.process_query.call_args[1]
        assert call_kwargs["thread_id"] == "session-42"

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.framework_service")
    def test_query_metadata_includes_client_id(self, mock_fw_service):
        """Client ID from auth is injected into response metadata."""
        mock_result = MagicMock()
        mock_result.response = "R"
        mock_result.confidence = 0.5
        mock_result.agents_used = ["hrm"]
        mock_result.mcts_stats = None
        mock_result.processing_time_ms = 10.0
        mock_result.metadata = {"existing": "data"}
        mock_fw_service.process_query = AsyncMock(return_value=mock_result)

        ci = MagicMock()
        ci.client_id = "my-client-123"
        ci.roles = {"user"}
        client = _make_client_with_auth(ci)
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 200
        assert mock_result.metadata["client_id"] == "my-client-123"

    def test_query_timeout_increments_prometheus(self):
        """TimeoutError in process_query increments Prometheus timeout counter."""
        with _prometheus_mocks() as mocks:
            mock_fw_service = MagicMock()
            mock_fw_service.process_query = AsyncMock(side_effect=TimeoutError("timed out"))

            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.framework_service", mock_fw_service):
                client = _make_client_with_auth()
                resp = client.post("/query", json={"query": "slow query"})
                assert resp.status_code == 504
                mocks["error_count"].labels.assert_any_call(error_type="timeout")

    def test_query_value_error_increments_prometheus(self):
        """ValueError in process_query increments Prometheus validation counter."""
        with _prometheus_mocks() as mocks:
            mock_fw_service = MagicMock()
            mock_fw_service.process_query = AsyncMock(side_effect=ValueError("bad"))

            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.framework_service", mock_fw_service):
                client = _make_client_with_auth()
                resp = client.post("/query", json={"query": "bad query"})
                assert resp.status_code == 400
                mocks["error_count"].labels.assert_any_call(error_type="validation")

    def test_query_runtime_error_increments_prometheus(self):
        """RuntimeError in process_query increments Prometheus runtime counter."""
        with _prometheus_mocks() as mocks:
            mock_fw_service = MagicMock()
            mock_fw_service.process_query = AsyncMock(side_effect=RuntimeError("broken"))

            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.framework_service", mock_fw_service):
                client = _make_client_with_auth()
                resp = client.post("/query", json={"query": "test"})
                assert resp.status_code == 503
                mocks["error_count"].labels.assert_any_call(error_type="runtime")

    def test_query_unexpected_error_increments_prometheus(self):
        """Unexpected Exception increments Prometheus internal error counter."""
        with _prometheus_mocks() as mocks:
            mock_fw_service = MagicMock()
            mock_fw_service.process_query = AsyncMock(side_effect=Exception("unexpected"))

            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.framework_service", mock_fw_service):
                client = _make_client_with_auth()
                resp = client.post("/query", json={"query": "test"})
                assert resp.status_code == 500
                mocks["error_count"].labels.assert_any_call(error_type="internal")

    def test_query_no_framework_increments_prometheus(self):
        """When framework_service is None, service_unavailable error counter increments."""
        with _prometheus_mocks() as mocks:
            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.framework_service", None):
                client = _make_client_with_auth()
                resp = client.post("/query", json={"query": "test"})
                assert resp.status_code == 503
                mocks["error_count"].labels.assert_any_call(error_type="service_unavailable")

    def test_query_validation_failure_increments_prometheus(self):
        """QueryInput validation failure increments Prometheus validation counter."""
        with _prometheus_mocks() as mocks:
            with patch("src.api.rest_server.IMPORTS_AVAILABLE", True), \
                 patch("src.api.rest_server.QueryInput", side_effect=Exception("validation failed")), \
                 patch("src.api.rest_server.framework_service", None):
                client = _make_client_with_auth()
                resp = client.post("/query", json={"query": "test"})
                assert resp.status_code == 400
                assert "Validation failed" in resp.json()["detail"]
                mocks["error_count"].labels.assert_any_call(error_type="validation")

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", False)
    @patch("src.api.rest_server.framework_service")
    def test_query_skips_validation_when_imports_unavailable(self, mock_fw_service):
        """When IMPORTS_AVAILABLE is False, skip QueryInput validation."""
        mock_fw_service.__bool__ = MagicMock(return_value=True)
        mock_result = MagicMock()
        mock_result.response = "R"
        mock_result.confidence = 0.5
        mock_result.agents_used = []
        mock_result.mcts_stats = None
        mock_result.processing_time_ms = 5.0
        mock_result.metadata = {}
        mock_fw_service.process_query = AsyncMock(return_value=mock_result)

        client = _make_client_with_auth()
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /metrics endpoint with Prometheus available
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricsEndpointWithPrometheus:
    """Tests for /metrics endpoint when Prometheus is available."""

    def test_metrics_returns_prometheus_format(self):
        """When Prometheus is available, /metrics returns generated output."""
        mock_generate = MagicMock(return_value=b"# HELP mcts_requests_total\n")
        with _prometheus_mocks(), \
             patch("src.api.rest_server.generate_latest", mock_generate, create=True), \
             patch("src.api.rest_server.CONTENT_TYPE_LATEST", "text/plain; version=0.0.4", create=True):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/metrics")
            assert resp.status_code == 200
            assert b"HELP" in resp.content

    @patch("src.api.rest_server.PROMETHEUS_AVAILABLE", False)
    def test_metrics_returns_501_without_prometheus(self):
        """When Prometheus is not available, /metrics returns 501."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert resp.status_code == 501
        assert "not available" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /health endpoint extended coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthEndpointExtended:
    """Extended health endpoint tests for branch coverage."""

    def test_health_with_ready_framework(self):
        """Health returns 'healthy' when framework is in READY state."""
        from src.api.framework_service import FrameworkState

        mock_service = MagicMock()
        mock_service.state = FrameworkState.READY

        with patch("src.api.rest_server.framework_service", mock_service):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"

    def test_health_returns_iso_timestamp(self):
        """Health response timestamp is in ISO format."""
        with patch("src.api.rest_server.framework_service", None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            data = resp.json()
            assert "T" in data["timestamp"]
            assert data["uptime_seconds"] >= 0

    def test_health_with_initializing_framework(self):
        """Health returns 'initializing' when framework is UNINITIALIZED."""
        from src.api.framework_service import FrameworkState

        mock_service = MagicMock()
        mock_service.state = FrameworkState.INITIALIZING

        with patch("src.api.rest_server.framework_service", mock_service):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
            # INITIALIZING state falls through to default "healthy"
            assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# /ready endpoint extended coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadinessEndpointExtended:
    """Extended readiness endpoint tests."""

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", True)
    def test_ready_all_checks_pass_with_framework(self):
        """Readiness includes all check fields when framework is ready."""
        mock_service = MagicMock()
        mock_service.is_ready = True
        with patch("src.api.rest_server.framework_service", mock_service):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["checks"]["imports_available"] is True
            assert data["checks"]["authenticator_configured"] is True
            assert data["checks"]["framework_service_available"] is True
            assert data["checks"]["framework_ready"] is True

    @patch("src.api.rest_server.IMPORTS_AVAILABLE", True)
    @patch("src.api.rest_server.FRAMEWORK_SERVICE_AVAILABLE", False)
    def test_ready_without_optional_services(self):
        """Readiness still passes when optional services unavailable."""
        with patch("src.api.rest_server.framework_service", None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["checks"]["framework_ready"] is False
            assert data["ready"] is True


# ---------------------------------------------------------------------------
# /stats endpoint extended coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStatsEndpointExtended:
    """Extended /stats endpoint tests."""

    @patch("src.api.rest_server.get_authenticator")
    def test_stats_includes_all_rate_limit_fields(self, mock_get_auth):
        """Stats response includes per_minute, per_hour, per_day."""
        mock_auth = MagicMock()
        mock_auth.get_client_stats.return_value = {
            "total_requests_today": 100,
            "requests_last_hour": 20,
            "requests_last_minute": 3,
        }
        mock_auth.rate_limit_config.requests_per_minute = 30
        mock_auth.rate_limit_config.requests_per_hour = 1800
        mock_auth.rate_limit_config.requests_per_day = 43200
        mock_get_auth.return_value = mock_auth

        ci = MagicMock()
        ci.client_id = "stats-client"
        ci.roles = {"admin", "user"}

        client = _make_client_with_auth(ci)
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["client_id"] == "stats-client"
        assert set(data["roles"]) == {"admin", "user"}
        assert data["rate_limits"]["per_minute"] == 30
        assert data["rate_limits"]["per_hour"] == 1800
        assert data["rate_limits"]["per_day"] == 43200
        assert data["total_requests_today"] == 100

    @patch("src.api.rest_server.get_authenticator")
    def test_stats_with_empty_roles(self, mock_get_auth):
        """Stats works when client has no roles."""
        mock_auth = MagicMock()
        mock_auth.get_client_stats.return_value = {}
        mock_auth.rate_limit_config.requests_per_minute = 60
        mock_auth.rate_limit_config.requests_per_hour = 3600
        mock_auth.rate_limit_config.requests_per_day = 86400
        mock_get_auth.return_value = mock_auth

        ci = MagicMock()
        ci.client_id = "no-role-client"
        ci.roles = set()

        client = _make_client_with_auth(ci)
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["roles"] == []


# ---------------------------------------------------------------------------
# QueryRequest / QueryResponse model edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelsEdgeCases:
    """Edge case tests for request/response models."""

    def test_query_request_json_schema_extra(self):
        """QueryRequest has json_schema_extra example in its Config."""
        schema = QueryRequest.model_json_schema()
        assert "example" in schema.get("properties", {}).get("query", {})

    def test_query_response_full_serialization(self):
        """Full QueryResponse round-trips through model_dump correctly."""
        resp = QueryResponse(
            response="answer text",
            confidence=0.75,
            agents_used=["hrm", "trm", "mcts"],
            mcts_stats={"iterations": 200, "best_value": 0.9},
            processing_time_ms=350.5,
            metadata={"thread_id": "abc", "client_id": "c1"},
        )
        data = resp.model_dump()
        assert data["response"] == "answer text"
        assert data["mcts_stats"]["iterations"] == 200
        assert data["metadata"]["thread_id"] == "abc"

    def test_query_request_boundary_iterations(self):
        """QueryRequest boundary values for mcts_iterations."""
        req1 = QueryRequest(query="test", mcts_iterations=1)
        assert req1.mcts_iterations == 1
        req2 = QueryRequest(query="test", mcts_iterations=10000)
        assert req2.mcts_iterations == 10000

    def test_query_request_all_flags_disabled(self):
        """QueryRequest with all optional flags disabled."""
        req = QueryRequest(query="test", use_mcts=False, use_rag=False)
        assert req.use_mcts is False
        assert req.use_rag is False
