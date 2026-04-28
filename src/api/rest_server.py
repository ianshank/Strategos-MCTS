"""
Production REST API server for LangGraph Multi-Agent MCTS Framework.

Provides:
- OpenAPI/Swagger documentation
- Authentication via API keys
- Rate limiting
- Health and readiness endpoints
- Request validation with Pydantic
- Prometheus metrics exposure
- Full framework integration

Best Practices 2025:
- Configuration-driven (no hardcoded values)
- Dependency injection via service layer
- Async-first design
- Comprehensive error handling
- Type-safe interfaces
"""

import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config.constants import DEFAULT_SERVER_HOST
from src.config.settings import get_settings
from src.observability.logging import get_logger

# Configure logging
logger = get_logger(__name__)

# Import framework components
try:
    from src.api.auth import (
        APIKeyAuthenticator,
        ClientInfo,
        RateLimitConfig,
        get_authenticator,
        set_authenticator,
    )
    from src.api.exceptions import (
        AuthenticationError,
        FrameworkError,
        RateLimitError,
        ValidationError,
    )
    from src.api.framework_service import (
        FrameworkConfig,
        FrameworkService,
        FrameworkState,
    )
    from src.models.validation import QueryInput

    IMPORTS_AVAILABLE = True
    FRAMEWORK_SERVICE_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    FRAMEWORK_SERVICE_AVAILABLE = False
    import_error = str(e)
    logger.warning(f"Import error: {e}")

# Prometheus metrics (optional)
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True

    # Define metrics
    REQUEST_COUNT = Counter("mcts_requests_total", "Total number of requests", ["method", "endpoint", "status"])
    REQUEST_LATENCY = Histogram("mcts_request_duration_seconds", "Request latency in seconds", ["method", "endpoint"])
    ACTIVE_REQUESTS = Gauge("mcts_active_requests", "Number of active requests")
    ERROR_COUNT = Counter("mcts_errors_total", "Total number of errors", ["error_type"])
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query processing."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User query to process",
        json_schema_extra={"example": "Recommend defensive positions for night attack scenario"},
    )
    use_mcts: bool = Field(default=True, description="Enable MCTS tactical simulation")
    use_rag: bool = Field(default=True, description="Enable RAG context retrieval")
    mcts_iterations: int | None = Field(default=None, ge=1, le=10000, description="Override default MCTS iterations")
    thread_id: str | None = Field(
        default=None,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Conversation thread ID for state persistence",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Recommend defensive positions for night attack",
                "use_mcts": True,
                "use_rag": True,
                "mcts_iterations": 200,
                "thread_id": "session_123",
            }
        }


class QueryResponse(BaseModel):
    """Response model for query results."""

    response: str = Field(..., description="Final synthesized response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    agents_used: list[str] = Field(..., description="List of agents that contributed")
    mcts_stats: dict[str, Any] | None = Field(default=None, description="MCTS simulation statistics")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(default="1.0.0", description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether service is ready")
    checks: dict[str, bool] = Field(..., description="Individual check results")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: bool = Field(default=True)
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    timestamp: str = Field(..., description="Error timestamp")


# Application startup
start_time = time.time()
framework_service: FrameworkService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager with framework initialization."""
    global framework_service

    # Startup
    logger.info("Starting MCTS Framework API server...")

    # Load settings for configuration
    settings = get_settings()

    # Initialize authenticator from settings (no hardcoded values)
    api_keys_env = os.environ.get("API_KEYS", "")
    api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    if not api_keys:
        # Fallback for development - generate a random key
        dev_key = f"dev-{secrets.token_hex(16)}"
        api_keys = [dev_key]
        # Log generic message without exposing the actual key value
        logger.warning("No API_KEYS configured. Generated temporary development key.")

    # Rate limits scale from per-minute base (theoretical max if sustained)
    authenticator = APIKeyAuthenticator(
        valid_keys=api_keys,
        rate_limit_config=RateLimitConfig(
            requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
            requests_per_hour=settings.RATE_LIMIT_REQUESTS_PER_MINUTE * 60,
            requests_per_day=settings.RATE_LIMIT_REQUESTS_PER_MINUTE * 60 * 24,
        ),
    )
    set_authenticator(authenticator)

    # Initialize framework service
    if FRAMEWORK_SERVICE_AVAILABLE:
        try:
            framework_config = FrameworkConfig.from_settings(settings)
            framework_service = await FrameworkService.get_instance(
                config=framework_config,
                settings=settings,
            )
            # Initialize the framework (lazy - will happen on first request if not here)
            init_success = await framework_service.initialize()
            if init_success:
                logger.info("Framework service initialized successfully")
            else:
                logger.warning("Framework service initialization deferred")
        except Exception as e:  # Broad catch: API stability requires catching unexpected errors
            logger.error(f"Failed to initialize framework service: {e}")
            framework_service = None
    else:
        logger.warning("Framework service not available due to import errors")

    logger.info("API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down API server...")
    if framework_service is not None:
        await framework_service.shutdown()
        await FrameworkService.reset_instance()
    logger.info("API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LangGraph Multi-Agent MCTS API",
    description="""
## Multi-Agent Reasoning API with MCTS Tactical Simulation

This API provides access to a sophisticated multi-agent reasoning framework that combines:
- **HRM Agent**: Hierarchical decomposition of complex queries
- **TRM Agent**: Iterative refinement for response quality
- **MCTS Engine**: Monte Carlo Tree Search for tactical simulation
- **RAG Integration**: Context retrieval from vector stores

### Features
- Secure API key authentication
- Rate limiting per client
- Real-time metrics (Prometheus)
- Distributed tracing (OpenTelemetry)
- Production-grade error handling

### Quick Start
1. Obtain an API key
2. Include `X-API-Key` header in requests
3. Send queries to `/query` endpoint
4. Monitor health via `/health` endpoint
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "query", "description": "Query processing operations"},
        {"name": "health", "description": "Health and readiness checks"},
        {"name": "metrics", "description": "Observability endpoints"},
    ],
    lifespan=lifespan,
)

# CORS middleware - secure by default, only enabled when explicitly configured.
# CORS settings are read once at import time. Changes require a server restart.
# Set CORS_ALLOWED_ORIGINS env var (comma-separated) to enable CORS.
# Security: credentials are disabled whenever wildcard origins are used.
_cors_settings = get_settings()
_cors_origins = _cors_settings.CORS_ALLOWED_ORIGINS

if _cors_origins:
    _has_wildcard_origin = "*" in _cors_origins
    if _has_wildcard_origin:
        # Normalize to explicit wildcard-only and disable credentials
        _cors_origins = ["*"]
        _cors_allow_credentials = False
    else:
        _cors_allow_credentials = _cors_settings.CORS_ALLOW_CREDENTIALS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    if PROMETHEUS_AVAILABLE:
        ACTIVE_REQUESTS.inc()

    start = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:  # Broad catch: API stability requires catching unexpected errors
        status = 500
        raise
    finally:
        if PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.dec()
            elapsed = time.perf_counter() - start
            REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=str(status)).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(elapsed)

    return response


# Authentication dependency
async def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    """Verify API key and return client info."""
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Authentication module not available")

    try:
        authenticator = get_authenticator()
        client_info = authenticator.require_auth(x_api_key)
        return client_info
    except AuthenticationError as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="authentication").inc()
        raise HTTPException(status_code=401, detail=e.user_message) from e
    except RateLimitError as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="rate_limit").inc()
        settings = get_settings()
        retry_after = e.retry_after_seconds or settings.RATE_LIMIT_RETRY_AFTER_SECONDS
        raise HTTPException(status_code=429, detail=e.user_message, headers={"Retry-After": str(retry_after)}) from e


# Exception handlers
@app.exception_handler(FrameworkError)
async def framework_error_handler(_request: Request, exc: FrameworkError):
    """Handle framework-specific errors."""
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNT.labels(error_type=exc.error_code).inc()

    return JSONResponse(status_code=500, content=exc.to_user_response())


@app.exception_handler(ValidationError)
async def validation_error_handler(_request: Request, exc: ValidationError):
    """Handle validation errors."""
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNT.labels(error_type="validation").inc()

    return JSONResponse(status_code=400, content=exc.to_user_response())


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Returns basic service health status. Use this for load balancer health checks.
    """
    # Determine health status based on framework state
    status = "healthy"
    if framework_service is not None:
        if framework_service.state == FrameworkState.ERROR:
            status = "degraded"
        elif framework_service.state == FrameworkState.UNINITIALIZED:
            status = "initializing"

    return HealthResponse(
        status=status,
        timestamp=datetime.now(UTC).isoformat(),
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["health"])
async def readiness_check():
    """
    Readiness check endpoint.

    Verifies all dependencies are available. Use this for Kubernetes readiness probes.
    """
    # Check framework service status
    framework_ready = False
    if framework_service is not None:
        framework_ready = framework_service.is_ready

    checks = {
        "imports_available": IMPORTS_AVAILABLE,
        "authenticator_configured": True,
        "framework_service_available": FRAMEWORK_SERVICE_AVAILABLE,
        "framework_ready": framework_ready,
        "prometheus_available": PROMETHEUS_AVAILABLE,
    }

    # Check if all critical services are available
    all_ready = all(
        [
            checks["imports_available"],
            checks["authenticator_configured"],
            # Framework readiness is optional - can still serve basic requests
        ]
    )

    if not all_ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    return ReadinessResponse(ready=all_ready, checks=checks)


@app.get("/metrics", tags=["metrics"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Prometheus metrics not available")

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        504: {"model": ErrorResponse, "description": "Request timeout"},
    },
)
async def process_query(request: QueryRequest, client_info: ClientInfo = Depends(verify_api_key)):  # noqa: B008
    """
    Process a query using the multi-agent MCTS framework.

    This endpoint:
    1. Validates the input query
    2. Optionally retrieves context via RAG
    3. Processes through HRM and TRM agents
    4. Optionally runs MCTS simulation
    5. Synthesizes a final response

    **Authentication**: Requires valid API key in X-API-Key header.

    **Rate Limiting**: Subject to rate limits per client.
    """
    request_start = time.perf_counter()

    # Validate input using validation models
    if IMPORTS_AVAILABLE:
        try:
            QueryInput(
                query=request.query,
                use_rag=request.use_rag,
                use_mcts=request.use_mcts,
                thread_id=request.thread_id,
            )
        except Exception as e:  # Broad catch: API stability requires catching unexpected errors
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNT.labels(error_type="validation").inc()
            raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}") from e

    # Check framework service availability
    if framework_service is None:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="service_unavailable").inc()
        raise HTTPException(
            status_code=503,
            detail="Framework service not available. Please try again later.",
        )

    # Process query through the actual framework
    try:
        result = await framework_service.process_query(
            query=request.query,
            use_mcts=request.use_mcts,
            use_rag=request.use_rag,
            thread_id=request.thread_id,
            mcts_iterations=request.mcts_iterations,
        )

        # Add client info to metadata
        result.metadata["client_id"] = client_info.client_id

        return QueryResponse(
            response=result.response,
            confidence=result.confidence,
            agents_used=result.agents_used,
            mcts_stats=result.mcts_stats,
            processing_time_ms=result.processing_time_ms,
            metadata=result.metadata,
        )

    except TimeoutError as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="timeout").inc()
        processing_time = (time.perf_counter() - request_start) * 1000
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {processing_time:.0f}ms: {str(e)}",
        ) from e

    except ValueError as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="validation").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e

    except RuntimeError as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="runtime").inc()
        logger.error(f"Runtime error processing query: {e}")
        raise HTTPException(
            status_code=503,
            detail="Framework temporarily unavailable. Please try again.",
        ) from e

    except Exception as e:  # Broad catch: API stability requires catching unexpected errors
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="internal").inc()
        logger.exception(f"Unexpected error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        ) from e


@app.get("/stats", tags=["metrics"])
async def get_stats(client_info: ClientInfo = Depends(verify_api_key)):  # noqa: B008
    """
    Get usage statistics for the authenticated client.

    Returns request counts and rate limit information.
    """
    authenticator = get_authenticator()
    stats = authenticator.get_client_stats(client_info.client_id)

    return {
        "client_id": client_info.client_id,
        "roles": list(client_info.roles),
        **stats,
        "rate_limits": {
            "per_minute": authenticator.rate_limit_config.requests_per_minute,
            "per_hour": authenticator.rate_limit_config.requests_per_hour,
            "per_day": authenticator.rate_limit_config.requests_per_day,
        },
    }


# Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.rest_server:app",
        host=DEFAULT_SERVER_HOST,
        port=8000,
        reload=False,
        workers=4,
        log_level="info",
        access_log=True,
    )
