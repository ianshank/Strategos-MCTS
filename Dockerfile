# Multi-stage build for LangGraph Multi-Agent MCTS Framework
# Stage 1: Build dependencies
FROM python:3.14-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source needed to install the package with its real deps
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Create virtual environment and install dependencies.
# requirements.txt covers the model/demo deps (torch, transformers, numpy);
# installing the package with the api+prometheus extras pulls the actual API runtime
# dependencies (fastapi, uvicorn, langgraph, pydantic-settings, opentelemetry, ...) that
# the served `uvicorn src.api.rest_server:app` process needs.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install ".[api,prometheus]"

# Stage 2: Production image
FROM python:3.14-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    # Application defaults
    LLM_PROVIDER=lmstudio \
    MCTS_DEFAULT_ITERATIONS=100 \
    LOG_LEVEL=INFO \
    # Security
    PYTHONHASHSEED=random

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/logs /app/models /app/data && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser tools/ ./tools/
COPY --chown=appuser:appuser examples/ ./examples/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser pyproject.toml ./
# Note: Examples are included for reference and demos

# Switch to non-root user
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
# 8000 - Main REST API server
# 9090 - Prometheus metrics (optional)
EXPOSE 8000 9090

# Default command - run the REST API server for production
CMD ["python", "-m", "uvicorn", "src.api.rest_server:app", "--host", "0.0.0.0", "--port", "8000"]

# Labels for metadata
LABEL maintainer="LangGraph MCTS Team" \
      version="1.0.0" \
      description="Multi-Agent MCTS Framework with LangGraph" \
      org.opencontainers.image.source="https://github.com/ianshank/langgraph_multi_agent_mcts"
