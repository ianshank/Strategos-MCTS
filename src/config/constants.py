"""
Configuration constants for application settings.

Centralizes default values and validation bounds to ensure
consistency and ease of customization across the application.
These constants provide shared defaults and bounds for
configuration-related code throughout the project.
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# MCTS Configuration
# ============================================================================

# MCTS iteration defaults and bounds
DEFAULT_MCTS_ITERATIONS: Final[int] = 100
MIN_MCTS_ITERATIONS: Final[int] = 1
MAX_MCTS_ITERATIONS: Final[int] = 10000

# MCTS exploration weight (UCB1 constant)
DEFAULT_MCTS_C: Final[float] = 1.414  # sqrt(2) is theoretically optimal
MIN_MCTS_C: Final[float] = 0.0
MAX_MCTS_C: Final[float] = 10.0

# Default seed for reproducibility
DEFAULT_SEED: Final[int] = 42

# ============================================================================
# Network Configuration
# ============================================================================

# HTTP timeout configuration
DEFAULT_HTTP_TIMEOUT_SECONDS: Final[int] = 30
MIN_HTTP_TIMEOUT_SECONDS: Final[int] = 1
MAX_HTTP_TIMEOUT_SECONDS: Final[int] = 300

# HTTP retry configuration
DEFAULT_HTTP_MAX_RETRIES: Final[int] = 3
MIN_HTTP_MAX_RETRIES: Final[int] = 0
MAX_HTTP_MAX_RETRIES: Final[int] = 10

# ============================================================================
# Security Configuration
# ============================================================================

# Query length limits
DEFAULT_MAX_QUERY_LENGTH: Final[int] = 10000
MIN_MAX_QUERY_LENGTH: Final[int] = 1
MAX_MAX_QUERY_LENGTH: Final[int] = 100000

# Rate limiting
DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 60
MIN_RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 1
MAX_RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 1000

# ============================================================================
# Framework Configuration
# ============================================================================

# Framework iteration limits
DEFAULT_FRAMEWORK_MAX_ITERATIONS: Final[int] = 3
MIN_FRAMEWORK_MAX_ITERATIONS: Final[int] = 1
MAX_FRAMEWORK_MAX_ITERATIONS: Final[int] = 100

# Consensus threshold for agent agreement
DEFAULT_CONSENSUS_THRESHOLD: Final[float] = 0.75
MIN_CONSENSUS_THRESHOLD: Final[float] = 0.0
MAX_CONSENSUS_THRESHOLD: Final[float] = 1.0

# RAG retrieval configuration
DEFAULT_TOP_K_RETRIEVAL: Final[int] = 5
MIN_TOP_K_RETRIEVAL: Final[int] = 1
MAX_TOP_K_RETRIEVAL: Final[int] = 100

# ============================================================================
# LLM Configuration
# ============================================================================

# Default temperature for LLM generation
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.7
MIN_LLM_TEMPERATURE: Final[float] = 0.0
MAX_LLM_TEMPERATURE: Final[float] = 2.0

# Confidence thresholds
DEFAULT_CONFIDENCE_WITH_RAG: Final[float] = 0.8
DEFAULT_CONFIDENCE_WITHOUT_RAG: Final[float] = 0.7
DEFAULT_CONFIDENCE_ON_ERROR: Final[float] = 0.3

# Error response preview length
DEFAULT_ERROR_QUERY_PREVIEW_LENGTH: Final[int] = 100

# ============================================================================
# S3 Storage Configuration
# ============================================================================

DEFAULT_S3_PREFIX: Final[str] = "mcts-artifacts"
DEFAULT_S3_REGION: Final[str] = "us-east-1"

# S3 bucket name validation bounds
MIN_S3_BUCKET_NAME_LENGTH: Final[int] = 3
MAX_S3_BUCKET_NAME_LENGTH: Final[int] = 63

# ============================================================================
# API Key Validation
# ============================================================================

# Minimum API key lengths for validation
MIN_OPENAI_API_KEY_LENGTH: Final[int] = 20
MIN_ANTHROPIC_API_KEY_LENGTH: Final[int] = 20
MIN_PINECONE_API_KEY_LENGTH: Final[int] = 20
MIN_LANGSMITH_API_KEY_LENGTH: Final[int] = 20
MIN_WANDB_API_KEY_LENGTH: Final[int] = 20

# API key prefixes
OPENAI_API_KEY_PREFIX: Final[str] = "sk-"
ANTHROPIC_API_KEY_PREFIX: Final[str] = "sk-ant-"

# Placeholder values that should be rejected
API_KEY_PLACEHOLDERS: Final[tuple[str, ...]] = (
    "",
    "your-api-key-here",
    "REPLACE_ME",
    "your_api_key_here",
    "YOUR_API_KEY",
)

# ============================================================================
# Project Names
# ============================================================================

DEFAULT_LANGSMITH_PROJECT: Final[str] = "langgraph-mcts"
DEFAULT_WANDB_PROJECT: Final[str] = "langgraph-mcts"
DEFAULT_WANDB_MODE: Final[str] = "online"

# ============================================================================
# API Endpoints
# ============================================================================

DEFAULT_LANGCHAIN_ENDPOINT: Final[str] = "https://api.smith.langchain.com"
DEFAULT_LMSTUDIO_URL: Final[str] = "http://localhost:1234/v1"
DEFAULT_SERVER_HOST: Final[str] = "0.0.0.0"
DEFAULT_OPENAI_BASE_URL: Final[str] = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL: Final[str] = "https://api.anthropic.com"
DEFAULT_OTLP_ENDPOINT: Final[str] = "localhost:4317"
DEFAULT_OTLP_HTTP_ENDPOINT: Final[str] = "http://localhost:4317"

# ============================================================================
# LLM Provider Defaults
# ============================================================================

# Default model names
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4-turbo-preview"
DEFAULT_ANTHROPIC_MODEL: Final[str] = "claude-3-5-sonnet-20241022"

# Default timeouts per provider (seconds)
DEFAULT_OPENAI_TIMEOUT: Final[float] = 60.0
DEFAULT_ANTHROPIC_TIMEOUT: Final[float] = 120.0
DEFAULT_LMSTUDIO_TIMEOUT: Final[float] = 300.0
DEFAULT_LLM_TIMEOUT: Final[float] = 60.0

# Default max tokens
DEFAULT_MAX_TOKENS: Final[int] = 4096
DEFAULT_MAX_RETRIES: Final[int] = 5

# ============================================================================
# S3 Storage Defaults
# ============================================================================

DEFAULT_S3_BUCKET_NAME: Final[str] = "mcts-framework-storage"

# ============================================================================
# Service Metadata
# ============================================================================

SERVICE_VERSION: Final[str] = "0.1.0"
DEFAULT_ENVIRONMENT: Final[str] = "development"
DEFAULT_OTEL_SERVICE_NAME: Final[str] = "mcts-framework"
