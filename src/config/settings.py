"""
Pydantic Settings v2 configuration management for LangGraph Multi-Agent MCTS.

Provides:
- Secure configuration loading from environment variables and .env files
- Type-safe settings with validation
- Secrets protection using SecretStr
- MCTS parameter bounds validation
- Support for multiple LLM providers
"""

from enum import Enum
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LMSTUDIO = "lmstudio"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MCTSImplementation(str, Enum):
    """MCTS implementation variants."""

    BASELINE = "baseline"  # Original MCTS core
    NEURAL = "neural"  # Neural-guided AlphaZero-style MCTS


class Settings(BaseSettings):
    """
    Application settings with security-first configuration.

    All sensitive values use SecretStr to prevent accidental exposure in logs.
    Configuration is loaded from environment variables with .env file support.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    # LLM Provider Configuration
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OPENAI, description="LLM provider to use (openai, anthropic, lmstudio)"
    )

    # API Keys (Secrets)
    OPENAI_API_KEY: SecretStr | None = Field(
        default=None, description="OpenAI API key (required if using OpenAI provider)"
    )

    ANTHROPIC_API_KEY: SecretStr | None = Field(
        default=None, description="Anthropic API key (required if using Anthropic provider)"
    )

    BRAINTRUST_API_KEY: SecretStr | None = Field(
        default=None, description="Braintrust API key for experiment tracking (optional)"
    )

    PINECONE_API_KEY: SecretStr | None = Field(
        default=None, description="Pinecone API key for vector storage (optional)"
    )

    PINECONE_HOST: str | None = Field(
        default=None,
        description="Pinecone host URL (e.g., https://index.svc.environment.pinecone.io)",
    )

    # Local LLM Configuration
    LMSTUDIO_BASE_URL: str | None = Field(
        default="http://localhost:1234/v1", description="LM Studio API base URL for local inference"
    )

    LMSTUDIO_MODEL: str | None = Field(default=None, description="LM Studio model identifier (e.g., liquid/lfm2-1.2b)")

    LMSTUDIO_PRESET: str | None = Field(
        default=None,
        description=(
            "Optional preset name override (e.g., 'phi4-reasoning'). If None, "
            "preset is auto-detected from LMSTUDIO_MODEL via name_pattern matching."
        ),
    )

    LMSTUDIO_REASONING_EFFORT: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description=("LM Studio reasoning effort hint. Only honored if the active preset " "has reasoning=True."),
    )

    LMSTUDIO_TEMPERATURE: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "LMStudio default sampling temperature override. None = let preset "
            "or DEFAULT_LMSTUDIO_TEMPERATURE constant decide."
        ),
    )

    # MCTS Configuration with bounds validation
    MCTS_ENABLED: bool = Field(default=True, description="Enable MCTS for agent decision-making")

    MCTS_IMPL: MCTSImplementation = Field(
        default=MCTSImplementation.BASELINE, description="MCTS implementation variant to use"
    )

    MCTS_ITERATIONS: int = Field(default=100, ge=1, le=10000, description="Number of MCTS iterations (1-10000)")

    MCTS_C: float = Field(
        default=1.414,
        ge=0.0,
        le=10.0,
        description="MCTS exploration weight (UCB1 constant, 0.0-10.0)",
    )

    # Random seed for reproducibility
    SEED: int | None = Field(default=None, ge=0, description="Random seed for reproducibility (optional)")

    # LangSmith Configuration for tracing and evaluation
    LANGSMITH_API_KEY: SecretStr | None = Field(
        default=None, description="LangSmith API key for tracing and evaluation (optional)"
    )

    LANGSMITH_PROJECT: str = Field(default="langgraph-mcts", description="LangSmith project name")

    LANGCHAIN_TRACING_V2: bool = Field(default=False, description="Enable LangChain tracing v2")

    LANGCHAIN_ENDPOINT: str = Field(default="https://api.smith.langchain.com", description="LangChain API endpoint")

    # Weights & Biases Configuration for experiment tracking
    WANDB_API_KEY: SecretStr | None = Field(
        default=None, description="Weights & Biases API key for experiment tracking (optional)"
    )

    WANDB_PROJECT: str = Field(default="langgraph-mcts", description="W&B project name")

    WANDB_ENTITY: str | None = Field(default=None, description="W&B entity (username or team name)")

    WANDB_MODE: str = Field(default="online", description="W&B mode: online, offline, or disabled")

    # Logging Configuration
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, description="Application log level")

    # OpenTelemetry Configuration
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = Field(
        default=None, description="OpenTelemetry OTLP exporter endpoint URL"
    )

    # S3 Storage Configuration
    S3_BUCKET: str | None = Field(default=None, description="S3 bucket name for artifact storage")

    S3_PREFIX: str = Field(default="mcts-artifacts", description="S3 key prefix for stored artifacts")

    S3_REGION: str = Field(default="us-east-1", description="AWS region for S3 bucket")

    # Network Configuration (security)
    HTTP_TIMEOUT_SECONDS: int = Field(default=30, ge=1, le=300, description="HTTP request timeout in seconds")

    HTTP_MAX_RETRIES: int = Field(default=3, ge=0, le=10, description="Maximum HTTP request retries")

    # LLM Provider-Specific Timeouts (seconds)
    OPENAI_TIMEOUT: float = Field(default=60.0, ge=1.0, le=600.0, description="OpenAI API request timeout in seconds")

    ANTHROPIC_TIMEOUT: float = Field(
        default=120.0, ge=1.0, le=600.0, description="Anthropic API request timeout in seconds"
    )

    LMSTUDIO_TIMEOUT: float = Field(
        default=300.0, ge=1.0, le=600.0, description="LMStudio local inference timeout in seconds"
    )

    # Circuit Breaker Settings
    CIRCUIT_BREAKER_RESET_SECONDS: float = Field(
        default=60.0, ge=10.0, le=600.0, description="Circuit breaker reset time in seconds"
    )

    # CORS Configuration (security)
    CORS_ALLOWED_ORIGINS: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins. Empty list disables CORS (secure default).",
    )

    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, description="Allow credentials in CORS requests")

    # Rate Limit Response Configuration
    RATE_LIMIT_RETRY_AFTER_SECONDS: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Default Retry-After header value for rate limit responses",
    )

    # Security Settings
    MAX_QUERY_LENGTH: int = Field(
        default=10000, ge=1, le=100000, description="Maximum allowed query length in characters"
    )

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60, ge=1, le=1000, description="Rate limit for API requests per minute"
    )

    # Framework Service Configuration
    FRAMEWORK_MAX_ITERATIONS: int = Field(
        default=3, ge=1, le=100, description="Maximum iterations for agent processing"
    )

    FRAMEWORK_CONSENSUS_THRESHOLD: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Consensus threshold for agent agreement"
    )

    FRAMEWORK_TOP_K_RETRIEVAL: int = Field(
        default=5, ge=1, le=100, description="Number of top documents to retrieve for RAG"
    )

    FRAMEWORK_ENABLE_PARALLEL_AGENTS: bool = Field(default=True, description="Enable parallel agent execution")

    # LLM Generation Configuration
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for LLM generation")

    # Confidence Thresholds
    CONFIDENCE_WITH_RAG: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score when RAG context is used"
    )

    CONFIDENCE_WITHOUT_RAG: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence score when RAG is disabled"
    )

    CONFIDENCE_ON_ERROR: float = Field(default=0.3, ge=0.0, le=1.0, description="Confidence score on error responses")

    # Error Response Configuration
    ERROR_QUERY_PREVIEW_LENGTH: int = Field(
        default=100, ge=10, le=1000, description="Length of query preview in error logs"
    )

    # HRM Agent Configuration
    HRM_H_DIM: int = Field(default=512, ge=1, le=4096, description="HRM high-level hidden dimension")

    HRM_L_DIM: int = Field(default=256, ge=1, le=4096, description="HRM low-level hidden dimension")

    HRM_NUM_H_LAYERS: int = Field(default=2, ge=1, le=10, description="Number of HRM high-level layers")

    HRM_NUM_L_LAYERS: int = Field(default=4, ge=1, le=10, description="Number of HRM low-level layers")

    HRM_MAX_OUTER_STEPS: int = Field(default=10, ge=1, le=100, description="Maximum outer steps for HRM")

    HRM_HALT_THRESHOLD: float = Field(
        default=0.95, ge=0.0, le=1.0, description="HRM halt threshold for adaptive computation"
    )

    # TRM Agent Configuration
    TRM_LATENT_DIM: int = Field(default=256, ge=1, le=4096, description="TRM latent dimension")

    TRM_HIDDEN_DIM: int = Field(default=512, ge=1, le=4096, description="TRM hidden dimension")

    TRM_NUM_RECURSIONS: int = Field(default=16, ge=1, le=100, description="Number of TRM recursions")

    TRM_CONVERGENCE_THRESHOLD: float = Field(default=0.01, ge=0.0, le=1.0, description="TRM convergence threshold")

    # Meta-Controller Configuration
    META_CONTROLLER_TYPE: str = Field(default="rnn", description="Meta-controller type: rnn, bert, hybrid, assembly")

    META_CONTROLLER_INPUT_DIM: int = Field(default=10, ge=1, le=1024, description="Meta-controller input dimension")

    META_CONTROLLER_HIDDEN_DIM: int = Field(default=64, ge=1, le=1024, description="Meta-controller hidden dimension")

    META_CONTROLLER_NUM_LAYERS: int = Field(default=1, ge=1, le=10, description="Number of meta-controller layers")

    META_CONTROLLER_NUM_AGENTS: int = Field(
        default=3, ge=1, le=20, description="Number of agents for meta-controller routing"
    )

    META_CONTROLLER_DROPOUT: float = Field(default=0.1, ge=0.0, le=1.0, description="Meta-controller dropout rate")

    # Hybrid Agent Cost Tracking
    HYBRID_NEURAL_COST_PER_CALL: float = Field(
        default=0.000001, ge=0.0, description="Neural network cost per inference call (USD)"
    )

    HYBRID_LLM_COST_PER_1K_TOKENS: float = Field(default=0.03, ge=0.0, description="LLM cost per 1000 tokens (USD)")

    HYBRID_MODE: str = Field(default="auto", description="Hybrid agent mode: auto, neural_only, llm_only, adaptive")

    HYBRID_POLICY_CONFIDENCE_THRESHOLD: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Policy network confidence threshold"
    )

    HYBRID_VALUE_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Value network confidence threshold"
    )

    # MCTS Advanced Configuration
    MCTS_PROGRESSIVE_WIDENING_K: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Progressive widening constant K"
    )

    MCTS_PROGRESSIVE_WIDENING_ALPHA: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Progressive widening exponent alpha"
    )

    MCTS_MAX_PARALLEL_ROLLOUTS: int = Field(default=4, ge=1, le=32, description="Maximum parallel MCTS rollouts")

    MCTS_CACHE_SIZE_LIMIT: int = Field(
        default=10000, ge=100, le=1000000, description="MCTS simulation cache size limit"
    )

    MCTS_MAX_ROLLOUT_DEPTH: int = Field(default=50, ge=1, le=500, description="Maximum MCTS rollout depth")

    # ========================================
    # Chess / Stockfish Engine Configuration
    # ========================================

    STOCKFISH_PATH: str | None = Field(
        default=None, description="Path to Stockfish executable (auto-detected if not set)"
    )

    STOCKFISH_HASH_SIZE_MB: int = Field(default=128, ge=1, le=4096, description="Stockfish hash table size in MB")

    STOCKFISH_THREADS: int = Field(default=1, ge=1, le=64, description="Stockfish thread count")

    STOCKFISH_DEFAULT_DEPTH: int = Field(default=20, ge=1, le=100, description="Stockfish default search depth")

    STOCKFISH_TIME_LIMIT_MS: int = Field(
        default=1000, ge=100, le=60000, description="Stockfish time limit per move in ms"
    )

    STOCKFISH_MULTIPV: int = Field(default=1, ge=1, le=10, description="Stockfish multi-PV lines to analyze")

    # ========================================
    # Chess Verification Configuration
    # ========================================

    CHESS_VERIFICATION_AGREEMENT_THRESHOLD: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Ensemble agent agreement threshold"
    )

    CHESS_VERIFICATION_DIVERGENCE_THRESHOLD: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Agent divergence threshold"
    )

    CHESS_VERIFICATION_ROUTING_THRESHOLD: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Routing consistency threshold"
    )

    CHESS_VERIFICATION_VALUE_DIVERGENCE_THRESHOLD: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Value estimation divergence threshold"
    )

    CHESS_VERIFICATION_DEFAULT_CONFIDENCE: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Default confidence when agent doesn't provide one"
    )

    CHESS_VERIFICATION_LOG_VALIDATIONS: bool = Field(
        default=False, description="Log individual move validations (verbose)"
    )

    # Chess Phase-Specific Routing Scores
    CHESS_ROUTING_SCORE_MATCH: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Score when primary agent matches expected"
    )

    CHESS_ROUTING_SCORE_MIDDLEGAME_FALLBACK: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Fallback score in middlegame"
    )

    CHESS_ROUTING_SCORE_PHASE_APPROPRIATE: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Score for phase-appropriate agent choice"
    )

    CHESS_ROUTING_SCORE_PHASE_MISMATCH: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Score for phase-mismatched agent choice"
    )

    CHESS_ROUTING_SCORE_DEFAULT: float = Field(default=0.5, ge=0.0, le=1.0, description="Default routing score")

    # ========================================
    # Chess Piece Values (Centipawns)
    # ========================================

    CHESS_PIECE_VALUE_PAWN: int = Field(default=100, ge=50, le=200, description="Pawn value in centipawns")

    CHESS_PIECE_VALUE_KNIGHT: int = Field(default=320, ge=250, le=400, description="Knight value in centipawns")

    CHESS_PIECE_VALUE_BISHOP: int = Field(default=330, ge=250, le=400, description="Bishop value in centipawns")

    CHESS_PIECE_VALUE_ROOK: int = Field(default=500, ge=400, le=600, description="Rook value in centipawns")

    CHESS_PIECE_VALUE_QUEEN: int = Field(default=900, ge=800, le=1100, description="Queen value in centipawns")

    # Chess FEN Logging Configuration
    CHESS_FEN_LOG_TRUNCATE_LENGTH: int = Field(
        default=40, ge=20, le=100, description="FEN truncation length for logging"
    )

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate OpenAI API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "sk-xxx", "REPLACE_ME"):
                raise ValueError("OpenAI API key appears to be a placeholder value")
            if not secret_value.startswith("sk-"):
                raise ValueError("OpenAI API key should start with 'sk-'")
            if len(secret_value) < 20:
                raise ValueError("OpenAI API key appears to be too short")
        return v

    @field_validator("ANTHROPIC_API_KEY")
    @classmethod
    def validate_anthropic_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Anthropic API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("Anthropic API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("Anthropic API key appears to be too short")
        return v

    @field_validator("BRAINTRUST_API_KEY")
    @classmethod
    def validate_braintrust_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Braintrust API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("Braintrust API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("Braintrust API key appears to be too short")
        return v

    @field_validator("PINECONE_API_KEY")
    @classmethod
    def validate_pinecone_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Pinecone API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            # Check for obviously invalid patterns
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("Pinecone API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("Pinecone API key appears to be too short")
        return v

    @field_validator("LANGSMITH_API_KEY")
    @classmethod
    def validate_langsmith_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate LangSmith API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("LangSmith API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("LangSmith API key appears to be too short")
        return v

    @field_validator("WANDB_API_KEY")
    @classmethod
    def validate_wandb_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Weights & Biases API key format without exposing the value."""
        if v is not None:
            secret_value = v.get_secret_value()
            if secret_value in ("", "your-api-key-here", "REPLACE_ME"):
                raise ValueError("W&B API key appears to be a placeholder value")
            if len(secret_value) < 20:
                raise ValueError("W&B API key appears to be too short")
        return v

    @field_validator("PINECONE_HOST")
    @classmethod
    def validate_pinecone_host(cls, v: str | None) -> str | None:
        """Validate Pinecone host URL format."""
        if v is not None and v != "":
            if not v.startswith("https://"):
                raise ValueError("Pinecone host must start with https://")
            if "pinecone.io" not in v:
                raise ValueError("Pinecone host should be a valid pinecone.io URL")
        return v

    @field_validator("LMSTUDIO_BASE_URL")
    @classmethod
    def validate_lmstudio_url(cls, v: str | None) -> str | None:
        """Validate LM Studio base URL format."""
        if v is not None:
            if not v.startswith(("http://", "https://")):
                raise ValueError("LM Studio base URL must start with http:// or https://")
            # Warn if not localhost (potential security concern)
            if not any(host in v for host in ("localhost", "127.0.0.1", "::1")):
                import warnings

                warnings.warn(
                    "LM Studio URL points to non-localhost address. Ensure this is intentional and secure.",
                    UserWarning,
                    stacklevel=2,
                )
        return v

    @field_validator("OTEL_EXPORTER_OTLP_ENDPOINT")
    @classmethod
    def validate_otel_endpoint(cls, v: str | None) -> str | None:
        """Validate OpenTelemetry endpoint URL."""
        if v is not None and v != "" and not v.startswith(("http://", "https://", "grpc://")):
            raise ValueError("OpenTelemetry endpoint must start with http://, https://, or grpc://")
        return v

    @field_validator("S3_BUCKET")
    @classmethod
    def validate_s3_bucket_name(cls, v: str | None) -> str | None:
        """Validate S3 bucket name format."""
        if v is not None:
            # S3 bucket naming rules
            if len(v) < 3 or len(v) > 63:
                raise ValueError("S3 bucket name must be 3-63 characters long")
            if not v.replace("-", "").replace(".", "").isalnum():
                raise ValueError("S3 bucket name can only contain lowercase letters, numbers, hyphens, and periods")
            if v.startswith("-") or v.endswith("-"):
                raise ValueError("S3 bucket name cannot start or end with a hyphen")
        return v

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "Settings":
        """Ensure required API keys are provided for the selected provider."""
        if self.LLM_PROVIDER == LLMProvider.OPENAI:
            if self.OPENAI_API_KEY is None:
                raise ValueError(
                    "OPENAI_API_KEY is required when using OpenAI provider. "
                    "Set the OPENAI_API_KEY environment variable."
                )
        elif self.LLM_PROVIDER == LLMProvider.ANTHROPIC:
            if self.ANTHROPIC_API_KEY is None:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when using Anthropic provider. "
                    "Set the ANTHROPIC_API_KEY environment variable."
                )
        elif self.LLM_PROVIDER == LLMProvider.LMSTUDIO and self.LMSTUDIO_BASE_URL is None:
            raise ValueError("LMSTUDIO_BASE_URL is required when using LM Studio provider.")
        return self

    def get_api_key(self) -> str | None:
        """
        Get the API key for the current provider.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.LLM_PROVIDER == LLMProvider.OPENAI and self.OPENAI_API_KEY:
            return str(self.OPENAI_API_KEY.get_secret_value())
        elif self.LLM_PROVIDER == LLMProvider.ANTHROPIC and self.ANTHROPIC_API_KEY:
            return str(self.ANTHROPIC_API_KEY.get_secret_value())
        return None

    def safe_dict(self) -> dict:
        """
        Return settings as dictionary with secrets masked.

        Safe for logging and display purposes.
        """
        data: dict = self.model_dump()
        # Mask all sensitive fields
        secret_fields = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "BRAINTRUST_API_KEY",
            "PINECONE_API_KEY",
            "LANGSMITH_API_KEY",
            "WANDB_API_KEY",
        ]
        for field in secret_fields:
            if field in data and data[field]:
                data[field] = "***MASKED***"
        return data

    def get_braintrust_api_key(self) -> str | None:
        """
        Get the Braintrust API key if configured.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.BRAINTRUST_API_KEY:
            return str(self.BRAINTRUST_API_KEY.get_secret_value())
        return None

    def get_pinecone_api_key(self) -> str | None:
        """
        Get the Pinecone API key if configured.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.PINECONE_API_KEY:
            return str(self.PINECONE_API_KEY.get_secret_value())
        return None

    def get_langsmith_api_key(self) -> str | None:
        """
        Get the LangSmith API key if configured.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.LANGSMITH_API_KEY:
            return str(self.LANGSMITH_API_KEY.get_secret_value())
        return None

    def get_wandb_api_key(self) -> str | None:
        """
        Get the Weights & Biases API key if configured.

        Returns the secret value - use with caution to avoid logging.
        """
        if self.WANDB_API_KEY:
            return str(self.WANDB_API_KEY.get_secret_value())
        return None

    def __repr__(self) -> str:
        """Safe string representation that doesn't expose secrets."""
        return f"Settings(LLM_PROVIDER={self.LLM_PROVIDER}, MCTS_ENABLED={self.MCTS_ENABLED}, MCTS_IMPL={self.MCTS_IMPL}, LOG_LEVEL={self.LOG_LEVEL})"


# Global settings instance (lazily loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Settings are loaded once and cached. To reload, call reset_settings() first.

    Returns:
        Settings: Application configuration instance

    Raises:
        ValidationError: If configuration is invalid
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """
    Reset the global settings instance.

    Forces settings to be reloaded from environment on next get_settings() call.
    Useful for testing.
    """
    global _settings
    _settings = None


# Type exports for external use
__all__ = [
    "Settings",
    "LLMProvider",
    "LogLevel",
    "MCTSImplementation",
    "get_settings",
    "reset_settings",
]
