"""
Input validation models for LangGraph Multi-Agent MCTS framework.

Provides:
- Pydantic models for all external inputs
- Query sanitization and length limits
- Configuration validation
- MCP tool input validation with strict type checking
- Security-focused input processing
"""

from datetime import datetime
import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Constants for validation
MAX_QUERY_LENGTH = 10000
MIN_QUERY_LENGTH = 1
MAX_CONTEXT_LENGTH = 50000
MAX_ITERATIONS = 10000
MIN_ITERATIONS = 1
MAX_EXPLORATION_WEIGHT = 10.0
MIN_EXPLORATION_WEIGHT = 0.0
MAX_BATCH_SIZE = 100


class QueryInput(BaseModel):
    """
    Validated query input for the multi-agent framework.

    Performs sanitization and security checks on user queries.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        extra="forbid",
    )

    query: str = Field(
        ..., min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH, description="User query to process"
    )

    use_rag: bool = Field(default=True, description="Enable RAG context retrieval")

    use_mcts: bool = Field(default=False, description="Enable MCTS simulation for tactical planning")

    thread_id: str | None = Field(
        default=None,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Conversation thread ID for state persistence",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """
        Sanitize query input for security.

        Removes potentially dangerous patterns while preserving legitimate content.
        """
        # Strip leading/trailing whitespace
        v = v.strip()

        # Check for empty query after stripping
        if not v:
            raise ValueError("Query cannot be empty or contain only whitespace")

        # Remove null bytes
        v = v.replace("\x00", "")

        # Limit consecutive whitespace
        v = re.sub(r"\s+", " ", v)

        # Check for suspicious patterns (basic injection prevention)
        suspicious_patterns = [
            r"<script[^>]*>",  # Script tags
            r"javascript:",  # JavaScript URLs
            r"on\w+\s*=",  # Event handlers
            r"\{\{.*\}\}",  # Template injection
            r"\$\{.*\}",  # Template literals
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Query contains potentially unsafe content matching pattern: {pattern}")

        return v

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, v: str | None) -> str | None:
        """Validate thread ID format for safe storage keys."""
        if v is not None:  # noqa: SIM102
            # Additional safety check beyond pattern
            if ".." in v or "/" in v or "\\" in v:
                raise ValueError("Thread ID contains invalid path characters")
        return v


class MCTSConfig(BaseModel):
    """
    Validated MCTS configuration parameters.

    Enforces bounds on exploration weight and iteration counts.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    iterations: int = Field(
        default=100, ge=MIN_ITERATIONS, le=MAX_ITERATIONS, description="Number of MCTS simulation iterations"
    )

    exploration_weight: float = Field(
        default=1.414,
        ge=MIN_EXPLORATION_WEIGHT,
        le=MAX_EXPLORATION_WEIGHT,
        description="UCB1 exploration constant (c parameter)",
    )

    max_depth: int = Field(default=10, ge=1, le=50, description="Maximum tree depth for MCTS expansion")

    simulation_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Timeout for MCTS simulation phase"
    )

    @field_validator("exploration_weight")
    @classmethod
    def validate_exploration_weight(cls, v: float) -> float:
        """Validate exploration weight is within reasonable bounds."""
        if not (MIN_EXPLORATION_WEIGHT <= v <= MAX_EXPLORATION_WEIGHT):
            raise ValueError(
                f"Exploration weight must be between {MIN_EXPLORATION_WEIGHT} and {MAX_EXPLORATION_WEIGHT}"
            )
        # Warn for unusual values
        if v < 0.5 or v > 3.0:
            import warnings

            warnings.warn(
                f"Exploration weight {v} is outside typical range (0.5-3.0). "
                "This may lead to suboptimal search behavior.",
                UserWarning,
                stacklevel=2,
            )
        return v


class AgentConfig(BaseModel):
    """
    Validated configuration for HRM/TRM agents.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    max_iterations: int = Field(default=3, ge=1, le=20, description="Maximum iterations for agent refinement")

    consensus_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Consensus threshold for agent agreement"
    )

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature for response generation")

    max_tokens: int = Field(default=2048, ge=1, le=128000, description="Maximum tokens in LLM response")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within LLM bounds."""
        if v < 0.0 or v > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class RAGConfig(BaseModel):
    """
    Validated RAG (Retrieval Augmented Generation) configuration.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    top_k: int = Field(default=5, ge=1, le=50, description="Number of documents to retrieve")

    similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum similarity score for retrieved documents"
    )

    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Document chunk size for embedding")

    chunk_overlap: int = Field(default=200, ge=0, le=2000, description="Overlap between document chunks")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "RAGConfig":
        """Ensure chunk overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return self


class MCPToolInput(BaseModel):
    """
    Base validation model for MCP (Model Context Protocol) tool inputs.

    Provides strict validation for external tool invocations.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    tool_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",
        description="Name of the MCP tool to invoke",
    )

    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters as key-value pairs")

    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Timeout for tool execution")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool name is safe and follows naming conventions."""
        # Prevent path traversal in tool names
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Tool name contains invalid characters")

        # Prevent overly long names
        if len(v) > 100:
            raise ValueError("Tool name exceeds maximum length of 100 characters")

        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate tool parameters for security."""
        # Check for reasonable size
        if len(str(v)) > 100000:
            raise ValueError("Tool parameters exceed maximum size")

        # Check parameter count
        if len(v) > 50:
            raise ValueError("Too many parameters (maximum 50)")

        # Validate parameter keys
        for key in v:
            if not isinstance(key, str):
                raise ValueError("Parameter keys must be strings")
            if len(key) > 100:
                raise ValueError(f"Parameter key '{key[:20]}...' exceeds maximum length")
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid parameter key format: {key}")

        return v


class FileReadInput(MCPToolInput):
    """
    Validated input for file reading operations.

    Implements path traversal protection.
    """

    tool_name: str = Field(default="read_file", frozen=True)

    file_path: str = Field(..., min_length=1, max_length=1000, description="Path to file to read")

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path for security concerns."""
        # Normalize path
        v = v.strip()

        # Check for path traversal attempts
        if ".." in v:
            raise ValueError("Path traversal detected: '..' not allowed in file path")

        # Check for absolute paths (may be allowed in some contexts)
        if v.startswith("/"):
            import warnings

            warnings.warn(
                "Absolute file path provided. Ensure this is within allowed directories.", UserWarning, stacklevel=2
            )

        # Check for suspicious patterns
        suspicious = [
            "/etc/",
            "/root/",
            "~/.ssh/",
            "/var/",
            "\\windows\\",
            "\\system32\\",
        ]
        for pattern in suspicious:
            if pattern.lower() in v.lower():
                raise ValueError(f"File path contains restricted directory: {pattern}")

        return v


class WebFetchInput(MCPToolInput):
    """
    Validated input for web fetch operations.

    Implements URL validation and security checks.
    """

    tool_name: str = Field(default="web_fetch", frozen=True)

    url: str = Field(..., min_length=1, max_length=2000, description="URL to fetch")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL for security."""
        v = v.strip()

        # Must start with https:// for security (http:// only for local)
        if not v.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("URL must use HTTPS protocol (except for localhost)")

        # Check for suspicious patterns
        if any(char in v for char in ["<", ">", "'", '"', ";"]):
            raise ValueError("URL contains invalid characters")

        # Validate basic URL structure
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, v, re.IGNORECASE):
            raise ValueError("Invalid URL format")

        return v


class BatchQueryInput(BaseModel):
    """
    Validated batch query input for processing multiple queries.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    queries: list[QueryInput] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE, description="List of queries to process in batch"
    )

    parallel: bool = Field(default=False, description="Process queries in parallel (if system supports)")

    @field_validator("queries")
    @classmethod
    def validate_batch_size(cls, v: list[QueryInput]) -> list[QueryInput]:
        """Validate batch doesn't exceed limits."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
        if len(v) == 0:
            raise ValueError("Batch must contain at least one query")
        return v


class APIRequestMetadata(BaseModel):
    """
    Metadata for API request tracking and audit logging.

    Used for security monitoring and rate limiting.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str = Field(
        ..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$", description="Unique request identifier"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp (UTC)")

    client_id: str | None = Field(
        default=None, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$", description="Client identifier for rate limiting"
    )

    source_ip: str | None = Field(default=None, description="Source IP address (for audit logging)")

    @field_validator("source_ip")
    @classmethod
    def validate_ip_address(cls, v: str | None) -> str | None:
        """Validate IP address format."""
        if v is not None:
            # Basic IPv4/IPv6 validation
            import ipaddress

            try:
                ipaddress.ip_address(v)
            except ValueError as e:
                raise ValueError(f"Invalid IP address format: {v}") from e
        return v


# Convenience functions for common validation patterns


def validate_query(query: str, **kwargs) -> QueryInput:
    """
    Validate a query string and return a validated QueryInput model.

    Args:
        query: Raw query string
        **kwargs: Additional query parameters

    Returns:
        QueryInput: Validated query model

    Raises:
        ValidationError: If validation fails
    """
    return QueryInput(query=query, **kwargs)


def validate_mcts_config(**kwargs) -> MCTSConfig:
    """
    Validate MCTS configuration parameters.

    Args:
        **kwargs: MCTS configuration parameters

    Returns:
        MCTSConfig: Validated configuration

    Raises:
        ValidationError: If validation fails
    """
    return MCTSConfig(**kwargs)


def validate_tool_input(tool_name: str, parameters: dict[str, Any], **kwargs) -> MCPToolInput:
    """
    Validate MCP tool input parameters.

    Args:
        tool_name: Name of the tool
        parameters: Tool parameters
        **kwargs: Additional options

    Returns:
        MCPToolInput: Validated tool input

    Raises:
        ValidationError: If validation fails
    """
    return MCPToolInput(tool_name=tool_name, parameters=parameters, **kwargs)


# Type exports
__all__ = [
    "QueryInput",
    "MCTSConfig",
    "AgentConfig",
    "RAGConfig",
    "MCPToolInput",
    "FileReadInput",
    "WebFetchInput",
    "BatchQueryInput",
    "APIRequestMetadata",
    "validate_query",
    "validate_mcts_config",
    "validate_tool_input",
]
