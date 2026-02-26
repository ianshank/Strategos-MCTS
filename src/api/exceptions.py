"""
Custom exception hierarchy for LangGraph Multi-Agent MCTS Framework.

Provides:
- Sanitized error messages for production
- Structured error information for logging
- Clear separation between user-facing and internal errors
"""

import re
from datetime import datetime
from typing import Any


class FrameworkError(Exception):
    """
    Base exception for all framework errors.

    Provides sanitized user-facing messages while preserving
    internal details for logging.
    """

    def __init__(
        self,
        user_message: str,
        internal_details: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize framework error.

        Args:
            user_message: Safe message to show to users
            internal_details: Detailed information for logs (may contain sensitive data)
            error_code: Machine-readable error code
            context: Additional context for debugging
        """
        self.user_message = user_message
        self.internal_details = internal_details or user_message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.timestamp = datetime.utcnow()

        super().__init__(user_message)

    def sanitize_details(self) -> str:
        """
        Remove sensitive information from internal details.

        Sanitizes:
        - File paths
        - API keys
        - Passwords
        - Connection strings
        - IP addresses
        """
        sanitized = self.internal_details

        # Remove file paths (Unix and Windows)
        sanitized = re.sub(r"/[\w/.-]+", "/***", sanitized)
        sanitized = re.sub(r"[A-Za-z]:\\[\w\\.-]+", "C:\\***", sanitized)

        # Remove API keys and secrets
        sanitized = re.sub(
            r"(api[_-]?key|secret|password|token|credential)[\s=:]+[\S]+", r"\1=***", sanitized, flags=re.IGNORECASE
        )

        # Remove connection strings
        sanitized = re.sub(r"(mongodb|postgresql|mysql|redis)://[^\s]+", r"\1://***", sanitized, flags=re.IGNORECASE)

        # Remove IP addresses
        sanitized = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "***.***.***", sanitized)

        # Remove email addresses
        sanitized = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "***@***", sanitized)

        return sanitized

    def to_log_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for structured logging.

        Returns sanitized version safe for logs.
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "user_message": self.user_message,
            "sanitized_details": self.sanitize_details(),
            "timestamp": self.timestamp.isoformat(),
            "context": {k: str(v) for k, v in self.context.items()},
        }

    def to_user_response(self) -> dict[str, Any]:
        """
        Convert exception to safe user-facing response.
        """
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.user_message,
            "timestamp": self.timestamp.isoformat(),
        }


class ValidationError(FrameworkError):
    """Raised when input validation fails."""

    def __init__(
        self,
        user_message: str = "Invalid input provided",
        internal_details: str | None = None,
        field_name: str | None = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if field_name:
            context["field_name"] = field_name
        super().__init__(
            user_message=user_message,
            internal_details=internal_details,
            error_code="VALIDATION_ERROR",
            context=context,
            **kwargs,
        )
        self.field_name = field_name


class AuthenticationError(FrameworkError):
    """Raised when authentication fails."""

    def __init__(self, user_message: str = "Authentication failed", internal_details: str | None = None, **kwargs):
        super().__init__(
            user_message=user_message, internal_details=internal_details, error_code="AUTH_ERROR", **kwargs
        )


class AuthorizationError(FrameworkError):
    """Raised when authorization fails."""

    def __init__(
        self,
        user_message: str = "Access denied",
        internal_details: str | None = None,
        required_permission: str | None = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if required_permission:
            context["required_permission"] = required_permission
        super().__init__(
            user_message=user_message,
            internal_details=internal_details,
            error_code="AUTHZ_ERROR",
            context=context,
            **kwargs,
        )


class RateLimitError(FrameworkError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        user_message: str = "Rate limit exceeded. Please try again later.",
        internal_details: str | None = None,
        retry_after_seconds: int | None = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if retry_after_seconds:
            context["retry_after_seconds"] = retry_after_seconds
        super().__init__(
            user_message=user_message,
            internal_details=internal_details,
            error_code="RATE_LIMIT",
            context=context,
            **kwargs,
        )
        self.retry_after_seconds = retry_after_seconds


class LLMError(FrameworkError):
    """Raised when LLM operations fail."""

    def __init__(
        self,
        user_message: str = "Language model service temporarily unavailable",
        internal_details: str | None = None,
        provider: str | None = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if provider:
            context["provider"] = provider
        super().__init__(
            user_message=user_message,
            internal_details=internal_details,
            error_code="LLM_ERROR",
            context=context,
            **kwargs,
        )


class MCTSError(FrameworkError):
    """Raised when MCTS simulation fails."""

    def __init__(
        self,
        user_message: str = "Tactical simulation failed",
        internal_details: str | None = None,
        iteration: int | None = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if iteration is not None:
            context["iteration"] = iteration
        super().__init__(
            user_message=user_message,
            internal_details=internal_details,
            error_code="MCTS_ERROR",
            context=context,
            **kwargs,
        )


class RAGError(FrameworkError):
    """Raised when RAG retrieval fails."""

    def __init__(self, user_message: str = "Context retrieval failed", internal_details: str | None = None, **kwargs):
        super().__init__(user_message=user_message, internal_details=internal_details, error_code="RAG_ERROR", **kwargs)


class TimeoutError(FrameworkError):
    """Raised when operation times out."""

    def __init__(
        self,
        user_message: str = "Operation timed out",
        internal_details: str | None = None,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if operation:
            context["operation"] = operation
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        super().__init__(
            user_message=user_message,
            internal_details=internal_details,
            error_code="TIMEOUT",
            context=context,
            **kwargs,
        )


class ConfigurationError(FrameworkError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        user_message: str = "System configuration error",
        internal_details: str | None = None,
        config_key: str | None = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if config_key:
            context["config_key"] = config_key
        super().__init__(
            user_message=user_message,
            internal_details=internal_details,
            error_code="CONFIG_ERROR",
            context=context,
            **kwargs,
        )


# Convenience function for wrapping exceptions
def wrap_exception(
    exc: Exception, user_message: str = "An unexpected error occurred", error_class: type = FrameworkError, **kwargs
) -> FrameworkError:
    """
    Wrap a standard exception in a FrameworkError with sanitized details.

    Args:
        exc: Original exception
        user_message: Safe user-facing message
        error_class: FrameworkError subclass to use
        **kwargs: Additional context

    Returns:
        FrameworkError instance with sanitized details
    """
    internal_details = f"{type(exc).__name__}: {str(exc)}"
    result = error_class(user_message=user_message, internal_details=internal_details, **kwargs)
    if not isinstance(result, FrameworkError):
        raise TypeError(f"error_class must produce a FrameworkError, got {type(result)}")
    return result
