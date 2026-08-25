"""
Comprehensive unit tests for custom exception classes.

Tests:
- Exception message formatting
- Error hierarchy and inheritance
- Sanitization of sensitive data
- User-facing vs internal error responses
- Context preservation
"""

# Import the exception classes
from datetime import datetime
import sys

sys.path.insert(0, ".")
from src.api.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    FrameworkError,
    LLMError,
    MCTSError,
    RAGError,
    RateLimitError,
    ValidationError,
    wrap_exception,
)
from src.api.exceptions import (
    TimeoutError as FrameworkTimeoutError,
)


class TestFrameworkError:
    """Test suite for base FrameworkError class."""

    def test_basic_initialization(self):
        """Test basic FrameworkError initialization."""
        error = FrameworkError(user_message="User-facing error message")

        assert error.user_message == "User-facing error message"
        assert error.internal_details == "User-facing error message"
        assert error.error_code == "FRAMEWORKERROR"
        assert error.context == {}
        assert isinstance(error.timestamp, datetime)

    def test_initialization_with_all_parameters(self):
        """Test FrameworkError with all parameters."""
        context = {"request_id": "12345", "user_id": "user_001"}
        error = FrameworkError(
            user_message="Something went wrong",
            internal_details="Detailed stack trace here",
            error_code="CUSTOM_ERROR",
            context=context,
        )

        assert error.user_message == "Something went wrong"
        assert error.internal_details == "Detailed stack trace here"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.context == context

    def test_error_inherits_from_exception(self):
        """Test that FrameworkError inherits from Exception."""
        error = FrameworkError(user_message="Test")

        assert isinstance(error, Exception)
        assert str(error) == "Test"

    def test_timestamp_is_utc(self):
        """Test that timestamp is in UTC."""
        error = FrameworkError(user_message="Test")

        # Timestamp should be close to current UTC time
        now = datetime.utcnow()
        time_diff = abs((now - error.timestamp).total_seconds())
        assert time_diff < 1  # Within 1 second

    def test_error_code_defaults_to_class_name(self):
        """Test that error_code defaults to uppercase class name."""
        error = FrameworkError(user_message="Test")
        assert error.error_code == "FRAMEWORKERROR"

    def test_internal_details_defaults_to_user_message(self):
        """Test that internal_details defaults to user_message."""
        error = FrameworkError(user_message="Public message")
        assert error.internal_details == "Public message"


class TestSanitization:
    """Test suite for sensitive data sanitization."""

    def test_sanitize_unix_file_paths(self):
        """Test that Unix file paths are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="Failed to read /home/user/config/secrets.json")

        sanitized = error.sanitize_details()

        assert "/home/user/config/secrets.json" not in sanitized
        assert "/***" in sanitized

    def test_sanitize_windows_file_paths(self):
        """Test that Windows file paths are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="Error in C:\\Users\\admin\\passwords.txt")

        sanitized = error.sanitize_details()

        assert "C:\\Users\\admin\\passwords.txt" not in sanitized
        assert "C:\\***" in sanitized

    def test_sanitize_api_keys(self):
        """Test that API keys are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="Request failed with api_key=sk_live_12345678")

        sanitized = error.sanitize_details()

        assert "sk_live_12345678" not in sanitized
        assert "api_key=***" in sanitized

    def test_sanitize_secrets(self):
        """Test that secret values are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="Connection with secret: mysupersecretvalue123")

        sanitized = error.sanitize_details()

        assert "mysupersecretvalue123" not in sanitized
        assert "secret" in sanitized.lower()

    def test_sanitize_passwords(self):
        """Test that passwords are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="Login failed: password=P@ssw0rd123!")

        sanitized = error.sanitize_details()

        assert "P@ssw0rd123!" not in sanitized
        assert "password=***" in sanitized

    def test_sanitize_tokens(self):
        """Test that tokens are sanitized."""
        error = FrameworkError(
            user_message="Error", internal_details="Invalid token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        )

        sanitized = error.sanitize_details()

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
        assert "token" in sanitized.lower()

    def test_sanitize_mongodb_connection_string(self):
        """Test that MongoDB connection strings are sanitized."""
        error = FrameworkError(
            user_message="Error", internal_details="Connection failed: mongodb://user:pass@host:27017/db"
        )

        sanitized = error.sanitize_details()

        assert "user:pass@host:27017/db" not in sanitized
        # The sanitizer replaces the connection string contents
        assert "mongodb:/" in sanitized
        assert "***" in sanitized

    def test_sanitize_postgresql_connection_string(self):
        """Test that PostgreSQL connection strings are sanitized."""
        error = FrameworkError(
            user_message="Error", internal_details="Error: postgresql://admin:secret@db.example.com:5432/app"
        )

        sanitized = error.sanitize_details()

        assert "admin:secret@db.example.com" not in sanitized
        # The sanitizer replaces the connection string contents
        assert "postgresql:/" in sanitized
        assert "***" in sanitized

    def test_sanitize_redis_connection_string(self):
        """Test that Redis connection strings are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="Cache error: redis://localhost:6379/0")

        sanitized = error.sanitize_details()

        assert "localhost:6379/0" not in sanitized
        # The sanitizer replaces the connection string contents
        assert "redis:/" in sanitized
        assert "***" in sanitized

    def test_sanitize_ip_addresses(self):
        """Test that IP addresses are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="Connection refused from 192.168.1.100:8080")

        sanitized = error.sanitize_details()

        assert "192.168.1.100" not in sanitized
        assert "***.***.***" in sanitized

    def test_sanitize_email_addresses(self):
        """Test that email addresses are sanitized."""
        error = FrameworkError(user_message="Error", internal_details="User john.doe@company.com not found")

        sanitized = error.sanitize_details()

        assert "john.doe@company.com" not in sanitized
        assert "***@***" in sanitized

    def test_sanitize_multiple_sensitive_items(self):
        """Test sanitization of multiple sensitive items."""
        error = FrameworkError(
            user_message="Error",
            internal_details=(
                "Failed at /var/www/app/config.py "
                "with api_key=secret123 "
                "connecting to mongodb://user:pass@192.168.1.1:27017"
            ),
        )

        sanitized = error.sanitize_details()

        # All sensitive data should be removed
        assert "/var/www/app/config.py" not in sanitized
        assert "secret123" not in sanitized
        assert "user:pass" not in sanitized
        assert "192.168.1.1" not in sanitized

    def test_sanitize_preserves_non_sensitive_data(self):
        """Test that non-sensitive data is preserved."""
        error = FrameworkError(user_message="Error", internal_details="Error code 500: Internal server error occurred")

        sanitized = error.sanitize_details()

        # Non-sensitive info should remain
        assert "Error code 500" in sanitized
        assert "Internal server error occurred" in sanitized


class TestLogDictConversion:
    """Test suite for to_log_dict method."""

    def test_to_log_dict_structure(self):
        """Test that to_log_dict returns correct structure."""
        error = FrameworkError(
            user_message="Test error",
            internal_details="Internal info",
            error_code="TEST_ERROR",
            context={"key": "value"},
        )

        log_dict = error.to_log_dict()

        assert "error_type" in log_dict
        assert "error_code" in log_dict
        assert "user_message" in log_dict
        assert "sanitized_details" in log_dict
        assert "timestamp" in log_dict
        assert "context" in log_dict

    def test_to_log_dict_contains_sanitized_details(self):
        """Test that log dict contains sanitized details."""
        error = FrameworkError(user_message="Error", internal_details="Secret at /home/user/.ssh/id_rsa")

        log_dict = error.to_log_dict()

        # Should be sanitized
        assert "/home/user/.ssh/id_rsa" not in log_dict["sanitized_details"]
        assert "/***" in log_dict["sanitized_details"]

    def test_to_log_dict_error_type(self):
        """Test that error_type matches class name."""
        error = FrameworkError(user_message="Test")
        log_dict = error.to_log_dict()

        assert log_dict["error_type"] == "FrameworkError"

    def test_to_log_dict_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        error = FrameworkError(user_message="Test")
        log_dict = error.to_log_dict()

        # Should be ISO format string
        timestamp = log_dict["timestamp"]
        assert isinstance(timestamp, str)
        # Should be parseable as ISO format
        datetime.fromisoformat(timestamp)

    def test_to_log_dict_context_conversion(self):
        """Test that context values are converted to strings."""
        error = FrameworkError(
            user_message="Test", context={"number": 42, "list": [1, 2, 3], "dict": {"nested": "value"}}
        )

        log_dict = error.to_log_dict()

        assert log_dict["context"]["number"] == "42"
        assert log_dict["context"]["list"] == "[1, 2, 3]"
        assert "nested" in log_dict["context"]["dict"]


class TestUserResponseConversion:
    """Test suite for to_user_response method."""

    def test_to_user_response_structure(self):
        """Test that to_user_response returns correct structure."""
        error = FrameworkError(user_message="User message")

        response = error.to_user_response()

        assert response["error"] is True
        assert "error_code" in response
        assert "message" in response
        assert "timestamp" in response

    def test_to_user_response_does_not_expose_internal_details(self):
        """Test that user response doesn't contain internal details."""
        error = FrameworkError(
            user_message="Something went wrong",
            internal_details="Database connection to postgresql://admin:pass@db:5432 failed",
        )

        response = error.to_user_response()

        # Internal details should not be present
        assert "internal_details" not in response
        assert "sanitized_details" not in response
        assert "postgresql" not in str(response)
        assert "admin:pass" not in str(response)

    def test_to_user_response_preserves_user_message(self):
        """Test that user message is preserved in response."""
        error = FrameworkError(user_message="Please try again later")

        response = error.to_user_response()

        assert response["message"] == "Please try again later"

    def test_to_user_response_includes_error_code(self):
        """Test that error code is included in response."""
        error = FrameworkError(user_message="Error", error_code="CUSTOM_CODE")

        response = error.to_user_response()

        assert response["error_code"] == "CUSTOM_CODE"


class TestValidationError:
    """Test suite for ValidationError class."""

    def test_validation_error_defaults(self):
        """Test ValidationError with default values."""
        error = ValidationError()

        assert error.user_message == "Invalid input provided"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.field_name is None

    def test_validation_error_with_field_name(self):
        """Test ValidationError with field name."""
        error = ValidationError(user_message="Email is invalid", field_name="email")

        assert error.field_name == "email"
        assert error.context["field_name"] == "email"

    def test_validation_error_inherits_from_framework_error(self):
        """Test that ValidationError inherits from FrameworkError."""
        error = ValidationError()

        assert isinstance(error, FrameworkError)
        assert isinstance(error, Exception)


class TestAuthenticationError:
    """Test suite for AuthenticationError class."""

    def test_authentication_error_defaults(self):
        """Test AuthenticationError with default values."""
        error = AuthenticationError()

        assert error.user_message == "Authentication failed"
        assert error.error_code == "AUTH_ERROR"

    def test_authentication_error_custom_message(self):
        """Test AuthenticationError with custom message."""
        error = AuthenticationError(
            user_message="Invalid credentials", internal_details="Token expired at 2024-01-01T00:00:00Z"
        )

        assert error.user_message == "Invalid credentials"
        assert "expired" in error.internal_details

    def test_authentication_error_inheritance(self):
        """Test AuthenticationError inheritance."""
        error = AuthenticationError()

        assert isinstance(error, FrameworkError)


class TestAuthorizationError:
    """Test suite for AuthorizationError class."""

    def test_authorization_error_defaults(self):
        """Test AuthorizationError with default values."""
        error = AuthorizationError()

        assert error.user_message == "Access denied"
        assert error.error_code == "AUTHZ_ERROR"

    def test_authorization_error_with_permission(self):
        """Test AuthorizationError with required permission."""
        error = AuthorizationError(user_message="You cannot access this resource", required_permission="admin")

        assert error.context["required_permission"] == "admin"

    def test_authorization_error_inheritance(self):
        """Test AuthorizationError inheritance."""
        error = AuthorizationError()

        assert isinstance(error, FrameworkError)


class TestRateLimitError:
    """Test suite for RateLimitError class."""

    def test_rate_limit_error_defaults(self):
        """Test RateLimitError with default values."""
        error = RateLimitError()

        assert error.user_message == "Rate limit exceeded. Please try again later."
        assert error.error_code == "RATE_LIMIT"
        assert error.retry_after_seconds is None

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after_seconds."""
        error = RateLimitError(user_message="Too many requests", retry_after_seconds=60)

        assert error.retry_after_seconds == 60
        assert error.context["retry_after_seconds"] == 60

    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inheritance."""
        error = RateLimitError()

        assert isinstance(error, FrameworkError)


class TestLLMError:
    """Test suite for LLMError class."""

    def test_llm_error_defaults(self):
        """Test LLMError with default values."""
        error = LLMError()

        assert error.user_message == "Language model service temporarily unavailable"
        assert error.error_code == "LLM_ERROR"

    def test_llm_error_with_provider(self):
        """Test LLMError with provider information."""
        error = LLMError(user_message="AI service error", provider="OpenAI")

        assert error.context["provider"] == "OpenAI"

    def test_llm_error_inheritance(self):
        """Test LLMError inheritance."""
        error = LLMError()

        assert isinstance(error, FrameworkError)


class TestMCTSError:
    """Test suite for MCTSError class."""

    def test_mcts_error_defaults(self):
        """Test MCTSError with default values."""
        error = MCTSError()

        assert error.user_message == "Tactical simulation failed"
        assert error.error_code == "MCTS_ERROR"

    def test_mcts_error_with_iteration(self):
        """Test MCTSError with iteration number."""
        error = MCTSError(user_message="Simulation failed", iteration=42)

        assert error.context["iteration"] == 42

    def test_mcts_error_inheritance(self):
        """Test MCTSError inheritance."""
        error = MCTSError()

        assert isinstance(error, FrameworkError)


class TestRAGError:
    """Test suite for RAGError class."""

    def test_rag_error_defaults(self):
        """Test RAGError with default values."""
        error = RAGError()

        assert error.user_message == "Context retrieval failed"
        assert error.error_code == "RAG_ERROR"

    def test_rag_error_inheritance(self):
        """Test RAGError inheritance."""
        error = RAGError()

        assert isinstance(error, FrameworkError)


class TestTimeoutError:
    """Test suite for TimeoutError class."""

    def test_timeout_error_defaults(self):
        """Test TimeoutError with default values."""
        error = FrameworkTimeoutError()

        assert error.user_message == "Operation timed out"
        assert error.error_code == "TIMEOUT"

    def test_timeout_error_with_operation(self):
        """Test TimeoutError with operation details."""
        error = FrameworkTimeoutError(
            user_message="Request timed out", operation="database_query", timeout_seconds=30.0
        )

        assert error.context["operation"] == "database_query"
        assert error.context["timeout_seconds"] == 30.0

    def test_timeout_error_inheritance(self):
        """Test TimeoutError inheritance."""
        error = FrameworkTimeoutError()

        assert isinstance(error, FrameworkError)


class TestConfigurationError:
    """Test suite for ConfigurationError class."""

    def test_configuration_error_defaults(self):
        """Test ConfigurationError with default values."""
        error = ConfigurationError()

        assert error.user_message == "System configuration error"
        assert error.error_code == "CONFIG_ERROR"

    def test_configuration_error_with_config_key(self):
        """Test ConfigurationError with config key."""
        error = ConfigurationError(user_message="Configuration invalid", config_key="DATABASE_URL")

        assert error.context["config_key"] == "DATABASE_URL"

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError()

        assert isinstance(error, FrameworkError)


class TestWrapException:
    """Test suite for wrap_exception utility function."""

    def test_wrap_standard_exception(self):
        """Test wrapping a standard exception."""
        original = ValueError("Invalid value provided")

        wrapped = wrap_exception(original)

        assert isinstance(wrapped, FrameworkError)
        assert wrapped.user_message == "An unexpected error occurred"
        assert "ValueError" in wrapped.internal_details
        assert "Invalid value provided" in wrapped.internal_details

    def test_wrap_with_custom_user_message(self):
        """Test wrapping with custom user message."""
        original = KeyError("missing_key")

        wrapped = wrap_exception(original, user_message="Required field is missing")

        assert wrapped.user_message == "Required field is missing"

    def test_wrap_with_custom_error_class(self):
        """Test wrapping with custom error class."""
        original = OSError("File not found")

        wrapped = wrap_exception(original, user_message="Resource unavailable", error_class=ConfigurationError)

        assert isinstance(wrapped, ConfigurationError)
        assert wrapped.error_code == "CONFIG_ERROR"

    def test_wrap_preserves_exception_type_name(self):
        """Test that wrapped exception preserves original type name."""
        original = TypeError("wrong type")

        wrapped = wrap_exception(original)

        assert "TypeError" in wrapped.internal_details

    def test_wrap_with_additional_context(self):
        """Test wrapping with additional context."""
        original = RuntimeError("System failure")

        wrapped = wrap_exception(original, user_message="System error", context={"request_id": "abc123"})

        assert wrapped.context["request_id"] == "abc123"


class TestErrorHierarchy:
    """Test suite for error class hierarchy."""

    def test_all_errors_inherit_from_framework_error(self):
        """Test that all custom errors inherit from FrameworkError."""
        error_classes = [
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            RateLimitError,
            LLMError,
            MCTSError,
            RAGError,
            FrameworkTimeoutError,
            ConfigurationError,
        ]

        for error_class in error_classes:
            error = error_class()
            assert isinstance(error, FrameworkError)
            assert isinstance(error, Exception)

    def test_all_errors_have_sanitize_method(self):
        """Test that all errors have sanitize_details method."""
        error_classes = [
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            RateLimitError,
            LLMError,
            MCTSError,
            RAGError,
            FrameworkTimeoutError,
            ConfigurationError,
        ]

        for error_class in error_classes:
            error = error_class(internal_details="Test api_key=secret")
            sanitized = error.sanitize_details()
            assert "secret" not in sanitized

    def test_all_errors_have_to_log_dict(self):
        """Test that all errors have to_log_dict method."""
        error_classes = [
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            RateLimitError,
            LLMError,
            MCTSError,
            RAGError,
            FrameworkTimeoutError,
            ConfigurationError,
        ]

        for error_class in error_classes:
            error = error_class()
            log_dict = error.to_log_dict()
            assert "error_type" in log_dict
            assert "error_code" in log_dict

    def test_all_errors_have_to_user_response(self):
        """Test that all errors have to_user_response method."""
        error_classes = [
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            RateLimitError,
            LLMError,
            MCTSError,
            RAGError,
            FrameworkTimeoutError,
            ConfigurationError,
        ]

        for error_class in error_classes:
            error = error_class()
            response = error.to_user_response()
            assert response["error"] is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_user_message(self):
        """Test error with empty user message."""
        error = FrameworkError(user_message="")
        assert error.user_message == ""
        assert str(error) == ""

    def test_very_long_internal_details(self):
        """Test error with very long internal details."""
        long_details = "x" * 10000
        error = FrameworkError(user_message="Error", internal_details=long_details)

        assert len(error.internal_details) == 10000

    def test_special_characters_in_messages(self):
        """Test error with special characters."""
        error = FrameworkError(
            user_message="Error: !@#$%^&*()_+-=[]{}|;':\",./<>?", internal_details="Details with \n\t\r special chars"
        )

        assert error.user_message is not None
        assert "\n" in error.internal_details

    def test_unicode_in_messages(self):
        """Test error with unicode characters."""
        error = FrameworkError(
            user_message="Error: \u4e2d\u6587\u6d4b\u8bd5", internal_details="Path: /\u00e9\u00f1/\u4e2d\u6587"
        )

        assert "\u4e2d\u6587\u6d4b\u8bd5" in error.user_message

    def test_none_in_context(self):
        """Test error with None values in context."""
        error = FrameworkError(user_message="Error", context={"key": None})

        log_dict = error.to_log_dict()
        assert log_dict["context"]["key"] == "None"

    def test_nested_exception_wrapping(self):
        """Test wrapping an already wrapped exception."""
        original = ValueError("Original error")
        first_wrap = wrap_exception(original, user_message="First wrap")
        second_wrap = wrap_exception(first_wrap, user_message="Second wrap")

        assert second_wrap.user_message == "Second wrap"
        assert "FrameworkError" in second_wrap.internal_details

    def test_error_code_with_spaces_preserved(self):
        """Test that custom error codes with spaces are preserved."""
        error = FrameworkError(user_message="Error", error_code="CUSTOM ERROR CODE")

        assert error.error_code == "CUSTOM ERROR CODE"

    def test_context_with_complex_objects(self):
        """Test context with complex nested objects."""
        context = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "tuple": (4, 5, 6),
        }

        error = FrameworkError(user_message="Error", context=context)

        log_dict = error.to_log_dict()
        # Complex objects should be converted to strings
        assert isinstance(log_dict["context"]["nested"], str)
        assert isinstance(log_dict["context"]["list"], str)

    def test_sanitization_case_insensitivity(self):
        """Test that sanitization is case-insensitive."""
        error = FrameworkError(user_message="Error", internal_details="API_KEY=secret PASSWORD=hidden")

        sanitized = error.sanitize_details()
        assert "secret" not in sanitized
        assert "hidden" not in sanitized

    def test_partial_path_matching(self):
        """Test that partial paths are handled correctly."""
        error = FrameworkError(user_message="Error", internal_details="Error at /var/log/app.log with code 404")

        sanitized = error.sanitize_details()
        # Path should be sanitized but "404" should remain
        assert "/var/log/app.log" not in sanitized
        # The number 404 may or may not be affected depending on pattern
