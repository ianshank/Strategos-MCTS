"""
Tests for custom exception hierarchy.

Tests FrameworkError, sanitization, all error subclasses,
and the wrap_exception utility.
"""

import pytest

from src.api.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    FrameworkError,
    LLMError,
    MCTSError,
    RAGError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    wrap_exception,
)


@pytest.mark.unit
class TestFrameworkError:
    def test_basic(self):
        err = FrameworkError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.user_message == "Something went wrong"
        assert err.internal_details == "Something went wrong"
        assert err.error_code == "FRAMEWORKERROR"
        assert err.context == {}
        assert err.timestamp is not None

    def test_with_details(self):
        err = FrameworkError(
            "Bad request",
            internal_details="Failed at /api/v1/query with key=sk-123",
            error_code="ERR001",
            context={"endpoint": "/query"},
        )
        assert err.internal_details == "Failed at /api/v1/query with key=sk-123"
        assert err.error_code == "ERR001"
        assert err.context["endpoint"] == "/query"

    def test_sanitize_api_keys(self):
        err = FrameworkError("error", internal_details="api_key=sk-abc123 failed")
        sanitized = err.sanitize_details()
        assert "sk-abc123" not in sanitized

    def test_sanitize_file_paths(self):
        err = FrameworkError("error", internal_details="Error in /home/user/secret.py")
        sanitized = err.sanitize_details()
        assert "/home/user/secret.py" not in sanitized

    def test_sanitize_connection_strings(self):
        err = FrameworkError("error", internal_details="mongodb://user:pass@host:27017/db")
        sanitized = err.sanitize_details()
        assert "user:pass" not in sanitized

    def test_sanitize_ip_addresses(self):
        err = FrameworkError("error", internal_details="Connected to 192.168.1.1")
        sanitized = err.sanitize_details()
        assert "192.168.1.1" not in sanitized

    def test_sanitize_emails(self):
        err = FrameworkError("error", internal_details="User admin@example.com")
        sanitized = err.sanitize_details()
        assert "admin@example.com" not in sanitized

    def test_to_log_dict(self):
        err = FrameworkError("test error", error_code="TEST")
        d = err.to_log_dict()
        assert d["error_type"] == "FrameworkError"
        assert d["error_code"] == "TEST"
        assert "timestamp" in d
        assert "sanitized_details" in d

    def test_to_user_response(self):
        err = FrameworkError("safe message", error_code="ERR")
        resp = err.to_user_response()
        assert resp["error"] is True
        assert resp["message"] == "safe message"
        assert resp["error_code"] == "ERR"


@pytest.mark.unit
class TestErrorSubclasses:
    def test_validation_error(self):
        err = ValidationError(field_name="email")
        assert err.error_code == "VALIDATION_ERROR"
        assert err.field_name == "email"
        assert err.context["field_name"] == "email"

    def test_authentication_error(self):
        err = AuthenticationError()
        assert err.error_code == "AUTH_ERROR"

    def test_authorization_error(self):
        err = AuthorizationError(required_permission="admin")
        assert err.error_code == "AUTHZ_ERROR"
        assert err.context["required_permission"] == "admin"

    def test_rate_limit_error(self):
        err = RateLimitError(retry_after_seconds=30)
        assert err.error_code == "RATE_LIMIT"
        assert err.retry_after_seconds == 30

    def test_llm_error(self):
        err = LLMError(provider="openai")
        assert err.error_code == "LLM_ERROR"
        assert err.context["provider"] == "openai"

    def test_mcts_error(self):
        err = MCTSError(iteration=42)
        assert err.error_code == "MCTS_ERROR"
        assert err.context["iteration"] == 42

    def test_rag_error(self):
        err = RAGError()
        assert err.error_code == "RAG_ERROR"

    def test_timeout_error(self):
        err = TimeoutError(operation="search", timeout_seconds=30.0)
        assert err.error_code == "TIMEOUT"
        assert err.context["operation"] == "search"

    def test_configuration_error(self):
        err = ConfigurationError(config_key="API_KEY")
        assert err.error_code == "CONFIG_ERROR"
        assert err.context["config_key"] == "API_KEY"

    def test_all_are_framework_errors(self):
        classes = [
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            RateLimitError,
            LLMError,
            MCTSError,
            RAGError,
            TimeoutError,
            ConfigurationError,
        ]
        for cls in classes:
            assert issubclass(cls, FrameworkError)


@pytest.mark.unit
class TestWrapException:
    def test_basic(self):
        orig = ValueError("bad value")
        wrapped = wrap_exception(orig)
        assert isinstance(wrapped, FrameworkError)
        assert "ValueError" in wrapped.internal_details

    def test_custom_message(self):
        wrapped = wrap_exception(RuntimeError("oops"), user_message="Service error")
        assert wrapped.user_message == "Service error"

    def test_custom_error_class(self):
        wrapped = wrap_exception(RuntimeError("x"), error_class=LLMError)
        assert isinstance(wrapped, LLMError)
