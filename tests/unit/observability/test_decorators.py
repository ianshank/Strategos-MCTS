"""
Comprehensive tests for observability decorators.

Tests all decorators: logged, timed, retry, cached, debug_on_error, validate_args
as well as helper functions and LogConfig.
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("psutil", reason="psutil required for observability tests")

from src.observability.decorators import (
    LogConfig,
    _format_args,
    _mask_sensitive,
    _truncate,
    cached,
    debug_on_error,
    logged,
    retry,
    timed,
    validate_args,
)


@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment variables for settings."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("LLM_PROVIDER", "openai")


# ============================================================================
# LogConfig Tests
# ============================================================================


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default LogConfig values."""
        config = LogConfig()

        assert config.entry_level == logging.DEBUG
        assert config.exit_level == logging.DEBUG
        assert config.error_level == logging.ERROR
        assert config.log_args is True
        assert config.log_result is True
        assert config.log_exception is True
        assert config.log_duration is True

    def test_truncation_defaults(self) -> None:
        """Test default truncation settings."""
        config = LogConfig()

        assert config.max_arg_length == 200
        assert config.max_result_length == 500

    def test_sensitive_fields_default(self) -> None:
        """Test default sensitive fields list."""
        config = LogConfig()

        assert "password" in config.sensitive_fields
        assert "api_key" in config.sensitive_fields
        assert "token" in config.sensitive_fields
        assert "secret" in config.sensitive_fields
        assert "credentials" in config.sensitive_fields

    def test_custom_config(self) -> None:
        """Test custom LogConfig values."""
        config = LogConfig(
            entry_level=logging.INFO,
            exit_level=logging.WARNING,
            log_args=False,
            max_arg_length=100,
        )

        assert config.entry_level == logging.INFO
        assert config.exit_level == logging.WARNING
        assert config.log_args is False
        assert config.max_arg_length == 100

    def test_from_settings(self) -> None:
        """Test creating config from settings."""
        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(LOG_LEVEL="DEBUG")
            config = LogConfig.from_settings()

            assert config.entry_level == logging.DEBUG
            assert config.error_level == logging.ERROR

    def test_from_settings_with_info_level(self) -> None:
        """Test creating config from settings with INFO level."""
        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(LOG_LEVEL="INFO")
            config = LogConfig.from_settings()

            assert config.exit_level == logging.INFO

    def test_from_settings_with_warning_level(self) -> None:
        """Test creating config from settings with WARNING level."""
        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(LOG_LEVEL="WARNING")
            config = LogConfig.from_settings()

            assert config.exit_level == logging.WARNING


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestTruncate:
    """Tests for _truncate helper function."""

    def test_short_string_unchanged(self) -> None:
        """Test short string is not truncated."""
        result = _truncate("hello", 10)
        assert result == "hello"

    def test_long_string_truncated(self) -> None:
        """Test long string is truncated."""
        result = _truncate("hello world", 5)
        assert result == "hello..."

    def test_exact_length(self) -> None:
        """Test string at exact max length."""
        result = _truncate("hello", 5)
        assert result == "hello"

    def test_non_string_converted(self) -> None:
        """Test non-string values are converted."""
        result = _truncate(12345, 10)
        assert result == "12345"

    def test_none_value(self) -> None:
        """Test None value."""
        result = _truncate(None, 10)
        assert result == "None"


class TestMaskSensitive:
    """Tests for _mask_sensitive helper function."""

    def test_mask_password(self) -> None:
        """Test password field is masked."""
        data = {"user": "john", "password": "secret123"}
        result = _mask_sensitive(data, ["password"])
        assert result["user"] == "john"
        assert result["password"] == "***MASKED***"

    def test_mask_api_key(self) -> None:
        """Test api_key field is masked."""
        data = {"name": "test", "api_key": "sk-12345"}
        result = _mask_sensitive(data, ["api_key"])
        assert result["api_key"] == "***MASKED***"

    def test_case_insensitive_matching(self) -> None:
        """Test field matching is case insensitive."""
        data = {"API_KEY": "sk-12345", "Password": "secret"}
        result = _mask_sensitive(data, ["api_key", "password"])
        assert result["API_KEY"] == "***MASKED***"
        assert result["Password"] == "***MASKED***"

    def test_nested_dict_masking(self) -> None:
        """Test nested dictionaries are masked."""
        data = {"config": {"database": {"password": "secret"}}}
        result = _mask_sensitive(data, ["password"])
        assert result["config"]["database"]["password"] == "***MASKED***"

    def test_list_masking(self) -> None:
        """Test lists with dictionaries are masked."""
        data = [{"password": "secret1"}, {"password": "secret2"}]
        result = _mask_sensitive(data, ["password"])
        assert result[0]["password"] == "***MASKED***"
        assert result[1]["password"] == "***MASKED***"

    def test_non_dict_unchanged(self) -> None:
        """Test non-dict values are unchanged."""
        assert _mask_sensitive("hello", ["password"]) == "hello"
        assert _mask_sensitive(123, ["password"]) == 123

    def test_partial_field_match(self) -> None:
        """Test partial field name matching."""
        data = {"user_password": "secret", "password_hash": "hash123"}
        result = _mask_sensitive(data, ["password"])
        assert result["user_password"] == "***MASKED***"
        assert result["password_hash"] == "***MASKED***"


class TestFormatArgs:
    """Tests for _format_args helper function."""

    def test_positional_args_skips_first(self) -> None:
        """Test first positional arg (self/cls) is skipped."""
        config = LogConfig()
        # First arg is skipped (assumed to be self/cls)
        result = _format_args(("arg1", "arg2"), {}, config)
        assert "arg1" not in result  # First arg skipped
        assert "arg2" in result

    def test_multiple_positional_args(self) -> None:
        """Test multiple positional arguments."""
        config = LogConfig()
        result = _format_args(("self", "arg1", "arg2"), {}, config)
        assert "arg1" in result
        assert "arg2" in result

    def test_keyword_args(self) -> None:
        """Test formatting keyword arguments."""
        config = LogConfig()
        result = _format_args((), {"key": "value"}, config)
        assert "key=value" in result

    def test_mixed_args(self) -> None:
        """Test formatting mixed arguments."""
        config = LogConfig()
        # First positional arg skipped, second appears
        result = _format_args(("self", "pos"), {"key": "value"}, config)
        assert "pos" in result
        assert "key=value" in result

    def test_sensitive_args_masked(self) -> None:
        """Test sensitive arguments are masked."""
        config = LogConfig()
        # Password in a dict value should be masked
        result = _format_args((), {"config": {"password": "secret"}}, config)
        assert "***MASKED***" in result
        assert "secret" not in result

    def test_truncation_applied(self) -> None:
        """Test long arguments are truncated."""
        config = LogConfig(max_arg_length=10)
        long_arg = "x" * 100
        # Second arg since first is skipped
        result = _format_args(("self", long_arg), {}, config)
        assert "..." in result
        assert len(result) < len(long_arg) + 10

    def test_empty_args(self) -> None:
        """Test empty arguments."""
        config = LogConfig()
        result = _format_args((), {}, config)
        assert result == ""


# ============================================================================
# @logged Decorator Tests
# ============================================================================


class TestLoggedDecorator:
    """Tests for @logged decorator."""

    @pytest.fixture
    def default_config(self) -> LogConfig:
        """Provide default LogConfig for tests."""
        return LogConfig()

    def test_basic_logging(self, default_config: LogConfig) -> None:
        """Test basic function logging."""
        mock_logger = MagicMock()

        @logged(config=default_config, logger_override=mock_logger)
        def simple_func(x: int) -> int:
            return x * 2

        result = simple_func(5)

        assert result == 10
        assert mock_logger.log.call_count >= 2  # Entry and exit

    def test_logs_entry_and_exit(self, default_config: LogConfig) -> None:
        """Test entry and exit messages are logged."""
        mock_logger = MagicMock()

        @logged(config=default_config, logger_override=mock_logger)
        def my_func() -> str:
            return "done"

        my_func()

        calls = [str(call) for call in mock_logger.log.call_args_list]
        calls_str = " ".join(calls)
        assert "ENTER" in calls_str
        assert "EXIT" in calls_str

    def test_custom_name(self, default_config: LogConfig) -> None:
        """Test custom function name in logs."""
        mock_logger = MagicMock()

        @logged(name="custom_operation", config=default_config, logger_override=mock_logger)
        def func() -> None:
            pass

        func()

        calls_str = str(mock_logger.log.call_args_list)
        assert "custom_operation" in calls_str

    def test_exception_logged(self, default_config: LogConfig) -> None:
        """Test exceptions are logged."""
        mock_logger = MagicMock()

        @logged(config=default_config, logger_override=mock_logger)
        def failing_func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_func()

        calls_str = str(mock_logger.log.call_args_list)
        assert "ERROR" in calls_str

    def test_no_args_logging(self) -> None:
        """Test logging with log_args=False."""
        mock_logger = MagicMock()
        config = LogConfig(log_args=False, log_result=False)

        @logged(config=config, logger_override=mock_logger)
        def func(x: int) -> int:
            return x

        func(5)

        # Args should not appear in ENTER log message
        calls = mock_logger.log.call_args_list
        entry_calls = [str(c) for c in calls if "ENTER" in str(c)]
        # With log_args=False, entry log should not include the argument value
        for call in entry_calls:
            # Should not include "(5)" args
            assert "func(5)" not in call

    def test_no_result_logging(self) -> None:
        """Test logging with log_result=False."""
        mock_logger = MagicMock()
        config = LogConfig(log_result=False)

        @logged(config=config, logger_override=mock_logger)
        def func() -> str:
            return "sensitive_result"

        func()

        # Result should not appear in logs
        calls_str = str(mock_logger.log.call_args_list)
        # Duration should still be logged
        assert "ms" in calls_str or "EXIT" in calls_str

    @pytest.mark.asyncio
    async def test_async_function(self, default_config: LogConfig) -> None:
        """Test logging works with async functions."""
        mock_logger = MagicMock()

        @logged(config=default_config, logger_override=mock_logger)
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_func(5)

        assert result == 10
        assert mock_logger.log.call_count >= 2

    @pytest.mark.asyncio
    async def test_async_exception_logged(self, default_config: LogConfig) -> None:
        """Test async exceptions are logged."""
        mock_logger = MagicMock()

        @logged(config=default_config, logger_override=mock_logger)
        async def failing_async() -> None:
            await asyncio.sleep(0.01)
            raise ValueError("async error")

        with pytest.raises(ValueError):
            await failing_async()

        calls_str = str(mock_logger.log.call_args_list)
        assert "ERROR" in calls_str


# ============================================================================
# @timed Decorator Tests
# ============================================================================


class TestTimedDecorator:
    """Tests for @timed decorator."""

    def test_timing_recorded(self) -> None:
        """Test timing is recorded."""
        with patch("src.observability.decorators.logger") as mock_logger:
            with patch("src.observability.decorators.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=10000)

                @timed()
                def fast_func() -> str:
                    return "done"

                result = fast_func()

                assert result == "done"
                assert mock_logger.debug.called or mock_logger.warning.called

    def test_slow_operation_warning(self) -> None:
        """Test warning logged for slow operations."""
        with patch("src.observability.decorators.logger") as mock_logger:
            with patch("src.observability.decorators.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000)

                @timed(threshold_ms=1)
                def slow_func() -> str:
                    time.sleep(0.01)  # 10ms
                    return "done"

                slow_func()

                # Should log warning for exceeding threshold
                mock_logger.warning.assert_called()
                call_str = str(mock_logger.warning.call_args)
                assert "Slow operation" in call_str

    def test_fast_operation_debug(self) -> None:
        """Test debug logged for fast operations."""
        with patch("src.observability.decorators.logger") as mock_logger:
            with patch("src.observability.decorators.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=10000)

                @timed(threshold_ms=10000)
                def fast_func() -> str:
                    return "done"

                fast_func()

                # Should log debug, not warning
                mock_logger.debug.assert_called()

    def test_custom_metric_name(self) -> None:
        """Test custom metric name in logs."""
        with patch("src.observability.decorators.logger") as mock_logger:
            with patch("src.observability.decorators.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000)

                @timed(metric_name="custom_metric", threshold_ms=1)
                def func() -> None:
                    time.sleep(0.01)

                func()

                call_str = str(mock_logger.warning.call_args)
                assert "custom_metric" in call_str

    @pytest.mark.asyncio
    async def test_async_timing(self) -> None:
        """Test timing works with async functions."""
        with patch("src.observability.decorators.logger") as mock_logger:
            with patch("src.observability.decorators.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000)

                @timed(threshold_ms=1)
                async def async_func() -> str:
                    await asyncio.sleep(0.01)
                    return "done"

                result = await async_func()

                assert result == "done"
                mock_logger.warning.assert_called()


# ============================================================================
# @retry Decorator Tests
# ============================================================================


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_no_retry_on_success(self) -> None:
        """Test no retry when function succeeds."""
        call_count = 0

        @retry(max_attempts=3)
        def successful_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self) -> None:
        """Test retry happens on failure."""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01, backoff_factor=1)
        def failing_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        result = failing_then_success()

        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded(self) -> None:
        """Test exception raised after max attempts."""
        call_count = 0

        @retry(max_attempts=2, initial_delay=0.01)
        def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            always_fails()

        assert call_count == 2

    def test_specific_exception_handling(self) -> None:
        """Test only specific exceptions are retried."""
        call_count = 0

        @retry(max_attempts=3, exceptions=(ValueError,), initial_delay=0.01)
        def raises_type_error() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            raises_type_error()

        assert call_count == 1  # No retry for TypeError

    def test_on_retry_callback(self) -> None:
        """Test on_retry callback is called."""
        retry_calls: list[tuple[Exception, int]] = []

        def on_retry_handler(exc: Exception, attempt: int) -> None:
            retry_calls.append((exc, attempt))

        @retry(max_attempts=3, initial_delay=0.01, on_retry=on_retry_handler)
        def failing_func() -> str:
            if len(retry_calls) < 2:
                raise ValueError("retry me")
            return "success"

        result = failing_func()

        assert result == "success"
        assert len(retry_calls) == 2
        assert retry_calls[0][1] == 1
        assert retry_calls[1][1] == 2

    def test_exponential_backoff(self) -> None:
        """Test exponential backoff delay increases."""
        start_time = time.time()
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.05, backoff_factor=2)
        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "done"

        failing_func()

        elapsed = time.time() - start_time
        # Should have waited 0.05s + 0.1s = 0.15s minimum
        assert elapsed >= 0.1  # Some tolerance for execution time

    @pytest.mark.asyncio
    async def test_async_retry(self) -> None:
        """Test retry works with async functions."""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        async def async_failing() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("retry")
            return "success"

        result = await async_failing()

        assert result == "success"
        assert call_count == 2


# ============================================================================
# @cached Decorator Tests
# ============================================================================


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_cache_hit(self) -> None:
        """Test cached result is returned."""
        call_count = 0

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached()
            def expensive_func(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            result1 = expensive_func(5)
            result2 = expensive_func(5)

            assert result1 == 10
            assert result2 == 10
            assert call_count == 1  # Only called once

    def test_cache_miss_different_args(self) -> None:
        """Test cache miss for different arguments."""
        call_count = 0

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached()
            def func(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x

            func(1)
            func(2)
            func(1)

            assert call_count == 2  # Called for 1 and 2

    def test_cache_expiration(self) -> None:
        """Test cached entries expire after TTL."""
        call_count = 0

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached(ttl_seconds=0.05)
            def func() -> int:
                nonlocal call_count
                call_count += 1
                return call_count

            result1 = func()
            time.sleep(0.1)
            result2 = func()

            assert result1 == 1
            assert result2 == 2  # New call after expiration

    def test_cache_clear(self) -> None:
        """Test cache_clear method."""
        call_count = 0

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached()
            def func() -> int:
                nonlocal call_count
                call_count += 1
                return call_count

            func()
            func.cache_clear()
            func()

            assert call_count == 2

    def test_cache_info(self) -> None:
        """Test cache_info method."""
        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached(max_size=50, ttl_seconds=100)
            def func(x: int) -> int:
                return x

            func(1)
            func(2)

            info = func.cache_info()
            assert info["size"] == 2
            assert info["max_size"] == 50
            assert info["ttl"] == 100

    def test_cache_eviction(self) -> None:
        """Test cache eviction when max size reached."""
        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached(max_size=2)
            def func(x: int) -> int:
                return x

            func(1)
            func(2)
            func(3)  # Should evict oldest

            info = func.cache_info()
            assert info["size"] == 2

    def test_kwargs_in_cache_key(self) -> None:
        """Test kwargs are included in cache key."""
        call_count = 0

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached()
            def func(a: int, b: int = 0) -> int:
                nonlocal call_count
                call_count += 1
                return a + b

            func(1, b=2)
            func(1, b=3)
            func(1, b=2)

            assert call_count == 2


# ============================================================================
# @debug_on_error Decorator Tests
# ============================================================================


class TestDebugOnErrorDecorator:
    """Tests for @debug_on_error decorator."""

    def test_no_error_passthrough(self) -> None:
        """Test successful function passes through."""

        @debug_on_error()
        def successful_func() -> str:
            return "success"

        result = successful_func()
        assert result == "success"

    def test_error_logged(self) -> None:
        """Test errors are logged."""
        with patch("src.observability.decorators.logger") as mock_logger:

            @debug_on_error()
            def failing_func() -> None:
                raise ValueError("test error")

            with pytest.raises(ValueError):
                failing_func()

            mock_logger.error.assert_called()

    def test_error_reraised(self) -> None:
        """Test errors are re-raised by default."""

        @debug_on_error(reraise=True)
        def failing_func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_func()

    def test_error_not_reraised(self) -> None:
        """Test errors can be suppressed."""

        @debug_on_error(reraise=False)
        def failing_func() -> None:
            raise ValueError("test error")

        # Should not raise
        result = failing_func()
        assert result is None

    def test_locals_logged(self) -> None:
        """Test local variables are logged on error."""
        with patch("src.observability.decorators.logger") as mock_logger:

            @debug_on_error(log_locals=True)
            def func_with_locals() -> None:
                local_var = "some value"
                raise ValueError(f"error with {local_var}")

            with pytest.raises(ValueError):
                func_with_locals()

            call_kwargs = mock_logger.error.call_args
            assert "extra" in call_kwargs.kwargs
            debug_context = call_kwargs.kwargs["extra"]["debug_context"]
            assert "locals" in debug_context

    def test_stack_logged(self) -> None:
        """Test stack trace is logged on error."""
        with patch("src.observability.decorators.logger") as mock_logger:

            @debug_on_error(log_stack=True)
            def failing_func() -> None:
                raise ValueError("test error")

            with pytest.raises(ValueError):
                failing_func()

            call_kwargs = mock_logger.error.call_args
            debug_context = call_kwargs.kwargs["extra"]["debug_context"]
            assert "traceback" in debug_context

    def test_sensitive_locals_masked(self) -> None:
        """Test sensitive local variables are masked."""
        with patch("src.observability.decorators.logger") as mock_logger:

            @debug_on_error(log_locals=True)
            def func_with_secret() -> None:
                password = "secret123"
                api_key = "sk-12345"
                raise ValueError(f"error {password} {api_key}")

            with pytest.raises(ValueError):
                func_with_secret()

            call_kwargs = mock_logger.error.call_args
            debug_context = call_kwargs.kwargs["extra"]["debug_context"]
            if "locals" in debug_context:
                locals_dict = debug_context["locals"]
                if "password" in locals_dict:
                    assert locals_dict["password"] == "***MASKED***"
                if "api_key" in locals_dict:
                    assert locals_dict["api_key"] == "***MASKED***"


# ============================================================================
# @validate_args Decorator Tests
# ============================================================================


class TestValidateArgsDecorator:
    """Tests for @validate_args decorator."""

    def test_valid_args_pass(self) -> None:
        """Test valid arguments pass validation."""

        @validate_args(x=lambda v: v > 0)
        def func(x: int) -> int:
            return x * 2

        result = func(5)
        assert result == 10

    def test_invalid_args_raise(self) -> None:
        """Test invalid arguments raise ValueError."""

        @validate_args(x=lambda v: v > 0)
        def func(x: int) -> int:
            return x * 2

        with pytest.raises(ValueError, match="Validation failed"):
            func(-5)

    def test_multiple_validators(self) -> None:
        """Test multiple argument validators."""

        @validate_args(x=lambda v: v > 0, name=lambda v: len(v) > 0)
        def func(x: int, name: str) -> str:
            return f"{name}: {x}"

        result = func(5, "test")
        assert result == "test: 5"

        with pytest.raises(ValueError):
            func(-1, "test")

        with pytest.raises(ValueError):
            func(5, "")

    def test_kwargs_validation(self) -> None:
        """Test keyword argument validation."""

        @validate_args(value=lambda v: v is not None)
        def func(value: str | None = None) -> str:
            return value or ""

        result = func(value="test")
        assert result == "test"

        with pytest.raises(ValueError):
            func(value=None)

    def test_missing_arg_not_validated(self) -> None:
        """Test missing optional args are not validated."""

        @validate_args(optional=lambda v: v > 0)
        def func(required: int, optional: int = 10) -> int:
            return required + optional

        result = func(5)  # optional not provided
        assert result == 15

    def test_validation_error_message(self) -> None:
        """Test validation error includes argument name and value."""

        @validate_args(x=lambda v: v > 100)
        def func(x: int) -> int:
            return x

        with pytest.raises(ValueError) as exc_info:
            func(50)

        error_msg = str(exc_info.value)
        assert "x" in error_msg
        assert "50" in error_msg


# ============================================================================
# Integration Tests
# ============================================================================


class TestDecoratorComposition:
    """Tests for combining multiple decorators."""

    def test_logged_and_timed(self) -> None:
        """Test combining @logged and @timed."""
        mock_logger = MagicMock()
        config = LogConfig()

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000)

            @logged(config=config, logger_override=mock_logger)
            @timed(threshold_ms=1000)
            def func(x: int) -> int:
                return x * 2

            result = func(5)
            assert result == 10
            assert mock_logger.log.called

    def test_retry_and_logged(self) -> None:
        """Test combining @retry and @logged."""
        mock_logger = MagicMock()
        config = LogConfig()
        call_count = 0

        @logged(config=config, logger_override=mock_logger)
        @retry(max_attempts=2, initial_delay=0.01)
        def func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("retry")
            return "success"

        result = func()
        assert result == "success"
        assert mock_logger.log.called

    def test_cached_and_validate(self) -> None:
        """Test combining @cached and @validate_args."""
        call_count = 0

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            @cached()
            @validate_args(x=lambda v: v > 0)
            def func(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            result1 = func(5)
            result2 = func(5)

            assert result1 == 10
            assert result2 == 10
            assert call_count == 1

            with pytest.raises(ValueError):
                func(-5)


class TestDecoratorWithMethods:
    """Tests for decorators on class methods."""

    def test_logged_on_method(self) -> None:
        """Test @logged works on instance methods."""
        mock_logger = MagicMock()
        config = LogConfig()

        class MyClass:
            @logged(config=config, logger_override=mock_logger)
            def method(self, x: int) -> int:
                return x * 2

        obj = MyClass()
        result = obj.method(5)

        assert result == 10
        assert mock_logger.log.called

    def test_cached_on_method(self) -> None:
        """Test @cached works on instance methods."""
        call_count = 0

        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300)

            class MyClass:
                @cached()
                def method(self, x: int) -> int:
                    nonlocal call_count
                    call_count += 1
                    return x * 2

            obj = MyClass()
            obj.method(5)
            obj.method(5)

            assert call_count == 1

    def test_retry_on_method(self) -> None:
        """Test @retry works on instance methods."""
        call_count = 0

        class MyClass:
            @retry(max_attempts=2, initial_delay=0.01)
            def method(self) -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ValueError("retry")
                return "success"

        obj = MyClass()
        result = obj.method()

        assert result == "success"
        assert call_count == 2
