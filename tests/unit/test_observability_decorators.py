"""
Unit tests for src/observability/decorators.py

Tests decorator behavior: logging, timing, retry, caching,
debug_on_error, and validate_args -- covering success paths,
error paths, async functions, and edge cases.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest

# Set environment variables before importing modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

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

pytestmark = pytest.mark.unit


# ────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────


class TestTruncate:
    """Tests for _truncate helper."""

    def test_short_string_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_long_string_truncated(self):
        result = _truncate("hello world", 5)
        assert result == "hello..."

    def test_non_string_converted(self):
        assert _truncate(42, 10) == "42"


class TestMaskSensitive:
    """Tests for _mask_sensitive helper."""

    def test_masks_password_field(self):
        data = {"username": "alice", "password": "secret123"}
        result = _mask_sensitive(data, ["password", "api_key", "token", "secret"])
        assert result["username"] == "alice"
        assert result["password"] == "***MASKED***"

    def test_masks_nested_dict(self):
        data = {"outer": {"api_key": "sk-123", "name": "test"}}
        result = _mask_sensitive(data, ["api_key"])
        assert result["outer"]["api_key"] == "***MASKED***"
        assert result["outer"]["name"] == "test"

    def test_masks_list_of_dicts(self):
        data = [{"token": "abc"}, {"name": "test"}]
        result = _mask_sensitive(data, ["token"])
        assert result[0]["token"] == "***MASKED***"
        assert result[1]["name"] == "test"

    def test_non_dict_non_list_passthrough(self):
        assert _mask_sensitive(42, ["password"]) == 42
        assert _mask_sensitive("hello", ["password"]) == "hello"

    def test_case_insensitive_matching(self):
        data = {"API_KEY": "sk-123"}
        result = _mask_sensitive(data, ["api_key"])
        assert result["API_KEY"] == "***MASKED***"

    def test_partial_field_match(self):
        data = {"my_secret_value": "hidden"}
        result = _mask_sensitive(data, ["secret"])
        assert result["my_secret_value"] == "***MASKED***"


class TestFormatArgs:
    """Tests for _format_args helper."""

    def test_positional_args(self):
        config = LogConfig(max_arg_length=200)
        # First arg with __class__ is treated as self and skipped,
        # so we use a plain int as first arg (int has __class__ too,
        # so it gets skipped). Pass three args to verify the rest appear.
        result = _format_args((42, "hello", 99), {}, config)
        assert "hello" in result
        assert "99" in result

    def test_keyword_args(self):
        config = LogConfig(max_arg_length=200)
        result = _format_args((), {"name": "alice"}, config)
        assert "name=alice" in result

    def test_skips_self_like_arg(self):
        """First arg with __class__ (self) is skipped."""
        obj = object()
        config = LogConfig(max_arg_length=200)
        result = _format_args((obj, "data"), {}, config)
        assert "data" in result
        # The object repr should not appear
        assert "object" not in result.split(",")[0] if "," in result else True

    def test_masks_sensitive_kwargs(self):
        """Sensitive fields are masked when values are dicts."""
        config = LogConfig(sensitive_fields=["password"])
        result = _format_args((), {"data": {"password": "secret", "user": "alice"}}, config)
        assert "***MASKED***" in result
        assert "secret" not in result


# ────────────────────────────────────────────────────────────────────
# LogConfig
# ────────────────────────────────────────────────────────────────────


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_values(self):
        cfg = LogConfig()
        assert cfg.entry_level == logging.DEBUG
        assert cfg.log_args is True
        assert cfg.log_result is True
        assert cfg.log_exception is True
        assert cfg.log_duration is True
        assert "password" in cfg.sensitive_fields

    def test_from_settings(self):
        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(LOG_LEVEL="WARNING")
            cfg = LogConfig.from_settings()
            assert cfg.entry_level == logging.DEBUG
            assert cfg.exit_level == logging.WARNING
            assert cfg.error_level == logging.ERROR

    def test_from_settings_unknown_level(self):
        with patch("src.observability.decorators.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(LOG_LEVEL="CUSTOM")
            cfg = LogConfig.from_settings()
            # Falls back to INFO
            assert cfg.exit_level == logging.INFO


# ────────────────────────────────────────────────────────────────────
# @logged decorator
# ────────────────────────────────────────────────────────────────────


class TestLoggedDecorator:
    """Tests for @logged decorator."""

    def _make_logger(self):
        mock_log = MagicMock(spec=logging.Logger)
        return mock_log

    def test_sync_success_logs_entry_and_exit(self):
        mock_log = self._make_logger()
        config = LogConfig(log_args=True, log_result=True)

        @logged(config=config, logger_override=mock_log)
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3
        # At least two log calls: entry and exit
        assert mock_log.log.call_count >= 2
        entry_call = mock_log.log.call_args_list[0]
        assert "[ENTER]" in entry_call[0][1]
        exit_call = mock_log.log.call_args_list[1]
        assert "[EXIT]" in exit_call[0][1]

    def test_sync_error_logs_error(self):
        mock_log = self._make_logger()
        config = LogConfig(log_exception=True)

        @logged(config=config, logger_override=mock_log)
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

        # Should have entry + error
        calls = [c[0][1] for c in mock_log.log.call_args_list]
        assert any("[ERROR]" in c for c in calls)

    def test_sync_no_args_logging(self):
        mock_log = self._make_logger()
        config = LogConfig(log_args=False)

        @logged(config=config, logger_override=mock_log)
        def noop():
            return 1

        noop()
        entry_msg = mock_log.log.call_args_list[0][0][1]
        assert "(" not in entry_msg  # no args logged

    def test_sync_no_result_but_duration(self):
        mock_log = self._make_logger()
        config = LogConfig(log_result=False, log_duration=True)

        @logged(config=config, logger_override=mock_log)
        def noop():
            return 1

        noop()
        exit_msg = mock_log.log.call_args_list[1][0][1]
        assert "[EXIT]" in exit_msg
        assert "ms" in exit_msg

    def test_sync_no_result_no_duration(self):
        mock_log = self._make_logger()
        config = LogConfig(log_result=False, log_duration=False)

        @logged(config=config, logger_override=mock_log)
        def noop():
            return 1

        noop()
        # Only entry log, no exit log
        assert mock_log.log.call_count == 1

    def test_sync_exception_not_logged_when_disabled(self):
        mock_log = self._make_logger()
        config = LogConfig(log_exception=False)

        @logged(config=config, logger_override=mock_log)
        def fail():
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            fail()

        calls = [c[0][1] for c in mock_log.log.call_args_list]
        assert not any("[ERROR]" in c for c in calls)

    def test_custom_name(self):
        mock_log = self._make_logger()

        @logged(name="my_custom_op", config=LogConfig(), logger_override=mock_log)
        def func():
            return 1

        func()
        entry_msg = mock_log.log.call_args_list[0][0][1]
        assert "my_custom_op" in entry_msg

    @pytest.mark.asyncio
    async def test_async_success(self):
        mock_log = self._make_logger()
        config = LogConfig(log_args=True, log_result=True)

        @logged(config=config, logger_override=mock_log)
        async def async_add(a, b):
            return a + b

        result = await async_add(3, 4)
        assert result == 7
        calls = [c[0][1] for c in mock_log.log.call_args_list]
        assert any("[ENTER]" in c for c in calls)
        assert any("[EXIT]" in c for c in calls)

    @pytest.mark.asyncio
    async def test_async_error(self):
        mock_log = self._make_logger()
        config = LogConfig(log_exception=True)

        @logged(config=config, logger_override=mock_log)
        async def async_fail():
            raise ValueError("async boom")

        with pytest.raises(ValueError, match="async boom"):
            await async_fail()

        calls = [c[0][1] for c in mock_log.log.call_args_list]
        assert any("[ERROR]" in c for c in calls)

    @pytest.mark.asyncio
    async def test_async_no_args(self):
        mock_log = self._make_logger()
        config = LogConfig(log_args=False)

        @logged(config=config, logger_override=mock_log)
        async def noop():
            return 1

        await noop()
        entry_msg = mock_log.log.call_args_list[0][0][1]
        assert "(" not in entry_msg

    @pytest.mark.asyncio
    async def test_async_duration_only(self):
        mock_log = self._make_logger()
        config = LogConfig(log_result=False, log_duration=True)

        @logged(config=config, logger_override=mock_log)
        async def noop():
            return 1

        await noop()
        exit_msg = mock_log.log.call_args_list[1][0][1]
        assert "[EXIT]" in exit_msg
        assert "ms" in exit_msg

    def test_preserves_function_name(self):
        @logged(config=LogConfig(), logger_override=MagicMock())
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."


# ────────────────────────────────────────────────────────────────────
# @timed decorator
# ────────────────────────────────────────────────────────────────────


class TestTimedDecorator:
    """Tests for @timed decorator."""

    @patch("src.observability.decorators.get_settings")
    def test_sync_under_threshold(self, mock_settings):
        mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000.0)

        @timed(threshold_ms=5000)
        def fast():
            return 42

        with patch("src.observability.decorators.logger") as mock_log:
            result = fast()
        assert result == 42
        mock_log.debug.assert_called()

    @patch("src.observability.decorators.get_settings")
    def test_sync_over_threshold(self, mock_settings):
        mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000.0)

        @timed(threshold_ms=0.001)
        def slow():
            time.sleep(0.01)
            return 1

        with patch("src.observability.decorators.logger") as mock_log:
            result = slow()
        assert result == 1
        mock_log.warning.assert_called()
        warning_msg = mock_log.warning.call_args[0][0]
        assert "Slow operation" in warning_msg

    @patch("src.observability.decorators.get_settings")
    def test_sync_exception_still_times(self, mock_settings):
        mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000.0)

        @timed(threshold_ms=5000)
        def fail():
            raise RuntimeError("oops")

        with patch("src.observability.decorators.logger") as mock_log:
            with pytest.raises(RuntimeError):
                fail()
            # Still logs timing even on failure (via finally)
            assert mock_log.debug.called or mock_log.warning.called

    @pytest.mark.asyncio
    @patch("src.observability.decorators.get_settings")
    async def test_async_under_threshold(self, mock_settings):
        mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000.0)

        @timed(threshold_ms=5000)
        async def fast():
            return 99

        with patch("src.observability.decorators.logger") as mock_log:
            result = await fast()
        assert result == 99
        mock_log.debug.assert_called()

    @pytest.mark.asyncio
    @patch("src.observability.decorators.get_settings")
    async def test_async_over_threshold(self, mock_settings):
        mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000.0)

        @timed(threshold_ms=0.001)
        async def slow():
            await asyncio.sleep(0.01)
            return 1

        with patch("src.observability.decorators.logger") as mock_log:
            result = await slow()
        assert result == 1
        mock_log.warning.assert_called()

    @patch("src.observability.decorators.get_settings")
    def test_default_threshold_from_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=0.001)

        @timed()  # no explicit threshold
        def func():
            time.sleep(0.01)
            return 1

        with patch("src.observability.decorators.logger") as mock_log:
            func()
        mock_log.warning.assert_called()

    @patch("src.observability.decorators.get_settings")
    def test_custom_metric_name(self, mock_settings):
        mock_settings.return_value = MagicMock(SLOW_OPERATION_THRESHOLD_MS=1000.0)

        @timed(metric_name="my_metric", threshold_ms=5000)
        def func():
            return 1

        with patch("src.observability.decorators.logger") as mock_log:
            func()
        msg = mock_log.debug.call_args[0][0]
        assert "my_metric" in msg


# ────────────────────────────────────────────────────────────────────
# @retry decorator
# ────────────────────────────────────────────────────────────────────


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_sync_no_failure(self):
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.001, exceptions=(ValueError,))
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_sync_retries_then_succeeds(self):
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.001, backoff_factor=1.0, exceptions=(ValueError,))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_sync_all_attempts_fail(self):
        @retry(max_attempts=2, initial_delay=0.001, exceptions=(ValueError,))
        def always_fail():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            always_fail()

    def test_sync_unmatched_exception_not_retried(self):
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.001, exceptions=(ValueError,))
        def wrong_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong")

        with pytest.raises(TypeError):
            wrong_error()
        assert call_count == 1

    def test_sync_on_retry_callback(self):
        callback = MagicMock()

        @retry(max_attempts=3, initial_delay=0.001, backoff_factor=1.0, exceptions=(ValueError,), on_retry=callback)
        def flaky():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            flaky()

        # on_retry called for attempt 1 and 2 (not the last one)
        assert callback.call_count == 2
        # Check args: (exception, attempt_number)
        assert callback.call_args_list[0][0][1] == 1
        assert callback.call_args_list[1][0][1] == 2

    @pytest.mark.asyncio
    async def test_async_no_failure(self):
        @retry(max_attempts=3, initial_delay=0.001, exceptions=(ValueError,))
        async def succeed():
            return "ok"

        assert await succeed() == "ok"

    @pytest.mark.asyncio
    async def test_async_retries_then_succeeds(self):
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.001, backoff_factor=1.0, exceptions=(ValueError,))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("not yet")
            return "ok"

        assert await flaky() == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_all_attempts_fail(self):
        @retry(max_attempts=2, initial_delay=0.001, exceptions=(RuntimeError,))
        async def always_fail():
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError, match="async fail"):
            await always_fail()

    @pytest.mark.asyncio
    async def test_async_on_retry_callback(self):
        callback = MagicMock()

        @retry(max_attempts=2, initial_delay=0.001, exceptions=(ValueError,), on_retry=callback)
        async def flaky():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await flaky()
        assert callback.call_count == 1


# ────────────────────────────────────────────────────────────────────
# @cached decorator
# ────────────────────────────────────────────────────────────────────


class TestCachedDecorator:
    """Tests for @cached decorator."""

    @patch("src.observability.decorators.get_settings")
    def test_cache_hit(self, mock_settings):
        mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300.0)
        call_count = 0

        @cached()
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(5) == 10
        assert call_count == 1  # second call is cache hit

    @patch("src.observability.decorators.get_settings")
    def test_cache_miss_different_args(self, mock_settings):
        mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300.0)
        call_count = 0

        @cached()
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(6) == 12
        assert call_count == 2

    @patch("src.observability.decorators.get_settings")
    def test_cache_expiry(self, mock_settings):
        mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300.0)
        call_count = 0

        @cached(ttl_seconds=0.01)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        time.sleep(0.02)
        compute(5)
        assert call_count == 2  # expired, recomputed

    @patch("src.observability.decorators.get_settings")
    def test_cache_eviction(self, mock_settings):
        mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300.0)

        @cached(max_size=2)
        def compute(x):
            return x * 2

        compute(1)
        compute(2)
        compute(3)  # should evict oldest
        info = compute.cache_info()
        assert info["size"] <= 2

    @patch("src.observability.decorators.get_settings")
    def test_cache_clear(self, mock_settings):
        mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300.0)

        @cached()
        def compute(x):
            return x * 2

        compute(1)
        assert compute.cache_info()["size"] == 1
        compute.cache_clear()
        assert compute.cache_info()["size"] == 0

    @patch("src.observability.decorators.get_settings")
    def test_kwargs_in_cache_key(self, mock_settings):
        mock_settings.return_value = MagicMock(DEFAULT_CACHE_TTL_SECONDS=300.0)
        call_count = 0

        @cached()
        def compute(x, multiplier=2):
            nonlocal call_count
            call_count += 1
            return x * multiplier

        compute(5, multiplier=2)
        compute(5, multiplier=3)
        assert call_count == 2  # different kwargs = different keys


# ────────────────────────────────────────────────────────────────────
# @debug_on_error decorator
# ────────────────────────────────────────────────────────────────────


class TestDebugOnError:
    """Tests for @debug_on_error decorator."""

    def test_no_error_passes_through(self):
        @debug_on_error()
        def safe():
            return 42

        assert safe() == 42

    def test_error_reraised_by_default(self):
        @debug_on_error()
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

    def test_error_suppressed_when_reraise_false(self):
        @debug_on_error(reraise=False)
        def fail():
            raise ValueError("boom")

        result = fail()
        assert result is None  # returns None

    def test_logs_error_with_context(self):
        with patch("src.observability.decorators.logger") as mock_log:

            @debug_on_error(log_locals=True, log_stack=True)
            def fail():
                x = 42  # noqa: F841
                raise ValueError("boom")

            with pytest.raises(ValueError):
                fail()

            mock_log.error.assert_called_once()
            call_kwargs = mock_log.error.call_args
            extra = call_kwargs[1].get("extra", {}) if call_kwargs[1] else {}
            debug_ctx = extra.get("debug_context", {})
            assert debug_ctx.get("function") == "fail"
            assert debug_ctx.get("exception") == "ValueError"

    def test_log_stack_false(self):
        with patch("src.observability.decorators.logger") as mock_log:

            @debug_on_error(log_stack=False, log_locals=False)
            def fail():
                raise ValueError("boom")

            with pytest.raises(ValueError):
                fail()

            extra = mock_log.error.call_args[1].get("extra", {})
            debug_ctx = extra.get("debug_context", {})
            assert "traceback" not in debug_ctx
            assert "locals" not in debug_ctx

    def test_sensitive_locals_masked(self):
        with patch("src.observability.decorators.logger") as mock_log:

            @debug_on_error(log_locals=True)
            def fail():
                my_password = "super_secret"  # noqa: F841
                raise ValueError("boom")

            with pytest.raises(ValueError):
                fail()

            extra = mock_log.error.call_args[1].get("extra", {})
            debug_ctx = extra.get("debug_context", {})
            local_vars = debug_ctx.get("locals", {})
            if "my_password" in local_vars:
                assert local_vars["my_password"] == "***MASKED***"


# ────────────────────────────────────────────────────────────────────
# @validate_args decorator
# ────────────────────────────────────────────────────────────────────


class TestValidateArgs:
    """Tests for @validate_args decorator."""

    def test_valid_args_pass(self):
        @validate_args(x=lambda v: v > 0, name=lambda v: len(v) > 0)
        def process(x, name):
            return f"{name}: {x}"

        assert process(1, "test") == "test: 1"

    def test_invalid_positional_arg_raises(self):
        @validate_args(x=lambda v: v > 0)
        def process(x):
            return x

        with pytest.raises(ValueError, match="Validation failed for argument 'x'"):
            process(-1)

    def test_invalid_kwarg_raises(self):
        @validate_args(name=lambda v: len(v) > 0)
        def process(name=""):
            return name

        with pytest.raises(ValueError, match="Validation failed for argument 'name'"):
            process(name="")

    def test_missing_validator_arg_ignored(self):
        @validate_args(x=lambda v: v > 0)
        def process(x, y):
            return x + y

        # y has no validator, should pass
        assert process(1, -5) == -4

    def test_validator_not_in_signature_ignored(self):
        @validate_args(z=lambda v: v > 0)
        def process(x):
            return x

        # z is not a parameter, validator skipped
        assert process(5) == 5

    def test_kwargs_override_positional(self):
        @validate_args(x=lambda v: v > 0)
        def process(x):
            return x

        # Passing as kwarg
        assert process(x=5) == 5
        with pytest.raises(ValueError):
            process(x=-1)
