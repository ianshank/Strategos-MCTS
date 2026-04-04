"""Unit tests for src/observability/logging.py."""

import asyncio
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.observability.logging import (
    CorrelationIdFilter,
    JSONFormatter,
    LogContext,
    PerformanceMetricsFilter,
    StructuredLogger,
    _correlation_id,
    _request_metadata,
    get_correlation_id,
    get_logger,
    get_request_metadata,
    get_structured_logger,
    log_execution_time,
    sanitize_dict,
    sanitize_message,
    set_correlation_id,
    set_request_metadata,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_context_vars():
    """Reset context variables between tests."""
    token_cid = _correlation_id.set(None)
    token_meta = _request_metadata.set(None)
    yield
    _correlation_id.reset(token_cid)
    _request_metadata.reset(token_meta)


# --- Correlation ID Tests ---


@pytest.mark.unit
class TestCorrelationId:
    def test_get_correlation_id_generates_new(self):
        cid = get_correlation_id()
        assert cid is not None
        assert len(cid) == 36  # UUID format

    def test_get_correlation_id_returns_same(self):
        cid1 = get_correlation_id()
        cid2 = get_correlation_id()
        assert cid1 == cid2

    def test_set_correlation_id(self):
        set_correlation_id("test-cid-123")
        assert get_correlation_id() == "test-cid-123"


# --- Request Metadata Tests ---


@pytest.mark.unit
class TestRequestMetadata:
    def test_get_request_metadata_default(self):
        assert get_request_metadata() == {}

    def test_set_and_get_request_metadata(self):
        metadata = {"user_id": "123", "action": "query"}
        set_request_metadata(metadata)
        assert get_request_metadata() == metadata


# --- Sanitize Message Tests ---


@pytest.mark.unit
class TestSanitizeMessage:
    def test_sanitize_api_key(self):
        msg = 'api_key: "sk-secret123"'
        result = sanitize_message(msg)
        assert "sk-secret123" not in result
        assert "***REDACTED***" in result

    def test_sanitize_password(self):
        msg = 'password = "my_password"'
        result = sanitize_message(msg)
        assert "my_password" not in result
        assert "***REDACTED***" in result

    def test_sanitize_token(self):
        msg = 'token: "abc-token-value"'
        result = sanitize_message(msg)
        assert "abc-token-value" not in result

    def test_sanitize_bearer(self):
        msg = "Authorization: Bearer eyJhbGciOiJIUz"
        result = sanitize_message(msg)
        assert "eyJhbGciOiJIUz" not in result
        assert "Bearer ***REDACTED***" in result

    def test_sanitize_basic_auth(self):
        msg = "Authorization: Basic dXNlcjpwYXNz"
        result = sanitize_message(msg)
        assert "dXNlcjpwYXNz" not in result
        assert "Basic ***REDACTED***" in result

    def test_sanitize_secret(self):
        msg = 'secret: "my-secret-val"'
        result = sanitize_message(msg)
        assert "my-secret-val" not in result

    def test_no_sanitize_safe_message(self):
        msg = "Processing request for user 42"
        assert sanitize_message(msg) == msg

    def test_sanitize_authorization_header(self):
        msg = 'authorization = "some-auth-value"'
        result = sanitize_message(msg)
        assert "some-auth-value" not in result

    def test_sanitize_aws_secret(self):
        msg = 'aws_secret_access_key: "AKIA123SECRET"'
        result = sanitize_message(msg)
        assert "AKIA123SECRET" not in result


# --- Sanitize Dict Tests ---


@pytest.mark.unit
class TestSanitizeDict:
    def test_sanitize_sensitive_keys(self):
        data = {"api_key": "sk-123", "name": "test"}
        result = sanitize_dict(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["name"] == "test"

    def test_sanitize_nested_dict(self):
        data = {"config": {"password": "secret123", "host": "localhost"}}
        result = sanitize_dict(data)
        assert result["config"]["password"] == "***REDACTED***"
        assert result["config"]["host"] == "localhost"

    def test_sanitize_list_of_dicts(self):
        data = {"items": [{"token": "abc"}, {"name": "ok"}]}
        result = sanitize_dict(data)
        assert result["items"][0]["token"] == "***REDACTED***"
        assert result["items"][1]["name"] == "ok"

    def test_sanitize_string_values(self):
        data = {"message": 'api_key: "sk-xyz"'}
        result = sanitize_dict(data)
        assert "sk-xyz" not in result["message"]

    def test_sanitize_non_string_non_dict_values(self):
        data = {"count": 42, "flag": True}
        result = sanitize_dict(data)
        assert result["count"] == 42
        assert result["flag"] is True

    def test_sanitize_credentials_key(self):
        data = {"credentials": "super-secret"}
        result = sanitize_dict(data)
        assert result["credentials"] == "***REDACTED***"

    def test_sanitize_private_key(self):
        data = {"private_key": "-----BEGIN RSA PRIVATE KEY-----"}
        result = sanitize_dict(data)
        assert result["private_key"] == "***REDACTED***"

    def test_sanitize_hyphenated_key(self):
        data = {"api-key": "sk-123"}
        result = sanitize_dict(data)
        assert result["api-key"] == "***REDACTED***"

    def test_sanitize_list_with_plain_items(self):
        data = {"tags": ["foo", "bar"]}
        result = sanitize_dict(data)
        assert result["tags"] == ["foo", "bar"]


# --- CorrelationIdFilter Tests ---


@pytest.mark.unit
class TestCorrelationIdFilter:
    def test_filter_adds_correlation_id(self):
        f = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        result = f.filter(record)
        assert result is True
        assert hasattr(record, "correlation_id")
        assert record.correlation_id is not None

    def test_filter_adds_request_metadata(self):
        set_request_metadata({"user": "test"})
        f = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        f.filter(record)
        assert record.request_metadata == {"user": "test"}


# --- PerformanceMetricsFilter Tests ---


@pytest.mark.unit
class TestPerformanceMetricsFilter:
    @patch("src.observability.logging.psutil.Process")
    def test_filter_adds_metrics(self, mock_process_cls):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.cpu_percent.return_value = 25.0
        mock_process.num_threads.return_value = 4
        mock_process_cls.return_value = mock_process

        f = PerformanceMetricsFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        result = f.filter(record)
        assert result is True
        assert record.memory_mb == pytest.approx(100.0, rel=0.01)
        assert record.cpu_percent == 25.0
        assert record.thread_count == 4


# --- JSONFormatter Tests ---


@pytest.mark.unit
class TestJSONFormatter:
    def test_format_basic(self):
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Hello world",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-cid"
        record.request_metadata = {}

        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "Hello world"
        assert data["correlation_id"] == "test-cid"
        assert data["line"] == 42
        assert "hostname" not in data
        assert "process" not in data

    def test_format_with_hostname(self):
        formatter = JSONFormatter(include_hostname=True, include_process=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        record.correlation_id = None
        record.request_metadata = {}
        output = formatter.format(record)
        data = json.loads(output)
        assert "hostname" in data

    def test_format_with_process_info(self):
        formatter = JSONFormatter(include_hostname=False, include_process=True)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        record.correlation_id = None
        record.request_metadata = {}
        output = formatter.format(record)
        data = json.loads(output)
        assert "process" in data
        assert "id" in data["process"]

    def test_format_with_performance_metrics(self):
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        record.correlation_id = None
        record.request_metadata = {}
        record.memory_mb = 150.123
        record.cpu_percent = 33.5
        record.thread_count = 8
        output = formatter.format(record)
        data = json.loads(output)
        assert data["performance"]["memory_mb"] == 150.12
        assert data["performance"]["cpu_percent"] == 33.5
        assert data["performance"]["thread_count"] == 8

    def test_format_with_request_metadata(self):
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        record.correlation_id = None
        record.request_metadata = {"user_id": "123", "token": "secret"}
        output = formatter.format(record)
        data = json.loads(output)
        assert data["request"]["user_id"] == "123"
        assert data["request"]["token"] == "***REDACTED***"

    def test_format_with_exception(self):
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="error", args=(), exc_info=exc_info
        )
        record.correlation_id = None
        record.request_metadata = {}
        output = formatter.format(record)
        data = json.loads(output)
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "test error"

    def test_format_sanitizes_message(self):
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg='api_key: "sk-secret"', args=(), exc_info=None
        )
        record.correlation_id = None
        record.request_metadata = {}
        output = formatter.format(record)
        data = json.loads(output)
        assert "sk-secret" not in data["message"]

    def test_format_with_extra_fields(self):
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        record.correlation_id = None
        record.request_metadata = {}
        record.custom_field = "custom_value"
        output = formatter.format(record)
        data = json.loads(output)
        assert data["extra"]["custom_field"] == "custom_value"

    def test_format_extra_dict_sanitized(self):
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="test", args=(), exc_info=None
        )
        record.correlation_id = None
        record.request_metadata = {}
        record.config_data = {"password": "secret123"}
        output = formatter.format(record)
        data = json.loads(output)
        assert data["extra"]["config_data"]["password"] == "***REDACTED***"


# --- setup_logging Tests ---


@pytest.mark.unit
class TestSetupLogging:
    def test_setup_logging_defaults(self):
        setup_logging(log_level="WARNING", json_output=True, include_performance_metrics=False)
        logger = logging.getLogger("mcts")
        assert logger is not None

    def test_setup_logging_with_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(
            log_level="DEBUG",
            log_file=log_file,
            include_performance_metrics=False,
            json_output=True,
        )
        logger = logging.getLogger("mcts")
        assert logger is not None

    def test_setup_logging_non_json(self):
        setup_logging(log_level="INFO", json_output=False, include_performance_metrics=False)

    def test_setup_logging_with_performance_metrics(self):
        setup_logging(log_level="INFO", include_performance_metrics=True)

    @patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"})
    def test_setup_logging_from_env(self):
        setup_logging(include_performance_metrics=False)


# --- get_logger Tests ---


@pytest.mark.unit
class TestGetLogger:
    def test_get_logger_prefixes_name(self):
        logger = get_logger("framework.graph")
        assert logger.name == "mcts.framework.graph"

    def test_get_logger_already_prefixed(self):
        logger = get_logger("mcts.agents")
        assert logger.name == "mcts.agents"


# --- LogContext Tests ---


@pytest.mark.unit
class TestLogContext:
    def test_log_context_sets_metadata(self):
        with LogContext(user_id="123"):
            metadata = get_request_metadata()
            assert metadata["user_id"] == "123"

    def test_log_context_restores_metadata(self):
        set_request_metadata({"existing": "value"})
        with LogContext(user_id="123"):
            meta = get_request_metadata()
            assert meta["user_id"] == "123"
            assert meta["existing"] == "value"
        assert get_request_metadata() == {"existing": "value"}

    def test_log_context_nested(self):
        with LogContext(a="1"):
            with LogContext(b="2"):
                meta = get_request_metadata()
                assert meta["a"] == "1"
                assert meta["b"] == "2"
            meta = get_request_metadata()
            assert meta["a"] == "1"
            assert "b" not in meta


# --- log_execution_time Tests ---


@pytest.mark.unit
class TestLogExecutionTime:
    @patch("src.observability.logging.psutil.Process")
    def test_sync_function_logging(self, mock_process_cls):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_cls.return_value = mock_process

        mock_logger = MagicMock()

        @log_execution_time(logger=mock_logger)
        def my_func(x):
            return x + 1

        result = my_func(5)
        assert result == 6
        mock_logger.log.assert_called_once()

    @patch("src.observability.logging.psutil.Process")
    def test_sync_function_exception(self, mock_process_cls):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_cls.return_value = mock_process

        mock_logger = MagicMock()

        @log_execution_time(logger=mock_logger)
        def failing_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            failing_func()
        mock_logger.log.assert_called_once()
        call_kwargs = mock_logger.log.call_args
        assert call_kwargs[1]["extra"]["timing"]["success"] is False

    @patch("src.observability.logging.psutil.Process")
    def test_async_function_logging(self, mock_process_cls):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_cls.return_value = mock_process

        mock_logger = MagicMock()

        @log_execution_time(logger=mock_logger)
        async def async_func(x):
            return x * 2

        result = asyncio.run(async_func(3))
        assert result == 6
        mock_logger.log.assert_called_once()

    @patch("src.observability.logging.psutil.Process")
    def test_async_function_exception(self, mock_process_cls):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_cls.return_value = mock_process

        mock_logger = MagicMock()

        @log_execution_time(logger=mock_logger)
        async def async_fail():
            raise ValueError("async fail")

        with pytest.raises(ValueError, match="async fail"):
            asyncio.run(async_fail())
        call_kwargs = mock_logger.log.call_args
        assert call_kwargs[1]["extra"]["timing"]["success"] is False

    @patch("src.observability.logging.psutil.Process")
    def test_default_logger(self, mock_process_cls):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_cls.return_value = mock_process

        @log_execution_time()
        def auto_logger_func():
            return 42

        result = auto_logger_func()
        assert result == 42


# --- StructuredLogger Tests ---


@pytest.mark.unit
class TestStructuredLogger:
    def test_info(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.info("hello", user_id="123")
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[0] == logging.INFO
            assert args[1] == "hello"

    def test_debug(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.debug("debug msg")
            args, _ = mock_log.call_args
            assert args[0] == logging.DEBUG

    def test_warning(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.warning("warn msg")
            args, _ = mock_log.call_args
            assert args[0] == logging.WARNING

    def test_error(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.error("error msg")
            args, _ = mock_log.call_args
            assert args[0] == logging.ERROR

    def test_critical(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.critical("critical msg")
            args, _ = mock_log.call_args
            assert args[0] == logging.CRITICAL

    def test_exception(self):
        with patch.object(logging.Logger, "exception") as mock_exc:
            sl = StructuredLogger("test.module")
            try:
                raise ValueError("exc test")
            except ValueError:
                sl.exception("error occurred")
            mock_exc.assert_called_once()

    def test_log_timing(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.log_timing("db_query", 45.2)
            mock_log.assert_called_once()

    def test_log_memory(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.log_memory("cache", 128.5)
            mock_log.assert_called_once()

    def test_log_mcts_iteration(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.log_mcts_iteration(
                iteration=5,
                tree_depth=10,
                nodes_explored=50,
                best_action="e2e4",
                ucb_score=1.234,
            )
            mock_log.assert_called_once()

    def test_log_mcts_iteration_no_optional(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.log_mcts_iteration(iteration=1, tree_depth=2, nodes_explored=3)
            mock_log.assert_called_once()

    def test_log_agent_execution_success(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.log_agent_execution("HRM", 100.5, 0.95, success=True)
            args, _ = mock_log.call_args
            assert args[0] == logging.INFO

    def test_log_agent_execution_failure(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.log_agent_execution("TRM", 50.0, 0.3, success=False)
            args, _ = mock_log.call_args
            assert args[0] == logging.WARNING

    def test_sanitizes_sensitive_extra(self):
        with patch.object(logging.Logger, "log") as mock_log:
            sl = StructuredLogger("test.module")
            sl.info("msg", api_key="secret-key-123")
            _, kwargs = mock_log.call_args
            assert kwargs["extra"]["api_key"] == "***REDACTED***"


@pytest.mark.unit
class TestGetStructuredLogger:
    def test_returns_structured_logger(self):
        sl = get_structured_logger("my.module")
        assert isinstance(sl, StructuredLogger)
