"""
JSON-structured logging infrastructure for multi-agent MCTS framework.

Provides:
- JSON-structured logging via logging.config.dictConfig
- Per-module loggers with proper hierarchy
- Correlation IDs for request tracking
- Log levels configurable via environment/settings
- Performance metrics in logs (timing, memory)
- Safe sanitization (no secrets in logs)
"""

import json
import logging
import logging.config
import os
import re
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any

import psutil

# Context variable for correlation ID tracking across async calls
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_request_metadata: ContextVar[dict | None] = ContextVar("request_metadata", default=None)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = _correlation_id.get()
    if cid is None:
        cid = str(uuid.uuid4())
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    _correlation_id.set(cid)


def set_request_metadata(metadata: dict) -> None:
    """Set request metadata for current context."""
    _request_metadata.set(metadata)


def get_request_metadata() -> dict:
    """Get request metadata for current context."""
    metadata = _request_metadata.get()
    return metadata if metadata is not None else {}


# Patterns for sensitive data sanitization
SENSITIVE_PATTERNS = [
    (re.compile(r'("?api[_-]?key"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?password"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?secret"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?token"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?authorization"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?aws[_-]?secret[_-]?access[_-]?key"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r"(Bearer\s+)\S+", re.IGNORECASE), r"\1***REDACTED***"),
    (re.compile(r"(Basic\s+)\S+", re.IGNORECASE), r"\1***REDACTED***"),
]


def sanitize_message(message: str) -> str:
    """Sanitize sensitive data from log messages."""
    for pattern, replacement in SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


def sanitize_dict(data: dict) -> dict:
    """Recursively sanitize sensitive data from dictionaries."""
    sensitive_keys = {
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "authorization",
        "auth",
        "credentials",
        "aws_secret_access_key",
        "private_key",
    }

    result: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")
        if key_lower in sensitive_keys:
            result[key] = "***REDACTED***"
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [sanitize_dict(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, str):
            result[key] = sanitize_message(value)
        else:
            result[key] = value
    return result


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID and request metadata to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        record.request_metadata = get_request_metadata()
        return True


class PerformanceMetricsFilter(logging.Filter):
    """Add performance metrics to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        process = psutil.Process()
        record.memory_mb = process.memory_info().rss / (1024 * 1024)
        record.cpu_percent = process.cpu_percent()
        record.thread_count = process.num_threads()
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON with comprehensive metadata."""

    def __init__(self, include_hostname: bool = True, include_process: bool = True):
        super().__init__()
        self.include_hostname = include_hostname
        self.include_process = include_process
        self.hostname: str | None = None
        if include_hostname:
            import socket

            self.hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_message(record.getMessage()),
            "correlation_id": getattr(record, "correlation_id", None),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add hostname if configured
        if self.hostname:
            log_data["hostname"] = self.hostname

        # Add process info if configured
        if self.include_process:
            log_data["process"] = {
                "id": record.process,
                "name": record.processName,
                "thread_id": record.thread,
                "thread_name": record.threadName,
            }

        # Add performance metrics if available
        if hasattr(record, "memory_mb"):
            log_data["performance"] = {
                "memory_mb": round(getattr(record, "memory_mb", 0), 2),
                "cpu_percent": round(getattr(record, "cpu_percent", 0), 2),
                "thread_count": getattr(record, "thread_count", 0),
            }

        # Add request metadata if available
        request_metadata = getattr(record, "request_metadata", {})
        if request_metadata:
            log_data["request"] = sanitize_dict(request_metadata)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add any extra fields (sanitized)
        extra_fields: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "correlation_id",
                "request_metadata",
                "memory_mb",
                "cpu_percent",
                "thread_count",
                "taskName",
            }:
                if isinstance(value, dict):
                    extra_fields[key] = sanitize_dict(value)
                elif isinstance(value, str):
                    extra_fields[key] = sanitize_message(value)
                else:
                    extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str)


def setup_logging(
    log_level: str | None = None,
    log_file: str | None = None,
    include_performance_metrics: bool = True,
    json_output: bool = True,
    include_hostname: bool = True,
    include_process: bool = True,
) -> None:
    """
    Configure JSON-structured logging for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  Defaults to LOG_LEVEL env var or INFO.
        log_file: Optional file path for log output.
        include_performance_metrics: Include memory/CPU metrics in logs.
        json_output: Use JSON formatter (default True).
        include_hostname: Include hostname in logs.
        include_process: Include process/thread info in logs.
    """
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Base configuration
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "correlation_id": {
                "()": CorrelationIdFilter,
            },
        },
        "formatters": {
            "json": {
                "()": JSONFormatter,
                "include_hostname": include_hostname,
                "include_process": include_process,
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(correlation_id)s): %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json" if json_output else "standard",
                "filters": ["correlation_id"],
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # Root logger
            "": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            # Framework loggers with hierarchy
            "mcts": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "mcts.framework": {
                "level": log_level,
                "propagate": True,
            },
            "mcts.agents": {
                "level": log_level,
                "propagate": True,
            },
            "mcts.observability": {
                "level": log_level,
                "propagate": True,
            },
            "mcts.storage": {
                "level": log_level,
                "propagate": True,
            },
            # Chess verification loggers
            "chess": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "chess.verification": {
                "level": log_level,
                "propagate": True,
            },
            "chess.engines": {
                "level": log_level,
                "propagate": True,
            },
            "chess.observability": {
                "level": log_level,
                "propagate": True,
            },
            "chess.move_selection": {
                "level": log_level,
                "propagate": True,
            },
            # Third-party loggers (quieter)
            "httpx": {
                "level": "WARNING",
                "propagate": True,
            },
            "opentelemetry": {
                "level": "WARNING",
                "propagate": True,
            },
            "aioboto3": {
                "level": "WARNING",
                "propagate": True,
            },
            "botocore": {
                "level": "WARNING",
                "propagate": True,
            },
        },
    }

    # Add performance metrics filter if requested
    if include_performance_metrics:
        config["filters"]["performance_metrics"] = {
            "()": PerformanceMetricsFilter,
        }
        config["handlers"]["console"]["filters"].append("performance_metrics")

    # Add file handler if requested
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_output else "standard",
            "filters": ["correlation_id"],
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        if include_performance_metrics:
            config["handlers"]["file"]["filters"].append("performance_metrics")

        # Add file handler to all loggers
        config["loggers"][""]["handlers"].append("file")
        config["loggers"]["mcts"]["handlers"].append("file")

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Uses hierarchical naming under 'mcts' root logger.
    Example: get_logger("framework.graph") returns logger "mcts.framework.graph"

    Args:
        name: Logger name (will be prefixed with 'mcts.')

    Returns:
        Configured logger instance
    """
    if not name.startswith("mcts."):
        name = f"mcts.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding temporary log context."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self._old_metadata: dict[Any, Any] = {}

    def __enter__(self) -> "LogContext":
        self._old_metadata = get_request_metadata().copy()
        new_metadata = {**self._old_metadata, **self.kwargs}
        set_request_metadata(new_metadata)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        set_request_metadata(self._old_metadata)


def log_execution_time(logger: logging.Logger | None = None, level: int = logging.INFO):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (defaults to function's module logger)
        level: Log level for timing message

    Example:
        @log_execution_time()
        def my_function():
            ...
    """

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss

            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                memory_delta_mb = (psutil.Process().memory_info().rss - start_memory) / (1024 * 1024)

                logger.log(
                    level,
                    f"Function {func.__name__} completed",
                    extra={
                        "timing": {
                            "function": func.__name__,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "success": success,
                            "error": error,
                            "memory_delta_mb": round(memory_delta_mb, 2),
                        }
                    },
                )

            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss

            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                memory_delta_mb = (psutil.Process().memory_info().rss - start_memory) / (1024 * 1024)

                logger.log(
                    level,
                    f"Async function {func.__name__} completed",
                    extra={
                        "timing": {
                            "function": func.__name__,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "success": success,
                            "error": error,
                            "memory_delta_mb": round(memory_delta_mb, 2),
                        }
                    },
                )

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Import asyncio for decorator
import asyncio  # noqa: E402


class StructuredLogger:
    """
    Convenience wrapper for structured logging with automatic correlation IDs.

    Provides a clean API for logging with structured data while automatically
    handling correlation ID injection and secret sanitization.

    Example:
        >>> logger = StructuredLogger("my.module")
        >>> logger.info("Processing request", user_id="123", action="query")
        >>> logger.log_timing("database_query", duration_ms=45.2)
    """

    def __init__(self, name: str) -> None:
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically __name__ from calling module)
        """
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **extra) -> None:
        """
        Internal log method with correlation ID and sanitization.

        Args:
            level: Logging level (e.g., logging.INFO)
            message: Log message
            **extra: Additional structured data to include
        """
        extra["correlation_id"] = get_correlation_id()
        sanitized_extra = sanitize_dict(extra) if extra else {}
        self._logger.log(level, sanitize_message(message), extra=sanitized_extra)

    def debug(self, message: str, **extra) -> None:
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **extra)

    def info(self, message: str, **extra) -> None:
        """Log info message with structured data."""
        self._log(logging.INFO, message, **extra)

    def warning(self, message: str, **extra) -> None:
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **extra)

    def error(self, message: str, **extra) -> None:
        """Log error message with structured data."""
        self._log(logging.ERROR, message, **extra)

    def critical(self, message: str, **extra) -> None:
        """Log critical message with structured data."""
        self._log(logging.CRITICAL, message, **extra)

    def exception(self, message: str, **extra) -> None:
        """Log exception with traceback and structured data."""
        extra["correlation_id"] = get_correlation_id()
        extra["traceback"] = traceback.format_exc()
        sanitized_extra = sanitize_dict(extra) if extra else {}
        self._logger.exception(sanitize_message(message), extra=sanitized_extra)

    def log_timing(self, operation: str, duration_ms: float, **extra) -> None:
        """
        Log operation timing.

        Args:
            operation: Name of the timed operation
            duration_ms: Duration in milliseconds
            **extra: Additional context
        """
        self.info(
            f"Operation completed: {operation}",
            timing_ms=round(duration_ms, 2),
            operation=operation,
            **extra,
        )

    def log_memory(self, component: str, memory_mb: float, **extra) -> None:
        """
        Log memory usage.

        Args:
            component: Component name
            memory_mb: Memory usage in megabytes
            **extra: Additional context
        """
        self.debug(
            f"Memory usage: {component}",
            memory_mb=round(memory_mb, 2),
            component=component,
            **extra,
        )

    def log_mcts_iteration(
        self,
        iteration: int,
        tree_depth: int,
        nodes_explored: int,
        best_action: str | None = None,
        ucb_score: float | None = None,
        **extra,
    ) -> None:
        """
        Log MCTS iteration details.

        Args:
            iteration: Current iteration number
            tree_depth: Current tree depth
            nodes_explored: Number of nodes explored
            best_action: Best action found (if any)
            ucb_score: UCB score of best action
            **extra: Additional context
        """
        self.debug(
            f"MCTS iteration {iteration}",
            iteration=iteration,
            tree_depth=tree_depth,
            nodes_explored=nodes_explored,
            best_action=best_action,
            ucb_score=round(ucb_score, 4) if ucb_score is not None else None,
            **extra,
        )

    def log_agent_execution(
        self,
        agent_name: str,
        duration_ms: float,
        confidence: float,
        success: bool = True,
        **extra,
    ) -> None:
        """
        Log agent execution details.

        Args:
            agent_name: Name of the agent
            duration_ms: Execution duration in milliseconds
            confidence: Agent confidence score
            success: Whether execution succeeded
            **extra: Additional context
        """
        level = logging.INFO if success else logging.WARNING
        self._log(
            level,
            f"Agent {agent_name} execution {'completed' if success else 'failed'}",
            agent_name=agent_name,
            timing_ms=round(duration_ms, 2),
            confidence=round(confidence, 4),
            success=success,
            **extra,
        )


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)
