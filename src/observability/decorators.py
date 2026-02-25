"""
Comprehensive logging decorators and debugging utilities.

Provides reusable decorators for:
- Function call logging with arguments
- Execution time tracking
- Error handling and logging
- Retry logic with backoff
- Input/output validation logging
- Cache hit/miss tracking

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 11
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class LogConfig:
    """Configuration for logging decorators."""

    # Log levels
    entry_level: int = logging.DEBUG
    exit_level: int = logging.DEBUG
    error_level: int = logging.ERROR

    # What to log
    log_args: bool = True
    log_result: bool = True
    log_exception: bool = True
    log_duration: bool = True

    # Truncation
    max_arg_length: int = 200
    max_result_length: int = 500

    # Sensitive field masking
    sensitive_fields: list[str] = field(
        default_factory=lambda: ["password", "api_key", "token", "secret", "credentials"]
    )

    @classmethod
    def from_settings(cls) -> LogConfig:
        """Create config from settings."""
        settings = get_settings()
        log_level = getattr(settings, "LOG_LEVEL", "INFO")
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        base_level = level_map.get(log_level, logging.INFO)

        return cls(
            entry_level=logging.DEBUG,
            exit_level=base_level,
            error_level=logging.ERROR,
        )


def _truncate(value: Any, max_length: int) -> str:
    """Truncate string representation if too long."""
    str_value = str(value)
    if len(str_value) > max_length:
        return str_value[:max_length] + "..."
    return str_value


def _mask_sensitive(value: Any, sensitive_fields: list[str]) -> Any:
    """Mask sensitive fields in dictionaries."""
    if isinstance(value, dict):
        masked = {}
        for k, v in value.items():
            if any(s in k.lower() for s in sensitive_fields):
                masked[k] = "***MASKED***"
            else:
                masked[k] = _mask_sensitive(v, sensitive_fields)
        return masked
    elif isinstance(value, list):
        return [_mask_sensitive(v, sensitive_fields) for v in value]
    return value


def _format_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    config: LogConfig,
) -> str:
    """Format function arguments for logging."""
    parts = []

    # Format positional args (skip self/cls)
    for i, arg in enumerate(args):
        if i == 0 and hasattr(arg, "__class__"):
            # Skip self/cls
            continue
        masked = _mask_sensitive(arg, config.sensitive_fields)
        parts.append(_truncate(masked, config.max_arg_length))

    # Format keyword args
    for key, value in kwargs.items():
        masked = _mask_sensitive(value, config.sensitive_fields)
        parts.append(f"{key}={_truncate(masked, config.max_arg_length)}")

    return ", ".join(parts)


def logged(
    name: str | None = None,
    config: LogConfig | None = None,
    logger_override: logging.Logger | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for comprehensive function call logging.

    Logs function entry, exit, duration, and exceptions.

    Args:
        name: Optional custom name for logging (defaults to function name)
        config: Logging configuration
        logger_override: Use specific logger instead of module logger

    Returns:
        Decorated function

    Example:
        >>> @logged()
        ... def process_data(data: dict) -> dict:
        ...     return {"result": data}

        >>> @logged(name="custom_name", config=LogConfig(log_args=False))
        ... async def async_process(x: int) -> int:
        ...     return x * 2
    """
    cfg = config or LogConfig.from_settings()
    log = logger_override or logger

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        func_name = name or func.__name__
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()

            # Log entry
            if cfg.log_args:
                args_str = _format_args(args, kwargs, cfg)
                log.log(cfg.entry_level, f"[ENTER] {func_name}({args_str})")
            else:
                log.log(cfg.entry_level, f"[ENTER] {func_name}")

            try:
                result = func(*args, **kwargs)

                # Log exit
                duration_ms = (time.time() - start_time) * 1000
                if cfg.log_result:
                    result_str = _truncate(
                        _mask_sensitive(result, cfg.sensitive_fields),
                        cfg.max_result_length,
                    )
                    log.log(
                        cfg.exit_level,
                        f"[EXIT] {func_name} -> {result_str} ({duration_ms:.2f}ms)",
                    )
                elif cfg.log_duration:
                    log.log(cfg.exit_level, f"[EXIT] {func_name} ({duration_ms:.2f}ms)")

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if cfg.log_exception:
                    log.log(
                        cfg.error_level,
                        f"[ERROR] {func_name} raised {type(e).__name__}: {e} ({duration_ms:.2f}ms)",
                        exc_info=True,
                    )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()

            # Log entry
            if cfg.log_args:
                args_str = _format_args(args, kwargs, cfg)
                log.log(cfg.entry_level, f"[ENTER] {func_name}({args_str})")
            else:
                log.log(cfg.entry_level, f"[ENTER] {func_name}")

            try:
                result = await func(*args, **kwargs)  # type: ignore

                # Log exit
                duration_ms = (time.time() - start_time) * 1000
                if cfg.log_result:
                    result_str = _truncate(
                        _mask_sensitive(result, cfg.sensitive_fields),
                        cfg.max_result_length,
                    )
                    log.log(
                        cfg.exit_level,
                        f"[EXIT] {func_name} -> {result_str} ({duration_ms:.2f}ms)",
                    )
                elif cfg.log_duration:
                    log.log(cfg.exit_level, f"[EXIT] {func_name} ({duration_ms:.2f}ms)")

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if cfg.log_exception:
                    log.log(
                        cfg.error_level,
                        f"[ERROR] {func_name} raised {type(e).__name__}: {e} ({duration_ms:.2f}ms)",
                        exc_info=True,
                    )
                raise

        if is_async:
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def timed(
    metric_name: str | None = None,
    threshold_ms: float | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for timing function execution.

    Logs warning if execution exceeds threshold.

    Args:
        metric_name: Name for timing metric (defaults to function name)
        threshold_ms: Warning threshold in milliseconds

    Returns:
        Decorated function

    Example:
        >>> @timed(threshold_ms=100)
        ... def slow_operation():
        ...     time.sleep(0.2)
    """
    settings = get_settings()
    default_threshold = getattr(settings, "SLOW_OPERATION_THRESHOLD_MS", 1000.0)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        name = metric_name or func.__name__
        threshold = threshold_ms or default_threshold
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                if duration_ms > threshold:
                    logger.warning(f"Slow operation: {name} took {duration_ms:.2f}ms (threshold: {threshold}ms)")
                else:
                    logger.debug(f"Timing: {name} took {duration_ms:.2f}ms")

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)  # type: ignore
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                if duration_ms > threshold:
                    logger.warning(f"Slow operation: {name} took {duration_ms:.2f}ms (threshold: {threshold}ms)")
                else:
                    logger.debug(f"Timing: {name} took {duration_ms:.2f}ms")

        if is_async:
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retrying failed operations with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Factor to multiply delay after each attempt
        exceptions: Tuple of exception types to catch
        on_retry: Optional callback on retry (exception, attempt_number)

    Returns:
        Decorated function

    Example:
        >>> @retry(max_attempts=3, exceptions=(ConnectionError,))
        ... def connect_to_service():
        ...     # May raise ConnectionError
        ...     pass
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        raise

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)
                    delay *= backoff_factor

            raise last_exception  # type: ignore

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        raise

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    await asyncio.sleep(delay)
                    delay *= backoff_factor

            raise last_exception  # type: ignore

        if is_async:
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def cached(
    ttl_seconds: float | None = None,
    max_size: int = 128,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Simple in-memory cache decorator with TTL.

    For more advanced caching, use src/framework/caching.py

    Args:
        ttl_seconds: Time-to-live in seconds (None for infinite)
        max_size: Maximum cache size

    Returns:
        Decorated function with caching

    Example:
        >>> @cached(ttl_seconds=60)
        ... def fetch_data(key: str) -> dict:
        ...     return expensive_operation(key)
    """
    settings = get_settings()
    default_ttl = getattr(settings, "DEFAULT_CACHE_TTL_SECONDS", 300.0)
    ttl = ttl_seconds if ttl_seconds is not None else default_ttl

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache: dict[str, tuple[T, float]] = {}

        def _make_key(args: tuple, kwargs: dict) -> str:
            """Create cache key from arguments."""
            key_parts = [str(a) for a in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            return ":".join(key_parts)

        def _is_expired(cached_time: float) -> bool:
            """Check if cached entry is expired."""
            if ttl is None or ttl <= 0:
                return False
            return time.time() - cached_time > ttl

        def _evict_if_needed() -> None:
            """Evict oldest entries if cache is full."""
            while len(cache) >= max_size:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
                logger.debug(f"Cache eviction: removed {oldest_key}")

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = _make_key(args, kwargs)

            # Check cache
            if key in cache:
                value, cached_time = cache[key]
                if not _is_expired(cached_time):
                    logger.debug(f"Cache hit: {func.__name__}({key[:50]}...)")
                    return value
                else:
                    del cache[key]
                    logger.debug(f"Cache expired: {func.__name__}({key[:50]}...)")

            # Cache miss
            logger.debug(f"Cache miss: {func.__name__}({key[:50]}...)")
            _evict_if_needed()

            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        # Expose cache control methods
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore
        wrapper.cache_info = lambda: {"size": len(cache), "max_size": max_size, "ttl": ttl}  # type: ignore

        return wrapper

    return decorator


def debug_on_error(
    log_locals: bool = True,
    log_stack: bool = True,
    reraise: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for debugging errors with detailed context.

    Args:
        log_locals: Log local variables on error
        log_stack: Log full stack trace
        reraise: Re-raise the exception after logging

    Returns:
        Decorated function

    Example:
        >>> @debug_on_error(log_locals=True)
        ... def complex_operation(data: dict):
        ...     # If this fails, locals will be logged
        ...     pass
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = {
                    "function": func.__name__,
                    "exception": type(e).__name__,
                    "message": str(e),
                }

                if log_stack:
                    error_context["traceback"] = traceback.format_exc()

                if log_locals:
                    # Get local variables from the frame where error occurred
                    try:
                        tb = e.__traceback__
                        if tb:
                            while tb.tb_next:
                                tb = tb.tb_next
                            local_vars = tb.tb_frame.f_locals
                            # Mask sensitive data
                            safe_locals = {}
                            for k, v in local_vars.items():
                                if any(s in k.lower() for s in ["password", "secret", "token", "key"]):
                                    safe_locals[k] = "***MASKED***"
                                else:
                                    try:
                                        safe_locals[k] = _truncate(repr(v), 200)
                                    except Exception:
                                        safe_locals[k] = "<unrepresentable>"
                            error_context["locals"] = safe_locals
                    except Exception:
                        pass

                logger.error(
                    f"Error in {func.__name__}: {e}",
                    extra={"debug_context": error_context},
                )

                if reraise:
                    raise

        return wrapper

    return decorator


def validate_args(**validators: Callable[[Any], bool]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for validating function arguments.

    Args:
        **validators: Mapping of argument name to validator function

    Returns:
        Decorated function with validation

    Example:
        >>> @validate_args(
        ...     x=lambda v: v > 0,
        ...     name=lambda v: len(v) > 0
        ... )
        ... def process(x: int, name: str):
        ...     pass
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        import inspect

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Bind arguments to parameter names
            bound_args = {}
            for i, arg in enumerate(args):
                if i < len(param_names):
                    bound_args[param_names[i]] = arg
            bound_args.update(kwargs)

            # Validate
            for arg_name, validator in validators.items():
                if arg_name in bound_args:
                    value = bound_args[arg_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for argument '{arg_name}': "
                            f"value {_truncate(value, 100)} did not pass validation"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "LogConfig",
    "logged",
    "timed",
    "retry",
    "cached",
    "debug_on_error",
    "validate_args",
]
