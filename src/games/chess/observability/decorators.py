"""
Chess Observability Decorators.

Provides decorators for tracing, verification, and debugging
chess gameplay operations.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.games.chess.constants import truncate_fen
from src.games.chess.observability.logger import get_chess_logger
from src.games.chess.observability.metrics import ChessMetricsCollector
from src.observability.logging import get_correlation_id

F = TypeVar("F", bound=Callable[..., Any])


def traced_move_selection(func: F) -> F:
    """Decorator to trace move selection with full context.

    Logs the start and end of move selection with timing,
    position details, and selected move.

    Example:
        >>> @traced_move_selection
        ... async def select_move(self, state: ChessGameState) -> str:
        ...     ...
    """

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_chess_logger("move_selection")
        correlation_id = get_correlation_id()

        # Try to extract state from arguments
        state = None
        for arg in args:
            if hasattr(arg, "fen"):
                state = arg
                break
        if state is None:
            state = kwargs.get("state")

        # Log start
        if state is not None:
            logger.debug(
                "Move selection started",
                fen=truncate_fen(state.fen),
                phase=(state.get_game_phase().value if hasattr(state, "get_game_phase") else "unknown"),
                legal_moves=(len(state.get_legal_actions()) if hasattr(state, "get_legal_actions") else 0),
                correlation_id=correlation_id,
            )

        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Log result
            if hasattr(result, "best_move"):
                logger.info(
                    "Move selected",
                    move=result.best_move,
                    confidence=getattr(result, "confidence", 0.0),
                    routing=(
                        (
                            pa.value
                            if (pa := getattr(result.routing_decision, "primary_agent", None)) is not None
                            else "unknown"
                        )
                        if hasattr(result, "routing_decision")
                        else "unknown"
                    ),
                    duration_ms=round(elapsed_ms, 2),
                    correlation_id=correlation_id,
                )
            else:
                logger.info(
                    "Move selection completed",
                    result=str(result)[:50],
                    duration_ms=round(elapsed_ms, 2),
                    correlation_id=correlation_id,
                )

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Move selection failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(elapsed_ms, 2),
                correlation_id=correlation_id,
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_chess_logger("move_selection")
        correlation_id = get_correlation_id()

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "Move selection completed",
                duration_ms=round(elapsed_ms, 2),
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Move selection failed",
                error=str(e),
                duration_ms=round(elapsed_ms, 2),
                correlation_id=correlation_id,
            )
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    return sync_wrapper  # type: ignore


def verified_game_play(verification_level: str = "full") -> Callable[[F], F]:
    """Decorator to verify games during play.

    Args:
        verification_level: Level of verification (full, basic, minimal)

    Returns:
        Decorated function

    Example:
        >>> @verified_game_play(verification_level="full")
        ... async def play_game(self) -> GameResult:
        ...     ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.games.chess.verification import ChessVerificationFactory

            logger = get_chess_logger("game_play")
            verifier = ChessVerificationFactory().create_game_verifier()

            start_time = time.perf_counter()

            # Execute the game
            result = await func(*args, **kwargs)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Post-play verification
            if hasattr(result, "moves") and verification_level != "minimal":
                moves = result.moves if hasattr(result, "moves") else []
                verification_result = await verifier.verify_full_game(moves)

                if not verification_result.is_valid:
                    logger.warning(
                        "Game verification failed",
                        issues=[str(i) for i in verification_result.issues[:5]],
                        moves_count=len(moves),
                    )
                else:
                    logger.info(
                        "Game verified successfully",
                        moves_count=len(moves),
                        result=verification_result.result.value,
                        duration_ms=round(elapsed_ms, 2),
                    )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_chess_logger("game_play")

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "Game play completed",
                duration_ms=round(elapsed_ms, 2),
            )

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def with_verification_context(
    log_moves: bool = True,
    log_positions: bool = False,
    collect_metrics: bool = True,
) -> Callable[[F], F]:
    """Decorator to add verification context to a function.

    Args:
        log_moves: Whether to log individual moves
        log_positions: Whether to log position details
        collect_metrics: Whether to collect metrics

    Returns:
        Decorated function

    Example:
        >>> @with_verification_context(log_moves=True)
        ... async def analyze_game(self, moves: list[str]) -> Analysis:
        ...     ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_chess_logger("verification")
            metrics = ChessMetricsCollector.get_instance() if collect_metrics else None

            # Set up context
            correlation_id = get_correlation_id()

            logger.debug(
                "Verification context started",
                function=func.__name__,
                log_moves=log_moves,
                log_positions=log_positions,
                correlation_id=correlation_id,
            )

            start_time = time.perf_counter()

            try:
                # Execute function
                result = await func(*args, **kwargs)

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Record metrics if enabled
                if metrics is not None:
                    # Try to extract game info from result
                    game_id = getattr(result, "game_id", None) or "unknown"
                    is_valid = getattr(result, "is_valid", True)
                    total_moves = getattr(result, "total_moves", 0)

                    metrics.record_game_verification(
                        game_id=game_id,
                        success=is_valid,
                        duration_ms=elapsed_ms,
                        moves_count=total_moves,
                    )

                logger.debug(
                    "Verification context completed",
                    function=func.__name__,
                    duration_ms=round(elapsed_ms, 2),
                    correlation_id=correlation_id,
                )

                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Record failure metrics
                if metrics is not None:
                    metrics.record_verification_error("execution_error")

                logger.error(
                    "Verification context failed",
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=round(elapsed_ms, 2),
                    correlation_id=correlation_id,
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_chess_logger("verification")
            correlation_id = get_correlation_id()

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                logger.debug(
                    "Verification context completed",
                    function=func.__name__,
                    duration_ms=round(elapsed_ms, 2),
                    correlation_id=correlation_id,
                )

                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                logger.error(
                    "Verification context failed",
                    function=func.__name__,
                    error=str(e),
                    duration_ms=round(elapsed_ms, 2),
                    correlation_id=correlation_id,
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def timed_verification(operation_name: str | None = None) -> Callable[[F], F]:
    """Decorator to time verification operations.

    Args:
        operation_name: Name of the operation (defaults to function name)

    Returns:
        Decorated function

    Example:
        >>> @timed_verification("move_encoding")
        ... def encode_move(self, move: str) -> int:
        ...     ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_chess_logger("timing")
            name = operation_name or func.__name__

            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                f"Verification operation: {name}",
                operation=name,
                duration_ms=round(elapsed_ms, 2),
            )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_chess_logger("timing")
            name = operation_name or func.__name__

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                f"Verification operation: {name}",
                operation=name,
                duration_ms=round(elapsed_ms, 2),
            )

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
