"""
Chess Error Injection Framework.

Provides utilities for injecting errors and testing error handling
in chess verification and gameplay components.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.games.chess.ensemble_agent import ChessEnsembleAgent

from src.games.chess.state import ChessGameState
from src.games.chess.verification.types import MoveType, MoveValidationResult


@dataclass
class InjectedError:
    """Represents an injected error."""

    error_type: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


class ChessErrorInjector:
    """Injects errors for robustness testing.

    Provides methods to simulate various error conditions
    in chess components for testing error handling.

    Example:
        >>> injector = ChessErrorInjector()
        >>> invalid_state = injector.corrupt_position(state)
        >>> # Test error handling with invalid state
    """

    def __init__(self) -> None:
        """Initialize the error injector."""
        self._injected_errors: list[InjectedError] = []
        self._patches: list[Any] = []

    @property
    def injected_errors(self) -> list[InjectedError]:
        """Get list of injected errors."""
        return self._injected_errors.copy()

    def corrupt_position(
        self,
        state: ChessGameState,
        corruption_type: str = "random",
    ) -> ChessGameState:
        """Create invalid position for testing error handling.

        Args:
            state: Valid chess state
            corruption_type: Type of corruption to apply

        Returns:
            Corrupted ChessGameState
        """
        import chess

        board = chess.Board(state.fen)

        if corruption_type == "remove_king":
            # Remove white king (creates invalid position)
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.KING and piece.color == chess.WHITE:
                    board.remove_piece_at(square)
                    break
        elif corruption_type == "double_king":
            # Add extra white king
            for square in chess.SQUARES:
                if board.piece_at(square) is None:
                    board.set_piece_at(square, chess.Piece(chess.KING, chess.WHITE))
                    break
        elif corruption_type == "pawn_on_first_rank":
            # Put pawn on 1st rank
            board.set_piece_at(chess.A1, chess.Piece(chess.PAWN, chess.WHITE))
        else:
            # Random corruption - swap pieces
            pieces = list(board.piece_map().items())
            if len(pieces) >= 2:
                sq1, sq2 = pieces[0][0], pieces[1][0]
                p1, p2 = board.piece_at(sq1), board.piece_at(sq2)
                board.remove_piece_at(sq1)
                board.remove_piece_at(sq2)
                if p2:
                    board.set_piece_at(sq1, p2)
                if p1:
                    board.set_piece_at(sq2, p1)

        self._injected_errors.append(
            InjectedError(
                error_type="corrupted_position",
                message=f"Applied {corruption_type} corruption",
                context={"original_fen": state.fen, "corrupted_fen": board.fen()},
            )
        )

        # Return state with corrupted FEN (may not be fully valid)
        try:
            return ChessGameState.from_fen(board.fen())
        except Exception:
            # If we can't create state, return original
            return state

    def create_invalid_move_response(
        self,
        state: ChessGameState,
    ) -> MoveValidationResult:
        """Create a mock invalid move response.

        Args:
            state: Chess position

        Returns:
            MoveValidationResult for an invalid move
        """
        from src.games.chess.verification.types import VerificationIssue, VerificationSeverity

        self._injected_errors.append(
            InjectedError(
                error_type="invalid_move_response",
                message="Injected invalid move response",
                context={"fen": state.fen},
            )
        )

        return MoveValidationResult(
            is_valid=False,
            move_uci="invalid",
            move_type=MoveType.NORMAL,
            issues=[
                VerificationIssue(
                    code="INJECTED_ERROR",
                    message="Injected invalid move for testing",
                    severity=VerificationSeverity.ERROR,
                )
            ],
            is_legal_in_position=False,
        )

    def patch_agent_with_failure(
        self,
        agent: ChessEnsembleAgent,
        failure_type: str = "exception",
    ) -> None:
        """Patch agent to simulate failure.

        Args:
            agent: Chess ensemble agent to patch
            failure_type: Type of failure (exception, timeout, invalid_move)
        """
        original_get_best_move = agent.get_best_move

        if failure_type == "exception":

            async def failing_get_best_move(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Injected agent failure")

            agent.get_best_move = failing_get_best_move  # type: ignore

        elif failure_type == "timeout":

            async def timeout_get_best_move(*args: Any, **kwargs: Any) -> Any:
                await asyncio.sleep(60)  # Simulate timeout
                return await original_get_best_move(*args, **kwargs)

            agent.get_best_move = timeout_get_best_move  # type: ignore

        elif failure_type == "invalid_move":

            async def invalid_move_get_best_move(*args: Any, **kwargs: Any) -> Any:
                result = await original_get_best_move(*args, **kwargs)
                # Replace best move with invalid move
                result.best_move = "invalid_move"
                return result

            agent.get_best_move = invalid_move_get_best_move  # type: ignore

        self._injected_errors.append(
            InjectedError(
                error_type=f"agent_{failure_type}",
                message=f"Patched agent with {failure_type} failure",
            )
        )

    def create_timeout_context(
        self,
        timeout_seconds: float,
    ) -> TimeoutContext:
        """Create a context that simulates timeout.

        Args:
            timeout_seconds: Timeout duration

        Returns:
            TimeoutContext for use in with statement
        """
        return TimeoutContext(timeout_seconds)

    def reset(self) -> None:
        """Reset all injected errors and patches."""
        self._injected_errors.clear()
        for p in self._patches:
            p.stop()
        self._patches.clear()


class TimeoutContext:
    """Context manager for simulating timeouts."""

    def __init__(self, timeout_seconds: float) -> None:
        """Initialize timeout context.

        Args:
            timeout_seconds: Timeout duration
        """
        self.timeout_seconds = timeout_seconds
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> TimeoutContext:
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context."""
        pass

    async def run_with_timeout(
        self,
        coro: Any,
    ) -> Any:
        """Run coroutine with timeout.

        Args:
            coro: Coroutine to run

        Returns:
            Result of coroutine

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        return await asyncio.wait_for(coro, timeout=self.timeout_seconds)


class MockEnsembleAgent:
    """Mock ensemble agent for testing.

    Provides configurable responses for testing verification
    without needing real neural networks.
    """

    def __init__(
        self,
        default_move: str = "e2e4",
        default_confidence: float = 0.8,
        default_value: float = 0.0,
        should_fail: bool = False,
    ) -> None:
        """Initialize mock agent.

        Args:
            default_move: Default move to return
            default_confidence: Default confidence
            default_value: Default value estimate
            should_fail: Whether to raise exceptions
        """
        self.default_move = default_move
        self.default_confidence = default_confidence
        self.default_value = default_value
        self.should_fail = should_fail
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def get_best_move(
        self,
        state: ChessGameState,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """Get best move (mocked).

        Args:
            state: Chess position
            temperature: Temperature for move selection
            **kwargs: Additional arguments

        Returns:
            Mocked ensemble response
        """
        self.call_count += 1
        self.calls.append({"state_fen": state.fen, "temperature": temperature})

        if self.should_fail:
            raise RuntimeError("Mock agent configured to fail")

        # Return mock response
        from dataclasses import dataclass

        @dataclass
        class MockResponse:
            best_move: str
            confidence: float
            value_estimate: float
            move_probabilities: dict[str, float]
            routing_decision: Any
            agent_responses: dict[str, Any]

        @dataclass
        class MockRouting:
            primary_agent: Any

        from src.games.chess.config import AgentType

        return MockResponse(
            best_move=self.default_move
            if self.default_move in state.get_legal_actions()
            else state.get_legal_actions()[0],
            confidence=self.default_confidence,
            value_estimate=self.default_value,
            move_probabilities={self.default_move: 0.9},
            routing_decision=MockRouting(primary_agent=AgentType.MCTS),
            agent_responses={},
        )


def create_error_injector() -> ChessErrorInjector:
    """Factory function to create a ChessErrorInjector.

    Returns:
        ChessErrorInjector instance
    """
    return ChessErrorInjector()


def create_mock_ensemble_agent(
    default_move: str = "e2e4",
    should_fail: bool = False,
) -> MockEnsembleAgent:
    """Factory function to create a MockEnsembleAgent.

    Args:
        default_move: Default move to return
        should_fail: Whether agent should fail

    Returns:
        MockEnsembleAgent instance
    """
    return MockEnsembleAgent(
        default_move=default_move,
        should_fail=should_fail,
    )
