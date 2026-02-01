"""
Unit Tests for Error Recovery.

Tests error handling and recovery in chess verification components.
"""

from __future__ import annotations

import pytest

from src.games.chess.state import ChessGameState
from src.games.chess.verification import (
    ChessGameVerifier,
    MoveValidator,
    create_game_verifier,
    create_move_validator,
)
from tests.games.chess.builders import initial_position
from tests.games.chess.error_injection import (
    ChessErrorInjector,
    create_error_injector,
    create_mock_ensemble_agent,
)
from tests.games.chess.fixtures import ERROR_INJECTION_SCENARIOS


class TestMoveValidatorErrorRecovery:
    """Tests for MoveValidator error recovery."""

    @pytest.fixture
    def validator(self) -> MoveValidator:
        """Create a move validator."""
        return create_move_validator()

    @pytest.fixture
    def error_injector(self) -> ChessErrorInjector:
        """Create an error injector."""
        return create_error_injector()

    @pytest.mark.unit
    def test_recovery_from_invalid_uci(self, validator: MoveValidator) -> None:
        """Test recovery from invalid UCI format."""
        state = initial_position()
        result = validator.validate_move(state, "not_a_move")

        assert result.is_valid is False
        assert any(issue.code == "INVALID_UCI_FORMAT" for issue in result.issues)
        # Validator should not crash

    @pytest.mark.unit
    def test_recovery_from_empty_move(self, validator: MoveValidator) -> None:
        """Test recovery from empty move string."""
        state = initial_position()
        result = validator.validate_move(state, "")

        assert result.is_valid is False
        # Should handle gracefully

    @pytest.mark.unit
    def test_recovery_from_illegal_move(self, validator: MoveValidator) -> None:
        """Test recovery from illegal move."""
        state = initial_position()
        result = validator.validate_move(state, "e1e8")  # King can't move there

        assert result.is_valid is False
        assert any(issue.code == "ILLEGAL_MOVE" for issue in result.issues)

    @pytest.mark.unit
    @pytest.mark.parametrize("scenario", ERROR_INJECTION_SCENARIOS, ids=lambda s: s["name"])
    def test_error_scenarios(self, validator: MoveValidator, scenario: dict) -> None:
        """Test various error injection scenarios."""
        if scenario["error_type"] == "invalid_fen":
            # Test FEN parsing through game verifier
            verifier = create_game_verifier()
            result = verifier.verify_position(scenario["fen"])
            assert result.is_valid is False
        elif scenario["error_type"] in ("invalid_uci", "illegal_move"):
            state = ChessGameState.from_fen(scenario["fen"])
            result = validator.validate_move(state, scenario["move"])
            assert result.is_valid is False


class TestGameVerifierErrorRecovery:
    """Tests for ChessGameVerifier error recovery."""

    @pytest.fixture
    def verifier(self) -> ChessGameVerifier:
        """Create a game verifier."""
        return create_game_verifier()

    @pytest.mark.unit
    def test_recovery_from_invalid_fen(self, verifier: ChessGameVerifier) -> None:
        """Test recovery from invalid FEN."""
        result = verifier.verify_position("invalid_fen_string")

        assert result.is_valid is False
        assert any(issue.code == "INVALID_FEN" for issue in result.issues)

    @pytest.mark.unit
    def test_recovery_from_corrupted_sequence(self, verifier: ChessGameVerifier) -> None:
        """Test recovery from corrupted move sequence."""
        moves = ["e2e4", "corrupted", "e7e5"]
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        result = verifier.verify_move_sequence(initial_fen, moves)

        assert result.is_valid is False
        # Should have processed some moves before failing

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_recovery_from_partial_game(self, verifier: ChessGameVerifier) -> None:
        """Test recovery from partially valid game."""
        # Valid moves followed by invalid
        moves = ["e2e4", "e7e5", "g1f3", "invalid_move"]

        result = await verifier.verify_full_game(moves=moves)

        assert result.is_valid is False
        # Should detect the invalid move

    @pytest.mark.unit
    def test_handles_empty_move_list(self, verifier: ChessGameVerifier) -> None:
        """Test handling of empty move list."""
        moves: list[str] = []
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        result = verifier.verify_move_sequence(initial_fen, moves)

        # Empty sequence should be valid
        assert result.is_valid is True
        assert result.total_moves == 0


class TestCorruptedPositionRecovery:
    """Tests for recovery from corrupted positions."""

    @pytest.fixture
    def error_injector(self) -> ChessErrorInjector:
        """Create an error injector."""
        return create_error_injector()

    @pytest.fixture
    def validator(self) -> MoveValidator:
        """Create a move validator."""
        return create_move_validator()

    @pytest.mark.unit
    def test_corrupted_position_handling(
        self,
        error_injector: ChessErrorInjector,
        validator: MoveValidator,
    ) -> None:
        """Test handling of corrupted position."""
        state = initial_position()

        # Corrupt the position
        corrupted = error_injector.corrupt_position(state, "random")

        # Validator should handle gracefully
        legal_moves = corrupted.get_legal_actions()
        if legal_moves:
            validator.validate_move(corrupted, legal_moves[0])
            # Should produce a result without crashing

    @pytest.mark.unit
    def test_injector_tracks_errors(
        self,
        error_injector: ChessErrorInjector,
    ) -> None:
        """Test that injector tracks injected errors."""
        state = initial_position()

        error_injector.corrupt_position(state, "random")
        error_injector.create_invalid_move_response(state)

        assert len(error_injector.injected_errors) == 2
        assert any(e.error_type == "corrupted_position" for e in error_injector.injected_errors)
        assert any(e.error_type == "invalid_move_response" for e in error_injector.injected_errors)

    @pytest.mark.unit
    def test_injector_reset(
        self,
        error_injector: ChessErrorInjector,
    ) -> None:
        """Test that injector can be reset."""
        state = initial_position()

        error_injector.corrupt_position(state, "random")
        assert len(error_injector.injected_errors) > 0

        error_injector.reset()
        assert len(error_injector.injected_errors) == 0


class TestMockEnsembleAgent:
    """Tests for mock ensemble agent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_agent_returns_move(self) -> None:
        """Test mock agent returns a move."""
        agent = create_mock_ensemble_agent(default_move="e2e4")
        state = initial_position()

        response = await agent.get_best_move(state)

        assert response.best_move == "e2e4"
        assert response.confidence == 0.8
        assert agent.call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_agent_failure(self) -> None:
        """Test mock agent failure mode."""
        agent = create_mock_ensemble_agent(should_fail=True)
        state = initial_position()

        with pytest.raises(RuntimeError, match="Mock agent configured to fail"):
            await agent.get_best_move(state)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_agent_tracks_calls(self) -> None:
        """Test mock agent tracks calls."""
        agent = create_mock_ensemble_agent()
        state1 = initial_position()
        state2 = ChessGameState.from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )

        await agent.get_best_move(state1, temperature=0.0)
        await agent.get_best_move(state2, temperature=0.5)

        assert agent.call_count == 2
        assert len(agent.calls) == 2
        assert agent.calls[0]["temperature"] == 0.0
        assert agent.calls[1]["temperature"] == 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_agent_fallback_to_legal_move(self) -> None:
        """Test mock agent falls back to legal move if default is illegal."""
        agent = create_mock_ensemble_agent(default_move="a1h8")  # Illegal move
        state = initial_position()

        response = await agent.get_best_move(state)

        # Should fall back to a legal move
        assert response.best_move in state.get_legal_actions()
