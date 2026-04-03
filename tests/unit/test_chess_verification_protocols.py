"""
Unit tests for chess verification protocols.

Tests that Protocol classes are properly defined and can be implemented/mocked.
Verifies structural subtyping contracts.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.games.chess.verification.protocols import (
    ChessGameVerifierProtocol,
    EnsembleConsistencyCheckerProtocol,
    MoveValidatorProtocol,
    SubAgentVerifierProtocol,
)

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Helpers - Concrete Implementations for Protocol Testing
# =============================================================================


class ConcreteMoveValidator:
    """Concrete implementation of MoveValidatorProtocol for testing."""

    def validate_move(self, state: Any, move_uci: str) -> Any:
        return MagicMock(is_valid=True, move_uci=move_uci)

    def validate_castling(self, state: Any, kingside: bool) -> Any:
        return MagicMock(is_valid=True)

    def validate_en_passant(self, state: Any, move_uci: str) -> Any:
        return MagicMock(is_valid=True)

    def validate_promotion(self, state: Any, move_uci: str) -> Any:
        return MagicMock(is_valid=True)

    def validate_encoding_roundtrip(self, state: Any, move_uci: str) -> Any:
        return MagicMock(is_valid=True)


class ConcreteGameVerifier:
    """Concrete implementation of ChessGameVerifierProtocol for testing."""

    async def verify_full_game(
        self, moves: list[str], expected_outcome: Any = None, game_id: str | None = None
    ) -> Any:
        return MagicMock(is_valid=True)

    def verify_position(self, fen: str) -> Any:
        return MagicMock(is_valid=True)

    def verify_move_sequence(self, initial_fen: str, moves: list[str]) -> Any:
        return MagicMock(is_valid=True)

    async def verify_game_playthrough(self, state: Any, max_moves: int | None = None) -> Any:
        return MagicMock(is_valid=True)


class ConcreteEnsembleChecker:
    """Concrete implementation of EnsembleConsistencyCheckerProtocol for testing."""

    async def check_position_consistency(self, state: Any) -> Any:
        return MagicMock(is_consistent=True)

    async def check_sequence_consistency(self, states: list[Any]) -> list[Any]:
        return [MagicMock(is_consistent=True) for _ in states]

    async def check_game_consistency(
        self, moves: list[str], initial_fen: str | None = None
    ) -> list[Any]:
        return [MagicMock(is_consistent=True)]

    def get_divergence_threshold(self) -> float:
        return 0.3


class ConcreteSubAgentVerifier:
    """Concrete implementation of SubAgentVerifierProtocol for testing."""

    async def verify_agent_response(self, state: Any, agent_name: str) -> Any:
        return MagicMock(is_valid=True)

    async def verify_routing_decision(self, state: Any) -> bool:
        return True


# =============================================================================
# Protocol Definition Tests
# =============================================================================


class TestMoveValidatorProtocol:
    """Tests for MoveValidatorProtocol."""

    def test_protocol_has_required_methods(self) -> None:
        """Test protocol defines all required methods."""
        required_methods = [
            "validate_move",
            "validate_castling",
            "validate_en_passant",
            "validate_promotion",
            "validate_encoding_roundtrip",
        ]
        for method in required_methods:
            assert hasattr(MoveValidatorProtocol, method)

    def test_concrete_implementation_satisfies_protocol(self) -> None:
        """Test concrete class satisfies the protocol structurally."""
        validator = ConcreteMoveValidator()
        # Structural subtyping: if it has all the methods, it satisfies the protocol
        assert hasattr(validator, "validate_move")
        assert hasattr(validator, "validate_castling")
        assert hasattr(validator, "validate_en_passant")
        assert hasattr(validator, "validate_promotion")
        assert hasattr(validator, "validate_encoding_roundtrip")

    def test_validate_move_callable(self) -> None:
        """Test validate_move can be called."""
        validator = ConcreteMoveValidator()
        state = MagicMock()
        result = validator.validate_move(state, "e2e4")
        assert result.is_valid is True

    def test_validate_castling_callable(self) -> None:
        """Test validate_castling can be called."""
        validator = ConcreteMoveValidator()
        state = MagicMock()
        result = validator.validate_castling(state, kingside=True)
        assert result.is_valid is True

    def test_validate_en_passant_callable(self) -> None:
        """Test validate_en_passant can be called."""
        validator = ConcreteMoveValidator()
        state = MagicMock()
        result = validator.validate_en_passant(state, "e5d6")
        assert result.is_valid is True

    def test_validate_promotion_callable(self) -> None:
        """Test validate_promotion can be called."""
        validator = ConcreteMoveValidator()
        state = MagicMock()
        result = validator.validate_promotion(state, "e7e8q")
        assert result.is_valid is True

    def test_validate_encoding_roundtrip_callable(self) -> None:
        """Test validate_encoding_roundtrip can be called."""
        validator = ConcreteMoveValidator()
        state = MagicMock()
        result = validator.validate_encoding_roundtrip(state, "e2e4")
        assert result.is_valid is True

    def test_mock_satisfies_protocol(self) -> None:
        """Test MagicMock can stand in for the protocol."""
        mock_validator = MagicMock(spec=ConcreteMoveValidator)
        mock_validator.validate_move.return_value = MagicMock(is_valid=True)

        result = mock_validator.validate_move(MagicMock(), "e2e4")
        assert result.is_valid is True


class TestChessGameVerifierProtocol:
    """Tests for ChessGameVerifierProtocol."""

    def test_protocol_has_required_methods(self) -> None:
        """Test protocol defines all required methods."""
        required_methods = [
            "verify_full_game",
            "verify_position",
            "verify_move_sequence",
            "verify_game_playthrough",
        ]
        for method in required_methods:
            assert hasattr(ChessGameVerifierProtocol, method)

    def test_concrete_implementation_satisfies_protocol(self) -> None:
        """Test concrete class satisfies the protocol."""
        verifier = ConcreteGameVerifier()
        assert hasattr(verifier, "verify_full_game")
        assert hasattr(verifier, "verify_position")
        assert hasattr(verifier, "verify_move_sequence")
        assert hasattr(verifier, "verify_game_playthrough")

    @pytest.mark.asyncio
    async def test_verify_full_game_async(self) -> None:
        """Test verify_full_game is async."""
        verifier = ConcreteGameVerifier()
        result = await verifier.verify_full_game(["e2e4", "e7e5"])
        assert result.is_valid is True

    def test_verify_position_sync(self) -> None:
        """Test verify_position is synchronous."""
        verifier = ConcreteGameVerifier()
        result = verifier.verify_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert result.is_valid is True

    def test_verify_move_sequence_sync(self) -> None:
        """Test verify_move_sequence is synchronous."""
        verifier = ConcreteGameVerifier()
        result = verifier.verify_move_sequence("start_fen", ["e2e4", "e7e5"])
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_verify_game_playthrough_async(self) -> None:
        """Test verify_game_playthrough is async."""
        verifier = ConcreteGameVerifier()
        state = MagicMock()
        result = await verifier.verify_game_playthrough(state, max_moves=50)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_verify_full_game_with_expected_outcome(self) -> None:
        """Test verify_full_game accepts optional parameters."""
        verifier = ConcreteGameVerifier()
        result = await verifier.verify_full_game(
            moves=["e2e4"], expected_outcome="white_wins", game_id="g-001"
        )
        assert result.is_valid is True


class TestEnsembleConsistencyCheckerProtocol:
    """Tests for EnsembleConsistencyCheckerProtocol."""

    def test_protocol_has_required_methods(self) -> None:
        """Test protocol defines all required methods."""
        required_methods = [
            "check_position_consistency",
            "check_sequence_consistency",
            "check_game_consistency",
            "get_divergence_threshold",
        ]
        for method in required_methods:
            assert hasattr(EnsembleConsistencyCheckerProtocol, method)

    def test_concrete_implementation_satisfies_protocol(self) -> None:
        """Test concrete class satisfies the protocol."""
        checker = ConcreteEnsembleChecker()
        assert hasattr(checker, "check_position_consistency")
        assert hasattr(checker, "check_sequence_consistency")
        assert hasattr(checker, "check_game_consistency")
        assert hasattr(checker, "get_divergence_threshold")

    @pytest.mark.asyncio
    async def test_check_position_consistency_async(self) -> None:
        """Test check_position_consistency is async."""
        checker = ConcreteEnsembleChecker()
        state = MagicMock()
        result = await checker.check_position_consistency(state)
        assert result.is_consistent is True

    @pytest.mark.asyncio
    async def test_check_sequence_consistency_async(self) -> None:
        """Test check_sequence_consistency is async."""
        checker = ConcreteEnsembleChecker()
        states = [MagicMock(), MagicMock()]
        results = await checker.check_sequence_consistency(states)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_check_game_consistency_async(self) -> None:
        """Test check_game_consistency is async."""
        checker = ConcreteEnsembleChecker()
        results = await checker.check_game_consistency(["e2e4", "e7e5"])
        assert len(results) >= 1

    def test_get_divergence_threshold(self) -> None:
        """Test get_divergence_threshold returns float."""
        checker = ConcreteEnsembleChecker()
        threshold = checker.get_divergence_threshold()
        assert isinstance(threshold, float)
        assert threshold > 0

    @pytest.mark.asyncio
    async def test_check_game_consistency_with_initial_fen(self) -> None:
        """Test check_game_consistency accepts initial_fen."""
        checker = ConcreteEnsembleChecker()
        results = await checker.check_game_consistency(
            ["e2e4"], initial_fen="custom_fen"
        )
        assert len(results) >= 1


class TestSubAgentVerifierProtocol:
    """Tests for SubAgentVerifierProtocol."""

    def test_protocol_has_required_methods(self) -> None:
        """Test protocol defines all required methods."""
        required_methods = [
            "verify_agent_response",
            "verify_routing_decision",
        ]
        for method in required_methods:
            assert hasattr(SubAgentVerifierProtocol, method)

    def test_concrete_implementation_satisfies_protocol(self) -> None:
        """Test concrete class satisfies the protocol."""
        verifier = ConcreteSubAgentVerifier()
        assert hasattr(verifier, "verify_agent_response")
        assert hasattr(verifier, "verify_routing_decision")

    @pytest.mark.asyncio
    async def test_verify_agent_response_async(self) -> None:
        """Test verify_agent_response is async."""
        verifier = ConcreteSubAgentVerifier()
        state = MagicMock()
        result = await verifier.verify_agent_response(state, "hrm")
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_verify_routing_decision_async(self) -> None:
        """Test verify_routing_decision is async."""
        verifier = ConcreteSubAgentVerifier()
        state = MagicMock()
        result = await verifier.verify_routing_decision(state)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_agent_response_different_agents(self) -> None:
        """Test verify_agent_response with different agent names."""
        verifier = ConcreteSubAgentVerifier()
        state = MagicMock()
        for agent_name in ["hrm", "trm", "mcts"]:
            result = await verifier.verify_agent_response(state, agent_name)
            assert result.is_valid is True


# =============================================================================
# Protocol Compatibility with AsyncMock Tests
# =============================================================================


class TestProtocolMocking:
    """Tests verifying protocols work well with unittest.mock."""

    @pytest.mark.asyncio
    async def test_async_mock_game_verifier(self) -> None:
        """Test AsyncMock can serve as ChessGameVerifierProtocol."""
        mock_verifier = AsyncMock()
        mock_verifier.verify_full_game.return_value = MagicMock(is_valid=True)
        # verify_position is sync in the protocol, use MagicMock for it
        mock_verifier.verify_position = MagicMock(return_value=MagicMock(is_valid=True))

        result = await mock_verifier.verify_full_game(["e2e4"])
        assert result.is_valid is True

        pos_result = mock_verifier.verify_position("test_fen")
        assert pos_result.is_valid is True

    @pytest.mark.asyncio
    async def test_async_mock_ensemble_checker(self) -> None:
        """Test AsyncMock can serve as EnsembleConsistencyCheckerProtocol."""
        mock_checker = AsyncMock()
        mock_checker.check_position_consistency.return_value = MagicMock(is_consistent=True)
        # get_divergence_threshold is sync in the protocol, use MagicMock for it
        mock_checker.get_divergence_threshold = MagicMock(return_value=0.3)

        result = await mock_checker.check_position_consistency(MagicMock())
        assert result.is_consistent is True

        threshold = mock_checker.get_divergence_threshold()
        assert threshold == 0.3

    @pytest.mark.asyncio
    async def test_async_mock_sub_agent_verifier(self) -> None:
        """Test AsyncMock can serve as SubAgentVerifierProtocol."""
        mock_verifier = AsyncMock()
        mock_verifier.verify_agent_response.return_value = MagicMock(is_valid=False)
        mock_verifier.verify_routing_decision.return_value = False

        result = await mock_verifier.verify_agent_response(MagicMock(), "hrm")
        assert result.is_valid is False

        routing = await mock_verifier.verify_routing_decision(MagicMock())
        assert routing is False
