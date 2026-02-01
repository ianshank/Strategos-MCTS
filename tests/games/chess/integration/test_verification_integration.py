"""
Integration Tests for Chess Verification.

Tests the integration of verification components including:
- Factory creation and component interaction
- Full game verification workflow
- Verification with different configurations
"""

from __future__ import annotations

import pytest

from src.games.chess.config import ChessConfig
from src.games.chess.state import ChessGameState
from src.games.chess.verification import (
    ChessVerificationFactory,
    VerificationBuilder,
    create_verification_factory,
)
from tests.games.chess.builders import (
    initial_position,
)


class TestVerificationFactoryIntegration:
    """Integration tests for ChessVerificationFactory."""

    @pytest.fixture
    def factory(self) -> ChessVerificationFactory:
        """Create a verification factory."""
        return create_verification_factory()

    @pytest.mark.integration
    def test_factory_creates_all_components(self, factory: ChessVerificationFactory) -> None:
        """Test that factory creates all verification components."""
        components = factory.create_all_verifiers()

        assert "move_validator" in components
        assert "game_verifier" in components
        assert "ensemble_checker" in components

        # Verify they are the right types
        assert components["move_validator"] is not None
        assert components["game_verifier"] is not None
        assert components["ensemble_checker"] is not None

    @pytest.mark.integration
    def test_factory_with_chess_config(self) -> None:
        """Test factory with custom chess configuration."""
        chess_config = ChessConfig.from_preset("small")
        factory = ChessVerificationFactory(chess_config=chess_config)

        validator = factory.create_move_validator()
        verifier = factory.create_game_verifier()

        assert validator is not None
        assert verifier is not None

    @pytest.mark.integration
    def test_factory_reuses_encoder(self, factory: ChessVerificationFactory) -> None:
        """Test that factory reuses action encoder."""
        validator1 = factory.create_move_validator(reuse_encoder=True)
        validator2 = factory.create_move_validator(reuse_encoder=True)

        # Should share the same encoder instance
        assert validator1.encoder is validator2.encoder

    @pytest.mark.integration
    def test_factory_creates_fresh_encoder(self, factory: ChessVerificationFactory) -> None:
        """Test that factory can create fresh encoders."""
        validator1 = factory.create_move_validator(reuse_encoder=False)
        validator2 = factory.create_move_validator(reuse_encoder=False)

        # Should have different encoder instances
        assert validator1.encoder is not validator2.encoder


class TestVerificationBuilderIntegration:
    """Integration tests for VerificationBuilder."""

    @pytest.mark.integration
    def test_builder_creates_move_validator(self) -> None:
        """Test builder creates move validator with configuration."""
        validator = (
            VerificationBuilder()
            .with_encoding_validation(True)
            .with_legality_validation(True)
            .with_logging(False)
            .build_move_validator()
        )

        state = initial_position()
        result = validator.validate_move(state, "e2e4")

        assert result.is_valid is True

    @pytest.mark.integration
    def test_builder_creates_game_verifier(self) -> None:
        """Test builder creates game verifier with configuration."""
        verifier = (
            VerificationBuilder()
            .with_stop_on_first_error(False)
            .with_result_verification(True)
            .build_game_verifier()
        )

        result = verifier.verify_move_sequence(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ["e2e4", "e7e5"],
        )

        assert result.is_valid is True

    @pytest.mark.integration
    def test_builder_creates_ensemble_checker(self) -> None:
        """Test builder creates ensemble checker with configuration."""
        checker = (
            VerificationBuilder()
            .with_agreement_threshold(0.7)
            .with_divergence_threshold(0.4)
            .build_ensemble_checker()
        )

        assert checker.config.agreement_threshold == 0.7
        assert checker.config.confidence_divergence_threshold == 0.4

    @pytest.mark.integration
    def test_builder_with_chess_config(self) -> None:
        """Test builder with chess configuration."""
        chess_config = ChessConfig.from_preset("small")

        verifier = (
            VerificationBuilder()
            .with_chess_config(chess_config)
            .build_game_verifier()
        )

        assert verifier is not None


class TestFullVerificationWorkflow:
    """Integration tests for complete verification workflows."""

    @pytest.fixture
    def factory(self) -> ChessVerificationFactory:
        """Create a verification factory."""
        return create_verification_factory()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_game_verification_workflow(
        self, factory: ChessVerificationFactory
    ) -> None:
        """Test complete game verification workflow."""
        # Create components
        verifier = factory.create_game_verifier()
        validator = factory.create_move_validator()

        # Verify a complete game
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]

        # First validate individual moves
        state = initial_position()
        for move in moves:
            result = validator.validate_move(state, move)
            assert result.is_valid is True
            state = state.apply_action(move)

        # Then verify the full game
        game_result = await verifier.verify_full_game(moves=moves)

        assert game_result.is_valid is True
        assert game_result.total_moves == len(moves)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_game_with_special_moves_verification(
        self, factory: ChessVerificationFactory
    ) -> None:
        """Test verification of game with special moves."""
        verifier = factory.create_game_verifier()

        # Game with castling
        moves = [
            "e2e4", "e7e5",
            "g1f3", "b8c6",
            "f1c4", "f8c5",
            "d2d3", "g8f6",
            "e1g1",  # Kingside castling
        ]

        result = await verifier.verify_full_game(moves=moves)

        assert result.is_valid is True
        assert result.move_sequence_result is not None
        assert result.move_sequence_result.castles >= 1

    @pytest.mark.integration
    def test_position_to_sequence_verification(
        self, factory: ChessVerificationFactory
    ) -> None:
        """Test verification from position through sequence."""
        verifier = factory.create_game_verifier()

        # Start from Italian Game position
        italian_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

        # Verify the position is valid
        position_result = verifier.verify_position(italian_fen)
        assert position_result.is_valid is True

        # Verify a continuation
        moves = ["f8c5", "c2c3", "g8f6"]
        sequence_result = verifier.verify_move_sequence(italian_fen, moves)

        assert sequence_result.is_valid is True
        assert sequence_result.final_fen is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(
        self, factory: ChessVerificationFactory
    ) -> None:
        """Test verification workflow with error recovery."""
        verifier = factory.create_game_verifier()

        # Game with an invalid move in the middle
        moves = ["e2e4", "e7e5", "invalid_move", "g1f3"]

        # Should detect the invalid move
        result = await verifier.verify_full_game(moves=moves)

        assert result.is_valid is False
        assert any(
            issue.code in ("INVALID_MOVE", "INVALID_UCI_FORMAT")
            for issue in result.issues
        )


class TestCrossComponentVerification:
    """Tests for verification across component boundaries."""

    @pytest.mark.integration
    def test_validator_verifier_consistency(self) -> None:
        """Test that validator and verifier produce consistent results."""
        factory = create_verification_factory()
        validator = factory.create_move_validator()
        verifier = factory.create_game_verifier()

        moves = ["e2e4", "e7e5", "g1f3"]
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        # Validate each move individually
        state = ChessGameState.from_fen(initial_fen)
        individual_valid = True
        for move in moves:
            result = validator.validate_move(state, move)
            if not result.is_valid:
                individual_valid = False
                break
            state = state.apply_action(move)

        # Verify as a sequence
        sequence_result = verifier.verify_move_sequence(initial_fen, moves)

        # Results should be consistent
        assert individual_valid == sequence_result.is_valid

    @pytest.mark.integration
    def test_encoding_verification_across_components(self) -> None:
        """Test that encoding verification is consistent."""
        factory = create_verification_factory()
        validator = factory.create_move_validator()

        state = initial_position()

        # Validate a move with encoding
        result = validator.validate_move(state, "e2e4")
        assert result.is_valid is True
        assert result.encoded_index is not None

        # Roundtrip should work
        roundtrip_result = validator.validate_encoding_roundtrip(state, "e2e4")
        assert roundtrip_result.is_valid is True
        assert roundtrip_result.encoded_index == result.encoded_index


class TestConfigurationIntegration:
    """Tests for configuration propagation."""

    @pytest.mark.integration
    def test_settings_propagation(self) -> None:
        """Test that settings are properly propagated."""
        from src.config.settings import get_settings

        settings = get_settings()
        factory = ChessVerificationFactory(settings=settings)

        components = factory.create_all_verifiers()

        # All components should be created
        assert all(v is not None for v in components.values())

    @pytest.mark.integration
    def test_log_level_configuration(self) -> None:
        """Test that log level configuration is respected."""
        verifier = (
            VerificationBuilder()
            .with_logging(True)
            .build_game_verifier()
        )

        assert verifier.config.log_verifications is True

        verifier_quiet = (
            VerificationBuilder()
            .with_logging(False)
            .build_game_verifier()
        )

        assert verifier_quiet.config.log_verifications is False
