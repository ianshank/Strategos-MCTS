"""Integration tests for chess ensemble and training pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.games.chess.config import (
    AgentType,
    ChessConfig,
    get_chess_small_config,
)
from src.games.chess.state import ChessGameState


class TestMetaControllerIntegration:
    """Integration tests for meta-controller with chess positions."""

    @pytest.fixture
    def config(self) -> ChessConfig:
        """Create small config for fast tests."""
        return get_chess_small_config()

    def test_meta_controller_routing_opening(self, config: ChessConfig) -> None:
        """Test meta-controller routes correctly in opening."""
        from src.games.chess.meta_controller import ChessMetaController

        controller = ChessMetaController(config.ensemble, device="cpu")
        state = ChessGameState.initial()

        decision = controller.route(state)

        assert decision.primary_agent in [AgentType.HRM, AgentType.TRM, AgentType.MCTS]
        assert 0 <= decision.confidence <= 1
        assert "phase" in decision.features

    def test_meta_controller_routing_endgame(self, config: ChessConfig) -> None:
        """Test meta-controller routes correctly in endgame."""
        from src.games.chess.meta_controller import ChessMetaController

        controller = ChessMetaController(config.ensemble, device="cpu")
        # King and pawn endgame
        fen = "8/8/4k3/8/4P3/8/8/4K3 w - - 0 50"
        state = ChessGameState.from_fen(fen)

        decision = controller.route(state)

        assert decision.primary_agent in [AgentType.HRM, AgentType.TRM, AgentType.MCTS]
        # Endgame should have different routing than opening

    def test_meta_controller_time_pressure(self, config: ChessConfig) -> None:
        """Test meta-controller handles time pressure."""
        from src.games.chess.meta_controller import ChessMetaController

        controller = ChessMetaController(config.ensemble, device="cpu")
        state = ChessGameState.initial()

        controller.route(state, time_pressure=False)
        decision_pressure = controller.route(state, time_pressure=True)

        # Time pressure should affect routing weights
        # TRM typically preferred under time pressure
        assert decision_pressure.agent_weights is not None


class TestFeatureExtractorIntegration:
    """Integration tests for feature extraction."""

    def test_feature_extraction_initial_position(self) -> None:
        """Test feature extraction for initial position."""
        from src.games.chess.meta_controller import ChessFeatureExtractor

        extractor = ChessFeatureExtractor()
        state = ChessGameState.initial()

        features = extractor.extract(state)

        assert features.is_opening is True
        assert features.move_number == 1
        assert features.has_queens is True
        assert features.is_check is False
        assert features.num_legal_moves == 20

    def test_feature_extraction_to_tensor(self) -> None:
        """Test feature conversion to tensor."""
        from src.games.chess.meta_controller import ChessFeatureExtractor

        extractor = ChessFeatureExtractor()
        state = ChessGameState.initial()

        features = extractor.extract(state)
        tensor = features.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32


class TestBoardRepresentationIntegration:
    """Integration tests for board representation."""

    def test_encode_decode_consistency(self) -> None:
        """Test that encoding preserves piece information."""
        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        state = ChessGameState.initial()

        tensor = rep.encode(state.board)

        # Check white pieces on first rank
        # King on e1 (file 4, rank 0) should be in plane 5 (white king)
        assert tensor[5, 0, 4] == 1.0  # White king

        # Check black pieces on eighth rank
        # King on e8 (file 4, rank 7) should be in plane 11 (black king)
        assert tensor[11, 7, 4] == 1.0  # Black king

    def test_encode_from_black_perspective(self) -> None:
        """Test encoding from black's perspective."""
        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        state = ChessGameState.initial()

        tensor_white = rep.encode(state.board, from_perspective=1)
        tensor_black = rep.encode(state.board, from_perspective=-1)

        # Tensors should be different (board flipped)
        assert not torch.allclose(tensor_white, tensor_black)


class TestActionEncoderIntegration:
    """Integration tests for action encoder with real positions."""

    def test_encode_all_legal_moves(self) -> None:
        """Test encoding all legal moves from initial position."""
        from src.games.chess.action_space import ChessActionEncoder

        encoder = ChessActionEncoder()
        state = ChessGameState.initial()

        legal_moves = state.get_legal_actions()
        encoded_indices = []

        for move in legal_moves:
            idx = encoder.encode_move(move)
            encoded_indices.append(idx)

        # All indices should be unique
        assert len(set(encoded_indices)) == len(encoded_indices)
        # All should be valid
        assert all(0 <= idx < encoder.action_size for idx in encoded_indices)

    def test_encode_decode_game_sequence(self) -> None:
        """Test encoding/decoding through a game sequence."""
        from src.games.chess.action_space import ChessActionEncoder

        encoder = ChessActionEncoder()
        state = ChessGameState.initial()

        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]

        for move in moves:
            # Encode and decode
            idx = encoder.encode_move(move, from_black_perspective=state.current_player == -1)
            decoded = encoder.decode_move(idx, from_black_perspective=state.current_player == -1)
            assert decoded == move

            state = state.apply_action(move)


class TestEnsembleAgentIntegration:
    """Integration tests for ensemble agent (requires mocking heavy components)."""

    @pytest.fixture
    def config(self) -> ChessConfig:
        """Create small config for fast tests."""
        config = get_chess_small_config()
        config.ensemble.use_learned_routing = False  # Disable neural routing for tests
        return config

    @pytest.mark.asyncio
    async def test_ensemble_agent_initialization(self, config: ChessConfig) -> None:
        """Test ensemble agent can be initialized."""
        # This is a lightweight test - full tests would need mocked networks
        from src.games.chess.ensemble_agent import ChessEnsembleAgent

        # Just test that we can create the agent without errors
        # The actual networks won't be loaded until accessed
        agent = ChessEnsembleAgent(config)
        assert agent.config == config
        assert agent.meta_controller is not None


class TestSelfPlayGameIntegration:
    """Integration tests for self-play game generation."""

    def test_selfplay_game_dataclass(self) -> None:
        """Test SelfPlayGame dataclass."""
        from src.games.chess.training import SelfPlayGame

        game = SelfPlayGame(
            positions=[torch.zeros(22, 8, 8)],
            policies=[np.zeros(4672)],
            values=[0.5],
            outcome=1.0,
            moves=["e2e4"],
            game_length=1,
        )

        assert game.game_length == 1
        assert game.outcome == 1.0


class TestDataAugmentationIntegration:
    """Integration tests for data augmentation."""

    @pytest.fixture
    def config(self) -> ChessConfig:
        """Create config with augmentation enabled."""
        config = get_chess_small_config()
        config.training.use_board_flip = True
        return config

    def test_augmentation_produces_multiple_samples(self, config: ChessConfig) -> None:
        """Test that augmentation produces multiple samples."""
        from src.games.chess.training import ChessDataAugmentation

        aug = ChessDataAugmentation(config)
        state_tensor = torch.randn(22, 8, 8)
        policy = np.random.randn(4672)

        augmented = aug.augment(state_tensor, policy)

        # Should produce original + flipped
        assert len(augmented) == 2

    def test_augmentation_disabled(self) -> None:
        """Test augmentation can be disabled."""
        from src.games.chess.training import ChessDataAugmentation

        config = get_chess_small_config()
        config.training.use_board_flip = False

        aug = ChessDataAugmentation(config)
        state_tensor = torch.randn(22, 8, 8)
        policy = np.random.randn(4672)

        augmented = aug.augment(state_tensor, policy)

        # Should only produce original
        assert len(augmented) == 1


class TestTrainingMetrics:
    """Tests for training metrics dataclass."""

    def test_training_metrics_creation(self) -> None:
        """Test creating training metrics."""
        from src.games.chess.training import ChessTrainingMetrics

        metrics = ChessTrainingMetrics(
            iteration=0,
            policy_loss=0.5,
            value_loss=0.3,
            total_loss=0.8,
            games_played=100,
            average_game_length=50.0,
            win_rate_white=0.45,
            win_rate_black=0.35,
            draw_rate=0.20,
            average_value_accuracy=0.7,
            learning_rate=0.01,
            elapsed_time_seconds=120.0,
        )

        assert metrics.iteration == 0
        assert metrics.total_loss == 0.8
        assert metrics.games_played == 100


class TestOpeningBook:
    """Tests for opening book functionality."""

    def test_opening_book_initialization(self) -> None:
        """Test opening book can be initialized."""
        from src.games.chess.training import ChessOpeningBook

        book = ChessOpeningBook()
        book.load()

        # Should have loaded builtin openings
        assert book._loaded is True

    def test_opening_book_get_move(self) -> None:
        """Test getting a book move."""
        from src.games.chess.training import ChessOpeningBook

        book = ChessOpeningBook()
        book.load()
        state = ChessGameState.initial()

        move = book.get_book_move(state)

        # Should return a common opening move
        if move is not None:
            assert move in ["e2e4", "d2d4", "c2c4", "g1f3"]

    def test_opening_book_out_of_book(self) -> None:
        """Test opening book returns None for out-of-book position."""
        from src.games.chess.training import ChessOpeningBook

        book = ChessOpeningBook()
        book.load()

        # Random position not in book
        fen = "8/8/4k3/8/4P3/8/8/4K3 w - - 0 50"
        state = ChessGameState.from_fen(fen)

        move = book.get_book_move(state)
        assert move is None


class TestConfigSystemConversion:
    """Tests for config conversion to system config."""

    def test_chess_config_to_system_config(self) -> None:
        """Test converting ChessConfig to SystemConfig."""
        from src.training.system_config import SystemConfig

        chess_config = get_chess_small_config()
        system_config = chess_config.to_system_config()

        assert isinstance(system_config, SystemConfig)
        assert system_config.mcts.num_simulations == chess_config.mcts.num_simulations
        assert system_config.neural_net.input_channels == chess_config.input_channels
        assert system_config.neural_net.action_size == chess_config.action_size
