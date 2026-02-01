"""Unit tests for chess configuration module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.games.chess.config import (
    AgentType,
    ChessActionSpaceConfig,
    ChessBoardConfig,
    ChessConfig,
    ChessEnsembleConfig,
    GamePhase,
    get_chess_large_config,
    get_chess_medium_config,
    get_chess_small_config,
)


class TestChessBoardConfig:
    """Tests for ChessBoardConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ChessBoardConfig()
        assert config.board_size == 8
        assert config.num_squares == 64
        assert config.num_piece_types == 6
        assert config.num_colors == 2
        assert config.piece_planes == 12

    def test_total_planes_without_history(self) -> None:
        """Test total planes calculation without history."""
        config = ChessBoardConfig(include_history=False)
        # 12 piece + 1 side + 4 castling + 1 ep + 1 halfmove + 1 fullmove + 2 repetition = 22
        assert config.total_planes == 22

    def test_total_planes_with_history(self) -> None:
        """Test total planes calculation with history."""
        config = ChessBoardConfig(include_history=True, history_length=8)
        base_planes = 22
        history_planes = 8 * 12  # 8 positions * 12 piece planes
        assert config.total_planes == base_planes + history_planes

    def test_input_shape(self) -> None:
        """Test input shape property."""
        config = ChessBoardConfig()
        shape = config.input_shape
        assert shape == (config.total_planes, 8, 8)


class TestChessActionSpaceConfig:
    """Tests for ChessActionSpaceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ChessActionSpaceConfig()
        assert config.queen_move_directions == 8
        assert config.queen_move_distances == 7
        assert config.knight_move_types == 8

    def test_total_actions(self) -> None:
        """Test total action space size."""
        config = ChessActionSpaceConfig()
        # 73 planes * 64 squares = 4672
        assert config.total_actions == 73 * 64

    def test_queen_moves(self) -> None:
        """Test queen moves calculation."""
        config = ChessActionSpaceConfig()
        assert config.queen_moves == 8 * 7  # 56


class TestChessEnsembleConfig:
    """Tests for ChessEnsembleConfig."""

    def test_default_weights(self) -> None:
        """Test default ensemble weights."""
        config = ChessEnsembleConfig()
        total = config.hrm_weight + config.trm_weight + config.mcts_weight
        assert abs(total - 1.0) < 0.01

    def test_get_phase_weights_opening(self) -> None:
        """Test phase weights for opening."""
        config = ChessEnsembleConfig()
        weights = config.get_phase_weights(GamePhase.OPENING)
        assert "hrm" in weights
        assert "trm" in weights
        assert "mcts" in weights
        assert weights["hrm"] == config.opening_hrm_preference

    def test_get_phase_weights_middlegame(self) -> None:
        """Test phase weights for middlegame."""
        config = ChessEnsembleConfig()
        weights = config.get_phase_weights(GamePhase.MIDDLEGAME)
        assert weights["mcts"] == config.middlegame_mcts_preference

    def test_get_phase_weights_endgame(self) -> None:
        """Test phase weights for endgame."""
        config = ChessEnsembleConfig()
        weights = config.get_phase_weights(GamePhase.ENDGAME)
        assert weights["trm"] == config.endgame_trm_preference


class TestChessConfig:
    """Tests for ChessConfig."""

    def test_default_initialization(self) -> None:
        """Test default configuration initialization."""
        config = ChessConfig()
        assert config.board is not None
        assert config.action_space is not None
        assert config.ensemble is not None
        assert config.mcts is not None
        assert config.neural_net is not None
        assert config.training is not None
        assert config.hrm is not None
        assert config.trm is not None

    def test_input_channels_property(self) -> None:
        """Test input_channels property."""
        config = ChessConfig()
        assert config.input_channels == config.board.total_planes

    def test_action_size_property(self) -> None:
        """Test action_size property."""
        config = ChessConfig()
        assert config.action_size == config.action_space.total_actions

    def test_to_dict(self) -> None:
        """Test configuration serialization to dict."""
        config = ChessConfig()
        config_dict = config.to_dict()
        assert "board" in config_dict
        assert "action_space" in config_dict
        assert "ensemble" in config_dict
        assert "device" in config_dict

    def test_from_dict(self) -> None:
        """Test configuration deserialization from dict."""
        original = ChessConfig()
        original.mcts.num_simulations = 500
        config_dict = original.to_dict()
        restored = ChessConfig.from_dict(config_dict)
        assert restored.mcts.num_simulations == 500

    def test_save_and_load(self) -> None:
        """Test configuration save and load."""
        config = ChessConfig()
        config.mcts.num_simulations = 999

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)
            loaded = ChessConfig.load(path)
            assert loaded.mcts.num_simulations == 999

    def test_from_preset_small(self) -> None:
        """Test small preset configuration."""
        config = ChessConfig.from_preset("small")
        assert config.mcts.num_simulations == 100
        assert config.neural_net.num_res_blocks == 9

    def test_from_preset_medium(self) -> None:
        """Test medium preset configuration."""
        config = ChessConfig.from_preset("medium")
        assert config.mcts.num_simulations == 400
        assert config.neural_net.num_res_blocks == 19

    def test_from_preset_large(self) -> None:
        """Test large preset configuration."""
        config = ChessConfig.from_preset("large")
        assert config.mcts.num_simulations == 1600
        assert config.neural_net.num_res_blocks == 39

    def test_from_preset_invalid(self) -> None:
        """Test invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            ChessConfig.from_preset("invalid")

    def test_env_override(self) -> None:
        """Test environment variable overrides."""
        os.environ["CHESS_MCTS_SIMULATIONS"] = "1234"
        os.environ["CHESS_LEARNING_RATE"] = "0.05"

        try:
            config = ChessConfig()
            assert config.mcts.num_simulations == 1234
            assert config.training.learning_rate == 0.05
        finally:
            del os.environ["CHESS_MCTS_SIMULATIONS"]
            del os.environ["CHESS_LEARNING_RATE"]

    def test_to_system_config(self) -> None:
        """Test conversion to SystemConfig."""
        config = ChessConfig()
        system_config = config.to_system_config()
        assert system_config.mcts.num_simulations == config.mcts.num_simulations
        assert system_config.neural_net.input_channels == config.input_channels
        assert system_config.neural_net.action_size == config.action_size


class TestPresetConfigs:
    """Tests for preset configuration functions."""

    def test_small_config(self) -> None:
        """Test get_chess_small_config."""
        config = get_chess_small_config()
        assert config.hrm.h_dim == 256
        assert config.training.games_per_iteration == 500

    def test_medium_config(self) -> None:
        """Test get_chess_medium_config."""
        config = get_chess_medium_config()
        assert config.neural_net.num_channels == 256
        assert config.training.games_per_iteration == 2500

    def test_large_config(self) -> None:
        """Test get_chess_large_config."""
        config = get_chess_large_config()
        assert config.hrm.h_dim == 768
        assert config.training.games_per_iteration == 25_000
        assert config.use_mixed_precision is True


class TestEnums:
    """Tests for configuration enums."""

    def test_game_phase_values(self) -> None:
        """Test GamePhase enum values."""
        assert GamePhase.OPENING.value == "opening"
        assert GamePhase.MIDDLEGAME.value == "middlegame"
        assert GamePhase.ENDGAME.value == "endgame"

    def test_agent_type_values(self) -> None:
        """Test AgentType enum values."""
        assert AgentType.HRM.value == "hrm"
        assert AgentType.TRM.value == "trm"
        assert AgentType.MCTS.value == "mcts"
        assert AgentType.ENSEMBLE.value == "ensemble"
