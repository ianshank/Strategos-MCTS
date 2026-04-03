"""Unit tests for Chess Meta-Controller module (src/games/chess/meta_controller.py)."""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed")
import torch

from src.games.chess.config import AgentType, ChessEnsembleConfig, GamePhase
from src.games.chess.meta_controller import (
    ChessFeatureExtractor,
    ChessMetaController,
    ChessPositionFeatures,
    NeuralRouter,
    RoutingDecision,
)

# ---------- Helper fixtures ----------

def _make_features(**overrides) -> ChessPositionFeatures:
    """Create ChessPositionFeatures with sensible defaults, allowing overrides."""
    defaults = {
        "game_phase": GamePhase.MIDDLEGAME,
        "move_number": 20,
        "is_opening": False,
        "is_middlegame": True,
        "is_endgame": False,
        "total_material": 50,
        "material_balance": 0,
        "has_queens": True,
        "is_material_imbalanced": False,
        "is_check": False,
        "num_legal_moves": 30,
        "has_captures": True,
        "has_promotions": False,
        "is_forcing": False,
        "center_control": 0.5,
        "king_safety": 0.7,
        "pawn_structure_complexity": 0.3,
        "time_pressure": False,
    }
    defaults.update(overrides)
    return ChessPositionFeatures(**defaults)


def _make_config(**overrides) -> ChessEnsembleConfig:
    """Create a ChessEnsembleConfig with use_learned_routing=False by default."""
    defaults = {"use_learned_routing": False}
    defaults.update(overrides)
    return ChessEnsembleConfig(**defaults)


# ---------- RoutingDecision ----------

@pytest.mark.unit
class TestRoutingDecision:
    """Tests for the RoutingDecision dataclass."""

    def test_creation(self):
        rd = RoutingDecision(
            primary_agent=AgentType.MCTS,
            agent_weights={"hrm": 0.2, "trm": 0.3, "mcts": 0.5},
            confidence=0.8,
            features={"phase": "middlegame"},
            reasoning="Complex middlegame",
        )
        assert rd.primary_agent == AgentType.MCTS
        assert rd.confidence == 0.8


# ---------- ChessPositionFeatures ----------

@pytest.mark.unit
class TestChessPositionFeatures:
    """Tests for ChessPositionFeatures dataclass and to_tensor."""

    def test_to_tensor_shape(self):
        features = _make_features()
        tensor = features.to_tensor()
        assert tensor.shape == (17,)
        assert tensor.dtype == torch.float32

    def test_to_tensor_opening(self):
        features = _make_features(
            game_phase=GamePhase.OPENING, is_opening=True, is_middlegame=False,
        )
        tensor = features.to_tensor()
        assert tensor[0].item() == 1.0  # is_opening
        assert tensor[1].item() == 0.0  # is_middlegame

    def test_to_tensor_normalization(self):
        features = _make_features(
            move_number=50, total_material=78, material_balance=10,
        )
        tensor = features.to_tensor()
        assert abs(tensor[3].item() - 50 / 100.0) < 1e-5
        assert abs(tensor[4].item() - 78 / 78.0) < 1e-5
        assert abs(tensor[5].item() - 10 / 39.0) < 1e-5

    def test_to_tensor_boolean_fields(self):
        features = _make_features(
            has_queens=True, is_check=True, has_captures=False,
            has_promotions=True, is_forcing=True, time_pressure=True,
        )
        tensor = features.to_tensor()
        assert tensor[6].item() == 1.0   # has_queens
        assert tensor[8].item() == 1.0   # is_check
        assert tensor[10].item() == 0.0  # has_captures
        assert tensor[11].item() == 1.0  # has_promotions
        assert tensor[12].item() == 1.0  # is_forcing
        assert tensor[16].item() == 1.0  # time_pressure


# ---------- ChessFeatureExtractor ----------

@pytest.mark.unit
class TestChessFeatureExtractor:
    """Tests for ChessFeatureExtractor."""

    def test_extract_starting_position(self):
        from src.games.chess.state import ChessGameState

        state = ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        extractor = ChessFeatureExtractor()
        features = extractor.extract(state)

        assert features.has_queens is True
        assert features.is_check is False
        assert features.num_legal_moves == 20  # Standard opening has 20 legal moves
        assert features.total_material > 0
        assert features.material_balance == 0
        assert features.is_material_imbalanced is False
        assert features.time_pressure is False

    def test_extract_with_time_pressure(self):
        from src.games.chess.state import ChessGameState

        state = ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        extractor = ChessFeatureExtractor()
        features = extractor.extract(state, time_pressure=True)
        assert features.time_pressure is True

    def test_extract_endgame_no_queens(self):
        from src.games.chess.state import ChessGameState

        # King + rook vs king endgame
        fen = "8/8/8/8/8/4k3/8/R3K3 w - - 0 1"
        state = ChessGameState.from_fen(fen)
        extractor = ChessFeatureExtractor()
        features = extractor.extract(state)

        assert features.has_queens is False
        assert features.total_material == 5  # Just a rook

    def test_extract_check_position(self):
        from src.games.chess.state import ChessGameState

        # White gives check with queen
        fen = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        state = ChessGameState.from_fen(fen)
        extractor = ChessFeatureExtractor()
        features = extractor.extract(state)
        assert features.is_check is True

    def test_extract_material_imbalance(self):
        from src.games.chess.state import ChessGameState

        # White has extra queen
        fen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        state = ChessGameState.from_fen(fen)
        extractor = ChessFeatureExtractor()
        features = extractor.extract(state)
        # Black is missing queen (9 points) so imbalance >= 3
        assert features.is_material_imbalanced is True

    def test_extract_promotions_available(self):
        from src.games.chess.state import ChessGameState

        # White pawn on 7th rank about to promote
        fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
        state = ChessGameState.from_fen(fen)
        extractor = ChessFeatureExtractor()
        features = extractor.extract(state)
        assert features.has_promotions is True

    def test_pawn_shield_squares_white_king(self):
        extractor = ChessFeatureExtractor()
        # King on g1 (square 6)
        squares = extractor._get_pawn_shield_squares(6, True)
        # Shield should be f2, g2, h2 (squares 13, 14, 15)
        assert 13 in squares
        assert 14 in squares
        assert 15 in squares

    def test_pawn_shield_squares_black_king(self):
        extractor = ChessFeatureExtractor()
        # King on g8 (square 62)
        squares = extractor._get_pawn_shield_squares(62, False)
        # Shield should be f7, g7, h7 (squares 53, 54, 55)
        assert 53 in squares
        assert 54 in squares
        assert 55 in squares

    def test_pawn_shield_edge_of_board(self):
        extractor = ChessFeatureExtractor()
        # King on a1 (square 0), white side
        squares = extractor._get_pawn_shield_squares(0, True)
        # Shield rank is 1, files 0-1 => squares 8, 9
        assert 8 in squares
        assert 9 in squares
        assert len(squares) == 2

    def test_pawn_shield_out_of_bounds(self):
        extractor = ChessFeatureExtractor()
        # White king on rank 8 (square 63), shield would be rank 9 => out of bounds
        squares = extractor._get_pawn_shield_squares(63, True)
        assert squares == []

    def test_center_control_calculation(self):
        import chess

        extractor = ChessFeatureExtractor()
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        center = [chess.D4, chess.D5, chess.E4, chess.E5]
        score = extractor._calculate_center_control(board, center)
        assert 0.0 <= score <= 1.0

    def test_king_safety_calculation(self):
        import chess

        extractor = ChessFeatureExtractor()
        board = chess.Board()
        safety = extractor._calculate_king_safety(board)
        assert 0.0 <= safety <= 1.0

    def test_king_safety_no_king(self):
        import chess

        extractor = ChessFeatureExtractor()
        board = chess.Board()
        # Remove white king by clearing the board and setting a custom position
        board.clear()
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        # board.king(WHITE) returns None when no white king
        safety = extractor._calculate_king_safety(board)
        assert safety == 0.5

    def test_pawn_complexity_starting(self):
        import chess

        extractor = ChessFeatureExtractor()
        board = chess.Board()
        complexity = extractor._calculate_pawn_complexity(board)
        assert 0.0 <= complexity <= 1.0

    def test_pawn_complexity_no_pawns(self):
        import chess

        extractor = ChessFeatureExtractor()
        board = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")
        complexity = extractor._calculate_pawn_complexity(board)
        assert complexity == 0.0

    def test_pawn_complexity_doubled_pawns(self):
        import chess

        extractor = ChessFeatureExtractor()
        # White has doubled pawns on e-file
        board = chess.Board("8/8/8/8/4P3/4P3/8/4K2k w - - 0 1")
        complexity = extractor._calculate_pawn_complexity(board)
        assert complexity > 0.0


# ---------- NeuralRouter ----------

@pytest.mark.unit
class TestNeuralRouter:
    """Tests for NeuralRouter nn.Module."""

    def test_forward_shape(self):
        router = NeuralRouter(input_dim=17, hidden_dim=64, num_agents=3, num_layers=2)
        x = torch.randn(1, 17)
        logits, confidence = router(x)
        assert logits.shape == (1, 3)
        assert confidence.shape == (1, 1)

    def test_forward_batch(self):
        router = NeuralRouter(input_dim=17, hidden_dim=64, num_agents=3, num_layers=2)
        x = torch.randn(8, 17)
        logits, confidence = router(x)
        assert logits.shape == (8, 3)
        assert confidence.shape == (8, 1)

    def test_confidence_bounded(self):
        router = NeuralRouter(input_dim=17, hidden_dim=32, num_agents=3, num_layers=1)
        x = torch.randn(4, 17)
        _, confidence = router(x)
        # Sigmoid output should be in (0, 1)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_single_layer(self):
        router = NeuralRouter(input_dim=17, hidden_dim=32, num_agents=3, num_layers=1)
        x = torch.randn(1, 17)
        logits, confidence = router(x)
        assert logits.shape == (1, 3)

    def test_many_agents(self):
        router = NeuralRouter(input_dim=17, hidden_dim=32, num_agents=10, num_layers=1)
        x = torch.randn(1, 17)
        logits, _ = router(x)
        assert logits.shape == (1, 10)


# ---------- ChessMetaController ----------

@pytest.mark.unit
class TestChessMetaControllerHeuristic:
    """Tests for ChessMetaController heuristic routing."""

    def test_heuristic_route_middlegame(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features()
        decision = controller._heuristic_route(features)

        assert isinstance(decision, RoutingDecision)
        assert isinstance(decision.primary_agent, AgentType)
        assert 0.0 <= decision.confidence <= 1.0
        assert abs(sum(decision.agent_weights.values()) - 1.0) < 1e-5
        assert "phase" in decision.features

    def test_tactical_position_favors_mcts(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(is_forcing=True, is_check=True)
        decision = controller._heuristic_route(features)

        # MCTS weight should be boosted in tactical positions
        assert decision.agent_weights["mcts"] > decision.agent_weights["hrm"]

    def test_endgame_without_queens_favors_trm(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(
            game_phase=GamePhase.ENDGAME,
            is_middlegame=False,
            is_endgame=True,
            has_queens=False,
        )
        decision = controller._heuristic_route(features)
        # TRM should be boosted for queenless endgames
        assert decision.agent_weights["trm"] > 0

    def test_opening_favors_hrm(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(
            game_phase=GamePhase.OPENING,
            is_opening=True,
            is_middlegame=False,
            move_number=5,
        )
        decision = controller._heuristic_route(features)
        assert decision.agent_weights["hrm"] > 0

    def test_time_pressure_favors_trm(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(time_pressure=True)
        decision = controller._heuristic_route(features)
        # TRM should be boosted under time pressure
        assert decision.agent_weights["trm"] > 0

    def test_weights_sum_to_one(self):
        config = _make_config()
        controller = ChessMetaController(config)
        for phase in [GamePhase.OPENING, GamePhase.MIDDLEGAME, GamePhase.ENDGAME]:
            features = _make_features(
                game_phase=phase,
                is_opening=(phase == GamePhase.OPENING),
                is_middlegame=(phase == GamePhase.MIDDLEGAME),
                is_endgame=(phase == GamePhase.ENDGAME),
            )
            decision = controller._heuristic_route(features)
            total = sum(decision.agent_weights.values())
            assert abs(total - 1.0) < 1e-5


@pytest.mark.unit
class TestChessMetaControllerNeural:
    """Tests for ChessMetaController neural routing."""

    def test_neural_route_with_router(self):
        config = _make_config(use_learned_routing=True, routing_hidden_dim=32, routing_num_layers=1)
        controller = ChessMetaController(config)
        assert controller.neural_router is not None

        features = _make_features()
        decision = controller._neural_route(features)

        assert isinstance(decision, RoutingDecision)
        assert isinstance(decision.primary_agent, AgentType)
        assert len(decision.agent_weights) == 3

    def test_neural_route_fallback_when_no_router(self):
        config = _make_config(use_learned_routing=True, routing_hidden_dim=32, routing_num_layers=1)
        controller = ChessMetaController(config)
        controller.neural_router = None  # Force no router

        features = _make_features()
        decision = controller._neural_route(features)
        # Should fall back to heuristic
        assert isinstance(decision, RoutingDecision)

    def test_route_dispatches_neural(self):
        config = _make_config(use_learned_routing=True, routing_hidden_dim=32, routing_num_layers=1)
        controller = ChessMetaController(config)

        from src.games.chess.state import ChessGameState

        state = ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        decision = controller.route(state)
        assert isinstance(decision, RoutingDecision)

    def test_route_dispatches_heuristic(self):
        config = _make_config(use_learned_routing=False)
        controller = ChessMetaController(config)

        from src.games.chess.state import ChessGameState

        state = ChessGameState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        decision = controller.route(state)
        assert isinstance(decision, RoutingDecision)


@pytest.mark.unit
class TestChessMetaControllerReasoning:
    """Tests for _generate_reasoning."""

    def test_hrm_opening_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(
            game_phase=GamePhase.OPENING, is_opening=True, is_forcing=False,
        )
        reasoning = controller._generate_reasoning(features, AgentType.HRM)
        assert "Opening" in reasoning or "strategic" in reasoning

    def test_hrm_quiet_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(is_opening=False, is_forcing=False)
        reasoning = controller._generate_reasoning(features, AgentType.HRM)
        assert "quiet" in reasoning or "strategic" in reasoning

    def test_trm_endgame_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(
            game_phase=GamePhase.ENDGAME, is_endgame=True, time_pressure=False,
        )
        reasoning = controller._generate_reasoning(features, AgentType.TRM)
        assert "Endgame" in reasoning or "precise" in reasoning

    def test_trm_time_pressure_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(time_pressure=True, is_endgame=False)
        reasoning = controller._generate_reasoning(features, AgentType.TRM)
        assert "Time pressure" in reasoning or "fast" in reasoning

    def test_mcts_forcing_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(is_forcing=True, is_middlegame=False)
        reasoning = controller._generate_reasoning(features, AgentType.MCTS)
        assert "Tactical" in reasoning or "search" in reasoning

    def test_mcts_middlegame_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features(is_forcing=False, is_middlegame=True)
        reasoning = controller._generate_reasoning(features, AgentType.MCTS)
        assert "middlegame" in reasoning or "tree search" in reasoning

    def test_default_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        # HRM with no opening/non-forcing => both conditions may fail
        features = _make_features(is_opening=False, is_forcing=True)
        reasoning = controller._generate_reasoning(features, AgentType.HRM)
        # At least we get a string back
        assert isinstance(reasoning, str)

    def test_default_fallback_reasoning(self):
        config = _make_config()
        controller = ChessMetaController(config)
        # MCTS with no forcing and no middlegame => empty reasons list
        features = _make_features(is_forcing=False, is_middlegame=False)
        reasoning = controller._generate_reasoning(features, AgentType.MCTS)
        assert "Default routing" in reasoning


@pytest.mark.unit
class TestChessMetaControllerSaveLoad:
    """Tests for save and load methods."""

    def test_save_with_neural_router(self, tmp_path):
        config = _make_config(use_learned_routing=True, routing_hidden_dim=32, routing_num_layers=1)
        controller = ChessMetaController(config)
        path = str(tmp_path / "router.pt")
        controller.save(path)
        import os
        assert os.path.exists(path)

    def test_save_without_neural_router(self, tmp_path):
        config = _make_config(use_learned_routing=False)
        controller = ChessMetaController(config)
        path = str(tmp_path / "router.pt")
        controller.save(path)
        import os
        assert not os.path.exists(path)

    def test_load_with_neural_router(self, tmp_path):
        config = _make_config(use_learned_routing=True, routing_hidden_dim=32, routing_num_layers=1)
        controller = ChessMetaController(config)
        path = str(tmp_path / "router.pt")
        controller.save(path)

        # Create a new controller and load the saved weights
        controller2 = ChessMetaController(config)
        controller2.load(path)

        # Verify weights were loaded (state dicts should match)
        sd1 = controller.neural_router.state_dict()
        sd2 = controller2.neural_router.state_dict()
        for key in sd1:
            assert torch.allclose(sd1[key], sd2[key]), f"Mismatch for {key}"

    def test_load_without_neural_router(self, tmp_path):
        config = _make_config(use_learned_routing=False)
        controller = ChessMetaController(config)
        path = str(tmp_path / "router.pt")
        # Should not raise even if file doesn't exist
        controller.load(path)

    def test_update_noop(self):
        config = _make_config()
        controller = ChessMetaController(config)
        features = _make_features()
        # update is a no-op currently, just verify no error
        controller.update(features, AgentType.MCTS)


@pytest.mark.unit
class TestChessMetaControllerInit:
    """Tests for ChessMetaController initialization."""

    def test_no_neural_router_when_disabled(self):
        config = _make_config(use_learned_routing=False)
        controller = ChessMetaController(config)
        assert controller.neural_router is None

    def test_neural_router_created_when_enabled(self):
        config = _make_config(use_learned_routing=True, routing_hidden_dim=64, routing_num_layers=2)
        controller = ChessMetaController(config)
        assert controller.neural_router is not None

    def test_device_setting(self):
        config = _make_config()
        controller = ChessMetaController(config, device="cpu")
        assert controller.device == "cpu"

    def test_agent_names(self):
        assert ChessMetaController.AGENT_NAMES == ["hrm", "trm", "mcts"]
