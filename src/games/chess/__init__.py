"""
Chess module implementing AlphaZero-style learning with ensemble agents.

This module provides a complete chess implementation including:
- ChessConfig: Comprehensive configuration for all chess parameters
- ChessGameState: Game state implementation following GameState interface
- ChessActionEncoder: Move encoding/decoding for neural network action space
- ChessBoardRepresentation: 22-plane tensor representation
- ChessEnsembleAgent: HRM + TRM + MCTS ensemble with meta-controller routing
- ChessTrainingOrchestrator: Self-play training pipeline
- ContinuousLearningSession: Real-time self-play learning with scorecard tracking
- Web UI: Gradio-based interface for playing and watching AI learn

Example usage:
    ```python
    from src.games.chess import ChessConfig, ChessGameState, ChessEnsembleAgent

    # Create configuration
    config = ChessConfig.from_preset("medium")

    # Create initial state
    state = ChessGameState()

    # Create ensemble agent
    agent = ChessEnsembleAgent(config)

    # Get best move
    move = await agent.get_best_move(state)

    # Or start a continuous learning session
    from src.games.chess import create_learning_session
    session = create_learning_session(preset="small", max_minutes=30)
    scorecard = await session.run_session()
    ```
"""

from src.games.chess.action_space import ChessActionEncoder
from src.games.chess.config import (
    AgentType,
    ChessActionSpaceConfig,
    ChessBoardConfig,
    ChessConfig,
    ChessEnsembleConfig,
    ChessHRMConfig,
    ChessMCTSConfig,
    ChessNeuralNetConfig,
    ChessTrainingConfig,
    ChessTRMConfig,
    GamePhase,
    get_chess_large_config,
    get_chess_medium_config,
    get_chess_small_config,
)
from src.games.chess.continuous_learning import (
    ContinuousLearningConfig,
    ContinuousLearningSession,
    GameRecord,
    GameResult,
    OnlineLearner,
    ScoreCard,
    create_learning_session,
)
from src.games.chess.ensemble_agent import (
    AgentResponse,
    ChessEnsembleAgent,
    ChessStateEncoder,
    EnsembleResponse,
)
from src.games.chess.meta_controller import (
    ChessFeatureExtractor,
    ChessMetaController,
    ChessPositionFeatures,
    RoutingDecision,
)
from src.games.chess.representation import ChessBoardRepresentation, board_to_tensor
from src.games.chess.state import ChessGameState, create_initial_state, create_state_from_fen
from src.games.chess.training import (
    ChessDataAugmentation,
    ChessOpeningBook,
    ChessTrainingMetrics,
    ChessTrainingOrchestrator,
    SelfPlayGame,
    create_chess_orchestrator,
)
from src.games.chess.ui import (
    GameSession,
    create_chess_ui,
    render_board_html,
)

__all__ = [
    # Config
    "AgentType",
    "ChessActionSpaceConfig",
    "ChessBoardConfig",
    "ChessConfig",
    "ChessEnsembleConfig",
    "ChessHRMConfig",
    "ChessMCTSConfig",
    "ChessNeuralNetConfig",
    "ChessTrainingConfig",
    "ChessTRMConfig",
    "GamePhase",
    "get_chess_large_config",
    "get_chess_medium_config",
    "get_chess_small_config",
    # Action space
    "ChessActionEncoder",
    # Representation
    "ChessBoardRepresentation",
    "board_to_tensor",
    # State
    "ChessGameState",
    "create_initial_state",
    "create_state_from_fen",
    # Meta-controller
    "ChessFeatureExtractor",
    "ChessMetaController",
    "ChessPositionFeatures",
    "RoutingDecision",
    # Ensemble agent
    "AgentResponse",
    "ChessEnsembleAgent",
    "ChessStateEncoder",
    "EnsembleResponse",
    # Training
    "ChessDataAugmentation",
    "ChessOpeningBook",
    "ChessTrainingMetrics",
    "ChessTrainingOrchestrator",
    "SelfPlayGame",
    "create_chess_orchestrator",
    # UI
    "GameSession",
    "create_chess_ui",
    "render_board_html",
    # Continuous Learning
    "ContinuousLearningConfig",
    "ContinuousLearningSession",
    "GameRecord",
    "GameResult",
    "OnlineLearner",
    "ScoreCard",
    "create_learning_session",
]
