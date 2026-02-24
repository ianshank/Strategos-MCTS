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

# LLM-powered chess engine (no numpy/torch dependencies)
from src.games.chess.llm_chess_engine import (  # noqa: F401 - re-exported
    ChessAnalysis,
    ChessMoveResult,
    LLMChessEngine,
    LLMChessHRMAgent,
    LLMChessMCTSAgent,
    LLMChessMetaController,
    LLMChessTRMAgent,
    RoutingDecision,
)

# Base exports always available (LLM chess engine — no numpy required)
__all__ = [
    # LLM Chess Engine
    "LLMChessEngine",
    "LLMChessHRMAgent",
    "LLMChessTRMAgent",
    "LLMChessMCTSAgent",
    "LLMChessMetaController",
    "ChessAnalysis",
    "ChessMoveResult",
    "RoutingDecision",
]

# Optional numpy-dependent imports (action space, state, config)
try:
    from src.games.chess.action_space import ChessActionEncoder  # noqa: F401
    from src.games.chess.config import (  # noqa: F401
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
    from src.games.chess.state import (  # noqa: F401
        ChessGameState,
        create_initial_state,
        create_state_from_fen,
    )

    __all__.extend([
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
        # State
        "ChessGameState",
        "create_initial_state",
        "create_state_from_fen",
    ])
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Optional torch-dependent representation imports
try:
    from src.games.chess.representation import (  # noqa: F401 - re-exported
        ChessBoardRepresentation,
        board_to_tensor,
    )
    __all__.extend([
        "ChessBoardRepresentation",
        "board_to_tensor",
    ])
except ImportError:
    pass

# Optional torch-dependent imports
try:
    from src.games.chess.continuous_learning import (  # noqa: F401 - re-exported
        ContinuousLearningConfig,
        ContinuousLearningSession,
        GameRecord,
        GameResult,
        OnlineLearner,
        ScoreCard,
        create_learning_session,
    )
    from src.games.chess.ensemble_agent import (  # noqa: F401 - re-exported
        AgentResponse,
        ChessEnsembleAgent,
        ChessStateEncoder,
        EnsembleResponse,
    )
    from src.games.chess.meta_controller import (  # noqa: F401 - re-exported
        ChessFeatureExtractor,
        ChessMetaController,
        ChessPositionFeatures,
        RoutingDecision,
    )
    from src.games.chess.training import (  # noqa: F401 - re-exported
        ChessDataAugmentation,
        ChessOpeningBook,
        ChessTrainingMetrics,
        ChessTrainingOrchestrator,
        SelfPlayGame,
        create_chess_orchestrator,
    )

    # Add torch-dependent exports
    __all__.extend([
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
        # Continuous Learning
        "ContinuousLearningConfig",
        "ContinuousLearningSession",
        "GameRecord",
        "GameResult",
        "OnlineLearner",
        "ScoreCard",
        "create_learning_session",
    ])
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional UI imports (requires gradio)
try:
    from src.games.chess.ui import (  # noqa: F401 - re-exported
        GameSession,
        create_chess_ui,
        render_board_html,
    )
    __all__.extend([
        "GameSession",
        "create_chess_ui",
        "render_board_html",
    ])
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
