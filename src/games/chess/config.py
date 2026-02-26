"""
Chess Configuration Module.

Provides comprehensive, validated configuration for AlphaZero-style chess training
with ensemble agents. All parameters are configurable via environment variables,
YAML files, or programmatic configuration.

No hard-coded values - everything is configurable through this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.training.system_config import SystemConfig

# Optional torch import for device configuration
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class GamePhase(Enum):
    """Chess game phases for meta-controller routing."""

    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


class AgentType(Enum):
    """Available agent types for ensemble."""

    HRM = "hrm"
    TRM = "trm"
    MCTS = "mcts"
    ENSEMBLE = "ensemble"


@dataclass
class ChessBoardConfig:
    """Configuration for chess board representation.

    Defines the tensor representation format for neural network input.
    Based on AlphaZero's 19-plane representation with enhancements.
    """

    # Board dimensions
    board_size: int = 8
    num_squares: int = 64

    # Piece plane configuration
    num_piece_types: int = 6  # Pawn, Knight, Bishop, Rook, Queen, King
    num_colors: int = 2  # White, Black
    piece_planes: int = 12  # 6 piece types * 2 colors

    # Game state planes
    side_to_move_planes: int = 1
    castling_planes: int = 4  # KQkq
    en_passant_planes: int = 1
    halfmove_clock_planes: int = 1
    fullmove_planes: int = 1
    repetition_planes: int = 2  # 1-fold, 2-fold repetition

    # History planes (optional, for temporal information)
    history_length: int = 8  # Number of past positions to include
    include_history: bool = False  # Whether to include position history

    @property
    def total_planes(self) -> int:
        """Calculate total number of input planes."""
        base_planes = (
            self.piece_planes
            + self.side_to_move_planes
            + self.castling_planes
            + self.en_passant_planes
            + self.halfmove_clock_planes
            + self.fullmove_planes
            + self.repetition_planes
        )
        if self.include_history:
            # Each historical position adds piece planes
            return base_planes + (self.history_length * self.piece_planes)
        return base_planes

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return expected input tensor shape (C, H, W)."""
        return (self.total_planes, self.board_size, self.board_size)


@dataclass
class ChessActionSpaceConfig:
    """Configuration for chess action space encoding.

    Uses AlphaZero-style action encoding:
    - 56 queen moves (8 directions * 7 distances)
    - 8 knight moves
    - 9 underpromotions (3 piece types * 3 directions: left, straight, right)
    Total: 73 move types per square = 73 * 64 = 4672 actions
    """

    # Move type counts
    queen_move_directions: int = 8  # N, NE, E, SE, S, SW, W, NW
    queen_move_distances: int = 7  # 1-7 squares
    knight_move_types: int = 8  # 8 possible knight moves

    # Promotion configuration
    promotion_piece_types: int = 3  # Knight, Bishop, Rook (Queen is default)
    promotion_directions: int = 3  # Left capture, straight, right capture

    # Action space dimensions
    move_planes: int = 73  # 56 queen + 8 knight + 9 underpromotions

    @property
    def total_actions(self) -> int:
        """Calculate total action space size."""
        return self.move_planes * 64  # 73 * 64 = 4672

    @property
    def queen_moves(self) -> int:
        """Number of queen-like move types."""
        return self.queen_move_directions * self.queen_move_distances


@dataclass
class ChessEnsembleConfig:
    """Configuration for ensemble agent combining HRM, TRM, and MCTS.

    Defines weights and routing behavior for the meta-controller.
    """

    # Agent weights for ensemble voting
    hrm_weight: float = 0.3
    trm_weight: float = 0.3
    mcts_weight: float = 0.4

    # Phase-specific agent preferences (0-1 scale)
    opening_hrm_preference: float = 0.6
    opening_trm_preference: float = 0.2
    opening_mcts_preference: float = 0.2

    middlegame_hrm_preference: float = 0.2
    middlegame_trm_preference: float = 0.3
    middlegame_mcts_preference: float = 0.5

    endgame_hrm_preference: float = 0.1
    endgame_trm_preference: float = 0.5
    endgame_mcts_preference: float = 0.4

    # Confidence thresholds for routing
    confidence_threshold: float = 0.8
    fallback_to_mcts: bool = True

    # Ensemble combination method
    combination_method: str = "weighted_vote"  # "weighted_vote", "max_confidence", "bayesian"

    # Meta-controller neural network
    use_learned_routing: bool = True
    routing_hidden_dim: int = 256
    routing_num_layers: int = 2

    def get_phase_weights(self, phase: GamePhase) -> dict[str, float]:
        """Get agent weights for a specific game phase."""
        if phase == GamePhase.OPENING:
            return {
                "hrm": self.opening_hrm_preference,
                "trm": self.opening_trm_preference,
                "mcts": self.opening_mcts_preference,
            }
        elif phase == GamePhase.MIDDLEGAME:
            return {
                "hrm": self.middlegame_hrm_preference,
                "trm": self.middlegame_trm_preference,
                "mcts": self.middlegame_mcts_preference,
            }
        else:  # ENDGAME
            return {
                "hrm": self.endgame_hrm_preference,
                "trm": self.endgame_trm_preference,
                "mcts": self.endgame_mcts_preference,
            }


@dataclass
class ChessMCTSConfig:
    """MCTS configuration specific to chess."""

    # Search parameters
    num_simulations: int = 800
    c_puct: float = 1.25

    # Dirichlet noise for exploration
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.3  # Chess-specific (lower than Go)

    # Temperature schedule
    temperature_threshold: int = 30  # Move number to switch to greedy
    temperature_init: float = 1.0
    temperature_final: float = 0.1

    # Virtual loss for parallel search
    virtual_loss: float = 3.0
    num_parallel: int = 8

    # Progressive widening (for large action spaces)
    use_progressive_widening: bool = True
    pw_k: float = 1.0
    pw_alpha: float = 0.5

    # Time management
    time_per_move_ms: int = 1000  # Milliseconds per move (when time-limited)
    use_time_management: bool = False


@dataclass
class ChessNeuralNetConfig:
    """Neural network configuration for chess."""

    # ResNet architecture
    num_res_blocks: int = 19
    num_channels: int = 256

    # Policy head
    policy_conv_channels: int = 32
    policy_fc_dim: int = 256

    # Value head
    value_conv_channels: int = 32
    value_fc_hidden: int = 256

    # Regularization
    use_batch_norm: bool = True
    dropout: float = 0.0
    weight_decay: float = 1e-4

    # SE (Squeeze-and-Excitation) blocks
    use_se_blocks: bool = True
    se_ratio: int = 4


@dataclass
class ChessTrainingConfig:
    """Training pipeline configuration for chess."""

    # Self-play generation
    games_per_iteration: int = 5000
    num_actors: int = 64

    # Experience replay
    buffer_size: int = 500_000
    min_buffer_size: int = 10_000  # Minimum before training starts
    batch_size: int = 2048

    # Optimization
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Learning rate schedule
    lr_schedule: str = "cosine"
    lr_decay_steps: int = 100
    lr_decay_gamma: float = 0.1
    warmup_steps: int = 1000

    # Training loop
    epochs_per_iteration: int = 1
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0

    # Checkpointing
    checkpoint_interval: int = 10
    keep_last_n_checkpoints: int = 5

    # Evaluation
    evaluation_games: int = 400
    win_rate_threshold: float = 0.55

    # Early stopping
    patience: int = 20
    min_delta: float = 0.01

    # Data augmentation
    use_board_flip: bool = True  # Flip board for color symmetry
    use_random_opening: bool = True  # Start from random opening positions

    # Opening book
    use_opening_book: bool = False
    opening_book_path: str = ""
    opening_book_moves: int = 10  # Number of moves from book

    # External evaluation
    evaluate_vs_stockfish: bool = False
    stockfish_path: str = ""
    stockfish_elo: int = 1500


@dataclass
class ChessHRMConfig:
    """HRM agent configuration for chess strategic planning."""

    h_dim: int = 512
    l_dim: int = 256
    num_h_layers: int = 2
    num_l_layers: int = 4
    max_outer_steps: int = 10
    halt_threshold: float = 0.95
    use_augmentation: bool = True
    dropout: float = 0.1
    ponder_epsilon: float = 0.01
    max_ponder_steps: int = 16

    # Chess-specific
    use_positional_encoding: bool = True
    position_embedding_dim: int = 64


@dataclass
class ChessTRMConfig:
    """TRM agent configuration for chess tactical refinement."""

    latent_dim: int = 256
    num_recursions: int = 16
    hidden_dim: int = 512
    deep_supervision: bool = True
    supervision_weight_decay: float = 0.5
    convergence_threshold: float = 0.01
    min_recursions: int = 3
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Chess-specific
    tactical_attention_heads: int = 8
    use_piece_attention: bool = True


@dataclass
class ChessConfig:
    """Master configuration for AlphaZero-style chess training.

    Aggregates all component configurations with sensible defaults
    and provides factory methods for different training scales.
    """

    # Component configurations
    board: ChessBoardConfig = field(default_factory=ChessBoardConfig)
    action_space: ChessActionSpaceConfig = field(default_factory=ChessActionSpaceConfig)
    ensemble: ChessEnsembleConfig = field(default_factory=ChessEnsembleConfig)
    mcts: ChessMCTSConfig = field(default_factory=ChessMCTSConfig)
    neural_net: ChessNeuralNetConfig = field(default_factory=ChessNeuralNetConfig)
    training: ChessTrainingConfig = field(default_factory=ChessTrainingConfig)
    hrm: ChessHRMConfig = field(default_factory=ChessHRMConfig)
    trm: ChessTRMConfig = field(default_factory=ChessTRMConfig)

    # System settings
    device: str = field(default_factory=lambda: "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
    seed: int = 42
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"

    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "chess-alphazero"
    wandb_entity: str | None = None

    # Paths
    checkpoint_dir: str = "./checkpoints/chess"
    data_dir: str = "./data/chess"
    log_dir: str = "./logs/chess"

    # Environment variable prefix
    ENV_PREFIX: ClassVar[str] = "CHESS_"

    def __post_init__(self) -> None:
        """Validate configuration and apply environment overrides."""
        self._apply_env_overrides()
        self._validate()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # System settings
        if env_device := os.getenv(f"{self.ENV_PREFIX}DEVICE"):
            self.device = env_device
        if env_seed := os.getenv(f"{self.ENV_PREFIX}SEED"):
            self.seed = int(env_seed)

        # MCTS settings
        if env_sims := os.getenv(f"{self.ENV_PREFIX}MCTS_SIMULATIONS"):
            self.mcts.num_simulations = int(env_sims)
        if env_cpuct := os.getenv(f"{self.ENV_PREFIX}MCTS_CPUCT"):
            self.mcts.c_puct = float(env_cpuct)

        # Training settings
        if env_lr := os.getenv(f"{self.ENV_PREFIX}LEARNING_RATE"):
            self.training.learning_rate = float(env_lr)
        if env_batch := os.getenv(f"{self.ENV_PREFIX}BATCH_SIZE"):
            self.training.batch_size = int(env_batch)
        if env_games := os.getenv(f"{self.ENV_PREFIX}GAMES_PER_ITERATION"):
            self.training.games_per_iteration = int(env_games)

        # Paths
        if env_checkpoint := os.getenv(f"{self.ENV_PREFIX}CHECKPOINT_DIR"):
            self.checkpoint_dir = env_checkpoint
        if env_data := os.getenv(f"{self.ENV_PREFIX}DATA_DIR"):
            self.data_dir = env_data

        # Wandb
        if env_wandb := os.getenv(f"{self.ENV_PREFIX}USE_WANDB"):
            self.use_wandb = env_wandb.lower() in ("true", "1", "yes")
        if env_wandb_project := os.getenv(f"{self.ENV_PREFIX}WANDB_PROJECT"):
            self.wandb_project = env_wandb_project

    def _validate(self) -> None:
        """Validate configuration values."""
        # Device validation - check torch availability first
        cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        if self.device.startswith("cuda") and not cuda_available:
            self.device = "cpu"
            self.use_mixed_precision = False
            self.distributed = False

        # Ensure positive values
        assert self.mcts.num_simulations > 0, "MCTS simulations must be positive"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.learning_rate > 0, "Learning rate must be positive"

        # Validate weights sum approximately to 1
        weight_sum = self.ensemble.hrm_weight + self.ensemble.trm_weight + self.ensemble.mcts_weight
        assert 0.99 <= weight_sum <= 1.01, f"Ensemble weights must sum to 1, got {weight_sum}"

    @property
    def input_channels(self) -> int:
        """Get total input channels for neural network."""
        return self.board.total_planes

    @property
    def action_size(self) -> int:
        """Get total action space size."""
        return self.action_space.total_actions

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "board": self.board.__dict__,
            "action_space": self.action_space.__dict__,
            "ensemble": self.ensemble.__dict__,
            "mcts": self.mcts.__dict__,
            "neural_net": self.neural_net.__dict__,
            "training": self.training.__dict__,
            "hrm": self.hrm.__dict__,
            "trm": self.trm.__dict__,
            "device": self.device,
            "seed": self.seed,
            "use_mixed_precision": self.use_mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "compile_model": self.compile_model,
            "distributed": self.distributed,
            "checkpoint_dir": self.checkpoint_dir,
            "data_dir": self.data_dir,
            "log_dir": self.log_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ChessConfig:
        """Create configuration from dictionary."""
        config = cls()

        # Update nested configs
        nested_configs = [
            ("board", config.board, ChessBoardConfig),
            ("action_space", config.action_space, ChessActionSpaceConfig),
            ("ensemble", config.ensemble, ChessEnsembleConfig),
            ("mcts", config.mcts, ChessMCTSConfig),
            ("neural_net", config.neural_net, ChessNeuralNetConfig),
            ("training", config.training, ChessTrainingConfig),
            ("hrm", config.hrm, ChessHRMConfig),
            ("trm", config.trm, ChessTRMConfig),
        ]

        for key, obj, _ in nested_configs:
            if key in config_dict:
                for attr, value in config_dict[key].items():
                    if hasattr(obj, attr):
                        setattr(obj, attr, value)

        # Update system settings
        system_keys = [
            "device",
            "seed",
            "use_mixed_precision",
            "gradient_checkpointing",
            "compile_model",
            "distributed",
            "checkpoint_dir",
            "data_dir",
            "log_dir",
            "use_wandb",
            "wandb_project",
        ]
        for key in system_keys:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ChessConfig:
        """Load configuration from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_preset(cls, preset: str) -> ChessConfig:
        """Create configuration from named preset.

        Args:
            preset: One of "small", "medium", "large", "competition"

        Returns:
            ChessConfig with preset values
        """
        presets = {
            "small": get_chess_small_config,
            "medium": get_chess_medium_config,
            "large": get_chess_large_config,
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return presets[preset]()

    def to_system_config(self) -> SystemConfig:
        """Convert to base SystemConfig for framework compatibility."""
        from src.training.system_config import (
            HRMConfig,
            MCTSConfig,
            NeuralNetworkConfig,
            SystemConfig,
            TrainingConfig,
            TRMConfig,
        )

        system_config = SystemConfig()

        # Map HRM config
        system_config.hrm = HRMConfig(
            h_dim=self.hrm.h_dim,
            l_dim=self.hrm.l_dim,
            num_h_layers=self.hrm.num_h_layers,
            num_l_layers=self.hrm.num_l_layers,
            max_outer_steps=self.hrm.max_outer_steps,
            halt_threshold=self.hrm.halt_threshold,
            use_augmentation=self.hrm.use_augmentation,
            dropout=self.hrm.dropout,
            ponder_epsilon=self.hrm.ponder_epsilon,
            max_ponder_steps=self.hrm.max_ponder_steps,
        )

        # Map TRM config
        system_config.trm = TRMConfig(
            latent_dim=self.trm.latent_dim,
            num_recursions=self.trm.num_recursions,
            hidden_dim=self.trm.hidden_dim,
            deep_supervision=self.trm.deep_supervision,
            supervision_weight_decay=self.trm.supervision_weight_decay,
            convergence_threshold=self.trm.convergence_threshold,
            min_recursions=self.trm.min_recursions,
            dropout=self.trm.dropout,
            use_layer_norm=self.trm.use_layer_norm,
        )

        # Map MCTS config
        system_config.mcts = MCTSConfig(
            num_simulations=self.mcts.num_simulations,
            c_puct=self.mcts.c_puct,
            dirichlet_epsilon=self.mcts.dirichlet_epsilon,
            dirichlet_alpha=self.mcts.dirichlet_alpha,
            temperature_threshold=self.mcts.temperature_threshold,
            temperature_init=self.mcts.temperature_init,
            temperature_final=self.mcts.temperature_final,
            virtual_loss=self.mcts.virtual_loss,
            num_parallel=self.mcts.num_parallel,
            use_progressive_widening=self.mcts.use_progressive_widening,
            pw_k=self.mcts.pw_k,
            pw_alpha=self.mcts.pw_alpha,
        )

        # Map Neural Network config
        system_config.neural_net = NeuralNetworkConfig(
            num_res_blocks=self.neural_net.num_res_blocks,
            num_channels=self.neural_net.num_channels,
            policy_conv_channels=self.neural_net.policy_conv_channels,
            policy_fc_dim=self.neural_net.policy_fc_dim,
            value_conv_channels=self.neural_net.value_conv_channels,
            value_fc_hidden=self.neural_net.value_fc_hidden,
            use_batch_norm=self.neural_net.use_batch_norm,
            dropout=self.neural_net.dropout,
            weight_decay=self.neural_net.weight_decay,
            input_channels=self.input_channels,
            action_size=self.action_size,
        )

        # Map Training config
        system_config.training = TrainingConfig(
            games_per_iteration=self.training.games_per_iteration,
            num_actors=self.training.num_actors,
            buffer_size=self.training.buffer_size,
            batch_size=self.training.batch_size,
            learning_rate=self.training.learning_rate,
            momentum=self.training.momentum,
            weight_decay=self.training.weight_decay,
            lr_schedule=self.training.lr_schedule,
            lr_decay_steps=self.training.lr_decay_steps,
            lr_decay_gamma=self.training.lr_decay_gamma,
            epochs_per_iteration=self.training.epochs_per_iteration,
            checkpoint_interval=self.training.checkpoint_interval,
            evaluation_games=self.training.evaluation_games,
            win_rate_threshold=self.training.win_rate_threshold,
            patience=self.training.patience,
            min_delta=self.training.min_delta,
        )

        # System settings
        system_config.device = self.device
        system_config.seed = self.seed
        system_config.use_mixed_precision = self.use_mixed_precision
        system_config.gradient_checkpointing = self.gradient_checkpointing
        system_config.compile_model = self.compile_model
        system_config.distributed = self.distributed
        system_config.world_size = self.world_size
        system_config.rank = self.rank
        system_config.backend = self.backend
        system_config.checkpoint_dir = self.checkpoint_dir
        system_config.data_dir = self.data_dir
        system_config.log_dir = self.log_dir
        system_config.use_wandb = self.use_wandb
        system_config.wandb_project = self.wandb_project
        system_config.wandb_entity = self.wandb_entity
        system_config.log_interval = self.log_interval

        return system_config


def get_chess_small_config() -> ChessConfig:
    """Configuration for fast experimentation with reduced model sizes."""
    config = ChessConfig()

    # Smaller models
    config.hrm.h_dim = 256
    config.hrm.l_dim = 128
    config.trm.latent_dim = 128
    config.trm.hidden_dim = 256
    config.neural_net.num_res_blocks = 9
    config.neural_net.num_channels = 128

    # Fewer simulations
    config.mcts.num_simulations = 100
    config.mcts.num_parallel = 4

    # Reduced training
    config.training.games_per_iteration = 500
    config.training.num_actors = 8
    config.training.batch_size = 256
    config.training.buffer_size = 50_000

    return config


def get_chess_medium_config() -> ChessConfig:
    """Configuration for balanced training with moderate resources."""
    config = ChessConfig()

    # Medium models
    config.neural_net.num_res_blocks = 19
    config.neural_net.num_channels = 256

    # Moderate simulations
    config.mcts.num_simulations = 400
    config.mcts.num_parallel = 8

    # Moderate training
    config.training.games_per_iteration = 2500
    config.training.num_actors = 32
    config.training.batch_size = 1024
    config.training.buffer_size = 250_000

    return config


def get_chess_large_config() -> ChessConfig:
    """Configuration for maximum performance with high resources."""
    config = ChessConfig()

    # Large models
    config.hrm.h_dim = 768
    config.hrm.l_dim = 384
    config.trm.latent_dim = 384
    config.trm.hidden_dim = 768
    config.neural_net.num_res_blocks = 39
    config.neural_net.num_channels = 384

    # More simulations
    config.mcts.num_simulations = 1600
    config.mcts.num_parallel = 16

    # Full training
    config.training.games_per_iteration = 25_000
    config.training.num_actors = 128
    config.training.batch_size = 4096
    config.training.buffer_size = 1_000_000

    # Optimization
    config.use_mixed_precision = True
    config.gradient_checkpointing = True

    return config
