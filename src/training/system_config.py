"""
System Configuration for LangGraph Multi-Agent MCTS with DeepMind-Style Learning.

This module provides centralized configuration management for all framework components
including HRM, TRM, Neural MCTS, and training infrastructure.
"""

from dataclasses import dataclass, field

import torch

from src.utils.device import get_default_device_str


@dataclass
class HRMConfig:
    """Configuration for Hierarchical Reasoning Model (HRM) Agent."""

    # Model dimensions
    h_dim: int = 512  # High-level planning dimension
    l_dim: int = 256  # Low-level execution dimension
    num_h_layers: int = 2  # Number of high-level layers
    num_l_layers: int = 4  # Number of low-level layers

    # Halting and iteration control
    max_outer_steps: int = 10  # Maximum planning steps
    halt_threshold: float = 0.95  # Confidence threshold for halting

    # Training features
    use_augmentation: bool = True  # Use tactical augmentation
    dropout: float = 0.1

    # ACT (Adaptive Computation Time) parameters
    ponder_epsilon: float = 0.01  # Small constant for numerical stability
    max_ponder_steps: int = 16  # Maximum pondering steps

    # Loss weights (for training)
    ponder_weight: float = 0.01  # Weight for ponder cost in loss
    consistency_weight: float = 0.1  # Weight for convergence consistency


@dataclass
class TRMConfig:
    """Configuration for Tiny Recursive Model (TRM) Agent."""

    # Model architecture
    latent_dim: int = 256  # Latent state dimension
    num_recursions: int = 16  # Maximum recursion depth
    hidden_dim: int = 512  # Hidden layer dimension

    # Deep supervision
    deep_supervision: bool = True  # Enable supervision at all recursion levels
    supervision_weight_decay: float = 0.5  # Decay factor for deeper levels

    # Convergence criteria
    convergence_threshold: float = 0.01  # L2 distance threshold
    min_recursions: int = 3  # Minimum recursions before checking convergence

    # Training
    dropout: float = 0.1
    use_layer_norm: bool = True


@dataclass
class MCTSConfig:
    """Configuration for Neural-Guided MCTS."""

    # Search parameters
    num_simulations: int = 1600  # AlphaGo Zero used 1600
    c_puct: float = 1.25  # Exploration constant for PUCT

    # Dirichlet noise for exploration (applied at root)
    dirichlet_epsilon: float = 0.25  # Mix of prior and noise
    dirichlet_alpha: float = 0.3  # Game/task specific

    # Temperature for action selection
    temperature_threshold: int = 30  # Move number to switch to greedy
    temperature_init: float = 1.0  # Initial temperature
    temperature_final: float = 0.1  # Final temperature

    # Virtual loss for parallel MCTS
    virtual_loss: float = 3.0  # Discourage simultaneous exploration
    num_parallel: int = 8  # Parallel search threads

    # Progressive widening
    use_progressive_widening: bool = True
    pw_k: float = 1.0
    pw_alpha: float = 0.5


@dataclass
class NeuralNetworkConfig:
    """Configuration for Policy-Value Networks."""

    # ResNet architecture
    num_res_blocks: int = 19  # AlphaGo Zero used 19 or 39
    num_channels: int = 256  # Feature channels

    # Policy head
    policy_conv_channels: int = 2
    policy_fc_dim: int = 256

    # Value head
    value_conv_channels: int = 1
    value_fc_hidden: int = 256

    # Regularization
    use_batch_norm: bool = True
    dropout: float = 0.0  # Usually 0 for ResNets with BN
    weight_decay: float = 1e-4

    # Input/Output
    input_channels: int = 17  # Game/task specific
    action_size: int = 362  # Game/task specific (e.g., Go: 19x19 + pass)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Self-play generation
    games_per_iteration: int = 25_000
    num_actors: int = 128  # Parallel self-play workers

    # Experience replay
    buffer_size: int = 500_000  # Keep last N positions
    batch_size: int = 2048

    # Optimization
    learning_rate: float = 0.2  # With LR schedule
    momentum: float = 0.9  # SGD momentum
    weight_decay: float = 1e-4

    # Learning rate schedule
    lr_schedule: str = "cosine"  # "cosine", "step", "constant"
    lr_decay_steps: int = 100  # For step schedule
    lr_decay_gamma: float = 0.1

    # Training loop
    epochs_per_iteration: int = 1
    checkpoint_interval: int = 10  # Save every N iterations

    # Agent-specific training
    hrm_train_batches: int = 10  # Batches per iteration for HRM
    trm_train_batches: int = 10  # Batches per iteration for TRM
    gradient_clip_norm: float = 1.0  # Max gradient norm for clipping

    # Evaluation
    evaluation_games: int = 400  # Games to evaluate new model
    win_rate_threshold: float = 0.55  # Required win rate to replace best model
    win_threshold: float = 0.55  # Alias for compatibility
    eval_temperature: float = 0.0  # Temperature for evaluation (0 = deterministic)

    # Early stopping
    patience: int = 20  # Iterations without improvement
    min_delta: float = 0.01  # Minimum improvement


@dataclass
class SystemConfig:
    """
    Master configuration for the entire LangGraph Multi-Agent MCTS system.

    This provides centralized configuration management with sensible defaults
    based on DeepMind's AlphaGo Zero and research on HRM/TRM architectures.
    """

    # Component configurations
    hrm: HRMConfig = field(default_factory=HRMConfig)
    trm: TRMConfig = field(default_factory=TRMConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    neural_net: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # System settings
    device: str = field(default_factory=lambda: get_default_device_str())
    seed: int = 42  # For reproducibility

    # Performance optimizations
    use_mixed_precision: bool = True  # FP16 training
    gradient_checkpointing: bool = False  # Trade compute for memory
    compile_model: bool = False  # PyTorch 2.0 compilation

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU

    # Logging and monitoring
    log_interval: int = 10  # Log every N iterations
    use_wandb: bool = False  # Weights & Biases integration
    wandb_project: str = "langgraph-mcts-deepmind"
    wandb_entity: str | None = None

    # Paths
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./data"
    log_dir: str = "./logs"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure device is valid
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        # Adjust settings based on device
        if self.device == "cpu":
            self.use_mixed_precision = False
            self.distributed = False
            self.backend = "gloo"

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "hrm": self.hrm.__dict__,
            "trm": self.trm.__dict__,
            "mcts": self.mcts.__dict__,
            "neural_net": self.neural_net.__dict__,
            "training": self.training.__dict__,
            "device": self.device,
            "seed": self.seed,
            "use_mixed_precision": self.use_mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "compile_model": self.compile_model,
            "distributed": self.distributed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SystemConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Update nested configs
        if "hrm" in config_dict:
            for key, value in config_dict["hrm"].items():
                setattr(config.hrm, key, value)

        if "trm" in config_dict:
            for key, value in config_dict["trm"].items():
                setattr(config.trm, key, value)

        if "mcts" in config_dict:
            for key, value in config_dict["mcts"].items():
                setattr(config.mcts, key, value)

        if "neural_net" in config_dict:
            for key, value in config_dict["neural_net"].items():
                setattr(config.neural_net, key, value)

        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                setattr(config.training, key, value)

        # Update system settings
        for key in [
            "device",
            "seed",
            "use_mixed_precision",
            "gradient_checkpointing",
            "compile_model",
            "distributed",
            "log_interval",
            "use_wandb",
        ]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    def save(self, path: str):
        """Save configuration to file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SystemConfig":
        """Load configuration from file."""
        import json

        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations for different use cases
def get_small_config() -> SystemConfig:
    """Configuration for fast experimentation (reduced model sizes)."""
    config = SystemConfig()

    # Smaller models
    config.hrm.h_dim = 256
    config.hrm.l_dim = 128
    config.trm.latent_dim = 128
    config.neural_net.num_res_blocks = 9
    config.neural_net.num_channels = 128

    # Fewer simulations
    config.mcts.num_simulations = 400
    config.training.games_per_iteration = 1000
    config.training.num_actors = 16

    return config


def get_medium_config() -> SystemConfig:
    """Configuration for balanced training (moderate resources)."""
    config = SystemConfig()

    # Default settings are already medium
    config.neural_net.num_res_blocks = 19
    config.mcts.num_simulations = 800
    config.training.games_per_iteration = 10_000
    config.training.num_actors = 64

    return config


def get_large_config() -> SystemConfig:
    """Configuration for maximum performance (high resources)."""
    config = SystemConfig()

    # Larger models
    config.hrm.h_dim = 768
    config.hrm.l_dim = 384
    config.trm.latent_dim = 384
    config.neural_net.num_res_blocks = 39
    config.neural_net.num_channels = 384

    # More simulations
    config.mcts.num_simulations = 3200
    config.training.games_per_iteration = 50_000
    config.training.num_actors = 256

    # Optimization
    config.use_mixed_precision = True
    config.gradient_checkpointing = True

    return config


def get_arc_agi_config() -> SystemConfig:
    """Configuration optimized for ARC-AGI benchmark tasks."""
    config = SystemConfig()

    # ARC-AGI specific settings
    config.hrm.h_dim = 512
    config.hrm.l_dim = 256
    config.hrm.max_outer_steps = 20  # Complex reasoning

    config.trm.num_recursions = 20  # Deep refinement
    config.trm.convergence_threshold = 0.005  # Precise solutions

    config.mcts.num_simulations = 1600
    config.mcts.c_puct = 1.5  # More exploration for puzzle solving

    # Input/output for grid tasks
    config.neural_net.input_channels = 11  # 10 colors + 1 empty
    config.neural_net.action_size = 100  # Depends on grid size

    return config
