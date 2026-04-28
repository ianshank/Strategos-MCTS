"""Internal submodule (split from component_factory.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import Settings


@dataclass
class TrainerConfig:
    """Configuration for trainer component creation."""

    batch_size: int = 32
    num_batches: int = 10
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = False
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"

    # HRM-specific
    ponder_weight: float = 0.01
    consistency_weight: float = 0.1

    # TRM-specific
    supervision_weight_decay: float = 0.5

    # Evaluation-specific
    num_eval_games: int = 20
    eval_temperature: float = 0.0
    mcts_iterations: int = 100
    win_threshold: float = 0.55

    # Replay buffer-specific
    buffer_capacity: int = 100000
    prioritized_alpha: float = 0.6
    prioritized_beta_start: float = 0.4
    prioritized_beta_frames: int = 100000

    additional_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> TrainerConfig:
        """Create configuration from settings."""
        return cls(
            batch_size=settings.MCTS_MAX_PARALLEL_ROLLOUTS * 8,
            gradient_clip_norm=1.0,
            use_mixed_precision=False,
            device="cuda" if settings.MCTS_IMPL.value == "neural" else "cpu",
            checkpoint_dir=settings.S3_PREFIX if settings.S3_BUCKET else "checkpoints",
            mcts_iterations=settings.MCTS_ITERATIONS,
        )


@dataclass
class MetricsConfig:
    """Configuration for metrics component creation."""

    window_size: int = 100
    enable_gpu_monitoring: bool = True
    alert_threshold_ms: float = 1000.0
    log_frequency: int = 1
    save_artifacts: bool = True
    project_name: str = "langgraph-mcts"
    wandb_entity: str | None = None
    offline_mode: bool = False

    additional_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> MetricsConfig:
        """Create configuration from settings."""
        return cls(
            project_name=settings.WANDB_PROJECT,
            wandb_entity=settings.WANDB_ENTITY,
            offline_mode=settings.WANDB_MODE == "offline",
            log_frequency=1,
        )


@dataclass
class DataLoaderConfig:
    """Configuration for data loader component creation."""

    cache_dir: str | None = None
    max_samples: int | None = None
    streaming: bool = True
    include_instruct: bool = True
    batch_size: int = 32

    additional_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> DataLoaderConfig:
        """Create configuration from settings."""
        cache_dir = str(Path.home() / ".cache" / "mcts_datasets")
        return cls(
            cache_dir=cache_dir,
            batch_size=settings.MCTS_MAX_PARALLEL_ROLLOUTS * 8,
        )
