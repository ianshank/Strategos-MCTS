"""Training profiles and predefined configurations for self-play convergence and domain training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.training.self_play_trainer import SelfPlayConfig
from src.training.system_config import MCTSConfig


class TrainingProfile(str, Enum):
    """Preset operational profiles for training self-play models."""

    SMOKE = "smoke"
    DEV = "dev"
    FULL = "full"


@dataclass(frozen=True)
class TrainingProfileSpec:
    """Combined configuration spec for a given training profile."""

    iterations: int
    num_simulations: int
    games_per_iteration: int
    batch_size: int
    learning_rate: float
    use_amp: bool
    compile_model: bool
    pin_memory: bool
    device: str


_PROFILES: dict[TrainingProfile, TrainingProfileSpec] = {
    TrainingProfile.SMOKE: TrainingProfileSpec(
        iterations=2,
        num_simulations=8,
        games_per_iteration=4,
        batch_size=8,
        learning_rate=1e-3,
        use_amp=False,
        compile_model=False,
        pin_memory=False,
        device="cpu",
    ),
    TrainingProfile.DEV: TrainingProfileSpec(
        iterations=20,
        num_simulations=200,
        games_per_iteration=50,
        batch_size=64,
        learning_rate=1e-3,
        use_amp=True,
        compile_model=False,
        pin_memory=True,
        device="auto",
    ),
    TrainingProfile.FULL: TrainingProfileSpec(
        iterations=200,
        num_simulations=800,
        games_per_iteration=500,
        batch_size=256,
        learning_rate=5e-4,
        use_amp=True,
        compile_model=True,
        pin_memory=True,
        device="auto",
    ),
}


def get_training_profile(profile: str | TrainingProfile) -> TrainingProfileSpec:
    """Retrieve pre-configured TrainingProfileSpec by name."""
    if isinstance(profile, str):
        try:
            profile_enum = TrainingProfile(profile.lower())
        except ValueError:
            valid = [p.value for p in TrainingProfile]
            raise ValueError(f"Unknown profile '{profile}'. Valid profiles: {valid}")
    else:
        profile_enum = profile

    return _PROFILES[profile_enum]


def profile_to_configs(
    spec: TrainingProfileSpec,
) -> tuple[SelfPlayConfig, MCTSConfig]:
    """Convert a TrainingProfileSpec to matching (SelfPlayConfig, MCTSConfig) instances."""
    self_play_cfg = SelfPlayConfig(
        num_games_per_iteration=spec.games_per_iteration,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        use_amp=spec.use_amp,
        compile_model=spec.compile_model,
        pin_memory=spec.pin_memory,
    )
    mcts_cfg = MCTSConfig(
        num_simulations=spec.num_simulations,
    )
    return self_play_cfg, mcts_cfg
