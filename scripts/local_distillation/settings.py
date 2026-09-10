"""Pydantic knobs for the local distillation driver (``LOCAL_DISTILLATION_*``)."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from scripts.local_distillation.schema import SCHEMA_VERSION
from src.games.connect_four.config import ConnectFourConfig

_C4 = ConnectFourConfig()


class DistillationSettings(BaseSettings):
    """Env-tunable distillation contract. Call sites read fields; they do not hardcode bounds."""

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_DISTILLATION_",
        case_sensitive=False,
        extra="ignore",
    )

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    min_simulations: int = Field(default=1, ge=1)
    default_simulations: int = Field(default=8, ge=1)
    temperature_init: float = Field(default=1.0, gt=0.0)
    temperature_final: float = Field(default=0.1, ge=0.0)
    temperature_threshold: int = Field(default=30, ge=0)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    value_loss_weight: float = Field(default=1.0, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    input_channels: int = Field(default=_C4.input_channels, ge=1)
    board_rows: int = Field(default=_C4.board_rows, ge=1)
    board_cols: int = Field(default=_C4.board_cols, ge=1)
    action_size: int = Field(default=_C4.action_space_size, ge=1)
    num_res_blocks: int = Field(default=2, ge=1)
    num_channels: int = Field(default=32, ge=1)
    recurrent_enabled: bool = Field(default=False)
    recurrences: int = Field(default=4, ge=1)
    promotion_min_delta: float = Field(default=0.0, ge=0.0)
    train_frac: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_frac: float = Field(default=0.1, gt=0.0, lt=1.0)
    buffer_capacity: int = Field(default=10_000, ge=1)
    wall_clock_repeat_cap: int = Field(default=1_000_000, ge=1)
    device: str = Field(default="cpu", min_length=1)

    @model_validator(mode="after")
    def _fractions_fit(self) -> DistillationSettings:
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must leave a test remainder")
        return self


def get_distillation_settings() -> DistillationSettings:
    """Load settings from the environment (cached construction is the caller's choice)."""
    return DistillationSettings()
