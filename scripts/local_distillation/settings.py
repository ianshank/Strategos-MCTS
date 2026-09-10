"""Pydantic knobs for the local distillation driver (``LOCAL_DISTILLATION_*``)."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from scripts.local_distillation.schema import SCHEMA_VERSION
from src.games.connect_four.config import ConnectFourConfig

_C4 = ConnectFourConfig()


def _distillation_env(name: str) -> str:
    return f"LOCAL_DISTILLATION_{name}"


class DistillationSettings(BaseSettings):
    """Env-tunable distillation contract. Call sites read fields; they do not hardcode bounds."""

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_DISTILLATION_",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
    )

    schema_version: int = Field(
        default=SCHEMA_VERSION, ge=1, validation_alias=_distillation_env("SCHEMA_VERSION")
    )
    min_simulations: int = Field(default=1, ge=1, validation_alias=_distillation_env("MIN_SIMULATIONS"))
    default_simulations: int = Field(default=8, ge=1, validation_alias=_distillation_env("DEFAULT_SIMULATIONS"))
    temperature_init: float = Field(default=1.0, gt=0.0, validation_alias=_distillation_env("TEMPERATURE_INIT"))
    temperature_final: float = Field(default=0.1, ge=0.0, validation_alias=_distillation_env("TEMPERATURE_FINAL"))
    temperature_threshold: int = Field(default=30, ge=0, validation_alias=_distillation_env("TEMPERATURE_THRESHOLD"))
    batch_size: int = Field(default=8, ge=1, validation_alias=_distillation_env("BATCH_SIZE"))
    learning_rate: float = Field(default=1e-3, gt=0.0, validation_alias=_distillation_env("LEARNING_RATE"))
    value_loss_weight: float = Field(default=1.0, ge=0.0, validation_alias=_distillation_env("VALUE_LOSS_WEIGHT"))
    grad_clip: float = Field(default=1.0, ge=0.0, validation_alias=_distillation_env("GRAD_CLIP"))
    input_channels: int = Field(default=_C4.input_channels, ge=1, validation_alias=_distillation_env("INPUT_CHANNELS"))
    board_rows: int = Field(default=_C4.board_rows, ge=1, validation_alias=_distillation_env("BOARD_ROWS"))
    board_cols: int = Field(default=_C4.board_cols, ge=1, validation_alias=_distillation_env("BOARD_COLS"))
    action_size: int = Field(default=_C4.action_space_size, ge=1, validation_alias=_distillation_env("ACTION_SIZE"))
    num_res_blocks: int = Field(default=2, ge=1, validation_alias=_distillation_env("NUM_RES_BLOCKS"))
    num_channels: int = Field(default=32, ge=1, validation_alias=_distillation_env("NUM_CHANNELS"))
    recurrent_enabled: bool = Field(default=False, validation_alias=_distillation_env("RECURRENT_ENABLED"))
    recurrences: int = Field(default=4, ge=1, validation_alias=_distillation_env("RECURRENCES"))
    promotion_min_delta: float = Field(default=0.0, ge=0.0, validation_alias=_distillation_env("PROMOTION_MIN_DELTA"))
    train_frac: float = Field(default=0.8, gt=0.0, lt=1.0, validation_alias=_distillation_env("TRAIN_FRAC"))
    val_frac: float = Field(default=0.1, gt=0.0, lt=1.0, validation_alias=_distillation_env("VAL_FRAC"))
    buffer_capacity: int = Field(default=10_000, ge=1, validation_alias=_distillation_env("BUFFER_CAPACITY"))
    wall_clock_repeat_cap: int = Field(
        default=1_000_000, ge=1, validation_alias=_distillation_env("WALL_CLOCK_REPEAT_CAP")
    )
    device: str = Field(default="cpu", min_length=1, validation_alias=_distillation_env("DEVICE"))

    @model_validator(mode="after")
    def _fractions_fit(self) -> DistillationSettings:
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must leave a test remainder")
        return self


def get_distillation_settings() -> DistillationSettings:
    """Load settings from the environment (cached construction is the caller's choice)."""
    return DistillationSettings()
