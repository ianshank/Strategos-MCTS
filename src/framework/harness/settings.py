"""Harness configuration via Pydantic Settings.

Standalone ``BaseSettings`` instance with the ``HARNESS_`` env prefix. Kept
separate from the global :class:`src.config.settings.Settings` so the harness
can evolve without disturbing existing callers, while still composable through
``HarnessFactory``. All thresholds, timeouts, paths, budgets, and feature
flags live here — no hardcoded values are permitted at call sites.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TopologyName(str, Enum):
    """Built-in multi-agent topologies."""

    PIPELINE = "pipeline"
    FAN_OUT_IN = "fan_out_in"
    EXPERT_POOL = "expert_pool"
    PRODUCER_REVIEWER = "producer_reviewer"
    SUPERVISOR = "supervisor"
    HIERARCHICAL = "hierarchical"


class AggregationPolicy(str, Enum):
    """Topology output aggregation policies."""

    FIRST_SUCCESS = "first_success"
    ALL_MUST_PASS = "all_must_pass"
    VERIFIER_RANKED = "verifier_ranked"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class HarnessPermissions(BaseSettings):
    """Permission flags governing tool capability.

    Loaded as ``HARNESS_PERM_*`` env vars (e.g. ``HARNESS_PERM_NETWORK=false``).
    """

    model_config = SettingsConfigDict(
        env_prefix="HARNESS_PERM_",
        case_sensitive=True,
        extra="ignore",
    )

    READ: bool = Field(default=True, description="Allow file reads")
    WRITE: bool = Field(default=True, description="Allow file writes (hashed edits)")
    SHELL: bool = Field(default=False, description="Allow shell command execution")
    NETWORK: bool = Field(default=False, description="Allow outbound network from tools")
    DELETE: bool = Field(default=False, description="Allow file deletion")


class HarnessSettings(BaseSettings):
    """Settings for the agent harness.

    All fields configurable via ``HARNESS_*`` env vars or a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    ENABLED: bool = Field(default=True, description="Master switch for harness")

    # Loop budgets
    MAX_ITERATIONS: int = Field(default=10, ge=1, le=10_000, description="Max control-loop iterations")
    ITERATION_TIMEOUT_SECONDS: float = Field(default=120.0, ge=0.1, le=3600.0)
    TOTAL_BUDGET_SECONDS: float = Field(default=1800.0, ge=1.0, le=86_400.0)
    TOKEN_BUDGET_PER_LOOP: int = Field(default=200_000, ge=1, le=10_000_000)

    # Tool output management
    TOOL_OUTPUT_HEAD_CHARS: int = Field(default=2000, ge=1, le=100_000)
    TOOL_OUTPUT_TAIL_CHARS: int = Field(default=2000, ge=0, le=100_000)
    TOOL_OUTPUT_TRUNCATION_MARKER: str = Field(default="\n…[truncated; full log at {path}]…\n")
    TOOL_OUTPUT_SPILLOVER_DIR: Path = Field(default=Path(".harness/spillover"))
    TOOL_DEFAULT_TIMEOUT_SECONDS: float = Field(default=60.0, ge=0.1, le=3600.0)

    # Memory
    MEMORY_ROOT: Path = Field(default=Path("memory"))
    MEMORY_INDEX_FILENAME: str = Field(default="MEMORY.md")
    MEMORY_EVENT_LOG_DIR: str = Field(default="episodic")
    MEMORY_EPISODIC_FILENAME_PATTERN: str = Field(default="{date:%Y-%m-%d}.md")
    MEMORY_COMPACTOR_INTERVAL_SECONDS: float = Field(default=30.0, ge=0.5, le=3600.0)
    MEMORY_HEARTBEAT_INTERVAL_SECONDS: float = Field(default=60.0, ge=1.0, le=86_400.0)
    MEMORY_HEARTBEAT_ENABLED: bool = Field(default=True)
    MEMORY_MAX_EVENT_BYTES: int = Field(default=4096, ge=128, le=1_048_576)

    # Topologies
    TOPOLOGY: TopologyName = Field(default=TopologyName.PRODUCER_REVIEWER)
    AGGREGATION_POLICY: AggregationPolicy = Field(default=AggregationPolicy.VERIFIER_RANKED)

    # Hashed-edit tool
    HASHED_EDIT_WINDOW: int = Field(default=1, ge=0, le=20, description="Lines on each side of anchor to hash")

    # Replay / record
    RECORD_DIR: Path | None = Field(default=None, description="If set, ReplayLLMClient records cassettes here")
    REPLAY_DIR: Path | None = Field(default=None, description="If set, ReplayLLMClient replays from this directory")
    DETERMINISTIC_CLOCK: bool = Field(default=False, description="Use frozen clock + counter UUIDs/RNG")
    SEED: int | None = Field(default=None, ge=0, description="Override Settings.SEED for harness runs")

    # Planner
    PLANNER_ENABLED: bool = Field(default=True)
    PLANNER_MAX_TOKENS: int = Field(default=2_000, ge=1, le=200_000)

    # Reasoner (per-iteration LLM call inside the control loop)
    REASON_MAX_TOKENS: int = Field(
        default=4_000,
        ge=1,
        le=200_000,
        description="Max tokens for the per-iteration Reason phase LLM call",
    )

    # Ralph loop
    RALPH_ENABLED: bool = Field(default=False)
    RALPH_COMPLETION_MARKER: str = Field(default="<!-- HARNESS:DONE -->")
    RALPH_MAX_LOOPS: int = Field(default=50, ge=1, le=10_000)
    RALPH_STUCK_BEHAVIOR: Literal["abort", "escalate", "pivot"] = Field(default="abort")
    RALPH_STUCK_THRESHOLD: int = Field(default=3, ge=1, le=100, description="Identical-failure count to declare stuck")

    # Hook chain
    HOOK_SHORT_CIRCUIT_DEFAULT: bool = Field(default=True)

    # Verification
    VERIFIER_FAIL_FAST: bool = Field(default=True)

    # Output
    OUTPUT_DIR: Path = Field(default=Path(".harness/runs"))

    @field_validator(
        "TOOL_OUTPUT_SPILLOVER_DIR",
        "MEMORY_ROOT",
        "OUTPUT_DIR",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        """Coerce ``str`` env vars into ``Path``."""
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        raise TypeError(f"Expected path-like value, got {type(v).__name__}")

    @field_validator("RECORD_DIR", "REPLAY_DIR", mode="before")
    @classmethod
    def _coerce_optional_path(cls, v: object) -> Path | None:
        """Coerce optional path env vars."""
        if v is None or v == "":
            return None
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        raise TypeError(f"Expected path-like value or None, got {type(v).__name__}")

    @model_validator(mode="after")
    def _validate_budgets(self) -> HarnessSettings:
        """Ensure budgets are mutually consistent."""
        if self.ITERATION_TIMEOUT_SECONDS > self.TOTAL_BUDGET_SECONDS:
            raise ValueError("HARNESS_ITERATION_TIMEOUT_SECONDS must be <= HARNESS_TOTAL_BUDGET_SECONDS")
        if self.RECORD_DIR is not None and self.REPLAY_DIR is not None and self.RECORD_DIR == self.REPLAY_DIR:
            raise ValueError("HARNESS_RECORD_DIR and HARNESS_REPLAY_DIR must differ if both set")
        return self

    def episodic_dir(self) -> Path:
        """Directory holding append-only episodic event logs."""
        return self.MEMORY_ROOT / self.MEMORY_EVENT_LOG_DIR

    def index_path(self) -> Path:
        """Resolved path to ``MEMORY.md`` (or whichever file is configured)."""
        return self.MEMORY_ROOT / self.MEMORY_INDEX_FILENAME

    def safe_dict(self) -> dict[str, object]:
        """Return settings as a dict suitable for logging (no secrets stored here)."""
        return self.model_dump(mode="json")


_harness_settings: HarnessSettings | None = None


def get_harness_settings() -> HarnessSettings:
    """Return the lazily-cached harness settings instance."""
    global _harness_settings
    if _harness_settings is None:
        _harness_settings = HarnessSettings()
    return _harness_settings


def reset_harness_settings() -> None:
    """Clear cached settings (test-only)."""
    global _harness_settings
    _harness_settings = None


__all__ = [
    "AggregationPolicy",
    "HarnessPermissions",
    "HarnessSettings",
    "TopologyName",
    "get_harness_settings",
    "reset_harness_settings",
]
