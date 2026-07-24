"""Dedicated settings for LangGraph orchestration hardening.

Kept separate from the monolithic application ``Settings`` so the graph-hardening config
(retry / tracing / checkpoint) validates on its own: a malformed ``GRAPH_*`` value fails fast
at construction, while unrelated concerns (e.g. a missing LLM API key, which ``Settings``
requires) never silently disable these features. All defaults and bounds come from
``src.config.constants`` — no hardcoded values here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.constants import (
    DEFAULT_CANDIDATE_SCORER,
    DEFAULT_GRAPH_NODE_RETRY_BACKOFF_FACTOR,
    DEFAULT_GRAPH_NODE_RETRY_EXCEPTIONS,
    DEFAULT_GRAPH_NODE_RETRY_INITIAL_DELAY_SECONDS,
    DEFAULT_GRAPH_NODE_RETRY_MAX_ATTEMPTS,
    MAX_GRAPH_NODE_RETRY_ATTEMPTS,
    MAX_GRAPH_NODE_RETRY_BACKOFF_FACTOR,
    MAX_GRAPH_NODE_RETRY_DELAY_SECONDS,
    MIN_GRAPH_NODE_RETRY_ATTEMPTS,
    MIN_GRAPH_NODE_RETRY_BACKOFF_FACTOR,
    MIN_GRAPH_NODE_RETRY_DELAY_SECONDS,
)


class GraphHardeningSettings(BaseSettings):
    """Validated configuration for graph node retry, tracing, and checkpointing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    # Retry-with-backoff at worker-node I/O boundaries.
    GRAPH_NODE_RETRY_ENABLED: bool = Field(
        default=True, description="Enable retry-with-backoff on transient graph worker-node I/O failures"
    )
    GRAPH_NODE_RETRY_MAX_ATTEMPTS: int = Field(
        default=DEFAULT_GRAPH_NODE_RETRY_MAX_ATTEMPTS,
        ge=MIN_GRAPH_NODE_RETRY_ATTEMPTS,
        le=MAX_GRAPH_NODE_RETRY_ATTEMPTS,
        description="Max attempts for a retryable node I/O call before propagating (stacks over adapter retries)",
    )
    GRAPH_NODE_RETRY_INITIAL_DELAY_SECONDS: float = Field(
        default=DEFAULT_GRAPH_NODE_RETRY_INITIAL_DELAY_SECONDS,
        ge=MIN_GRAPH_NODE_RETRY_DELAY_SECONDS,
        le=MAX_GRAPH_NODE_RETRY_DELAY_SECONDS,
        description="Initial backoff delay (seconds) before the first retry of a node I/O call",
    )
    GRAPH_NODE_RETRY_BACKOFF_FACTOR: float = Field(
        default=DEFAULT_GRAPH_NODE_RETRY_BACKOFF_FACTOR,
        ge=MIN_GRAPH_NODE_RETRY_BACKOFF_FACTOR,
        le=MAX_GRAPH_NODE_RETRY_BACKOFF_FACTOR,
        description="Multiplier applied to the retry delay after each failed attempt",
    )
    GRAPH_NODE_RETRY_EXCEPTIONS: list[str] = Field(
        default_factory=lambda: list(DEFAULT_GRAPH_NODE_RETRY_EXCEPTIONS),
        description="Allowlist of retryable exceptions (bare builtin names or dotted import paths)",
    )

    # Execution trace logging (structured event per node transition).
    GRAPH_TRACE_ENABLED: bool = Field(
        default=True, description="Emit a structured trace event for every graph node transition"
    )
    GRAPH_TRACE_DIR: str | None = Field(
        default=None,
        description="Directory for per-run JSONL trace files; None emits to structured logs/metrics only",
    )

    # Checkpoint backend selection (LangGraph checkpointer).
    GRAPH_CHECKPOINT_BACKEND: Literal["memory", "sqlite"] = Field(
        default="memory",
        description=(
            "LangGraph checkpoint backend: 'memory' (in-process, default) or 'sqlite'. "
            "'sqlite' is durable only when GRAPH_CHECKPOINT_SQLITE_PATH is set; with no path it "
            "uses an in-memory sqlite DB (not durable)."
        ),
    )
    GRAPH_CHECKPOINT_SQLITE_PATH: str | None = Field(
        default=None,
        description="SQLite DB path for the 'sqlite' checkpoint backend (None => in-memory sqlite)",
    )

    # MCTS candidate-scoring seam. Chooses how the winning candidate action is selected
    # after search returns per-candidate statistics. 'identity' preserves the engine's own
    # MAX_VISITS selection (behaviour-preserving default); 'value' re-ranks by mean value.
    GRAPH_MCTS_CANDIDATE_SCORER: Literal["identity", "value"] = Field(
        # The default lives in constants.py (str-typed); cast narrows it to the field's Literal so
        # `mypy src/` accepts it without a context-dependent type: ignore.
        default=cast(Literal["identity", "value"], DEFAULT_CANDIDATE_SCORER),
        description=(
            "Candidate scorer for the MCTS node: 'identity' preserves the engine's MAX_VISITS "
            "selection (default); 'value' re-ranks candidates by mean value."
        ),
    )


@lru_cache(maxsize=1)
def get_graph_hardening_settings() -> GraphHardeningSettings:
    """Return the process-wide graph-hardening settings (cached)."""
    return GraphHardeningSettings()


def reset_graph_hardening_settings() -> None:
    """Clear the cached settings (test helper)."""
    get_graph_hardening_settings.cache_clear()
