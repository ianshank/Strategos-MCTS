"""
Benchmark configuration using Pydantic Settings v2.

All configuration values are loaded from environment variables,
following the no-hardcoded-values principle from CLAUDE.md.

Environment Variable Prefixes:
- BENCHMARK_*: Global benchmark settings
- BENCHMARK_RUN_*: Run-level configuration
- BENCHMARK_SCORING_*: LLM-as-judge configuration
- BENCHMARK_COST_*: Cost estimation configuration
- BENCHMARK_LG_*: LangGraph-specific settings
- BENCHMARK_ADK_*: ADK-specific settings
- BENCHMARK_REPORT_*: Report generation settings
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class BenchmarkRunConfig(BaseSettings):
    """
    Run-level benchmark configuration.

    Controls execution parameters like iterations, timeout, and parallelism.
    """

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_RUN_",
        extra="ignore",
    )

    num_iterations: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of iterations per task per system",
    )
    task_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600.0,
        description="Timeout for individual task execution",
    )
    max_concurrent_tasks: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum concurrent task executions",
    )
    retry_on_failure: bool = Field(
        default=True,
        description="Retry failed task executions",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts per task",
    )
    retry_backoff_base_seconds: float = Field(
        default=2.0,
        gt=0,
        le=30.0,
        description="Base delay for exponential backoff retries",
    )
    warmup_runs: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of warmup runs before actual benchmark",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class ScoringConfig(BaseSettings):
    """
    LLM-as-judge scoring configuration.

    Controls the model, temperature, and retry behavior for scoring.
    """

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_SCORING_",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable LLM-as-judge scoring",
    )
    provider: str = Field(
        default="openai",
        description="LLM provider for scoring (openai, anthropic, google)",
    )
    model: str = Field(
        default="gpt-4-turbo-preview",
        description="Model to use for scoring",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for scoring (low = more deterministic)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Maximum tokens for scoring response",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for scoring",
    )
    retry_backoff_base_seconds: float = Field(
        default=2.0,
        gt=0,
        le=30.0,
        description="Base delay for exponential backoff retries",
    )
    min_score: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum valid score",
    )
    max_score: float = Field(
        default=5.0,
        le=10.0,
        description="Maximum valid score",
    )
    scoring_dimensions: list[str] = Field(
        default_factory=lambda: [
            "task_completion",
            "reasoning_depth",
            "accuracy",
            "coherence",
        ],
        description="Dimensions to score",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid = ["openai", "anthropic", "google"]
        if v.lower() not in valid:
            raise ValueError(f"Scoring provider must be one of {valid}")
        return v.lower()


class CostConfig(BaseSettings):
    """
    Cost estimation configuration.

    Token pricing per provider and model for cost tracking.
    """

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_COST_",
        extra="ignore",
    )

    # OpenAI pricing (per 1K tokens, USD)
    openai_input_per_1k: float = Field(
        default=0.01,
        ge=0.0,
        description="OpenAI input token cost per 1K tokens (USD)",
    )
    openai_output_per_1k: float = Field(
        default=0.03,
        ge=0.0,
        description="OpenAI output token cost per 1K tokens (USD)",
    )

    # Anthropic pricing (per 1K tokens, USD)
    anthropic_input_per_1k: float = Field(
        default=0.008,
        ge=0.0,
        description="Anthropic input token cost per 1K tokens (USD)",
    )
    anthropic_output_per_1k: float = Field(
        default=0.024,
        ge=0.0,
        description="Anthropic output token cost per 1K tokens (USD)",
    )

    # Google Gemini pricing (per 1K tokens, USD)
    gemini_pro_input_per_1k: float = Field(
        default=0.00125,
        ge=0.0,
        description="Gemini Pro input token cost per 1K tokens (USD)",
    )
    gemini_pro_output_per_1k: float = Field(
        default=0.005,
        ge=0.0,
        description="Gemini Pro output token cost per 1K tokens (USD)",
    )
    gemini_flash_input_per_1k: float = Field(
        default=0.000075,
        ge=0.0,
        description="Gemini Flash input token cost per 1K tokens (USD)",
    )
    gemini_flash_output_per_1k: float = Field(
        default=0.0003,
        ge=0.0,
        description="Gemini Flash output token cost per 1K tokens (USD)",
    )

    def get_rates(self, provider: str, model: str) -> tuple[float, float]:
        """
        Get (input_rate, output_rate) per 1K tokens for a provider/model.

        Args:
            provider: LLM provider name
            model: Model identifier

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k)
        """
        if provider == "google" or provider == "gemini":
            if "flash" in model.lower():
                return (self.gemini_flash_input_per_1k, self.gemini_flash_output_per_1k)
            return (self.gemini_pro_input_per_1k, self.gemini_pro_output_per_1k)
        elif provider == "anthropic":
            return (self.anthropic_input_per_1k, self.anthropic_output_per_1k)
        # Default to OpenAI pricing
        return (self.openai_input_per_1k, self.openai_output_per_1k)


class LangGraphBenchmarkConfig(BaseSettings):
    """LangGraph MCTS-specific benchmark configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_LG_",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable LangGraph MCTS benchmarking",
    )
    mcts_iterations: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="MCTS iterations for benchmark runs",
    )
    mcts_exploration_weight: float = Field(
        default=1.414,
        ge=0.0,
        le=10.0,
        description="UCB1 exploration constant",
    )
    use_parallel_agents: bool = Field(
        default=True,
        description="Enable parallel agent execution",
    )
    consensus_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Consensus threshold for agent agreement",
    )
    max_graph_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum LangGraph iteration cycles",
    )


class ADKBenchmarkConfig(BaseSettings):
    """Google ADK-specific benchmark configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_ADK_",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable ADK benchmarking",
    )
    coordinator_model: str = Field(
        default="gemini-2.5-pro",
        description="Model for ADK coordinator agent",
    )
    sub_agent_model: str = Field(
        default="gemini-2.5-flash",
        description="Model for ADK sub-agents",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        description="Google API key for Gemini models",
    )
    google_project_id: str | None = Field(
        default=None,
        description="Google Cloud project ID",
    )
    google_region: str = Field(
        default="us-central1",
        description="Google Cloud region",
    )
    max_agent_turns: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum turns for ADK agent execution",
    )
    use_vertex_search: bool = Field(
        default=False,
        description="Enable Vertex AI Search grounding",
    )
    vertex_search_datastore: str | None = Field(
        default=None,
        description="Vertex AI Search datastore path",
    )


class ReportConfig(BaseSettings):
    """Report generation configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_REPORT_",
        extra="ignore",
    )

    output_dir: str = Field(
        default="benchmark_output",
        description="Directory for benchmark output files",
    )
    report_filename: str = Field(
        default="benchmark_report.md",
        description="Filename for the comparison report",
    )
    results_filename: str = Field(
        default="benchmark_results.json",
        description="Filename for raw results JSON",
    )
    include_raw_responses: bool = Field(
        default=False,
        description="Include full raw responses in report",
    )
    include_agent_traces: bool = Field(
        default=False,
        description="Include agent trace data in report",
    )
    max_response_preview_length: int = Field(
        default=500,
        ge=0,
        le=10000,
        description="Maximum characters of response to include in report preview",
    )


class BenchmarkSettings(BaseSettings):
    """
    Master benchmark configuration.

    Centralizes all benchmark configurations with validation.
    Loaded from .env file and environment variables.

    Example:
        >>> settings = get_benchmark_settings()
        >>> settings.run.num_iterations
        1
        >>> settings.scoring.model
        'gpt-4-turbo-preview'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Global benchmark settings
    benchmark_enabled: bool = Field(
        default=True,
        alias="BENCHMARK_ENABLED",
        description="Enable benchmark framework",
    )
    benchmark_name: str = Field(
        default="langgraph_mcts_vs_adk",
        alias="BENCHMARK_NAME",
        description="Name for this benchmark suite",
    )

    # Nested configurations (created lazily for performance)
    _run: BenchmarkRunConfig | None = None
    _scoring: ScoringConfig | None = None
    _cost: CostConfig | None = None
    _langgraph: LangGraphBenchmarkConfig | None = None
    _adk: ADKBenchmarkConfig | None = None
    _report: ReportConfig | None = None

    @property
    def run(self) -> BenchmarkRunConfig:
        """Get run configuration."""
        if self._run is None:
            self._run = BenchmarkRunConfig()
        return self._run

    @property
    def scoring(self) -> ScoringConfig:
        """Get scoring configuration."""
        if self._scoring is None:
            self._scoring = ScoringConfig()
        return self._scoring

    @property
    def cost(self) -> CostConfig:
        """Get cost configuration."""
        if self._cost is None:
            self._cost = CostConfig()
        return self._cost

    @property
    def langgraph(self) -> LangGraphBenchmarkConfig:
        """Get LangGraph configuration."""
        if self._langgraph is None:
            self._langgraph = LangGraphBenchmarkConfig()
        return self._langgraph

    @property
    def adk(self) -> ADKBenchmarkConfig:
        """Get ADK configuration."""
        if self._adk is None:
            self._adk = ADKBenchmarkConfig()
        return self._adk

    @property
    def report(self) -> ReportConfig:
        """Get report configuration."""
        if self._report is None:
            self._report = ReportConfig()
        return self._report

    def safe_dict(self) -> dict[str, Any]:
        """Return configuration with secrets masked. Safe for logging."""
        return {
            "benchmark_enabled": self.benchmark_enabled,
            "benchmark_name": self.benchmark_name,
            "run": {
                "num_iterations": self.run.num_iterations,
                "task_timeout_seconds": self.run.task_timeout_seconds,
                "max_concurrent_tasks": self.run.max_concurrent_tasks,
                "seed": self.run.seed,
            },
            "scoring": {
                "enabled": self.scoring.enabled,
                "provider": self.scoring.provider,
                "model": self.scoring.model,
            },
            "langgraph": {
                "enabled": self.langgraph.enabled,
                "mcts_iterations": self.langgraph.mcts_iterations,
            },
            "adk": {
                "enabled": self.adk.enabled,
                "coordinator_model": self.adk.coordinator_model,
                "google_api_key": "***" if self.adk.google_api_key else None,
            },
        }


# Singleton pattern (consistent with existing codebase)
_benchmark_settings: BenchmarkSettings | None = None


def get_benchmark_settings() -> BenchmarkSettings:
    """
    Get global benchmark settings instance.

    Returns:
        BenchmarkSettings instance (singleton)
    """
    global _benchmark_settings
    if _benchmark_settings is None:
        _benchmark_settings = BenchmarkSettings()
        logger.info("Benchmark settings loaded", extra=_benchmark_settings.safe_dict())
    return _benchmark_settings


def reset_benchmark_settings() -> None:
    """
    Reset the global settings instance.

    Useful for testing and configuration reloading.
    """
    global _benchmark_settings
    _benchmark_settings = None
    logger.debug("Benchmark settings reset")
