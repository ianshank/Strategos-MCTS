"""
Tests for benchmark configuration settings.

Validates Pydantic Settings v2 configuration loading, validation,
defaults, and environment variable override behavior.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.benchmark.config.benchmark_settings import (
    ADKBenchmarkConfig,
    BenchmarkRunConfig,
    BenchmarkSettings,
    CostConfig,
    LangGraphBenchmarkConfig,
    ReportConfig,
    ScoringConfig,
    get_benchmark_settings,
    reset_benchmark_settings,
)


class TestBenchmarkRunConfig:
    """Test BenchmarkRunConfig defaults and validation."""

    def test_defaults(self) -> None:
        config = BenchmarkRunConfig()
        assert config.num_iterations == 1
        assert config.task_timeout_seconds == 120.0
        assert config.max_concurrent_tasks == 1
        assert config.retry_on_failure is True
        assert config.max_retries == 3
        assert config.seed == 42

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"BENCHMARK_RUN_NUM_ITERATIONS": "5", "BENCHMARK_RUN_SEED": "123"}):
            config = BenchmarkRunConfig()
            assert config.num_iterations == 5
            assert config.seed == 123

    def test_validation_min(self) -> None:
        with pytest.raises(ValueError):
            BenchmarkRunConfig(num_iterations=0)

    def test_validation_max(self) -> None:
        with pytest.raises(ValueError):
            BenchmarkRunConfig(num_iterations=101)


class TestScoringConfig:
    """Test ScoringConfig defaults and validation."""

    def test_defaults(self) -> None:
        config = ScoringConfig()
        assert config.enabled is True
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo-preview"
        assert config.temperature == 0.1
        assert config.min_score == 1.0
        assert config.max_score == 5.0

    def test_scoring_dimensions(self) -> None:
        config = ScoringConfig()
        assert "task_completion" in config.scoring_dimensions
        assert "reasoning_depth" in config.scoring_dimensions
        assert "accuracy" in config.scoring_dimensions
        assert "coherence" in config.scoring_dimensions

    def test_invalid_provider(self) -> None:
        with pytest.raises(ValueError, match="Scoring provider must be one of"):
            ScoringConfig(provider="invalid")

    def test_valid_providers(self) -> None:
        for provider in ["openai", "anthropic", "google"]:
            config = ScoringConfig(provider=provider)
            assert config.provider == provider


class TestCostConfig:
    """Test CostConfig pricing and rate lookup."""

    def test_defaults(self) -> None:
        config = CostConfig()
        assert config.openai_input_per_1k > 0
        assert config.openai_output_per_1k > 0
        assert config.anthropic_input_per_1k > 0
        assert config.gemini_pro_input_per_1k > 0
        assert config.gemini_flash_input_per_1k > 0

    def test_get_rates_openai(self) -> None:
        config = CostConfig()
        input_rate, output_rate = config.get_rates("openai", "gpt-4")
        assert input_rate == config.openai_input_per_1k
        assert output_rate == config.openai_output_per_1k

    def test_get_rates_anthropic(self) -> None:
        config = CostConfig()
        input_rate, output_rate = config.get_rates("anthropic", "claude-3")
        assert input_rate == config.anthropic_input_per_1k
        assert output_rate == config.anthropic_output_per_1k

    def test_get_rates_gemini_pro(self) -> None:
        config = CostConfig()
        input_rate, output_rate = config.get_rates("google", "gemini-2.5-pro")
        assert input_rate == config.gemini_pro_input_per_1k
        assert output_rate == config.gemini_pro_output_per_1k

    def test_get_rates_gemini_flash(self) -> None:
        config = CostConfig()
        input_rate, output_rate = config.get_rates("google", "gemini-2.5-flash")
        assert input_rate == config.gemini_flash_input_per_1k
        assert output_rate == config.gemini_flash_output_per_1k

    def test_get_rates_unknown_defaults_to_openai(self) -> None:
        config = CostConfig()
        input_rate, output_rate = config.get_rates("unknown", "some-model")
        assert input_rate == config.openai_input_per_1k
        assert output_rate == config.openai_output_per_1k


class TestLangGraphBenchmarkConfig:
    """Test LangGraph-specific config."""

    def test_defaults(self) -> None:
        config = LangGraphBenchmarkConfig()
        assert config.enabled is True
        assert config.mcts_iterations == 50
        assert config.mcts_exploration_weight == pytest.approx(1.414)
        assert config.use_parallel_agents is True
        assert config.consensus_threshold == 0.75


class TestADKBenchmarkConfig:
    """Test ADK-specific config."""

    def test_defaults(self) -> None:
        config = ADKBenchmarkConfig()
        assert config.enabled is True
        assert config.coordinator_model == "gemini-2.5-pro"
        assert config.sub_agent_model == "gemini-2.5-flash"
        assert config.google_api_key is None
        assert config.max_agent_turns == 10

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"BENCHMARK_ADK_COORDINATOR_MODEL": "gemini-1.5-pro"}):
            config = ADKBenchmarkConfig()
            assert config.coordinator_model == "gemini-1.5-pro"


class TestReportConfig:
    """Test report configuration."""

    def test_defaults(self) -> None:
        config = ReportConfig()
        assert config.output_dir == "benchmark_output"
        assert config.report_filename == "benchmark_report.md"
        assert config.results_filename == "benchmark_results.json"
        assert config.include_raw_responses is False
        assert config.include_agent_traces is False


class TestBenchmarkSettings:
    """Test master benchmark settings."""

    def setup_method(self) -> None:
        reset_benchmark_settings()

    def teardown_method(self) -> None:
        reset_benchmark_settings()

    def test_defaults(self) -> None:
        settings = BenchmarkSettings()
        assert settings.benchmark_enabled is True
        assert settings.benchmark_name == "langgraph_mcts_vs_adk"

    def test_nested_configs_lazy_init(self) -> None:
        settings = BenchmarkSettings()
        # Accessing nested configs should create them lazily
        assert isinstance(settings.run, BenchmarkRunConfig)
        assert isinstance(settings.scoring, ScoringConfig)
        assert isinstance(settings.cost, CostConfig)
        assert isinstance(settings.langgraph, LangGraphBenchmarkConfig)
        assert isinstance(settings.adk, ADKBenchmarkConfig)
        assert isinstance(settings.report, ReportConfig)

    def test_safe_dict_no_secrets(self) -> None:
        settings = BenchmarkSettings()
        safe = settings.safe_dict()
        assert "benchmark_enabled" in safe
        assert "benchmark_name" in safe
        # API key should be masked
        assert safe["adk"]["google_api_key"] is None

    def test_singleton_pattern(self) -> None:
        reset_benchmark_settings()
        s1 = get_benchmark_settings()
        s2 = get_benchmark_settings()
        assert s1 is s2

    def test_reset_clears_singleton(self) -> None:
        s1 = get_benchmark_settings()
        reset_benchmark_settings()
        s2 = get_benchmark_settings()
        assert s1 is not s2
