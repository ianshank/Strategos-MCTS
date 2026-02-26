"""Tests for LLM-Guided MCTS Configuration."""

import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.config.settings import LLMProvider
from src.framework.mcts.llm_guided.config import (
    BALANCED_LLM_MCTS_CONFIG,
    DATA_COLLECTION_CONFIG,
    DEFAULT_LLM_MCTS_CONFIG,
    FAST_LLM_MCTS_CONFIG,
    THOROUGH_LLM_MCTS_CONFIG,
    GeneratorConfig,
    LLMGuidedMCTSConfig,
    LLMGuidedMCTSPreset,
    ReflectorConfig,
    create_llm_mcts_preset,
)


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GeneratorConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.num_variants == 3
        assert config.top_p == 0.95
        assert config.include_test_errors is True
        assert config.include_previous_attempts is True
        assert config.max_previous_attempts == 3
        assert config.track_tokens is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GeneratorConfig(
            model="gpt-4o-mini",
            temperature=0.5,
            num_variants=5,
        )
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.5
        assert config.num_variants == 5

    def test_validation_temperature_bounds(self):
        """Test temperature validation."""
        # Valid temperature
        config = GeneratorConfig(temperature=1.5)
        config.validate()  # Should not raise

        # Invalid temperature
        with pytest.raises(ValueError, match="temperature must be in"):
            config = GeneratorConfig(temperature=2.5)
            config.validate()

    def test_validation_max_tokens_bounds(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError, match="max_tokens must be in"):
            config = GeneratorConfig(max_tokens=10000)
            config.validate()

    def test_validation_num_variants_bounds(self):
        """Test num_variants validation."""
        with pytest.raises(ValueError, match="num_variants must be in"):
            config = GeneratorConfig(num_variants=0)
            config.validate()


class TestReflectorConfig:
    """Tests for ReflectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReflectorConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_tokens == 1000
        assert config.include_test_results is True
        assert config.include_code_context is True
        assert config.min_score == 0.0
        assert config.max_score == 1.0

    def test_validation_score_bounds(self):
        """Test score bounds validation."""
        with pytest.raises(ValueError, match="min_score must be < max_score"):
            config = ReflectorConfig(min_score=1.0, max_score=0.5)
            config.validate()


class TestLLMGuidedMCTSConfig:
    """Tests for LLMGuidedMCTSConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMGuidedMCTSConfig()
        assert config.num_iterations == 30
        assert config.seed == 42
        assert config.exploration_weight == 1.414
        assert config.max_depth == 10
        assert config.max_children == 5
        assert config.early_termination_on_solution is True
        assert config.solution_confidence_threshold == 0.95
        assert config.collect_training_data is True
        assert config.execution_timeout_seconds == 5.0
        assert config.llm_provider == LLMProvider.OPENAI
        assert config.name == "default"

    def test_nested_configs(self):
        """Test nested generator and reflector configs."""
        config = LLMGuidedMCTSConfig()
        assert isinstance(config.generator_config, GeneratorConfig)
        assert isinstance(config.reflector_config, ReflectorConfig)

    def test_validation_iterations_bounds(self):
        """Test num_iterations validation."""
        with pytest.raises(ValueError, match="num_iterations must be in"):
            LLMGuidedMCTSConfig(num_iterations=0)

    def test_validation_exploration_weight_bounds(self):
        """Test exploration_weight validation."""
        with pytest.raises(ValueError, match="exploration_weight must be in"):
            LLMGuidedMCTSConfig(exploration_weight=15.0)

    def test_validation_max_depth_bounds(self):
        """Test max_depth validation."""
        with pytest.raises(ValueError, match="max_depth must be in"):
            LLMGuidedMCTSConfig(max_depth=100)

    def test_validation_timeout_bounds(self):
        """Test execution_timeout_seconds validation."""
        with pytest.raises(ValueError, match="execution_timeout_seconds must be in"):
            LLMGuidedMCTSConfig(execution_timeout_seconds=0.01)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = LLMGuidedMCTSConfig(name="test")
        d = config.to_dict()

        assert d["name"] == "test"
        assert d["num_iterations"] == 30
        assert d["llm_provider"] == "openai"
        assert "generator_config" in d
        assert "reflector_config" in d

    def test_to_json(self):
        """Test serialization to JSON."""
        config = LLMGuidedMCTSConfig(name="test")
        json_str = config.to_json()

        data = json.loads(json_str)
        assert data["name"] == "test"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "name": "from_dict_test",
            "num_iterations": 50,
            "llm_provider": "anthropic",
            "generator_config": {"model": "claude-3", "temperature": 0.5},
            "reflector_config": {"model": "claude-3", "temperature": 0.2},
        }

        config = LLMGuidedMCTSConfig.from_dict(d)
        assert config.name == "from_dict_test"
        assert config.num_iterations == 50
        assert config.llm_provider == LLMProvider.ANTHROPIC
        assert config.generator_config.model == "claude-3"

    def test_from_json(self):
        """Test deserialization from JSON."""
        json_str = '{"name": "json_test", "num_iterations": 25}'
        config = LLMGuidedMCTSConfig.from_json(json_str)
        assert config.name == "json_test"
        assert config.num_iterations == 25

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config = LLMGuidedMCTSConfig(name="save_load_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)

            assert path.exists()

            loaded = LLMGuidedMCTSConfig.load(path)
            assert loaded.name == "save_load_test"
            assert loaded.num_iterations == config.num_iterations

    def test_copy_with_overrides(self):
        """Test copying configuration with overrides."""
        config = LLMGuidedMCTSConfig(name="original", num_iterations=30)
        copied = config.copy(name="copied", num_iterations=50)

        assert copied.name == "copied"
        assert copied.num_iterations == 50
        # Original unchanged
        assert config.name == "original"
        assert config.num_iterations == 30


class TestPresets:
    """Tests for configuration presets."""

    def test_fast_preset(self):
        """Test FAST preset configuration."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.FAST)
        assert config.name == "fast"
        assert config.num_iterations == 10
        assert config.max_depth == 5
        assert config.generator_config.model == "gpt-4o-mini"
        assert config.collect_training_data is False

    def test_balanced_preset(self):
        """Test BALANCED preset configuration."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.BALANCED)
        assert config.name == "balanced"
        assert config.num_iterations == 30
        assert config.collect_training_data is True

    def test_thorough_preset(self):
        """Test THOROUGH preset configuration."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.THOROUGH)
        assert config.name == "thorough"
        assert config.num_iterations == 100
        assert config.exploration_weight == 2.0
        assert config.verbose is True

    def test_data_collection_preset(self):
        """Test DATA_COLLECTION preset configuration."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.DATA_COLLECTION)
        assert config.name == "data_collection"
        assert config.early_termination_on_solution is False
        assert config.save_mcts_policy is True

    def test_benchmark_preset(self):
        """Test BENCHMARK preset configuration."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.BENCHMARK)
        assert config.name == "benchmark"
        assert config.execution_timeout_seconds == 10.0

    def test_default_config_instances(self):
        """Test default config instances."""
        assert DEFAULT_LLM_MCTS_CONFIG.name == "default"
        assert FAST_LLM_MCTS_CONFIG.name == "fast"
        assert BALANCED_LLM_MCTS_CONFIG.name == "balanced"
        assert THOROUGH_LLM_MCTS_CONFIG.name == "thorough"
        assert DATA_COLLECTION_CONFIG.name == "data_collection"

    def test_invalid_preset(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_llm_mcts_preset("invalid")  # type: ignore
