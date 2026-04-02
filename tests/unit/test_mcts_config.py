"""Unit tests for MCTS Configuration module (src/framework/mcts/config.py).

Tests cover:
- MCTSConfig dataclass creation and validation
- Parameter bounds validation
- Serialization/deserialization (to_dict, to_json, from_dict, from_json)
- Preset configurations (fast, balanced, thorough, exploration_heavy, exploitation_heavy)
- Config copy with overrides
"""

from __future__ import annotations

import json

import pytest

from src.framework.mcts.config import (
    BALANCED_CONFIG,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    THOROUGH_CONFIG,
    ConfigPreset,
    MCTSConfig,
    create_preset_config,
)
from src.framework.mcts.policies import SelectionPolicy


@pytest.mark.unit
class TestMCTSConfigDefaults:
    """Test MCTSConfig default values."""

    def test_default_config_creation(self):
        config = MCTSConfig()
        assert config.num_iterations == 100
        assert config.seed == 42
        assert config.exploration_weight == 1.414
        assert config.progressive_widening_k == 1.0
        assert config.progressive_widening_alpha == 0.5
        assert config.max_rollout_depth == 10
        assert config.rollout_policy == "hybrid"
        assert config.selection_policy == SelectionPolicy.MAX_VISITS
        assert config.max_parallel_rollouts == 4
        assert config.enable_cache is True
        assert config.cache_size_limit == 10000
        assert config.max_tree_depth == 20
        assert config.max_children_per_node == 50
        assert config.early_termination_threshold == 0.95
        assert config.min_iterations_before_termination == 50
        assert config.early_stop_threshold == 0.01
        assert config.early_stop_patience == 10
        assert config.min_value == 0.0
        assert config.max_value == 1.0
        assert config.name == "default"
        assert config.description == ""

    def test_custom_config(self):
        config = MCTSConfig(
            num_iterations=200,
            seed=123,
            exploration_weight=2.0,
            name="custom",
            description="Test config",
        )
        assert config.num_iterations == 200
        assert config.seed == 123
        assert config.exploration_weight == 2.0
        assert config.name == "custom"
        assert config.description == "Test config"


@pytest.mark.unit
class TestMCTSConfigValidation:
    """Test MCTSConfig validation."""

    def test_invalid_num_iterations_zero(self):
        with pytest.raises(ValueError, match="num_iterations"):
            MCTSConfig(num_iterations=0)

    def test_invalid_num_iterations_too_large(self):
        with pytest.raises(ValueError, match="num_iterations"):
            MCTSConfig(num_iterations=200000)

    def test_invalid_exploration_weight_negative(self):
        with pytest.raises(ValueError, match="exploration_weight"):
            MCTSConfig(exploration_weight=-1)

    def test_invalid_exploration_weight_too_large(self):
        with pytest.raises(ValueError, match="exploration_weight"):
            MCTSConfig(exploration_weight=15)

    def test_invalid_progressive_widening_k(self):
        with pytest.raises(ValueError, match="progressive_widening_k"):
            MCTSConfig(progressive_widening_k=0)

    def test_invalid_progressive_widening_alpha_zero(self):
        with pytest.raises(ValueError, match="progressive_widening_alpha"):
            MCTSConfig(progressive_widening_alpha=0)

    def test_invalid_progressive_widening_alpha_one(self):
        with pytest.raises(ValueError, match="progressive_widening_alpha"):
            MCTSConfig(progressive_widening_alpha=1)

    def test_invalid_rollout_depth(self):
        with pytest.raises(ValueError, match="max_rollout_depth"):
            MCTSConfig(max_rollout_depth=0)

    def test_invalid_rollout_policy(self):
        with pytest.raises(ValueError, match="rollout_policy"):
            MCTSConfig(rollout_policy="invalid")

    def test_invalid_parallel_rollouts_zero(self):
        with pytest.raises(ValueError, match="max_parallel_rollouts"):
            MCTSConfig(max_parallel_rollouts=0)

    def test_invalid_parallel_rollouts_too_large(self):
        with pytest.raises(ValueError, match="max_parallel_rollouts"):
            MCTSConfig(max_parallel_rollouts=200)

    def test_invalid_cache_size_negative(self):
        with pytest.raises(ValueError, match="cache_size_limit"):
            MCTSConfig(cache_size_limit=-1)

    def test_invalid_tree_depth(self):
        with pytest.raises(ValueError, match="max_tree_depth"):
            MCTSConfig(max_tree_depth=0)

    def test_invalid_max_children(self):
        with pytest.raises(ValueError, match="max_children_per_node"):
            MCTSConfig(max_children_per_node=0)

    def test_invalid_early_termination_threshold_zero(self):
        with pytest.raises(ValueError, match="early_termination_threshold"):
            MCTSConfig(early_termination_threshold=0)

    def test_invalid_early_termination_threshold_above_one(self):
        with pytest.raises(ValueError, match="early_termination_threshold"):
            MCTSConfig(early_termination_threshold=1.5)

    def test_invalid_min_iterations_before_termination(self):
        with pytest.raises(ValueError, match="min_iterations_before_termination"):
            MCTSConfig(min_iterations_before_termination=0)

    def test_min_iterations_exceeds_num_iterations(self):
        with pytest.raises(ValueError, match="min_iterations_before_termination"):
            MCTSConfig(num_iterations=10, min_iterations_before_termination=20)

    def test_invalid_early_stop_threshold_negative(self):
        with pytest.raises(ValueError, match="early_stop_threshold"):
            MCTSConfig(early_stop_threshold=-0.1)

    def test_invalid_early_stop_patience(self):
        with pytest.raises(ValueError, match="early_stop_patience"):
            MCTSConfig(early_stop_patience=0)

    def test_invalid_value_bounds(self):
        with pytest.raises(ValueError, match="min_value"):
            MCTSConfig(min_value=1.0, max_value=0.0)

    def test_valid_llm_rollout_policy(self):
        config = MCTSConfig(rollout_policy="llm")
        assert config.rollout_policy == "llm"


@pytest.mark.unit
class TestMCTSConfigSerialization:
    """Test MCTSConfig serialization."""

    def test_to_dict(self):
        config = MCTSConfig(name="test")
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["num_iterations"] == 100
        assert d["selection_policy"] == "max_visits"

    def test_to_json(self):
        config = MCTSConfig(name="json_test")
        json_str = config.to_json()
        parsed = json.loads(json_str)
        assert parsed["name"] == "json_test"

    def test_to_json_custom_indent(self):
        config = MCTSConfig()
        json_str = config.to_json(indent=4)
        assert "    " in json_str

    def test_from_dict(self):
        data = {"num_iterations": 50, "seed": 99, "name": "from_dict"}
        config = MCTSConfig.from_dict(data)
        assert config.num_iterations == 50
        assert config.seed == 99
        assert config.name == "from_dict"

    def test_from_dict_with_selection_policy_string(self):
        data = {"selection_policy": "max_value"}
        config = MCTSConfig.from_dict(data)
        assert config.selection_policy == SelectionPolicy.MAX_VALUE

    def test_from_json(self):
        json_str = '{"num_iterations": 75, "name": "from_json"}'
        config = MCTSConfig.from_json(json_str)
        assert config.num_iterations == 75
        assert config.name == "from_json"

    def test_roundtrip_dict(self):
        original = MCTSConfig(
            num_iterations=200,
            exploration_weight=2.0,
            name="roundtrip",
        )
        restored = MCTSConfig.from_dict(original.to_dict())
        assert restored.num_iterations == original.num_iterations
        assert restored.exploration_weight == original.exploration_weight
        assert restored.name == original.name

    def test_roundtrip_json(self):
        original = MCTSConfig(
            num_iterations=300,
            selection_policy=SelectionPolicy.ROBUST_CHILD,
        )
        restored = MCTSConfig.from_json(original.to_json())
        assert restored.num_iterations == original.num_iterations
        assert restored.selection_policy == original.selection_policy


@pytest.mark.unit
class TestMCTSConfigCopy:
    """Test MCTSConfig copy with overrides."""

    def test_copy_no_overrides(self):
        original = MCTSConfig(name="original")
        copied = original.copy()
        assert copied.name == "original"
        assert copied.num_iterations == original.num_iterations

    def test_copy_with_overrides(self):
        original = MCTSConfig(name="original", num_iterations=100)
        copied = original.copy(num_iterations=200, name="modified")
        assert copied.num_iterations == 200
        assert copied.name == "modified"
        # Original unchanged
        assert original.num_iterations == 100

    def test_repr(self):
        config = MCTSConfig(name="test_repr")
        r = repr(config)
        assert "test_repr" in r
        assert "MCTSConfig" in r


@pytest.mark.unit
class TestPresetConfigs:
    """Test preset configurations."""

    def test_fast_preset(self):
        config = create_preset_config(ConfigPreset.FAST)
        assert config.name == "fast"
        assert config.num_iterations == 25
        assert config.max_rollout_depth == 5
        assert config.rollout_policy == "random"

    def test_balanced_preset(self):
        config = create_preset_config(ConfigPreset.BALANCED)
        assert config.name == "balanced"
        assert config.num_iterations == 100
        assert config.rollout_policy == "hybrid"

    def test_thorough_preset(self):
        config = create_preset_config(ConfigPreset.THOROUGH)
        assert config.name == "thorough"
        assert config.num_iterations == 500
        assert config.selection_policy == SelectionPolicy.ROBUST_CHILD

    def test_exploration_heavy_preset(self):
        config = create_preset_config(ConfigPreset.EXPLORATION_HEAVY)
        assert config.name == "exploration_heavy"
        assert config.exploration_weight == 2.5

    def test_exploitation_heavy_preset(self):
        config = create_preset_config(ConfigPreset.EXPLOITATION_HEAVY)
        assert config.name == "exploitation_heavy"
        assert config.exploration_weight == 0.5
        assert config.selection_policy == SelectionPolicy.MAX_VALUE

    def test_all_presets_valid(self):
        """All presets should create valid configs."""
        for preset in ConfigPreset:
            config = create_preset_config(preset)
            # Validation happens in __post_init__, so if we get here it's valid
            assert config.name != ""

    def test_module_level_configs(self):
        """Module-level config constants should be valid."""
        assert DEFAULT_CONFIG.name == "default"
        assert FAST_CONFIG.name == "fast"
        assert BALANCED_CONFIG.name == "balanced"
        assert THOROUGH_CONFIG.name == "thorough"
