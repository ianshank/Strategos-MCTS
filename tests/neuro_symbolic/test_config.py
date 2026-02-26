"""
Unit tests for neuro-symbolic configuration module.

Tests:
- Configuration dataclass initialization
- Validation and normalization
- Preset configurations
- Serialization/deserialization
- Environment variable handling

Best Practices 2025:
- Property-based testing with hypothesis
- Parameterized tests for coverage
- Mocking for external dependencies
"""

import os
import tempfile
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

from src.neuro_symbolic.config import (
    ConstraintConfig,
    ConstraintEnforcement,
    LogicEngineConfig,
    NeuralEmbeddingConfig,
    NeuroSymbolicConfig,
    ProofConfig,
    ProofStrategy,
    SolverBackend,
    SymbolicAgentConfig,
    get_default_config,
    get_high_precision_config,
    get_hybrid_mcts_config,
    get_low_latency_config,
)


class TestLogicEngineConfig:
    """Tests for LogicEngineConfig."""

    def test_default_initialization(self):
        """Test default values are set correctly."""
        config = LogicEngineConfig()

        assert config.backend == SolverBackend.Z3
        assert config.fallback_backend == SolverBackend.SYMPY
        assert config.solver_timeout_ms > 0
        assert config.max_proof_depth > 0
        assert config.proof_strategy == ProofStrategy.BIDIRECTIONAL
        assert config.enable_memoization is True

    def test_custom_initialization(self):
        """Test custom values are respected."""
        config = LogicEngineConfig(
            backend=SolverBackend.PROLOG,
            solver_timeout_ms=1000,
            max_proof_depth=20,
            proof_strategy=ProofStrategy.BACKWARD_CHAINING,
        )

        assert config.backend == SolverBackend.PROLOG
        assert config.solver_timeout_ms == 1000
        assert config.max_proof_depth == 20
        assert config.proof_strategy == ProofStrategy.BACKWARD_CHAINING

    @pytest.mark.parametrize(
        "env_var,value,attr",
        [
            ("NEURO_SYMBOLIC_SOLVER_TIMEOUT_MS", "2000", "solver_timeout_ms"),
            ("NEURO_SYMBOLIC_MAX_PROOF_DEPTH", "100", "max_proof_depth"),
            ("NEURO_SYMBOLIC_CACHE_SIZE", "5000", "cache_size"),
        ],
    )
    def test_environment_variable_override(self, env_var, value, attr):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {env_var: value}):
            config = LogicEngineConfig()
            assert getattr(config, attr) == int(value)


class TestConstraintConfig:
    """Tests for ConstraintConfig."""

    def test_default_enforcement(self):
        """Test default constraint enforcement is HARD."""
        config = ConstraintConfig()
        assert config.default_enforcement == ConstraintEnforcement.HARD

    def test_all_enforcement_types(self):
        """Test all enforcement types can be set."""
        for enforcement in ConstraintEnforcement:
            config = ConstraintConfig(default_enforcement=enforcement)
            assert config.default_enforcement == enforcement

    @pytest.mark.property
    @given(
        penalty_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_satisfaction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_property_valid_threshold_ranges(
        self,
        penalty_weight: float,
        min_satisfaction: float,
    ):
        """Property: Threshold values should be valid floats."""
        config = ConstraintConfig(
            soft_constraint_penalty_weight=penalty_weight,
            min_satisfaction_ratio=min_satisfaction,
        )
        assert isinstance(config.soft_constraint_penalty_weight, float)
        assert isinstance(config.min_satisfaction_ratio, float)


class TestSymbolicAgentConfig:
    """Tests for SymbolicAgentConfig."""

    def test_weight_normalization(self):
        """Test that confidence weights are normalized to sum to 1."""
        config = SymbolicAgentConfig(
            neural_confidence_weight=0.6,
            symbolic_confidence_weight=0.8,
        )

        total = config.neural_confidence_weight + config.symbolic_confidence_weight
        assert abs(total - 1.0) < 1e-6

    def test_fallback_defaults(self):
        """Test neural fallback defaults."""
        config = SymbolicAgentConfig()

        assert config.fallback_to_neural is True
        assert 0.0 <= config.neural_fallback_confidence_threshold <= 1.0

    @pytest.mark.property
    @given(
        neural_w=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        symbolic_w=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_property_weights_always_normalize(self, neural_w: float, symbolic_w: float):
        """Property: Weights should always normalize to 1.0."""
        config = SymbolicAgentConfig(
            neural_confidence_weight=neural_w,
            symbolic_confidence_weight=symbolic_w,
        )
        total = config.neural_confidence_weight + config.symbolic_confidence_weight
        assert abs(total - 1.0) < 1e-6


class TestNeuroSymbolicConfig:
    """Tests for master NeuroSymbolicConfig."""

    def test_nested_configs_initialized(self):
        """Test that all nested configs are initialized."""
        config = NeuroSymbolicConfig()

        assert isinstance(config.logic_engine, LogicEngineConfig)
        assert isinstance(config.constraints, ConstraintConfig)
        assert isinstance(config.proof, ProofConfig)
        assert isinstance(config.agent, SymbolicAgentConfig)
        assert isinstance(config.embedding, NeuralEmbeddingConfig)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_device_fallback_to_cpu(self):
        """Test device falls back to CPU when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            config = NeuroSymbolicConfig(device="cuda:0")
            assert config.device == "cpu"

    def test_device_fallback_to_cpu_no_torch(self):
        """Test device is CPU when torch not available."""
        # When torch is not available, device should always be cpu
        config = NeuroSymbolicConfig(device="cuda:0")
        if not TORCH_AVAILABLE:
            assert config.device == "cpu"

    def test_to_dict_serialization(self):
        """Test configuration can be serialized to dict."""
        config = NeuroSymbolicConfig()
        config_dict = config.to_dict()

        assert "logic_engine" in config_dict
        assert "constraints" in config_dict
        assert "proof" in config_dict
        assert "agent" in config_dict
        assert "embedding" in config_dict
        assert "device" in config_dict

    def test_from_dict_deserialization(self):
        """Test configuration can be deserialized from dict."""
        original = NeuroSymbolicConfig()
        original.logic_engine.solver_timeout_ms = 3000
        original.constraints.min_satisfaction_ratio = 0.9

        config_dict = original.to_dict()
        restored = NeuroSymbolicConfig.from_dict(config_dict)

        assert restored.logic_engine.solver_timeout_ms == 3000
        assert restored.constraints.min_satisfaction_ratio == 0.9

    def test_save_and_load(self):
        """Test configuration can be saved and loaded from file."""
        config = NeuroSymbolicConfig()
        config.seed = 12345

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.save(f.name)
            loaded = NeuroSymbolicConfig.load(f.name)

        assert loaded.seed == 12345
        os.unlink(f.name)


class TestPresetConfigurations:
    """Tests for preset configurations."""

    def test_default_config(self):
        """Test default config is valid."""
        config = get_default_config()

        assert isinstance(config, NeuroSymbolicConfig)
        assert config.device in ("cpu", "cuda", "cuda:0")

    def test_high_precision_config(self):
        """Test high precision config has stricter settings."""
        config = get_high_precision_config()

        assert config.constraints.default_enforcement == ConstraintEnforcement.HARD
        assert config.constraints.min_satisfaction_ratio >= 0.9
        assert config.proof.min_proof_confidence >= 0.8
        assert config.agent.symbolic_confidence_weight > config.agent.neural_confidence_weight

    def test_low_latency_config(self):
        """Test low latency config has faster settings."""
        default = get_default_config()
        low_latency = get_low_latency_config()

        assert low_latency.logic_engine.solver_timeout_ms < default.logic_engine.solver_timeout_ms
        assert low_latency.logic_engine.max_proof_depth < default.logic_engine.max_proof_depth
        assert low_latency.proof.generate_proof_trees is False

    def test_hybrid_mcts_config(self):
        """Test hybrid MCTS config is balanced."""
        config = get_hybrid_mcts_config()

        assert config.logic_engine.enable_memoization is True
        assert config.logic_engine.parallel_proof_search is True
        assert config.constraints.precompile_constraints is True
        # Balanced weights
        weight_diff = abs(config.agent.neural_confidence_weight - config.agent.symbolic_confidence_weight)
        assert weight_diff < 0.3

    def test_all_presets_are_valid(self):
        """Test all preset configs can be serialized and deserialized."""
        presets = [
            get_default_config,
            get_high_precision_config,
            get_low_latency_config,
            get_hybrid_mcts_config,
        ]

        for preset_fn in presets:
            config = preset_fn()
            config_dict = config.to_dict()
            restored = NeuroSymbolicConfig.from_dict(config_dict)

            assert isinstance(restored, NeuroSymbolicConfig)


class TestSolverBackendEnum:
    """Tests for SolverBackend enum."""

    def test_all_backends_defined(self):
        """Test all expected backends are defined."""
        expected = {"Z3", "PROLOG", "DATALOG", "SYMPY", "CUSTOM"}
        actual = {b.name for b in SolverBackend}
        assert expected == actual

    def test_backend_from_string(self):
        """Test backend can be created from string."""
        assert SolverBackend["Z3"] == SolverBackend.Z3
        assert SolverBackend["PROLOG"] == SolverBackend.PROLOG


class TestProofStrategyEnum:
    """Tests for ProofStrategy enum."""

    def test_all_strategies_defined(self):
        """Test all expected strategies are defined."""
        expected = {"FORWARD_CHAINING", "BACKWARD_CHAINING", "BIDIRECTIONAL", "NEURAL_GUIDED"}
        actual = {s.name for s in ProofStrategy}
        assert expected == actual
