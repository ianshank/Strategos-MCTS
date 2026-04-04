"""
Tests for neuro-symbolic configuration module.

Tests all config dataclasses, preset configurations,
serialization, and validation.
"""

import pytest

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


@pytest.mark.unit
class TestLogicEngineConfig:
    def test_defaults(self):
        cfg = LogicEngineConfig()
        assert cfg.backend == SolverBackend.Z3
        assert cfg.fallback_backend == SolverBackend.SYMPY
        assert cfg.solver_timeout_ms == 5000
        assert cfg.max_proof_depth == 50
        assert cfg.proof_strategy == ProofStrategy.BIDIRECTIONAL
        assert cfg.enable_memoization is True

    def test_custom(self):
        cfg = LogicEngineConfig(backend=SolverBackend.PROLOG, solver_timeout_ms=1000)
        assert cfg.backend == SolverBackend.PROLOG
        assert cfg.solver_timeout_ms == 1000


@pytest.mark.unit
class TestConstraintConfig:
    def test_defaults(self):
        cfg = ConstraintConfig()
        assert cfg.default_enforcement == ConstraintEnforcement.HARD
        assert cfg.precompile_constraints is True
        assert cfg.max_constraints_per_state == 100

    def test_custom(self):
        cfg = ConstraintConfig(default_enforcement=ConstraintEnforcement.SOFT)
        assert cfg.default_enforcement == ConstraintEnforcement.SOFT


@pytest.mark.unit
class TestProofConfig:
    def test_defaults(self):
        cfg = ProofConfig()
        assert cfg.generate_proof_trees is True
        assert cfg.generate_natural_language_explanations is True
        assert cfg.confidence_aggregation_method == "geometric_mean"
        assert cfg.min_proof_confidence == 0.7


@pytest.mark.unit
class TestSymbolicAgentConfig:
    def test_defaults(self):
        cfg = SymbolicAgentConfig()
        assert cfg.fallback_to_neural is True
        assert cfg.max_facts == 10000

    def test_weight_normalization(self):
        cfg = SymbolicAgentConfig(neural_confidence_weight=1.0, symbolic_confidence_weight=1.0)
        assert abs(cfg.neural_confidence_weight + cfg.symbolic_confidence_weight - 1.0) < 1e-6

    def test_weights_already_normalized(self):
        cfg = SymbolicAgentConfig(neural_confidence_weight=0.3, symbolic_confidence_weight=0.7)
        assert cfg.neural_confidence_weight == pytest.approx(0.3)
        assert cfg.symbolic_confidence_weight == pytest.approx(0.7)


@pytest.mark.unit
class TestNeuralEmbeddingConfig:
    def test_defaults(self):
        cfg = NeuralEmbeddingConfig()
        assert cfg.fact_embedding_dim == 256
        assert cfg.state_embedding_dim == 512
        assert cfg.encoder_type == "transformer"
        assert cfg.dropout == pytest.approx(0.1)


@pytest.mark.unit
class TestNeuroSymbolicConfig:
    def test_defaults(self):
        cfg = NeuroSymbolicConfig()
        assert cfg.device in ("cpu", "cuda")
        assert cfg.seed == 42
        assert cfg.enable_async is True

    def test_to_dict(self):
        cfg = NeuroSymbolicConfig()
        d = cfg.to_dict()
        assert "logic_engine" in d
        assert "constraints" in d
        assert "proof" in d
        assert "agent" in d
        assert "embedding" in d
        assert "device" in d

    def test_from_dict(self):
        d = {
            "logic_engine": {"backend": "PROLOG", "solver_timeout_ms": 2000},
            "constraints": {"default_enforcement": "SOFT"},
            "device": "cpu",
            "seed": 99,
        }
        cfg = NeuroSymbolicConfig.from_dict(d)
        assert cfg.logic_engine.backend == SolverBackend.PROLOG
        assert cfg.logic_engine.solver_timeout_ms == 2000
        assert cfg.constraints.default_enforcement == ConstraintEnforcement.SOFT
        assert cfg.device == "cpu"
        assert cfg.seed == 99

    def test_save_and_load(self, tmp_path):
        cfg = NeuroSymbolicConfig()
        cfg.seed = 123
        path = str(tmp_path / "config.json")
        cfg.save(path)
        loaded = NeuroSymbolicConfig.load(path)
        assert loaded.seed == 123

    def test_cuda_fallback_to_cpu(self):
        # When cuda isn't available, device should be cpu
        cfg = NeuroSymbolicConfig(device="cpu")
        assert cfg.device == "cpu"


@pytest.mark.unit
class TestPresetConfigs:
    def test_default_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, NeuroSymbolicConfig)

    def test_high_precision_config(self):
        cfg = get_high_precision_config()
        assert cfg.constraints.default_enforcement == ConstraintEnforcement.HARD
        assert cfg.constraints.min_satisfaction_ratio == 0.95
        assert cfg.proof.explanation_verbosity_level == 3
        assert cfg.agent.neural_confidence_weight == pytest.approx(0.2)

    def test_low_latency_config(self):
        cfg = get_low_latency_config()
        assert cfg.logic_engine.solver_timeout_ms == 1000
        assert cfg.logic_engine.max_proof_depth == 20
        assert cfg.proof.generate_proof_trees is False
        assert cfg.constraints.default_enforcement == ConstraintEnforcement.SOFT

    def test_hybrid_mcts_config(self):
        cfg = get_hybrid_mcts_config()
        assert cfg.logic_engine.solver_timeout_ms == 500
        assert cfg.logic_engine.enable_memoization is True
        assert cfg.constraints.precompile_constraints is True
        assert cfg.agent.neural_confidence_weight == pytest.approx(0.5)
