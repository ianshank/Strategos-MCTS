"""
Neuro-Symbolic Configuration Module.

Provides centralized configuration management for neuro-symbolic AI components
following 2025 best practices:
- Dataclass-based configurations for type safety
- No hardcoded values - all configurable
- Preset configurations for common use cases
- Environment-aware defaults
- Validation and normalization

Based on research:
- "Neuro-Symbolic AI: The 3rd Wave" (Garcez & Lamb, 2023)
- "Differentiable Reasoning over Symbolic Knowledge" (Rocktäschel, 2024)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# Optional torch import for environments without GPU support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


class SolverBackend(Enum):
    """Available symbolic solver backends."""

    Z3 = auto()  # Microsoft Z3 SMT solver
    PROLOG = auto()  # SWI-Prolog logic programming
    DATALOG = auto()  # Datalog for recursive queries
    SYMPY = auto()  # SymPy for algebraic reasoning
    CUSTOM = auto()  # Custom solver implementation


class ProofStrategy(Enum):
    """Proof search strategies."""

    FORWARD_CHAINING = auto()  # Data-driven reasoning
    BACKWARD_CHAINING = auto()  # Goal-driven reasoning
    BIDIRECTIONAL = auto()  # Combined approach
    NEURAL_GUIDED = auto()  # Neural network guides search


class ConstraintEnforcement(Enum):
    """How constraints are enforced."""

    HARD = auto()  # Must be satisfied (prune if violated)
    SOFT = auto()  # Prefer satisfaction (penalize violations)
    ADVISORY = auto()  # Log violations but don't enforce


@dataclass
class LogicEngineConfig:
    """Configuration for the symbolic logic engine."""

    # Solver selection
    backend: SolverBackend = SolverBackend.Z3
    fallback_backend: SolverBackend | None = SolverBackend.SYMPY

    # Timeout and limits
    solver_timeout_ms: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_SOLVER_TIMEOUT_MS", "5000")))
    max_proof_depth: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_PROOF_DEPTH", "50")))
    max_unification_attempts: int = field(
        default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_UNIFICATION", "1000"))
    )

    # Proof strategy
    proof_strategy: ProofStrategy = ProofStrategy.BIDIRECTIONAL
    enable_memoization: bool = True
    cache_size: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_CACHE_SIZE", "10000")))

    # Parallelization
    parallel_proof_search: bool = True
    max_parallel_proofs: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_PARALLEL_PROOFS", "4")))

    # Debug and tracing
    trace_proofs: bool = False
    export_proof_trees: bool = False


@dataclass
class ConstraintConfig:
    """Configuration for symbolic constraints."""

    # Enforcement behavior
    default_enforcement: ConstraintEnforcement = ConstraintEnforcement.HARD

    # Validation thresholds
    soft_constraint_penalty_weight: float = field(
        default_factory=lambda: float(os.getenv("NEURO_SYMBOLIC_SOFT_PENALTY_WEIGHT", "0.5"))
    )
    min_satisfaction_ratio: float = field(
        default_factory=lambda: float(os.getenv("NEURO_SYMBOLIC_MIN_SATISFACTION", "0.8"))
    )

    # Constraint compilation
    precompile_constraints: bool = True
    constraint_cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_CONSTRAINT_TTL", "3600"))
    )

    # Safety limits
    max_constraints_per_state: int = field(
        default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_CONSTRAINTS", "100"))
    )
    max_constraint_variables: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_VARS", "50")))

    # Conflict resolution
    enable_conflict_analysis: bool = True
    max_conflict_resolution_attempts: int = field(
        default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_CONFLICT_ATTEMPTS", "10"))
    )


@dataclass
class ProofConfig:
    """Configuration for proof generation and explanation."""

    # Proof tree generation
    generate_proof_trees: bool = True
    max_proof_tree_nodes: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_PROOF_NODES", "500")))

    # Explanation generation
    generate_natural_language_explanations: bool = True
    explanation_verbosity_level: int = field(
        default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_EXPLANATION_VERBOSITY", "2"))
    )  # 1=minimal, 2=standard, 3=detailed

    # Confidence scoring
    confidence_aggregation_method: str = "geometric_mean"  # mean, min, geometric_mean
    min_proof_confidence: float = field(
        default_factory=lambda: float(os.getenv("NEURO_SYMBOLIC_MIN_PROOF_CONFIDENCE", "0.7"))
    )

    # Caching
    cache_proof_trees: bool = True
    proof_cache_max_size: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_PROOF_CACHE_SIZE", "1000")))


@dataclass
class SymbolicAgentConfig:
    """Configuration for the symbolic reasoning agent."""

    # Agent behavior
    fallback_to_neural: bool = True
    neural_fallback_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEURO_SYMBOLIC_NEURAL_FALLBACK_THRESHOLD", "0.6"))
    )

    # Knowledge base
    max_facts: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_FACTS", "10000")))
    fact_expiry_seconds: int | None = None  # None = never expire

    # Query processing
    max_query_complexity: int = field(
        default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_MAX_QUERY_COMPLEXITY", "20"))
    )
    enable_query_rewriting: bool = True

    # Integration with neural components
    use_neural_for_ambiguity_resolution: bool = True
    neural_confidence_weight: float = field(
        default_factory=lambda: float(os.getenv("NEURO_SYMBOLIC_NEURAL_WEIGHT", "0.3"))
    )
    symbolic_confidence_weight: float = field(
        default_factory=lambda: float(os.getenv("NEURO_SYMBOLIC_SYMBOLIC_WEIGHT", "0.7"))
    )

    def __post_init__(self):
        """Validate configuration."""
        total_weight = self.neural_confidence_weight + self.symbolic_confidence_weight
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            self.neural_confidence_weight /= total_weight
            self.symbolic_confidence_weight /= total_weight


@dataclass
class NeuralEmbeddingConfig:
    """Configuration for neural embeddings in neuro-symbolic system."""

    # Embedding dimensions
    fact_embedding_dim: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_FACT_DIM", "256")))
    state_embedding_dim: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_STATE_DIM", "512")))

    # Encoder architecture
    encoder_type: str = "transformer"  # transformer, lstm, mlp
    num_encoder_layers: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_ENCODER_LAYERS", "4")))
    encoder_hidden_dim: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_ENCODER_HIDDEN", "512")))
    num_attention_heads: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_ATTENTION_HEADS", "8")))

    # Regularization
    dropout: float = field(default_factory=lambda: float(os.getenv("NEURO_SYMBOLIC_DROPOUT", "0.1")))
    use_layer_norm: bool = True


@dataclass
class NeuroSymbolicConfig:
    """
    Master configuration for the neuro-symbolic AI system.

    Centralizes all neuro-symbolic configuration with sensible defaults
    based on current research best practices.
    """

    # Component configurations
    logic_engine: LogicEngineConfig = field(default_factory=LogicEngineConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    proof: ProofConfig = field(default_factory=ProofConfig)
    agent: SymbolicAgentConfig = field(default_factory=SymbolicAgentConfig)
    embedding: NeuralEmbeddingConfig = field(default_factory=NeuralEmbeddingConfig)

    # System settings
    device: str = field(default_factory=lambda: "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
    seed: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_SEED", "42")))

    # Performance settings
    enable_async: bool = True
    batch_symbolic_queries: bool = True
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_BATCH_SIZE", "32")))

    # Logging and monitoring
    log_level: str = field(default_factory=lambda: os.getenv("NEURO_SYMBOLIC_LOG_LEVEL", "INFO"))
    enable_metrics: bool = True
    metrics_export_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("NEURO_SYMBOLIC_METRICS_INTERVAL", "60"))
    )

    # Persistence
    checkpoint_dir: str = field(
        default_factory=lambda: os.getenv("NEURO_SYMBOLIC_CHECKPOINT_DIR", "./checkpoints/neuro_symbolic")
    )
    knowledge_base_path: str | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure device is valid
        if self.device.startswith("cuda") and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            self.device = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for logging/serialization."""
        return {
            "logic_engine": {
                "backend": self.logic_engine.backend.name,
                "solver_timeout_ms": self.logic_engine.solver_timeout_ms,
                "max_proof_depth": self.logic_engine.max_proof_depth,
                "proof_strategy": self.logic_engine.proof_strategy.name,
            },
            "constraints": {
                "default_enforcement": self.constraints.default_enforcement.name,
                "soft_constraint_penalty_weight": self.constraints.soft_constraint_penalty_weight,
                "min_satisfaction_ratio": self.constraints.min_satisfaction_ratio,
            },
            "proof": {
                "generate_proof_trees": self.proof.generate_proof_trees,
                "generate_explanations": self.proof.generate_natural_language_explanations,
                "min_proof_confidence": self.proof.min_proof_confidence,
            },
            "agent": {
                "fallback_to_neural": self.agent.fallback_to_neural,
                "neural_fallback_threshold": self.agent.neural_fallback_confidence_threshold,
                "max_facts": self.agent.max_facts,
            },
            "embedding": {
                "fact_embedding_dim": self.embedding.fact_embedding_dim,
                "state_embedding_dim": self.embedding.state_embedding_dim,
                "encoder_type": self.embedding.encoder_type,
            },
            "device": self.device,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> NeuroSymbolicConfig:
        """Create configuration from dictionary."""
        config = cls()

        if "logic_engine" in config_dict:
            le = config_dict["logic_engine"]
            if "backend" in le:
                config.logic_engine.backend = SolverBackend[le["backend"]]
            if "solver_timeout_ms" in le:
                config.logic_engine.solver_timeout_ms = le["solver_timeout_ms"]
            if "max_proof_depth" in le:
                config.logic_engine.max_proof_depth = le["max_proof_depth"]
            if "proof_strategy" in le:
                config.logic_engine.proof_strategy = ProofStrategy[le["proof_strategy"]]

        if "constraints" in config_dict:
            c = config_dict["constraints"]
            if "default_enforcement" in c:
                config.constraints.default_enforcement = ConstraintEnforcement[c["default_enforcement"]]
            if "soft_constraint_penalty_weight" in c:
                config.constraints.soft_constraint_penalty_weight = c["soft_constraint_penalty_weight"]
            if "min_satisfaction_ratio" in c:
                config.constraints.min_satisfaction_ratio = c["min_satisfaction_ratio"]

        if "device" in config_dict:
            config.device = config_dict["device"]
        if "seed" in config_dict:
            config.seed = config_dict["seed"]

        return config

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> NeuroSymbolicConfig:
        """Load configuration from JSON file."""
        import json

        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations for common use cases


def get_default_config() -> NeuroSymbolicConfig:
    """Get default balanced configuration."""
    return NeuroSymbolicConfig()


def get_high_precision_config() -> NeuroSymbolicConfig:
    """
    Configuration optimized for high precision and explainability.

    Use when correctness and auditability are paramount.
    """
    config = NeuroSymbolicConfig()

    # Strict constraints
    config.constraints.default_enforcement = ConstraintEnforcement.HARD
    config.constraints.min_satisfaction_ratio = 0.95

    # Detailed proofs
    config.proof.generate_proof_trees = True
    config.proof.generate_natural_language_explanations = True
    config.proof.explanation_verbosity_level = 3
    config.proof.min_proof_confidence = 0.85

    # More thorough search
    config.logic_engine.max_proof_depth = 100
    config.logic_engine.solver_timeout_ms = 10000

    # Less neural reliance
    config.agent.neural_confidence_weight = 0.2
    config.agent.symbolic_confidence_weight = 0.8

    return config


def get_low_latency_config() -> NeuroSymbolicConfig:
    """
    Configuration optimized for low latency.

    Use when response time is critical.
    """
    config = NeuroSymbolicConfig()

    # Faster timeouts
    config.logic_engine.solver_timeout_ms = 1000
    config.logic_engine.max_proof_depth = 20
    config.logic_engine.max_unification_attempts = 500

    # Lighter proofs
    config.proof.generate_proof_trees = False
    config.proof.generate_natural_language_explanations = False

    # More neural fallback
    config.agent.fallback_to_neural = True
    config.agent.neural_fallback_confidence_threshold = 0.4

    # Soft constraints for flexibility
    config.constraints.default_enforcement = ConstraintEnforcement.SOFT

    return config


def get_hybrid_mcts_config() -> NeuroSymbolicConfig:
    """
    Configuration optimized for integration with MCTS.

    Balances symbolic constraint checking with search efficiency.
    """
    config = NeuroSymbolicConfig()

    # Fast constraint checking for MCTS expansion
    config.logic_engine.solver_timeout_ms = 500
    config.logic_engine.enable_memoization = True
    config.logic_engine.cache_size = 50000

    # Parallel proof search for parallel MCTS
    config.logic_engine.parallel_proof_search = True

    # Precompile constraints for speed
    config.constraints.precompile_constraints = True

    # Balance neural and symbolic
    config.agent.neural_confidence_weight = 0.5
    config.agent.symbolic_confidence_weight = 0.5

    return config
