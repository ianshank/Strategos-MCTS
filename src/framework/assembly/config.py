"""
Configuration for Assembly Theory components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class AssemblyConfig:
    """
    Configuration for Assembly Theory integration.

    Attributes:
        mcts_ucb_weight: Weight for assembly factor in MCTS UCB (default: 0.2)
        max_complexity_threshold: Maximum assembly index before pruning (default: 15)
        routing_simple_threshold: Assembly index threshold for simple queries (default: 3)
        routing_medium_threshold: Assembly index threshold for medium queries (default: 7)
        trm_complexity_penalty: Lambda for complexity penalty in TRM (default: 0.1)
        substructure_reuse_threshold: Minimum similarity for pattern reuse (default: 0.7)
        enable_assembly_routing: Enable assembly-based routing (default: True)
        enable_assembly_mcts: Enable assembly enhancements in MCTS (default: True)
        enable_assembly_hrm: Enable assembly enhancements in HRM (default: True)
        enable_assembly_trm: Enable assembly enhancements in TRM (default: True)
        enable_assembly_consensus: Enable assembly-based consensus (default: True)
        cache_assembly_indices: Cache computed assembly indices (default: True)
        max_cache_size: Maximum cache entries (default: 10000)
    """

    # MCTS settings
    mcts_ucb_weight: float = 0.2
    max_complexity_threshold: int = 15

    # Routing thresholds
    routing_simple_threshold: int = 3
    routing_medium_threshold: int = 7

    # TRM settings
    trm_complexity_penalty: float = 0.1
    trm_convergence_threshold: float = 0.01

    # Substructure library
    substructure_reuse_threshold: float = 0.7
    substructure_max_size: int = 10000

    # Feature flags
    enable_assembly_routing: bool = True
    enable_assembly_mcts: bool = True
    enable_assembly_hrm: bool = True
    enable_assembly_trm: bool = True
    enable_assembly_consensus: bool = True

    # Performance
    cache_assembly_indices: bool = True
    max_cache_size: int = 10000

    # Consensus
    consensus_pathway_weight: float = 0.7
    consensus_complexity_weight: float = 0.3
    complexity_selection_weight: float = 0.1

    # Concept extraction
    min_concept_frequency: int = 1
    max_concepts: int = 100
    use_technical_terms: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "AssemblyConfig":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()  # Return defaults

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assembly_config = data.get("assembly", {})

        # Flatten nested structure
        config_dict = {}

        # MCTS settings
        mcts = assembly_config.get("mcts", {})
        config_dict["mcts_ucb_weight"] = mcts.get("ucb_weight", cls.mcts_ucb_weight)
        config_dict["max_complexity_threshold"] = mcts.get("max_complexity_threshold", cls.max_complexity_threshold)

        # Routing
        routing = assembly_config.get("routing", {})
        config_dict["routing_simple_threshold"] = routing.get("simple_threshold", cls.routing_simple_threshold)
        config_dict["routing_medium_threshold"] = routing.get("medium_threshold", cls.routing_medium_threshold)

        # TRM
        trm = assembly_config.get("trm", {})
        config_dict["trm_complexity_penalty"] = trm.get("complexity_penalty", cls.trm_complexity_penalty)
        config_dict["trm_convergence_threshold"] = trm.get("convergence_threshold", cls.trm_convergence_threshold)

        # Substructure
        substructure = assembly_config.get("substructure", {})
        config_dict["substructure_reuse_threshold"] = substructure.get(
            "reuse_threshold", cls.substructure_reuse_threshold
        )
        config_dict["substructure_max_size"] = substructure.get("max_size", cls.substructure_max_size)

        # Feature flags
        flags = assembly_config.get("feature_flags", {})
        config_dict["enable_assembly_routing"] = flags.get("routing", cls.enable_assembly_routing)
        config_dict["enable_assembly_mcts"] = flags.get("mcts", cls.enable_assembly_mcts)
        config_dict["enable_assembly_hrm"] = flags.get("hrm", cls.enable_assembly_hrm)
        config_dict["enable_assembly_trm"] = flags.get("trm", cls.enable_assembly_trm)
        config_dict["enable_assembly_consensus"] = flags.get("consensus", cls.enable_assembly_consensus)

        # Performance
        perf = assembly_config.get("performance", {})
        config_dict["cache_assembly_indices"] = perf.get("cache_indices", cls.cache_assembly_indices)
        config_dict["max_cache_size"] = perf.get("max_cache_size", cls.max_cache_size)

        # Consensus
        consensus = assembly_config.get("consensus", {})
        config_dict["consensus_pathway_weight"] = consensus.get("pathway_weight", cls.consensus_pathway_weight)
        config_dict["consensus_complexity_weight"] = consensus.get("complexity_weight", cls.consensus_complexity_weight)
        config_dict["complexity_selection_weight"] = consensus.get("selection_weight", cls.complexity_selection_weight)

        # Concept extraction
        concepts = assembly_config.get("concept_extraction", {})
        config_dict["min_concept_frequency"] = concepts.get("min_frequency", cls.min_concept_frequency)
        config_dict["max_concepts"] = concepts.get("max_concepts", cls.max_concepts)
        config_dict["use_technical_terms"] = concepts.get("use_technical_terms", cls.use_technical_terms)

        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mcts": {
                "ucb_weight": self.mcts_ucb_weight,
                "max_complexity_threshold": self.max_complexity_threshold,
            },
            "routing": {
                "simple_threshold": self.routing_simple_threshold,
                "medium_threshold": self.routing_medium_threshold,
            },
            "trm": {
                "complexity_penalty": self.trm_complexity_penalty,
                "convergence_threshold": self.trm_convergence_threshold,
            },
            "substructure": {
                "reuse_threshold": self.substructure_reuse_threshold,
                "max_size": self.substructure_max_size,
            },
            "feature_flags": {
                "routing": self.enable_assembly_routing,
                "mcts": self.enable_assembly_mcts,
                "hrm": self.enable_assembly_hrm,
                "trm": self.enable_assembly_trm,
                "consensus": self.enable_assembly_consensus,
            },
            "performance": {
                "cache_indices": self.cache_assembly_indices,
                "max_cache_size": self.max_cache_size,
            },
            "consensus": {
                "pathway_weight": self.consensus_pathway_weight,
                "complexity_weight": self.consensus_complexity_weight,
                "selection_weight": self.complexity_selection_weight,
            },
            "concept_extraction": {
                "min_frequency": self.min_concept_frequency,
                "max_concepts": self.max_concepts,
                "use_technical_terms": self.use_technical_terms,
            },
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump({"assembly": self.to_dict()}, f, default_flow_style=False)

    def validate(self) -> None:
        """Validate configuration values."""
        assert 0.0 <= self.mcts_ucb_weight <= 1.0, "UCB weight must be in [0, 1]"
        assert self.max_complexity_threshold > 0, "Max complexity threshold must be positive"
        assert self.routing_simple_threshold < self.routing_medium_threshold, (
            "Simple threshold must be less than medium threshold"
        )
        assert 0.0 <= self.trm_complexity_penalty <= 1.0, "Complexity penalty must be in [0, 1]"
        assert 0.0 <= self.substructure_reuse_threshold <= 1.0, "Reuse threshold must be in [0, 1]"
        assert self.max_cache_size > 0, "Cache size must be positive"
        assert 0.0 <= self.consensus_pathway_weight <= 1.0, "Pathway weight must be in [0, 1]"
        assert 0.0 <= self.consensus_complexity_weight <= 1.0, "Complexity weight must be in [0, 1]"
