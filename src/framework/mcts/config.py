"""
MCTS Configuration Module - Parameter management and presets.

Provides:
- MCTSConfig dataclass with all parameters
- Validation of parameter bounds
- Preset configurations (fast, balanced, thorough)
- Serialization support for experiment tracking
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .policies import SelectionPolicy


class ConfigPreset(Enum):
    """Preset configuration names."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    EXPLORATION_HEAVY = "exploration_heavy"
    EXPLOITATION_HEAVY = "exploitation_heavy"


@dataclass
class MCTSConfig:
    """
    Complete configuration for MCTS engine.

    All MCTS parameters are centralized here with validation.
    Supports serialization for experiment tracking and reproducibility.
    """

    # Core MCTS parameters
    num_iterations: int = 100
    """Number of MCTS iterations to run."""

    seed: int = 42
    """Random seed for deterministic behavior."""

    exploration_weight: float = 1.414
    """UCB1 exploration constant (c). Higher = more exploration."""

    # Progressive widening
    progressive_widening_k: float = 1.0
    """Progressive widening coefficient. Higher = more conservative."""

    progressive_widening_alpha: float = 0.5
    """Progressive widening exponent. Lower = more aggressive expansion."""

    # Rollout configuration
    max_rollout_depth: int = 10
    """Maximum depth for rollout simulations."""

    rollout_policy: str = "hybrid"
    """Rollout policy: 'random', 'greedy', 'hybrid'."""

    # Action selection
    selection_policy: SelectionPolicy = SelectionPolicy.MAX_VISITS
    """Policy for final action selection."""

    # Parallelization
    max_parallel_rollouts: int = 4
    """Maximum concurrent rollout simulations."""

    # Caching
    enable_cache: bool = True
    """Enable simulation result caching."""

    cache_size_limit: int = 10000
    """Maximum number of cached simulation results."""

    # Tree structure
    max_tree_depth: int = 20
    """Maximum depth of MCTS tree."""

    max_children_per_node: int = 50
    """Maximum children per node (action branching limit)."""

    # Early termination (visit-fraction based)
    early_termination_threshold: float = 0.95
    """Stop if best action has this fraction of total visits."""

    min_iterations_before_termination: int = 50
    """Minimum iterations before early termination check."""

    # Early termination (value-convergence based)
    early_stop_threshold: float = 0.01
    """Minimum change in best action value to continue search. Set to 0 to disable."""

    early_stop_patience: int = 10
    """Number of consecutive iterations with no improvement before stopping."""

    # Value bounds
    min_value: float = 0.0
    """Minimum value for normalization."""

    max_value: float = 1.0
    """Maximum value for normalization."""

    # Metadata
    name: str = "default"
    """Configuration name for tracking."""

    description: str = ""
    """Description of this configuration."""

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all configuration parameters.

        Raises:
            ValueError: If any parameter is out of valid bounds.
        """
        errors = []

        # Core parameters
        if self.num_iterations < 1:
            errors.append("num_iterations must be >= 1")
        if self.num_iterations > 100000:
            errors.append("num_iterations should be <= 100000 for practical use")

        if self.exploration_weight < 0:
            errors.append("exploration_weight must be >= 0")
        if self.exploration_weight > 10:
            errors.append("exploration_weight should be <= 10")

        # Progressive widening
        if self.progressive_widening_k <= 0:
            errors.append("progressive_widening_k must be > 0")
        if not 0 < self.progressive_widening_alpha < 1:
            errors.append("progressive_widening_alpha must be in (0, 1)")

        # Rollout
        if self.max_rollout_depth < 1:
            errors.append("max_rollout_depth must be >= 1")
        if self.rollout_policy not in ["random", "greedy", "hybrid", "llm"]:
            errors.append("rollout_policy must be one of: random, greedy, hybrid, llm")

        # Parallelization
        if self.max_parallel_rollouts < 1:
            errors.append("max_parallel_rollouts must be >= 1")
        if self.max_parallel_rollouts > 100:
            errors.append("max_parallel_rollouts should be <= 100")

        # Caching
        if self.cache_size_limit < 0:
            errors.append("cache_size_limit must be >= 0")

        # Tree structure
        if self.max_tree_depth < 1:
            errors.append("max_tree_depth must be >= 1")
        if self.max_children_per_node < 1:
            errors.append("max_children_per_node must be >= 1")

        # Early termination (visit-fraction based)
        if not 0 < self.early_termination_threshold <= 1:
            errors.append("early_termination_threshold must be in (0, 1]")
        if self.min_iterations_before_termination < 1:
            errors.append("min_iterations_before_termination must be >= 1")
        if self.min_iterations_before_termination > self.num_iterations:
            errors.append("min_iterations_before_termination must be <= num_iterations")

        # Early termination (value-convergence based)
        if self.early_stop_threshold < 0:
            errors.append("early_stop_threshold must be >= 0")
        if self.early_stop_patience < 1:
            errors.append("early_stop_patience must be >= 1")

        # Value bounds
        if self.min_value >= self.max_value:
            errors.append("min_value must be < max_value")

        if errors:
            raise ValueError("Invalid MCTS configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of config.
        """
        d = asdict(self)
        # Convert enum to string
        d["selection_policy"] = self.selection_policy.value
        return d

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize configuration to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCTSConfig:
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration parameters

        Returns:
            MCTSConfig instance
        """
        # Convert selection_policy string back to enum
        if "selection_policy" in data and isinstance(data["selection_policy"], str):
            data["selection_policy"] = SelectionPolicy(data["selection_policy"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> MCTSConfig:
        """
        Deserialize configuration from JSON string.

        Args:
            json_str: JSON string

        Returns:
            MCTSConfig instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def copy(self, **overrides) -> MCTSConfig:
        """
        Create a copy with optional parameter overrides.

        Args:
            **overrides: Parameters to override

        Returns:
            New MCTSConfig instance
        """
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"MCTSConfig(name={self.name!r}, "
            f"iterations={self.num_iterations}, "
            f"c={self.exploration_weight}, "
            f"widening_k={self.progressive_widening_k}, "
            f"widening_alpha={self.progressive_widening_alpha})"
        )


def create_preset_config(preset: ConfigPreset) -> MCTSConfig:
    """
    Create a preset configuration.

    Args:
        preset: Preset type to create

    Returns:
        MCTSConfig with preset parameters
    """
    if preset == ConfigPreset.FAST:
        return MCTSConfig(
            name="fast",
            description="Fast search with minimal iterations",
            num_iterations=25,
            exploration_weight=1.414,
            progressive_widening_k=0.5,  # Aggressive widening
            progressive_widening_alpha=0.5,
            max_rollout_depth=5,
            rollout_policy="random",
            selection_policy=SelectionPolicy.MAX_VISITS,
            max_parallel_rollouts=8,
            cache_size_limit=1000,
            early_termination_threshold=0.8,
            min_iterations_before_termination=10,
        )

    elif preset == ConfigPreset.BALANCED:
        return MCTSConfig(
            name="balanced",
            description="Balanced search for typical use cases",
            num_iterations=100,
            exploration_weight=1.414,
            progressive_widening_k=1.0,
            progressive_widening_alpha=0.5,
            max_rollout_depth=10,
            rollout_policy="hybrid",
            selection_policy=SelectionPolicy.MAX_VISITS,
            max_parallel_rollouts=4,
            cache_size_limit=10000,
            early_termination_threshold=0.9,
            min_iterations_before_termination=50,
        )

    elif preset == ConfigPreset.THOROUGH:
        return MCTSConfig(
            name="thorough",
            description="Thorough search for high-stakes decisions",
            num_iterations=500,
            exploration_weight=1.414,
            progressive_widening_k=2.0,  # Conservative widening
            progressive_widening_alpha=0.6,
            max_rollout_depth=20,
            rollout_policy="hybrid",
            selection_policy=SelectionPolicy.ROBUST_CHILD,
            max_parallel_rollouts=4,
            cache_size_limit=50000,
            early_termination_threshold=0.95,
            min_iterations_before_termination=200,
        )

    elif preset == ConfigPreset.EXPLORATION_HEAVY:
        return MCTSConfig(
            name="exploration_heavy",
            description="High exploration for diverse action discovery",
            num_iterations=200,
            exploration_weight=2.5,  # High exploration
            progressive_widening_k=0.8,  # More widening
            progressive_widening_alpha=0.4,  # Aggressive
            max_rollout_depth=15,
            rollout_policy="random",
            selection_policy=SelectionPolicy.MAX_VISITS,
            max_parallel_rollouts=6,
            cache_size_limit=20000,
            early_termination_threshold=0.95,
            min_iterations_before_termination=100,
        )

    elif preset == ConfigPreset.EXPLOITATION_HEAVY:
        return MCTSConfig(
            name="exploitation_heavy",
            description="High exploitation for known-good action refinement",
            num_iterations=150,
            exploration_weight=0.5,  # Low exploration
            progressive_widening_k=3.0,  # Conservative
            progressive_widening_alpha=0.7,  # Very conservative
            max_rollout_depth=10,
            rollout_policy="greedy",
            selection_policy=SelectionPolicy.MAX_VALUE,
            max_parallel_rollouts=4,
            cache_size_limit=10000,
            early_termination_threshold=0.85,
            min_iterations_before_termination=75,
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")


# Default configurations for easy access
DEFAULT_CONFIG = MCTSConfig()
FAST_CONFIG = create_preset_config(ConfigPreset.FAST)
BALANCED_CONFIG = create_preset_config(ConfigPreset.BALANCED)
THOROUGH_CONFIG = create_preset_config(ConfigPreset.THOROUGH)
