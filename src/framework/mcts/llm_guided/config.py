"""
Configuration for LLM-Guided MCTS.

Provides:
- LLMGuidedMCTSConfig: Main configuration dataclass
- GeneratorConfig: Configuration for the generator agent
- ReflectorConfig: Configuration for the reflector agent
- Presets for different use cases
- Validation and serialization
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.config.settings import LLMProvider

from . import constants as C


class LLMGuidedMCTSPreset(Enum):
    """Preset configurations for LLM-guided MCTS."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    DATA_COLLECTION = "data_collection"
    BENCHMARK = "benchmark"


@dataclass
class GeneratorConfig:
    """Configuration for the Generator agent."""

    # Model settings
    model: str = C.DEFAULT_GENERATOR_MODEL
    """LLM model to use for generation."""

    temperature: float = C.DEFAULT_GENERATOR_TEMPERATURE
    """Temperature for generation (higher = more diverse)."""

    max_tokens: int = C.DEFAULT_GENERATOR_MAX_TOKENS
    """Maximum tokens for generated code."""

    # Generation behavior
    num_variants: int = C.DEFAULT_NUM_VARIANTS
    """Number of code variants to generate per expansion."""

    top_p: float = C.DEFAULT_TOP_P
    """Nucleus sampling parameter."""

    # Prompt settings
    include_test_errors: bool = True
    """Include test errors in the prompt."""

    include_previous_attempts: bool = True
    """Include previous failed attempts in context."""

    max_previous_attempts: int = 3
    """Maximum number of previous attempts to include."""

    # Cost tracking
    track_tokens: bool = True
    """Track token usage for cost estimation."""

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if self.temperature < C.TEMPERATURE_MIN or self.temperature > C.TEMPERATURE_MAX:
            errors.append(f"temperature must be in [{C.TEMPERATURE_MIN}, {C.TEMPERATURE_MAX}]")
        if self.max_tokens < C.MAX_TOKENS_MIN or self.max_tokens > C.MAX_TOKENS_MAX:
            errors.append(f"max_tokens must be in [{C.MAX_TOKENS_MIN}, {C.MAX_TOKENS_MAX}]")
        if self.num_variants < C.NUM_VARIANTS_MIN or self.num_variants > C.NUM_VARIANTS_MAX:
            errors.append(f"num_variants must be in [{C.NUM_VARIANTS_MIN}, {C.NUM_VARIANTS_MAX}]")
        if self.top_p < C.TOP_P_MIN or self.top_p > C.TOP_P_MAX:
            errors.append(f"top_p must be in [{C.TOP_P_MIN}, {C.TOP_P_MAX}]")
        if self.max_previous_attempts < 0:
            errors.append("max_previous_attempts must be >= 0")

        if errors:
            raise ValueError("Invalid GeneratorConfig:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class ReflectorConfig:
    """Configuration for the Reflector agent."""

    # Model settings
    model: str = C.DEFAULT_REFLECTOR_MODEL
    """LLM model to use for reflection."""

    temperature: float = C.DEFAULT_REFLECTOR_TEMPERATURE
    """Temperature for reflection (lower = more deterministic)."""

    max_tokens: int = C.DEFAULT_REFLECTOR_MAX_TOKENS
    """Maximum tokens for reflection output."""

    # Evaluation settings
    include_test_results: bool = True
    """Include test execution results in evaluation."""

    include_code_context: bool = True
    """Include original problem context."""

    # Scoring settings
    min_score: float = 0.0
    """Minimum score value."""

    max_score: float = 1.0
    """Maximum score value."""

    # Cost tracking
    track_tokens: bool = True
    """Track token usage for cost estimation."""

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if self.temperature < C.TEMPERATURE_MIN or self.temperature > C.TEMPERATURE_MAX:
            errors.append(f"temperature must be in [{C.TEMPERATURE_MIN}, {C.TEMPERATURE_MAX}]")
        if self.max_tokens < C.MAX_TOKENS_MIN or self.max_tokens > C.REFLECTOR_MAX_TOKENS_MAX:
            errors.append(f"max_tokens must be in [{C.MAX_TOKENS_MIN}, {C.REFLECTOR_MAX_TOKENS_MAX}]")
        if self.min_score >= self.max_score:
            errors.append("min_score must be < max_score")

        if errors:
            raise ValueError("Invalid ReflectorConfig:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class LLMGuidedMCTSConfig:
    """
    Complete configuration for LLM-Guided MCTS.

    Centralizes all parameters with validation and serialization support.
    """

    # MCTS Core Parameters
    num_iterations: int = C.DEFAULT_ITERATIONS_BALANCED
    """Number of MCTS iterations to run."""

    seed: int = C.DEFAULT_SEED
    """Random seed for reproducibility."""

    exploration_weight: float = C.UCB1_EXPLORATION_CONSTANT
    """UCB1 exploration constant (c). Higher = more exploration."""

    # Tree Structure
    max_depth: int = C.DEFAULT_MAX_DEPTH
    """Maximum depth of MCTS tree."""

    max_children: int = C.DEFAULT_MAX_CHILDREN
    """Maximum children per node (branching factor)."""

    # Early Termination
    early_termination_on_solution: bool = True
    """Stop search when a solution is found."""

    solution_confidence_threshold: float = C.DEFAULT_SOLUTION_CONFIDENCE_THRESHOLD
    """Confidence threshold to accept a solution."""

    # Agent Configurations
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)
    """Configuration for generator agent."""

    reflector_config: ReflectorConfig = field(default_factory=ReflectorConfig)
    """Configuration for reflector agent."""

    # Data Collection
    collect_training_data: bool = True
    """Enable training data collection for neural network training."""

    training_data_dir: str = C.DEFAULT_TRAINING_DATA_DIR
    """Directory to save training data."""

    save_mcts_policy: bool = True
    """Save MCTS visit distribution as improved policy target."""

    # Code Execution
    execution_timeout_seconds: float = C.DEFAULT_EXECUTION_TIMEOUT
    """Timeout for code execution in seconds."""

    max_memory_mb: int = C.DEFAULT_MAX_MEMORY_MB
    """Maximum memory for code execution in MB."""

    allow_network: bool = False
    """Allow network access in code execution sandbox."""

    # Provider Settings
    llm_provider: LLMProvider = LLMProvider.OPENAI
    """LLM provider to use."""

    # Logging and Debugging
    verbose: bool = False
    """Enable verbose logging."""

    log_tree_stats: bool = True
    """Log tree statistics after search."""

    # Metadata
    name: str = "default"
    """Configuration name for tracking."""

    description: str = ""
    """Description of this configuration."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        errors = []

        # MCTS parameters
        if self.num_iterations < C.ITERATIONS_MIN or self.num_iterations > C.ITERATIONS_MAX:
            errors.append(f"num_iterations must be in [{C.ITERATIONS_MIN}, {C.ITERATIONS_MAX}]")
        if self.exploration_weight < C.EXPLORATION_WEIGHT_MIN or self.exploration_weight > C.EXPLORATION_WEIGHT_MAX:
            errors.append(f"exploration_weight must be in [{C.EXPLORATION_WEIGHT_MIN}, {C.EXPLORATION_WEIGHT_MAX}]")
        if self.max_depth < C.MIN_DEPTH or self.max_depth > C.MAX_DEPTH_LIMIT:
            errors.append(f"max_depth must be in [{C.MIN_DEPTH}, {C.MAX_DEPTH_LIMIT}]")
        if self.max_children < C.MIN_CHILDREN or self.max_children > C.MAX_CHILDREN_LIMIT:
            errors.append(f"max_children must be in [{C.MIN_CHILDREN}, {C.MAX_CHILDREN_LIMIT}]")

        # Early termination
        if (
            self.solution_confidence_threshold < C.CONFIDENCE_MIN
            or self.solution_confidence_threshold > C.CONFIDENCE_MAX
        ):
            errors.append(f"solution_confidence_threshold must be in [{C.CONFIDENCE_MIN}, {C.CONFIDENCE_MAX}]")

        # Code execution
        if (
            self.execution_timeout_seconds < C.EXECUTION_TIMEOUT_MIN
            or self.execution_timeout_seconds > C.EXECUTION_TIMEOUT_MAX
        ):
            errors.append(
                f"execution_timeout_seconds must be in [{C.EXECUTION_TIMEOUT_MIN}, {C.EXECUTION_TIMEOUT_MAX}]"
            )
        if self.max_memory_mb < C.MAX_MEMORY_MIN_MB or self.max_memory_mb > C.MAX_MEMORY_MAX_MB:
            errors.append(f"max_memory_mb must be in [{C.MAX_MEMORY_MIN_MB}, {C.MAX_MEMORY_MAX_MB}]")

        # Validate nested configs
        try:
            self.generator_config.validate()
        except ValueError as e:
            errors.append(f"Generator config: {e}")

        try:
            self.reflector_config.validate()
        except ValueError as e:
            errors.append(f"Reflector config: {e}")

        if errors:
            raise ValueError("Invalid LLMGuidedMCTSConfig:\n" + "\n".join(f"  - {e}" for e in errors))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["llm_provider"] = self.llm_provider.value
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMGuidedMCTSConfig:
        """Create configuration from dictionary."""
        # Convert nested configs
        if "generator_config" in data and isinstance(data["generator_config"], dict):
            data["generator_config"] = GeneratorConfig(**data["generator_config"])
        if "reflector_config" in data and isinstance(data["reflector_config"], dict):
            data["reflector_config"] = ReflectorConfig(**data["reflector_config"])
        if "llm_provider" in data and isinstance(data["llm_provider"], str):
            data["llm_provider"] = LLMProvider(data["llm_provider"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> LLMGuidedMCTSConfig:
        """Deserialize configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str | Path) -> LLMGuidedMCTSConfig:
        """Load configuration from file."""
        with open(path) as f:
            return cls.from_json(f.read())

    def copy(self, **overrides) -> LLMGuidedMCTSConfig:
        """Create a copy with optional parameter overrides."""
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"LLMGuidedMCTSConfig(name={self.name!r}, "
            f"iterations={self.num_iterations}, "
            f"c={self.exploration_weight}, "
            f"provider={self.llm_provider.value})"
        )


def create_llm_mcts_preset(preset: LLMGuidedMCTSPreset) -> LLMGuidedMCTSConfig:
    """
    Create a preset configuration.

    Args:
        preset: Preset type to create

    Returns:
        LLMGuidedMCTSConfig with preset parameters
    """
    if preset == LLMGuidedMCTSPreset.FAST:
        return LLMGuidedMCTSConfig(
            name="fast",
            description="Fast search for quick iteration",
            num_iterations=C.DEFAULT_ITERATIONS_FAST,
            max_depth=5,
            max_children=3,
            exploration_weight=C.UCB1_EXPLORATION_CONSTANT,
            generator_config=GeneratorConfig(
                model=C.DEFAULT_GENERATOR_MODEL_FAST,
                temperature=C.DEFAULT_GENERATOR_TEMPERATURE,
                num_variants=2,
            ),
            reflector_config=ReflectorConfig(
                model=C.DEFAULT_REFLECTOR_MODEL_FAST,
                temperature=C.DEFAULT_REFLECTOR_TEMPERATURE,
            ),
            collect_training_data=False,
            verbose=False,
        )

    elif preset == LLMGuidedMCTSPreset.BALANCED:
        return LLMGuidedMCTSConfig(
            name="balanced",
            description="Balanced search for typical use cases",
            num_iterations=C.DEFAULT_ITERATIONS_BALANCED,
            max_depth=C.DEFAULT_MAX_DEPTH,
            max_children=C.DEFAULT_MAX_CHILDREN,
            exploration_weight=C.UCB1_EXPLORATION_CONSTANT,
            generator_config=GeneratorConfig(
                model=C.DEFAULT_GENERATOR_MODEL,
                temperature=C.DEFAULT_GENERATOR_TEMPERATURE,
                num_variants=C.DEFAULT_NUM_VARIANTS,
            ),
            reflector_config=ReflectorConfig(
                model=C.DEFAULT_REFLECTOR_MODEL,
                temperature=C.DEFAULT_REFLECTOR_TEMPERATURE,
            ),
            collect_training_data=True,
            verbose=False,
        )

    elif preset == LLMGuidedMCTSPreset.THOROUGH:
        return LLMGuidedMCTSConfig(
            name="thorough",
            description="Thorough search for difficult problems",
            num_iterations=C.DEFAULT_ITERATIONS_THOROUGH,
            max_depth=15,
            max_children=C.DEFAULT_MAX_CHILDREN,
            exploration_weight=2.0,
            generator_config=GeneratorConfig(
                model=C.DEFAULT_GENERATOR_MODEL,
                temperature=0.8,
                num_variants=4,
                include_previous_attempts=True,
                max_previous_attempts=5,
            ),
            reflector_config=ReflectorConfig(
                model=C.DEFAULT_REFLECTOR_MODEL,
                temperature=0.2,
            ),
            collect_training_data=True,
            verbose=True,
            log_tree_stats=True,
        )

    elif preset == LLMGuidedMCTSPreset.DATA_COLLECTION:
        return LLMGuidedMCTSConfig(
            name="data_collection",
            description="Optimized for training data collection",
            num_iterations=C.DEFAULT_ITERATIONS_BENCHMARK,
            max_depth=C.DEFAULT_MAX_DEPTH,
            max_children=C.DEFAULT_MAX_CHILDREN,
            exploration_weight=1.5,
            early_termination_on_solution=False,  # Keep exploring for more data
            generator_config=GeneratorConfig(
                model=C.DEFAULT_GENERATOR_MODEL,
                temperature=C.DEFAULT_GENERATOR_TEMPERATURE,
                num_variants=C.DEFAULT_NUM_VARIANTS,
                include_previous_attempts=True,
            ),
            reflector_config=ReflectorConfig(
                model=C.DEFAULT_REFLECTOR_MODEL,
                temperature=C.DEFAULT_REFLECTOR_TEMPERATURE,
            ),
            collect_training_data=True,
            save_mcts_policy=True,
            verbose=False,
        )

    elif preset == LLMGuidedMCTSPreset.BENCHMARK:
        return LLMGuidedMCTSConfig(
            name="benchmark",
            description="Configuration for HumanEval benchmarking",
            num_iterations=C.DEFAULT_ITERATIONS_BALANCED,
            max_depth=C.DEFAULT_MAX_DEPTH,
            max_children=C.DEFAULT_MAX_CHILDREN,
            exploration_weight=C.UCB1_EXPLORATION_CONSTANT,
            execution_timeout_seconds=10.0,
            generator_config=GeneratorConfig(
                model=C.DEFAULT_GENERATOR_MODEL,
                temperature=C.DEFAULT_GENERATOR_TEMPERATURE,
                num_variants=C.DEFAULT_NUM_VARIANTS,
            ),
            reflector_config=ReflectorConfig(
                model=C.DEFAULT_REFLECTOR_MODEL,
                temperature=C.DEFAULT_REFLECTOR_TEMPERATURE,
            ),
            collect_training_data=True,
            save_mcts_policy=True,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")


def get_preset_config(preset: str | LLMGuidedMCTSPreset) -> LLMGuidedMCTSConfig:
    """
    Get a preset configuration by name.

    Args:
        preset: Preset name string or LLMGuidedMCTSPreset enum

    Returns:
        LLMGuidedMCTSConfig with preset parameters

    Raises:
        ValueError: If preset name is not recognized
    """
    if isinstance(preset, str):
        preset_map = {
            "fast": LLMGuidedMCTSPreset.FAST,
            "balanced": LLMGuidedMCTSPreset.BALANCED,
            "thorough": LLMGuidedMCTSPreset.THOROUGH,
            "data_collection": LLMGuidedMCTSPreset.DATA_COLLECTION,
            "benchmark": LLMGuidedMCTSPreset.BENCHMARK,
        }
        preset_enum = preset_map.get(preset.lower())
        if preset_enum is None:
            raise ValueError(f"Unknown preset: {preset}. Available presets: {list(preset_map.keys())}")
        return create_llm_mcts_preset(preset_enum)
    else:
        return create_llm_mcts_preset(preset)


# Default configuration for easy access
DEFAULT_LLM_MCTS_CONFIG = LLMGuidedMCTSConfig()
FAST_LLM_MCTS_CONFIG = create_llm_mcts_preset(LLMGuidedMCTSPreset.FAST)
BALANCED_LLM_MCTS_CONFIG = create_llm_mcts_preset(LLMGuidedMCTSPreset.BALANCED)
THOROUGH_LLM_MCTS_CONFIG = create_llm_mcts_preset(LLMGuidedMCTSPreset.THOROUGH)
DATA_COLLECTION_CONFIG = create_llm_mcts_preset(LLMGuidedMCTSPreset.DATA_COLLECTION)
