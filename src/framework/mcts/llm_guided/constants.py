"""
Configuration Constants for LLM-Guided MCTS.

Centralizes all configurable values, magic numbers, and defaults
to ensure consistency and ease of customization across the module.

Environment variables can override these defaults where noted.
"""

from __future__ import annotations

import os
from typing import Final

# ============================================================================
# Model Configuration
# ============================================================================

# Default model names - can be overridden via environment
DEFAULT_GENERATOR_MODEL: Final[str] = os.getenv("MCTS_GENERATOR_MODEL", "gpt-4o")
DEFAULT_REFLECTOR_MODEL: Final[str] = os.getenv("MCTS_REFLECTOR_MODEL", "gpt-4o")
DEFAULT_TOKENIZER_NAME: Final[str] = os.getenv("MCTS_TOKENIZER_NAME", "gpt2")

# Fast preset models (lighter weight for quick iterations)
DEFAULT_GENERATOR_MODEL_FAST: Final[str] = os.getenv("MCTS_GENERATOR_MODEL_FAST", "gpt-4o-mini")
DEFAULT_REFLECTOR_MODEL_FAST: Final[str] = os.getenv("MCTS_REFLECTOR_MODEL_FAST", "gpt-4o-mini")

# Model parameters
DEFAULT_GENERATOR_TEMPERATURE: Final[float] = 0.7
DEFAULT_REFLECTOR_TEMPERATURE: Final[float] = 0.3
DEFAULT_GENERATOR_MAX_TOKENS: Final[int] = 2000
DEFAULT_REFLECTOR_MAX_TOKENS: Final[int] = 1000
DEFAULT_NUM_VARIANTS: Final[int] = 3
DEFAULT_TOP_P: Final[float] = 0.95

# Temperature validation bounds
TEMPERATURE_MIN: Final[float] = 0.0
TEMPERATURE_MAX: Final[float] = 2.0

# Token validation bounds
MAX_TOKENS_MIN: Final[int] = 1
MAX_TOKENS_MAX: Final[int] = 8000
REFLECTOR_MAX_TOKENS_MAX: Final[int] = 4000

# Variant validation bounds
NUM_VARIANTS_MIN: Final[int] = 1
NUM_VARIANTS_MAX: Final[int] = 10

# Top-p validation bounds
TOP_P_MIN: Final[float] = 0.0
TOP_P_MAX: Final[float] = 1.0

# ============================================================================
# MCTS Algorithm Configuration
# ============================================================================

# UCB1 exploration constant (sqrt(2) is theoretically optimal)
UCB1_EXPLORATION_CONSTANT: Final[float] = 1.414

# Default iterations for different presets
DEFAULT_ITERATIONS_FAST: Final[int] = 10
DEFAULT_ITERATIONS_BALANCED: Final[int] = 30
DEFAULT_ITERATIONS_THOROUGH: Final[int] = 100
DEFAULT_ITERATIONS_BENCHMARK: Final[int] = 50

# Iteration validation bounds
ITERATIONS_MIN: Final[int] = 1
ITERATIONS_MAX: Final[int] = 1000

# Exploration weight validation bounds
EXPLORATION_WEIGHT_MIN: Final[float] = 0.0
EXPLORATION_WEIGHT_MAX: Final[float] = 10.0

# Tree depth and breadth limits
DEFAULT_MAX_DEPTH: Final[int] = 10
DEFAULT_MAX_CHILDREN: Final[int] = 5
MIN_DEPTH: Final[int] = 1
MIN_CHILDREN: Final[int] = 1
MAX_DEPTH_LIMIT: Final[int] = 50
MAX_CHILDREN_LIMIT: Final[int] = 20

# Solution confidence threshold and bounds
DEFAULT_SOLUTION_CONFIDENCE_THRESHOLD: Final[float] = 0.95
CONFIDENCE_MIN: Final[float] = 0.0
CONFIDENCE_MAX: Final[float] = 1.0

# Best node selection weight for visit count
BEST_NODE_VISIT_WEIGHT: Final[float] = 0.1

# Logging interval for iterations
LOG_PROGRESS_EVERY_N_ITERATIONS: Final[int] = 10

# ============================================================================
# Execution Configuration
# ============================================================================

# Timeouts (can be overridden via environment)
DEFAULT_EXECUTION_TIMEOUT: Final[float] = float(os.getenv("MCTS_EXECUTION_TIMEOUT", "5.0"))
BENCHMARK_TIMEOUT_PER_PROBLEM: Final[float] = float(os.getenv("BENCHMARK_TIMEOUT_PER_PROBLEM", "60.0"))
TIMEOUT_SIGNAL_PADDING: Final[int] = 1

# Execution timeout validation bounds
EXECUTION_TIMEOUT_MIN: Final[float] = 0.1
EXECUTION_TIMEOUT_MAX: Final[float] = 60.0

# Memory limits (can be overridden via environment)
DEFAULT_MAX_MEMORY_MB: Final[int] = int(os.getenv("MCTS_MAX_MEMORY_MB", "256"))

# Memory validation bounds
MAX_MEMORY_MIN_MB: Final[int] = 32
MAX_MEMORY_MAX_MB: Final[int] = 2048

# Benchmark settings
DEFAULT_BENCHMARK_MAX_CONCURRENT: Final[int] = 1

# ============================================================================
# Agent Fallback Values
# ============================================================================

# Confidence values for various fallback scenarios
DEFAULT_CONFIDENCE_FALLBACK: Final[float] = 0.5
CONFIDENCE_ON_PARSE_FAILURE: Final[float] = 0.5
LAST_RESORT_CONFIDENCE: Final[float] = 0.3

# Value fallbacks for Reflector
DEFAULT_VALUE_FALLBACK: Final[float] = 0.5
VALUE_ON_PARSE_FAILURE: Final[float] = 0.5

# Truncation limits
REFLECTION_TRUNCATE_CHARS: Final[int] = 500

# ============================================================================
# RAG Configuration
# ============================================================================

DEFAULT_RAG_TOP_K: Final[int] = 5
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.7
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 300
DEFAULT_CACHE_MAX_SIZE: Final[int] = 1000

# Context limits for RAG
CONTEXT_MAX_SOLUTIONS: Final[int] = 3
CONTEXT_MAX_PATTERNS: Final[int] = 2
CONTEXT_MAX_DOCS: Final[int] = 2
MAX_CONTEXT_LENGTH: Final[int] = 4000

# Prompt builder limits
MAX_DOCS_IN_PROMPT: Final[int] = 2
DOC_CONTENT_MAX_CHARS: Final[int] = 500
STDOUT_MAX_CHARS: Final[int] = 500
BRIEF_CODE_LINES: Final[int] = 10

# ============================================================================
# Training Configuration
# ============================================================================

DEFAULT_NUM_EPOCHS: Final[int] = 10
DEFAULT_LEARNING_RATE: Final[float] = 1e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 0.01
DEFAULT_WARMUP_STEPS: Final[int] = 100
DEFAULT_MAX_GRAD_NORM: Final[float] = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 5

# Network dimensions
GPT2_VOCAB_SIZE: Final[int] = 50257
DEFAULT_HIDDEN_DIM: Final[int] = 256
DEFAULT_NUM_TRANSFORMER_LAYERS: Final[int] = 4
DEFAULT_NUM_ATTENTION_HEADS: Final[int] = 8
DEFAULT_FEEDFORWARD_DIM: Final[int] = 1024
DEFAULT_MAX_SEQUENCE_LENGTH: Final[int] = 2048
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_MAX_ACTIONS: Final[int] = 10
DEFAULT_POLICY_NUM_LAYERS: Final[int] = 2
DEFAULT_VALUE_NUM_LAYERS: Final[int] = 2

# Dataset configuration
DEFAULT_MAX_CODE_LENGTH: Final[int] = 2048
DEFAULT_MAX_PROBLEM_LENGTH: Final[int] = 1024
DEFAULT_MIN_VISITS: Final[int] = 1

# Checkpoint configuration
DEFAULT_SAVE_EVERY_EPOCHS: Final[int] = 1
DEFAULT_KEEP_LAST_N_CHECKPOINTS: Final[int] = 3
DEFAULT_LOG_EVERY_STEPS: Final[int] = 100
DEFAULT_EVAL_EVERY_EPOCHS: Final[int] = 1

# Data collection
DEFAULT_BATCH_SIZE: Final[int] = 100

# ============================================================================
# Benchmark Metrics
# ============================================================================

DEFAULT_PASS_AT_K_VALUES: Final[tuple[int, ...]] = (1, 5, 10)

# Difficulty estimation thresholds
EASY_MAX_LINES: Final[int] = 5
EASY_MAX_TESTS: Final[int] = 3
MEDIUM_MAX_LINES: Final[int] = 15
MEDIUM_MAX_TESTS: Final[int] = 6

# Display limits
MAX_TEST_CASES_TO_SHOW: Final[int] = 3

# ============================================================================
# Security Configuration
# ============================================================================

# Dangerous patterns for code validation
DANGEROUS_PATTERNS: Final[tuple[str, ...]] = (
    "os.system",
    "subprocess",
    "eval(",
    "exec(",
    "__import__",
    "open(",
    "file(",
)

# Allowed imports for sandboxed execution
ALLOWED_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        "math",
        "string",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "heapq",
        "bisect",
        "array",
        "copy",
        "pprint",
        "reprlib",
        "enum",
        "graphlib",
        "typing",
        "types",
        "dataclasses",
        "abc",
        "contextlib",
        "decimal",
        "fractions",
        "numbers",
        "cmath",
        "statistics",
        "random",
    }
)

# ============================================================================
# File Paths and Directories
# ============================================================================

DEFAULT_TRAINING_DATA_DIR: Final[str] = os.getenv("MCTS_TRAINING_DATA_DIR", "./training_data")
DEFAULT_CHECKPOINT_DIR: Final[str] = os.getenv("MCTS_CHECKPOINT_DIR", "./checkpoints")
DEFAULT_BENCHMARK_OUTPUT_DIR: Final[str] = os.getenv("BENCHMARK_OUTPUT_DIR", "./benchmark_results")

# ============================================================================
# Random Seed
# ============================================================================

DEFAULT_SEED: Final[int] = 42

# ============================================================================
# Reward Transformation
# ============================================================================

# Transform reflector value [0,1] to reward [-1,1]: reward = value * 2 - 1
REWARD_SCALING_FACTOR: Final[float] = 2.0
REWARD_OFFSET: Final[float] = 1.0
