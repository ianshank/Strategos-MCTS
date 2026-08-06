"""
Configuration constants for application settings.

Centralizes default values and validation bounds to ensure
consistency and ease of customization across the application.
These constants provide shared defaults and bounds for
configuration-related code throughout the project.
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# MCTS Configuration
# ============================================================================

# MCTS iteration defaults and bounds
DEFAULT_MCTS_ITERATIONS: Final[int] = 100
MIN_MCTS_ITERATIONS: Final[int] = 1
MAX_MCTS_ITERATIONS: Final[int] = 10000

# MCTS exploration weight (UCB1 constant)
DEFAULT_MCTS_C: Final[float] = 1.414  # sqrt(2) is theoretically optimal
MIN_MCTS_C: Final[float] = 0.0
MAX_MCTS_C: Final[float] = 10.0

# MCTS candidate-scoring seam. The scorer decides which candidate action wins after
# the search returns per-candidate statistics. "identity" preserves the engine's own
# MAX_VISITS selection (behaviour-preserving default); "value" re-ranks by mean value.
CANDIDATE_SCORER_IDENTITY: Final[str] = "identity"
CANDIDATE_SCORER_VALUE: Final[str] = "value"
CANDIDATE_SCORER_NAMES: Final[tuple[str, ...]] = (CANDIDATE_SCORER_IDENTITY, CANDIDATE_SCORER_VALUE)
DEFAULT_CANDIDATE_SCORER: Final[str] = CANDIDATE_SCORER_IDENTITY

# Risk-averse subgoal scoring: score = value - lambda * dispersion (off by default).
# lambda is the dispersion-penalty weight (>= 0); dispersion is read from candidate
# metadata under this key (populated by the coarse-dynamics MDN once integrated).
# NOTE: CANDIDATE_SCORER_RISK_AVERSE is intentionally NOT in CANDIDATE_SCORER_NAMES /
# the GRAPH_MCTS_CANDIDATE_SCORER enum. The risk scorer needs a lambda + dispersion
# source (not name-only construction), so it is wired directly and flag-gated via
# ENABLE_UNCERTAINTY_SUBGOAL_PENALTY rather than selected through the string registry
# factory (create_candidate_scorer). The constant names the scorer for logs/telemetry.
CANDIDATE_SCORER_RISK_AVERSE: Final[str] = "risk_averse"
DEFAULT_SUBGOAL_UNCERTAINTY_LAMBDA: Final[float] = 1.0
MIN_SUBGOAL_UNCERTAINTY_LAMBDA: Final[float] = 0.0
MAX_SUBGOAL_UNCERTAINTY_LAMBDA: Final[float] = 100.0
RISK_DISPERSION_METADATA_KEY: Final[str] = "dispersion"

# Default seed for reproducibility
DEFAULT_SEED: Final[int] = 42

# Coarse-dynamics Mixture Density Network (MDN) — defaults and bounds.
# Number of low-level states aggregated into one coarse (multi-step) transition.
DEFAULT_COARSE_WINDOW: Final[int] = 4
MIN_COARSE_WINDOW: Final[int] = 1
MAX_COARSE_WINDOW: Final[int] = 1024
# Mixture component count (K) of the diagonal-Gaussian MDN head.
DEFAULT_MDN_COMPONENTS: Final[int] = 5
MIN_MDN_COMPONENTS: Final[int] = 1
MAX_MDN_COMPONENTS: Final[int] = 64
# Hidden width of the MDN trunk.
DEFAULT_MDN_HIDDEN_DIM: Final[int] = 128

# ============================================================================
# Network Configuration
# ============================================================================

# HTTP timeout configuration
DEFAULT_HTTP_TIMEOUT_SECONDS: Final[int] = 30
MIN_HTTP_TIMEOUT_SECONDS: Final[int] = 1
MAX_HTTP_TIMEOUT_SECONDS: Final[int] = 300

# HTTP retry configuration
DEFAULT_HTTP_MAX_RETRIES: Final[int] = 3
MIN_HTTP_MAX_RETRIES: Final[int] = 0
MAX_HTTP_MAX_RETRIES: Final[int] = 10

# ============================================================================
# Graph Orchestration Hardening
# ============================================================================

# Retry-with-backoff for LangGraph worker-node I/O boundaries.
DEFAULT_GRAPH_NODE_RETRY_MAX_ATTEMPTS: Final[int] = 3
MIN_GRAPH_NODE_RETRY_ATTEMPTS: Final[int] = 1
MAX_GRAPH_NODE_RETRY_ATTEMPTS: Final[int] = 10

DEFAULT_GRAPH_NODE_RETRY_INITIAL_DELAY_SECONDS: Final[float] = 0.5
MIN_GRAPH_NODE_RETRY_DELAY_SECONDS: Final[float] = 0.0
MAX_GRAPH_NODE_RETRY_DELAY_SECONDS: Final[float] = 30.0

DEFAULT_GRAPH_NODE_RETRY_BACKOFF_FACTOR: Final[float] = 2.0
MIN_GRAPH_NODE_RETRY_BACKOFF_FACTOR: Final[float] = 1.0
MAX_GRAPH_NODE_RETRY_BACKOFF_FACTOR: Final[float] = 10.0

# Default transient-exception allowlist (bare builtins + dotted LLM-adapter exceptions).
# Deliberately excludes non-transient errors (auth, invalid-request, context-length) and
# CircuitBreakerOpenError (retrying would defeat the breaker).
DEFAULT_GRAPH_NODE_RETRY_EXCEPTIONS: Final[tuple[str, ...]] = (
    "TimeoutError",
    "ConnectionError",
    "src.adapters.llm.exceptions.LLMTimeoutError",
    "src.adapters.llm.exceptions.LLMConnectionError",
    "src.adapters.llm.exceptions.LLMServerError",
    "src.adapters.llm.exceptions.LLMRateLimitError",
)

# Execution-trace logging for graph node transitions.
DEFAULT_TRACE_DIGEST_HEX_CHARS: Final[int] = 16
GRAPH_TRACE_LOGGER_NAME: Final[str] = "graph.trace"

# Per-invocation run-id length (hex chars) used for graph trace correlation.
GRAPH_RUN_ID_HEX_CHARS: Final[int] = 12

# Benchmark run-store layout (kill-safe incremental persistence).
BENCHMARK_RUNS_SUBDIR: Final[str] = "runs"
BENCHMARK_RESULTS_LOG_FILENAME: Final[str] = "results.jsonl"
BENCHMARK_RUN_MANIFEST_FILENAME: Final[str] = "run.json"

# ============================================================================
# Security Configuration
# ============================================================================

# Query length limits
DEFAULT_MAX_QUERY_LENGTH: Final[int] = 10000
MIN_MAX_QUERY_LENGTH: Final[int] = 1
MAX_MAX_QUERY_LENGTH: Final[int] = 100000

# Rate limiting
DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 60
MIN_RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 1
MAX_RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 1000

# ============================================================================
# Framework Configuration
# ============================================================================

# Framework iteration limits
DEFAULT_FRAMEWORK_MAX_ITERATIONS: Final[int] = 3
MIN_FRAMEWORK_MAX_ITERATIONS: Final[int] = 1
MAX_FRAMEWORK_MAX_ITERATIONS: Final[int] = 100

# Consensus threshold for agent agreement
DEFAULT_CONSENSUS_THRESHOLD: Final[float] = 0.75
MIN_CONSENSUS_THRESHOLD: Final[float] = 0.0
MAX_CONSENSUS_THRESHOLD: Final[float] = 1.0

# RAG retrieval configuration
DEFAULT_TOP_K_RETRIEVAL: Final[int] = 5
MIN_TOP_K_RETRIEVAL: Final[int] = 1
MAX_TOP_K_RETRIEVAL: Final[int] = 100

# ============================================================================
# LLM Configuration
# ============================================================================

# Default temperature for LLM generation
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.7
MIN_LLM_TEMPERATURE: Final[float] = 0.0
MAX_LLM_TEMPERATURE: Final[float] = 2.0

# Per-role LLM temperature defaults (intentionally distinct: reasoning agents are
# more deterministic than open-ended generation; chess move selection lower still).
# Centralized here so the individual agent modules reference a single source.
DEFAULT_HRM_TEMPERATURE: Final[float] = 0.5
DEFAULT_TRM_TEMPERATURE: Final[float] = 0.5
DEFAULT_CHESS_LLM_TEMPERATURE: Final[float] = 0.3

# Confidence thresholds
DEFAULT_CONFIDENCE_WITH_RAG: Final[float] = 0.8
DEFAULT_CONFIDENCE_WITHOUT_RAG: Final[float] = 0.7
DEFAULT_CONFIDENCE_ON_ERROR: Final[float] = 0.3

# Error response preview length
DEFAULT_ERROR_QUERY_PREVIEW_LENGTH: Final[int] = 100

# ============================================================================
# S3 Storage Configuration
# ============================================================================

DEFAULT_S3_PREFIX: Final[str] = "mcts-artifacts"
DEFAULT_S3_REGION: Final[str] = "us-east-1"

# S3 bucket name validation bounds
MIN_S3_BUCKET_NAME_LENGTH: Final[int] = 3
MAX_S3_BUCKET_NAME_LENGTH: Final[int] = 63

# ============================================================================
# API Key Validation
# ============================================================================

# Minimum API key lengths for validation
MIN_OPENAI_API_KEY_LENGTH: Final[int] = 20
MIN_ANTHROPIC_API_KEY_LENGTH: Final[int] = 20
MIN_PINECONE_API_KEY_LENGTH: Final[int] = 20
MIN_LANGSMITH_API_KEY_LENGTH: Final[int] = 20
MIN_WANDB_API_KEY_LENGTH: Final[int] = 20

# API key prefixes
OPENAI_API_KEY_PREFIX: Final[str] = "sk-"
ANTHROPIC_API_KEY_PREFIX: Final[str] = "sk-ant-"

# Placeholder values that should be rejected
API_KEY_PLACEHOLDERS: Final[tuple[str, ...]] = (
    "",
    "your-api-key-here",
    "REPLACE_ME",
    "your_api_key_here",
    "YOUR_API_KEY",
)

# ============================================================================
# Project Names
# ============================================================================

DEFAULT_LANGSMITH_PROJECT: Final[str] = "langgraph-mcts"
DEFAULT_WANDB_PROJECT: Final[str] = "langgraph-mcts"
DEFAULT_WANDB_MODE: Final[str] = "online"

# ============================================================================
# API Endpoints
# ============================================================================

DEFAULT_LANGCHAIN_ENDPOINT: Final[str] = "https://api.smith.langchain.com"
DEFAULT_LMSTUDIO_URL: Final[str] = "http://localhost:1234/v1"
DEFAULT_SERVER_HOST: Final[str] = "0.0.0.0"
DEFAULT_OPENAI_BASE_URL: Final[str] = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL: Final[str] = "https://api.anthropic.com"
DEFAULT_OTLP_ENDPOINT: Final[str] = "localhost:4317"
DEFAULT_OTLP_HTTP_ENDPOINT: Final[str] = "http://localhost:4317"

# ============================================================================
# LLM Provider Defaults
# ============================================================================

# Default model names
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4-turbo-preview"
DEFAULT_ANTHROPIC_MODEL: Final[str] = "claude-3-5-sonnet-20241022"
DEFAULT_LMSTUDIO_MODEL: Final[str] = "local-model"
DEFAULT_GOOGLE_GEMINI_MODEL: Final[str] = "gemini-2.0-flash-001"

# LLM-guided MCTS provider defaults (distinct from the adapter defaults above;
# tests pin these exact values)
DEFAULT_LLM_MCTS_OPENAI_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_LLM_MCTS_ANTHROPIC_MODEL: Final[str] = "claude-sonnet-4-20250514"

# Anthropic Messages API version header (``anthropic-version``). Centralized
# here so the REST adapter and the LLM-guided MCTS provider config stay in
# lockstep instead of duplicating the literal.
DEFAULT_ANTHROPIC_API_VERSION: Final[str] = "2023-06-01"

# Default timeouts per provider (seconds)
DEFAULT_OPENAI_TIMEOUT: Final[float] = 60.0
DEFAULT_ANTHROPIC_TIMEOUT: Final[float] = 120.0
DEFAULT_LMSTUDIO_TIMEOUT: Final[float] = 300.0
DEFAULT_LLM_TIMEOUT: Final[float] = 60.0

# Default max tokens
DEFAULT_MAX_TOKENS: Final[int] = 4096
DEFAULT_MAX_RETRIES: Final[int] = 5

# ============================================================================
# S3 Storage Defaults
# ============================================================================

DEFAULT_S3_BUCKET_NAME: Final[str] = "mcts-framework-storage"

# ============================================================================
# Service Metadata
# ============================================================================

SERVICE_VERSION: Final[str] = "0.1.0"
DEFAULT_ENVIRONMENT: Final[str] = "development"
DEFAULT_OTEL_SERVICE_NAME: Final[str] = "mcts-framework"

# ============================================================================
# Persistence / Serialization
# ============================================================================

# On-disk format versions for safe, forward-compatible (de)serialization.
# Bump when the persisted record schema changes.
SUBSTRUCTURE_LIBRARY_FORMAT_VERSION: Final[int] = 1
EXPERIENCE_BUFFER_FORMAT_VERSION: Final[int] = 1

# Substructure library sizing/matching defaults and bounds
DEFAULT_SUBSTRUCTURE_MAX_SIZE: Final[int] = 10000
MIN_SUBSTRUCTURE_MAX_SIZE: Final[int] = 1
MAX_SUBSTRUCTURE_MAX_SIZE: Final[int] = 1000000

DEFAULT_SUBSTRUCTURE_SIMILARITY_THRESHOLD: Final[float] = 0.7
MIN_SUBSTRUCTURE_SIMILARITY_THRESHOLD: Final[float] = 0.0
MAX_SUBSTRUCTURE_SIMILARITY_THRESHOLD: Final[float] = 1.0

# Experience replay buffer sizing defaults and bounds
DEFAULT_EXPERIENCE_BUFFER_MAX_SIZE: Final[int] = 100000
MIN_EXPERIENCE_BUFFER_MAX_SIZE: Final[int] = 1
MAX_EXPERIENCE_BUFFER_MAX_SIZE: Final[int] = 100000000

# Default device for loading persisted tensors. "cpu" is the safe default so a
# buffer saved on GPU can be reloaded on a CPU-only host; callers move tensors
# to the training device afterwards.
DEFAULT_TENSOR_LOAD_MAP_LOCATION: Final[str] = "cpu"

# ============================================================================
# Hybrid Agent Parsing Fallbacks
# ============================================================================

# Returned when an LLM response cannot be parsed into an action/value.
DEFAULT_HYBRID_ACTION_FALLBACK: Final[int] = 0
DEFAULT_HYBRID_VALUE_FALLBACK: Final[float] = 0.0

# ============================================================================
# Diagram Rendering (Kroki)
# ============================================================================

# Kroki service used to render Mermaid diagrams to image formats. Overridable so
# deployments can point at a self-hosted Kroki instance.
DEFAULT_KROKI_BASE_URL: Final[str] = "https://kroki.io"
DEFAULT_KROKI_TIMEOUT_SECONDS: Final[float] = 30.0

# ============================================================================
# Chess Agent Routing
# ============================================================================

# Confidence boost applied when a single agent clearly dominates the routing
# weights. Shared by the chess meta-controller and the LLM chess engine.
CHESS_ROUTING_CONFIDENCE_BOOST: Final[float] = 0.3

# ============================================================================
# Mock / Fallback (test & dev only)
# ============================================================================

# Text returned by the in-process mock LLM client. Centralized so it is greppable
# and can never be mistaken for a real model response.
MOCK_LLM_RESPONSE_TEXT: Final[str] = "This is a mock response for testing purposes."

# ============================================================================
# M5 Policy-Lift Benchmark (decision-quality gate)
# ============================================================================

# Lift (%) the confidence-interval lower bound must clear to pass the M5 gate.
M5_TARGET_LIFT_PCT: Final[float] = 20.0

# Relative lift is meaningless against a near-zero baseline; below this floor the
# benchmark reports an absolute-points delta instead of a ratio.
M5_MIN_BASELINE: Final[float] = 0.05

# Per-metric game counts: enough that the CI lower bound can clear the target when
# the effect is real (Wilson at n=100 resolves ~±0.10 on win-rate).
M5_DEFAULT_GAMES_WIN_RATE: Final[int] = 100
M5_DEFAULT_GAMES_MEAN_REWARD: Final[int] = 30

# MCTS simulations per move for the self-play convergence driver's default (cheap
# smoke/plumbing runs). Deliberately tiny — the full neural default
# (system_config.MCTSConfig.num_simulations = 1600) is far too expensive for a driver
# smoke run; real chess convergence overrides it via --num-simulations.
M5_DEFAULT_SELF_PLAY_SIMULATIONS: Final[int] = 16

# Confidence level for the lift interval (must be a key of the z-table in
# src/utils/stats.py: 0.90 / 0.95 / 0.99).
M5_DEFAULT_CONFIDENCE: Final[float] = 0.95

# Default MLP hidden layout mirrored from MLPPolicyValueNetwork's constructor default,
# used when reconstructing a network for a checkpoint without an architecture spec.
M5_DEFAULT_MLP_HIDDEN_DIMS: Final[tuple[int, ...]] = (512, 256)

# Board size assumed when deriving a default conv architecture for adversarial
# board domains (chess).
M5_DEFAULT_ADVERSARIAL_BOARD_SIZE: Final[int] = 8

# ============================================================================
# GPU & Training Configuration
# ============================================================================

DEFAULT_CUDA_MEMORY_FRACTION: Final[float] = 0.9
MIN_CUDA_MEMORY_FRACTION: Final[float] = 0.1
MAX_CUDA_MEMORY_FRACTION: Final[float] = 1.0
MIN_GPU_MEMORY_GB: Final[float] = 2.0
SUPPORTED_CUDA_BACKENDS: Final[tuple[str, ...]] = ("nccl", "gloo")

# ============================================================================
# Model Checkpoint Integrity
# ============================================================================

# Magic prefix of a Git-LFS pointer file. A repository cloned without
# `git lfs install && git lfs pull` leaves ~130-byte text stubs in place of the
# real weights, and `Path.exists()` happily returns True for them — so existence
# is not a usable readiness signal. Detecting the prefix lets callers degrade
# with an actionable message instead of dying inside a deserializer.
GIT_LFS_POINTER_MAGIC: Final[bytes] = b"version https://git-lfs.github.com/spec/v1"

# Pointer files are a few short lines; anything larger is real content. Bounds
# how much of a candidate file is read during inspection.
GIT_LFS_POINTER_MAX_BYTES: Final[int] = 1024

# Leading bytes of a Python pickle stream (protocol 2+), used by legacy
# (pre-1.6) torch checkpoints that are not zip archives.
PICKLE_PROTOCOL2_MAGIC: Final[bytes] = b"\x80"

# Weight file suffixes recognized when inspecting a checkpoint directory
# (e.g. a PEFT/LoRA adapter directory rather than a single file).
CHECKPOINT_WEIGHT_SUFFIXES: Final[tuple[str, ...]] = (".pt", ".pth", ".bin", ".safetensors", ".ckpt")

# Remediation hint surfaced when a checkpoint turns out to be an LFS pointer.
GIT_LFS_REMEDIATION: Final[str] = "git lfs install && git lfs pull"
