# Comprehensive Implementation Plan - LangGraph Multi-Agent MCTS

> **Design Principle:** This plan treats implementation as constraint-satisfaction, not checklist completion. Define success criteria, blockers, and verification commands—then systematically resolve.

**Version:** 3.0
**Created:** 2026-01-27
**Current Status:** 90% Feature Complete | 88.4% Tests Passing (771/872)
**Core MCTS Coverage:** 97.66%

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Immediate Priorities (Blocking Deployment)](#3-immediate-priorities-blocking-deployment)
4. [Short-Term Goals (2-4 Weeks)](#4-short-term-goals-2-4-weeks)
5. [Medium-Term Goals (1-2 Months)](#5-medium-term-goals-1-2-months)
6. [Long-Term Vision (Phase 4+)](#6-long-term-vision-phase-4)
7. [Implementation Patterns](#7-implementation-patterns)
8. [Sub-Agent Orchestration Strategy](#8-sub-agent-orchestration-strategy)
9. [Testing Strategy](#9-testing-strategy)
10. [Observability & Debugging](#10-observability--debugging)
11. [Backwards Compatibility](#11-backwards-compatibility)
12. [Verification Checklist](#12-verification-checklist)

---

## 1. Executive Summary

### 1.1 Project Overview

The LangGraph Multi-Agent MCTS framework is a production-ready DeepMind-style self-improving AI system combining:

- **Hierarchical Reasoning Module (HRM):** DeBERTa-based adaptive computation time agent
- **Task Refinement Module (TRM):** Iterative solution refinement with deep supervision
- **Neural Meta-Controller:** Dynamic routing across agents (RNN, BERT, Hybrid, Assembly variants)
- **Neural MCTS Engine:** AlphaZero-style tree search with progressive widening
- **LangGraph Orchestration:** StateGraph-based workflow with checkpointing
- **Enterprise Integrations:** Pinecone, W&B, LangSmith, S3, Braintrust

### 1.2 Current Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Overall Completion | 90% | 100% |
| Test Pass Rate | 88.4% (771/872) | 100% |
| Core MCTS Coverage | 97.66% | 95%+ |
| API Latency (p95) | ~100ms | <500ms |
| Memory Usage | ~500MB | <2GB |
| Docker Image Size | 1.08GB | <2GB |

### 1.3 Critical Blockers Summary

| Priority | Task | Test Failures | Effort |
|----------|------|---------------|--------|
| **CRITICAL** | MCTS Policy Interface Fix | 24 | 0.5 days |
| **HIGH** | RAG Integration Pipeline | - | 1 day |
| **HIGH** | HRMAgent Export Fix | 14 | 0.5 days |
| **MEDIUM** | DABStep Dataset Split | 6 | 0.5 days |
| **MEDIUM** | Collection Error Resolution | 47 | 1 day |

---

## 2. Current State Analysis

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REST API Layer                                │
│                     (FastAPI + Uvicorn)                             │
│  src/api/rest_server.py                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Framework Service                                │
│                 src/api/framework_service.py                        │
│  - Singleton pattern with lazy initialization                       │
│  - Supports full framework or lightweight fallback                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LangGraph Orchestration                          │
│                    src/framework/graph.py                           │
│  Nodes: entry → retrieve_context → route_decision → agents →        │
│         aggregate_results → evaluate_consensus → output              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │    HRM    │   │    TRM    │   │   MCTS    │
            │   Agent   │   │   Agent   │   │  Engine   │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                        ┌───────────────────┐
                        │  Meta-Controller  │
                        │ (RNN/BERT/Hybrid) │
                        └───────────────────┘
```

### 2.2 Key File Locations

```
CONFIGURATION
├── src/config/settings.py           # Pydantic Settings v2 (all config)
├── .env                             # Environment variables (secrets)
└── pyproject.toml                   # Dependencies, tool config

CORE FRAMEWORK
├── src/framework/graph.py           # LangGraph orchestration
├── src/framework/factories.py       # Component factories
├── src/framework/component_factory.py # Dynamic component creation
└── src/framework/mcts/
    ├── core.py                      # MCTS engine
    ├── config.py                    # MCTS configuration
    ├── policies.py                  # Selection/rollout policies (BLOCKER)
    ├── neural_mcts.py               # AlphaZero-style neural MCTS
    └── progressive_widening.py      # Action space management

AGENTS
├── src/agents/hrm_agent.py          # Hierarchical Reasoning Module
├── src/agents/trm_agent.py          # Task Refinement Module
├── src/agents/hybrid_agent.py       # LLM + Neural hybrid
└── src/agents/meta_controller/
    ├── base.py                      # Base interfaces
    ├── rnn_controller.py            # GRU-based routing
    ├── bert_controller.py           # DeBERTa-based routing
    ├── hybrid_controller.py         # Combined approaches
    └── assembly_router.py           # Assembly theory routing

LLM ADAPTERS
├── src/adapters/llm/base.py         # Protocol-based interface
├── src/adapters/llm/openai_client.py
├── src/adapters/llm/anthropic_client.py
└── src/adapters/llm/lmstudio_client.py

OBSERVABILITY
├── src/observability/logging.py     # Structured JSON logging
├── src/observability/metrics.py     # Prometheus metrics
├── src/observability/tracing.py     # OpenTelemetry integration
└── src/observability/profiling.py   # Performance profiling

TESTS
├── tests/unit/                      # Fast, isolated tests
├── tests/integration/               # Component interaction tests
├── tests/e2e/                       # End-to-end scenarios
├── tests/framework/mcts/            # MCTS-specific tests
└── tests/conftest.py                # Root fixtures & configuration
```

### 2.3 Test Failure Breakdown

| Category | Failures | Root Cause | Resolution Path |
|----------|----------|------------|-----------------|
| MCTS Policy Interface | 24 | Signature mismatch in `RolloutPolicy.evaluate()` | Align interface signatures |
| HRM Agent Export | 14 | Module export issues in `__init__.py` | Fix exports, add missing imports |
| DABStep Dataset | 6 | Missing `split` parameter | Add split parameter support |
| Collection Errors | 47 | Optional dependencies not installed | Install `[dev,neural,vectordb]` |
| Other | 10 | Various | Individual investigation |

---

## 3. Immediate Priorities (Blocking Deployment)

### 3.1 CRITICAL: MCTS Policy Interface Fix

**Status:** 🔄 IN PROGRESS
**Location:** `src/framework/mcts/policies.py`
**Failures:** 24 tests
**Effort:** 0.5 days

**Problem:**
The `RolloutPolicy` abstract base class signature doesn't match implementations.

**Current Interface:**
```python
class RolloutPolicy(ABC):
    @abstractmethod
    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
    ) -> float:
        pass
```

**Required Interface:**
```python
class RolloutPolicy(ABC):
    @abstractmethod
    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """
        Evaluate state and return value in [0, 1].

        Args:
            state: Current MCTS state
            rng: Random number generator for reproducibility
            max_depth: Maximum rollout depth (default: 10)

        Returns:
            Value estimate between 0 and 1
        """
        pass
```

**Implementation Steps:**

1. **Update abstract interface** in `src/framework/mcts/policies.py`:
   ```python
   @abstractmethod
   async def evaluate(
       self,
       state: MCTSState,
       rng: np.random.Generator,
       max_depth: int = 10,
   ) -> float:
   ```

2. **Update all implementations:**
   - `RandomRolloutPolicy`
   - `NeuralRolloutPolicy`
   - `HybridRolloutPolicy`
   - `LLMGuidedRolloutPolicy` (if exists)

3. **Add backwards compatibility wrapper** (see Section 11):
   ```python
   def _ensure_max_depth_param(func):
       """Backwards compatibility for old evaluate signatures."""
       sig = inspect.signature(func)
       if 'max_depth' not in sig.parameters:
           @functools.wraps(func)
           async def wrapper(self, state, rng, max_depth=10):
               return await func(self, state, rng)
           return wrapper
       return func
   ```

4. **Run verification:**
   ```bash
   pytest tests/unit/test_mcts_core.py -v -k "policy"
   pytest tests/framework/mcts/ -v
   ```

---

### 3.2 HIGH: RAG Integration Pipeline

**Status:** 🔄 IN PROGRESS
**Location:** `src/api/framework_service.py`, `src/storage/pinecone_store.py`
**Effort:** 1 day

**Current State:**
RAG retriever exists but not fully integrated into the query pipeline.

**Required Integration:**

```python
# src/api/framework_service.py
class FrameworkService:
    async def process_query(
        self,
        query: str,
        use_rag: bool = True,
        thread_id: str | None = None,
        **kwargs,
    ) -> QueryResult:
        context = ""
        retrieved_docs = []

        # RAG Integration
        if use_rag and self._rag_retriever:
            try:
                retrieved_docs = await self._rag_retriever.search(
                    query=query,
                    top_k=self._config.top_k_retrieval,
                    filters=kwargs.get("rag_filters"),
                )
                context = self._format_retrieved_context(retrieved_docs)

                self._logger.info(
                    "RAG context retrieved",
                    extra={
                        "correlation_id": get_correlation_id(),
                        "num_docs": len(retrieved_docs),
                        "context_length": len(context),
                    }
                )
            except RAGError as e:
                self._logger.warning(
                    "RAG retrieval failed, continuing without context",
                    extra={"error": str(e)},
                )

        # Process with context
        initial_state: AgentState = {
            "query": query,
            "use_mcts": kwargs.get("use_mcts", self._config.mcts_enabled),
            "use_rag": use_rag,
            "rag_context": context,
            "retrieved_docs": [doc.model_dump() for doc in retrieved_docs],
            "iteration": 0,
            "max_iterations": self._config.max_iterations,
            "agent_outputs": [],
        }

        result = await self._graph.ainvoke(initial_state)
        return QueryResult.from_state(result)

    def _format_retrieved_context(
        self,
        docs: list[RetrievedDocument],
    ) -> str:
        """Format retrieved documents into context string."""
        if not docs:
            return ""

        formatted = ["## Retrieved Context\n"]
        for i, doc in enumerate(docs, 1):
            formatted.append(f"### Document {i} (relevance: {doc.score:.2f})")
            formatted.append(doc.content)
            formatted.append("")

        return "\n".join(formatted)
```

**Verification:**
```bash
pytest tests/integration/test_rag_integration.py -v
pytest tests/api/test_rest_endpoints.py -v -k "rag"
```

---

### 3.3 HIGH: HRM Agent Export Fix

**Status:** 🔴 NOT STARTED
**Location:** `src/agents/__init__.py`, `src/agents/hrm_agent.py`
**Failures:** 14 tests
**Effort:** 0.5 days

**Problem:**
HRMAgent and related classes not properly exported from module.

**Fix:**

```python
# src/agents/__init__.py
from src.agents.hrm_agent import (
    HRMAgent,
    HRMConfig,
    HRMOutput,
    SubProblem,
    AdaptiveComputationTime,
)
from src.agents.trm_agent import (
    TRMAgent,
    TRMConfig,
    TRMOutput,
    RecursiveBlock,
)
from src.agents.hybrid_agent import (
    HybridAgent,
    HybridConfig,
    DecisionSource,
)

__all__ = [
    # HRM
    "HRMAgent",
    "HRMConfig",
    "HRMOutput",
    "SubProblem",
    "AdaptiveComputationTime",
    # TRM
    "TRMAgent",
    "TRMConfig",
    "TRMOutput",
    "RecursiveBlock",
    # Hybrid
    "HybridAgent",
    "HybridConfig",
    "DecisionSource",
]
```

---

### 3.4 MEDIUM: DABStep Dataset Split Parameter

**Status:** 🔴 NOT STARTED
**Location:** Dataset loading utilities
**Failures:** 6 tests
**Effort:** 0.5 days

**Problem:**
DABStep dataset loader doesn't support `split` parameter.

**Fix:**
```python
def load_dabstep_dataset(
    path: str,
    split: str = "train",  # Add this parameter
    **kwargs,
) -> Dataset:
    """Load DABStep dataset with split support."""
    return load_dataset(
        path,
        split=split,
        **kwargs,
    )
```

---

## 4. Short-Term Goals (2-4 Weeks)

### 4.1 Achieve 100% Test Pass Rate

**Target:** All 872 tests passing

**Strategy:**
1. Fix MCTS Policy Interface (24 tests)
2. Fix HRM Agent Exports (14 tests)
3. Fix DABStep Dataset (6 tests)
4. Resolve collection errors (47 tests)
5. Investigate remaining 10 failures

**Verification:**
```bash
# Install all optional dependencies
pip install -e ".[dev,neural,experiment,vectordb]"

# Run full test suite
pytest tests/ -v --tb=short

# Generate coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### 4.2 Deploy to Staging

**Prerequisites:**
- [ ] 100% test pass rate
- [ ] Docker build successful
- [ ] Smoke tests pass
- [ ] Monitoring configured

**Deployment Steps:**
```bash
# Build Docker image
docker build -t langgraph-mcts:staging .

# Run smoke tests
docker run --rm langgraph-mcts:staging pytest tests/smoke -v

# Deploy to staging
kubectl apply -f kubernetes/staging/

# Verify health
curl https://staging.api/health
```

### 4.3 Production Hardening

**Performance Targets:**

| Metric | Target | How to Achieve |
|--------|--------|----------------|
| Query Latency (p95) | <500ms | Response streaming, caching |
| MCTS Iterations/sec | >100 | Parallel rollouts, GPU acceleration |
| Memory Usage | <2GB | LRU cache tuning, memory pooling |

**Implementation:**

```python
# src/framework/mcts/parallel_mcts.py
class ParallelMCTS:
    """MCTS with parallel rollouts for performance."""

    def __init__(
        self,
        config: MCTSConfig,
        max_workers: int | None = None,
    ):
        self._config = config
        self._max_workers = max_workers or min(32, os.cpu_count() or 4)
        self._semaphore = asyncio.Semaphore(self._max_workers)

    async def search(
        self,
        initial_state: MCTSState,
    ) -> SearchResult:
        """Execute parallel MCTS search."""
        tasks = []
        for _ in range(self._config.num_iterations):
            tasks.append(self._run_iteration(initial_state))

        results = await asyncio.gather(*tasks)
        return self._aggregate_results(results)

    async def _run_iteration(self, state: MCTSState) -> IterationResult:
        async with self._semaphore:
            # Selection, Expansion, Simulation, Backpropagation
            ...
```

---

## 5. Medium-Term Goals (1-2 Months)

### 5.1 Full Production Deployment

**Infrastructure:**
- Kubernetes cluster with auto-scaling
- Multi-region deployment
- Blue-green deployment strategy
- Comprehensive monitoring (Prometheus + Grafana)

### 5.2 Multi-GPU Distributed Training

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Orchestrator                         │
│               src/training/unified_orchestrator.py              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │  GPU 0  │     │  GPU 1  │     │  GPU N  │
        │  HRM    │     │  TRM    │     │  MCTS   │
        └─────────┘     └─────────┘     └─────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Gradient Sync  │
                    │  (DistributedDP) │
                    └─────────────────┘
```

**Configuration:**
```python
# src/training/distributed_config.py
@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""

    # From environment - no hardcoded values
    world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1)))
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", 0)))
    backend: str = field(default_factory=lambda: os.environ.get("DIST_BACKEND", "nccl"))

    # Configurable via settings
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    find_unused_parameters: bool = False
```

### 5.3 Hyperparameter Tuning

**Tools:** Optuna, Ray Tune, W&B Sweeps

**Search Space:**
```python
# src/training/hyperparameter_search.py
def create_search_space() -> dict:
    """Define hyperparameter search space."""
    return {
        # MCTS Parameters
        "mcts_iterations": optuna.distributions.IntDistribution(50, 500),
        "mcts_exploration_weight": optuna.distributions.FloatDistribution(0.5, 2.5),
        "mcts_progressive_widening_k": optuna.distributions.FloatDistribution(0.5, 2.0),
        "mcts_progressive_widening_alpha": optuna.distributions.FloatDistribution(0.25, 0.75),

        # Agent Parameters
        "hrm_max_depth": optuna.distributions.IntDistribution(3, 10),
        "hrm_confidence_threshold": optuna.distributions.FloatDistribution(0.5, 0.9),
        "trm_max_iterations": optuna.distributions.IntDistribution(5, 20),
        "trm_convergence_threshold": optuna.distributions.FloatDistribution(0.001, 0.1),

        # Training Parameters
        "learning_rate": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        "batch_size": optuna.distributions.CategoricalDistribution([8, 16, 32, 64]),
        "weight_decay": optuna.distributions.FloatDistribution(0.0, 0.1),
    }
```

---

## 6. Long-Term Vision (Phase 4+)

### 6.1 Continuous Self-Play Training

**AlphaZero-style training loop:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Play Training Loop                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate games/episodes with current model                  │
│  2. Store (state, action, value) tuples in replay buffer        │
│  3. Sample mini-batch from replay buffer                        │
│  4. Train policy and value networks                             │
│  5. Evaluate new model against previous best                    │
│  6. If improved, update best model checkpoint                   │
│  7. Repeat from step 1                                          │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
# src/training/self_play.py
class SelfPlayTrainer:
    """Continuous self-play training loop."""

    def __init__(
        self,
        config: SelfPlayConfig,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        replay_buffer: ReplayBuffer,
        logger: logging.Logger,
    ):
        self._config = config
        self._policy = policy_network
        self._value = value_network
        self._buffer = replay_buffer
        self._logger = logger
        self._best_elo = 0.0

    async def run_training_loop(self) -> None:
        """Run continuous self-play training."""
        for iteration in range(self._config.max_iterations):
            # Generate games
            experiences = await self._generate_games(
                num_games=self._config.games_per_iteration,
            )

            # Store experiences
            self._buffer.add_batch(experiences)

            # Train networks
            train_metrics = await self._train_step()

            # Evaluate
            if iteration % self._config.eval_interval == 0:
                elo = await self._evaluate_model()
                if elo > self._best_elo:
                    self._best_elo = elo
                    await self._save_checkpoint(iteration)

            # Log progress
            self._logger.info(
                "Training iteration complete",
                extra={
                    "iteration": iteration,
                    "buffer_size": len(self._buffer),
                    "metrics": train_metrics,
                }
            )
```

### 6.2 Multi-Domain Training Expansion

**Planned Domains:**

| Domain | State Space | Actions | Reward Signal |
|--------|-------------|---------|---------------|
| Code Generation | AST + context | Edit, Insert, Delete | Test pass rate |
| Strategic Reasoning | Game state | Move, Evaluate | Win/lose/draw |
| Document Analysis | Document tree | Extract, Summarize | Info gain |
| Scientific Research | Paper + hypotheses | Experiment, Cite | Citation impact |

### 6.3 Neural-Symbolic Integration

**Architecture:**
```python
# src/neuro_symbolic/integration.py
class NeuroSymbolicAgent:
    """Combines neural networks with symbolic reasoning."""

    def __init__(
        self,
        neural_encoder: NeuralEncoder,
        symbolic_reasoner: SymbolicReasoner,
        constraint_solver: ConstraintSolver,
    ):
        self._encoder = neural_encoder
        self._reasoner = symbolic_reasoner
        self._solver = constraint_solver

    async def reason(self, query: str, context: dict) -> ReasoningResult:
        """
        1. Neural: Encode query and context into latent representation
        2. Symbolic: Extract logical constraints from representation
        3. Solve: Find solutions satisfying constraints
        4. Neural: Rank solutions using learned preferences
        """
        # Step 1: Neural encoding
        latent = await self._encoder.encode(query, context)

        # Step 2: Extract constraints
        constraints = self._reasoner.extract_constraints(latent)

        # Step 3: Solve constraints
        solutions = self._solver.solve(constraints)

        # Step 4: Rank solutions
        ranked = await self._encoder.rank(solutions, latent)

        return ReasoningResult(
            best_solution=ranked[0],
            alternatives=ranked[1:],
            constraints=constraints,
        )
```

---

## 7. Implementation Patterns

### 7.1 Configuration Pattern (No Hardcoded Values)

```python
# CORRECT: All configuration from Pydantic Settings
from src.config.settings import get_settings

settings = get_settings()
mcts = MCTSEngine(
    seed=settings.SEED,
    exploration_weight=settings.MCTS_C,
    num_iterations=settings.MCTS_ITERATIONS,
)

# WRONG: Hardcoded values
mcts = MCTSEngine(
    seed=42,                    # Should be settings.SEED
    exploration_weight=1.414,   # Should be settings.MCTS_C
    num_iterations=100,         # Should be settings.MCTS_ITERATIONS
)
```

### 7.2 Factory Pattern (Dependency Injection)

```python
# src/framework/factories.py
class LLMClientFactory:
    """Factory for creating LLM clients with settings injection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(
        self,
        provider: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> LLMClient:
        provider = provider or self._settings.LLM_PROVIDER.value

        creators = {
            "openai": self._create_openai,
            "anthropic": self._create_anthropic,
            "lmstudio": self._create_lmstudio,
        }

        if provider not in creators:
            raise ValueError(f"Unknown provider: {provider}")

        return creators[provider](model, **kwargs)

    def _create_openai(self, model: str | None, **kwargs) -> OpenAIClient:
        return OpenAIClient(
            api_key=self._settings.get_api_key(),
            model=model or "gpt-4",
            timeout=self._settings.HTTP_TIMEOUT_SECONDS,
            **kwargs,
        )
```

### 7.3 Protocol-Based Interfaces

```python
# src/adapters/llm/base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM interface using structural typing."""

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response from LLM."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Interface for all agents."""

    async def process(self, query: str, context: dict) -> AgentResult:
        ...

    def get_confidence(self) -> float:
        ...
```

### 7.4 Agent Pattern with Logging

```python
# src/agents/base.py
class BaseAgent:
    """Base agent with dependency injection and structured logging."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._last_confidence: float = 0.0

    async def process(self, query: str, context: dict) -> AgentResult:
        """Process with logging and error handling."""
        correlation_id = get_correlation_id()
        start_time = time.perf_counter()

        self._logger.info(
            "Processing query",
            extra={
                "correlation_id": correlation_id,
                "query_length": len(query),
                "agent": self.__class__.__name__,
            }
        )

        try:
            result = await self._do_process(query, context)
            duration_ms = (time.perf_counter() - start_time) * 1000

            self._logger.info(
                "Processing complete",
                extra={
                    "correlation_id": correlation_id,
                    "duration_ms": duration_ms,
                    "confidence": result.confidence,
                }
            )

            self._last_confidence = result.confidence
            return result

        except Exception as e:
            self._logger.error(
                "Processing failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
```

---

## 8. Sub-Agent Orchestration Strategy

### 8.1 Claude Code Sub-Agent Types

| Agent Type | Purpose | Tools | When to Use |
|------------|---------|-------|-------------|
| **Explore** | Codebase exploration | Glob, Grep, Read, WebFetch | Finding patterns, understanding structure |
| **Plan** | Architecture design | All tools | Complex implementation planning |
| **Bash** | Command execution | Bash | Git, npm, testing, linting |
| **general-purpose** | Multi-step tasks | All tools | Complex debugging, refactoring |

### 8.2 Sub-Agent Usage Patterns

```markdown
## Explore Agent - Use for:
- "Where is X implemented?"
- "How does Y work?"
- "What patterns are used for Z?"
- "Find all files matching pattern"

## Plan Agent - Use for:
- "Design implementation for feature X"
- "Create architecture for module Y"
- "Identify files needing modification"
- "Plan migration strategy"

## Bash Agent - Use for:
- "Run test suite"
- "Execute git operations"
- "Install dependencies"
- "Run formatters/linters"

## General-Purpose Agent - Use for:
- Complex multi-file refactoring
- Debugging across multiple components
- Research requiring web search
- Tasks needing iteration
```

### 8.3 Parallel Agent Execution

```python
# When tasks are independent, run in parallel
async def parallel_agent_execution(
    tasks: list[AgentTask],
) -> list[AgentResult]:
    """Execute independent agent tasks in parallel."""
    async def run_task(task: AgentTask) -> AgentResult:
        agent = get_agent(task.agent_type)
        return await agent.process(task.query, task.context)

    results = await asyncio.gather(
        *[run_task(task) for task in tasks],
        return_exceptions=True,
    )

    return results
```

---

## 9. Testing Strategy

### 9.1 Test Categories

| Category | Location | Purpose | Target Coverage |
|----------|----------|---------|-----------------|
| Unit | `tests/unit/` | Individual functions | 90% |
| Integration | `tests/integration/` | Component interactions | 70% |
| E2E | `tests/e2e/` | Complete scenarios | 50% |
| Property | `tests/property/` | Invariant testing | Core algorithms |
| Benchmark | `tests/benchmark/` | Performance | Key metrics |

### 9.2 Test Fixtures Pattern

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def test_settings():
    """Test settings with isolation."""
    reset_settings()
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-key-for-testing",
        "MCTS_ENABLED": "true",
        "MCTS_ITERATIONS": "10",  # Low for fast tests
        "SEED": "42",
    }):
        yield get_settings()
    reset_settings()

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock(spec=LLMClient)
    client.generate.return_value = LLMResponse(
        content="Test response",
        model="gpt-4-test",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )
    return client

@pytest.fixture
def mcts_engine(test_settings):
    """Configured MCTS engine for testing."""
    return MCTSEngine(
        seed=test_settings.SEED,
        exploration_weight=test_settings.MCTS_C,
    )
```

### 9.3 Test Examples

```python
# tests/unit/test_mcts_core.py
@pytest.mark.unit
def test_mcts_engine_deterministic_with_seed(mcts_engine):
    """Test MCTS produces deterministic results with same seed."""
    state = MCTSState("test", {"query": "test"})

    result1 = mcts_engine.select(state)
    mcts_engine.reset()
    result2 = mcts_engine.select(state)

    assert result1 == result2

@pytest.mark.unit
@pytest.mark.asyncio
async def test_rollout_policy_interface(mock_rollout_policy):
    """Test rollout policy implements required interface."""
    state = MCTSState("test", {})
    rng = np.random.default_rng(42)

    # Should work with default max_depth
    value = await mock_rollout_policy.evaluate(state, rng)
    assert 0.0 <= value <= 1.0

    # Should work with explicit max_depth
    value = await mock_rollout_policy.evaluate(state, rng, max_depth=5)
    assert 0.0 <= value <= 1.0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_framework_service_with_mcts(
    mock_llm_client,
    test_settings,
):
    """Test framework service integrates MCTS correctly."""
    service = await FrameworkService.get_instance(
        config=FrameworkConfig.from_settings(test_settings),
    )

    result = await service.process_query(
        query="Test query",
        use_mcts=True,
    )

    assert result.response is not None
    assert result.mcts_stats is not None
```

---

## 10. Observability & Debugging

### 10.1 Structured Logging

```python
# src/observability/logging.py
from contextvars import ContextVar

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

def get_correlation_id() -> str | None:
    return _correlation_id.get()

def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)

class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID and sanitization."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_message(record.getMessage()),
            "correlation_id": get_correlation_id(),
        }

        if hasattr(record, "extra"):
            log_obj.update(sanitize_dict(record.extra))

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)
```

### 10.2 Metrics Collection

```python
# src/observability/metrics.py
@dataclass
class MCTSMetrics:
    """MCTS performance metrics."""
    iterations: int = 0
    total_simulations: int = 0
    tree_depth: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    selection_times_ms: list[float] = field(default_factory=list)
    simulation_times_ms: list[float] = field(default_factory=list)

class MetricsCollector:
    """Central metrics collection with Prometheus export."""

    def record_mcts_iteration(
        self,
        search_id: str,
        phase_times: dict[str, float],
    ) -> None:
        metrics = self._mcts_metrics[search_id]
        metrics.iterations += 1
        # Record phase times...
```

### 10.3 Debugging Checklist

```markdown
When debugging issues:

1. [ ] Check logs with correlation ID: `grep <correlation_id> logs/`
2. [ ] Verify configuration: `settings.safe_dict()`
3. [ ] Check MCTS tree state: `engine.get_tree_stats()`
4. [ ] Verify agent confidence: `agent.get_confidence()`
5. [ ] Check async task status: `asyncio.all_tasks()`
6. [ ] Review metrics: `/metrics` endpoint
7. [ ] Check distributed trace: LangSmith/Jaeger
8. [ ] Verify rate limits: check 429 responses
9. [ ] Check cache stats: `engine.cache_hits, engine.cache_misses`
10. [ ] Memory profiling: `tracemalloc.get_traced_memory()`
```

---

## 11. Backwards Compatibility

### 11.1 Version Compatibility Matrix

| Component | v0.1.x | v0.2.x | Notes |
|-----------|--------|--------|-------|
| Settings API | ✓ | ✓ | Fully compatible |
| MCTS Engine | ✓ | ✓ | New params have defaults |
| Agent Protocol | ✓ | ✓ | Additive changes only |
| REST API | ✓ | ✓ | Versioned endpoints |

### 11.2 Deprecation Pattern

```python
import warnings
from functools import wraps

def deprecated(reason: str, removal_version: str):
    """Mark function as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in "
                f"v{removal_version}. {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@deprecated("Use MCTSConfig.from_preset() instead", "0.3.0")
def create_mcts_config(iterations: int) -> MCTSConfig:
    return MCTSConfig(num_iterations=iterations)
```

### 11.3 API Versioning

```python
# src/api/rest_server.py
from fastapi import APIRouter

# V1 API (current stable)
v1_router = APIRouter(prefix="/v1")

@v1_router.post("/query")
async def query_v1(request: QueryRequest) -> QueryResponse:
    ...

# V2 API (with new features)
v2_router = APIRouter(prefix="/v2")

@v2_router.post("/query")
async def query_v2(request: QueryRequestV2) -> QueryResponseV2:
    ...

# Mount both
app.include_router(v1_router)
app.include_router(v2_router)
```

### 11.4 Configuration Migration

```python
# src/config/migration.py
def migrate_settings(old_settings: dict) -> dict:
    """Migrate old settings format to new format."""
    new_settings = old_settings.copy()

    # Rename deprecated keys
    migrations = {
        "MCTS_NUM_ITERATIONS": "MCTS_ITERATIONS",
        "EXPLORATION_CONSTANT": "MCTS_C",
    }

    for old_key, new_key in migrations.items():
        if old_key in new_settings:
            new_settings[new_key] = new_settings.pop(old_key)
            warnings.warn(
                f"Setting {old_key} is deprecated, use {new_key}",
                DeprecationWarning,
            )

    return new_settings
```

---

## 12. Verification Checklist

### 12.1 Pre-Commit Checks

```bash
# 1. Format code
black src/ tests/ --line-length 120

# 2. Sort imports
isort src/ tests/ --profile black

# 3. Lint
ruff check src/ tests/ --fix
ruff check src/ tests/

# 4. Type check
mypy src/ --strict

# 5. Unit tests
pytest tests/unit -v --tb=short -x

# 6. Integration tests
pytest tests/integration -v --tb=short

# 7. Coverage check
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=50

# 8. Security scan
bandit -r src/ -ll

# 9. Check for hardcoded values
grep -r "api_key.*=.*['\"]sk-" src/ && echo "FAIL" || echo "OK"
```

### 12.2 Deployment Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Type checking passes
- [ ] No hardcoded secrets (verified)
- [ ] Environment variables documented
- [ ] Docker build successful
- [ ] Smoke tests pass in staging
- [ ] Monitoring dashboards configured
- [ ] Alerting rules configured
- [ ] Rollback plan documented
- [ ] Load testing completed
- [ ] Security review completed

### 12.3 Success Criteria Verification

```bash
# Mechanically verify all success criteria:

# 1. Tests pass
pytest tests/ -v --tb=short && echo "PASS: Tests" || echo "FAIL: Tests"

# 2. Type check passes
mypy src/ --strict && echo "PASS: Types" || echo "FAIL: Types"

# 3. Coverage meets threshold
pytest tests/ --cov=src --cov-fail-under=80 && echo "PASS: Coverage" || echo "FAIL: Coverage"

# 4. No hardcoded values
! grep -r "api_key.*=.*['\"]" src/ && echo "PASS: No hardcoded" || echo "FAIL: Hardcoded found"

# 5. Logging with correlation IDs
grep -r "correlation_id" src/observability/ && echo "PASS: Correlation IDs" || echo "FAIL: Missing correlation IDs"

# 6. Configuration via Settings
grep -r "get_settings()" src/ | wc -l | xargs -I {} echo "Settings usage: {} files"
```

---

## Quick Reference

### Commands

```bash
# Setup
pip install -e ".[dev,neural]"
cp .env.example .env

# Development
black src/ tests/                    # Format
ruff check src/ tests/ --fix        # Lint
mypy src/                           # Type check

# Testing
pytest tests/unit -v                 # Unit tests
pytest tests/ -k "mcts"             # MCTS tests
pytest tests/ --cov=src             # With coverage

# Running
python -m src.api.server            # Start API
curl http://localhost:8000/health   # Health check
```

### Key Environment Variables

```bash
# Required
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# MCTS
MCTS_ENABLED=true
MCTS_ITERATIONS=100
MCTS_C=1.414

# Optional
LOG_LEVEL=INFO
PINECONE_API_KEY=...
WANDB_API_KEY=...
```

---

**Document Version:** 3.0
**Last Updated:** 2026-01-27
**Aligned with:** LangGraph Multi-Agent MCTS v0.2.0
