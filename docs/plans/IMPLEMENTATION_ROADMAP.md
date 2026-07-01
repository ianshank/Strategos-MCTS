# Implementation Roadmap - LangGraph Multi-Agent MCTS

> Comprehensive guide for completing production-ready features.
> **Last Updated:** 2026-01-24
> **Current Status:** 90% Feature Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 1: Production Unblock](#phase-1-production-unblock)
3. [Phase 2: Core Training](#phase-2-core-training)
4. [Phase 3: Production Hardening](#phase-3-production-hardening)
5. [Phase 4: Advanced Features](#phase-4-advanced-features)
6. [Implementation Patterns](#implementation-patterns)
7. [Testing Strategy](#testing-strategy)
8. [Verification Checklist](#verification-checklist)

---

## Executive Summary

### Current Project State

| Metric | Value |
|--------|-------|
| Overall Completion | 90% |
| Test Pass Rate | 88.4% (771/872) |
| Core MCTS Coverage | 97.66% |
| Documentation | Excellent (36+ files) |

### Priority Tasks

| Priority | Task | Status | Effort |
|----------|------|--------|--------|
| **CRITICAL** | REST API → Framework Connection | Complete | - |
| **HIGH** | MCTS Policy Interface Fix | In Progress | 0.5 days |
| **HIGH** | RAG Integration in Query Pipeline | In Progress | 1 day |
| **MEDIUM** | Comprehensive Logging | Complete | - |
| **MEDIUM** | Full Test Suite Pass | In Progress | 1 day |

---

## Phase 1: Production Unblock

### 1.1 REST API Framework Connection

**Status:** ✅ COMPLETE

**Location:** `src/api/rest_server.py`, `src/api/framework_service.py`

**Implementation Details:**
- Framework service uses singleton pattern with lazy initialization
- Supports full framework or lightweight fallback mode
- Configuration-driven (no hardcoded values)
- Proper error handling with typed exceptions

**Key Components:**
```python
# Framework initialization in rest_server.py (lifespan)
framework_service = await FrameworkService.get_instance(
    config=FrameworkConfig.from_settings(settings),
    settings=settings,
)

# Query processing
result = await framework_service.process_query(
    query=request.query,
    use_mcts=request.use_mcts,
    use_rag=request.use_rag,
    thread_id=request.thread_id,
)
```

**Verification:**
```bash
# Run API tests
pytest tests/api/test_rest_endpoints.py -v
pytest tests/unit/test_framework_service.py -v
```

---

### 1.2 MCTS Policy Interface

**Status:** 🔄 IN PROGRESS

**Location:** `src/framework/mcts/policies.py`

**Issue:** 24 test failures due to signature mismatch in rollout policy interface.

**Required Changes:**
1. Ensure `RolloutPolicy.evaluate()` signature matches interface
2. Update all policy implementations for consistency
3. Add proper type hints

**Interface Definition:**
```python
class RolloutPolicy(ABC):
    @abstractmethod
    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """Evaluate state and return value in [0, 1]."""
        pass
```

**Verification:**
```bash
pytest tests/unit/test_mcts_core.py -v
pytest tests/framework/mcts/ -v
```

---

### 1.3 Test Suite Completion

**Status:** 🔄 IN PROGRESS

**Current Pass Rate:** 88.4%

**Known Failures:**
| Category | Count | Resolution |
|----------|-------|------------|
| MCTS Policy Interface | 24 | Fix signature mismatch |
| DABStep Dataset | 6 | Add split parameter |
| HRM Agent Export | 14 | Fix module exports |
| Collection Errors | 47 | Install optional deps |

**Fix Commands:**
```bash
# Install all optional dependencies
pip install -e ".[dev,neural,experiment,vectordb]"

# Run tests with verbose output
pytest tests/unit -v --tb=short

# Run specific failing tests
pytest tests/unit/test_mcts_core.py -v -k "policy"
```

---

## Phase 2: Core Training

### 2.1 HRM Agent Training Loop

**Status:** ✅ IMPLEMENTED

**Location:** `src/training/unified_orchestrator.py`, `src/training/agent_trainer.py`

**Implementation:**
```python
# In unified_orchestrator.py
async def _train_hrm_agent(self) -> dict[str, float]:
    from .agent_trainer import HRMTrainer, HRMTrainingConfig

    trainer = HRMTrainer(
        agent=self.hrm_agent,
        optimizer=self.hrm_optimizer,
        loss_fn=self.hrm_loss_fn,
        config=HRMTrainingConfig(...),
        device=self.device,
    )

    return await trainer.train_epoch(data_loader)
```

**Key Features:**
- Adaptive Computation Time (ACT) loss
- Ponder cost regularization
- Convergence consistency loss
- Mixed precision support
- Gradient clipping

---

### 2.2 TRM Agent Training Loop

**Status:** ✅ IMPLEMENTED

**Location:** `src/training/unified_orchestrator.py`, `src/training/agent_trainer.py`

**Implementation:**
```python
# In unified_orchestrator.py
async def _train_trm_agent(self) -> dict[str, float]:
    from .agent_trainer import TRMTrainer, TRMTrainingConfig

    trainer = TRMTrainer(
        agent=self.trm_agent,
        optimizer=self.trm_optimizer,
        loss_fn=self.trm_loss_fn,
        config=TRMTrainingConfig(...),
        device=self.device,
    )

    return await trainer.train_epoch(data_loader)
```

**Key Features:**
- Deep supervision at all recursion levels
- Convergence monitoring
- Residual norm tracking
- Weight decay for intermediate losses

---

### 2.3 RAG Integration

**Status:** 🔄 IN PROGRESS

**Location:** `src/api/framework_service.py`, `src/storage/pinecone_store.py`

**Required Integration:**
```python
# Enhanced framework service with RAG
async def process_query(
    self,
    query: str,
    use_rag: bool = True,
    ...
) -> QueryResult:
    context = ""

    if use_rag and self._rag_retriever:
        retrieved_docs = await self._rag_retriever.search(
            query=query,
            top_k=self._config.top_k_retrieval,
        )
        context = self._format_retrieved_context(retrieved_docs)

    # Process with context
    result = await self._framework.process(
        query=query,
        context=context,
        ...
    )
```

---

## Phase 3: Production Hardening

### 3.1 Performance Optimization

**Targets:**
| Metric | Target | Current |
|--------|--------|---------|
| Query Latency | <500ms | ~100ms |
| MCTS Iterations/sec | >100 | TBD |
| Memory Usage | <2GB | ~500MB |
| Docker Image | <2GB | 1.08GB |

**Optimization Areas:**
1. MCTS simulation caching
2. Batch inference for neural networks
3. Connection pooling for external services
4. Response streaming for long queries

---

### 3.2 Error Recovery

**Implementation:**
```python
# Circuit breaker pattern
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: float,
    ):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "closed"

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise CircuitBreakerOpen()

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

---

### 3.3 Observability

**Implemented:**
- Structured logging with correlation IDs
- Prometheus metrics (request count, latency, errors)
- OpenTelemetry tracing

**Dashboards:**
- Grafana: `monitoring/grafana/dashboards/`
- Prometheus: `monitoring/prometheus/prometheus.yml`

---

## Phase 4: Advanced Features

### 4.1 Continuous Self-Play

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│                Self-Play Loop                        │
├─────────────────────────────────────────────────────┤
│  1. Generate games with current model               │
│  2. Store experiences in replay buffer              │
│  3. Sample batch and train networks                 │
│  4. Evaluate against previous best                  │
│  5. Update best model if improved                   │
│  6. Repeat                                          │
└─────────────────────────────────────────────────────┘
```

### 4.2 Multi-Domain Training

**Planned Support:**
- Code generation (programming)
- Strategic reasoning (games)
- Document analysis (enterprise)
- Scientific research (papers)

---

## Implementation Patterns

### Dynamic Component Factory

```python
from src.config.settings import get_settings

class ComponentFactory:
    """Factory for creating framework components dynamically."""

    @classmethod
    def create_mcts(cls, settings=None) -> MCTSEngine:
        settings = settings or get_settings()

        if settings.MCTS_IMPL == MCTSImplementation.NEURAL:
            return NeuralMCTS(
                num_simulations=settings.MCTS_ITERATIONS,
                c_puct=settings.MCTS_C,
            )
        else:
            return MCTSEngine(
                num_iterations=settings.MCTS_ITERATIONS,
                exploration_weight=settings.MCTS_C,
            )

    @classmethod
    def create_agent(cls, agent_type: str, settings=None):
        settings = settings or get_settings()

        creators = {
            "hrm": cls._create_hrm,
            "trm": cls._create_trm,
            "meta": cls._create_meta_controller,
        }

        return creators[agent_type](settings)
```

### Configuration-Driven Design

```python
# CORRECT - All values from configuration
settings = get_settings()
mcts = MCTSEngine(
    num_iterations=settings.MCTS_ITERATIONS,
    exploration_weight=settings.MCTS_C,
    max_depth=settings.MCTS_MAX_ROLLOUT_DEPTH,
)

# WRONG - Hardcoded values
mcts = MCTSEngine(
    num_iterations=100,  # Don't do this!
    exploration_weight=1.414,
)
```

### Async-First Pattern

```python
# All I/O operations are async
async def process_with_timeout(
    self,
    query: str,
    timeout: float,
) -> Result:
    try:
        result = await asyncio.wait_for(
            self._process(query),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Query timed out after {timeout}s")
        raise TimeoutError(f"Processing exceeded {timeout}s")
```

---

## Testing Strategy

### Test Categories

| Category | Location | Command |
|----------|----------|---------|
| Unit | `tests/unit/` | `pytest tests/unit -v` |
| Integration | `tests/integration/` | `pytest tests/integration -v` |
| E2E | `tests/e2e/` | `pytest tests/e2e -v` |
| Performance | `tests/performance/` | `pytest tests/performance -v` |

### Test Fixtures

```python
# tests/conftest.py
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.MCTS_ENABLED = True
    settings.MCTS_ITERATIONS = 10  # Low for fast tests
    settings.MCTS_C = 1.414
    settings.SEED = 42
    return settings

@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = MagicMock()

    async def mock_generate(*args, **kwargs):
        response = MagicMock()
        response.text = "Mock response"
        return response

    client.generate = mock_generate
    return client
```

### Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| MCTS Core | 95% | 97.66% |
| Neural Networks | 85% | 82-87% |
| API Layer | 85% | 84% |
| Training | 80% | TBD |
| Overall | 85% | 88% |

---

## Verification Checklist

### Pre-Commit Checks

```bash
# 1. Format code
black src/ tests/ --line-length 120

# 2. Sort imports
isort src/ tests/ --profile black

# 3. Lint
ruff check src/ tests/ --fix

# 4. Type check
mypy src/ --strict

# 5. Run tests
pytest tests/unit -v

# 6. Check for hardcoded values
grep -r "api_key.*=.*['\"]sk-" src/ && echo "FAIL" || echo "OK"
```

### Deployment Checklist

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] No hardcoded secrets
- [ ] Environment variables documented
- [ ] Docker build successful
- [ ] Smoke tests pass
- [ ] Monitoring configured
- [ ] Rollback plan documented

---

## Quick Reference

### Key Files

| Purpose | File |
|---------|------|
| Settings | `src/config/settings.py` |
| REST API | `src/api/rest_server.py` |
| Framework Service | `src/api/framework_service.py` |
| MCTS Engine | `src/framework/mcts/core.py` |
| MCTS Policies | `src/framework/mcts/policies.py` |
| HRM Agent | `src/agents/hrm_agent.py` |
| TRM Agent | `src/agents/trm_agent.py` |
| Training | `src/training/unified_orchestrator.py` |
| Agent Trainer | `src/training/agent_trainer.py` |

### Environment Variables

```bash
# Required
LLM_PROVIDER=openai  # or anthropic, lmstudio
OPENAI_API_KEY=sk-...  # or ANTHROPIC_API_KEY

# MCTS Configuration
MCTS_ENABLED=true
MCTS_ITERATIONS=100
MCTS_C=1.414

# Optional
PINECONE_API_KEY=...  # For RAG
WANDB_API_KEY=...     # For experiment tracking
LOG_LEVEL=INFO
```

### Common Commands

```bash
# Development
pip install -e ".[dev,neural]"
pytest tests/unit -v

# Production
docker build -t mcts-api .
docker-compose up -d

# Monitoring
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

---

*Document Version: 1.0*
*Generated: 2026-01-24*
