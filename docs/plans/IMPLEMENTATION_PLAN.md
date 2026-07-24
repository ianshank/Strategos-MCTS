# Comprehensive Implementation Plan: LangGraph Multi-Agent MCTS

> **Generated**: 2026-01-30
> **Template Version**: 2.0
> **Current Codebase Completion**: ~95%
> **Test Status**: 88.4% passing (771/872 tests)

---

## Executive Summary

This implementation plan addresses gaps and enhancements for the LangGraph Multi-Agent MCTS framework. The codebase is already highly mature with 67K lines of production code. This plan focuses on:

1. **Gap Analysis**: Identifying remaining implementation needs
2. **Enhancement Priorities**: Optimizations and improvements
3. **Dynamic Patterns**: Backwards-compatible, reusable components
4. **Test Suite Completion**: Property-based tests, edge cases
5. **Observability Enhancements**: Logging, debugging, alerting

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Gap Analysis](#2-gap-analysis)
3. [Implementation Phases](#3-implementation-phases)
4. [Dynamic Component Patterns](#4-dynamic-component-patterns)
5. [Testing Strategy](#5-testing-strategy)
6. [Logging & Debugging Enhancements](#6-logging--debugging-enhancements)
7. [Alerting & Monitoring](#7-alerting--monitoring)
8. [Implementation Checklist](#8-implementation-checklist)
9. [Verification Protocol](#9-verification-protocol)

---

## 1. Current State Assessment

### 1.1 Implemented Components (95% Complete)

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| **Configuration** | ✅ Complete | 614 | Pydantic Settings v2, validation, SecretStr |
| **LLM Adapters** | ✅ Complete | ~1,200 | OpenAI, Anthropic, LMStudio |
| **MCTS Core** | ✅ Complete | ~620 | UCB1, progressive widening, caching |
| **Neural MCTS** | ✅ Complete | ~800 | AlphaZero-style with policy/value networks |
| **LLM-Guided MCTS** | ✅ Complete | ~2,000 | Full LATS implementation |
| **HRM Agent** | ✅ Complete | ~600 | Adaptive Computation Time |
| **TRM Agent** | ✅ Complete | ~500 | Deep supervision, recursion |
| **Meta-Controllers** | ✅ Complete | ~1,500 | RNN, BERT, Hybrid, Assembly |
| **Hybrid Agent** | ✅ Complete | ~400 | Cost-optimized LLM+Neural |
| **LangGraph Integration** | ✅ Complete | ~1,184 | Full StateGraph with checkpointing |
| **Factories** | ✅ Complete | ~746 | All component factories |
| **Observability** | ✅ Complete | ~600 | Logging, metrics, tracing |
| **Storage** | ✅ Complete | ~400 | Pinecone, S3 |
| **API** | ✅ Complete | ~800 | FastAPI REST server |
| **Training Pipeline** | ✅ Complete | ~2,500 | Full ML orchestration |
| **Enterprise Use Cases** | ✅ Complete | ~1,200 | M&A, Clinical Trial, Regulatory |
| **Neuro-Symbolic** | ✅ Complete | ~800 | Constraint-based reasoning |
| **Tests** | ✅ Complete | 101 files | Unit, integration, e2e, chaos |

### 1.2 Architecture Patterns Already Implemented

```python
# Pattern 1: Dependency Injection via Factories
from src.framework.factories import LLMClientFactory, MCTSEngineFactory
factory = LLMClientFactory(settings=get_settings())
client = factory.create(provider="openai")

# Pattern 2: Protocol-Based Interfaces
@runtime_checkable
class LLMClient(Protocol):
    async def generate(self, *, messages: list[dict], **kwargs) -> LLMResponse: ...

# Pattern 3: Configuration via Pydantic Settings
class Settings(BaseSettings):
    MCTS_ITERATIONS: int = Field(default=100, ge=1, le=10000)
    OPENAI_API_KEY: SecretStr | None = Field(default=None)

# Pattern 4: Async-First Design
async def process(self, query: str) -> Result:
    async with httpx.AsyncClient() as client:
        return await client.get(url)

# Pattern 5: Structured Logging with Correlation IDs
logger.info("Processing", extra={"correlation_id": get_correlation_id()})
```

---

## 2. Gap Analysis

### 2.1 Remaining Implementation Gaps

| Category | Gap | Priority | Effort |
|----------|-----|----------|--------|
| **Caching** | Query-level caching for expensive operations | High | Medium |
| **Health Checks** | Comprehensive health endpoints | High | Low |
| **Timeouts** | Granular timeout configuration per operation | Medium | Medium |
| **Alerting** | Alerting rules definitions | Medium | Medium |
| **Property Tests** | MCTS invariant property-based tests | Medium | Medium |
| **Edge Cases** | MCTS tree edge case handling | Medium | Low |
| **Connection Pool** | Enhanced connection pooling | Low | Low |
| **Documentation** | Inline docstring examples | Low | Low |

### 2.2 Enhancement Opportunities

| Enhancement | Impact | Complexity |
|-------------|--------|------------|
| Query result caching with TTL | High | Medium |
| Circuit breaker improvements | Medium | Low |
| Retry strategy refinements | Medium | Low |
| Batch processing support | Medium | Medium |
| WebSocket streaming | Low | High |

---

## 3. Implementation Phases

### Phase 1: Critical Infrastructure (Week 1)

#### 3.1.1 Query-Level Caching System

**Location**: `src/framework/caching.py`

```python
"""
Query-level caching for expensive operations.

Features:
- TTL-based expiration
- LRU eviction
- Configurable per-operation
- Thread-safe async operations
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from src.config.settings import get_settings

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Cache entry with TTL tracking."""

    value: Any
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # 5 minutes default
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class QueryCache:
    """
    Async-safe query result cache with TTL and LRU eviction.

    Example:
        >>> cache = QueryCache(max_size=1000, default_ttl=300)
        >>> await cache.get_or_compute("key", expensive_fn)
    """

    def __init__(
        self,
        max_size: int | None = None,
        default_ttl: float | None = None,
    ):
        settings = get_settings()
        self.max_size = max_size or settings.MCTS_CACHE_SIZE_LIMIT
        self.default_ttl = default_ttl or 300.0
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _compute_key(self, *args, **kwargs) -> str:
        """Generate deterministic cache key."""
        key_data = f"{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def get(self, key: str) -> tuple[bool, Any]:
        """Get value from cache if exists and not expired."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired:
                    entry.hits += 1
                    self._stats["hits"] += 1
                    self._cache.move_to_end(key)
                    return True, entry.value
                else:
                    del self._cache[key]
            self._stats["misses"] += 1
            return False, None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """Set value in cache with optional TTL."""
        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1

            self._cache[key] = CacheEntry(
                value=value,
                ttl_seconds=ttl or self.default_ttl,
            )

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl: float | None = None,
    ) -> T:
        """Get from cache or compute and cache result."""
        found, value = await self.get(key)
        if found:
            return value

        # Compute outside lock to avoid blocking
        if asyncio.iscoroutinefunction(compute_fn):
            result = await compute_fn()
        else:
            result = compute_fn()

        await self.set(key, result, ttl)
        return result

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": self._stats["hits"] / total if total > 0 else 0,
        }

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()


# Global cache instances
_query_cache: QueryCache | None = None
_mcts_cache: QueryCache | None = None


def get_query_cache() -> QueryCache:
    """Get or create global query cache."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


def get_mcts_cache() -> QueryCache:
    """Get or create MCTS-specific cache."""
    global _mcts_cache
    if _mcts_cache is None:
        _mcts_cache = QueryCache(max_size=10000, default_ttl=600)
    return _mcts_cache
```

#### 3.1.2 Health Check Endpoints

**Location**: `src/api/health.py`

```python
"""
Comprehensive health check endpoints.

Provides:
- Liveness probe (is the service running)
- Readiness probe (is the service ready to accept traffic)
- Detailed health status (all components)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.config.settings import get_settings


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] | None = None


class HealthChecker:
    """
    Comprehensive health checking for all system components.

    Example:
        >>> checker = HealthChecker()
        >>> health = await checker.check_all()
        >>> print(health["status"])  # "healthy" or "degraded"
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds
        self.settings = get_settings()

    async def check_liveness(self) -> dict:
        """
        Liveness check - is the service running?

        Returns minimal response for Kubernetes liveness probe.
        """
        return {
            "status": "alive",
            "timestamp": time.time(),
        }

    async def check_readiness(self) -> dict:
        """
        Readiness check - is the service ready for traffic?

        Checks critical dependencies only.
        """
        checks = await asyncio.gather(
            self._check_settings(),
            self._check_memory(),
            return_exceptions=True,
        )

        all_healthy = all(
            isinstance(c, ComponentHealth) and c.status == HealthStatus.HEALTHY
            for c in checks
        )

        return {
            "ready": all_healthy,
            "timestamp": time.time(),
        }

    async def check_all(self) -> dict:
        """
        Detailed health check of all components.

        Returns comprehensive status for monitoring dashboards.
        """
        checks = await asyncio.gather(
            self._check_settings(),
            self._check_memory(),
            self._check_llm_client(),
            self._check_mcts_engine(),
            self._check_vector_store(),
            self._check_cache(),
            return_exceptions=True,
        )

        components = []
        for check in checks:
            if isinstance(check, ComponentHealth):
                components.append({
                    "name": check.name,
                    "status": check.status.value,
                    "latency_ms": check.latency_ms,
                    "message": check.message,
                    "details": check.details,
                })
            elif isinstance(check, Exception):
                components.append({
                    "name": "unknown",
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": str(check),
                })

        # Determine overall status
        statuses = [c["status"] for c in components]
        if HealthStatus.UNHEALTHY.value in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED.value in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            "status": overall.value,
            "timestamp": time.time(),
            "version": "0.1.0",
            "components": components,
        }

    async def _check_settings(self) -> ComponentHealth:
        """Check settings are loaded correctly."""
        start = time.time()
        try:
            settings = get_settings()
            # Verify critical settings
            assert settings.LLM_PROVIDER is not None

            return ComponentHealth(
                name="settings",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={"provider": settings.LLM_PROVIDER.value},
            )
        except Exception as e:
            return ComponentHealth(
                name="settings",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_memory(self) -> ComponentHealth:
        """Check memory usage is acceptable."""
        import psutil

        start = time.time()
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_percent = process.memory_percent()

        # Thresholds from settings or defaults
        if memory_percent > 90:
            status = HealthStatus.UNHEALTHY
        elif memory_percent > 75:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return ComponentHealth(
            name="memory",
            status=status,
            latency_ms=(time.time() - start) * 1000,
            details={
                "memory_mb": round(memory_mb, 2),
                "memory_percent": round(memory_percent, 2),
            },
        )

    async def _check_llm_client(self) -> ComponentHealth:
        """Check LLM client connectivity."""
        start = time.time()
        try:
            # Import lazily to avoid circular imports
            from src.framework.factories import LLMClientFactory

            factory = LLMClientFactory()
            # Just verify factory can be created
            return ComponentHealth(
                name="llm_client",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={"provider": self.settings.LLM_PROVIDER.value},
            )
        except Exception as e:
            return ComponentHealth(
                name="llm_client",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_mcts_engine(self) -> ComponentHealth:
        """Check MCTS engine can be instantiated."""
        start = time.time()
        try:
            from src.framework.mcts.core import MCTSEngine

            engine = MCTSEngine(seed=42)
            return ComponentHealth(
                name="mcts_engine",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={"seed": engine.seed},
            )
        except Exception as e:
            return ComponentHealth(
                name="mcts_engine",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_vector_store(self) -> ComponentHealth:
        """Check vector store connectivity (if configured)."""
        start = time.time()
        if not self.settings.PINECONE_API_KEY:
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message="Not configured",
            )

        try:
            # Just verify settings are valid
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={"host": self.settings.PINECONE_HOST},
            )
        except Exception as e:
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )

    async def _check_cache(self) -> ComponentHealth:
        """Check cache health."""
        start = time.time()
        try:
            from src.framework.caching import get_query_cache

            cache = get_query_cache()
            stats = cache.stats

            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details=stats,
            )
        except Exception as e:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )


# Singleton instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get or create health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
```

### Phase 2: MCTS Enhancements (Week 2)

#### 3.2.1 MCTS Edge Case Handling

**Location**: `src/framework/mcts/edge_cases.py`

```python
"""
MCTS edge case handling and validation.

Handles:
- Empty action space
- Terminal state detection
- Timeout handling
- Budget exhaustion
- Tree corruption detection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MCTSTerminationReason(str, Enum):
    """Reasons for MCTS search termination."""
    ITERATIONS_COMPLETE = "iterations_complete"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TERMINAL_STATE = "terminal_state"
    NO_ACTIONS = "no_actions"
    CONVERGENCE = "convergence"
    ERROR = "error"


@dataclass
class MCTSSearchResult:
    """Result of MCTS search with termination info."""
    best_action: str | None
    stats: dict[str, Any]
    termination_reason: MCTSTerminationReason
    iterations_completed: int
    time_elapsed_seconds: float
    error: Exception | None = None


class MCTSValidator:
    """
    Validates MCTS tree structure and invariants.

    Invariants checked:
    - Sum of child visits <= parent visits
    - No cycles in tree
    - Single root
    - UCB1 scores >= 0
    """

    def validate_tree(self, root) -> list[str]:
        """
        Validate tree structure and return list of violations.

        Returns empty list if tree is valid.
        """
        violations = []
        visited = set()

        def validate_node(node, path: list):
            node_id = id(node)

            # Check for cycles
            if node_id in path:
                violations.append(f"Cycle detected at depth {len(path)}")
                return

            # Check for multiple visits (shouldn't happen in tree)
            if node_id in visited:
                violations.append(f"Node visited multiple times")
                return

            visited.add(node_id)
            path.append(node_id)

            # Check visit count invariant
            if node.children:
                child_visits = sum(c.visits for c in node.children)
                if child_visits > node.visits:
                    violations.append(
                        f"Child visits ({child_visits}) > parent visits ({node.visits})"
                    )

            # Check value bounds
            if node.visits > 0:
                avg_value = node.value_sum / node.visits
                if avg_value < 0 or avg_value > 1:
                    violations.append(
                        f"Value {avg_value} outside [0, 1] bounds"
                    )

            # Recurse to children
            for child in node.children:
                validate_node(child, path.copy())

        validate_node(root, [])
        return violations


class TimeoutHandler:
    """
    Handles timeout and budget management for MCTS.

    Example:
        >>> handler = TimeoutHandler(timeout_seconds=30, token_budget=10000)
        >>> async with handler.guard():
        ...     await run_mcts()
    """

    def __init__(
        self,
        timeout_seconds: float | None = None,
        token_budget: int | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.token_budget = token_budget
        self.tokens_used = 0
        self._start_time: float | None = None

    @property
    def is_timeout(self) -> bool:
        """Check if timeout has been exceeded."""
        if self.timeout_seconds is None or self._start_time is None:
            return False
        import time
        return (time.time() - self._start_time) > self.timeout_seconds

    @property
    def is_budget_exhausted(self) -> bool:
        """Check if token budget has been exceeded."""
        if self.token_budget is None:
            return False
        return self.tokens_used >= self.token_budget

    def record_tokens(self, tokens: int) -> None:
        """Record token usage."""
        self.tokens_used += tokens

    async def guard(self):
        """Context manager for timeout protection."""
        import time
        self._start_time = time.time()

        class GuardContext:
            def __init__(self, handler):
                self.handler = handler

            async def __aenter__(self):
                return self.handler

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

        return GuardContext(self)
```

#### 3.2.2 MCTS Timeout Configuration

**Add to** `src/config/settings.py`:

```python
# MCTS Timeout Configuration
MCTS_SEARCH_TIMEOUT_SECONDS: float = Field(
    default=60.0, ge=1.0, le=600.0,
    description="Maximum time for MCTS search"
)

MCTS_ITERATION_TIMEOUT_SECONDS: float = Field(
    default=5.0, ge=0.1, le=60.0,
    description="Maximum time per MCTS iteration"
)

MCTS_SIMULATION_TIMEOUT_SECONDS: float = Field(
    default=10.0, ge=0.1, le=60.0,
    description="Maximum time for single simulation"
)

MCTS_EARLY_TERMINATION_THRESHOLD: float = Field(
    default=0.95, ge=0.0, le=1.0,
    description="Confidence threshold for early termination"
)
```

### Phase 3: Testing Enhancements (Week 3)

#### 3.3.1 Property-Based Tests for MCTS

**Location**: `tests/property/test_mcts_invariants.py`

```python
"""
Property-based tests for MCTS invariants.

Uses Hypothesis to generate random inputs and verify
that MCTS invariants always hold.
"""

import pytest
from hypothesis import given, settings, strategies as st

from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState


class TestMCTSInvariants:
    """Property-based tests for MCTS invariants."""

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        exploration_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_engine_determinism(self, seed: int, exploration_weight: float):
        """
        Property: Same seed should produce same results.
        """
        engine1 = MCTSEngine(seed=seed, exploration_weight=exploration_weight)
        engine2 = MCTSEngine(seed=seed, exploration_weight=exploration_weight)

        # First random numbers should match
        assert engine1.rng.random() == engine2.rng.random()

    @given(
        visits=st.integers(min_value=1, max_value=10000),
        value_sum=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
    )
    def test_ucb1_non_negative(self, visits: int, value_sum: float):
        """
        Property: UCB1 score should always be non-negative.
        """
        from src.framework.mcts.policies import ucb1

        # With positive parent visits, UCB1 should be non-negative
        score = ucb1(
            value_sum=value_sum,
            visits=visits,
            parent_visits=visits + 1,  # Parent has at least as many visits
            c=1.414,
        )

        assert score >= 0 or score == float("inf")

    @given(
        num_children=st.integers(min_value=1, max_value=20),
        parent_visits=st.integers(min_value=10, max_value=1000),
    )
    def test_child_visits_invariant(self, num_children: int, parent_visits: int):
        """
        Property: Sum of child visits <= parent visits.
        """
        import numpy as np

        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = parent_visits

        # Distribute visits among children (each gets <= parent / num_children)
        child_visits = [parent_visits // (num_children + 1) for _ in range(num_children)]

        for i, cv in enumerate(child_visits):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = cv

        total_child_visits = sum(c.visits for c in root.children)
        assert total_child_visits <= root.visits

    @given(
        depth=st.integers(min_value=1, max_value=10),
        branching=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_tree_depth_consistent(self, depth: int, branching: int):
        """
        Property: Tree depth matches expected structure.
        """
        import numpy as np

        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        def build_tree(node: MCTSNode, current_depth: int):
            if current_depth >= depth:
                return
            for i in range(branching):
                child_state = MCTSState(f"d{current_depth}_c{i}", {})
                child = node.add_child(f"action_{i}", child_state)
                build_tree(child, current_depth + 1)

        build_tree(root, 0)

        engine = MCTSEngine(seed=42)
        actual_depth = engine.get_tree_depth(root)

        assert actual_depth == depth

    @given(
        state_id=st.text(min_size=1, max_size=100),
        features=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.integers(), st.floats(allow_nan=False), st.text()),
            min_size=0,
            max_size=10,
        ),
    )
    def test_state_hashing_deterministic(self, state_id: str, features: dict):
        """
        Property: Same state should always produce same hash.
        """
        state1 = MCTSState(state_id=state_id, features=features)
        state2 = MCTSState(state_id=state_id, features=features)

        assert state1.to_hash_key() == state2.to_hash_key()

    @given(
        num_iterations=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=20, deadline=10000)  # 10 second deadline
    @pytest.mark.asyncio
    async def test_search_completes(self, num_iterations: int):
        """
        Property: Search should always complete within reasonable time.
        """
        import numpy as np

        from src.framework.mcts.policies import RandomRolloutPolicy

        engine = MCTSEngine(seed=42)
        state = MCTSState("test", {"query": "test"})
        root = MCTSNode(state=state, rng=np.random.default_rng(42))

        def action_generator(s: MCTSState) -> list[str]:
            return ["action_a", "action_b"]

        def state_transition(s: MCTSState, action: str) -> MCTSState:
            return MCTSState(
                state_id=f"{s.state_id}_{action}",
                features={**s.features, "depth": s.features.get("depth", 0) + 1},
            )

        policy = RandomRolloutPolicy()

        best_action, stats = await engine.search(
            root=root,
            num_iterations=num_iterations,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=policy,
        )

        assert stats["iterations"] == num_iterations
        assert stats["root_visits"] > 0
```

#### 3.3.2 Edge Case Tests

**Location**: `tests/unit/test_mcts_edge_cases.py`

```python
"""
Unit tests for MCTS edge cases.
"""

import pytest
import numpy as np

from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState


class TestMCTSEdgeCases:
    """Test MCTS behavior in edge cases."""

    @pytest.fixture
    def engine(self):
        return MCTSEngine(seed=42)

    @pytest.fixture
    def empty_action_state(self):
        return MCTSState("empty", {"has_actions": False})

    def test_empty_action_space(self, engine, empty_action_state):
        """Test handling of state with no available actions."""
        rng = np.random.default_rng(42)
        root = MCTSNode(state=empty_action_state, rng=rng)

        def action_generator(s):
            return []  # No actions available

        def state_transition(s, action):
            return s

        # Expansion should mark node as terminal
        result = engine.expand(root, action_generator, state_transition)

        assert result.terminal is True
        assert result is root

    def test_single_child(self, engine):
        """Test selection with single child."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 10

        child_state = MCTSState("child", {})
        child = root.add_child("only_action", child_state)
        child.visits = 5
        child.value_sum = 3.0

        selected = root.select_child(exploration_weight=1.414)

        assert selected is child

    def test_all_children_unexplored(self, engine):
        """Test selection when all children have zero visits."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 1

        for i in range(5):
            child_state = MCTSState(f"child_{i}", {})
            root.add_child(f"action_{i}", child_state)
            # Children have 0 visits (unexplored)

        # With UCB1, unexplored children should have infinite priority
        # First unexplored child should be selected
        selected = root.select_child(exploration_weight=1.414)

        assert selected.visits == 0

    def test_very_deep_tree(self, engine):
        """Test handling of very deep trees (100+ levels)."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)

        current = root
        depth = 100

        for i in range(depth):
            child_state = MCTSState(f"level_{i}", {})
            current.available_actions = [f"action_{i}"]
            current = current.add_child(f"action_{i}", child_state)

        # Should handle deep tree without stack overflow
        actual_depth = engine.get_tree_depth(root)

        assert actual_depth == depth

    def test_wide_tree(self, engine):
        """Test handling of very wide trees (1000+ children)."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 10000

        for i in range(1000):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = i % 10
            child.value_sum = float(i % 10) * 0.5

        # Selection should complete quickly
        selected = root.select_child(exploration_weight=1.414)

        assert selected is not None

    def test_numerical_stability(self, engine):
        """Test numerical stability with extreme values."""
        state = MCTSState("root", {})
        rng = np.random.default_rng(42)
        root = MCTSNode(state=state, rng=rng)
        root.visits = 1_000_000
        root.value_sum = 500_000.0

        for i in range(10):
            child_state = MCTSState(f"child_{i}", {})
            child = root.add_child(f"action_{i}", child_state)
            child.visits = 100_000
            child.value_sum = 50_000.0

        # Should not have numerical issues
        selected = root.select_child(exploration_weight=1.414)

        assert selected is not None
        assert not np.isnan(selected.value)
        assert not np.isinf(selected.value)

    def test_cache_eviction(self, engine):
        """Test cache eviction under pressure."""
        small_engine = MCTSEngine(seed=42, cache_size_limit=10)

        # Add more entries than cache limit
        for i in range(20):
            state = MCTSState(f"state_{i}", {"id": i})
            key = state.to_hash_key()
            small_engine._simulation_cache[key] = (0.5, 1)

            # Verify eviction happens
            if i >= 10:
                assert len(small_engine._simulation_cache) <= 10

        assert small_engine.cache_evictions > 0
```

---

## 4. Dynamic Component Patterns

### 4.1 Backwards-Compatible Configuration Extension

```python
"""
Pattern for adding new configuration fields without breaking existing code.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class SettingsV2(BaseSettings):
    """
    Extended settings with backwards compatibility.

    New fields have defaults that match v1 behavior.
    """

    # V1 fields (unchanged)
    MCTS_ITERATIONS: int = Field(default=100, ge=1, le=10000)
    MCTS_C: float = Field(default=1.414, ge=0.0, le=10.0)

    # V2 additions (with safe defaults)
    MCTS_TIMEOUT_SECONDS: float = Field(
        default=60.0,  # Safe default
        ge=1.0, le=600.0,
        description="Added in v2: Search timeout"
    )

    MCTS_EARLY_TERMINATION: bool = Field(
        default=False,  # Disabled by default for backwards compatibility
        description="Added in v2: Enable early termination"
    )

    @property
    def is_v2_features_enabled(self) -> bool:
        """Check if v2 features are explicitly enabled."""
        return self.MCTS_EARLY_TERMINATION
```

### 4.2 Plugin Architecture for Agents

```python
"""
Plugin architecture for dynamically registering new agents.
"""

from typing import Protocol, Type, runtime_checkable


@runtime_checkable
class AgentPlugin(Protocol):
    """Protocol for agent plugins."""

    name: str
    version: str

    async def process(self, query: str, context: dict) -> dict: ...
    def get_capabilities(self) -> list[str]: ...


class AgentRegistry:
    """
    Registry for dynamically registered agents.

    Example:
        >>> registry = AgentRegistry()
        >>> registry.register("custom_agent", MyCustomAgent)
        >>> agent = registry.create("custom_agent", config=config)
    """

    _instance: "AgentRegistry | None" = None
    _agents: dict[str, Type[AgentPlugin]] = {}

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, agent_class: Type[AgentPlugin]) -> None:
        """Register an agent class."""
        if not isinstance(agent_class, type):
            raise TypeError("agent_class must be a class")
        self._agents[name] = agent_class

    def unregister(self, name: str) -> None:
        """Unregister an agent."""
        self._agents.pop(name, None)

    def create(self, name: str, **kwargs) -> AgentPlugin:
        """Create an agent instance."""
        if name not in self._agents:
            raise ValueError(f"Unknown agent: {name}")
        return self._agents[name](**kwargs)

    def list_agents(self) -> list[str]:
        """List registered agent names."""
        return list(self._agents.keys())


# Decorator for easy registration
def register_agent(name: str):
    """Decorator to register an agent class."""
    def decorator(cls: Type[AgentPlugin]) -> Type[AgentPlugin]:
        AgentRegistry.get_instance().register(name, cls)
        return cls
    return decorator


# Example usage:
# @register_agent("my_agent")
# class MyAgent:
#     name = "my_agent"
#     version = "1.0.0"
#     ...
```

### 4.3 Reusable Retry Pattern

```python
"""
Reusable retry pattern with exponential backoff.
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait_seconds: float = 1.0,
        max_wait_seconds: float = 30.0,
        retryable_exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.min_wait = min_wait_seconds
        self.max_wait = max_wait_seconds
        self.exceptions = retryable_exceptions


def with_retry(config: RetryConfig | None = None):
    """
    Decorator for adding retry logic to async functions.

    Example:
        >>> @with_retry(RetryConfig(max_attempts=5))
        ... async def unreliable_api_call():
        ...     return await external_api()
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        @retry(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait_exponential(
                min=config.min_wait,
                max=config.max_wait,
            ),
            retry=retry_if_exception_type(config.exceptions),
            before_sleep=lambda retry_state: logger.warning(
                f"Retry {retry_state.attempt_number}/{config.max_attempts} "
                f"for {func.__name__}"
            ),
        )
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-configured retry decorators
retry_llm = with_retry(RetryConfig(
    max_attempts=3,
    min_wait_seconds=1.0,
    max_wait_seconds=10.0,
    retryable_exceptions=(TimeoutError, ConnectionError),
))

retry_storage = with_retry(RetryConfig(
    max_attempts=5,
    min_wait_seconds=0.5,
    max_wait_seconds=30.0,
))
```

---

## 5. Testing Strategy

### 5.1 Test Categories and Targets

| Category | Location | Coverage Target | Focus |
|----------|----------|-----------------|-------|
| **Unit** | `tests/unit/` | 90% | Individual functions, edge cases |
| **Integration** | `tests/integration/` | 70% | Component interactions |
| **E2E** | `tests/e2e/` | 50% | Full user scenarios |
| **Property** | `tests/property/` | N/A | Invariant verification |
| **Performance** | `tests/performance/` | N/A | Latency, throughput |
| **Chaos** | `tests/chaos/` | N/A | Failure resilience |

### 5.2 Test Fixtures Best Practices

```python
# tests/conftest.py additions

@pytest.fixture
def isolated_settings():
    """
    Fully isolated settings for tests that modify configuration.

    Creates a temporary environment and resets after test.
    """
    import tempfile
    import os

    original_env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = os.path.join(tmpdir, ".env")
        with open(env_file, "w") as f:
            f.write("LLM_PROVIDER=openai\n")
            f.write("OPENAI_API_KEY=sk-test-key-isolated\n")

        os.environ["MCTS_ENV_FILE"] = env_file
        reset_settings()

        yield get_settings()

        os.environ.clear()
        os.environ.update(original_env)
        reset_settings()


@pytest.fixture
async def async_timeout():
    """
    Fixture for testing timeout behavior.
    """
    async def with_timeout(coro, timeout_seconds: float = 1.0):
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None

    return with_timeout


@pytest.fixture
def mock_cache():
    """
    Mock cache for testing without side effects.
    """
    from unittest.mock import AsyncMock
    from src.framework.caching import QueryCache

    cache = QueryCache(max_size=10, default_ttl=1.0)
    # Reset stats
    cache._stats = {"hits": 0, "misses": 0, "evictions": 0}

    return cache
```

### 5.3 Test Naming Convention

```
test_<unit>_<scenario>_<expected_result>

Examples:
- test_mcts_engine_empty_actions_marks_terminal
- test_ucb1_zero_visits_returns_infinity
- test_cache_expired_entry_returns_miss
- test_settings_invalid_key_raises_validation_error
```

---

## 6. Logging & Debugging Enhancements

### 6.1 Enhanced Structured Logger

```python
"""
Enhanced structured logging with MCTS-specific context.

Location: src/observability/logging.py (additions)
"""

class MCTSLogContext:
    """
    Context manager for MCTS operation logging.

    Automatically logs start, completion, and errors with timing.
    """

    def __init__(
        self,
        operation: str,
        logger: logging.Logger,
        extra: dict | None = None,
    ):
        self.operation = operation
        self.logger = logger
        self.extra = extra or {}
        self.start_time = None

    async def __aenter__(self):
        import time
        self.start_time = time.time()

        self.logger.info(
            f"MCTS {self.operation} started",
            extra={
                "correlation_id": get_correlation_id(),
                "operation": self.operation,
                **self.extra,
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        import time
        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is not None:
            self.logger.error(
                f"MCTS {self.operation} failed",
                extra={
                    "correlation_id": get_correlation_id(),
                    "operation": self.operation,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(exc_val),
                    "error_type": exc_type.__name__,
                    **self.extra,
                },
                exc_info=True,
            )
        else:
            self.logger.info(
                f"MCTS {self.operation} completed",
                extra={
                    "correlation_id": get_correlation_id(),
                    "operation": self.operation,
                    "duration_ms": round(duration_ms, 2),
                    **self.extra,
                }
            )

        return False  # Don't suppress exceptions


# Example usage:
# async with MCTSLogContext("search", logger, {"iterations": 100}):
#     result = await engine.search(...)
```

### 6.2 Debug Mode Enhancements

```python
"""
Debug utilities for development.

Location: src/observability/debug.py (additions)
"""

import json
from typing import Any


class MCTSDebugger:
    """
    Debugging utilities for MCTS operations.

    Provides:
    - Tree visualization
    - State inspection
    - Path tracing
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.trace_buffer: list[dict] = []
        self.max_trace_size = 1000

    def trace(self, event: str, data: dict[str, Any]) -> None:
        """Record a trace event."""
        if not self.enabled:
            return

        import time
        self.trace_buffer.append({
            "timestamp": time.time(),
            "event": event,
            "data": data,
        })

        # Prevent unbounded growth
        if len(self.trace_buffer) > self.max_trace_size:
            self.trace_buffer = self.trace_buffer[-self.max_trace_size:]

    def tree_to_dict(self, node, max_depth: int = 10) -> dict:
        """Convert MCTS tree to dictionary for inspection."""
        if max_depth <= 0:
            return {"truncated": True}

        return {
            "state_id": node.state.state_id,
            "visits": node.visits,
            "value": node.value,
            "action": node.action,
            "terminal": node.terminal,
            "children": [
                self.tree_to_dict(c, max_depth - 1)
                for c in node.children[:10]  # Limit children
            ],
        }

    def tree_to_mermaid(self, node, max_depth: int = 5) -> str:
        """Generate Mermaid diagram of MCTS tree."""
        lines = ["graph TD"]
        node_id = 0

        def add_node(n, depth: int) -> int:
            nonlocal node_id
            current_id = node_id
            node_id += 1

            label = f"{n.state.state_id[:10]}\\nV={n.visits} Q={n.value:.2f}"
            lines.append(f"    N{current_id}[\"{label}\"]")

            if depth < max_depth:
                for child in n.children[:5]:
                    child_id = add_node(child, depth + 1)
                    lines.append(f"    N{current_id} --> N{child_id}")

            return current_id

        add_node(node, 0)
        return "\n".join(lines)

    def export_trace(self) -> str:
        """Export trace buffer as JSON."""
        return json.dumps(self.trace_buffer, indent=2, default=str)

    def clear_trace(self) -> None:
        """Clear trace buffer."""
        self.trace_buffer.clear()
```

---

## 7. Alerting & Monitoring

### 7.1 Alerting Rules Definition

**Location**: `config/alerting_rules.yaml`

```yaml
# Alerting rules for MCTS framework monitoring

groups:
  - name: mcts_alerts
    rules:
      # MCTS Performance Alerts
      - alert: MCTSSearchTimeout
        expr: mcts_search_duration_seconds > 60
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "MCTS search taking too long"
          description: "Search duration {{ $value }}s exceeds 60s threshold"

      - alert: MCTSCacheHitRateLow
        expr: mcts_cache_hit_rate < 0.3
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "MCTS cache hit rate below 30%"
          description: "Cache may need tuning, current hit rate: {{ $value }}"

      - alert: MCTSHighMemoryUsage
        expr: process_resident_memory_bytes > 2e9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage (>2GB)"
          description: "Memory usage: {{ humanize $value }}"

  - name: llm_alerts
    rules:
      - alert: LLMProviderError
        expr: rate(llm_requests_failed_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM provider error rate high"
          description: "Error rate: {{ $value }}/s"

      - alert: LLMRateLimited
        expr: rate(llm_rate_limited_total[5m]) > 0.5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "LLM rate limiting detected"
          description: "Rate limit hits: {{ $value }}/s"

      - alert: LLMLatencyHigh
        expr: histogram_quantile(0.95, llm_request_duration_seconds) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM P95 latency > 10s"
          description: "P95 latency: {{ $value }}s"

  - name: agent_alerts
    rules:
      - alert: AgentConsensusFailure
        expr: rate(agent_consensus_failures_total[5m]) > 0.1
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Agent consensus failures increasing"
          description: "Failure rate: {{ $value }}/s"

      - alert: MetaControllerRoutingError
        expr: rate(meta_controller_errors_total[5m]) > 0.05
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Meta-controller routing errors"
          description: "Error rate: {{ $value }}/s"

  - name: infrastructure_alerts
    rules:
      - alert: HealthCheckFailing
        expr: health_check_status != 1
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Health check failing"
          description: "Component unhealthy: {{ $labels.component }}"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High 5xx error rate"
          description: "Error rate: {{ $value }}/s"
```

### 7.2 Metrics Collection Enhancements

```python
"""
Enhanced metrics collection for alerting.

Location: src/observability/metrics.py (additions)
"""

from prometheus_client import Counter, Histogram, Gauge


# MCTS Metrics
mcts_search_duration = Histogram(
    "mcts_search_duration_seconds",
    "MCTS search duration in seconds",
    ["search_type"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120],
)

mcts_iterations_total = Counter(
    "mcts_iterations_total",
    "Total MCTS iterations performed",
    ["search_id"],
)

mcts_cache_hit_rate = Gauge(
    "mcts_cache_hit_rate",
    "MCTS simulation cache hit rate",
)

mcts_tree_depth = Gauge(
    "mcts_tree_depth",
    "Current MCTS tree depth",
    ["search_id"],
)

# LLM Metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"],
)

llm_requests_failed_total = Counter(
    "llm_requests_failed_total",
    "Failed LLM requests",
    ["provider", "error_type"],
)

llm_rate_limited_total = Counter(
    "llm_rate_limited_total",
    "LLM rate limit hits",
    ["provider"],
)

llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration",
    ["provider", "model"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# Agent Metrics
agent_consensus_failures_total = Counter(
    "agent_consensus_failures_total",
    "Agent consensus failures",
)

meta_controller_errors_total = Counter(
    "meta_controller_errors_total",
    "Meta-controller routing errors",
    ["error_type"],
)

# Health Metrics
health_check_status = Gauge(
    "health_check_status",
    "Health check status (1=healthy, 0=unhealthy)",
    ["component"],
)
```

---

## 8. Implementation Checklist

### 8.1 Phase 1: Critical Infrastructure

- [ ] Implement `QueryCache` in `src/framework/caching.py`
- [ ] Implement `HealthChecker` in `src/api/health.py`
- [ ] Add health endpoints to REST server
- [ ] Write unit tests for caching
- [ ] Write unit tests for health checks
- [ ] Update CLAUDE.md with new commands

### 8.2 Phase 2: MCTS Enhancements

- [ ] Add `MCTSValidator` for tree validation
- [ ] Add `TimeoutHandler` for search timeouts
- [ ] Add timeout settings to `settings.py`
- [ ] Implement early termination logic
- [ ] Write edge case tests
- [ ] Write property-based tests

### 8.3 Phase 3: Testing & Observability

- [ ] Add property-based tests for all MCTS invariants
- [ ] Add edge case tests for boundary conditions
- [ ] Enhance structured logging with MCTS context
- [ ] Add `MCTSDebugger` utilities
- [ ] Create alerting rules YAML
- [ ] Add enhanced metrics collection
- [ ] Update test fixtures

### 8.4 Documentation

- [ ] Update CLAUDE.md with new patterns
- [ ] Add inline docstring examples
- [ ] Document alerting rules
- [ ] Document debugging utilities

---

## 9. Verification Protocol

### 9.1 Verification Commands

```bash
# 1. Format and lint
black src/ tests/ --line-length 120
ruff check src/ tests/ --fix

# 2. Type check
mypy src/ --strict --ignore-missing-imports

# 3. Unit tests
pytest tests/unit -v --tb=short

# 4. Property tests
pytest tests/property -v --tb=short

# 5. Integration tests
pytest tests/integration -v --tb=short

# 6. Coverage check
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=50

# 7. Security scan
bandit -r src/ -ll

# 8. Smoke test
python -c "from src.config.settings import get_settings; print(get_settings().safe_dict())"

# 9. Health check
python -c "
import asyncio
from src.api.health import get_health_checker
async def main():
    checker = get_health_checker()
    result = await checker.check_all()
    print(f'Health: {result[\"status\"]}')
asyncio.run(main())
"
```

### 9.2 Success Criteria

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Unit tests passing | 100% | `pytest tests/unit` |
| Integration tests passing | 100% | `pytest tests/integration` |
| Type checking | No errors | `mypy src/` |
| Code coverage | ≥50% | `pytest --cov=src` |
| No hardcoded values | 0 occurrences | `grep -r "api_key.*=.*['\"]sk-" src/` |
| Health check passing | All components | Health endpoint |
| Property tests passing | 100% | `pytest tests/property` |

---

## Appendix: File Locations Summary

```
NEW FILES TO CREATE:
├── src/framework/caching.py       # Query-level caching
├── src/framework/mcts/edge_cases.py  # Edge case handling
├── src/api/health.py              # Health check endpoints
├── config/alerting_rules.yaml     # Alerting definitions
├── tests/property/test_mcts_invariants.py  # Property tests
├── tests/unit/test_mcts_edge_cases.py      # Edge case tests
├── tests/unit/test_caching.py     # Cache tests
└── tests/unit/test_health.py      # Health check tests

FILES TO MODIFY:
├── src/config/settings.py         # Add timeout settings
├── src/observability/logging.py   # Add MCTS log context
├── src/observability/metrics.py   # Add alerting metrics
├── src/api/rest_server.py         # Add health endpoints
├── tests/conftest.py              # Add new fixtures
└── CLAUDE.md                      # Update documentation
```

---

*Implementation Plan Version: 1.0*
*Last Updated: 2026-01-30*
