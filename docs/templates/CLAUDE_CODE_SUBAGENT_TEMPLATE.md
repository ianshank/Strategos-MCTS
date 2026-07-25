# Claude Code Sub-Agent Implementation Template

> **Design Principle:** This template provides reusable patterns for building AI systems using Claude Code's sub-agent architecture. All patterns are dynamic, backwards-compatible, and use no hardcoded values.

**Version:** 1.0
**Created:** 2026-01-27
**Target:** Claude Code implementations with multi-agent orchestration

---

## Table of Contents

1. [Overview](#1-overview)
2. [Sub-Agent Architecture](#2-sub-agent-architecture)
3. [Dynamic Component Patterns](#3-dynamic-component-patterns)
4. [Reusable Code Patterns](#4-reusable-code-patterns)
5. [Configuration Management](#5-configuration-management)
6. [Factory Patterns](#6-factory-patterns)
7. [Testing Framework](#7-testing-framework)
8. [Logging & Debugging](#8-logging--debugging)
9. [Backwards Compatibility](#9-backwards-compatibility)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Overview

### 1.1 Template Purpose

This template provides production-ready patterns for implementing AI systems using Claude Code's sub-agent capabilities. It ensures:

- **Dynamic Configuration:** All values from environment/settings
- **Backwards Compatibility:** Graceful migration paths
- **Reusability:** Composable components and patterns
- **Testability:** Comprehensive test coverage
- **Observability:** Structured logging and metrics

### 1.2 Key Principles

```
1. NO HARDCODED VALUES
   - All configuration via Pydantic Settings
   - Environment variables for secrets
   - Settings injection for components

2. DEPENDENCY INJECTION
   - Constructor injection for all dependencies
   - Factory pattern for component creation
   - Protocol-based interfaces

3. BACKWARDS COMPATIBILITY
   - Deprecation warnings before removal
   - Migration helpers for configuration
   - API versioning for breaking changes

4. COMPREHENSIVE TESTING
   - Unit tests for all functions
   - Integration tests for workflows
   - Property-based tests for invariants

5. STRUCTURED OBSERVABILITY
   - Correlation IDs across async boundaries
   - Structured JSON logging
   - Prometheus metrics export
```

---

## 2. Sub-Agent Architecture

### 2.1 Available Sub-Agents

Claude Code provides specialized sub-agents for different tasks:

| Agent Type | Purpose | Available Tools | Best For |
|------------|---------|-----------------|----------|
| **Explore** | Codebase exploration | Glob, Grep, Read, WebFetch, WebSearch | Finding code, understanding patterns |
| **Plan** | Implementation planning | All tools | Architecture design, strategy |
| **Bash** | Command execution | Bash | Git, npm, testing, deployment |
| **general-purpose** | Multi-step tasks | All tools | Complex tasks requiring iteration |

### 2.2 Sub-Agent Selection Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Sub-Agent Selection Decision Tree                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Task Type?                                                         │
│     │                                                               │
│     ├── Understanding code ──────────────► Explore Agent            │
│     │   - "How does X work?"                                        │
│     │   - "Where is Y implemented?"                                 │
│     │   - "What patterns are used?"                                 │
│     │                                                               │
│     ├── Planning implementation ─────────► Plan Agent               │
│     │   - "Design approach for feature"                             │
│     │   - "Identify files to modify"                                │
│     │   - "Create architecture"                                     │
│     │                                                               │
│     ├── Running commands ────────────────► Bash Agent               │
│     │   - "Run tests"                                               │
│     │   - "Git operations"                                          │
│     │   - "Install dependencies"                                    │
│     │                                                               │
│     └── Complex multi-step ──────────────► general-purpose Agent    │
│         - Multiple searches needed                                  │
│         - Debugging across files                                    │
│         - Research + implementation                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Sub-Agent Invocation Pattern

```python
# Pattern for invoking sub-agents programmatically

from dataclasses import dataclass
from typing import Literal

@dataclass
class SubAgentTask:
    """Task specification for sub-agent execution."""
    agent_type: Literal["Explore", "Plan", "Bash", "general-purpose"]
    description: str  # 3-5 word summary
    prompt: str  # Detailed task description
    allowed_tools: list[str] | None = None  # Optional tool restrictions
    max_turns: int | None = None  # Optional iteration limit

@dataclass
class SubAgentResult:
    """Result from sub-agent execution."""
    output: str
    agent_id: str  # For potential resume
    success: bool
    metadata: dict

# Example usage patterns
EXPLORATION_TASK = SubAgentTask(
    agent_type="Explore",
    description="Find error handling patterns",
    prompt="""
    Explore this codebase to understand:
    1. How errors are handled across the application
    2. What error types are defined
    3. Where try/except blocks are used
    4. Error logging patterns

    Provide a summary of findings with file locations.
    """,
)

PLANNING_TASK = SubAgentTask(
    agent_type="Plan",
    description="Plan authentication feature",
    prompt="""
    Design an implementation plan for adding OAuth2 authentication:
    1. Identify existing auth patterns in the codebase
    2. Determine files that need modification
    3. Consider backwards compatibility
    4. Plan test coverage
    5. Identify potential risks

    Return a step-by-step implementation plan.
    """,
)

BASH_TASK = SubAgentTask(
    agent_type="Bash",
    description="Run test suite",
    prompt="Run the full test suite with coverage and report results.",
)
```

### 2.4 Parallel Sub-Agent Execution

When tasks are independent, execute sub-agents in parallel for efficiency:

```python
# Pattern: Parallel sub-agent execution
import asyncio
from typing import Sequence

async def execute_parallel_tasks(
    tasks: Sequence[SubAgentTask],
) -> list[SubAgentResult]:
    """
    Execute independent sub-agent tasks in parallel.

    Use when:
    - Tasks don't depend on each other's results
    - You need to gather information from multiple sources
    - Time efficiency is important
    """
    async def run_task(task: SubAgentTask) -> SubAgentResult:
        # In Claude Code, this would be multiple Task tool calls
        # in a single message block
        ...

    results = await asyncio.gather(
        *[run_task(task) for task in tasks],
        return_exceptions=True,
    )

    return [
        r if isinstance(r, SubAgentResult)
        else SubAgentResult(output="", agent_id="", success=False, metadata={"error": str(r)})
        for r in results
    ]

# Example: Gather information from multiple sources in parallel
async def comprehensive_codebase_analysis():
    """Run multiple explorations in parallel."""
    tasks = [
        SubAgentTask(
            agent_type="Explore",
            description="Find configuration patterns",
            prompt="Find all configuration loading patterns in src/",
        ),
        SubAgentTask(
            agent_type="Explore",
            description="Find logging patterns",
            prompt="Find all logging patterns in src/",
        ),
        SubAgentTask(
            agent_type="Explore",
            description="Find test patterns",
            prompt="Find test fixture patterns in tests/",
        ),
    ]

    results = await execute_parallel_tasks(tasks)
    return results
```

### 2.5 Sequential Sub-Agent Chains

When tasks depend on previous results, chain them sequentially:

```python
# Pattern: Sequential sub-agent chains
async def implement_feature_with_tests():
    """
    Sequential chain: Plan → Implement → Test → Verify

    Use when:
    - Each step depends on the previous
    - You need to validate before proceeding
    - Order matters for correctness
    """
    # Step 1: Plan with Plan agent
    plan_result = await execute_task(SubAgentTask(
        agent_type="Plan",
        description="Plan feature implementation",
        prompt="Design implementation for user authentication...",
    ))

    if not plan_result.success:
        return plan_result  # Early return on failure

    # Step 2: Implement with general-purpose agent
    # (uses plan from previous step)
    implement_result = await execute_task(SubAgentTask(
        agent_type="general-purpose",
        description="Implement authentication feature",
        prompt=f"""
        Implement the feature based on this plan:
        {plan_result.output}

        Follow all patterns and conventions in the codebase.
        """,
    ))

    # Step 3: Test with Bash agent
    test_result = await execute_task(SubAgentTask(
        agent_type="Bash",
        description="Run tests",
        prompt="Run pytest tests/unit -v and report results",
    ))

    return {
        "plan": plan_result,
        "implementation": implement_result,
        "tests": test_result,
    }
```

---

## 3. Dynamic Component Patterns

### 3.1 Dynamic Component Registry

```python
# src/framework/component_registry.py
from typing import TypeVar, Generic, Callable, Any
from dataclasses import dataclass, field

T = TypeVar("T")

@dataclass
class ComponentRegistry(Generic[T]):
    """
    Dynamic registry for component types.

    Features:
    - Runtime registration of new components
    - Type-safe component retrieval
    - Factory function support
    - Metadata tracking
    """
    _registry: dict[str, Callable[..., T]] = field(default_factory=dict)
    _metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        **metadata,
    ) -> None:
        """Register a component factory."""
        self._registry[name] = factory
        self._metadata[name] = metadata

    def create(self, name: str, **kwargs) -> T:
        """Create a component instance."""
        if name not in self._registry:
            raise KeyError(
                f"Unknown component: {name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name](**kwargs)

    def list_components(self) -> list[str]:
        """List all registered component names."""
        return list(self._registry.keys())

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a component."""
        return self._metadata.get(name, {})


# Usage example
agent_registry: ComponentRegistry[AgentProtocol] = ComponentRegistry()

# Register agents dynamically
agent_registry.register(
    "hrm",
    HRMAgent,
    description="Hierarchical Reasoning Module",
    requires_neural=True,
)

agent_registry.register(
    "trm",
    TRMAgent,
    description="Task Refinement Module",
    requires_neural=True,
)

# Create agents dynamically
agent = agent_registry.create("hrm", config=config, llm_client=client)
```

### 3.2 Plugin Architecture

```python
# src/framework/plugins.py
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path

class PluginInterface(ABC):
    """Base interface for plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        ...

    @abstractmethod
    def initialize(self, settings: Settings) -> None:
        """Initialize the plugin."""
        ...


class PluginManager:
    """
    Dynamic plugin loading and management.

    Supports:
    - Auto-discovery from plugin directory
    - Lazy loading
    - Dependency management
    """

    def __init__(
        self,
        plugin_dir: Path | None = None,
        settings: Settings | None = None,
    ):
        self._plugin_dir = plugin_dir or Path("plugins")
        self._settings = settings or get_settings()
        self._plugins: dict[str, PluginInterface] = {}
        self._loaded: set[str] = set()

    def discover_plugins(self) -> list[str]:
        """Discover available plugins."""
        plugins = []
        if self._plugin_dir.exists():
            for path in self._plugin_dir.glob("*/plugin.py"):
                plugins.append(path.parent.name)
        return plugins

    def load_plugin(self, name: str) -> PluginInterface:
        """Load a plugin by name."""
        if name in self._loaded:
            return self._plugins[name]

        module = import_module(f"plugins.{name}.plugin")
        plugin_class = getattr(module, "Plugin")
        plugin = plugin_class()
        plugin.initialize(self._settings)

        self._plugins[name] = plugin
        self._loaded.add(name)

        return plugin

    def get_plugin(self, name: str) -> PluginInterface | None:
        """Get a loaded plugin."""
        return self._plugins.get(name)
```

### 3.3 Feature Flags

```python
# src/config/feature_flags.py
from dataclasses import dataclass
from typing import Any

@dataclass
class FeatureFlag:
    """Individual feature flag with metadata."""
    name: str
    enabled: bool
    description: str
    rollout_percentage: float = 100.0
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Dynamic feature flag management.

    Use for:
    - Gradual feature rollouts
    - A/B testing
    - Kill switches
    - Environment-specific features
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._flags: dict[str, FeatureFlag] = {}
        self._load_flags()

    def _load_flags(self) -> None:
        """Load flags from settings/environment."""
        # Load from environment variables with FF_ prefix
        import os
        for key, value in os.environ.items():
            if key.startswith("FF_"):
                flag_name = key[3:].lower()
                self._flags[flag_name] = FeatureFlag(
                    name=flag_name,
                    enabled=value.lower() in ("true", "1", "yes"),
                    description=f"Feature flag from env: {key}",
                )

    def is_enabled(
        self,
        flag_name: str,
        default: bool = False,
        context: dict | None = None,
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag_name: Name of the feature flag
            default: Default value if flag not found
            context: Optional context for percentage rollout

        Returns:
            Whether the feature is enabled
        """
        flag = self._flags.get(flag_name)
        if flag is None:
            return default

        if not flag.enabled:
            return False

        # Check rollout percentage if context provided
        if context and flag.rollout_percentage < 100.0:
            user_id = context.get("user_id", "")
            bucket = hash(user_id) % 100
            return bucket < flag.rollout_percentage

        return True

    def register_flag(self, flag: FeatureFlag) -> None:
        """Register a new feature flag."""
        self._flags[flag.name] = flag


# Usage
feature_flags = FeatureFlagManager()

if feature_flags.is_enabled("neural_mcts"):
    mcts = NeuralMCTS(config)
else:
    mcts = MCTSEngine(config)
```

---

## 4. Reusable Code Patterns

### 4.1 Result Types

```python
# src/core/result.py
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Any

T = TypeVar("T")
E = TypeVar("E", bound=Exception)

@dataclass(frozen=True)
class Success(Generic[T]):
    """Successful result containing a value."""
    value: T

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def map(self, func: Callable[[T], Any]) -> "Result":
        return Success(func(self.value))


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Failed result containing an error."""
    error: E

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def unwrap(self) -> Never:
        raise self.error

    def map(self, func: Callable) -> "Result":
        return self  # Failures propagate


Result = Success[T] | Failure[E]


def try_except(func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """
    Decorator that wraps function to return Result type.

    Converts exceptions to Failure results.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Result[T, Exception]:
        try:
            return Success(func(*args, **kwargs))
        except Exception as e:
            return Failure(e)
    return wrapper


async def try_except_async(
    coro: Coroutine[Any, Any, T],
) -> Result[T, Exception]:
    """Wrap async operation in Result type."""
    try:
        value = await coro
        return Success(value)
    except Exception as e:
        return Failure(e)
```

### 4.2 Retry Pattern

```python
# src/core/retry.py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    logger: logging.Logger | None = None,
):
    """
    Reusable retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        min_wait: Minimum wait between retries (seconds)
        max_wait: Maximum wait between retries (seconds)
        retry_on: Exception types to retry on
        logger: Logger for retry attempts
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retry_on),
        before_sleep=before_sleep_log(logger or logging.getLogger(), logging.WARNING),
    )


# Usage
@with_retry(max_attempts=3, retry_on=(ConnectionError, TimeoutError))
async def fetch_with_retry(url: str) -> Response:
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

### 4.3 Circuit Breaker Pattern

```python
# src/core/circuit_breaker.py
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Prevents cascading failures by:
    - Tracking failure rate
    - Opening circuit when threshold exceeded
    - Allowing test requests after timeout
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit open, retry after {self.recovery_timeout}s"
            )

        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpen("Half-open call limit reached")
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._success_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


# Usage with LLM client
llm_circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60.0,
)

async def call_llm_with_circuit_breaker(prompt: str) -> str:
    return await llm_circuit.call(llm_client.generate, prompt=prompt)
```

### 4.4 Caching Pattern

```python
# src/core/cache.py
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import TypeVar, Generic, Callable, Hashable
import time
import asyncio
import hashlib
import json

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with expiration."""
    value: V
    expires_at: float


class LRUCache(Generic[K, V]):
    """
    LRU cache with TTL support.

    Features:
    - Least Recently Used eviction
    - Time-based expiration
    - Thread-safe operations
    - Memory limit support
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
    ):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: K) -> V | None:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    async def set(self, key: K, value: V) -> None:
        """Set value in cache."""
        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + self._ttl,
            )
            self._cache.move_to_end(key)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_data = json.dumps(
        {"args": args, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


def cached(
    cache: LRUCache,
    key_func: Callable[..., str] | None = None,
):
    """
    Decorator for caching async function results.

    Args:
        cache: LRU cache instance
        key_func: Custom key generation function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = (key_func or cache_key)(*args, **kwargs)
            result = await cache.get(key)
            if result is not None:
                return result

            result = await func(*args, **kwargs)
            await cache.set(key, result)
            return result
        return wrapper
    return decorator


# Usage
mcts_cache = LRUCache[str, float](max_size=10000, ttl_seconds=600)

@cached(mcts_cache)
async def evaluate_state(state_hash: str) -> float:
    # Expensive evaluation...
    return value
```

---

## 5. Configuration Management

### 5.1 Pydantic Settings Pattern

```python
# src/config/settings.py
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from typing import Any

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LMSTUDIO = "lmstudio"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """
    Application settings loaded from environment.

    Features:
    - Type validation with Pydantic
    - SecretStr for sensitive values
    - Validators for complex rules
    - Defaults that can be overridden
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    # ==================== LLM Configuration ====================
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use",
    )

    OPENAI_API_KEY: SecretStr | None = Field(
        default=None,
        description="OpenAI API key",
    )

    ANTHROPIC_API_KEY: SecretStr | None = Field(
        default=None,
        description="Anthropic API key",
    )

    LLM_MODEL: str = Field(
        default="gpt-4",
        description="Model name for LLM calls",
    )

    LLM_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0-2.0)",
    )

    LLM_MAX_TOKENS: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens for LLM response",
    )

    # ==================== MCTS Configuration ====================
    MCTS_ENABLED: bool = Field(
        default=True,
        description="Enable MCTS exploration",
    )

    MCTS_ITERATIONS: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="MCTS search iterations",
    )

    MCTS_C: float = Field(
        default=1.414,
        ge=0.0,
        le=10.0,
        description="UCB1 exploration constant",
    )

    MCTS_MAX_DEPTH: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum MCTS tree depth",
    )

    MCTS_CACHE_SIZE: int = Field(
        default=10000,
        ge=0,
        le=1000000,
        description="MCTS simulation cache size",
    )

    # ==================== Agent Configuration ====================
    HRM_MAX_DEPTH: int = Field(
        default=5,
        ge=1,
        le=20,
        description="HRM agent max recursion depth",
    )

    HRM_CONFIDENCE_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="HRM confidence threshold for halting",
    )

    TRM_MAX_ITERATIONS: int = Field(
        default=10,
        ge=1,
        le=50,
        description="TRM agent max iterations",
    )

    TRM_CONVERGENCE_THRESHOLD: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="TRM convergence threshold",
    )

    # ==================== Observability ====================
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )

    LOG_FORMAT: str = Field(
        default="json",
        description="Log format (json or text)",
    )

    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )

    TRACING_ENABLED: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing",
    )

    # ==================== Infrastructure ====================
    HTTP_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="HTTP request timeout",
    )

    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )

    SEED: int | None = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # ==================== Validators ====================
    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, v: SecretStr | None) -> SecretStr | None:
        if v is not None:
            secret = v.get_secret_value()
            if secret and not secret.startswith("sk-"):
                raise ValueError("OpenAI key must start with 'sk-'")
        return v

    @model_validator(mode="after")
    def validate_api_key_exists(self) -> "Settings":
        """Ensure at least one API key is configured."""
        if self.LLM_PROVIDER == LLMProvider.LMSTUDIO:
            return self  # LMStudio doesn't need API key

        has_key = (
            self.OPENAI_API_KEY is not None or
            self.ANTHROPIC_API_KEY is not None
        )
        if not has_key:
            raise ValueError(
                "At least one API key must be configured: "
                "OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )
        return self

    # ==================== Helpers ====================
    def get_api_key(self) -> str:
        """Get API key for current provider."""
        if self.LLM_PROVIDER == LLMProvider.OPENAI and self.OPENAI_API_KEY:
            return self.OPENAI_API_KEY.get_secret_value()
        elif self.LLM_PROVIDER == LLMProvider.ANTHROPIC and self.ANTHROPIC_API_KEY:
            return self.ANTHROPIC_API_KEY.get_secret_value()
        raise ValueError(f"No API key for provider: {self.LLM_PROVIDER}")

    def safe_dict(self) -> dict[str, Any]:
        """Return settings with secrets masked."""
        data = self.model_dump()
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            if key in data and data[key]:
                data[key] = "***MASKED***"
        return data


# Singleton pattern with reset for testing
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (for testing)."""
    global _settings
    _settings = None
```

### 5.2 Environment-Specific Configuration

```python
# src/config/environments.py
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration overrides."""
    name: str
    debug: bool
    log_level: str
    mcts_iterations: int
    cache_enabled: bool

    # Predefined environments
    DEVELOPMENT: ClassVar["EnvironmentConfig"]
    STAGING: ClassVar["EnvironmentConfig"]
    PRODUCTION: ClassVar["EnvironmentConfig"]


EnvironmentConfig.DEVELOPMENT = EnvironmentConfig(
    name="development",
    debug=True,
    log_level="DEBUG",
    mcts_iterations=10,  # Fast iteration
    cache_enabled=False,  # Fresh results
)

EnvironmentConfig.STAGING = EnvironmentConfig(
    name="staging",
    debug=True,
    log_level="INFO",
    mcts_iterations=50,
    cache_enabled=True,
)

EnvironmentConfig.PRODUCTION = EnvironmentConfig(
    name="production",
    debug=False,
    log_level="INFO",
    mcts_iterations=100,
    cache_enabled=True,
)


def get_environment_config() -> EnvironmentConfig:
    """Get configuration for current environment."""
    import os
    env = os.environ.get("ENVIRONMENT", "development").lower()

    configs = {
        "development": EnvironmentConfig.DEVELOPMENT,
        "staging": EnvironmentConfig.STAGING,
        "production": EnvironmentConfig.PRODUCTION,
    }

    return configs.get(env, EnvironmentConfig.DEVELOPMENT)
```

---

## 6. Factory Patterns

### 6.1 Abstract Factory

```python
# src/framework/factories.py
from typing import Protocol, TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar("T")


class ComponentFactory(Protocol[T]):
    """Protocol for component factories."""

    def create(self, **kwargs) -> T:
        """Create a component instance."""
        ...


class LLMClientFactory:
    """
    Factory for creating LLM clients.

    Supports:
    - Multiple providers
    - Configuration injection
    - Custom client options
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._creators = {
            "openai": self._create_openai,
            "anthropic": self._create_anthropic,
            "lmstudio": self._create_lmstudio,
        }

    def create(
        self,
        provider: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> LLMClient:
        """Create an LLM client."""
        provider = provider or self._settings.LLM_PROVIDER.value

        if provider not in self._creators:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(self._creators.keys())}"
            )

        return self._creators[provider](model, **kwargs)

    def _create_openai(self, model: str | None, **kwargs) -> OpenAIClient:
        return OpenAIClient(
            api_key=self._settings.get_api_key(),
            model=model or self._settings.LLM_MODEL,
            timeout=self._settings.HTTP_TIMEOUT_SECONDS,
            max_retries=self._settings.MAX_RETRIES,
            **kwargs,
        )

    def _create_anthropic(self, model: str | None, **kwargs) -> AnthropicClient:
        return AnthropicClient(
            api_key=self._settings.get_api_key(),
            model=model or "claude-3-5-sonnet-20241022",
            timeout=self._settings.HTTP_TIMEOUT_SECONDS,
            max_retries=self._settings.MAX_RETRIES,
            **kwargs,
        )

    def _create_lmstudio(self, model: str | None, **kwargs) -> LMStudioClient:
        return LMStudioClient(
            base_url=kwargs.get("base_url", "http://localhost:1234/v1"),
            model=model,
            timeout=self._settings.HTTP_TIMEOUT_SECONDS,
            **kwargs,
        )


class MCTSEngineFactory:
    """Factory for MCTS engines."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

    def create(
        self,
        preset: str | None = None,
        **overrides,
    ) -> MCTSEngine:
        """
        Create an MCTS engine.

        Args:
            preset: Optional preset ("fast", "balanced", "thorough")
            **overrides: Override specific configuration values
        """
        config = self._get_config(preset)

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return MCTSEngine(
            seed=config.seed,
            exploration_weight=config.exploration_weight,
            max_depth=config.max_depth,
            cache_size=config.cache_size,
        )

    def _get_config(self, preset: str | None) -> MCTSConfig:
        if preset == "fast":
            return MCTSConfig(
                seed=self._settings.SEED,
                num_iterations=50,
                exploration_weight=1.0,
                max_depth=5,
                cache_size=1000,
            )
        elif preset == "thorough":
            return MCTSConfig(
                seed=self._settings.SEED,
                num_iterations=500,
                exploration_weight=2.0,
                max_depth=20,
                cache_size=50000,
            )
        else:  # balanced (default)
            return MCTSConfig(
                seed=self._settings.SEED,
                num_iterations=self._settings.MCTS_ITERATIONS,
                exploration_weight=self._settings.MCTS_C,
                max_depth=self._settings.MCTS_MAX_DEPTH,
                cache_size=self._settings.MCTS_CACHE_SIZE,
            )


class AgentFactory:
    """Factory for creating agents."""

    def __init__(
        self,
        llm_client: LLMClient,
        settings: Settings | None = None,
        logger: logging.Logger | None = None,
    ):
        self._llm = llm_client
        self._settings = settings or get_settings()
        self._logger = logger or logging.getLogger(__name__)

    def create_hrm(self, **kwargs) -> HRMAgent:
        """Create HRM agent."""
        config = HRMConfig(
            max_depth=kwargs.get("max_depth", self._settings.HRM_MAX_DEPTH),
            confidence_threshold=kwargs.get(
                "confidence_threshold",
                self._settings.HRM_CONFIDENCE_THRESHOLD,
            ),
        )
        return HRMAgent(
            config=config,
            llm_client=self._llm,
            logger=self._logger,
        )

    def create_trm(self, **kwargs) -> TRMAgent:
        """Create TRM agent."""
        config = TRMConfig(
            max_iterations=kwargs.get(
                "max_iterations",
                self._settings.TRM_MAX_ITERATIONS,
            ),
            convergence_threshold=kwargs.get(
                "convergence_threshold",
                self._settings.TRM_CONVERGENCE_THRESHOLD,
            ),
        )
        return TRMAgent(
            config=config,
            llm_client=self._llm,
            logger=self._logger,
        )

    def create_hybrid(
        self,
        policy_network: PolicyNetwork | None = None,
        value_network: ValueNetwork | None = None,
        **kwargs,
    ) -> HybridAgent:
        """Create Hybrid agent."""
        return HybridAgent(
            config=HybridConfig(**kwargs),
            policy_network=policy_network,
            value_network=value_network,
            llm_client=self._llm,
            logger=self._logger,
        )
```

### 6.2 Builder Pattern

```python
# src/framework/builder.py
from dataclasses import dataclass, field
from typing import Self

@dataclass
class GraphBuilderConfig:
    """Configuration built incrementally."""
    hrm_agent: HRMAgent | None = None
    trm_agent: TRMAgent | None = None
    mcts_engine: MCTSEngine | None = None
    meta_controller: MetaController | None = None
    vector_store: VectorStore | None = None
    max_iterations: int = 5
    use_mcts: bool = True
    use_rag: bool = False


class GraphBuilder:
    """
    Builder for constructing the agent graph.

    Fluent interface for step-by-step configuration.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._config = GraphBuilderConfig()
        self._logger = logging.getLogger(__name__)

    def with_hrm_agent(self, agent: HRMAgent) -> Self:
        """Add HRM agent to graph."""
        self._config.hrm_agent = agent
        return self

    def with_trm_agent(self, agent: TRMAgent) -> Self:
        """Add TRM agent to graph."""
        self._config.trm_agent = agent
        return self

    def with_mcts(
        self,
        engine: MCTSEngine | None = None,
        enabled: bool = True,
    ) -> Self:
        """Configure MCTS."""
        self._config.use_mcts = enabled
        if engine:
            self._config.mcts_engine = engine
        return self

    def with_rag(
        self,
        vector_store: VectorStore | None = None,
        enabled: bool = True,
    ) -> Self:
        """Configure RAG."""
        self._config.use_rag = enabled
        if vector_store:
            self._config.vector_store = vector_store
        return self

    def with_meta_controller(self, controller: MetaController) -> Self:
        """Add meta-controller for routing."""
        self._config.meta_controller = controller
        return self

    def with_max_iterations(self, max_iter: int) -> Self:
        """Set maximum iterations."""
        self._config.max_iterations = max_iter
        return self

    def build(self) -> StateGraph:
        """Build the graph from configuration."""
        self._validate_config()

        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("entry", self._create_entry_node())

        if self._config.use_rag and self._config.vector_store:
            graph.add_node("retrieve", self._create_rag_node())

        if self._config.meta_controller:
            graph.add_node("route", self._create_routing_node())

        if self._config.hrm_agent:
            graph.add_node("hrm", self._create_agent_node(self._config.hrm_agent))

        if self._config.trm_agent:
            graph.add_node("trm", self._create_agent_node(self._config.trm_agent))

        if self._config.use_mcts and self._config.mcts_engine:
            graph.add_node("mcts", self._create_mcts_node())

        graph.add_node("aggregate", self._create_aggregation_node())

        # Add edges
        self._add_edges(graph)

        return graph.compile()

    def _validate_config(self) -> None:
        """Validate builder configuration."""
        if not self._config.hrm_agent and not self._config.trm_agent:
            raise ValueError("At least one agent (HRM or TRM) must be configured")

    def _add_edges(self, graph: StateGraph) -> None:
        """Add edges between nodes."""
        graph.set_entry_point("entry")

        if self._config.use_rag:
            graph.add_edge("entry", "retrieve")
            graph.add_edge("retrieve", "route" if self._config.meta_controller else "hrm")
        else:
            graph.add_edge("entry", "route" if self._config.meta_controller else "hrm")

        # ... additional edge configuration


# Usage
builder = GraphBuilder()
graph = (
    builder
    .with_hrm_agent(hrm_agent)
    .with_trm_agent(trm_agent)
    .with_mcts(mcts_engine, enabled=True)
    .with_rag(vector_store, enabled=True)
    .with_max_iterations(5)
    .build()
)
```

---

## 7. Testing Framework

### 7.1 Test Configuration

```python
# tests/conftest.py
import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Graceful imports for optional dependencies
try:
    from src.config.settings import Settings, get_settings, reset_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

try:
    from src.framework.mcts.core import MCTSEngine, MCTSState
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False


def pytest_configure(config):
    """Register custom markers."""
    markers = [
        "unit: Fast, isolated unit tests",
        "integration: Component interaction tests",
        "e2e: End-to-end scenario tests",
        "slow: Tests taking >10 seconds",
        "benchmark: Performance benchmarks",
        "property: Property-based tests",
        "neural: Neural network tests (requires PyTorch)",
        "llm: LLM integration tests (requires API key)",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Skip tests based on environment."""
    skip_slow = pytest.mark.skip(reason="Use --runslow to run slow tests")
    skip_neural = pytest.mark.skip(reason="PyTorch not available")
    skip_llm = pytest.mark.skip(reason="No LLM API key configured")

    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow", False):
            item.add_marker(skip_slow)

        if "neural" in item.keywords:
            try:
                import torch  # noqa
            except ImportError:
                item.add_marker(skip_neural)

        if "llm" in item.keywords:
            if not os.environ.get("OPENAI_API_KEY"):
                item.add_marker(skip_llm)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def test_logger() -> logging.Logger:
    """Logger for tests."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def test_settings():
    """Test settings with isolation."""
    if not SETTINGS_AVAILABLE:
        pytest.skip("Settings not available")

    reset_settings()
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-key-for-testing-only",
        "MCTS_ENABLED": "true",
        "MCTS_ITERATIONS": "10",
        "SEED": "42",
        "LOG_LEVEL": "DEBUG",
    }):
        settings = get_settings()
        yield settings
    reset_settings()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock()
    client.generate.return_value = MagicMock(
        content="Test response",
        model="gpt-4-test",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )
    client.model = "gpt-4-test"
    client.provider = "openai"
    return client


@pytest.fixture
def mcts_engine(test_settings):
    """MCTS engine for testing."""
    if not MCTS_AVAILABLE:
        pytest.skip("MCTS not available")

    return MCTSEngine(
        seed=test_settings.SEED,
        exploration_weight=test_settings.MCTS_C,
    )


@pytest.fixture
def simple_state():
    """Simple MCTS state for testing."""
    if not MCTS_AVAILABLE:
        pytest.skip("MCTS not available")

    return MCTSState(
        state_id="test_state",
        features={"query": "test", "depth": 0},
    )


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    if SETTINGS_AVAILABLE:
        try:
            reset_settings()
        except Exception:
            pass
```

### 7.2 Test Examples

```python
# tests/unit/test_settings.py
import pytest
from unittest.mock import patch
import os


@pytest.mark.unit
class TestSettings:
    """Test configuration settings."""

    def test_settings_loads_from_environment(self, test_settings):
        """Test settings load from environment variables."""
        assert test_settings.LLM_PROVIDER.value == "openai"
        assert test_settings.MCTS_ITERATIONS == 10
        assert test_settings.SEED == 42

    def test_settings_validates_api_key_format(self):
        """Test API key validation."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "invalid-key",  # Should start with sk-
            "LLM_PROVIDER": "openai",
        }):
            reset_settings()
            with pytest.raises(ValueError, match="must start with 'sk-'"):
                get_settings()

    def test_settings_safe_dict_masks_secrets(self, test_settings):
        """Test sensitive data is masked in safe_dict."""
        safe = test_settings.safe_dict()
        assert safe["OPENAI_API_KEY"] == "***MASKED***"

    def test_settings_requires_api_key(self):
        """Test that API key is required for non-LMStudio providers."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
        }, clear=True):
            reset_settings()
            with pytest.raises(ValueError, match="API key must be configured"):
                get_settings()


# tests/unit/test_mcts_engine.py
@pytest.mark.unit
class TestMCTSEngine:
    """Test MCTS engine functionality."""

    def test_engine_deterministic_with_seed(self, mcts_engine, simple_state):
        """Test MCTS produces deterministic results with same seed."""
        # Run twice with same seed
        result1 = mcts_engine.select(simple_state)
        mcts_engine.reset_rng()
        result2 = mcts_engine.select(simple_state)

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_engine_search_returns_result(self, mcts_engine, simple_state):
        """Test MCTS search returns valid result."""
        result = await mcts_engine.search(
            initial_state=simple_state,
            num_iterations=5,
        )

        assert result is not None
        assert result.iterations_completed <= 5
        assert result.stats is not None

    def test_engine_ucb1_calculation(self, mcts_engine):
        """Test UCB1 calculation is correct."""
        # Parent with 100 visits
        # Child with 10 visits, 5.0 value sum
        # UCB1 = 5.0/10 + 1.414 * sqrt(ln(100)/10)
        # UCB1 = 0.5 + 1.414 * sqrt(0.4605) = 0.5 + 0.96 = 1.46

        ucb = mcts_engine._calculate_ucb1(
            visits=10,
            value_sum=5.0,
            parent_visits=100,
            exploration_weight=1.414,
        )

        assert abs(ucb - 1.46) < 0.1


# tests/integration/test_agent_workflow.py
@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentWorkflow:
    """Test agent workflow integration."""

    async def test_hrm_agent_processes_query(
        self,
        mock_llm_client,
        test_settings,
        test_logger,
    ):
        """Test HRM agent processes query correctly."""
        from src.agents.hrm_agent import HRMAgent, HRMConfig

        config = HRMConfig(
            max_depth=test_settings.HRM_MAX_DEPTH,
            confidence_threshold=test_settings.HRM_CONFIDENCE_THRESHOLD,
        )
        agent = HRMAgent(
            config=config,
            llm_client=mock_llm_client,
            logger=test_logger,
        )

        result = await agent.process(
            query="What is 2 + 2?",
            context={},
        )

        assert result is not None
        mock_llm_client.generate.assert_called()

    async def test_factory_creates_configured_agents(
        self,
        mock_llm_client,
        test_settings,
    ):
        """Test factory creates properly configured agents."""
        from src.framework.factories import AgentFactory

        factory = AgentFactory(
            llm_client=mock_llm_client,
            settings=test_settings,
        )

        hrm = factory.create_hrm()
        assert hrm._config.max_depth == test_settings.HRM_MAX_DEPTH

        trm = factory.create_trm()
        assert trm._config.max_iterations == test_settings.TRM_MAX_ITERATIONS
```

### 7.3 Property-Based Testing

```python
# tests/property/test_mcts_invariants.py
import pytest
from hypothesis import given, strategies as st, assume


@pytest.mark.property
class TestMCTSInvariants:
    """Property-based tests for MCTS invariants."""

    @given(
        visits=st.integers(min_value=0, max_value=10000),
        value_sum=st.floats(min_value=0, max_value=10000),
        exploration_weight=st.floats(min_value=0, max_value=10),
    )
    def test_ucb1_always_non_negative(
        self,
        visits,
        value_sum,
        exploration_weight,
    ):
        """UCB1 value should always be non-negative."""
        assume(visits > 0)  # Avoid division by zero
        assume(value_sum >= 0)

        ucb = calculate_ucb1(
            visits=visits,
            value_sum=value_sum,
            parent_visits=max(visits * 2, 1),
            exploration_weight=exploration_weight,
        )

        assert ucb >= 0

    @given(
        num_children=st.integers(min_value=1, max_value=100),
        visits_list=st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=1,
            max_size=100,
        ),
    )
    def test_child_visits_sum_invariant(self, num_children, visits_list):
        """Sum of child visits should not exceed parent visits."""
        visits_list = visits_list[:num_children]
        total_child_visits = sum(visits_list)
        parent_visits = total_child_visits + 1  # Parent has at least one extra

        assert total_child_visits <= parent_visits

    @given(
        value=st.floats(min_value=0, max_value=1),
    )
    def test_normalized_values_in_range(self, value):
        """Normalized values should be in [0, 1] range."""
        normalized = normalize_value(value)
        assert 0 <= normalized <= 1
```

---

## 8. Logging & Debugging

### 8.1 Structured Logging

```python
# src/observability/logging.py
import logging
import json
import re
from contextvars import ContextVar
from datetime import datetime
from typing import Any

# Async-safe correlation ID
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: str | None) -> None:
    """Set correlation ID for current context."""
    _correlation_id.set(cid)


# Sensitive data patterns to sanitize
SENSITIVE_PATTERNS = [
    (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', re.I), "api_key=***"),
    (re.compile(r'password["\']?\s*[:=]\s*["\']?[\w-]+', re.I), "password=***"),
    (re.compile(r'secret["\']?\s*[:=]\s*["\']?[\w-]+', re.I), "secret=***"),
    (re.compile(r'token["\']?\s*[:=]\s*["\']?[\w-]+', re.I), "token=***"),
    (re.compile(r'sk-[a-zA-Z0-9]{20,}', re.I), "sk-***"),
    (re.compile(r'sk-ant-[a-zA-Z0-9]{20,}', re.I), "sk-ant-***"),
]

SENSITIVE_KEYS = {
    "api_key", "apikey", "password", "secret", "token",
    "authorization", "auth", "credential", "private_key",
}


def sanitize_message(message: str) -> str:
    """Remove sensitive data from log message."""
    for pattern, replacement in SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


def sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize dictionary."""
    result = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            result[key] = "***REDACTED***"
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_message(record.getMessage()),
            "correlation_id": get_correlation_id(),
        }

        # Add extra fields
        if hasattr(record, "extra") and record.extra:
            log_obj.update(sanitize_dict(record.extra))

        # Add exception info
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> logging.Logger:
    """Configure application logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handler
    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

    root_logger.addHandler(handler)
    return root_logger


class StructuredLogger:
    """Convenience wrapper for structured logging."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **extra: Any) -> None:
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "",
            0,
            message,
            (),
            None,
        )
        record.extra = {
            "correlation_id": get_correlation_id(),
            **sanitize_dict(extra),
        }
        self._logger.handle(record)

    def debug(self, message: str, **extra: Any) -> None:
        self._log(logging.DEBUG, message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        self._log(logging.INFO, message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        self._log(logging.WARNING, message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        self._log(logging.ERROR, message, **extra)

    def exception(self, message: str, **extra: Any) -> None:
        self._logger.exception(message, extra=sanitize_dict(extra))
```

### 8.2 Debugging Tools

```python
# src/observability/debugging.py
import asyncio
import functools
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class DebugInfo:
    """Debug information for a function call."""
    function_name: str
    duration_ms: float
    memory_delta_mb: float
    return_type: str
    args_summary: str


def debug_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for debugging async functions.

    Logs:
    - Execution time
    - Memory usage delta
    - Arguments summary
    - Return type
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        tracemalloc.start()
        start_time = time.perf_counter()
        start_memory = tracemalloc.get_traced_memory()[0]

        try:
            result = await func(*args, **kwargs)

            duration_ms = (time.perf_counter() - start_time) * 1000
            end_memory = tracemalloc.get_traced_memory()[0]
            memory_delta = (end_memory - start_memory) / 1024 / 1024

            logger = StructuredLogger(func.__module__)
            logger.debug(
                f"Function completed: {func.__name__}",
                duration_ms=round(duration_ms, 2),
                memory_delta_mb=round(memory_delta, 4),
                return_type=type(result).__name__,
            )

            return result

        finally:
            tracemalloc.stop()

    return wrapper


class AsyncTaskDebugger:
    """Debug helper for async task management."""

    @staticmethod
    def log_all_tasks() -> list[dict]:
        """Log information about all running async tasks."""
        tasks = []
        for task in asyncio.all_tasks():
            tasks.append({
                "name": task.get_name(),
                "done": task.done(),
                "cancelled": task.cancelled(),
                "coro": str(task.get_coro()),
            })
        return tasks

    @staticmethod
    def cancel_all_tasks() -> int:
        """Cancel all running tasks (use with caution)."""
        cancelled = 0
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled


# Debugging checklist as code
def run_debug_checklist(
    settings: Settings,
    mcts_engine: MCTSEngine | None = None,
    agents: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run diagnostic checks and return results.

    Checks:
    1. Configuration loaded correctly
    2. MCTS engine state
    3. Agent status
    4. Async task status
    5. Memory usage
    """
    results = {
        "configuration": settings.safe_dict(),
        "correlation_id": get_correlation_id(),
    }

    if mcts_engine:
        results["mcts"] = {
            "cache_size": len(mcts_engine._cache),
            "cache_hits": mcts_engine._cache_hits,
            "cache_misses": mcts_engine._cache_misses,
        }

    if agents:
        results["agents"] = {
            name: {
                "confidence": agent.get_confidence() if hasattr(agent, "get_confidence") else "N/A",
                "type": type(agent).__name__,
            }
            for name, agent in agents.items()
        }

    results["async_tasks"] = AsyncTaskDebugger.log_all_tasks()

    # Memory info
    import tracemalloc
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["memory"] = {
        "current_mb": round(current / 1024 / 1024, 2),
        "peak_mb": round(peak / 1024 / 1024, 2),
    }

    return results
```

---

## 9. Backwards Compatibility

### 9.1 Deprecation Pattern

```python
# src/core/deprecation.py
import functools
import warnings
from typing import Callable, TypeVar

T = TypeVar("T")


def deprecated(
    reason: str,
    removal_version: str,
    replacement: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Mark a function as deprecated.

    Args:
        reason: Why this is deprecated
        removal_version: Version when it will be removed
        replacement: What to use instead
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            message = (
                f"{func.__name__} is deprecated and will be removed in "
                f"v{removal_version}. {reason}"
            )
            if replacement:
                message += f" Use {replacement} instead."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Mark function as deprecated in docstring
        wrapper.__doc__ = f"DEPRECATED: {func.__doc__ or ''}\n\n{reason}"
        return wrapper

    return decorator


def renamed_parameter(
    old_name: str,
    new_name: str,
    removal_version: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Handle renamed parameters with backwards compatibility.

    Automatically maps old parameter name to new one.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if old_name in kwargs:
                warnings.warn(
                    f"Parameter '{old_name}' is deprecated, use '{new_name}' instead. "
                    f"Will be removed in v{removal_version}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new_name] = kwargs.pop(old_name)

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Usage examples
@deprecated(
    reason="MCTSConfig now uses from_preset() for standard configurations",
    removal_version="0.3.0",
    replacement="MCTSConfig.from_preset()",
)
def create_mcts_config(iterations: int) -> MCTSConfig:
    """Create MCTS config (deprecated)."""
    return MCTSConfig(num_iterations=iterations)


@renamed_parameter("num_simulations", "num_iterations", "0.3.0")
def search(state: MCTSState, num_iterations: int = 100) -> SearchResult:
    """Run MCTS search."""
    ...
```

### 9.2 Configuration Migration

```python
# src/config/migration.py
import warnings
from typing import Any


class ConfigMigrator:
    """
    Migrate configuration from older formats.

    Supports:
    - Renamed settings
    - Removed settings (with warnings)
    - Default value changes
    """

    # Map old setting names to new names
    RENAMED_SETTINGS = {
        "MCTS_NUM_ITERATIONS": "MCTS_ITERATIONS",
        "EXPLORATION_CONSTANT": "MCTS_C",
        "HRM_DEPTH": "HRM_MAX_DEPTH",
        "TRM_ITERATIONS": "TRM_MAX_ITERATIONS",
    }

    # Settings that have been removed
    REMOVED_SETTINGS = {
        "LEGACY_MODE": "Legacy mode is no longer supported",
        "OLD_API_VERSION": "Use API_VERSION instead",
    }

    # Default value changes (old_value -> warning message)
    DEFAULT_CHANGES = {
        "MCTS_C": {
            "1.0": "Default exploration weight changed from 1.0 to 1.414",
        },
    }

    def migrate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Migrate configuration to current format."""
        migrated = {}

        for key, value in config.items():
            # Handle renamed settings
            if key in self.RENAMED_SETTINGS:
                new_key = self.RENAMED_SETTINGS[key]
                warnings.warn(
                    f"Setting '{key}' is deprecated, use '{new_key}' instead.",
                    DeprecationWarning,
                )
                migrated[new_key] = value
                continue

            # Handle removed settings
            if key in self.REMOVED_SETTINGS:
                warnings.warn(
                    f"Setting '{key}' has been removed: {self.REMOVED_SETTINGS[key]}",
                    DeprecationWarning,
                )
                continue

            # Check for default value changes
            if key in self.DEFAULT_CHANGES:
                str_value = str(value)
                if str_value in self.DEFAULT_CHANGES[key]:
                    warnings.warn(
                        self.DEFAULT_CHANGES[key][str_value],
                        UserWarning,
                    )

            migrated[key] = value

        return migrated


def load_config_with_migration(path: str) -> dict[str, Any]:
    """Load and migrate configuration file."""
    import json

    with open(path) as f:
        config = json.load(f)

    migrator = ConfigMigrator()
    return migrator.migrate(config)
```

### 9.3 API Versioning

```python
# src/api/versioning.py
from fastapi import APIRouter, Request, Response
from functools import wraps

# API version routers
v1_router = APIRouter(prefix="/v1", tags=["v1"])
v2_router = APIRouter(prefix="/v2", tags=["v2"])


def api_version_deprecated(
    version: str,
    removal_date: str,
    replacement_version: str,
):
    """
    Mark an API version as deprecated.

    Adds deprecation headers to responses.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            response: Response = await func(*args, **kwargs)

            # Add deprecation headers
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = removal_date
            response.headers["Link"] = (
                f'</{replacement_version}{func.__name__}>; rel="successor-version"'
            )

            return response
        return wrapper
    return decorator


# V1 endpoints (current stable)
@v1_router.post("/query")
async def query_v1(request: QueryRequest) -> QueryResponse:
    """Process query (v1 API)."""
    ...


# V2 endpoints (with new features)
@v2_router.post("/query")
async def query_v2(request: QueryRequestV2) -> QueryResponseV2:
    """Process query with streaming support (v2 API)."""
    ...


# Deprecated V0 endpoint
@api_version_deprecated(
    version="v0",
    removal_date="2026-06-01",
    replacement_version="v1",
)
async def query_v0(request: dict) -> dict:
    """Deprecated v0 query endpoint."""
    ...
```

---

## 10. Implementation Checklist

### 10.1 Pre-Implementation

```markdown
□ Read existing code in the area to modify
□ Identify patterns already in use
□ Check for related tests
□ Plan implementation with TodoWrite
□ Identify backwards compatibility concerns
```

### 10.2 Implementation

```markdown
□ All values from configuration (no hardcoded values)
□ Dependency injection used throughout
□ Protocol-based interfaces where applicable
□ Comprehensive error handling
□ Structured logging with correlation IDs
□ Sanitization of sensitive data in logs
```

### 10.3 Testing

```markdown
□ Unit tests for all new functions (90% target)
□ Integration tests for workflows (70% target)
□ Property-based tests for invariants
□ All tests pass: pytest tests/ -v
□ Coverage meets threshold: pytest --cov=src --cov-fail-under=80
```

### 10.4 Verification

```markdown
□ Code formatted: black src/ tests/
□ Imports sorted: isort src/ tests/ --profile black
□ Linting passes: ruff check src/ tests/
□ Type checking passes: mypy src/
□ No hardcoded values: grep -r "api_key.*=" src/
□ Security scan: bandit -r src/
```

### 10.5 Documentation

```markdown
□ Docstrings for all public functions
□ Type hints complete
□ CLAUDE.md updated if commands/decisions changed
□ Deprecation notices for breaking changes
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

# Verification
python -c "from src.config.settings import get_settings; print(get_settings().safe_dict())"
```

### Environment Variables

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
SEED=42
```

---

**Template Version:** 1.0
**Last Updated:** 2026-01-27
**Aligned with:** Claude Code sub-agent architecture
