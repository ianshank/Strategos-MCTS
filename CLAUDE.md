# CLAUDE.md - Project Context for AI Assistants

> Quick reference for Claude Code and other AI assistants working on this codebase.
> For implementation templates, see:
> - `MULTI_AGENT_MCTS_TEMPLATE.md` - Comprehensive template with C4 architecture (v2.0)
> - `CLAUDE_CODE_IMPLEMENTATION_TEMPLATE.md` - Original implementation patterns

---

## Quick Start

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"           # Development only
pip install -e ".[dev,neural]"    # Include PyTorch for neural MCTS

# Configure environment
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)

# Verify installation
pytest tests/unit -v --tb=short -q
```

---

## Build Commands

| Command | Purpose |
|---------|---------|
| `pip install -e ".[dev]"` | Install with dev dependencies |
| `pip install -e ".[dev,benchmark]"` | Include benchmark framework |
| `pip install -e ".[dev,neural]"` | Include PyTorch for neural MCTS |
| `black src/ tests/ --line-length 120` | Format code |
| `isort src/ tests/ --profile black` | Sort imports |
| `ruff check src/ tests/ --fix` | Lint with auto-fix |
| `mypy src/ --strict` | Type check |

---

## Test Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit -v` | Run unit tests |
| `pytest tests/integration -v` | Run integration tests |
| `pytest tests/ -k "mcts"` | Run MCTS-related tests |
| `pytest tests/ --cov=src --cov-report=term-missing` | Run with coverage |
| `pytest tests/ -m "not slow"` | Skip slow tests |
| `pytest tests/unit -x` | Stop on first failure |
| `pytest tests/unit/benchmark -v` | Run benchmark framework tests |

---

## Benchmark Commands

| Command | Purpose |
|---------|---------|
| `python -m src.benchmark` | Run full benchmark with defaults |
| `python -m src.benchmark --systems langgraph_mcts` | Benchmark single system |
| `python -m src.benchmark --tasks A1 A2 B1` | Run specific tasks |
| `python -m src.benchmark --iterations 3` | Run 3 iterations for significance |
| `python -m src.benchmark --dry-run` | Preview tasks/systems without running |
| `python -m src.benchmark --no-scoring` | Run without LLM scoring |
| `python -m src.benchmark --output-dir ./results` | Custom output directory |

---

## Key File Locations

```
CONFIGURATION
├── src/config/settings.py       # Pydantic Settings (all config here)
├── .env                         # Environment variables (secrets)
└── pyproject.toml               # Dependencies, tool config

CORE FRAMEWORK
├── src/framework/graph.py       # LangGraph orchestration
├── src/framework/mcts/core.py   # MCTS engine
└── src/framework/factories.py   # Component factories

AGENTS
├── src/agents/hrm_agent.py      # Hierarchical Reasoning Module
├── src/agents/trm_agent.py      # Task Refinement Module
├── src/agents/hybrid_agent.py   # LLM + Neural hybrid
└── src/agents/meta_controller/  # Neural routing

LLM ADAPTERS
├── src/adapters/llm/base.py     # Protocol & interfaces
├── src/adapters/llm/openai_client.py
├── src/adapters/llm/anthropic_client.py
└── src/adapters/llm/lmstudio_client.py

OBSERVABILITY
├── src/observability/logging.py # Structured logging
├── src/observability/metrics.py # Prometheus metrics
└── src/observability/tracing.py # Distributed tracing

BENCHMARK FRAMEWORK
├── src/benchmark/cli.py         # CLI entry point (python -m src.benchmark)
├── src/benchmark/factory.py     # BenchmarkFactory (wires all components)
├── src/benchmark/config/        # Pydantic Settings benchmark configuration
├── src/benchmark/tasks/         # Task models, registry, default task sets
├── src/benchmark/adapters/      # System adapters (LangGraph, ADK) + factory
├── src/benchmark/evaluation/    # Harness, scorer, cost calculator, models
└── src/benchmark/reporting/     # Metrics aggregator, report generator

TESTS
├── tests/unit/                  # Unit tests
├── tests/unit/benchmark/        # Benchmark framework unit tests
├── tests/integration/           # Integration tests
└── tests/fixtures/              # Shared test fixtures
```

---

## Architecture Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024-01 | Pydantic Settings v2 for config | Type safety, validation, env loading |
| 2024-01 | LangGraph for orchestration | Native async, checkpointing, visualization |
| 2024-02 | Protocol-based LLM adapters | Provider agnosticism without ABC overhead |
| 2024-03 | TypedDict for AgentState | Better IDE support than dataclass for state machines |
| 2024-04 | ContextVar for correlation IDs | Async-safe request tracking |
| 2024-05 | Factory pattern for components | Testability, dependency injection |

---

## Configuration Patterns

### All Configuration via Pydantic Settings

```python
# CORRECT - Use settings
from src.config.settings import get_settings
settings = get_settings()
api_key = settings.get_api_key()
iterations = settings.MCTS_ITERATIONS

# WRONG - Hardcoded values
api_key = "sk-xxx"  # Never!
iterations = 100    # Use settings
```

### Required Environment Variables

```bash
# One of these is required
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Provider selection
LLM_PROVIDER=openai  # openai | anthropic | lmstudio
```

### Optional Environment Variables

```bash
LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR
MCTS_ENABLED=true           # Enable MCTS exploration
MCTS_ITERATIONS=100         # Search iterations
MCTS_C=1.414                # Exploration weight (UCB1)
SEED=42                     # For reproducibility
LANGSMITH_API_KEY=ls-...    # For tracing
PINECONE_API_KEY=...        # For vector storage
```

---

## Common Patterns

### Dependency Injection

```python
class MyComponent:
    def __init__(
        self,
        config: MyConfig,           # Configuration injected
        llm_client: LLMClient,      # Dependencies injected
        logger: logging.Logger,     # Logger injected
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._logger = logger
```

### Async Operations

```python
# All I/O is async
async def process(self, query: str) -> Result:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return Result(data=response.json())
```

### Logging with Correlation ID

```python
from src.observability.logging import get_correlation_id, sanitize_dict

self._logger.info(
    "Processing request",
    extra={
        "correlation_id": get_correlation_id(),
        "data": sanitize_dict(sensitive_data),  # Masks secrets
    }
)
```

---

## Known Issues & Workarounds

| Issue | Workaround |
|-------|------------|
| LMStudio tests fail without local server | Set `LMSTUDIO_SKIP=1` to skip |
| Pinecone tests require valid API key | Use mocks in CI, real key locally |
| Neural MCTS slow on CPU | Use CUDA or reduce iterations |
| Type errors with langchain | Use `# type: ignore[import]` |

---

## Test Markers

```python
@pytest.mark.unit          # Fast, isolated tests
@pytest.mark.integration   # Component interaction tests
@pytest.mark.e2e           # End-to-end scenarios
@pytest.mark.slow          # Tests >10 seconds
@pytest.mark.benchmark     # Performance tests
@pytest.mark.property      # Property-based tests
```

---

## Verification Checklist

Before committing, verify:

```bash
# 1. Format
black src/ tests/ --check

# 2. Lint
ruff check src/ tests/

# 3. Types
mypy src/

# 4. Tests
pytest tests/unit -v

# 5. No hardcoded values
grep -r "api_key.*=.*['\"]sk-" src/ && echo "FAIL: Hardcoded keys!" || echo "OK"
```

---

## Getting Help

- **Template (v2.0)**: See `MULTI_AGENT_MCTS_TEMPLATE.md` for comprehensive template with:
  - Full C4 architecture diagrams
  - All sub-agent specifications (HRM, TRM, Meta-Controller, MCTS)
  - Dynamic component patterns and factories
  - Complete test suite patterns
  - Logging and observability patterns
- **Original Template**: See `CLAUDE_CODE_IMPLEMENTATION_TEMPLATE.md` for detailed patterns
- **Architecture**: See `docs/C4_ARCHITECTURE.md` for system diagrams
- **Training**: See `docs/LOCAL_TRAINING_GUIDE.md` for ML pipeline
- **Deployment**: See `docs/DEPLOYMENT_REPORT.md` for deployment status

---

*Last Updated: 2025-01*
