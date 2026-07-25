# CLAUDE.md - Project Context for AI Assistants

> Quick reference for Claude Code and other AI assistants working on this codebase.
> For the implementation template, see:
> - `docs/templates/MULTI_AGENT_MCTS_TEMPLATE.md` - Comprehensive template with C4 architecture (v2.0)

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
| `black . --line-length 120` | Format code |
| `ruff check . --select I --fix` | Sort imports (ruff owns isort rules) |
| `ruff check . --fix` | Lint with auto-fix |
| `mypy src/ --strict` | Type check |

> **Tooling is pinned for CI/local parity.** `ruff` and `mypy` are pinned to a validated
> minor in the `[dev]` extra (the CI lint job installs `.[dev]`). Bump them deliberately and
> re-run the full local gate (`ruff`/`black`/`mypy`/`pytest`) before changing the pins —
> e.g. mypy's `no-redef`/`unused-ignore` behavior shifts across releases.

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
├── src/config/constants.py      # Shared defaults & bounds (model names, URLs, magic numbers)
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
├── src/adapters/llm/resilience.py   # Shared CircuitBreaker (provider-agnostic)
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
black . --check

# 2. Lint
ruff check .

# 3. Types
mypy src/

# 4. Tests
pytest tests/unit -v

# 5. No hardcoded values
grep -r "api_key.*=.*['\"]sk-" src/ && echo "FAIL: Hardcoded keys!" || echo "OK"
```

---

## Agent Harness Framework

| Path | Purpose |
|------|---------|
| `src/framework/harness/settings.py` | `HarnessSettings` — `HARNESS_*` env vars |
| `src/framework/harness/loop/runner.py` | `HarnessRunner` — six-phase deterministic loop |
| `src/framework/harness/loop/facade.py` | `HarnessAgentAdapter` — `AsyncAgentBase` facade |
| `src/framework/harness/memory/` | Append-only event log + compactor → `MEMORY.md` |
| `src/framework/harness/tools/` | `AsyncToolExecutor` + builtins (file/shell/test/lint/types) |
| `src/framework/harness/hooks/` | `HookChain` + secret/size/required-keys hooks |
| `src/framework/harness/topology/` | Six topologies (pipeline, fan-out-in, expert pool, producer-reviewer, supervisor, hierarchical) |
| `src/framework/harness/ralph/` | Spec-driven outer loop |
| `src/framework/harness/replay/` | Record/replay cassettes + deterministic clock |
| `src/framework/harness/cli.py` | `harness` console script (`run`, `dry-run`, `replay`, `validate-spec`) |
| `AGENTS.md` (root) | Routing ledger for autonomous agents |

CLI quick reference:

```bash
harness validate-spec specs/*.SPEC.md  # schema v2 validation; errors exit 1
harness dry-run --spec spec.md     # plan only, no LLM calls
harness run --spec spec.md         # full deterministic loop
harness run --spec spec.md --ralph # outer Ralph loop
```

## Getting Help

- **Template (v2.0)**: See `docs/templates/MULTI_AGENT_MCTS_TEMPLATE.md` for comprehensive template with:
  - Full C4 architecture diagrams
  - All sub-agent specifications (HRM, TRM, Meta-Controller, MCTS)
  - Dynamic component patterns and factories
  - Complete test suite patterns
  - Logging and observability patterns
- **Architecture**: See `docs/C4_ARCHITECTURE.md` for system diagrams
- **Training**: See `docs/LOCAL_TRAINING_GUIDE.md` for ML pipeline
- **Deployment**: See `docs/DOCKER_DEPLOYMENT.md` for the deployment guide and `docs/STATUS.md` for
  current status (the historical `docs/archive/reports/DEPLOYMENT_REPORT.md` is retained for reference)
- **Current status (source of truth)**: See `docs/STATUS.md` for the reproducible test/coverage
  baseline, and `docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md` for the active roadmap.

## Spec-Driven Development

Phase work is specified as Markdown specs under `specs/<id>.SPEC.md` (schema v2: frontmatter
`id`/`goal`/`module`/`status` lifecycle + Goal / Acceptance Criteria with `AC-n:` IDs /
Constraints), parsed by `src/framework/harness/intent/spec_loader.py` and enforced by
`src/framework/harness/intent/spec_validator.py`:

```bash
harness validate-spec specs/*.SPEC.md   # schema v2 validation (errors exit 1)
harness dry-run --spec specs/phase_1_correctness.SPEC.md  # plan only, no LLM
```

### Spec-driven workflow (SDD Phase 1 enforcement)

- `/spec-new <id> <module>` scaffolds a `draft` spec (deterministic refusal on malformed ids and
  module overlap with open specs); the `spec-review` subagent gates draft→approved (a human flips
  the status); `/spec-implement <id>` requires `approved`, then creates/switches to the
  `spec/<id>` branch cut from `origin/main`.
- A **PreToolUse gate** (`.claude/hooks/spec_gate.py`, committed `.claude/settings.json`) warns on
  Edit/Write/MultiEdit/NotebookEdit under `src/**` unless the branch is `spec/<id>` with an
  approved/implemented spec. Warn mode during the pilot; `SPEC_GATE_BYPASS=1` is the hotfix
  bypass. Known holes: Bash-based writes are not gated; native Windows without `python3` degrades
  to non-blocking errors.
- **CI traceability** (`harness spec-trace`, run by the `spec-validate` job on PRs): diffs
  touching `src/**` need a `spec/<id>` branch whose spec is `approved` on the base branch, or a
  `No-Spec: <reason>` commit trailer. Flips to `verified` require same-line `spec-id AC-n`
  mappings under `tests/`. Until the first approved spec merges, the trailer is the expected
  channel for all `src/**` work.

Reusable project skills live in `.claude/skills/`: `quality-gate` (full local gate),
`validate-specs` (validate all specs), `coverage-baseline` (refresh `docs/STATUS.md`).

---

*Last Updated: 2026-06-30*
