# Testing Infrastructure - 2025 Best Practices

This document describes the comprehensive testing infrastructure for the LangGraph Multi-Agent MCTS Framework.

## Overview

The testing suite implements modern testing practices including:
- **Contract Testing**: Validates interface compliance
- **Property-Based Testing**: Tests invariants across wide input ranges
- **Performance Testing**: Ensures acceptable performance characteristics
- **Builder Pattern**: Fluent API for test data creation
- **Test Markers**: Organized test categorization

## Test Categories

### 1. Contract Tests (`test_contracts.py`)

Contract tests ensure that components adhere to their defined interfaces and protocols.

```python
@pytest.mark.contract
def test_llm_client_implements_protocol():
    """Ensures all LLM clients follow the same interface."""
    assert hasattr(client, 'generate')
    assert callable(client.generate)
```

**Coverage:**
- LLM client protocol compliance (OpenAI, Anthropic, LM Studio)
- Agent interface validation (AgentContext, AgentResult)
- MCTS component contracts (MCTSState, MCTSNode)
- Factory method contracts
- Validation model contracts

**Run contract tests:**
```bash
pytest -m contract
```

### 2. Property-Based Tests (`test_properties.py`)

Property-based tests use hypothesis to validate invariants across thousands of generated inputs.

```python
@pytest.mark.property
@given(visits=st.integers(min_value=0, max_value=10000))
def test_node_value_invariant(visits: int):
    """Tests that node.value == value_sum / visits always holds."""
    # Test runs with hundreds of different visit counts
```

**Coverage:**
- MCTS node invariants (value calculation, depth tracking)
- State hash determinism
- Token counting invariants
- Configuration validation
- Agent context preservation

**Run property tests:**
```bash
pytest -m property
```

### 3. Performance Tests (`test_performance.py`)

Performance tests ensure the framework maintains acceptable performance characteristics.

```python
@pytest.mark.performance
def test_node_expansion_speed():
    """Ensures node expansion < 1ms per operation."""
    # Measures actual performance
```

**Coverage:**
- MCTS operation speed (node expansion, hashing, UCB1)
- Agent operation overhead (context creation, serialization)
- Factory instantiation performance
- Validation overhead
- Async task scheduling
- Memory efficiency
- Throughput tests

**Run performance tests:**
```bash
pytest -m performance
```

### 4. Component Tests

Existing component tests for specific modules (HRM, TRM, MCTS agents).

```bash
pytest -m component
```

### 5. Integration & E2E Tests

```bash
make test-e2e        # what CI runs: pytest tests/e2e -m "not ui"
pytest -m integration
```

**What belongs in `tests/e2e/`.** A module qualifies only if it drives a *real entry
point* — an installed console script, `python -m`, or the FastAPI app through its
lifespan — with every component inside the process real and only the network boundary
stubbed, through the paths the code already offers (`ALLOW_MOCK_LLM_FALLBACK`, the offline
hub flags). Re-implementing an algorithm in the test is not an end-to-end test. The full
contract and the reasoning behind it are in
`docs/plans/2026-09-04-e2e-device-agnostic.md`.

**The device matrix.** Anything that moves a tensor is parametrized over `cpu`, `cuda` and
`mps` by the `device_case` / `device` fixtures in `tests/e2e/conftest.py`, which are built
on `tests/utils/device_matrix.py`. Never write a device literal in a test; take the
fixture. The matrix is static so that unavailable devices are reported as **skipped with a
reason** rather than being absent — on a CPU-only runner "all green" must not read as "the
CUDA path is tested". Non-CPU cases carry the `gpu` marker, so `-m "not gpu"` deselects
them. Use `accelerator_case` for a test that compares an accelerator *against* CPU; it
collects as a single reasoned skip where no accelerator exists.

```bash
make test-e2e                      # every available device; others skip with a reason
E2E_DEVICES=cpu make test-e2e      # pin the matrix
E2E_DEVICES=cuda make test-e2e     # REQUIRE cuda: fails, not skips, if the host lacks it
```

`E2E_DEVICES` is test-side only; nothing under `src/` reads it. It is unrelated to
`FORCE_GPU_TESTS`, which belongs to the Docker smoke tests.

**Subprocesses.** Use the `run_script` / `run_module` fixtures rather than `subprocess`
directly. They strip every provider and tracker credential from the child's environment
before it starts (this suite also runs in a post-merge workflow that exports real API
keys), bound each child, kill whole process groups on timeout, and render a failure as
argv, exit code and both captured streams.

## Test Data Builders (`builders.py`)

Builders provide a fluent API for creating test fixtures.

### AgentContextBuilder

```python
from tests.builders import AgentContextBuilder

context = (
    AgentContextBuilder()
    .with_query("Secure defensive perimeter")
    .with_session_id("test-123")
    .with_rag_context("Tactical doctrine...")
    .with_confidence(0.95)
    .build()
)
```

### AgentResultBuilder

```python
from tests.builders import AgentResultBuilder

result = (
    AgentResultBuilder()
    .with_response("Recommended positions...")
    .with_confidence(0.88)
    .with_agent_name("HRM")
    .with_success(True)
    .build()
)
```

### MCTSStateBuilder

```python
from tests.builders import MCTSStateBuilder

state = (
    MCTSStateBuilder()
    .with_state_id("tactical_state_1")
    .with_feature("position", [10, 20])
    .with_feature("score", 85)
    .build()
)
```

### LLMResponseBuilder

```python
from tests.builders import LLMResponseBuilder

response = (
    LLMResponseBuilder()
    .with_text("Analysis complete...")
    .with_model("gpt-4")
    .with_tokens(prompt=100, completion=200)
    .build()
)
```

### TacticalScenarioBuilder

```python
from tests.builders import TacticalScenarioBuilder

scenario = (
    TacticalScenarioBuilder()
    .with_type("defensive")
    .with_complexity("high")
    .with_units(150)
    .add_objective("Secure perimeter")
    .add_constraint("Minimize casualties")
    .build()
)
```

### Convenience Functions

```python
from tests.builders import (
    minimal_agent_context,
    successful_agent_result,
    failed_agent_result,
    defensive_scenario,
    offensive_scenario,
)

# Quick test data creation
context = minimal_agent_context()
result = successful_agent_result("HRM")
scenario = defensive_scenario()
```

## Factory Pattern (`src/framework/factories.py`)

Factories enable dependency injection and modular component creation.

### LLMClientFactory

```python
from src.framework.factories import LLMClientFactory

factory = LLMClientFactory()

# Create with specific config
client = factory.create(
    provider="openai",
    model="gpt-4",
    timeout=60.0
)

# Create from environment settings
client = factory.create_from_settings()
```

### MCTSEngineFactory

```python
from src.framework.factories import MCTSEngineFactory

factory = MCTSEngineFactory()

engine = factory.create(
    seed=42,
    exploration_weight=1.414,
    config_preset="balanced"  # fast, balanced, or thorough
)
```

### FrameworkFactory

```python
from src.framework.factories import FrameworkFactory

factory = FrameworkFactory()

framework = factory.create_framework(
    llm_provider="openai",
    mcts_enabled=True,
    mcts_seed=42
)

# Or use convenience function
from src.framework.factories import create_framework

framework = create_framework(llm_provider="anthropic")
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
pytest -m contract          # Contract tests only
pytest -m property          # Property-based tests only
pytest -m performance       # Performance tests only
pytest -m "contract or property"  # Multiple categories
```

### Run with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run with verbose output
```bash
pytest -v --tb=short
```

### Run specific test file
```bash
pytest tests/test_contracts.py -v
```

### Run specific test
```bash
pytest tests/test_contracts.py::TestLLMClientContract::test_openai_client_implements_protocol -v
```

## Test Markers

All available pytest markers:

- `unit`: Unit tests
- `integration`: Integration tests
- `e2e`: End-to-end tests
- `component`: Component-level tests
- `contract`: Contract/interface tests
- `property`: Property-based tests
- `performance`: Performance benchmarks
- `slow`: Slow-running tests
- `api`: API endpoint tests
- `training`: Training pipeline tests
- `dataset`: Dataset integration tests
- `security`: Security validation tests
- `chaos`: Chaos engineering tests

View all markers:
```bash
pytest --markers
```

## Best Practices

### 1. Use Builders for Test Data

❌ **Avoid:**
```python
context = AgentContext(
    query="test",
    session_id="123",
    rag_context=None,
    metadata={},
    conversation_history=[],
    max_iterations=5,
    temperature=0.7
)
```

✅ **Prefer:**
```python
context = AgentContextBuilder().with_query("test").build()
```

### 2. Use Property Tests for Invariants

❌ **Avoid:**
```python
def test_token_count():
    response = create_response(10, 20)
    assert response.total_tokens == 30
```

✅ **Prefer:**
```python
@given(
    prompt=st.integers(min_value=0, max_value=10000),
    completion=st.integers(min_value=0, max_value=10000)
)
def test_token_count_invariant(prompt, completion):
    # Tests hundreds of combinations automatically
```

### 3. Use Contract Tests for Interfaces

❌ **Avoid:**
```python
def test_client_has_generate():
    client = OpenAIClient(...)
    client.generate("test")  # Might raise if interface changes
```

✅ **Prefer:**
```python
@pytest.mark.contract
def test_client_implements_protocol():
    assert hasattr(client, 'generate')
    assert callable(client.generate)
```

### 4. Use Performance Tests for Critical Paths

```python
@pytest.mark.performance
def test_critical_operation_speed():
    start = time.perf_counter()
    for _ in range(1000):
        critical_operation()
    elapsed = time.perf_counter() - start
    
    assert elapsed / 1000 < target_ms
```

## Continuous Integration

The CI pipeline (`.github/workflows/ci.yml`) automatically runs:

1. **Linting** (ruff)
2. **Type checking** (mypy)
3. **Security scanning** (bandit, pip-audit)
4. **Tests** (pytest with coverage)
5. **Docker build** (with health checks)

All tests must pass before merging to main.

## Code Coverage

Current coverage targets:
- **Minimum**: branch coverage at or above `fail_under` in `pyproject.toml`
  `[tool.coverage.report]` — **85%**, enforced by CI and by `make test`. Read the value
  there rather than trusting this line; `CHARTER.md` NG-5 forbids lowering it.
- **Scope**: the gate measures `tests/unit/` only. The e2e suite runs outside the `--cov`
  invocation on purpose, so end-to-end tests can neither dilute nor inflate the unit
  denominator (`docs/plans/EVIDENCE_FIRST_PROGRAM.md` R3).
- **Measured**: see `docs/STATUS.md`, which is regenerated from a real run.

View coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Adding New Tests

### 1. Add a Contract Test

```python
# tests/test_contracts.py

@pytest.mark.contract
def test_my_component_contract(self):
    """Contract: Component should implement required interface."""
    from src.my_module import MyComponent
    
    component = MyComponent()
    
    # Check required methods
    assert hasattr(component, 'required_method')
    assert callable(component.required_method)
```

### 2. Add a Property Test

```python
# tests/test_properties.py

from hypothesis import given, strategies as st

@pytest.mark.property
@given(value=st.floats(min_value=0.0, max_value=1.0))
def test_my_invariant(self, value):
    """Property: Some invariant should always hold."""
    result = my_function(value)
    assert 0.0 <= result <= 1.0
```

### 3. Add a Performance Test

```python
# tests/test_performance.py

@pytest.mark.performance
def test_my_operation_speed(self):
    """Test: Operation should be fast enough."""
    import time
    
    start = time.perf_counter()
    iterations = 1000
    
    for _ in range(iterations):
        my_operation()
    
    elapsed = time.perf_counter() - start
    per_op_ms = (elapsed / iterations) * 1000
    
    assert per_op_ms < 1.0  # < 1ms per operation
```

## References

- **pytest**: https://docs.pytest.org/
- **hypothesis**: https://hypothesis.readthedocs.io/
- **pytest markers**: https://docs.pytest.org/en/stable/how-to/mark.html
- **Pydantic**: https://docs.pydantic.dev/
