# Phase 4: Agent Builder Benchmark - LangGraph MCTS vs Google ADK

> **Template Version:** 2.0 | **Phase:** 4 of N | **Status:** Implementation Ready
> **Design Principle:** Constraint programming over instruction writing. Define the feasible region,
> objective function, and search parameters -- then let the agent solve.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building a production-grade benchmark framework that systematically compares
the LangGraph Multi-Agent MCTS system against Google's ADK Agent Builder across
equivalent multi-agent coordination tasks in software quality engineering,
regulatory compliance, and strategic decision-making domains.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All unit tests pass: `pytest tests/unit/benchmark -v`
- [ ] All integration tests pass: `pytest tests/integration/benchmark -v`
- [ ] Type checking passes: `mypy src/benchmark/`
- [ ] Linting passes: `ruff check src/benchmark/ tests/unit/benchmark/`
- [ ] Code coverage >= 80% on benchmark core logic
- [ ] No hardcoded values -- all config via Pydantic Settings
- [ ] Both LangGraph MCTS and ADK adapters implement identical BenchmarkSystemProtocol
- [ ] Evaluation harness produces structured BenchmarkResult for every task
- [ ] LLM-as-judge scoring returns valid scores (1-5) for all quality dimensions
- [ ] Comparison report generates valid markdown with summary table
- [ ] All sensitive data (API keys, tokens) sanitized in logs and reports
- [ ] Factory pattern used for all component creation
- [ ] Benchmark tasks are data-driven (loaded from config, not inline)
```

### 1.3 Problem Description (The "Three Paragraphs")

```
Paragraph 1 -- Core Coordination Logic:
The benchmark framework orchestrates parallel evaluation of two multi-agent systems
(LangGraph MCTS and Google ADK) across three task domains. Each system receives
identical input tasks and produces structured outputs that are scored by an
LLM-as-judge evaluator. The coordinator manages task routing, adapter selection,
timing instrumentation, and result aggregation while maintaining strict isolation
between system runs to prevent cross-contamination.

Paragraph 2 -- Data Flows and State:
Tasks flow from a registry (BenchmarkTaskRegistry) through system-specific adapters
(LangGraphAdapter, ADKAdapter) that normalize I/O to a common BenchmarkResult schema.
State includes: pending tasks, in-flight executions, timing measurements, token counts,
agent traces, and scored results. The evaluation harness maintains a session-scoped
results store that accumulates across task sets (A: QE, B: Compliance, C: Strategic).
All timing, cost, and quality metrics feed into a ReportGenerator that produces
comparative analysis.

Paragraph 3 -- Failure Modes and Invariants:
Failure modes: adapter timeout (configurable per-system), LLM judge scoring failure
(retry with backoff), API rate limits (token bucket), missing API keys (graceful skip
with warning). Invariants: every task produces exactly one BenchmarkResult per system;
scoring dimensions are always in [1, 5]; cost estimates are non-negative; the total
number of results equals (num_tasks * num_systems); no benchmark run modifies the
systems under test.
```

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.10+ (match existing pyproject.toml)
- Required Dependencies: pydantic>=2.0, pydantic-settings, langgraph, httpx
- Optional Dependencies: google-adk (for ADK adapter), google-generativeai (for scoring)
- Security: No hardcoded API keys or secrets; all via SecretStr/env vars
- Compatibility: Must integrate with existing src/ architecture (factories, protocols, settings)
- Configuration: All thresholds, weights, timeouts via Pydantic Settings
- Testing: pytest with async support, markers for unit/integration/benchmark
- Backwards Compatibility: No breaking changes to existing src/ modules
```

### 2.2 Soft Constraints (Preferences)

```
- Style: Black formatting (120 char), isort (black profile), ruff linting
- Architecture: Protocol-based interfaces, factory pattern, dependency injection
- Performance: Async I/O for all external calls, concurrent task execution where safe
- Testing: >80% coverage on core logic, property-based tests for scoring bounds
- Logging: Structured JSON with correlation IDs, sensitive data sanitization
- Cost Tracking: Per-task token usage and estimated USD cost
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Add new Pydantic Settings classes for benchmark configuration
- Create new factory classes following existing patterns
- Add pytest markers (benchmark, adk, scoring)
- Add optional dependencies to pyproject.toml [benchmark] extra
- Choose LLM-as-judge prompting strategy
- Design task schema and scoring rubrics
- Create adapter interfaces for future system integrations
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
- src/benchmark/**          (new module)
- tests/unit/benchmark/**   (new tests)
- tests/integration/benchmark/** (new tests)
- pyproject.toml            (add [benchmark] optional dependency group)
- .env.example              (add benchmark env vars)
- CLAUDE.md                 (update with benchmark commands)

OUT OF SCOPE:
- src/framework/            (read-only reference)
- src/agents/               (read-only reference)
- src/adapters/llm/         (read-only reference)
- src/enterprise/           (read-only reference, pattern reference)
- src/observability/        (import only, no modifications)
- Production deployment configs
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation within src/benchmark/ and tests/
- Factory class creation following existing patterns
- Test creation and execution
- Configuration class creation
- Adapter implementation

CONFIRM FIRST (ask before proceeding):
- Modifications to existing src/ modules
- Changes to pyproject.toml dependencies
- Breaking API changes to existing protocols

PROHIBITED (do not attempt):
- Commits to main branch
- External API calls with side effects during tests
- Modifications to src/framework/ or src/agents/
- Hardcoded API keys or secrets
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5
- Max files to modify in single pass: 20
- Benchmark task definitions: 3 task sets (A, B, C) with 3-4 tasks each
- LLM judge calls: Configurable, default 1 per result
- Retry budget: 3 retries per failed scoring call
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:
1. black src/benchmark/ tests/unit/benchmark/ tests/integration/benchmark/ --check --line-length 120
2. isort src/benchmark/ tests/unit/benchmark/ tests/integration/benchmark/ --check --profile black
3. ruff check src/benchmark/ tests/unit/benchmark/ tests/integration/benchmark/
4. mypy src/benchmark/ --strict
5. pytest tests/unit/benchmark/ -v --tb=short
6. pytest tests/integration/benchmark/ -v --tb=short -m "not slow"
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
-> Fix automatically with black/isort/ruff --fix, re-run

ON TYPE ERROR:
-> Analyze error, fix types (prefer Protocol over ABC), re-run

ON TEST FAILURE:
-> Read failure output
-> Identify root cause (implementation bug vs test bug)
-> Fix implementation (not test, unless test is wrong)
-> Re-run

ON REPEATED FAILURE (same error 3x):
-> Stop and report analysis
-> Document in KNOWN_ISSUES section
-> Request human guidance
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass
2. Coverage report: pytest tests/unit/benchmark/ --cov=src/benchmark --cov-report=term-missing
3. No hardcoded values: grep -r "api_key.*=.*['\"]sk-" src/benchmark/ && echo "FAIL" || echo "OK"
4. All protocols implemented: verify BenchmarkSystemProtocol compliance
5. Generate sample benchmark report (dry run with mocks)
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Architecture Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01 | Protocol-based BenchmarkSystemProtocol | Provider agnosticism (LangGraph, ADK, future systems) |
| 2025-01 | Pydantic Settings for BenchmarkConfig | Consistent with existing config pattern, env var support |
| 2025-01 | Factory pattern for adapter creation | Testability, dependency injection, loose coupling |
| 2025-01 | LLM-as-judge with configurable provider | Avoid manual scoring bias, support multiple judge models |
| 2025-01 | Data-driven task definitions | Extensibility, no code changes for new tasks |
| 2025-01 | Structured BenchmarkResult dataclass | Type safety, serialization, aggregation support |

### 5.2 Information to Preserve Across Sessions

```
- Benchmark task schemas and expected outputs
- Scoring rubric definitions
- Adapter interface contracts
- Cost estimation formulas per provider
- Known ADK API limitations and workarounds
```

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Implementation Order

```
Phase 4A: Core Framework (Week 1)
  1. Benchmark configuration (Pydantic Settings)
  2. Data models (BenchmarkTask, BenchmarkResult, ScoringResult)
  3. Protocols (BenchmarkSystemProtocol, ScorerProtocol)
  4. Task registry (data-driven task definitions)
  5. Unit tests for data models and config

Phase 4B: System Adapters (Week 2)
  6. LangGraph MCTS adapter (wraps existing framework)
  7. ADK adapter (wraps Google ADK agents)
  8. Adapter factory
  9. Unit tests for adapters (mocked external calls)

Phase 4C: Evaluation Engine (Week 3)
  10. LLM-as-judge scorer
  11. Evaluation harness (orchestrates benchmark runs)
  12. Cost calculator
  13. Integration tests for evaluation pipeline

Phase 4D: Reporting & Analysis (Week 4)
  14. Report generator (markdown comparison tables)
  15. Metrics aggregator
  16. End-to-end integration tests
  17. Documentation updates (CLAUDE.md, .env.example)
```

### 6.2 Completion Checklist

```
[ ] All success criteria from Section 1.2 met
[ ] All verification commands from Section 4.1 pass
[ ] CLAUDE.md updated with benchmark commands
[ ] .env.example updated with benchmark env vars
[ ] No redundant or unused code
[ ] Logging with correlation IDs in all async paths
[ ] Sensitive data sanitized in all outputs
[ ] Factory pattern for all component creation
[ ] Full test suite with unit + integration coverage
[ ] Phase 4 plan document committed
```

---

## SECTION 7: COMPONENT SPECIFICATIONS

### 7.1 Module Structure

```
src/benchmark/
  __init__.py                    # Public API exports
  config/
    __init__.py
    benchmark_settings.py        # Pydantic Settings for all benchmark config
  tasks/
    __init__.py
    models.py                    # BenchmarkTask, TaskCategory, TaskComplexity
    registry.py                  # BenchmarkTaskRegistry (data-driven)
    task_sets.py                 # Default task set definitions
  adapters/
    __init__.py
    protocol.py                  # BenchmarkSystemProtocol
    langgraph_adapter.py         # LangGraph MCTS adapter
    adk_adapter.py               # Google ADK adapter
    factory.py                   # BenchmarkAdapterFactory
  evaluation/
    __init__.py
    models.py                    # BenchmarkResult, ScoringResult
    scorer.py                    # LLM-as-judge scorer
    harness.py                   # Evaluation harness (orchestrator)
    cost_calculator.py           # Token/cost estimation
  reporting/
    __init__.py
    report_generator.py          # Markdown report generation
    metrics_aggregator.py        # Statistical aggregation
```

### 7.2 Key Protocols

```python
# BenchmarkSystemProtocol - Adapter contract
class BenchmarkSystemProtocol(Protocol):
    @property
    def name(self) -> str: ...
    async def execute(self, task: BenchmarkTask) -> BenchmarkResult: ...
    async def health_check(self) -> bool: ...

# ScorerProtocol - Scoring contract
class ScorerProtocol(Protocol):
    async def score(self, result: BenchmarkResult, task: BenchmarkTask) -> ScoringResult: ...
```

### 7.3 Configuration Hierarchy

```
BenchmarkSettings (root)
  +-- BenchmarkRunConfig        (run-level: iterations, timeout, parallelism)
  +-- ScoringConfig             (judge model, retry, temperature)
  +-- CostConfig                (per-provider token pricing)
  +-- LangGraphBenchmarkConfig  (LangGraph-specific settings)
  +-- ADKBenchmarkConfig        (ADK-specific settings)
  +-- ReportConfig              (output format, paths)
```

---

## SECTION 8: TASK SPECIFICATIONS

### Task Set A: Software Quality Engineering

| ID | Description | Complexity | Expected Elements |
|----|-------------|-----------|-------------------|
| A1 | Code Review: Identify bugs in PR diff | Medium | division-by-zero, leaf nodes, UCB1 correctness, thread safety |
| A2 | Security Vulnerability Analysis | High | OWASP top-10 mapping, severity scoring, remediation priority |
| A3 | Test Plan Generation from Requirements | High | unit/integration/e2e/performance/security test types |
| A4 | Architecture Decision Record Review | Very High | risk identification, alternative evaluation, tradeoff analysis |

### Task Set B: Regulatory Compliance

| ID | Description | Complexity | Expected Elements |
|----|-------------|-----------|-------------------|
| B1 | Compliance Requirement Extraction | Medium | structured requirements, regulation references |
| B2 | Control Gap Analysis | High | control mapping, gap identification, coverage metrics |
| B3 | Remediation Plan Generation | Very High | prioritized actions, timeline, budget allocation, risk mitigation |

### Task Set C: Strategic Decision Making

| ID | Description | Complexity | Expected Elements |
|----|-------------|-----------|-------------------|
| C1 | Investment Strategy Evaluation | High | multiple strategies, risk/return analysis, recommendations |
| C2 | Resource-Constrained Project Planning | Very High | strategic options, resource allocation, timeline, competitive differentiation |
| C3 | Competitive Analysis with Scenarios | Very High | scenario modeling, market positioning, contingency planning |

---

## SECTION 9: SCORING RUBRIC

### Dimensions (1-5 scale)

| Dimension | 1 (Poor) | 3 (Adequate) | 5 (Excellent) |
|-----------|----------|--------------|---------------|
| Task Completion | Misses most aspects | Addresses core aspects | Comprehensive coverage |
| Reasoning Depth | Surface-level | Multi-step reasoning | Deep, layered analysis |
| Accuracy | Major errors | Minor inaccuracies | Factually correct |
| Coherence | Disorganized | Logically structured | Seamlessly integrated |

### Additional Metrics

| Metric | Unit | Collection Method |
|--------|------|-------------------|
| Total Latency | milliseconds | Wall-clock timing |
| Time to First Token | milliseconds | Streaming callback |
| Agent Call Count | integer | Adapter instrumentation |
| Tool Call Count | integer | Adapter instrumentation |
| Input Tokens | integer | API response metadata |
| Output Tokens | integer | API response metadata |
| Estimated Cost | USD | Token count * rate |

---

## SECTION 10: COST ESTIMATES

| Component | Estimated Cost | Notes |
|-----------|---------------|-------|
| Gemini 2.5 Pro (coordinator, ~50 runs) | ~$15-30 | ADK coordinator agent |
| Gemini 2.5 Flash (sub-agents, ~200 runs) | ~$5-10 | ADK sub-agents |
| Gemini 2.0 Flash (scoring judge, ~100 runs) | ~$2-5 | LLM-as-judge |
| Agent Engine runtime (if deployed) | ~$10-20 | Cloud hosting |
| OpenAI GPT-4 (LangGraph runs, ~50) | ~$20-40 | LangGraph LLM backend |
| **Subtotal** | **~$52-105** | |

---

## SECTION 11: NEXT STEPS (Post-Phase 4)

1. **A2A Protocol Integration** -- Expose LangGraph agents via A2A for cross-framework comms
2. **Gemini Backend for MCTS** -- Test Gemini 2.5 Pro as LLM backend in MCTS reasoning chains
3. **Vertex AI Search Integration** -- Per-customer data stores for grounded responses
4. **Production Deployment** -- Agent Engine deployment with monitoring
5. **Customer Demo Pipeline** -- End-to-end demo flow with benchmark results
