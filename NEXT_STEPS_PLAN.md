# LangGraph Multi-Agent MCTS: Next Steps Plan

> ⚠️ **SUPERSEDED (2026-06-30).** This v1.0 plan is retained for history but several of its
> "critical" items are already resolved in the source tree (the iteration-counter bug, the
> DABStep `split` parameter, REST `/query` wiring, RAG wiring, and the LLM-adapter/hybrid-agent/
> monitoring/observability test gaps). For the current, code-verified plan see
> **`docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md`**, whose §1.1 Verified-Status table governs
> wherever it disagrees with this document.

> **Version:** 1.0.0
> **Date:** 2026-02-01
> **Status:** Superseded by docs/NEXT_STEPS_IMPLEMENTATION_PLAN_2026H2.md

---

## Executive Summary

This document outlines the logical next steps for the LangGraph Multi-Agent MCTS framework based on a comprehensive codebase analysis. The project is **~90% feature complete** with **88.4% test pass rate** (771/872 tests). The plan prioritizes critical bugs, test coverage gaps, and feature enhancements.

> **Status update (CI hardening):** the CI pipeline is now **green and deterministic** —
> lint/type/test/build/docker jobs pass, tool versions (`ruff`/`mypy`) are pinned for
> CI/local parity, and the unit suite holds **~88.85% coverage** against an 85% gate.
> Phase 2.1 (LLM adapter tests) has progressed: the shared `CircuitBreaker`
> (`src/adapters/llm/resilience.py`) is now covered by dedicated regression tests
> (`tests/unit/adapters/test_resilience.py`), including enforcement of `half_open_max_calls`.

---

## Phase 1: Critical Bug Fixes (Priority: 🔴 CRITICAL)

### 1.1 Fix Iteration Counter Bug in LangGraph Orchestration

**Problem:** The iteration counter in `src/framework/graph.py` is never incremented when looping back to `route_decision`, potentially causing infinite loops.

**Location:** `src/framework/graph.py:998-1001`

```python
# Current (broken):
def _check_consensus(self, state: AgentState) -> str:
    if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
        return "synthesize"
    return "iterate"  # BUG: iteration counter never incremented!
```

**Acceptance Criteria:**
- [ ] Add `iteration += 1` in `evaluate_consensus` or `aggregate_results` node
- [ ] Add unit test verifying iteration increments correctly
- [ ] Add integration test verifying max_iterations causes loop termination
- [ ] Verify no infinite loops with `max_iterations=3` test case
- [ ] Update any logging to track iteration progress

---

### 1.2 Fix MCTS Policy Interface Signature Mismatch

**Problem:** 24 test failures due to `RolloutPolicy.evaluate()` signature mismatch between interface and implementations.

**Location:** `src/framework/mcts/policies.py`

**Acceptance Criteria:**
- [ ] Audit all `RolloutPolicy` implementations for consistent signatures
- [ ] Standardize `evaluate(state, action) -> float` or `evaluate(state) -> float`
- [ ] Update all implementations to match the standardized interface
- [ ] All 24 currently failing policy tests pass
- [ ] Add type hints and protocol definition for clarity

---

## Phase 2: Test Coverage Improvements (Priority: 🟠 HIGH)

### 2.1 Add LLM Adapter Unit Tests

**Problem:** 0% dedicated test coverage for LLM adapters (OpenAI, Anthropic, LMStudio).

**Files Needing Tests:**
- `src/adapters/llm/openai_client.py`
- `src/adapters/llm/anthropic_client.py`
- `src/adapters/llm/lmstudio_client.py`

**Acceptance Criteria:**
- [ ] Create `tests/unit/adapters/test_openai_client.py` with:
  - [ ] Mock HTTP responses for chat completions
  - [ ] Test retry logic with exponential backoff
  - [ ] Test streaming response handling
  - [ ] Test token counting
  - [ ] Test error handling (rate limits, auth failures)
- [ ] Create `tests/unit/adapters/test_anthropic_client.py` with equivalent coverage
- [ ] Create `tests/unit/adapters/test_lmstudio_client.py` with equivalent coverage
- [ ] Achieve >80% line coverage for adapter module
- [ ] All tests pass with mocked dependencies (no real API calls)

---

### 2.2 Add Hybrid Agent Unit Tests

**Problem:** No comprehensive unit tests for `HybridAgent` despite being a critical component.

**File:** `src/agents/hybrid_agent.py`

**Acceptance Criteria:**
- [ ] Create `tests/unit/test_hybrid_agent.py` with:
  - [ ] Test `auto` routing mode (neural first, LLM fallback)
  - [ ] Test `neural_only` mode
  - [ ] Test `llm_only` mode
  - [ ] Test `adaptive` mode with confidence threshold
  - [ ] Test cost tracking calculations
  - [ ] Test fallback behavior when components missing
  - [ ] Test blending of neural + LLM estimates
- [ ] Test all four routing modes return valid outputs
- [ ] Verify cost savings calculations are mathematically correct
- [ ] Achieve >85% line coverage

---

### 2.3 Add Google ADK Integration Tests

**Problem:** 0% test coverage for `src/integrations/google_adk/` (13 source files).

**Acceptance Criteria:**
- [ ] Create `tests/integration/test_google_adk.py` with:
  - [ ] Mock ADK agent initialization
  - [ ] Test agent routing from meta-controller
  - [ ] Test error handling for ADK failures
  - [ ] Test timeout handling
- [ ] Add skip markers for environments without ADK dependencies
- [ ] Document ADK test requirements in `tests/README.md`

---

### 2.4 Add Monitoring/Observability Tests

**Problem:** 0-11% coverage for monitoring and observability modules.

**Files Needing Tests:**
- `src/monitoring/otel_tracing.py`
- `src/monitoring/prometheus_metrics.py`
- `src/observability/braintrust_tracker.py`

**Acceptance Criteria:**
- [ ] Create `tests/unit/test_otel_tracing.py`:
  - [ ] Test span creation and context propagation
  - [ ] Test trace ID extraction
  - [ ] Test OTLP exporter configuration
- [ ] Create `tests/unit/test_prometheus_metrics.py`:
  - [ ] Test counter increments
  - [ ] Test histogram observations
  - [ ] Test gauge updates
  - [ ] Test metric endpoint output
- [ ] Achieve >70% coverage for observability modules

---

## Phase 3: Feature Enhancements (Priority: 🟡 MEDIUM)

### 3.1 Implement LangGraph Streaming Support

**Problem:** No streaming implementation despite being in design docs and CLAUDE.md.

**Location:** `src/framework/graph.py`

**Acceptance Criteria:**
- [ ] Add `astream_events()` method to `IntegratedFramework`
- [ ] Implement token-level streaming for LLM responses
- [ ] Implement node-level state snapshots during execution
- [ ] Add streaming configuration options in `GraphConfig`
- [ ] Create `tests/integration/test_graph_streaming.py`:
  - [ ] Test token-by-token output
  - [ ] Test node completion events
  - [ ] Test error handling during streams
- [ ] Update API endpoints to support streaming responses
- [ ] Document streaming usage in CLAUDE.md

---

### 3.2 Add LangGraph Visualization

**Problem:** MCTS tree visualization exists, but LangGraph state graph visualization is missing.

**Acceptance Criteria:**
- [ ] Add `visualize_graph()` method returning Mermaid diagram
- [ ] Add `/graph/visualize` API endpoint
- [ ] Generate ASCII representation for CLI users
- [ ] Export to PNG/SVG for documentation
- [ ] Add node metadata (execution count, avg duration)
- [ ] Create unit tests for visualization output

---

### 3.3 Implement MCTS Early Termination

**Problem:** `config.py` has early termination settings but `search()` runs all iterations.

**Location:** `src/framework/mcts/core.py`

**Acceptance Criteria:**
- [ ] Add convergence threshold check in main search loop
- [ ] Terminate when best action has stable value (< threshold change over N iterations)
- [ ] Add `early_stop_threshold` and `early_stop_patience` config
- [ ] Log early termination events
- [ ] Add unit tests verifying early stopping behavior
- [ ] Benchmark performance improvement

---

### 3.4 Complete RAG Pipeline Integration

**Problem:** RAG retrieval works, but full pipeline integration is incomplete.

**Acceptance Criteria:**
- [ ] Wire RAG context into agent state flow
- [ ] Add relevance scoring to route decisions
- [ ] Implement context windowing for long documents
- [ ] Add caching for repeated queries
- [ ] Create integration tests for RAG → Agent → Response flow
- [ ] Document RAG configuration in CLAUDE.md

---

## Phase 4: Training & Learning (Priority: 🟢 STANDARD)

### 4.1 Complete Meta-Controller Training Loop

**Problem:** Training framework exists but fine-tuning loop needs work.

**Acceptance Criteria:**
- [ ] Implement data collection from live routing decisions
- [ ] Create training script for meta-controller fine-tuning
- [ ] Add validation loop with held-out test set
- [ ] Implement automatic weight tuning for hybrid controller
- [ ] Add training metrics (accuracy, loss curves)
- [ ] Document training procedure in `docs/META_CONTROLLER_TRAINING.md`
- [ ] Create `tests/training/test_meta_controller_training.py`

---

### 4.2 Add Assembly Router Learning

**Problem:** Assembly router rules are manually defined, not learned from data.

**Acceptance Criteria:**
- [ ] Collect routing decision outcomes (success/failure)
- [ ] Implement threshold optimization based on outcomes
- [ ] Add A/B testing framework for routing rules
- [ ] Track and log routing rule performance
- [ ] Create automated rule update pipeline

---

### 4.3 Generalize Self-Play Training Loop

**Problem:** Self-play is fully implemented for chess but not generalized.

**Acceptance Criteria:**
- [ ] Abstract chess-specific components into generic interface
- [ ] Create `SelfPlayTrainer` base class
- [ ] Implement for at least one additional domain (reasoning, planning)
- [ ] Add curriculum learning hooks
- [ ] Document generalization pattern in template

---

## Phase 5: Production Readiness (Priority: 🔵 ENHANCEMENT)

### 5.1 Performance Optimization

**Acceptance Criteria:**
- [ ] Benchmark MCTS iterations/second (target: >1000/sec)
- [ ] Add batched inference for meta-controller
- [ ] Implement model quantization (INT8) for BERT controllers
- [ ] Add connection pooling for LLM clients
- [ ] Profile memory usage and reduce allocations
- [ ] Add performance regression tests

---

### 5.2 Distributed MCTS

**Acceptance Criteria:**
- [ ] Design distributed tree search architecture
- [ ] Implement worker nodes for parallel simulation
- [ ] Add coordinator for result aggregation
- [ ] Handle worker failures gracefully
- [ ] Create Kubernetes deployment manifests
- [ ] Benchmark scalability (2, 4, 8 workers)

---

### 5.3 Explainability Module

**Acceptance Criteria:**
- [ ] Add decision tree extraction from MCTS paths
- [ ] Implement attention visualization for meta-controller
- [ ] Generate human-readable reasoning traces
- [ ] Add `/explain` API endpoint
- [ ] Create UI component for visualization
- [ ] Document explainability features

---

## Implementation Timeline

| Phase | Focus | Duration | Dependencies |
|-------|-------|----------|--------------|
| **1** | Critical Bug Fixes | Week 1 | None |
| **2** | Test Coverage | Weeks 2-3 | Phase 1 complete |
| **3** | Feature Enhancements | Weeks 4-6 | Core tests passing |
| **4** | Training & Learning | Weeks 7-9 | Stable codebase |
| **5** | Production Readiness | Weeks 10-12 | All features complete |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test Pass Rate | 88.4% | >95% |
| Source Files with Tests | 60% | >85% |
| Critical Bugs | 2 | 0 |
| Documentation Coverage | 80% | >95% |
| MCTS Iterations/sec | Unknown | >1000 |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Iteration bug causes production issues | High | Critical | Fix immediately (Phase 1.1) |
| LLM adapter untested failures | Medium | High | Add comprehensive mocks (Phase 2.1) |
| Streaming complexity delays | Medium | Medium | Start with basic token streaming |
| Distributed MCTS coordination overhead | Medium | Medium | Benchmark early, iterate design |

---

## Quick Reference: File Locations

**Critical Fixes:**
- `src/framework/graph.py:998-1001` - Iteration bug
- `src/framework/mcts/policies.py` - Policy interface

**Test Gaps:**
- `src/adapters/llm/` - LLM clients
- `src/agents/hybrid_agent.py` - Hybrid agent
- `src/integrations/google_adk/` - ADK integration
- `src/monitoring/` - Observability

**Feature Additions:**
- `src/framework/graph.py` - Streaming, visualization
- `src/framework/mcts/core.py` - Early termination
- `src/api/rag_retriever.py` - RAG integration

---

*Document generated from comprehensive codebase analysis.*
