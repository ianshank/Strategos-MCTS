# MVP Gaps Analysis & Roadmap

> Analysis of gaps between current codebase state and a demo-able MVP,
> with short/medium/long-term implementation plans.

---

## Executive Summary

The codebase contains **217 Python source files** with sophisticated architecture,
but the path from code to live demo has critical gaps. The core issue: **the MCTS
engine works mechanically but doesn't drive actual LLM reasoning**, and the existing
demo (`app.py`) requires heavy dependencies (PyTorch, Gradio, PEFT) that are fragile
to install.

**What was built to close these gaps:**

| Deliverable | File | Status |
|-------------|------|--------|
| LLM-powered MCTS engine | `src/framework/mcts/llm_mcts.py` | Done |
| CLI demo (zero deps) | `demo.py` | Done |
| Tests | `tests/unit/test_llm_mcts.py` | Done |
| This roadmap | `MVP_ROADMAP.md` | Done |

---

## Gap Analysis

### Critical Gaps (Demo Blockers)

| # | Gap | Impact | Resolution |
|---|-----|--------|------------|
| 1 | **No lightweight demo entry point** | Cannot demo without PyTorch/Gradio install | Created `demo.py` - works with stdlib only |
| 2 | **MCTS doesn't drive LLM reasoning** | Core value prop not demonstrated | Created `llm_mcts.py` - MCTS explores reasoning strategies via real LLM calls |
| 3 | **Agent handlers are simulated** | `app.py` agents use `asyncio.sleep()` + template strings | `llm_mcts.py` makes real LLM calls |
| 4 | **Dependency chain is fragile** | `pip install -e ".[dev]"` fails in many environments | Demo uses zero external deps |
| 5 | **No mock mode for offline demos** | Can't demo without API keys | `MockLLMClient` provides realistic mock responses |

### Structural Gaps (Reduce Demo Impact)

| # | Gap | Impact |
|---|-----|--------|
| 6 | No visualization of MCTS tree exploration | Hard to explain what MCTS is doing |
| 7 | No A/B comparison (MCTS vs single-shot) | Can't prove MCTS improves quality |
| 8 | LangGraph graph nodes are wired but don't call real LLMs | The orchestration framework is mechanical |
| 9 | Benchmark framework can't run without full dep chain | No quantitative quality metrics |
| 10 | No streaming output during MCTS exploration | Demo feels slow during LLM calls |

### Technical Debt

| # | Gap | Impact |
|---|-----|--------|
| 11 | `core.py` MCTS depends on numpy | Can't run in minimal environments |
| 12 | Settings require pydantic-settings | Config fails without it |
| 13 | 42 test collection errors from missing deps | Test suite appears broken |
| 14 | No CI/CD pipeline | No automated quality gates |

---

## Short-Term Plan (Implemented)

### What was built

#### 1. `src/framework/mcts/llm_mcts.py` - LLM-Powered MCTS Engine

**Purpose:** Bridge the gap between the mechanical MCTS algorithm and actual
LLM-powered reasoning.

**Architecture:**
```
Query
  |
  v
MCTSTreeNode (root)
  |-- direct          (straightforward answer)
  |-- decomposition   (break into sub-problems, HRM-style)
  |-- refinement      (iterative improvement, TRM-style)
  |-- analogy         (reason by analogy)
  |-- adversarial     (argue against own answer)

For each iteration:
  1. SELECT: UCB1 picks the most promising strategy
  2. SIMULATE: LLM call generates a response using that strategy
  3. SCORE: Response quality evaluated (LLM-as-judge or heuristic)
  4. BACKPROPAGATE: Score updates the tree

After all iterations:
  - Best strategy identified (most visited = most robust)
  - Consensus synthesis combines top strategies
```

**Key classes:**
- `StdlibLLMClient` - HTTP client using only `urllib` (supports OpenAI + Anthropic)
- `MockLLMClient` - Realistic mock for offline demos
- `LLMMCTSEngine` - The core search engine
- `ResponseScorer` - LLM-as-judge or heuristic scoring
- `ConsensusBuilder` - Synthesizes top strategies into final answer
- `MultiAgentMCTSPipeline` - End-to-end pipeline

**Zero external dependencies** - works with Python 3.10+ standard library only.

#### 2. `demo.py` - CLI Demo

**Usage:**
```bash
# Mock mode (no API key needed)
python demo.py

# With real LLM calls
OPENAI_API_KEY=sk-... python demo.py --provider openai
ANTHROPIC_API_KEY=sk-... python demo.py --provider anthropic

# Custom query
python demo.py --query "How should I design a rate limiter?"

# Interactive REPL
python demo.py --interactive

# JSON output for programmatic use
python demo.py --json

# Adjust MCTS parameters
python demo.py --iterations 15 --exploration 2.0
```

**Output includes:**
- MCTS exploration progress with strategy scores
- UCB1-guided visit distribution (bar charts)
- Best strategy response
- Consensus synthesis from top strategies
- Performance statistics (time, tokens, calls)
- Explanation of why MCTS adds value

#### 3. `tests/unit/test_llm_mcts.py` - Test Suite

30+ tests covering:
- Tree node mechanics (UCB1, value averaging)
- Mock client behavior
- Response scoring (heuristic)
- MCTS engine search (determinism, exploration, strategy coverage)
- Consensus building
- Pipeline end-to-end
- Strategy prompt validation

---

## Medium-Term Plan (Next Steps)

### M1: Wire LangGraph Graph to Real LLM Calls

**Goal:** Make `src/framework/graph.py` nodes call real LLMs instead of simulating.

**Approach:**
1. Create `src/framework/agents/llm_hrm.py` - LLM-based HRM that decomposes queries
2. Create `src/framework/agents/llm_trm.py` - LLM-based TRM that refines responses
3. Update `graph.py` node implementations to use `StdlibLLMClient`
4. Keep backward compatibility with existing imports

**Files to modify:**
- `src/framework/graph.py` (node implementations)
- New: `src/framework/agents/llm_hrm.py`
- New: `src/framework/agents/llm_trm.py`

### M2: Add Comparison Mode to Demo

**Goal:** Show MCTS vs. single-shot prompting side by side.

**Approach:**
```bash
python demo.py --compare --query "your question"
```

Output:
```
--- Single-Shot (direct prompt) ---
[response A]
Score: 0.65

--- MCTS (10 iterations, 5 strategies) ---
[response B]
Score: 0.82

Improvement: +26%
```

### M3: MCTS Tree Visualization

**Goal:** ASCII tree visualization showing the exploration.

```
root (20 visits)
├── direct (6 visits, avg=0.72) ***
├── decomposition (5 visits, avg=0.68)
├── refinement (4 visits, avg=0.65)
├── analogy (3 visits, avg=0.61)
└── adversarial (2 visits, avg=0.55)
```

### M4: Streaming Output During Exploration

**Goal:** Show progress as MCTS iterates (not just final results).

```
[1/10] Trying: direct         -> score: 0.72
[2/10] Trying: decomposition  -> score: 0.68
[3/10] Trying: direct         -> score: 0.75  (UCB1 selected best)
...
```

### M5: Fix Dependency Chain

**Goal:** Make `pip install -e ".[dev]"` work reliably.

**Approach:**
1. Pin dependency versions more precisely in `pyproject.toml`
2. Add a `requirements-minimal.txt` for demo-only deps
3. Add a `Makefile` or `justfile` with setup commands
4. Test installation in clean Python 3.10, 3.11, 3.12 environments

---

## Long-Term Plan (Production Readiness)

### L1: Docker-Based Demo

```dockerfile
FROM python:3.12-slim
COPY . /app
RUN pip install -e ".[dev]"
CMD ["python", "demo.py", "--interactive"]
```

### L2: Web UI (Lightweight)

Replace the heavy Gradio dependency with a simple HTML/JS interface served by
Python's built-in `http.server`. The LLM-MCTS engine already produces JSON output
(`demo.py --json`), so the UI just needs to consume that.

### L3: Benchmark Results

Run the benchmark framework against real queries and publish:
- MCTS vs. single-shot quality comparison
- Token efficiency (quality per token spent)
- Strategy effectiveness by query type
- Comparison with Google ADK (the existing benchmark adapter)

### L4: Neural Meta-Controller Integration

Connect the trained RNN/BERT meta-controllers to the LLM-MCTS pipeline:
- Meta-controller predicts which strategies to prioritize
- MCTS validates and refines the prediction
- Feedback loop improves the meta-controller over time

### L5: CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
- Run unit tests (no API key needed - mock mode)
- Run linting (ruff, black --check)
- Run type checking (mypy)
- Build Docker image
- Deploy demo to staging
```

---

## Architecture: How It All Connects

```
                    ┌─────────────────────────────────┐
                    │          User Query              │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────v────────────────────┐
                    │    MultiAgentMCTSPipeline       │
                    │    (demo.py entry point)         │
                    └────────────┬────────────────────┘
                                 │
            ┌────────────────────v──────────────────────┐
            │           LLMMCTSEngine                    │
            │  ┌─────────────────────────────────────┐  │
            │  │  UCB1 Selection → LLM Call → Score   │  │
            │  │  → Backpropagation → Repeat          │  │
            │  └─────────────────────────────────────┘  │
            │                                            │
            │  Strategies:                               │
            │  ├── direct (straightforward)              │
            │  ├── decomposition (HRM-style)             │
            │  ├── refinement (TRM-style)                │
            │  ├── analogy (creative)                    │
            │  └── adversarial (self-critique)           │
            └────────────────────┬──────────────────────┘
                                 │
                    ┌────────────v────────────────────┐
                    │     ConsensusBuilder             │
                    │  (synthesize top strategies)     │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────v────────────────────┐
                    │       Final Response             │
                    └─────────────────────────────────┘

LLM Backend:
  StdlibLLMClient (OpenAI/Anthropic via urllib)
  MockLLMClient (offline demos)

Existing Framework Integration:
  src/framework/graph.py      ← LangGraph orchestration (medium-term)
  src/framework/mcts/core.py  ← Original MCTS engine (numpy-based)
  src/adapters/llm/           ← Production LLM adapters (httpx-based)
```

---

## File Inventory: New Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/framework/mcts/llm_mcts.py` | ~520 | LLM-powered MCTS engine |
| `demo.py` | ~310 | CLI demo entry point |
| `tests/unit/test_llm_mcts.py` | ~250 | Test suite |
| `MVP_ROADMAP.md` | This file | Gaps analysis and roadmap |

---

## Demo Script (What to Show)

### 1. Zero-dependency demo (30 seconds)
```bash
python demo.py
```
Shows: MCTS exploring 5 strategies, UCB1 converging, consensus synthesis.
**Key talking point:** "MCTS doesn't just try once - it systematically explores
different reasoning approaches and converges on the best one."

### 2. Real LLM demo (2 minutes)
```bash
OPENAI_API_KEY=sk-... python demo.py --provider openai --query "Design a rate limiter"
```
Shows: Real LLM responses being explored and scored.
**Key talking point:** "Each MCTS iteration is a real LLM call with a different
reasoning strategy. The tree search focuses on the approaches that score highest."

### 3. Interactive mode (open-ended)
```bash
python demo.py --interactive
```
Let the audience ask questions and see MCTS explore strategies in real time.

---

*Last updated: 2026-02-23*
