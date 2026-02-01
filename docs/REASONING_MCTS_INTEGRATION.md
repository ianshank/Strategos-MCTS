# Reasoning-Enhanced MCTS Integration

This document describes the integration of modern AI reasoning techniques into the LangGraph multi-agent MCTS system, including Process Reward Models (PRMs), extended thinking, hybrid search strategies, and dual-agent architecture.

## Overview

Modern reasoning models (OpenAI o1/o3, DeepSeek-R1, Claude's extended thinking) demonstrate that **test-time compute scaling** can be more effective than parameter scaling for complex reasoning tasks. This integration brings these techniques to MCTS, creating a synergistic architecture where:

- **PRMs** provide step-level evaluation signals (replacing sparse terminal rewards)
- **Extended thinking** enables deeper analysis for critical decisions
- **Parallel sampling** ensures robustness against "overthinking"
- **Dual-agent architecture** separates strategic reasoning from action execution

## Architecture

```
                    User Query
                        │
                        ▼
        ┌───────────────────────────────┐
        │   LangGraph Reasoning Graph   │
        │   (Enhanced Orchestration)    │
        └───────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Reasoner  │  │    Actor    │  │   Hybrid    │
│   Agent     │  │   Agent     │  │   Search    │
│ (Extended   │  │ (Fast       │  │ (Parallel + │
│  Thinking)  │  │  Execution) │  │  Serial)    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────────┐    ┌───────────────────┐
│  Process Reward   │    │   Reasoning       │
│  Model (PRM)      │    │   MCTS Node       │
│  - Step scoring   │    │   - Traces        │
│  - Filtering      │    │   - PRM scores    │
│  - Training data  │    │   - Verification  │
└───────────────────┘    └───────────────────┘
```

## Components

### 1. Process Reward Model (PRM)

PRMs score individual reasoning steps rather than just final outcomes, enabling more precise credit assignment.

**Key Classes:**
- `ProcessRewardModel` - Abstract base class for all PRMs
- `LLMProcessRewardModel` - Uses LLM to evaluate step quality
- `MonteCarloProcessRewardModel` - Monte Carlo estimation (ReST-MCTS* approach)
- `HeuristicProcessRewardModel` - Domain-specific heuristics
- `EnsemblePRM` - Combines multiple PRMs

**Usage:**
```python
from src.framework.mcts import (
    LLMProcessRewardModel,
    ReasoningStep,
    ReasoningTrajectory,
)

# Create PRM
async def evaluate_fn(prompt: str) -> dict:
    response = await model.generate(prompt)
    return {"text": response.text}

prm = LLMProcessRewardModel(
    evaluate_fn=evaluate_fn,
    cache_size=1000,
)

# Create trajectory
trajectory = ReasoningTrajectory(query="What is 2 + 2?")
trajectory.add_step(ReasoningStep(
    content="First, identify the operands",
    step_type="reasoning",
))

# Score steps
scores = await prm.score_trajectory(trajectory)
for score in scores:
    print(f"Step score: {score.step_score:.2f}")
```

**MCTS Integration:**
```python
from src.framework.mcts import PRMMCTSIntegration, PRMEnhancedMCTSConfig

config = PRMEnhancedMCTSConfig(
    prm_selection_weight=0.5,  # Weight for PRM in UCB
    prm_expansion_threshold=0.3,  # Minimum score to expand
    prm_expansion_top_k=5,  # Keep top-k candidates
)

integration = PRMMCTSIntegration(prm, config)

# Enhanced UCT score
score = await integration.enhanced_uct_score(
    node_value=5.0,
    node_visits=10,
    parent_visits=100,
    step=step,
    trajectory=trajectory,
)
```

### 2. Extended Thinking

Adaptive token budget allocation for reasoning models.

**Key Classes:**
- `ThinkingBudget` - Configures thinking token budgets
- `ThinkingMode` - Enum for thinking intensity levels
- `ExtendedThinkingEvaluator` - Abstract evaluator interface
- `ClaudeExtendedThinkingEvaluator` - Claude-specific implementation
- `ParallelThinkingEvaluator` - Best-of-N parallel evaluation
- `AdaptiveThinkingRouter` - Routes based on task complexity

**Usage:**
```python
from src.framework.mcts import (
    ThinkingBudget,
    ThinkingMode,
    AdaptiveThinkingRouter,
)

# Configure budget
budget = ThinkingBudget(
    min_tokens=1024,
    max_tokens=65536,
    default_tokens=8192,
    depth_multiplier=1.2,  # More tokens for deeper nodes
    uncertainty_multiplier=1.5,  # More tokens when uncertain
)

# Compute adaptive budget
tokens = budget.compute_budget(
    depth=3,
    visits=5,
    ucb_score=0.7,
    uncertainty=0.6,
)
print(f"Allocated {tokens} thinking tokens")

# Get thinking mode
mode = budget.get_mode(tokens)
print(f"Mode: {mode.value}")  # e.g., "standard"
```

**Parallel Evaluation:**
```python
from src.framework.mcts import ParallelThinkingEvaluator

parallel_eval = ParallelThinkingEvaluator(
    base_evaluator=claude_evaluator,
    num_samples=3,
    aggregation="max",  # or "mean", "median", "vote"
)

result = await parallel_eval.evaluate(state, context, budget)
print(f"Score: {result.score} (from {len(result.metadata['all_scores'])} samples)")
```

### 3. Hybrid Search Strategy

Combines parallel generation with serial deep analysis.

**Phases:**
1. **Parallel Generation** (~25% compute) - Sample multiple candidates
2. **PRM Filtering** (~25% compute) - Score and filter with PRM
3. **Extended Evaluation** (~50% compute) - Deep thinking for top candidates
4. **Backpropagation** - Update tree with results

**Usage:**
```python
from src.framework.mcts import (
    HybridMCTSSearch,
    HybridSearchConfig,
    MCTSEngine,
)

config = HybridSearchConfig(
    parallel_budget_ratio=0.25,
    prm_budget_ratio=0.25,
    extended_budget_ratio=0.50,
    num_parallel_candidates=8,
    prm_top_k=3,
    early_terminate_confidence=0.95,
)

search = HybridMCTSSearch(
    mcts_engine=MCTSEngine(seed=42),
    prm=prm,
    thinking_evaluator=thinking_eval,
    config=config,
)

result = await search.search(
    root=root_node,
    action_generator=get_actions,
    state_transition=transition,
    rollout_policy=rollout,
    query="Solve this problem...",
)

print(f"Best action: {result.best_action}")
print(f"Confidence: {result.best_candidate.confidence}")
print(f"Phases: {[p.value for p in result.phases_completed]}")
```

### 4. Reasoning Node

Enhanced MCTS node with reasoning capabilities.

**Key Classes:**
- `ReasoningMCTSNode` - Node with reasoning traces and PRM scores
- `ReasoningMetadata` - Stores thinking traces, PRM scores, agent attribution

**Usage:**
```python
from src.framework.mcts import (
    ReasoningMCTSNode,
    ReasoningStep,
    MCTSState,
)

state = MCTSState(state_id="root", features={"query": "test"})
node = ReasoningMCTSNode(state=state)

# Add reasoning step
node.add_reasoning_step(ReasoningStep(
    content="Analyze the problem structure",
    step_type="reasoning",
    confidence=0.9,
))

# Set PRM score
node.set_prm_score(PRMScore(step_score=0.85, cumulative_score=0.85))

# Add child with reasoning
child = node.add_reasoning_child(
    action="decompose",
    child_state=MCTSState(state_id="child", features={}),
    reasoning_step=ReasoningStep(content="Break into subproblems"),
)

# Get full trajectory
trajectory = child.get_trajectory()
print(trajectory.to_text())
```

### 5. Dual-Agent Architecture

Separates strategic reasoning from action execution.

**Agents:**
- `ReasonerAgent` - Extended thinking, no tool access, proposes strategies
- `ActorAgent` - Fast execution, tool access, implements strategies

**Usage:**
```python
from src.framework.mcts import (
    ReasonerAgent,
    ActorAgent,
    DualAgentMCTSController,
)

# Create agents
async def reasoner_model(prompt, tokens):
    return await claude.think(prompt, budget=tokens)

async def actor_model(prompt):
    return await gpt4.execute(prompt)

reasoner = ReasonerAgent(
    model_fn=reasoner_model,
    default_thinking_tokens=16384,
)

actor = ActorAgent(
    model_fn=actor_model,
    tools=[search_tool, calculator],
)

# Create controller
controller = DualAgentMCTSController(
    reasoner=reasoner,
    actor=actor,
    prm=prm,
)

# Expand node with reasoning
children = await controller.expand_with_reasoning(
    node=current_node,
    context="Problem context...",
    n_strategies=5,
)
```

### 6. LangGraph Integration

Complete workflow for reasoning-enhanced MCTS.

**Usage:**
```python
from src.framework.mcts import (
    create_reasoning_graph,
    run_reasoning_search,
    ReasoningGraphConfig,
)

# Configure
config = ReasoningGraphConfig(
    mcts_iterations=50,
    prm_enabled=True,
    prm_selection_weight=0.5,
    thinking_enabled=True,
    hybrid_search_enabled=True,
    verification_enabled=True,
)

# Create and run
result = await run_reasoning_search(
    query="Solve this complex problem...",
    model_adapter=model_adapter,
    logger=logger,
    config=config,
)

print(f"Solution: {result['solution']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Iterations: {result['iterations']}")
print(f"Thinking tokens: {result['thinking_tokens']}")
```

## Configuration Reference

### PRMEnhancedMCTSConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prm_selection_weight` | float | 0.5 | Weight for PRM in UCB selection |
| `prm_expansion_threshold` | float | 0.3 | Minimum PRM score to expand |
| `prm_expansion_top_k` | int | 5 | Max candidates after filtering |
| `use_prm_for_backprop` | bool | True | Use PRM for backpropagation |
| `prm_backprop_discount` | float | 0.95 | Discount for step scores |

### ThinkingBudget

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_tokens` | int | 1024 | Minimum thinking tokens |
| `max_tokens` | int | 65536 | Maximum thinking tokens |
| `default_tokens` | int | 8192 | Default budget |
| `depth_multiplier` | float | 1.2 | Scale by tree depth |
| `uncertainty_multiplier` | float | 1.5 | Scale by uncertainty |

### HybridSearchConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_budget_ratio` | float | 0.25 | Compute for phase 1 |
| `prm_budget_ratio` | float | 0.25 | Compute for phase 2 |
| `extended_budget_ratio` | float | 0.50 | Compute for phase 3 |
| `num_parallel_candidates` | int | 8 | Parallel samples |
| `prm_top_k` | int | 3 | Keep after PRM filtering |
| `early_terminate_confidence` | float | 0.95 | Early stop threshold |

### ReasoningGraphConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mcts_iterations` | int | 50 | Search iterations |
| `prm_enabled` | bool | True | Enable PRM |
| `thinking_enabled` | bool | True | Enable extended thinking |
| `hybrid_search_enabled` | bool | True | Enable hybrid search |
| `adaptive_complexity` | bool | True | Route by complexity |

## Best Practices

### 1. Budget Allocation

For complex reasoning tasks:
- Start with 8K-16K thinking tokens for standard evaluation
- Use 32K+ for critical decision points
- Apply parallel evaluation (Best-of-3) for high-uncertainty nodes

### 2. PRM Integration

- Use PRM weights between 0.3-0.5 for UCB selection
- Set expansion threshold based on task difficulty (0.3-0.5)
- Train domain-specific PRMs using `PRMTrainingCollector`

### 3. Avoiding Overthinking

- Monitor `TaskComplexity.overthinking_risk`
- Use `AdaptiveThinkingRouter` for automatic routing
- Set reasonable token caps per iteration

### 4. Verification

- Enable verification for code and math problems
- Use `VerifiedHybridSearch` for production systems
- Combine multiple verifiers for robustness

## References

1. "Let's Verify Step by Step" - OpenAI
2. "ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search"
3. "Scaling Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters"
4. "Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners"
5. "Language Agent Tree Search" (LATS) - LangGraph implementation
