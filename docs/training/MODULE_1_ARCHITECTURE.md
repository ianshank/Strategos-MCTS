# Module 1: System & Architecture Deep Dive

**Duration:** 8 hours (1.5 days)
**Format:** Lecture + Architecture Lab
**Difficulty:** Foundation
**Prerequisites:** Basic Python, REST API concepts

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Explain the system architecture** using C4 diagrams as reference
2. **Identify the roles** of HRM, TRM, and MCTS agents
3. **Navigate the codebase** confidently using architecture documentation
4. **Trace data flow** from API request to final response
5. **Design simple extensions** to the existing architecture

---

## Session 1: Architecture Overview (2 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [architecture.md](../architecture.md) - Full system architecture
- [C4_ARCHITECTURE.md](../C4_ARCHITECTURE.md) - C4 model suite

### Lecture: System Context (45 minutes)

#### External Actors
- **End Users:** Analysts requesting tactical or cybersecurity analysis
- **API Clients:** Applications integrating with the framework
- **Administrators:** System operators and ML engineers

#### System Boundary
What's inside vs. outside the LangGraph Multi-Agent MCTS system:

**Inside:**
- HRM, TRM, MCTS agents
- LangGraph orchestration layer
- Vector storage (Pinecone/local)
- Observability (LangSmith, OpenTelemetry)

**Outside:**
- LLM providers (OpenAI, Anthropic, LM Studio)
- External data sources
- Monitoring infrastructure (Prometheus, Grafana)

#### Key System Properties
- **Scalability:** Horizontal scaling via stateless agents
- **Observability:** Full E2E tracing with LangSmith
- **Modularity:** Swap agents or LLM providers independently
- **Extensibility:** Add new agent types without core changes

### Lecture: Container View (45 minutes)

#### Container Architecture Diagram

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────────────────────────────┐
│         API Gateway (FastAPI)           │
│  - Authentication (JWT)                 │
│  - Rate limiting                        │
│  - Request validation                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    LangGraph Orchestration Layer        │
│  - State machine routing                │
│  - Agent coordination                   │
│  - Error handling & retries             │
└──┬────────┬────────┬────────────────────┘
   │        │        │
   ▼        ▼        ▼
┌────┐  ┌────┐  ┌────────┐
│HRM │  │TRM │  │ MCTS   │
│    │  │    │  │ Engine │
└────┘  └────┘  └────────┘
   │        │        │
   └────────┴────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│           Storage Layer                  │
│  - Vector DB (Pinecone/local FAISS)     │
│  - Metadata store (PostgreSQL)          │
│  - Cache (Redis)                        │
└─────────────────────────────────────────┘
```

#### Container Responsibilities

**1. API Gateway Container**
- **Location:** [src/api/](../../src/api/)
- **Purpose:** External interface, auth, validation
- **Key files:**
  - `inference_server.py` - FastAPI routes
  - `auth.py` - JWT authentication

**2. LangGraph Orchestration**
- **Location:** [examples/langgraph_multi_agent_mcts.py](../../examples/langgraph_multi_agent_mcts.py)
- **Purpose:** State machine, agent routing, workflow control
- **Key concepts:**
  - `AgentState` - Typed state dictionary
  - Conditional edges - Dynamic routing
  - Checkpointing - State persistence

**3. Agent Containers**
- **Location:** [src/agents/](../../src/agents/)
- **Purpose:** Domain-specific reasoning
- **Files:**
  - `hrm_agent.py` - High-level reasoning
  - `trm_agent.py` - Tactical refinement
  - [src/framework/mcts/](../../src/framework/mcts/) - MCTS engine

**4. Storage Layer**
- **Location:** [src/storage/](../../src/storage/)
- **Purpose:** Persistence, retrieval, caching
- **Integrations:** Pinecone, FAISS, PostgreSQL, Redis

**5. Observability**
- **LangSmith:** Distributed tracing
- **OpenTelemetry:** Metrics and logs
- **Prometheus:** Time-series metrics
- **Grafana:** Dashboards

### Discussion & Q&A (30 minutes)

**Key Questions to Explore:**
1. Why separate HRM, TRM, and MCTS instead of one monolithic agent?
2. What are the tradeoffs of using LangGraph vs. a custom orchestrator?
3. How does the system handle LLM provider failures?
4. What happens if the vector DB is unavailable?

---

## Session 2: Agent Responsibilities (2 hours)

### Pre-Reading (30 minutes)

- [DEEPMIND_IMPLEMENTATION.md](../DEEPMIND_IMPLEMENTATION.md) - Agent design philosophy
- [AGENT_TRACING_GUIDE.md](../AGENT_TRACING_GUIDE.md) - Agent behavior and tracing

### Lecture: HRM Agent (High-Level Reasoning Module) (30 minutes)

#### Purpose
Decompose complex queries into structured subtasks and objectives.

#### Key Responsibilities
1. **Intent Analysis:** Understand user's high-level goal
2. **Task Decomposition:** Break into actionable subtasks
3. **Objective Identification:** Define success criteria
4. **Confidence Calibration:** Estimate decomposition quality

#### Input/Output
- **Input:** Raw user query (string)
- **Output:** Structured task list with objectives

#### Code Location
- [src/agents/hrm_agent.py](../../src/agents/hrm_agent.py)
- Tests: [tests/components/test_hrm_agent_traced.py](../../tests/components/test_hrm_agent_traced.py)

#### Example Flow
```python
Query: "What's the best tactical approach for urban warfare?"

HRM Output:
{
  "tasks": [
    "Identify urban terrain characteristics",
    "Analyze force composition requirements",
    "Evaluate civilian consideration factors",
    "Assess logistics and supply chain needs"
  ],
  "objectives": {
    "primary": "Develop comprehensive urban tactics",
    "secondary": ["Minimize collateral damage", "Optimize resource usage"]
  },
  "confidence": 0.87
}
```

### Lecture: TRM Agent (Tactical Refinement Module) (30 minutes)

#### Purpose
Iteratively refine solutions through multi-round reasoning and self-critique.

#### Key Responsibilities
1. **Initial Solution Generation:** Create baseline answer
2. **Self-Critique:** Identify weaknesses and gaps
3. **Iterative Refinement:** Improve through multiple rounds
4. **Convergence Detection:** Stop when improvements plateau
5. **Alternative Ranking:** Score multiple candidate solutions

#### Input/Output
- **Input:** Task description + initial context
- **Output:** Refined solution + alternatives + convergence flag

#### Code Location
- [src/agents/trm_agent.py](../../src/agents/trm_agent.py)
- Tests: [tests/components/test_trm_agent_traced.py](../../tests/components/test_trm_agent_traced.py)

#### Refinement Loop
```
Round 1: Generate initial solution
Round 2: Critique → Refine → Improve by 15%
Round 3: Critique → Refine → Improve by 8%
Round 4: Critique → Refine → Improve by 2%  ← Converged!
```

### Lecture: MCTS Engine (Monte Carlo Tree Search) (45 minutes)

#### Purpose
Explore decision space using tree search and win probability estimation.

#### Key Responsibilities
1. **Selection:** Choose promising nodes using UCB1 policy
2. **Expansion:** Add new child nodes to explore
3. **Simulation:** Rollout to estimate value
4. **Backpropagation:** Update ancestors with results

#### Core Algorithm (UCB1)
```python
def ucb1_score(node, exploration_constant=1.41):
    if node.visit_count == 0:
        return float('inf')  # Explore unvisited nodes first

    exploitation = node.win_rate
    exploration = exploration_constant * sqrt(
        log(node.parent.visit_count) / node.visit_count
    )
    return exploitation + exploration
```

#### Input/Output
- **Input:** Decision tree + current state + iteration budget
- **Output:** Best path + win probabilities + visit statistics

#### Code Location
- [src/framework/mcts/](../../src/framework/mcts/)
- Tests: [tests/components/test_mcts_agent_traced.py](../../tests/components/test_mcts_agent_traced.py)

#### Key Hyperparameters
- **Iterations:** 100-500 (more = better quality, higher latency)
- **Exploration constant:** 1.0-2.0 (higher = more exploration)
- **Simulation depth:** 3-10 moves ahead

### Discussion & Q&A (15 minutes)

**Key Questions:**
1. When should you use HRM-only vs. full-stack (HRM+TRM+MCTS)?
2. How do you tune TRM refinement iterations?
3. What's the latency tradeoff for MCTS iterations?

---

## Session 3: LangGraph Integration (2 hours)

### Pre-Reading (30 minutes)

- [langgraph_mcts_architecture.md](../langgraph_mcts_architecture.md)
- [examples/langgraph_multi_agent_mcts.py](../../examples/langgraph_multi_agent_mcts.py) (first 200 lines)

### Lecture: State Machine Design (45 minutes)

#### LangGraph Concepts

**1. State Schema (AgentState)**
```python
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str                    # Original user query
    decomposed_tasks: List[str]   # From HRM
    refined_solution: str         # From TRM
    mcts_result: Optional[dict]   # From MCTS
    current_phase: str            # 'hrm' | 'trm' | 'mcts' | 'complete'
    metadata: dict                # Tracing and performance data
```

**2. Node Functions**
Each node is a Python function:
```python
def hrm_node(state: AgentState) -> AgentState:
    """Process query through HRM agent."""
    decomposed_tasks = hrm_agent.decompose(state["query"])
    return {
        **state,
        "decomposed_tasks": decomposed_tasks,
        "current_phase": "trm",
        "metadata": {...}
    }
```

**3. Conditional Edges**
Dynamic routing based on state:
```python
def should_run_mcts(state: AgentState) -> str:
    """Decide if MCTS is needed."""
    if state["query"].requires_decision_tree:
        return "run_mcts"
    else:
        return "skip_mcts"
```

**4. Graph Definition**
```python
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("hrm", hrm_node)
graph.add_node("trm", trm_node)
graph.add_node("mcts", mcts_node)

# Add edges
graph.add_edge("hrm", "trm")
graph.add_conditional_edges(
    "trm",
    should_run_mcts,
    {"run_mcts": "mcts", "skip_mcts": END}
)

# Compile
app = graph.compile()
```

### Lecture: Comparison with CrewAI (30 minutes)

| Feature | LangGraph | CrewAI |
|---------|-----------|--------|
| **State Management** | Explicit TypedDict | Implicit task memory |
| **Routing** | Conditional edges | Sequential by default |
| **Checkpointing** | Built-in | Manual |
| **Debugging** | State inspection | Limited visibility |
| **Flexibility** | Full control | Opinionated |

**Why LangGraph for This Framework?**
- Need explicit control over routing (HRM → TRM → MCTS)
- Benefit from state checkpointing for long-running MCTS
- Better integration with LangSmith tracing
- More flexible error handling and retries

### Live Demo: Tracing a Query (15 minutes)

**Instructor Demo:**
1. Run `scripts/smoke_test_traced.py`
2. Open LangSmith UI
3. Show trace hierarchy:
   - Workflow (top level)
   - HRM phase
   - TRM phase
   - MCTS phase
   - Individual LLM calls

---

## Session 4: Hands-On Lab (2 hours)

### Lab 1: Navigate the Codebase (30 minutes)

**Objective:** Use architecture docs to find key code locations.

**Tasks:**
1. Find the FastAPI route that handles `/analyze` requests
2. Locate the HRM agent's task decomposition logic
3. Find the MCTS UCB1 selection implementation
4. Identify where LangSmith tracing is initialized

**Deliverable:** Document with file paths and line numbers

---

### Lab 2: Trace a Sample Query (45 minutes)

**Objective:** Follow a complete query through the system.

**Tasks:**
1. Run this command:
   ```bash
   python scripts/smoke_test_traced.py
   ```

2. Open LangSmith and find your trace

3. Answer these questions:
   - What was the total latency?
   - How many LLM calls were made?
   - What tasks did HRM identify?
   - Did TRM converge? After how many rounds?
   - Was MCTS invoked? If so, with how many iterations?

4. Create a sequence diagram showing the execution flow

**Deliverable:** Sequence diagram + answers document

---

### Lab 3: Identify Extension Point (30 minutes)

**Objective:** Plan where to add a new capability.

**Scenario:** You need to add a new agent called "Risk Assessment Module (RAM)" that evaluates risks before TRM refinement.

**Tasks:**
1. Where in the LangGraph would you add the RAM node?
2. What changes are needed to `AgentState`?
3. What conditional logic determines if RAM should run?
4. How would you trace RAM's execution in LangSmith?

**Deliverable:** Architecture modification plan (1-page)

---

### Lab 4: Architecture Quiz (15 minutes)

**Take the quiz:** [MODULE_1_QUIZ.md](MODULE_1_QUIZ.md)

**Passing Score:** 80% (16/20 questions correct)

---

## Post-Session Resources

### Additional Reading
- [SCALABILITY_ANALYSIS.md](../reports/SCALABILITY_ANALYSIS.md) - Performance characteristics
- [DEPLOYMENT_REPORT.md](../reports/DEPLOYMENT_REPORT.md) - Production deployment patterns

### Video Recordings
- Architecture walkthrough (to be recorded)
- Live trace demo (to be recorded)

### Office Hours
- When: [Schedule TBD]
- Where: [Link TBD]
- Topics: Architecture questions, debugging codebase navigation issues

---

## Assessment Rubric

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Lab Completion** | 40% | All 4 labs submitted with correct answers |
| **Quiz Score** | 30% | Minimum 80% on architecture quiz |
| **Sequence Diagram** | 20% | Accurate representation of query flow |
| **Extension Plan** | 10% | Thoughtful design for RAM agent addition |

**Minimum Passing:** 70% overall

---

## Next Module

Continue to [MODULE_2_AGENTS.md](MODULE_2_AGENTS.md) - Agents Deep Dive

**Prerequisites for Module 2:**
- Completed Module 1 labs
- Scored 80%+ on architecture quiz
- Familiarity with pytest (review [pytest documentation](https://docs.pytest.org/) if needed)
