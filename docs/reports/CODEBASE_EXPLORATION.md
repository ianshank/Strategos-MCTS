# LangGraph Multi-Agent MCTS Framework - Complete Codebase Exploration

## Executive Summary

This is a sophisticated, production-ready multi-agent framework combining **MCTS (Monte Carlo Tree Search)**, **HRM (Hierarchical Reasoning Model)**, **TRM (Task Response Model)**, and **Neural Meta-Controllers** with **LangGraph** state machine orchestration. The project contains **~22K lines of Python code** across **64 files**, with existing implementations of training pipelines, observability systems, and provider-agnostic LLM adapters.

---

## 1. COMPLETE DIRECTORY STRUCTURE

```

├── .claude/                           # Claude AI configuration
├── .github/                           # GitHub Actions CI/CD
├── .git/                              # Git repository
├── .env.example                       # Environment template with 20+ variables
├── .gitignore                         # Git ignore patterns
├── .pre-commit-config.yaml            # Pre-commit hooks (ruff, black, bandit)
├── .secrets.baseline                  # Security baseline
├── pyproject.toml                     # Project config (setuptools, ruff, black, mypy, pytest)
├── requirements.txt                   # Basic dependencies (45 lines)
├── README.md                          # Main documentation
├── langgraph_mcts_architecture.md     # Detailed architecture spec (19K)
├── NEURAL_TRAINING_SUMMARY.md         # Training results summary
├── INTEGRATION_STATUS.md              # Service integration status
├── API_CONFIGURATION_GUIDE.md         # API setup guide
├── SETUP_COMPLETE.md                  # Setup documentation
│
├── src/                               # Main source code (8,340 lines)
│   ├── __init__.py
│   ├── adapters/
│   │   └── llm/
│   │       ├── base.py                # LLMClient Protocol & LLMResponse dataclasses
│   │       ├── openai_client.py       # OpenAI adapter with retries
│   │       ├── anthropic_client.py    # Anthropic Messages API adapter
│   │       ├── lmstudio_client.py     # Local LM Studio support
│   │       ├── exceptions.py          # Custom exceptions
│   │       └── __init__.py            # Factory function: create_client()
│   │
│   ├── config/
│   │   ├── settings.py                # Pydantic Settings v2 with validation
│   │   └── __init__.py                # get_settings() helper
│   │
│   ├── framework/
│   │   ├── agents/
│   │   │   ├── base.py                # AsyncAgentBase, AgentContext, AgentResult
│   │   │   │                          # CompositeAgent, ParallelAgent, SequentialAgent
│   │   │   └── __init__.py
│   │   │
│   │   ├── mcts/
│   │   │   ├── core.py                # MCTSEngine, MCTSNode, MCTSState
│   │   │   ├── config.py              # MCTSConfig with 20+ parameters + presets
│   │   │   ├── policies.py            # UCB1, SelectionPolicy enum, rollout policies
│   │   │   ├── experiments.py         # ExperimentTracker for tracking runs
│   │   │   └── __init__.py
│   │   │
│   │   ├── graph.py                   # LangGraph integration with AgentState
│   │   └── __init__.py
│   │
│   ├── agents/
│   │   └── meta_controller/
│   │       ├── base.py                # AbstractMetaController, MetaControllerFeatures
│   │       ├── rnn_controller.py      # GRU-based RNN meta-controller (99.78% accuracy)
│   │       ├── bert_controller.py     # BERT+LoRA meta-controller
│   │       ├── config_loader.py       # MetaControllerConfigLoader
│   │       ├── utils.py               # Helper functions (normalize_features, etc.)
│   │       └── __init__.py
│   │
│   ├── models/
│   │   ├── validation.py              # Pydantic validation models
│   │   └── __init__.py
│   │
│   ├── observability/
│   │   ├── logging.py                 # JSON structured logging
│   │   ├── tracing.py                 # OpenTelemetry integration
│   │   ├── metrics.py                 # Performance metrics collection
│   │   ├── profiling.py               # Performance profiling
│   │   ├── debug.py                   # MCTS tree visualization (text + DOT)
│   │   ├── braintrust_tracker.py      # Braintrust experiment tracking
│   │   └── __init__.py
│   │
│   ├── storage/
│   │   ├── s3_client.py               # Async S3 client with retries
│   │   ├── pinecone_store.py          # Pinecone vector store integration
│   │   └── __init__.py
│   │
│   └── training/
│       ├── __init__.py
│       ├── data_generator.py          # MetaControllerDataGenerator (23K lines)
│       ├── train_rnn.py               # RNNTrainer with full training pipeline (33K lines)
│       └── train_bert_lora.py         # BERTLoRATrainer with HF Trainer integration (24K lines)
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_integration_e2e.py        # End-to-end integration tests
│   ├── test_e2e_providers.py          # Provider-specific tests
│   ├── test_meta_controller.py        # Meta-controller unit tests
│   └── fixtures/
│       ├── __init__.py
│       ├── mcts_fixtures.py           # MCTS test fixtures
│       └── meta_controller_fixtures.py # Meta-controller fixtures
│
├── tools/
│   ├── mcp/
│   │   └── server.py                  # MCP (Model Context Protocol) server
│   └── cli/                           # CLI tools (if any)
│
├── demos/
│   └── neural_meta_controller_demo.py # Demo of neural meta-controller
│
├── examples/                          # Usage examples
│
├── models/                            # Trained model artifacts
│   ├── rnn_meta_controller.pt         # Trained RNN model (317KB)
│   ├── rnn_meta_controller.history.json # Training history
│   └── bert_lora/
│       ├── final_model/
│       │   ├── adapter_model.safetensors # BERT LoRA weights
│       │   └── README.md
│       └── generated_dataset.json     # Generated training data
│
├── docs/
│   ├── architecture.md                # Architecture documentation
│   ├── PINECONE_INTEGRATION.md        # Pinecone setup guide
│   ├── EXPERIMENT_TRACKING_GUIDE.md   # Braintrust tracking guide
│   ├── security_audit.md              # Security findings
│   └── mermaid/                       # Mermaid diagrams
│
└── scripts/
    └── export_architecture_diagrams.py # Export diagrams to images
```

---

## 2. KEY FILE LOCATIONS & PURPOSES

### Core Agent Framework
- **Base Classes**: `src/framework/agents/base.py`
  - `AsyncAgentBase`: Abstract base for all agents
  - `ParallelAgent`: Run multiple agents concurrently
  - `SequentialAgent`: Chain agents together
  - `AgentContext`: Input data structure
  - `AgentResult`: Output data structure

### MCTS Implementation (New Deterministic Core)
- **Core Engine**: `src/framework/mcts/core.py`
  - `MCTSEngine`: Main MCTS orchestrator
  - `MCTSNode`: Tree node with UCB1 scoring
  - `MCTSState`: Hashable state for caching
  - Features: Seeded RNG, progressive widening, simulation caching, parallel rollouts

- **Configuration**: `src/framework/mcts/config.py`
  - `MCTSConfig`: 20+ configurable parameters
  - Presets: FAST (25 iter), BALANCED (100 iter), THOROUGH (500 iter)
  - Plus: EXPLORATION_HEAVY, EXPLOITATION_HEAVY variants

- **Policies**: `src/framework/mcts/policies.py`
  - `ucb1()`: Upper Confidence Bound formula
  - `SelectionPolicy`: Enum for action selection (MAX_VISITS, MAX_VALUE, ROBUST_CHILD)
  - `RandomRolloutPolicy`, `GreedyRolloutPolicy`, `HybridRolloutPolicy`

### LangGraph Orchestration
- **Graph Building**: `src/framework/graph.py`
  - `AgentState`: TypedDict for shared state across graph
  - Node functions for: initialization, RAG retrieval, routing, agent execution
  - Integration with new MCTS core
  - Meta-controller support (optional)

### Neural Meta-Controllers
- **Base Classes**: `src/agents/meta_controller/base.py`
  - `AbstractMetaController`: Base for all controllers
  - `MetaControllerFeatures`: 8-dimensional feature vector (confidence, iteration, etc.)
  - `MetaControllerPrediction`: Agent selection output

- **RNN Controller** (RECOMMENDED): `src/agents/meta_controller/rnn_controller.py`
  - `RNNMetaControllerModel`: GRU-based architecture
  - `RNNMetaController`: Wrapper with prediction logic
  - **Accuracy**: 99.78% on test set

- **BERT Controller**: `src/agents/meta_controller/bert_controller.py`
  - `BERTMetaControllerModel`: BERT+LoRA architecture
  - `BERTMetaController`: Wrapper with tokenization
  - **Accuracy**: 47.68% (experimental, slower)

### LLM Adapters (Provider-Agnostic)
- **Base Protocol**: `src/adapters/llm/base.py`
  - `LLMClient` Protocol: Structural subtyping interface
  - `LLMResponse`: Standardized response format
  - `ToolCall`: Tool/function call representation

- **Implementations**:
  - OpenAI: `src/adapters/llm/openai_client.py`
  - Anthropic: `src/adapters/llm/anthropic_client.py`
  - LM Studio: `src/adapters/llm/lmstudio_client.py`

### Configuration Management
- **Settings**: `src/config/settings.py`
  - Pydantic Settings v2 with environment variable validation
  - LLM provider selection
  - MCTS parameters
  - API keys (SecretStr for security)
  - Supports: OpenAI, Anthropic, LM Studio, Pinecone, Braintrust

### Training Pipelines
- **Data Generator**: `src/training/data_generator.py` (23K lines)
  - `MetaControllerDataGenerator`: Generates synthetic training data
  - 3-class classification (HRM, TRM, MCTS)
  - Supports balanced/unbalanced datasets

- **RNN Training**: `src/training/train_rnn.py` (33K lines)
  - `RNNTrainer`: Full training pipeline
  - Features: Early stopping, checkpointing, per-class metrics
  - Braintrust integration for experiment tracking

- **BERT LoRA Training**: `src/training/train_bert_lora.py` (24K lines)
  - `BERTLoRATrainer`: Training with HuggingFace Trainer
  - LoRA adapters for parameter efficiency
  - Full evaluation pipeline

### Observability
- **Logging**: JSON structured logs with correlation IDs
- **Tracing**: OpenTelemetry integration for full request lifecycle
- **Metrics**: Performance metrics collection and aggregation
- **Profiling**: CPU/memory profiling and bottleneck detection
- **Braintrust**: Experiment tracking integration
- **Debugging**: MCTS tree visualization (text format + DOT for Graphviz)

---

## 3. ARCHITECTURE PATTERNS & DESIGN

### Multi-Agent Orchestration Pattern
```
User Query
    ↓
LangGraph State Machine
    ├─ Initialize State
    ├─ RAG Retrieval (optional)
    ├─ Route to Agents
    ├─ Execute:
    │  ├─ HRM (Hierarchical Reasoning)
    │  ├─ TRM (Task Response Model) 
    │  ├─ MCTS (Monte Carlo Tree Search)
    │  └─ Meta-Controller (Neural agent selector)
    ├─ Aggregate Results
    ├─ Evaluate Consensus
    ├─ Loop if needed (convergence check)
    └─ Synthesize Final Response
```

### MCTS Algorithm (Deterministic)
```
Loop N iterations:
    1. Selection: Navigate tree using UCB1 policy
    2. Expansion: Add new child node (progressive widening)
    3. Simulation: Run rollout using rollout policy
    4. Backpropagation: Update statistics from leaf to root
Return best action based on selection policy
```

### Neural Meta-Controller Pattern
```
Agent State Features (8-dimensional):
    - hrm_confidence: float [0, 1]
    - trm_confidence: float [0, 1]
    - mcts_value: float [0, 1]
    - consensus_score: float [0, 1]
    - last_agent: str (categorical)
    - iteration: int [0, 20]
    - query_length: int [0, 10000]
    - has_rag_context: bool
            ↓
    Neural Network
            ↓
    Agent Prediction: {hrm: 0.3, trm: 0.5, mcts: 0.2}
```

### Provider Abstraction Pattern
```python
# Works identically with any provider
client = create_client("openai", model="gpt-4")
client = create_client("anthropic", model="claude-3-sonnet")
client = create_client("lmstudio", base_url="http://localhost:1234/v1")

# All use LLMClient protocol
response = await client.generate(messages=[...], temperature=0.7)
```

---

## 4. EXISTING AGENT CLASSES

### HRM Agent (Hierarchical Reasoning Model)
- **Purpose**: Hierarchical decomposition of complex queries
- **Interface**: Via `AsyncAgentBase`
- **Status**: Base implementation available
- **Location**: Framework integration in `src/framework/graph.py`

### TRM Agent (Task Response Model)  
- **Purpose**: Iterative refinement and quality improvement
- **Interface**: Via `AsyncAgentBase`
- **Status**: Base implementation available
- **Location**: Framework integration in `src/framework/graph.py`

### MCTS Agent
- **Purpose**: Monte Carlo Tree Search for exploring action spaces
- **Core**: `MCTSEngine` with deterministic RNG
- **Features**:
  - Configurable iterations (25-500+)
  - Progressive widening control
  - Simulation caching (SHA-256 based)
  - Parallel rollouts with bounded concurrency
  - Multiple rollout policies (random, greedy, hybrid, LLM)
- **Location**: `src/framework/mcts/`
- **Status**: Fully implemented and tested

### Neural Meta-Controller
- **Purpose**: Dynamic agent selection based on system state
- **RNN Version** (RECOMMENDED):
  - Architecture: GRU with 64 hidden units
  - Accuracy: 99.78%
  - Parameters: 17,211 (tiny model)
  - Training time: ~5 seconds
  
- **BERT Version** (Experimental):
  - Architecture: BERT-mini + LoRA adapters
  - Accuracy: 47.68%
  - Trainable parameters: 17,155 of 11,188,486 (0.15%)
  - Training time: ~11 seconds

- **Location**: `src/agents/meta_controller/`
- **Status**: Fully implemented with trained models in `models/`

---

## 5. LANGGRAPH ORCHESTRATION PATTERNS

### Graph State Definition
```python
class AgentState(TypedDict):
    query: str
    use_mcts: bool
    use_rag: bool
    rag_context: NotRequired[str]
    retrieved_docs: NotRequired[List[Dict]]
    hrm_results: NotRequired[Dict]
    trm_results: NotRequired[Dict]
    agent_outputs: Annotated[List[Dict], operator.add]
    mcts_root: NotRequired[MCTSNode]
    mcts_iterations: NotRequired[int]
    mcts_best_action: NotRequired[str]
    mcts_stats: NotRequired[Dict]
    confidence_scores: NotRequired[Dict[str, float]]
    iteration: NotRequired[int]
```

### Graph Node Functions
1. **initialize_state()**: Parse input, setup context
2. **retrieve_context()**: RAG retrieval from vector stores
3. **route_to_agents()**: Conditional routing logic
4. **execute_hrm()**: Run HRM agent
5. **execute_trm()**: Run TRM agent
6. **execute_mcts()**: Run MCTS simulation
7. **aggregate_results()**: Combine outputs
8. **evaluate_consensus()**: Convergence check
9. **synthesize()**: Final response generation

### Graph Building
```python
graph = StateGraph(AgentState)
graph.add_node("init", initialize_state)
graph.add_node("rag", retrieve_context)
# ... add more nodes
graph.add_edge("init", "rag")
graph.add_conditional_edges("route", route_logic)
# ... add more edges
app = graph.compile(checkpointer=MemorySaver())
```

---

## 6. EXISTING TRAINING & EVALUATION CODE

### Training Pipelines
1. **RNN Meta-Controller Training** (`src/training/train_rnn.py`)
   - `RNNTrainer` class with 150+ lines of methods
   - Full training loop with validation
   - Early stopping with patience
   - Per-class F1/Precision/Recall metrics
   - Model checkpointing
   - Braintrust experiment tracking (optional)

2. **BERT LoRA Training** (`src/training/train_bert_lora.py`)
   - `BERTLoRATrainer` class
   - HuggingFace Trainer integration
   - LoRA adapter fine-tuning
   - Dataset preparation utilities
   - Full evaluation suite

### Data Generation
- **MetaControllerDataGenerator** (`src/training/data_generator.py`)
  - Generates synthetic training data
  - 3-class labels (HRM, TRM, MCTS)
  - Feature generation: confidence scores, iteration counts, etc.
  - Supports balanced/unbalanced datasets
  - Export formats: tensors, text, JSON

### Existing Training Results
- **RNN**: 99.78% accuracy, F1=0.9988 (best class HRM=1.0000)
- **BERT**: 47.68% accuracy (less suitable for this task)
- Models saved in `models/`

---

## 7. CURRENT DEPENDENCIES & VERSIONS

### Core Dependencies
```
langgraph>=0.0.20          # State machine orchestration
langchain>=0.1.0           # LLM framework
langchain-core>=0.1.0      # Core abstractions
langchain-openai>=0.0.2    # OpenAI integration
openai>=1.0.0              # OpenAI API client
```

### Data & Storage
```
chromadb>=0.4.0            # Vector store
faiss-cpu>=1.7.4           # Similarity search
pinecone>=7.0.0            # Vector database (optional)
aioboto3>=12.0.0           # Async S3 client
botocore>=1.31.0           # AWS SDK
```

### ML/Training (Optional)
```
torch>=2.0.0               # Deep learning
transformers>=4.30.0       # HuggingFace models
peft>=0.7.0                # LoRA adapters
datasets>=2.14.0           # Dataset tools
PyYAML>=6.0.0              # Config files
```

### Observability
```
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp-proto-grpc>=1.20.0
opentelemetry-instrumentation-httpx>=0.42b0
psutil>=5.9.0              # System monitoring
braintrust>=0.0.100        # Experiment tracking (optional)
```

### Development
```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
mypy>=1.7.0
ruff>=0.1.9
black>=23.12.0
pre-commit>=3.6.0
bandit[toml]>=1.7.0
pip-audit>=2.6.0
```

### Async & Utilities
```
httpx>=0.25.0              # Async HTTP client
tenacity>=8.2.0            # Retry logic
pydantic>=2.0.0            # Validation
```

---

## 8. CONFIGURATION PATTERNS

### Environment Variables (via Settings class)
```python
from src.config.settings import get_settings

settings = get_settings()  # Loads from .env

# LLM Configuration
settings.LLM_PROVIDER      # "openai" | "anthropic" | "lmstudio"
settings.OPENAI_API_KEY    # SecretStr
settings.ANTHROPIC_API_KEY # SecretStr

# MCTS Parameters
settings.MCTS_ITERATIONS   # int (default: 100)
settings.MCTS_C            # float (default: 1.414)
settings.SEED              # int (default: 42)

# Storage
settings.PINECONE_API_KEY  # SecretStr (optional)
settings.BRAINTRUST_API_KEY # SecretStr (optional)

# Logging
settings.LOG_LEVEL         # "DEBUG" | "INFO" | "WARNING"
```

### MCTS Configuration Presets
```python
from src.framework.mcts.config import FAST_CONFIG, BALANCED_CONFIG, THOROUGH_CONFIG

# Preset configs available:
# FAST: 25 iterations, aggressive exploration
# BALANCED: 100 iterations (default), balanced exploration/exploitation
# THOROUGH: 500 iterations, conservative exploration

# Custom config
config = MCTSConfig(
    num_iterations=100,
    exploration_weight=1.414,
    progressive_widening_k=1.0,
    rollout_policy="hybrid",
    max_parallel_rollouts=4,
    enable_cache=True,
)
```

### Meta-Controller Configuration
```python
from src.agents.meta_controller.config_loader import MetaControllerConfigLoader

# Load from YAML
config = MetaControllerConfigLoader.load_from_yaml("config.yaml")

# Configuration properties:
# - enabled: bool
# - type: "rnn" | "bert"
# - model_path: str
# - threshold: float
```

---

## 9. INTEGRATION POINTS FOR NEW TRAINING PIPELINE

### Where to Hook New Training Code

1. **Data Generation Enhancement**
   - Location: `src/training/data_generator.py`
   - Current: `MetaControllerDataGenerator` generates 8-dim feature vectors
   - Enhancement: Collect real production data instead of synthetic
   - Interface: Same `MetaControllerFeatures` dataclass

2. **Training Script Entry Points**
   - RNN: `src/training/train_rnn.py`
   - BERT: `src/training/train_bert_lora.py`
   - Both have `main()` functions for CLI execution
   - Can be called as Python modules or CLI scripts

3. **Model Evaluation**
   - Both trainers have `.evaluate()` methods
   - Metrics: accuracy, per-class F1/Precision/Recall
   - Confusion matrices, loss curves

4. **Experiment Tracking**
   - Location: `src/observability/braintrust_tracker.py`
   - Can log metrics to Braintrust
   - Function: `create_training_tracker()`

5. **Model Persistence**
   - RNN: Saves to `.pt` (PyTorch) format
   - BERT: Uses HuggingFace safe tensor format
   - Location: `models/`

6. **Graph Integration**
   - Location: `src/framework/graph.py`
   - Import new controllers: `from src.agents.meta_controller.rnn_controller import RNNMetaController`
   - Add to graph nodes: `graph.add_node("meta_control", select_agent_via_controller)`

---

## 10. PROJECT STATISTICS

| Metric | Count |
|--------|-------|
| Total Python Files | 64 |
| Total Lines of Code | ~22,219 |
| Source Code Lines (src/) | ~8,340 |
| Test Files | 7 |
| Documentation Files | 10+ |
| Training Scripts | 2 (RNN + BERT) |
| Trained Models | 2 (RNN + BERT LoRA) |
| LLM Adapters | 3 (OpenAI, Anthropic, LM Studio) |
| Observability Modules | 6 (logging, tracing, metrics, profiling, debug, braintrust) |

---

## 11. READY-TO-USE PATTERNS & EXAMPLES

### Basic Usage Pattern
```python
import asyncio
from src.adapters.llm import create_client
from src.config.settings import get_settings
from src.framework.agents.base import AsyncAgentBase

async def main():
    settings = get_settings()
    
    # Create LLM client (provider-agnostic)
    client = create_client(
        settings.LLM_PROVIDER,
        model="gpt-4-turbo-preview",
        timeout=60.0,
    )
    
    # Create custom agent
    class MyAgent(AsyncAgentBase):
        async def _process_impl(self, context):
            # Your logic here
            response = await self.generate_llm_response(
                prompt=f"Answer: {context.query}"
            )
            return AgentResult(response=response.text, confidence=0.9)
    
    agent = MyAgent(client, name="MyAgent")
    result = await agent.process(query="Test query")
    print(result)

asyncio.run(main())
```

### MCTS Usage Pattern
```python
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.config import BALANCED_CONFIG

async def run_mcts():
    engine = MCTSEngine(seed=42)
    
    root = MCTSNode(
        state=MCTSState(state_id="root", features={}),
        rng=engine.rng,
    )
    
    best_action, stats = await engine.search(
        root=root,
        num_iterations=100,
        action_generator=lambda s: ["a1", "a2", "a3"],
        state_transition=lambda s, a: MCTSState(f"{s.state_id}_{a}"),
        rollout_policy=rollout_policy,
    )
    
    print(f"Best action: {best_action}")
    print(f"Stats: {stats}")
```

---

## 12. SECURITY & COMPLIANCE

### Implemented
- Environment variables for API keys (never hardcoded)
- Pydantic validation on all external inputs
- SecretStr type for sensitive data
- Pre-commit hooks: bandit (security), ruff, black
- CI/CD pipeline with security scanning (pip-audit, bandit)
- Secrets redaction from logs
- Security audit documentation available

### Tested
- Provider-agnostic abstraction (injection resistance)
- Request sanitization
- Error handling without information leakage

---

## 13. RECOMMENDED NEXT STEPS FOR TRAINING PIPELINE

1. **Inherit from existing classes**:
   - Extend `MetaControllerDataGenerator` for production data collection
   - Extend `RNNTrainer` or `BERTLoRATrainer` for custom training logic

2. **Use established patterns**:
   - `AsyncAgentBase` for agent implementation
   - `AgentContext` / `AgentResult` for I/O contracts
   - `LLMClient` protocol for model access

3. **Leverage built-in tools**:
   - `BraintrustTracker` for experiment tracking
   - `ExperimentTracker` for MCTS stats
   - `MetricsCollector` for performance monitoring
   - OpenTelemetry for distributed tracing

4. **Integration checkpoints**:
   - Update `src/framework/graph.py` to use new controllers
   - Add training script to CI/CD in `.github/workflows/`
   - Document in README and architecture.md

---

## File Index for Quick Reference

| Purpose | File Path |
|---------|-----------|
| **Main Framework** | |
| LangGraph Orchestration | `src/framework/graph.py` |
| Agent Base Classes | `src/framework/agents/base.py` |
| MCTS Core | `src/framework/mcts/core.py` |
| MCTS Config | `src/framework/mcts/config.py` |
| **Agents** | |
| Meta-Controller Base | `src/agents/meta_controller/base.py` |
| RNN Controller | `src/agents/meta_controller/rnn_controller.py` |
| BERT Controller | `src/agents/meta_controller/bert_controller.py` |
| **Training** | |
| Data Generator | `src/training/data_generator.py` |
| RNN Trainer | `src/training/train_rnn.py` |
| BERT Trainer | `src/training/train_bert_lora.py` |
| **Configuration** | |
| Settings | `src/config/settings.py` |
| LLM Adapters | `src/adapters/llm/*.py` |
| **Observability** | |
| Logging | `src/observability/logging.py` |
| Tracing | `src/observability/tracing.py` |
| Metrics | `src/observability/metrics.py` |
| Braintrust | `src/observability/braintrust_tracker.py` |
| **Tests** | |
| Integration Tests | `tests/test_integration_e2e.py` |
| Meta-Controller Tests | `tests/test_meta_controller.py` |
| MCTS Fixtures | `tests/fixtures/mcts_fixtures.py` |

