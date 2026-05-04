# C4 Architecture Diagrams: DeepMind-Style Self-Improving AI System

This document provides C4 architecture diagrams (Context, Container, Component, and Code levels) for the LangGraph Multi-Agent MCTS framework with DeepMind-style learning.

## Table of Contents

1. [Level 1: System Context Diagram](#level-1-system-context-diagram)
2. [Level 2: Container Diagram](#level-2-container-diagram)
3. [Level 3: Component Diagrams](#level-3-component-diagrams)
4. [Level 4: Code Diagrams](#level-4-code-diagrams)
5. [Deployment Architecture](#deployment-architecture)
6. [Data Flow Diagrams](#data-flow-diagrams)

---

## Level 1: System Context Diagram

Shows the system in its environment with external actors and systems.

```mermaid
graph TB
    subgraph "External Systems & Users"
        User[AI Researcher/Developer]
        Client[Client Applications]
        WandB[Weights & Biases]
        Storage[Cloud Storage S3]
        Monitor[Monitoring Systems]
        Pinecone[Vector Database]
        ArXiv[ArXiv API]
    end

    subgraph "LangGraph Multi-Agent MCTS System"
        System[DeepMind-Style<br/>Self-Improving AI System<br/><br/>Combines HRM, TRM, and Neural MCTS<br/>for hierarchical reasoning and<br/>self-improving capabilities]
    end

    User -->|Configure & Train| System
    User -->|Monitor Progress| System
    Client -->|Inference Requests<br/>REST API| System
    System -->|Log Experiments<br/>Metrics| WandB
    System -->|Store Checkpoints<br/>Training Data| Storage
    System -->|Performance Metrics<br/>Telemetry| Monitor
    System -->|Index & Retrieve<br/>Knowledge| Pinecone
    System -->|Fetch Papers<br/>Research Corpus| ArXiv
    System -->|Trained Models<br/>Predictions| Client

    style System fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style User fill:#95E1D3,stroke:#38A89D,stroke-width:2px
    style Client fill:#95E1D3,stroke:#38A89D,stroke-width:2px
    style WandB fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
    style Storage fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
    style Monitor fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
    style Pinecone fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
    style ArXiv fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px
```

### Key Relationships

| From | To | Description |
|------|-----|-------------|
| **AI Researcher** | System | Configures training parameters, monitors experiments |
| **Client Applications** | System | Makes inference requests via REST API |
| **System** | Weights & Biases | Logs training metrics, experiments, model performance |
| **System** | Cloud Storage | Persists checkpoints, training data, replay buffer |
| **System** | Monitoring | Sends telemetry, performance metrics, alerts |
| **System** | Pinecone | Stores/Retrieves vector embeddings for RAG |
| **System** | ArXiv API | Fetches research papers for knowledge corpus |

---

## Level 2: Container Diagram

Shows the high-level technical building blocks (applications, data stores, services).

```mermaid
graph TB
    subgraph "User & Client Layer"
        User[AI Researcher]
        Client[Client App]
    end

    subgraph "LangGraph Multi-Agent MCTS System"
        subgraph "Training System"
            Orchestrator[Training Orchestrator<br/>Python/AsyncIO<br/><br/>Coordinates training pipeline]
            DataGen[Synthetic Generator<br/>Python/LLM<br/><br/>Generates training data]
            CorpusBuilder[Corpus Builder<br/>Python<br/><br/>Fetches & indexes papers]
            Monitor[Performance Monitor<br/>Python<br/><br/>Tracks metrics]
        end

        subgraph "Core Models"
            HRM[HRM Agent<br/>PyTorch DeBERTa<br/><br/>Hierarchical reasoning]
            TRM[TRM Agent<br/>PyTorch DeBERTa<br/><br/>Recursive refinement]
            MCTS[Neural MCTS<br/>Python/PyTorch<br/><br/>Tree search with NN]
            MetaController[Meta Controller<br/>PyTorch GRU<br/><br/>Agent routing]
            ParallelMCTS[Parallel MCTS<br/>AsyncIO/PyTorch<br/><br/>Virtual loss parallelization]
            PVNet[Policy-Value Network<br/>PyTorch ResNet<br/><br/>Action probabilities + value]
        end

        subgraph "Inference System"
            API[FastAPI Server<br/>FastAPI/Uvicorn<br/><br/>REST API endpoints]
            InferenceEngine[Inference Engine<br/>Python/PyTorch<br/><br/>Model inference]
        end

        subgraph "Data Layer"
            ReplayBuffer[Replay Buffer<br/>Python/NumPy<br/><br/>Experience storage]
            Cache[Evaluation Cache<br/>Python Dict<br/><br/>MCTS caching]
            VectorStore[Vector Store<br/>Pinecone<br/><br/>RAG Knowledge Base]
        end
    end

    subgraph "External Services"
        WandB[(Weights & Biases<br/>Experiment Tracking)]
        S3[(Cloud Storage<br/>S3/MinIO)]
        Prometheus[(Monitoring<br/>Prometheus/Grafana)]
        LLMProvider[LLM Provider<br/>OpenAI/Anthropic]
    end

    User -->|Configure| Orchestrator
    Client -->|HTTP POST| API

    Orchestrator -->|Orchestrates| DataGen
    Orchestrator -->|Orchestrates| CorpusBuilder
    Orchestrator -->|Trains| HRM
    Orchestrator -->|Trains| TRM
    Orchestrator -->|Trains| PVNet
    Orchestrator -->|Trains| MetaController
    Orchestrator -->|Uses| Monitor

    DataGen -->|Uses| LLMProvider
    DataGen -->|Stores| ReplayBuffer
    CorpusBuilder -->|Indexes to| VectorStore

    MCTS -->|Evaluates with| PVNet
    MCTS -->|Caches in| Cache
    MCTS -->|Guided by| HRM
    MCTS -->|Refines with| TRM
    MCTS -->|Retrieves from| VectorStore

    MetaController -->|Routes to| HRM
    MetaController -->|Routes to| TRM
    MetaController -->|Routes to| MCTS

    API -->|Routes to| InferenceEngine
    InferenceEngine -->|Uses| MetaController
    InferenceEngine -->|Uses| HRM
    InferenceEngine -->|Uses| TRM
    InferenceEngine -->|Uses| MCTS

    Orchestrator -->|Logs to| WandB
    Monitor -->|Exports to| Prometheus
    Orchestrator -->|Saves checkpoints| S3
    ReplayBuffer -->|Persists to| S3

    style Orchestrator fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
    style DataGen fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
    style HRM fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style TRM fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style MCTS fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style MetaController fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style API fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
    style VectorStore fill:#F39C12,stroke:#B9770E,stroke-width:2px
```

### Container Descriptions

| Container | Technology | Responsibility |
|-----------|-----------|----------------|
| **Training Orchestrator** | Python, PyTorch, AsyncIO | Coordinates complete training pipeline, curriculum learning |
| **Synthetic Generator** | Python, LLM | Generates high-quality Q&A pairs for training |
| **Corpus Builder** | Python | Fetches arXiv papers and builds vector index |
| **HRM Agent** | PyTorch, DeBERTa | Hierarchical problem decomposition |
| **TRM Agent** | PyTorch, DeBERTa | Recursive solution refinement |
| **Neural MCTS** | Python, NumPy, PyTorch | Tree search with neural guidance |
| **Meta Controller** | PyTorch, GRU | Dynamic routing of queries to optimal agents |
| **Policy-Value Network** | PyTorch, ResNet | Predicts action probabilities and values |
| **FastAPI Server** | FastAPI, Uvicorn | REST API for inference |
| **Inference Engine** | PyTorch | Model inference and prediction |
| **Replay Buffer** | Python, NumPy | Stores and samples experiences |
| **Vector Store** | Pinecone | RAG knowledge base for retrieval |

---

## Level 3: Component Diagrams

### 3.1 Training Orchestrator Components

```mermaid
graph TB
    subgraph "Training Orchestrator Container"
        subgraph "Core Orchestration"
            OrcMain[UnifiedTrainingOrchestrator<br/><br/>Main coordinator]
            PhaseMgr[Phase Manager<br/><br/>Curriculum control]
            EvalMgr[Evaluation Manager<br/><br/>Model evaluation]
        end

        subgraph "Model Training"
            PVTrainer[PV Network Trainer<br/><br/>Policy-value training]
            HRMTrainer[HRM Trainer<br/><br/>Hierarchical reasoning]
            TRMTrainer[TRM Trainer<br/><br/>Recursive refinement]
            MetaTrainer[Meta Trainer<br/><br/>Router training]
        end

        subgraph "Optimizers"
            PVOptim[AdamW Optimizer]
            HRMOptim[AdamW Optimizer]
            TRMOptim[AdamW Optimizer]
            MetaOptim[AdamW Optimizer]
            LRScheduler[Linear Warmup]
        end

        subgraph "Data Pipeline"
            DataOrch[Data Orchestrator<br/><br/>Data loading]
            SelfPlay[Self-Play Generator<br/><br/>MCTS games]
            SyntheticGen[Synthetic Gen<br/><br/>LLM Q&A]
            RAGBuilder[RAG Builder<br/><br/>Vector indexing]
        end

        subgraph "Persistence"
            CheckpointMgr[Checkpoint Manager<br/><br/>Save/load models]
            ConfigMgr[Config Manager<br/><br/>System configuration]
            ModelIntegrator[Model Integrator<br/><br/>Deployment]
        end

        subgraph "Monitoring"
            PerfMon[Performance Monitor<br/><br/>Metrics tracking]
            WandBLogger[WandB Logger<br/><br/>Experiment logging]
        end
    end

    OrcMain -->|Manages| PhaseMgr
    OrcMain -->|Uses| EvalMgr

    PhaseMgr -->|Phase 1| RAGBuilder
    PhaseMgr -->|Phase 2| HRMTrainer
    PhaseMgr -->|Phase 3| TRMTrainer
    PhaseMgr -->|Phase 4| SelfPlay
    PhaseMgr -->|Phase 5| MetaTrainer

    SelfPlay -->|Feeds| PVTrainer
    DataOrch -->|Feeds| HRMTrainer
    DataOrch -->|Feeds| TRMTrainer
    SyntheticGen -->|Feeds| DataOrch

    HRMTrainer -->|Uses| HRMOptim
    TRMTrainer -->|Uses| TRMOptim
    MetaTrainer -->|Uses| MetaOptim

    OrcMain -->|Persists with| CheckpointMgr
    OrcMain -->|Deploys via| ModelIntegrator
    
    OrcMain -->|Monitors with| PerfMon
    PerfMon -->|Logs to| WandBLogger

    style OrcMain fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
```

### 3.2 Neural Network Components (DeepMind Implementation)

```mermaid
graph TB
    subgraph "Neural Components"
        subgraph "HRM (Hierarchical Reasoning)"
            HRMBase[DeBERTa Base<br/><br/>Pre-trained Transformer]
            HRMLoRA[LoRA Adapter<br/><br/>Efficient fine-tuning]
            DecompHead[Decomposition Head<br/><br/>Subtask generation]
            DepthHead[Depth Predictor<br/><br/>Recursion depth]
        end

        subgraph "TRM (Task Refinement)"
            TRMBase[DeBERTa Base<br/><br/>Pre-trained Transformer]
            TRMLoRA[LoRA Adapter<br/><br/>Efficient fine-tuning]
            RefineHead[Refinement Head<br/><br/>Response improvement]
            ScoreHead[Score Predictor<br/><br/>Quality estimation]
        end

        subgraph "Neural MCTS"
            StateEnc[State Encoder<br/><br/>Feature extraction]
            PolicyNet[Policy Head<br/><br/>Action probabilities]
            ValueNet[Value Head<br/><br/>Position evaluation]
        end

        subgraph "Meta Controller"
            RouterNet[Router Network<br/><br/>Agent selection]
            FeatureExt[Feature Extractor<br/><br/>Query analysis]
        end
    end

    HRMBase -->|Wrapped by| HRMLoRA
    HRMLoRA -->|Feeds| DecompHead
    HRMLoRA -->|Feeds| DepthHead

    TRMBase -->|Wrapped by| TRMLoRA
    TRMLoRA -->|Feeds| RefineHead
    TRMLoRA -->|Feeds| ScoreHead

    StateEnc -->|Shared| PolicyNet
    StateEnc -->|Shared| ValueNet

    FeatureExt -->|Feeds| RouterNet

    style HRMBase fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style TRMBase fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
```

### 3.3 Agent Harness — Producer-Reviewer Subsystem (Local-LLM Pipeline)

Sibling subsystem to the neural agents above. Wires the existing
`ProducerReviewerTopology` to **any** OpenAI-compatible local LLM
(LM Studio, vLLM, Ollama). Designed for sequential, latency-bounded
inference on a single local model — producer and reviewer share one
`LLMClient` to avoid concurrent VRAM pressure on memory-bound GPUs
such as the Tesla P40 (24 GB).

```mermaid
graph TB
    subgraph "demos / harness CLI"
        Demo[producer_reviewer_phi4_demo<br/><br/>argparse + env-driven defaults]
    end

    subgraph "Agent Harness — agents/"
        Producer[LLMProducerAgent<br/><br/>drafts response]
        Reviewer[LLMReviewerAgent<br/><br/>ACCEPT/REJECT + score]
        Prompts[prompts.py<br/><br/>provider-agnostic templates]
        Topology[ProducerReviewerTopology<br/><br/>max_rounds loop]
    end

    subgraph "Benchmark bridge"
        Adapter[BenchmarkTaskAdapter<br/><br/>BenchmarkTask -> harness Task]
        Template[benchmark_task_template.md<br/><br/>parametric markdown spec]
        Tasks[ALL_TASKS<br/><br/>A1 .. C3]
    end

    subgraph "LLM adapter layer"
        Factory[LLMClientFactory<br/><br/>provider switch]
        LMS[LMStudioClient<br/><br/>OpenAI-compatible HTTP]
        PresetReg[ModelPreset registry<br/><br/>regex match by model name]
        Applier[apply_preset<br/><br/>merge into payload]
    end

    subgraph "Configuration"
        Settings[Settings<br/><br/>LMSTUDIO_*]
        HSettings[HarnessSettings<br/><br/>HARNESS_PRODUCER_*<br/>HARNESS_REVIEWER_*<br/>HARNESS_BENCHMARK_TASK_ID]
    end

    subgraph "Local model (out of process)"
        LMStudio[LM Studio server<br/><br/>Phi-4 14B Q4_K_M]
    end

    Demo -->|--task A1| Adapter
    Adapter -->|lookup| Tasks
    Adapter -->|to_task / to_spec_text| Template
    Demo -->|create_llm| Factory
    Factory -->|auto-attach preset| PresetReg
    Factory -->|build| LMS
    LMS -->|apply_preset payload| Applier
    Applier -->|stop / temp / reasoning| LMS
    Demo -->|run_pipeline| Topology
    Topology -->|round 1..N| Producer
    Topology -->|review| Reviewer
    Producer -->|generate| LMS
    Reviewer -->|generate| LMS
    LMS -->|HTTP /chat/completions| LMStudio
    Producer -.->|uses| Prompts
    Reviewer -.->|uses| Prompts
    Settings --> Factory
    HSettings --> Demo
    HSettings --> Topology

    style Producer fill:#9B59B6,stroke:#6C3483,stroke-width:2px,color:#fff
    style Reviewer fill:#9B59B6,stroke:#6C3483,stroke-width:2px,color:#fff
    style Topology fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style PresetReg fill:#F39C12,stroke:#B9770E,stroke-width:2px
    style LMStudio fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
```

**Key invariants enforced by this subsystem:**

| Invariant | Where enforced |
|---|---|
| Provider-agnostic agents (no model-specific strings in `agents/`) | `prompts.py` test asserts no `"phi"` / `"openai"` / `"anthropic"` / `"lmstudio"` literal |
| Per-model decoding rules attach by **regex** match on model name | `ModelPreset.name_pattern`, `get_preset(model_name)` |
| Reviewer accept-gate cannot be bypassed by injection in the body | `LLMReviewerAgent.run` returns `response=marker` only; full body in `metadata["text"]` |
| Producer + reviewer never run concurrently | Topology runs serially; helpers share one `LLMClient` instance |
| All defaults are env-driven (no hardcoded values in `demos/` body) | `Settings`, `HarnessSettings`, `constants.DEFAULT_LMSTUDIO_TEMPERATURE` |

**New environment variables (all optional, defaults preserve current behaviour):**

| Variable | Default | Purpose |
|---|---|---|
| `LMSTUDIO_PRESET` | `None` | Override preset lookup by name |
| `LMSTUDIO_REASONING_EFFORT` | `None` | Reasoning hint (`low`/`medium`/`high`) |
| `LMSTUDIO_TEMPERATURE` | `None` | Default sampling temperature |
| `HARNESS_PRODUCER_REVIEWER_ROUNDS` | `3` | Loop cap |
| `HARNESS_PRODUCER_MAX_TOKENS` | `4000` | Producer draft budget |
| `HARNESS_REVIEWER_MAX_TOKENS` | `1500` | Reviewer review budget |
| `HARNESS_BENCHMARK_TASK_ID` | `None` | Default benchmark task id for the demo |

---

## Deployment Architecture

### Docker & Production Deployment

```mermaid
graph TB
    subgraph "Docker Environment"
        subgraph "Training Container"
            TrainApp[Training CLI]
            GPU[GPU Runtime]
            VolCache[Cache Volume]
            VolCheck[Checkpoint Volume]
        end

        subgraph "Inference Container"
            APIApp[FastAPI App]
            ProdModels[Production Models]
            ModelLoader[Model Loader]
        end
    end

    subgraph "Host Infrastructure"
        NVIDIA[NVIDIA Driver]
        DockerEngine[Docker Engine]
        FileSystem[File System]
    end

    TrainApp -->|Uses| GPU
    TrainApp -->|Writes| VolCache
    TrainApp -->|Saves| VolCheck

    APIApp -->|Uses| ModelLoader
    ModelLoader -->|Loads| ProdModels
    ProdModels -->|Mapped from| VolCheck

    DockerEngine -->|Manages| TrainApp
    DockerEngine -->|Manages| APIApp
    NVIDIA -->|Powers| GPU

    style TrainApp fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
    style APIApp fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
```

---

## Summary

This updated C4 architecture reflects the **current state** of the application, incorporating:

1.  **Neural Networks**: Explicit integration of HRM, TRM, MCTS, and Meta-Controller models with LoRA adapters.
2.  **Training Pipeline**: Comprehensive orchestration including synthetic data generation and corpus building.
3.  **RAG Integration**: Pinecone vector database for retrieval-augmented generation.
4.  **Docker Deployment**: Containerized training and inference workflows.
5.  **External Services**: Integration with W&B, S3, and ArXiv.

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Core ML** | PyTorch 2.1+, Transformers, PEFT, NumPy |
| **Models** | DeBERTa-v3, ResNet, GRU |
| **Local LLM** | LM Studio (OpenAI-compatible HTTP), Phi-4 14B reasoning preset |
| **Orchestration** | Python AsyncIO, LangGraph, Producer-Reviewer harness topology |
| **Data** | Pinecone, ArXiv API, OpenAI API, Anthropic API, LM Studio |
| **Monitoring** | Weights & Biases, Prometheus, structured stdlib logging |
| **Deployment** | Docker, Docker Compose |
