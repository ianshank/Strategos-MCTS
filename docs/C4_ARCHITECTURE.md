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
| **Neural MCTS** | Python, NumPy, PyTorch | AlphaZero-style tree search with neural guidance |
| **Parallel MCTS** | Python, AsyncIO | Tree-parallel search with virtual-loss collision avoidance |
| **Meta Controller** | PyTorch, GRU/BERT | Dynamic routing of queries to optimal agents |
| **Assembly Router** | Python, NLP | Feature extraction (`ConceptExtractor`) → HRM/TRM/MCTS routing heuristics |
| **Policy-Value Network** | PyTorch, ResNet | Predicts action probabilities and values for square & rectangular boards (`(C, H, W)`) |
| **Gameplay Domains** | Python, NumPy, PyTorch | Multi-domain environment suite (`chess`, `connect_four`, `othello`, reasoning, planning) |
| **GPU Introspection** | Python, PyTorch | Memory tracking (`GPUMemoryTracker`), pre-flight checks, CUDA fraction clamping |
| **Training Profiles** | Python | Standardized operational presets (`smoke`, `dev`, `full`) for MCTS self-play training |
| **FastAPI Server** | FastAPI, Uvicorn | REST API for inference (streaming, graph viz, comparison) |
| **Inference Engine** | PyTorch | Model inference and prediction |
| **Replay Buffer** | Python, NumPy | Stores and samples experiences (torch-safe checkpoints) |
| **Vector Store** | Pinecone | RAG knowledge base for retrieval |
| **Prometheus Metrics** | prometheus-client | Counters/histograms for agents, MCTS, LLM calls; `/metrics` endpoint |


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

### 3.5 Self-Play Training & Gameplay Domains

```mermaid
graph TB
    subgraph "Self-Play Training Component"
        CLI[SelfPlayConvergence<br/><br/>CLI entry point]
        Trainer[SelfPlayTrainer<br/><br/>Training loop]
        Profiles[TrainingProfiles<br/><br/>smoke/dev/full]
        SysConfig[SystemConfig<br/><br/>device/AMP/compile]
        Registry[DomainRegistry<br/><br/>Domain dispatch]
        
        subgraph "Game Domains"
            Chess[Chess]
            ConnectFour[ConnectFour]
            Othello[Othello]
            Reasoning[Reasoning]
            Planning[Planning]
        end
        
        PVNet[PolicyValueNetwork<br/><br/>MLP for reasoning/planning, ResNet for board games]
        GPUUtils[GPUUtils<br/><br/>Memory tracking, pre-flight checks]
    end

    CLI -->|Configures via| Profiles
    CLI -->|Configures via| SysConfig
    CLI -->|Runs| Trainer
    Trainer -->|Dispatches via| Registry
    Registry --> Chess
    Registry --> ConnectFour
    Registry --> Othello
    Registry --> Reasoning
    Registry --> Planning
    Trainer -->|Trains| PVNet
    Trainer -->|Monitors via| GPUUtils
    
    style CLI fill:#E74C3C,stroke:#922B21,stroke-width:2px,color:#fff
    style Trainer fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style PVNet fill:#8E44AD,stroke:#5B2C6F,stroke-width:2px,color:#fff
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
            AssemblyRouter[AssemblyRouter<br/><br/>HRM / TRM / MCTS heuristics]
            ConceptExtractor[ConceptExtractor<br/><br/>NLP concept & complexity scoring]
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

    FeatureExt --> RouterNet
    ConceptExtractor --> AssemblyRouter
    AssemblyRouter --> RouterNet

    style HRMBase fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style TRMBase fill:#3498DB,stroke:#1F618D,stroke-width:2px,color:#fff
    style AssemblyRouter fill:#8E44AD,stroke:#5B2C6F,stroke-width:2px,color:#fff
    style ConceptExtractor fill:#8E44AD,stroke:#5B2C6F,stroke-width:2px,color:#fff
```

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

### Authentication & Secrets

- **API authentication** is configuration-selected via `AUTH_MODE` (`src/config/settings.py`):
  `api_key` (default — `X-API-Key` validated by `APIKeyAuthenticator`) or `jwt` (validated by
  `JWTAuthenticator` from `JWT_SECRET`/`JWT_ALGORITHM`/`JWT_EXPIRY_HOURS`). The selection lives in
  `rest_server.verify_api_key`; the API-key path is unchanged by default and `JWT_SECRET` is required
  at startup when `AUTH_MODE=jwt`.
- **Secrets** are never stored in the repo or image. In Kubernetes the External Secrets Operator
  materializes the `llm-secrets` Secret from an external store at runtime (`kubernetes/deployment.yaml`
  `ExternalSecret`); locally they resolve via env → Pydantic `SecretStr`. See
  `docs/SECRETS_MANAGEMENT.md`. A CI `spec-validate` job also greps for committed key material.

### Serving, streaming & visualization (Phase 4)

Framework capabilities are exposed through thin REST endpoints that delegate to coverage-bearing
service modules (so `rest_server.py`, which is omitted from coverage, stays logic-free):

- `src/api/streaming.py` (`StreamingService`) → `POST /query-stream` (SSE over LangGraph
  `astream_events`).
- `src/api/graph_service.py` (`GraphService`) → `GET /graph/structure|mermaid`, `POST /graph/render`
  (Kroki).
- `src/api/comparison_service.py` (`ComparisonService`) → `POST /compare` (MCTS vs single-shot); also
  drives `demo.py --compare` and the Gradio UI (`app.py`, `[ui]` extra). All gated by `ENABLE_*` flags.

### Neural self-play & multi-domain learning (M5 / Phase 5)

```mermaid
graph LR
    Registry["DomainRegistry<br/>reasoning / planning (single-agent)<br/>chess / connect_four / othello (adversarial)<br/>lazy-loaded game domains"] --> Trainer["SelfPlayTrainer<br/>single_agent flag"]
    GameRegs["game registrations<br/>chess / connect_four / othello<br/>lazy-loaded via DomainRegistry"] -.-> Registry
    Trainer --> NMCTS[NeuralMCTS<br/>+ SelfPlayCollector]
    NMCTS --> Buffer[ExperienceBuffer<br/>torch-safe]
    Buffer --> Loss[AlphaZeroLoss]
    Loss --> Net[PolicyValueNetwork]
    Net --> NMCTS
    Trainer --> Bench[policy_comparison<br/>lift + CI]
    Stats[utils/stats<br/>Wilson / diff CIs] --> Bench
    Stats --> Eval[EvaluationService]
    Bench --> CLI[policy_lift CLI<br/>exit-code gate + JSON artifact]
    MCtrl[MetaControllerDataCollector] --> MCtrlTrain[train_and_validate]
    style Trainer fill:#8E44AD,stroke:#5B2C6F,stroke-width:2px,color:#fff
    style Bench fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
    style CLI fill:#2ECC71,stroke:#1E8449,stroke-width:2px,color:#fff
```

- `SelfPlayTrainer` (`src/training/self_play_trainer.py`) composes `NeuralMCTS` + `ExperienceBuffer` +
  `AlphaZeroLoss`; `single_agent=True` bypasses the two-player negamax assumptions for non-adversarial
  domains. Domains are selected via `DomainRegistry` (`src/framework/domain_registry.py`); dict-action
  states are made hashable by `single_agent_domains.StringActionGameState`.
  `save_checkpoint(..., metadata=...)` optionally writes a `<checkpoint>.meta.json` architecture
  sidecar so tools can rebuild the network without guessing.
- The built-in **reasoning/planning domains are synthetic smoke tests** (gameable rewards — they
  validate plumbing, not decision quality). **Chess is the adversarial M5 domain**, registered lazily
  on first `DomainRegistry.get("chess")` via `src/games/chess/registration.py` behind the optional
  `chess` extra (a clean no-op without `python-chess`); a dedicated `chess-tests` CI job installs the
  extra and runs the chess subset without touching the coverage gate.
- Decision-quality lift is measured by `src/benchmark/policy_comparison.py` (mean-reward for
  single-agent, win-rate for adversarial) with confidence intervals from the shared scipy-free
  `src/utils/stats.py` (Wilson score for win-rate; difference-of-means for rewards — also used by
  `EvaluationService`). The M5 gate is **fail-closed on the CI lower bound** (`meets_target`; the
  point estimate is reporting-only via `point_meets_target`) and is runnable end-to-end with
  `python -m src.benchmark.policy_lift` (`policy-lift` console script): JSON artifact + exit code
  0 (gate met) / 1 (not met) / 2 (error). The meta-controller learning loop lives in
  `src/training/meta_controller_data_collector.py` (`docs/META_CONTROLLER_TRAINING.md`).

---

## Summary

This updated C4 architecture reflects the **current state** of the application, incorporating:

1.  **Neural Networks**: Explicit integration of HRM, TRM, MCTS, and Meta-Controller models with LoRA adapters.
2.  **Training Pipeline**: Comprehensive orchestration including synthetic data generation and corpus building.
3.  **RAG Integration**: Pinecone vector database for retrieval-augmented generation.
4.  **Docker Deployment**: Containerized training and inference workflows.
5.  **External Services**: Integration with W&B, S3, and ArXiv.
6.  **Parallel MCTS**: Tree-parallel search engine with virtual-loss collision avoidance and adaptive scaling (`src/framework/mcts/parallel_mcts.py`).
7.  **Assembly Router & Concept Extractor**: NLP-driven routing heuristics — `ConceptExtractor` classifies query concepts into `technical_term`, `domain_entity`, or `process_action` with a complexity score; `AssemblyRouter` maps these features to HRM/TRM/MCTS (`src/framework/assembly/`, `src/agents/meta_controller/assembly_router.py`).
8.  **Prometheus Observability**: Full counter/histogram instrumentation for agent latency, MCTS iterations, LLM call outcomes, and active operations (`src/monitoring/prometheus_metrics.py`; `/metrics` endpoint via `rest_server.py`).
9.  **Test hardening (2026-07-20)**: 10 101 tests passing at 93.82% branch coverage; `ruff`, `black`, and `mypy` all clean across 305 source files.

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Core ML** | PyTorch 2.1+, Transformers, PEFT, NumPy |
| **Models** | DeBERTa-v3, ResNet, GRU |
| **Orchestration** | Python AsyncIO, LangGraph |
| **Data** | Pinecone, ArXiv API, OpenAI API |
| **Monitoring** | Weights & Biases, Prometheus, OpenTelemetry |
| **Deployment** | Docker, Docker Compose |
| **LLM adapters** | Provider-agnostic clients (OpenAI, Anthropic, LM Studio) over a shared resilience layer (`src/adapters/llm/resilience.py` — `CircuitBreaker`) with tenacity retries |

> **Cross-cutting:** LLM client resilience (circuit breaker + exponential-backoff retries)
> lives in `src/adapters/llm/resilience.py` and is shared by all provider clients, rather
> than duplicated per provider.

### CI/CD & Quality Gates

The `.github/workflows/ci.yml` pipeline includes the local `quality-gate` skill's checks, run with the
same pinned tool versions (the lint job installs the `[dev]` extra; the type-check job installs
`[dev,neural]`, which includes it): `black --check` → `ruff check` → `mypy src/` (strictness comes from
`[tool.mypy]` in `pyproject.toml`, not a `--strict` flag; CI adds `--no-error-summary`) → `pytest` with
branch coverage (`--cov-fail-under=85`, **achieved 93.82%** as of 2026-07-20) → a hardcoded-secret grep.
On top of that gate, CI-only jobs add `bandit` (HIGH-severity gate), `pip-audit` (CRITICAL gate), spec
validation (`harness validate-spec` — error-level against spec schema v2 via
`src/framework/harness/intent/spec_validator.py`: required `id`/`goal`/`status` frontmatter with a closed
status lifecycle, authored `AC-n` criterion IDs, filename↔id and duplicate-id checks across `specs/`, and
a no-changelog rule rejecting inline done-markers; one multi-path invocation so cross-file rules fire —
plus, on PRs, `harness spec-trace` traceability: `src/**` diffs need a `spec/<id>` branch approved on the
base branch or a `No-Spec: <reason>` commit trailer, and `verified` flips need same-line spec-id+`AC-n`
mappings under `tests/`; the `spec-validate` job now gates the CI `summary` aggregate on failure), and a
Docker build with a Trivy image scan whose SARIF results upload to GitHub code scanning (the `docker-build`
job carries `security-events: write`; the upload is advisory and non-blocking). Configuration is
centralized in `src/config/constants.py` + `src/config/settings.py` (Pydantic Settings) with domain-specific
constant modules; there are no hardcoded secrets or magic numbers in the routing/adapter layers.

**Current gate status (2026-07-20):**
| Gate | Status |
|---|---|
| `ruff check src/ tests/` | ✅ Clean (0 issues) |
| `black src/ tests/ --check --line-length 120` | ✅ Clean |
| `mypy src/` | ✅ Clean — 0 errors in 305 source files |
| `pytest tests/ -m "not slow" --cov=src` | ✅ 10 101 passed, 43 skipped, **93.82%** coverage |

> **Planned (not yet built):** the Phase 2 pilot (first spec driven through the full lifecycle;
> gate flips warn→block on exit) and Phase 3 plugin packaging/extraction into
> `claude-code-foundry` — see `docs/plans/SDD_PLUGIN_EXTRACTION_PLAN.md`. (Phases 0–1 — schema,
> error-level validator, `.claude/` enforcement surfaces, and CI traceability — are built; the
> session gate runs in warn mode during the pilot.)
