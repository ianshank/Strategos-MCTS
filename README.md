# LangGraph Multi-Agent MCTS Framework

**Production-Ready DeepMind-Style AI System with Neural MCTS and Hierarchical Reasoning**

![Architecture](docs/img/architecture_overview.png)

This framework implements a state-of-the-art multi-agent system combining hierarchical reasoning (HRM), iterative refinement (TRM), and Monte Carlo Tree Search (MCTS) guided by neural networks. It features a complete training pipeline, synthetic data generation, and RAG integration.

## 🚀 Key Features

### 🧠 Core Architecture
- **HRM (Hierarchical Reasoning Module)**: DeBERTa-based agent for complex problem decomposition.
- **TRM (Task Refinement Module)**: Iterative agent for refining and optimizing solutions.
- **Neural MCTS**: AlphaZero-style tree search guided by policy/value networks.
- **Meta-Controller**: Neural router (GRU/BERT) that dynamically assigns tasks to the best agent.

### 🛠️ Training Pipeline
- **End-to-End Orchestration**: Automated multi-stage training (Pre-training → Fine-tuning → Self-Play).
- **Synthetic Data Generation**: LLM-powered generator for creating high-quality training datasets.
- **Research Corpus Builder**: Automated fetching and indexing of arXiv papers for RAG.
- **Docker Support**: Fully containerized training and inference environments.

### 📊 Observability & RAG
- **RAG Integration**: Pinecone vector database for retrieving domain knowledge.
- **Experiment Tracking**: Full integration with Weights & Biases.
- **Production Monitoring**: Prometheus/Grafana metrics for latency, memory, and model performance.

### 🧪 Agent Harness — Local-LLM Reasoning Pipeline
- **Producer-Reviewer Topology**: Sequential draft-then-critique loop wired to any
  OpenAI-compatible local server (LM Studio / vLLM / Ollama).
- **Provider-Agnostic Agents**: `LLMProducerAgent` and `LLMReviewerAgent` work with
  any `LLMClient`; per-model decoding rules (stop tokens, reasoning hint, default
  temperature) attach via a regex-matched **model preset registry** — no model
  literals in agent code.
- **Built-in Phi-4 Preset**: `PHI4_REASONING` matches `phi-4` model identifiers
  and ships sane defaults for Tesla P40 (24 GB) class hardware.
- **Benchmark Bridge**: `BenchmarkTaskAdapter` converts existing `BenchmarkTask`
  fixtures (A1–C3) directly into harness `Task` objects without content
  duplication.

## 📦 Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for containerized workflow)
- NVIDIA GPU (recommended for training)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ianshank/langgraph_multi_agent_mcts.git
   cd langgraph_multi_agent_mcts
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (OpenAI, Pinecone, W&B)
   ```

3. **Run with Docker (Recommended):**
   ```bash
   # Run the demo pipeline (builds image, generates data, trains models)
   bash scripts/run_docker_training.sh
   ```

## 🏗️ Training Workflow

The framework supports a comprehensive training lifecycle:

1.  **Data Generation**:
    ```bash
    # Generate synthetic Q&A pairs
    python -m scripts.generate_synthetic_training_data --num-samples 1000
    ```

2.  **Corpus Building**:
    ```bash
    # Fetch and index arXiv papers
    python -m training.examples.build_arxiv_corpus --mode keywords --max-papers 200
    ```

3.  **Production Training**:
    ```bash
    # Run full training pipeline
    bash scripts/run_production_training.sh
    ```

## 🤖 Local-LLM Reasoning Demo (LM Studio + Phi-4)

Run a producer-reviewer pipeline against an LM Studio model on any benchmark task.
Defaults are env-driven; nothing is hardcoded.

```bash
# Point the framework at your local LM Studio instance.
export LLM_PROVIDER=lmstudio
export LMSTUDIO_BASE_URL=http://localhost:1234/v1
export LMSTUDIO_MODEL="phi-4-reasoning-q4_k_m"   # auto-attaches PHI4_REASONING preset
export LMSTUDIO_REASONING_EFFORT=medium          # optional: low | medium | high

# Run task A1 (code review) with three review rounds.
python -m demos.producer_reviewer_phi4_demo --task A1 --rounds 3

# Or take the task from an env var instead of a flag:
HARNESS_BENCHMARK_TASK_ID=A2 python -m demos.producer_reviewer_phi4_demo --json
```

**New environment variables (all optional; defaults preserve existing behaviour):**

| Variable | Default | Purpose |
|---|---|---|
| `LMSTUDIO_PRESET` | `None` | Override preset lookup by name |
| `LMSTUDIO_REASONING_EFFORT` | `None` | Reasoning hint (`low` / `medium` / `high`) |
| `LMSTUDIO_TEMPERATURE` | `None` | Default sampling temperature |
| `HARNESS_PRODUCER_REVIEWER_ROUNDS` | `3` | Producer-reviewer iteration cap |
| `HARNESS_PRODUCER_MAX_TOKENS` | `4000` | Producer draft token budget |
| `HARNESS_REVIEWER_MAX_TOKENS` | `1500` | Reviewer review token budget |
| `HARNESS_BENCHMARK_TASK_ID` | `None` | Default benchmark task id for the demo |

See [`docs/C4_ARCHITECTURE.md`](docs/C4_ARCHITECTURE.md) → **Agent Harness — Producer-Reviewer Subsystem** for the component diagram.

## 🧪 Testing

Run the comprehensive test suite to verify system integrity:

```bash
# Run all tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run specific deployed model tests
pytest tests/integration/test_deployed_models.py

# Run only the harness / agent / adapter unit tests touched by the local-LLM demo
pytest tests/unit/adapters/llm/ tests/unit/framework/harness/agents/ \
       tests/unit/benchmark/test_harness_bridge.py tests/unit/demos/ \
       tests/integration/test_producer_reviewer_phi4_demo.py
```

## 📚 Documentation

- **[Architecture Guide](docs/C4_ARCHITECTURE.md)**: Detailed C4 diagrams of system components.
- **[Training Guide](docs/LOCAL_TRAINING_GUIDE.md)**: How to train models locally or in the cloud.
- **[Synthetic Data](training/SYNTHETIC_DATA_GENERATION_GUIDE.md)**: Guide to generating training data.
- **[Changelog](CHANGELOG.md)**: Release history including the producer-reviewer / LM Studio addition.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📜 License

MIT License - see the [LICENSE](LICENSE) file for details.
