# LangGraph Multi-Agent MCTS Framework

**Production-quality components for a DeepMind-style AI system with Neural MCTS and Hierarchical Reasoning** (pre-integration — see [Known Limitations](#-known-limitations))

![Architecture](docs/img/architecture_overview.png)

This framework implements a multi-agent system combining hierarchical reasoning (HRM), iterative refinement (TRM), and Monte Carlo Tree Search (MCTS) guided by neural networks. It features a training pipeline, synthetic data generation, and RAG integration. The individual components are well-tested and engineered to a production standard; full end-to-end integration is still in progress.

## ⚠️ Known Limitations

- **Mock/lightweight fallbacks are opt-in.** When the configured LLM or the full integrated
  framework can't initialize, the service fails loud by default rather than silently serving
  mock output. Enable `ALLOW_MOCK_LLM_FALLBACK` / `ALLOW_LIGHTWEIGHT_FRAMEWORK_FALLBACK` for
  tests/dev (see `.env.example`).
- **Persisted artifacts changed format.** The substructure library now persists as JSON and
  the experience buffer via `torch.save(weights_only=True)`. Legacy `pickle` files are only
  read when `ASSEMBLY_TRUST_LEGACY_PICKLE` / `TRAINING_TRUST_LEGACY_PICKLE` are set, then
  migrated in place. See [CHANGELOG.md](CHANGELOG.md).
- **Some training and hybrid-agent paths are extension points.** Domain-specific prompt/parse
  logic and certain training loops ship as overridable defaults, not finished implementations.
- See `GAP_ANALYSIS_REPORT.md` for the current component-by-component status.

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

## 🧪 Testing

Run the comprehensive test suite to verify system integrity:

```bash
# Run all tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run specific deployed model tests
pytest tests/integration/test_deployed_models.py
```

## 📚 Documentation

- **[Architecture Guide](docs/C4_ARCHITECTURE.md)**: Detailed C4 diagrams of system components.
- **[Training Guide](docs/LOCAL_TRAINING_GUIDE.md)**: How to train models locally or in the cloud.
- **[Synthetic Data](training/SYNTHETIC_DATA_GENERATION_GUIDE.md)**: Guide to generating training data.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📜 License

MIT License - see the [LICENSE](LICENSE) file for details.
