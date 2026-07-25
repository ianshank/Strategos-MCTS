# Google ADK Integration for LangGraph Multi-Agent MCTS Framework

This integration brings Google's Agent Development Kit (ADK) specialized agents into your LangGraph-based multi-agent MCTS framework, enabling powerful hybrid workflows that combine custom HRM/TRM agents with Google's state-of-the-art agent capabilities.

## Overview

The Google ADK integration provides five specialized agents:

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **ML Engineering** | Model training & optimization | SOTA model search, code refinement, ensemble strategies |
| **Data Science** | Data analysis & ML | NL2SQL, BigQuery integration, BQML, visualization |
| **Academic Research** | Research synthesis | Paper analysis, citation discovery, future directions |
| **Data Engineering** | Data pipeline development | Dataform pipelines, SQLx generation, troubleshooting |
| **Deep Search** | Production research | Human-in-the-loop planning, comprehensive reports |

## Quick Start

### Installation

```bash
# Install with Google ADK dependencies
pip install -e ".[google-adk]"
```

### Basic Configuration

```python
from src.integrations.google_adk import MLEngineeringAgent, ADKConfig, ADKBackend

# Create configuration
config = ADKConfig(
    backend=ADKBackend.LOCAL,  # or VERTEX_AI for production
    model_name="gemini-2.0-flash-001",
    workspace_dir="./workspace/adk",
    enable_search=True,
)

# Initialize agent
agent = MLEngineeringAgent(config)
await agent.initialize()

# Use agent
response = await agent.train_model(
    task_name="my_model",
    data_path="./data/training.csv",
    task_type="Tabular Regression",
    query="Train a model to predict customer lifetime value",
)

print(response.result)
```

### Environment Setup

1. **Copy environment template:**
   ```bash
   cp src/integrations/google_adk/config/.env.example .env
   ```

2. **Configure credentials (for Vertex AI):**
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Set environment variables in `.env`:**
   ```bash
   GOOGLE_CLOUD_PROJECT=your-project-id
   ADK_BACKEND=vertex_ai
   ROOT_AGENT_MODEL=gemini-2.0-flash-001
   ```

## Architecture

### Integration Design

```
┌─────────────────────────────────────────────────┐
│         LangGraph Multi-Agent Framework         │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   HRM    │  │   TRM    │  │  MCTS    │     │
│  │  Agent   │  │  Agent   │  │  Search  │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │             │             │            │
│       └─────────────┴─────────────┘            │
│                     │                          │
│            ┌────────▼────────┐                 │
│            │  ADK Adapter    │                 │
│            │  (Base Layer)   │                 │
│            └────────┬────────┘                 │
│                     │                          │
│       ┌─────────────┼─────────────┐            │
│       │             │             │            │
│  ┌────▼───┐   ┌────▼───┐   ┌─────▼──┐        │
│  │   ML   │   │  Data  │   │Academic│        │
│  │  Eng   │   │Science │   │Research│  ...   │
│  └────────┘   └────────┘   └────────┘        │
│                                                 │
└─────────────────────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │  Google ADK & Gemini    │
         │  (Vertex AI / ML Dev)   │
         └─────────────────────────┘
```

### Key Components

1. **ADKAgentAdapter** (`base.py`)
   - Base class for all ADK agents
   - Handles initialization, configuration, and execution
   - Manages authentication and environment setup
   - Provides common interface for LangGraph integration

2. **Specialized Agents** (`agents/`)
   - Each agent wraps specific Google ADK functionality
   - Implements domain-specific methods and workflows
   - Provides structured request/response interfaces

3. **Configuration** (`config/`)
   - Environment variable management
   - Backend selection (Local, ML Dev, Vertex AI)
   - Authentication and credentials

## Use Cases

### 1. Enhanced Model Training

Combine your HRM agent's hierarchical reasoning with ML Engineering agent's SOTA model search:

```python
from src.agents.hrm_agent import create_hrm_agent
from src.integrations.google_adk import MLEngineeringAgent

# Decompose problem with HRM
hrm = create_hrm_agent(hrm_config)
subproblems = await hrm.decompose_problem(query, state)

# Use ML Engineering for model training subproblems
ml_agent = MLEngineeringAgent(adk_config)
for sp in subproblems:
    if "model" in sp.description.lower():
        result = await ml_agent.train_model(...)
```

### 2. Research-Driven Development

Build training corpus from latest academic research:

```python
from src.integrations.google_adk import AcademicResearchAgent
from training.research_corpus_builder import ResearchCorpusBuilder

# Find latest research on MCTS
research_agent = AcademicResearchAgent(config)
citations = await research_agent.find_citations("Monte Carlo Tree Search")

# Build training corpus
corpus_builder = ResearchCorpusBuilder()
corpus_builder.add_research_papers(citations.artifacts)
```

### 3. Data Pipeline Automation

Automate training data pipeline development:

```python
from src.integrations.google_adk import DataEngineeringAgent

de_agent = DataEngineeringAgent(config)

# Design pipeline for MCTS trajectory data
pipeline = await de_agent.design_pipeline(
    query="Build ETL pipeline for agent training data",
    pipeline_name="mcts_training_etl",
    source_tables=["raw_trajectories", "agent_states", "rewards"],
    target_table="training_dataset",
)
```

### 4. Performance Analysis

Analyze agent performance with Data Science agent:

```python
from src.integrations.google_adk import DataScienceAgent

ds_agent = DataScienceAgent(config)

# Analyze what makes agents successful
analysis = await ds_agent.analyze_data(
    query="Identify factors correlating with high MCTS success rates",
    data_source="agent_performance_metrics.csv",
)

# Query training database
metrics = await ds_agent.query_database(
    nl_query="Show me agents with best improvement trajectory over last month",
    dataset_name="training_metrics",
)
```

## Agent Details

### ML Engineering Agent

**Capabilities:**
- SOTA model search via web
- Iterative code refinement
- Ensemble strategies
- Debugging & data leakage detection

**Example:**
```python
response = await ml_agent.train_model(
    task_name="housing_prices",
    data_path="./data/housing.csv",
    task_type="Tabular Regression",
    metric="rmse",
    lower_is_better=True,
)
```

### Data Science Agent

**Capabilities:**
- NL2SQL (BigQuery/AlloyDB)
- Python data analysis
- BigQuery ML training
- Cross-dataset operations

**Example:**
```python
# Natural language to SQL
sql = await ds_agent.query_database(
    nl_query="Top customers by revenue last quarter",
    dataset_name="sales",
)

# Train BQML model
model = await ds_agent.train_bqml_model(
    query="Forecast monthly sales",
    model_type="arima",
    target_column="revenue",
)
```

### Academic Research Agent

**Capabilities:**
- Paper analysis (PDF/URL)
- Citation discovery (2023+)
- Future research directions
- Research synthesis

**Example:**
```python
# Analyze paper
analysis = await research_agent.analyze_paper(
    paper_title="Attention Is All You Need",
    query="Analyze transformer architecture impact",
)

# Find citations
citations = await research_agent.find_citations(
    paper_title="BERT: Pre-training of Deep Bidirectional Transformers"
)
```

### Data Engineering Agent

**Capabilities:**
- Dataform pipeline design
- SQLx file generation
- Pipeline troubleshooting
- Performance optimization

**Example:**
```python
# Design pipeline
pipeline = await de_agent.design_pipeline(
    query="Transform raw event data to analytics tables",
    pipeline_name="events_etl",
    source_tables=["raw_events"],
)

# Generate SQLx
sqlx = await de_agent.generate_sqlx(
    query="Aggregate daily metrics",
    table_name="daily_metrics",
)
```

### Deep Search Agent

**Capabilities:**
- Human-in-the-loop planning
- Autonomous research execution
- Iterative refinement
- Comprehensive reports with citations

**Example:**
```python
# Two-phase workflow
plan = await deep_search.create_research_plan(
    topic="Monte Carlo Tree Search optimization techniques"
)
# User reviews and approves plan

report = await deep_search.execute_research(
    research_id=plan.session_id
)
```

## Configuration Reference

### Backend Options

| Backend | Use Case | Requirements |
|---------|----------|--------------|
| `LOCAL` | Testing without Google Cloud | None |
| `ML_DEV` | Local dev with Gemini API | Google Cloud project, API key |
| `VERTEX_AI` | Production with full features | Vertex AI enabled, credentials |

### Environment Variables

See `config/.env.example` for complete list. Key variables:

```bash
# Core
GOOGLE_CLOUD_PROJECT=your-project-id
ADK_BACKEND=vertex_ai
ROOT_AGENT_MODEL=gemini-2.0-flash-001

# Features
ADK_ENABLE_SEARCH=true
ADK_ENABLE_TRACING=true
ADK_WORKSPACE_DIR=./workspace/adk

# Agent-specific
BIGQUERY_DATASET_ID=analytics
DATAFORM_REPOSITORY_NAME=pipelines
```

## API Reference

### Common Interface

All agents inherit from `ADKAgentAdapter`:

```python
class ADKAgentAdapter:
    async def initialize() -> None
    async def invoke(request: ADKAgentRequest) -> ADKAgentResponse
    async def cleanup() -> None
    def get_capabilities() -> dict[str, Any]
```

### Request/Response

```python
@dataclass
class ADKAgentRequest:
    query: str                          # User query
    context: dict[str, Any]            # Additional context
    session_id: Optional[str]          # Session tracking
    parameters: dict[str, Any]         # Agent-specific params

@dataclass
class ADKAgentResponse:
    result: str                         # Agent output
    metadata: dict[str, Any]           # Execution metadata
    artifacts: list[str]               # Generated files
    status: str                        # success/error
    error: Optional[str]               # Error message
    session_id: Optional[str]          # Session ID
```

## Examples

See the repo-level `examples/google_adk/` directory for comprehensive examples:

- `example_ml_engineering.py` - Model training workflows
- `example_data_science.py` - Data analysis and BQML
- `example_academic_research.py` - Paper analysis and corpus building

Run examples:
```bash
python -m examples.google_adk.example_ml_engineering
```

## Testing

```bash
# Run integration tests
pytest src/integrations/google_adk/tests/

# Run specific agent tests
pytest src/integrations/google_adk/tests/test_ml_engineering.py
```

## Troubleshooting

### Common Issues

1. **Import errors:**
   ```bash
   pip install -e ".[google-adk]"
   ```

2. **Authentication errors:**
   ```bash
   gcloud auth application-default login
   ```

3. **API not enabled:**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

See `config/README.md` for detailed troubleshooting.

## Contributing

To extend the integration:

1. Create new agent in `agents/`
2. Inherit from `ADKAgentAdapter`
3. Implement required methods
4. Add examples and tests
5. Update documentation

## Resources

- [Google ADK Documentation](https://cloud.google.com/vertex-ai/docs/agent-development-kit)
- [ADK Samples Repository](https://github.com/google/adk-samples)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Configuration Guide](config/README.md)
- [Examples](../../../examples/google_adk/README.md)

## License

This integration follows the same MIT license as the main project.
