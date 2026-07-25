  # Google ADK Integration Examples

This directory contains comprehensive examples demonstrating how to use Google ADK agents
within your LangGraph multi-agent MCTS framework.

## Examples Overview

### 1. ML Engineering Agent (`example_ml_engineering.py`)

Demonstrates model training, code refinement, and integration with your training pipeline:

```bash
python -m src.integrations.google_adk.examples.example_ml_engineering
```

**Use Cases:**
- Train state-of-the-art ML models for your agents
- Refine model components through ablation studies
- Optimize neural architectures for HRM/TRM agents
- Automate hyperparameter tuning

### 2. Data Science Agent (`example_data_science.py`)

Shows NL2SQL, data analysis, and BigQuery ML capabilities:

```bash
python -m src.integrations.google_adk.examples.example_data_science
```

**Use Cases:**
- Analyze agent performance data
- Query training metrics databases
- Build forecasting models for agent behavior
- Perform exploratory data analysis on MCTS results

### 3. Academic Research Agent (`example_academic_research.py`)

Demonstrates paper analysis, citation discovery, and corpus building:

```bash
python -m src.integrations.google_adk.examples.example_academic_research
```

**Use Cases:**
- Build training corpus from latest research papers
- Keep agents updated with state-of-the-art techniques
- Discover new MCTS algorithms and optimizations
- Track developments in hierarchical reasoning

### 4. Data Engineering Agent (`example_data_engineering.py`)

Shows Dataform pipeline development and data quality management:

```bash
python -m src.integrations.google_adk.examples.example_data_engineering
```

**Use Cases:**
- Build data pipelines for agent training data
- Transform and clean MCTS trajectory data
- Manage training dataset schemas
- Ensure data quality for agent development

### 5. Deep Search Agent (`example_deep_search.py`)

Demonstrates production-ready research with human-in-the-loop:

```bash
python -m src.integrations.google_adk.examples.example_deep_search
```

**Use Cases:**
- Research best practices for agent architectures
- Deep dive into MCTS optimization techniques
- Compile comprehensive reports on ML frameworks
- Investigate specific agent behavior patterns

## Running Examples

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -e ".[google-adk]"
   ```

2. **Configure environment:**
   ```bash
   cp src/integrations/google_adk/config/.env.example .env
   # Edit .env with your configuration
   ```

3. **Authenticate (for Vertex AI backend):**
   ```bash
   gcloud auth application-default login
   ```

### Local Testing (No Google Cloud Required)

All examples can run in `LOCAL` mode for testing:

```python
config = ADKConfig(
    backend=ADKBackend.LOCAL,
    workspace_dir="./workspace/adk_examples",
)
```

**Note:** Local mode has limited functionality and doesn't use actual Gemini models.

### Production Use (Vertex AI Backend)

For full functionality with Gemini models:

```python
config = ADKConfig(
    backend=ADKBackend.VERTEX_AI,
    project_id="your-project-id",
    model_name="gemini-2.0-flash-001",
    enable_search=True,
)
```

## Integration Patterns

### Pattern 1: Hybrid Agent Workflow

Combine ADK agents with your existing HRM/TRM agents:

```python
from src.agents.hrm_agent import HRMAgent
from src.integrations.google_adk import MLEngineeringAgent

# Your existing agent
hrm_agent = HRMAgent(config)

# Google ADK agent
ml_agent = MLEngineeringAgent(adk_config)

# HRM decomposes problem
subproblems = await hrm_agent.decompose_problem(query, state)

# ML agent handles model training subproblem
for sp in subproblems:
    if "model training" in sp.description:
        result = await ml_agent.train_model(...)
```

### Pattern 2: Training Data Enrichment

Use Academic Research agent to build better training datasets:

```python
from src.integrations.google_adk import AcademicResearchAgent
from training.research_corpus_builder import ResearchCorpusBuilder

research_agent = AcademicResearchAgent(config)
corpus_builder = ResearchCorpusBuilder()

# Find latest MCTS research
citations = await research_agent.find_citations("Monte Carlo Tree Search")

# Add to training corpus
corpus_builder.add_research(citations)
```

### Pattern 3: Performance Analysis

Use Data Science agent to analyze agent metrics:

```python
from src.integrations.google_adk import DataScienceAgent

ds_agent = DataScienceAgent(config)

# Analyze agent performance
analysis = await ds_agent.analyze_data(
    query="Identify factors that improve MCTS success rate",
    data_source="agent_metrics.csv",
)

# Query training logs
sql_result = await ds_agent.query_database(
    nl_query="Show agents with highest reward improvement over time",
    dataset_name="training_logs",
)
```

### Pattern 4: Automated Pipeline Development

Use Data Engineering agent for data infrastructure:

```python
from src.integrations.google_adk import DataEngineeringAgent

de_agent = DataEngineeringAgent(config)

# Design training data pipeline
pipeline = await de_agent.design_pipeline(
    query="Build ETL pipeline for MCTS trajectory data",
    pipeline_name="mcts_training_pipeline",
    source_tables=["raw_trajectories", "agent_states"],
)
```

## Advanced Examples

### Multi-Agent Research Workflow

```python
async def research_and_implement(topic: str):
    # Step 1: Research with Deep Search
    deep_search = DeepSearchAgent(config)
    research = await deep_search.full_research(topic)

    # Step 2: Analyze papers with Academic Research
    academic = AcademicResearchAgent(config)
    citations = await academic.find_citations(topic)

    # Step 3: Implement with ML Engineering
    ml_eng = MLEngineeringAgent(config)
    implementation = await ml_eng.train_model(
        query=f"Implement {topic} based on research",
        ...
    )

    return {
        "research": research,
        "citations": citations,
        "implementation": implementation,
    }
```

### Training Pipeline with All Agents

```python
async def build_complete_training_pipeline():
    # Data Engineering: Build pipeline
    de_agent = DataEngineeringAgent(config)
    pipeline = await de_agent.design_pipeline(...)

    # Data Science: Analyze existing data
    ds_agent = DataScienceAgent(config)
    analysis = await ds_agent.analyze_data(...)

    # Academic Research: Get latest techniques
    ar_agent = AcademicResearchAgent(config)
    research = await ar_agent.find_citations(...)

    # ML Engineering: Train optimized models
    ml_agent = MLEngineeringAgent(config)
    model = await ml_agent.train_model(...)

    return pipeline, analysis, research, model
```

## Troubleshooting

### Import Errors

```bash
# Ensure package is installed in editable mode
pip install -e .

# Or install with ADK dependencies
pip install -e ".[google-adk]"
```

### Authentication Errors

```bash
# Verify authentication
gcloud auth application-default print-access-token

# Check current project
gcloud config get-value project

# Re-authenticate if needed
gcloud auth application-default login
```

### API Not Enabled

```bash
# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
```

## Next Steps

1. **Review agent capabilities:**
   ```python
   capabilities = agent.get_capabilities()
   print(capabilities)
   ```

2. **Explore full API documentation:**
   - See docstrings in each agent class
   - Review `base.py` for common interfaces

3. **Integrate into your workflows:**
   - Add agents to your LangGraph graphs
   - Combine with MCTS search strategies
   - Enhance training pipelines

4. **Customize for your needs:**
   - Extend agent classes
   - Add custom prompts
   - Integrate with your monitoring systems

## Support

- **ADK Documentation:** [Google ADK Docs](https://cloud.google.com/vertex-ai/docs/agent-development-kit)
- **GitHub Issues:** Report integration issues
- **Configuration:** See `config/README.md` for detailed setup

Happy integrating! 🚀
