"""
Data Science agent wrapper for NL2SQL, data analysis, and BigQuery integration.

Based on: https://github.com/google/adk-samples/tree/main/python/agents/data-science
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..base import ADKAgentAdapter, ADKAgentRequest, ADKAgentResponse, ADKConfig


class DataScienceAgent(ADKAgentAdapter):
    """
    Multi-agent Data Science framework for sophisticated data analysis.

    This agent orchestrates specialized sub-agents for:
    - Natural language to SQL (NL2SQL) translation
    - Data analysis and visualization
    - Machine learning with BigQuery ML
    - Cross-dataset operations

    Supports BigQuery and AlloyDB backends.
    """

    def __init__(self, config: ADKConfig):
        """
        Initialize Data Science agent.

        Args:
            config: ADK configuration
        """
        super().__init__(config, agent_name="data_science")

        # Data Science specific configuration
        self.dataset_config_dir = Path(config.workspace_dir) / "dataset_configs"
        self.query_history_dir = Path(config.workspace_dir) / "query_history"
        self.visualization_dir = Path(config.workspace_dir) / "visualizations"

        self.dataset_config_dir.mkdir(parents=True, exist_ok=True)
        self.query_history_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # Sub-agent roles
        self.sub_agents = {
            "database": "NL2SQL translation for BigQuery/AlloyDB",
            "analysis": "Python code execution for data analysis",
            "bqml": "BigQuery ML model training",
        }

    async def _agent_initialize(self) -> None:
        """Initialize Data Science agent resources."""
        try:
            import google.adk  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "google-adk not installed. Install with: pip install 'langgraph-multi-agent-mcts[google-adk]'"
            ) from None

        # Setup environment variables for database connections
        self._setup_database_config()

    def _setup_database_config(self) -> None:
        """Setup database connection configuration."""
        # Check for required environment variables
        required_vars = {
            "bigquery": ["GOOGLE_CLOUD_PROJECT"],
            "alloydb": ["ALLOYDB_INSTANCE", "ALLOYDB_DATABASE", "ALLOYDB_USER"],
        }

        self.available_backends = []
        if all(os.getenv(var) for var in required_vars["bigquery"]):
            self.available_backends.append("bigquery")
        if all(os.getenv(var) for var in required_vars["alloydb"]):
            self.available_backends.append("alloydb")

    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Execute data science task.

        Args:
            request: Agent request with query

        Returns:
            Agent response with analysis results
        """
        # Determine task type from request
        task_type = request.parameters.get("task_type", "nl2sql")
        dataset_name = request.parameters.get("dataset_name", "default")

        if task_type == "nl2sql":
            return await self._handle_nl2sql(request, dataset_name)
        elif task_type == "analysis":
            return await self._handle_analysis(request)
        elif task_type == "bqml":
            return await self._handle_bqml(request)
        else:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"Unknown task type: {task_type}. Supported: nl2sql, analysis, bqml",
            )

    async def _handle_nl2sql(
        self,
        request: ADKAgentRequest,
        dataset_name: str,
    ) -> ADKAgentResponse:
        """
        Handle natural language to SQL translation.

        Args:
            request: Agent request
            dataset_name: Dataset name

        Returns:
            Agent response with SQL query
        """
        nl_query = request.query
        backend = request.parameters.get("backend", "bigquery")

        if backend not in self.available_backends:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"Backend {backend} not configured. Available: {self.available_backends}",
            )

        # In full implementation, this would use CHASE-SQL or Gemini NL2SQL
        # For now, provide structured response
        sql_plan = self._generate_sql_plan(nl_query, dataset_name, backend)

        # Save query to history
        query_record = {
            "natural_language": nl_query,
            "dataset": dataset_name,
            "backend": backend,
            "sql_plan": sql_plan,
        }

        history_file = self.query_history_dir / f"{dataset_name}_queries.jsonl"
        with open(history_file, "a") as f:
            f.write(json.dumps(query_record) + "\n")

        return ADKAgentResponse(
            result=sql_plan,
            metadata={
                "task_type": "nl2sql",
                "dataset": dataset_name,
                "backend": backend,
            },
            artifacts=[str(history_file)],
            status="success",
            session_id=request.session_id,
        )

    async def _handle_analysis(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Handle data analysis task.

        Args:
            request: Agent request

        Returns:
            Agent response with analysis code and results
        """
        analysis_type = request.parameters.get("analysis_type", "exploratory")
        data_source = request.parameters.get("data_source")

        analysis_plan = self._generate_analysis_plan(
            request.query,
            analysis_type,
            data_source,
        )

        return ADKAgentResponse(
            result=analysis_plan,
            metadata={
                "task_type": "analysis",
                "analysis_type": analysis_type,
            },
            status="success",
            session_id=request.session_id,
        )

    async def _handle_bqml(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Handle BigQuery ML model training.

        Args:
            request: Agent request

        Returns:
            Agent response with BQML training plan
        """
        model_type = request.parameters.get("model_type", "arima")
        target_column = request.parameters.get("target_column")

        bqml_plan = self._generate_bqml_plan(request.query, model_type, target_column)

        return ADKAgentResponse(
            result=bqml_plan,
            metadata={
                "task_type": "bqml",
                "model_type": model_type,
            },
            status="success",
            session_id=request.session_id,
        )

    def _generate_sql_plan(
        self,
        nl_query: str,
        dataset_name: str,
        backend: str,
    ) -> str:
        """Generate NL2SQL execution plan."""
        return f"""
# NL2SQL Translation Plan

## Query
{nl_query}

## Target
- **Dataset**: {dataset_name}
- **Backend**: {backend}

## Translation Strategy

### Step 1: Schema Understanding
- Retrieve dataset schema and metadata
- Identify relevant tables and columns
- Understand relationships and foreign keys

### Step 2: Query Generation
- Parse natural language intent
- Map entities to database objects
- Generate SQL query structure

### Step 3: Optimization
- Apply query optimization techniques
- Ensure efficient execution plan
- Add appropriate indexes if needed

### Step 4: Validation
- Verify query syntax
- Check for potential issues
- Validate expected results

## Expected SQL
```sql
-- SQL query will be generated here based on:
-- - Dataset schema from {dataset_name}
-- - Natural language query: "{nl_query}"
-- - Backend-specific optimizations for {backend}
```

*Note: Full SQL generation requires dataset schema configuration*
""".strip()

    def _generate_analysis_plan(
        self,
        query: str,
        analysis_type: str,
        data_source: str | None,
    ) -> str:
        """Generate data analysis plan."""
        return f"""
# Data Analysis Plan

## Objective
{query}

## Analysis Type
{analysis_type}

## Data Source
{data_source or "To be specified"}

## Analysis Steps

### 1. Data Loading
- Connect to data source
- Load relevant datasets
- Validate data quality

### 2. Exploratory Data Analysis
- Summary statistics
- Distribution analysis
- Correlation analysis
- Missing value analysis

### 3. Data Visualization
- Create relevant plots
- Interactive dashboards
- Statistical visualizations

### 4. Statistical Analysis
- Hypothesis testing
- Trend analysis
- Pattern detection

### 5. Insights & Recommendations
- Key findings
- Actionable insights
- Recommendations

## Code Template
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
# data = pd.read_csv('{data_source}')  # or from BigQuery

# Analysis code will be generated here
```

*Note: Complete analysis requires data source specification*
""".strip()

    def _generate_bqml_plan(
        self,
        query: str,
        model_type: str,
        target_column: str | None,
    ) -> str:
        """Generate BigQuery ML training plan."""
        return f"""
# BigQuery ML Training Plan

## Objective
{query}

## Model Configuration
- **Type**: {model_type}
- **Target**: {target_column or "To be specified"}

## Supported BQML Models
- **ARIMA**: Time series forecasting
- **Exponential Smoothing**: Seasonal forecasting
- **TFT (Temporal Fusion Transformer)**: Advanced time series
- **Linear Regression**: Regression tasks
- **Logistic Regression**: Classification tasks
- **Boosted Trees**: Ensemble methods

## Training Steps

### 1. Data Preparation
```sql
-- Prepare training dataset
CREATE OR REPLACE TABLE `project.dataset.training_data` AS
SELECT
  -- feature columns
  {target_column or "target"} as target
FROM `project.dataset.source_table`
WHERE conditions...;
```

### 2. Model Creation
```sql
-- Create BQML model
CREATE OR REPLACE MODEL `project.dataset.{model_type}_model`
OPTIONS(
  model_type='{model_type.upper()}',
  time_series_timestamp_col='timestamp',
  time_series_data_col='{target_column or "target"}'
  -- Additional options...
) AS
SELECT * FROM `project.dataset.training_data`;
```

### 3. Model Evaluation
```sql
-- Evaluate model performance
SELECT * FROM ML.EVALUATE(
  MODEL `project.dataset.{model_type}_model`,
  TABLE `project.dataset.test_data`
);
```

### 4. Predictions
```sql
-- Generate predictions
SELECT * FROM ML.FORECAST(
  MODEL `project.dataset.{model_type}_model`,
  STRUCT(30 AS horizon)  -- 30 periods ahead
);
```

*Note: Requires BigQuery dataset and table configuration*
""".strip()

    async def query_database(
        self,
        nl_query: str,
        dataset_name: str = "default",
        backend: str = "bigquery",
    ) -> ADKAgentResponse:
        """
        Execute natural language query against database.

        Args:
            nl_query: Natural language query
            dataset_name: Dataset name
            backend: Database backend (bigquery or alloydb)

        Returns:
            Agent response with SQL and results
        """
        request = ADKAgentRequest(
            query=nl_query,
            parameters={
                "task_type": "nl2sql",
                "dataset_name": dataset_name,
                "backend": backend,
            },
        )

        return await self.invoke(request)

    async def analyze_data(
        self,
        query: str,
        data_source: str | None = None,
        analysis_type: str = "exploratory",
    ) -> ADKAgentResponse:
        """
        Perform data analysis.

        Args:
            query: Analysis objective
            data_source: Path or identifier of data source
            analysis_type: Type of analysis (exploratory, statistical, etc.)

        Returns:
            Agent response with analysis code and results
        """
        request = ADKAgentRequest(
            query=query,
            parameters={
                "task_type": "analysis",
                "data_source": data_source,
                "analysis_type": analysis_type,
            },
        )

        return await self.invoke(request)

    async def train_bqml_model(
        self,
        query: str,
        model_type: str = "arima",
        target_column: str | None = None,
    ) -> ADKAgentResponse:
        """
        Train BigQuery ML model.

        Args:
            query: Training objective
            model_type: BQML model type (arima, linear_reg, etc.)
            target_column: Target column name

        Returns:
            Agent response with BQML training plan
        """
        request = ADKAgentRequest(
            query=query,
            parameters={
                "task_type": "bqml",
                "model_type": model_type,
                "target_column": target_column,
            },
        )

        return await self.invoke(request)

    def get_capabilities(self) -> dict[str, Any]:
        """Get Data Science agent capabilities."""
        base_caps = super().get_capabilities()
        base_caps.update(
            {
                "agent_type": "data_science",
                "sub_agents": self.sub_agents,
                "supported_backends": self.available_backends,
                "features": [
                    "nl2sql_translation",
                    "data_analysis",
                    "visualization",
                    "bigquery_ml",
                    "cross_dataset_joins",
                    "session_memory",
                ],
                "bqml_models": [
                    "arima",
                    "exponential_smoothing",
                    "tft",
                    "linear_reg",
                    "logistic_reg",
                    "boosted_tree",
                ],
            }
        )
        return base_caps
