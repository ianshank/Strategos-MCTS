"""
Data Engineering agent for Dataform pipeline development and management.

Based on: https://github.com/google/adk-samples/tree/main/python/agents/data-engineering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base import ADKAgentAdapter, ADKAgentRequest, ADKAgentResponse, ADKConfig


class DataEngineeringAgent(ADKAgentAdapter):
    """
    Data Engineering agent for Dataform pipeline assistance.

    This agent helps with:
    - Building and modifying Dataform pipelines
    - Creating/updating SQLx files for data transformations
    - Integrating UDFs and stored procedures
    - Managing table schemas and data types
    - Diagnosing and troubleshooting pipeline issues
    - Optimizing pipeline performance
    - Ensuring data quality
    """

    def __init__(self, config: ADKConfig):
        """
        Initialize Data Engineering agent.

        Args:
            config: ADK configuration
        """
        super().__init__(config, agent_name="data_engineering")

        # Data engineering specific directories
        self.pipelines_dir = Path(config.workspace_dir) / "pipelines"
        self.schemas_dir = Path(config.workspace_dir) / "schemas"
        self.udfs_dir = Path(config.workspace_dir) / "udfs"
        self.logs_dir = Path(config.workspace_dir) / "logs"

        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.udfs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Dataform configuration
        self.dataform_repo: str | None = None
        self.dataform_workspace: str | None = None

    async def _agent_initialize(self) -> None:
        """Initialize Data Engineering agent resources."""
        try:
            import google.adk  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "google-adk not installed. Install with: pip install 'langgraph-multi-agent-mcts[google-adk]'"
            ) from None

        # Load Dataform configuration from environment
        import os

        self.dataform_repo = os.getenv("DATAFORM_REPOSITORY_NAME")
        self.dataform_workspace = os.getenv("DATAFORM_WORKSPACE_NAME")

    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Execute data engineering task.

        Args:
            request: Agent request with pipeline task

        Returns:
            Agent response with pipeline design or fix
        """
        task_type = request.parameters.get("task_type", "pipeline_design")

        if task_type == "pipeline_design":
            return await self._handle_pipeline_design(request)
        elif task_type == "sqlx_generation":
            return await self._handle_sqlx_generation(request)
        elif task_type == "troubleshooting":
            return await self._handle_troubleshooting(request)
        elif task_type == "optimization":
            return await self._handle_optimization(request)
        elif task_type == "schema_design":
            return await self._handle_schema_design(request)
        else:
            return ADKAgentResponse(
                result="",
                status="error",
                error=f"Unknown task type: {task_type}",
            )

    async def _handle_pipeline_design(
        self,
        request: ADKAgentRequest,
    ) -> ADKAgentResponse:
        """
        Design Dataform pipeline.

        Args:
            request: Agent request

        Returns:
            Agent response with pipeline design
        """
        pipeline_name = request.parameters.get("pipeline_name", "new_pipeline")
        source_tables = request.parameters.get("source_tables", [])
        target_table = request.parameters.get("target_table")

        design = self._generate_pipeline_design(
            query=request.query,
            pipeline_name=pipeline_name,
            source_tables=source_tables,
            target_table=target_table,
        )

        # Save design
        design_file = self.pipelines_dir / f"{pipeline_name}_design.md"
        with open(design_file, "w") as f:
            f.write(design)

        return ADKAgentResponse(
            result=design,
            metadata={
                "task_type": "pipeline_design",
                "pipeline_name": pipeline_name,
                "design_file": str(design_file),
            },
            artifacts=[str(design_file)],
            status="success",
            session_id=request.session_id,
        )

    async def _handle_sqlx_generation(
        self,
        request: ADKAgentRequest,
    ) -> ADKAgentResponse:
        """
        Generate SQLx transformation file.

        Args:
            request: Agent request

        Returns:
            Agent response with SQLx code
        """
        table_name = request.parameters.get("table_name", "new_table")
        transformation_type = request.parameters.get("transformation_type", "table")

        sqlx = self._generate_sqlx(
            query=request.query,
            table_name=table_name,
            transformation_type=transformation_type,
        )

        # Save SQLx file
        sqlx_file = self.pipelines_dir / f"{table_name}.sqlx"
        with open(sqlx_file, "w") as f:
            f.write(sqlx)

        return ADKAgentResponse(
            result=sqlx,
            metadata={
                "task_type": "sqlx_generation",
                "table_name": table_name,
                "sqlx_file": str(sqlx_file),
            },
            artifacts=[str(sqlx_file)],
            status="success",
            session_id=request.session_id,
        )

    async def _handle_troubleshooting(
        self,
        request: ADKAgentRequest,
    ) -> ADKAgentResponse:
        """
        Troubleshoot pipeline issues.

        Args:
            request: Agent request

        Returns:
            Agent response with troubleshooting steps
        """
        error_log = request.context.get("error_log")
        compilation_log = request.context.get("compilation_log")

        troubleshooting = self._generate_troubleshooting_plan(
            query=request.query,
            error_log=error_log,
            compilation_log=compilation_log,
        )

        return ADKAgentResponse(
            result=troubleshooting,
            metadata={
                "task_type": "troubleshooting",
            },
            status="success",
            session_id=request.session_id,
        )

    async def _handle_optimization(
        self,
        request: ADKAgentRequest,
    ) -> ADKAgentResponse:
        """
        Optimize pipeline performance.

        Args:
            request: Agent request

        Returns:
            Agent response with optimization recommendations
        """
        pipeline_name = request.parameters.get("pipeline_name")
        current_performance = request.context.get("performance_metrics")

        optimization = self._generate_optimization_plan(
            query=request.query,
            pipeline_name=pipeline_name,
            current_performance=current_performance,
        )

        return ADKAgentResponse(
            result=optimization,
            metadata={
                "task_type": "optimization",
                "pipeline_name": pipeline_name,
            },
            status="success",
            session_id=request.session_id,
        )

    async def _handle_schema_design(
        self,
        request: ADKAgentRequest,
    ) -> ADKAgentResponse:
        """
        Design table schema.

        Args:
            request: Agent request

        Returns:
            Agent response with schema definition
        """
        table_name = request.parameters.get("table_name")
        data_requirements = request.parameters.get("data_requirements", {})

        schema = self._generate_schema_design(
            query=request.query,
            table_name=table_name,
            data_requirements=data_requirements,
        )

        # Save schema
        schema_file = self.schemas_dir / f"{table_name}_schema.json"
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

        return ADKAgentResponse(
            result=json.dumps(schema, indent=2),
            metadata={
                "task_type": "schema_design",
                "table_name": table_name,
                "schema_file": str(schema_file),
            },
            artifacts=[str(schema_file)],
            status="success",
            session_id=request.session_id,
        )

    def _generate_pipeline_design(
        self,
        query: str,
        pipeline_name: str,
        source_tables: list[str],
        target_table: str | None,
    ) -> str:
        """Generate Dataform pipeline design."""
        return f"""
# Dataform Pipeline Design: {pipeline_name}

## Objective
{query}

## Pipeline Configuration
- **Name**: {pipeline_name}
- **Repository**: {self.dataform_repo or "Not configured"}
- **Workspace**: {self.dataform_workspace or "Not configured"}

## Data Flow

### Source Tables
{chr(10).join(f"- {table}" for table in source_tables) if source_tables else "- To be defined"}

### Target Table
{target_table or "To be defined"}

## Pipeline Architecture

### Stage 1: Data Ingestion
```sqlx
-- Raw data ingestion
-- Source: {", ".join(source_tables) if source_tables else "TBD"}
-- Destination: raw_{pipeline_name}
```

### Stage 2: Data Cleaning
```sqlx
-- Data quality checks
-- Null handling
-- Type conversions
-- Duplicate removal
```

### Stage 3: Transformations
```sqlx
-- Business logic transformations
-- Aggregations
-- Joins
-- Calculations
```

### Stage 4: Data Quality
```sqlx
-- Assertions
-- Data validation
-- Completeness checks
-- Consistency verification
```

### Stage 5: Final Output
```sqlx
-- Materialized table/view
-- Partitioning strategy
-- Clustering keys
-- Description and tags
```

## Dependencies

### Upstream Dependencies
{chr(10).join(f"- {table}" for table in source_tables) if source_tables else "- None"}

### Downstream Dependencies
- To be identified based on usage

## Data Quality Rules

1. **Completeness**: All required fields must be non-null
2. **Uniqueness**: Primary keys must be unique
3. **Consistency**: Foreign keys must reference valid records
4. **Accuracy**: Values must be within expected ranges
5. **Timeliness**: Data must be fresh (define SLA)

## Deployment Strategy

1. **Development**: Test in dev workspace
2. **Staging**: Validate with production data snapshot
3. **Production**: Deploy with monitoring

## Monitoring

- **Execution Time**: Track runtime
- **Row Counts**: Monitor data volumes
- **Error Rates**: Alert on failures
- **Data Quality**: Track assertion pass rates

---
*Dataform Pipeline Design - Ready for SQLx Implementation*
""".strip()

    def _generate_sqlx(
        self,
        query: str,
        table_name: str,
        transformation_type: str,
    ) -> str:
        """Generate SQLx file content."""
        return f"""
config {{
    type: "{transformation_type}",
    schema: "analytics",
    description: "{query}",
    tags: ["auto_generated"],
    assertions: {{
        uniqueKey: ["id"],
        nonNull: ["id", "created_at"]
    }}
}}

-- {query}
-- Generated for: {table_name}

SELECT
    -- Add your columns here
    id,
    created_at,
    updated_at
FROM
    ${{ref("source_table")}}
WHERE
    -- Add your filters here
    created_at >= CURRENT_DATE() - 30

-- Example transformations:
-- - Use ${{ref("table_name")}} for dependencies
-- - Add pre_operations for temp tables
-- - Add post_operations for grants/cleanup
-- - Define assertions for data quality
""".strip()

    def _generate_troubleshooting_plan(
        self,
        query: str,
        error_log: str | None,
        compilation_log: str | None,
    ) -> str:
        """Generate troubleshooting plan."""
        return f"""
# Pipeline Troubleshooting

## Issue Description
{query}

## Error Analysis

### Compilation Errors
{compilation_log if compilation_log else "No compilation log provided"}

### Execution Errors
{error_log if error_log else "No error log provided"}

## Troubleshooting Steps

### Step 1: Verify Dependencies
- Check all referenced tables exist
- Verify ${{ref()}} syntax is correct
- Confirm upstream tables are up-to-date

### Step 2: Check SQL Syntax
- Validate SQL in each SQLX file
- Check for common syntax errors
- Verify BigQuery-specific functions

### Step 3: Data Quality Issues
- Check for null values in non-null columns
- Verify uniqueness constraints
- Look for type mismatches

### Step 4: Permission Issues
- Confirm service account has read access to sources
- Verify write permissions to destination dataset
- Check BigQuery quotas

### Step 5: Logic Errors
- Review transformation logic
- Check join conditions
- Verify aggregation logic

## Common Issues & Solutions

### Issue: Reference Error
**Solution**: Ensure referenced table is declared in dependencies

### Issue: Compilation Timeout
**Solution**: Simplify complex queries, add intermediate tables

### Issue: Assertion Failure
**Solution**: Review data quality rules, adjust assertions

### Issue: Circular Dependency
**Solution**: Refactor pipeline to remove cycles

## Next Actions

1. Review error messages in detail
2. Test SQL queries in BigQuery console
3. Validate data at each transformation step
4. Check Dataform documentation for specific errors
5. Consult pipeline dependencies graph

---
*Dataform Troubleshooting Guide*
""".strip()

    def _generate_optimization_plan(
        self,
        query: str,
        pipeline_name: str | None,
        current_performance: dict | None,
    ) -> str:
        """Generate optimization plan."""
        return f"""
# Pipeline Optimization: {pipeline_name or "Pipeline"}

## Objective
{query}

## Current Performance
{json.dumps(current_performance, indent=2) if current_performance else "Metrics not provided"}

## Optimization Strategies

### 1. Query Optimization
- **Partitioning**: Use date/timestamp partitioning
- **Clustering**: Add clustering keys for common filters
- **Pruning**: Ensure partition pruning is effective

### 2. Materialization Strategy
- **Tables vs Views**: Materialize frequently accessed transformations
- **Incremental Models**: Use incremental updates for large tables
- **Snapshots**: Consider snapshot tables for historical data

### 3. Dependency Management
- **Parallel Execution**: Minimize sequential dependencies
- **Batch Size**: Optimize batch sizes for transformations
- **Scheduling**: Stagger execution to avoid contention

### 4. BigQuery Optimizations
```sql
-- Partition by date
PARTITION BY DATE(timestamp_column)

-- Cluster for faster queries
CLUSTER BY category, region

-- Use array/struct for denormalization
-- Avoid SELECT * in production
-- Use approximate aggregations where appropriate
```

### 5. Cost Optimization
- Avoid full table scans
- Use cached results when possible
- Implement data lifecycle policies
- Monitor and optimize slot usage

## Performance Targets

- **Execution Time**: < 5 minutes for most transformations
- **Data Freshness**: Within SLA requirements
- **Cost**: Minimize bytes processed
- **Reliability**: > 99% success rate

## Monitoring & Iteration

1. Track key metrics over time
2. A/B test optimization changes
3. Review execution plans
4. Adjust based on usage patterns

---
*Dataform Optimization Recommendations*
""".strip()

    def _generate_schema_design(
        self,
        query: str,
        table_name: str | None,
        data_requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate table schema design."""
        return {
            "table_name": table_name,
            "description": query,
            "schema": {
                "fields": [
                    {"name": "id", "type": "STRING", "mode": "REQUIRED", "description": "Unique identifier"},
                    {
                        "name": "created_at",
                        "type": "TIMESTAMP",
                        "mode": "REQUIRED",
                        "description": "Record creation timestamp",
                    },
                    {
                        "name": "updated_at",
                        "type": "TIMESTAMP",
                        "mode": "NULLABLE",
                        "description": "Record update timestamp",
                    },
                    # Add fields based on requirements
                ]
            },
            "partitioning": {"type": "DAY", "field": "created_at"},
            "clustering": {"fields": ["id"]},
            "data_requirements": data_requirements,
        }

    async def design_pipeline(
        self,
        query: str,
        pipeline_name: str,
        source_tables: list[str],
        target_table: str | None = None,
    ) -> ADKAgentResponse:
        """
        Design a Dataform pipeline.

        Args:
            query: Pipeline objective
            pipeline_name: Name of the pipeline
            source_tables: List of source table names
            target_table: Target table name

        Returns:
            Agent response with pipeline design
        """
        request = ADKAgentRequest(
            query=query,
            parameters={
                "task_type": "pipeline_design",
                "pipeline_name": pipeline_name,
                "source_tables": source_tables,
                "target_table": target_table,
            },
        )

        return await self.invoke(request)

    async def generate_sqlx(
        self,
        query: str,
        table_name: str,
        transformation_type: str = "table",
    ) -> ADKAgentResponse:
        """
        Generate SQLx transformation file.

        Args:
            query: Transformation description
            table_name: Name of the table
            transformation_type: Type (table, view, incremental)

        Returns:
            Agent response with SQLx code
        """
        request = ADKAgentRequest(
            query=query,
            parameters={
                "task_type": "sqlx_generation",
                "table_name": table_name,
                "transformation_type": transformation_type,
            },
        )

        return await self.invoke(request)

    async def troubleshoot(
        self,
        query: str,
        error_log: str | None = None,
        compilation_log: str | None = None,
    ) -> ADKAgentResponse:
        """
        Troubleshoot pipeline issues.

        Args:
            query: Issue description
            error_log: Execution error log
            compilation_log: Compilation error log

        Returns:
            Agent response with troubleshooting steps
        """
        request = ADKAgentRequest(
            query=query,
            context={
                "error_log": error_log,
                "compilation_log": compilation_log,
            },
            parameters={
                "task_type": "troubleshooting",
            },
        )

        return await self.invoke(request)

    def get_capabilities(self) -> dict[str, Any]:
        """Get Data Engineering agent capabilities."""
        base_caps = super().get_capabilities()
        base_caps.update(
            {
                "agent_type": "data_engineering",
                "dataform_repo": self.dataform_repo,
                "dataform_workspace": self.dataform_workspace,
                "features": [
                    "pipeline_design",
                    "sqlx_generation",
                    "troubleshooting",
                    "optimization",
                    "schema_design",
                    "udf_integration",
                    "data_quality",
                ],
                "supported_transforms": ["table", "view", "incremental", "assertion"],
            }
        )
        return base_caps
