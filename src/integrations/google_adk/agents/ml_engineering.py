"""
Machine Learning Engineering agent wrapper (MLE-STAR).

This agent implements the MLE-STAR approach for training state-of-the-art models
through web search and targeted code block refinement.

Based on: https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base import ADKAgentAdapter, ADKAgentRequest, ADKAgentResponse, ADKConfig


class MLEngineeringAgent(ADKAgentAdapter):
    """
    Machine Learning Engineering agent for model training and optimization.

    This agent handles:
    - Initial solution generation via web search
    - Iterative code refinement through ablation studies
    - Ensemble strategies for improved performance
    - Debugging and data leakage detection
    - Data usage verification

    The agent achieves medal rankings in 63.6% of MLE-Bench-Lite competitions
    when paired with Gemini-2.5-Pro.
    """

    def __init__(self, config: ADKConfig):
        """
        Initialize ML Engineering agent.

        Args:
            config: ADK configuration
        """
        super().__init__(config, agent_name="ml_engineering")

        # MLE-specific configuration
        self.task_dir = Path(config.workspace_dir) / "tasks"
        self.output_dir = Path(config.workspace_dir) / "outputs"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default task configuration
        self.default_task_config = {
            "task_type": "Tabular Regression",
            "metric": "rmse",
            "lower_is_better": True,
        }

    async def _agent_initialize(self) -> None:
        """Initialize MLE agent resources."""
        # Check for google-adk installation
        try:
            import google.adk  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "google-adk not installed. Install with: pip install 'langgraph-multi-agent-mcts[google-adk]'"
            ) from None

        # Setup task directory structure
        self._setup_task_environment()

    def _setup_task_environment(self) -> None:
        """Setup directory structure for ML tasks."""
        # Create subdirectories for different task types
        for task_type in ["regression", "classification", "clustering", "forecasting"]:
            (self.task_dir / task_type).mkdir(exist_ok=True)

    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        """
        Execute ML engineering task.

        Args:
            request: Agent request with task specification

        Returns:
            Agent response with model training results
        """
        # Extract task parameters
        task_name = request.parameters.get("task_name", "custom_task")
        task_type = request.parameters.get("task_type", self.default_task_config["task_type"])
        data_path = request.parameters.get("data_path")
        metric = request.parameters.get("metric", self.default_task_config["metric"])
        lower_is_better = request.parameters.get("lower_is_better", self.default_task_config["lower_is_better"])

        # Validate data path
        if not data_path:
            return ADKAgentResponse(
                result="",
                status="error",
                error="data_path parameter is required for ML engineering tasks",
            )

        # Create task configuration
        task_config = {
            "task_name": task_name,
            "task_type": task_type,
            "data_path": data_path,
            "metric": metric,
            "lower_is_better": lower_is_better,
            "query": request.query,
        }

        # Save task configuration
        task_config_path = self.task_dir / f"{task_name}_config.json"
        with open(task_config_path, "w") as f:
            json.dump(task_config, f, indent=2)

        # In a full implementation, this would invoke the actual MLE-STAR agent
        # For now, we provide a structured response for integration
        result_summary = self._generate_ml_task_plan(task_config)

        return ADKAgentResponse(
            result=result_summary,
            metadata={
                "task_name": task_name,
                "task_type": task_type,
                "metric": metric,
                "config_path": str(task_config_path),
            },
            artifacts=[str(task_config_path)],
            status="success",
            session_id=request.session_id,
        )

    def _generate_ml_task_plan(self, task_config: dict[str, Any]) -> str:
        """
        Generate execution plan for ML task.

        Args:
            task_config: Task configuration

        Returns:
            Formatted execution plan
        """
        plan = f"""
# ML Engineering Task Plan

## Task Overview
- **Name**: {task_config["task_name"]}
- **Type**: {task_config["task_type"]}
- **Data**: {task_config["data_path"]}
- **Metric**: {task_config["metric"]} ({"lower is better" if task_config["lower_is_better"] else "higher is better"})

## Execution Strategy (MLE-STAR)

### Phase 1: Initial Solution Generation
1. **Web Search for SOTA Models**
   - Search for state-of-the-art approaches for {task_config["task_type"]}
   - Identify top-performing architectures and techniques
   - Consolidate best candidate solutions

2. **Baseline Implementation**
   - Implement 3-5 baseline models
   - Establish performance benchmarks
   - Identify promising directions

### Phase 2: Code Block Refinement
1. **Component Analysis**
   - Decompose ML pipeline into components:
     - Data preprocessing
     - Feature engineering
     - Model architecture
     - Training loop
     - Evaluation

2. **Ablation Studies**
   - Iteratively refine each component
   - Test variations and improvements
   - Measure impact on {task_config["metric"]}

### Phase 3: Ensemble Strategies
1. **Model Combination**
   - Develop ensemble methods
   - Weighted averaging
   - Stacking strategies

2. **Performance Optimization**
   - Hyperparameter tuning
   - Cross-validation
   - Final model selection

### Phase 4: Robustness & Validation
1. **Debugging**
   - Error analysis
   - Edge case handling
   - Stability testing

2. **Data Validation**
   - Leakage detection
   - Usage verification
   - Distribution analysis

## Query
{task_config["query"]}

## Next Steps
To execute this plan, the MLE agent will:
1. Initialize the search for SOTA models
2. Generate and test baseline implementations
3. Iteratively refine components
4. Combine best solutions into ensemble
5. Validate and deliver final model

---
*Generated by ML Engineering Agent (MLE-STAR)*
"""
        return plan.strip()

    async def train_model(
        self,
        task_name: str,
        data_path: str,
        task_type: str = "Tabular Regression",
        metric: str = "rmse",
        lower_is_better: bool = True,
        query: str | None = None,
    ) -> ADKAgentResponse:
        """
        Convenience method for model training.

        Args:
            task_name: Name of the ML task
            data_path: Path to training data
            task_type: Type of ML task (regression, classification, etc.)
            metric: Evaluation metric
            lower_is_better: Whether lower metric values are better
            query: Optional natural language description

        Returns:
            Agent response with training results
        """
        request = ADKAgentRequest(
            query=query or f"Train {task_type} model for {task_name}",
            parameters={
                "task_name": task_name,
                "task_type": task_type,
                "data_path": data_path,
                "metric": metric,
                "lower_is_better": lower_is_better,
            },
        )

        return await self.invoke(request)

    async def refine_code_block(
        self,
        code: str,
        component: str,
        improvement_goal: str,
    ) -> ADKAgentResponse:
        """
        Refine a specific code block in the ML pipeline.

        Args:
            code: Code to refine
            component: Component name (e.g., 'feature_engineering', 'model_architecture')
            improvement_goal: Goal for refinement

        Returns:
            Agent response with refined code
        """
        request = ADKAgentRequest(
            query=f"Refine {component}: {improvement_goal}",
            context={"code": code},
            parameters={
                "component": component,
                "improvement_goal": improvement_goal,
            },
        )

        return await self.invoke(request)

    def get_capabilities(self) -> dict[str, Any]:
        """Get ML Engineering agent capabilities."""
        base_caps = super().get_capabilities()
        base_caps.update(
            {
                "agent_type": "ml_engineering",
                "supported_tasks": [
                    "Tabular Regression",
                    "Tabular Classification",
                    "Time Series Forecasting",
                    "Clustering",
                ],
                "supported_metrics": [
                    "rmse",
                    "mse",
                    "mae",
                    "r2",  # regression
                    "accuracy",
                    "f1",
                    "precision",
                    "recall",
                    "auc",  # classification
                    "silhouette",
                    "davies_bouldin",  # clustering
                ],
                "features": [
                    "sota_model_search",
                    "code_refinement",
                    "ensemble_strategies",
                    "debugging",
                    "data_leakage_detection",
                ],
            }
        )
        return base_caps
