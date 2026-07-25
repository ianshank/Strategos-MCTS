"""
Example: Using the ML Engineering Agent (MLE-STAR)

This example demonstrates how to use the Google ADK ML Engineering agent
to train state-of-the-art models for machine learning tasks.
"""

import asyncio

from src.integrations.google_adk import MLEngineeringAgent
from src.integrations.google_adk.base import ADKBackend, ADKConfig


async def example_basic_training():
    """Example: Basic model training task."""
    print("=" * 80)
    print("Example 1: Basic Model Training")
    print("=" * 80)

    # Create configuration
    config = ADKConfig(
        backend=ADKBackend.LOCAL,  # Use LOCAL for testing without Google Cloud
        model_name="gemini-2.0-flash-001",
        workspace_dir="./workspace/adk_examples/ml_engineering",
        enable_search=True,
    )

    # Initialize agent
    agent = MLEngineeringAgent(config)
    await agent.initialize()

    # Train a regression model
    response = await agent.train_model(
        task_name="california_housing",
        data_path="./data/california_housing.csv",
        task_type="Tabular Regression",
        metric="rmse",
        lower_is_better=True,
        query="Train a model to predict California housing prices using SOTA techniques",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nTraining Plan:\n{response.result}")
    print(f"\nArtifacts: {response.artifacts}")

    # Cleanup
    await agent.cleanup()


async def example_code_refinement():
    """Example: Refining a specific code block."""
    print("\n" + "=" * 80)
    print("Example 2: Code Block Refinement")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/ml_engineering",
    )

    agent = MLEngineeringAgent(config)
    await agent.initialize()

    # Existing feature engineering code
    existing_code = """
def create_features(df):
    df['age'] = 2024 - df['year_built']
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    return df
"""

    # Refine the feature engineering component
    response = await agent.refine_code_block(
        code=existing_code,
        component="feature_engineering",
        improvement_goal="Add more sophisticated features and handle edge cases",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nRefined Code Plan:\n{response.result}")

    await agent.cleanup()


async def example_classification_task():
    """Example: Classification task."""
    print("\n" + "=" * 80)
    print("Example 3: Classification Task")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/ml_engineering",
        enable_search=True,
    )

    agent = MLEngineeringAgent(config)
    await agent.initialize()

    # Train a classification model
    response = await agent.train_model(
        task_name="customer_churn",
        data_path="./data/customer_churn.csv",
        task_type="Tabular Classification",
        metric="f1",
        lower_is_better=False,
        query="Build a customer churn prediction model with high F1 score",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nClassification Plan:\n{response.result}")

    # Get agent capabilities
    capabilities = agent.get_capabilities()
    print("\nAgent Capabilities:")
    print(f"- Supported Tasks: {capabilities['supported_tasks']}")
    print(f"- Supported Metrics: {capabilities['supported_metrics']}")
    print(f"- Features: {capabilities['features']}")

    await agent.cleanup()


async def example_with_langgraph_integration():
    """Example: Integration with LangGraph workflow."""
    print("\n" + "=" * 80)
    print("Example 4: LangGraph Integration")
    print("=" * 80)

    # This shows how the ADK agent can be used within a LangGraph workflow
    # alongside your existing HRM/TRM agents

    config = ADKConfig.from_env()  # Load from environment variables
    config.workspace_dir = "./workspace/adk_examples/ml_engineering"

    agent = MLEngineeringAgent(config)
    await agent.initialize()

    # Simulate a multi-agent workflow where:
    # 1. HRM agent identifies need for ML model
    # 2. ML Engineering agent handles model development
    # 3. TRM agent evaluates and integrates the model

    print("\n[HRM Agent] Analyzing task...")
    print("[HRM Agent] Decomposed problem: Need ML model for prediction")

    print("\n[ML Engineering Agent] Developing model...")
    response = await agent.train_model(
        task_name="workflow_task",
        data_path="./data/training_data.csv",
        task_type="Tabular Regression",
        metric="rmse",
        query="Build optimized regression model for workflow integration",
    )

    print(f"\n[ML Engineering Agent] {response.status}")
    print(f"[ML Engineering Agent] Model plan generated: {len(response.result)} chars")

    print("\n[TRM Agent] Evaluating model quality...")
    print("[TRM Agent] Model approved for integration")

    await agent.cleanup()


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Google ADK ML Engineering Agent Examples")
    print("=" * 80)

    # Run examples
    await example_basic_training()
    await example_code_refinement()
    await example_classification_task()
    await example_with_langgraph_integration()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Configure your Google Cloud credentials (see config/README.md)")
    print("2. Set ADK_BACKEND=vertex_ai for full functionality")
    print("3. Integrate agents into your LangGraph workflows")


if __name__ == "__main__":
    asyncio.run(main())
