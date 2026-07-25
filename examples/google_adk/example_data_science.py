"""
Example: Using the Data Science Agent

This example demonstrates NL2SQL, data analysis, and BigQuery ML capabilities.
"""

import asyncio

from src.integrations.google_adk import DataScienceAgent
from src.integrations.google_adk.base import ADKBackend, ADKConfig


async def example_nl2sql():
    """Example: Natural language to SQL translation."""
    print("=" * 80)
    print("Example 1: NL2SQL Query")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/data_science",
        enable_search=True,
    )

    agent = DataScienceAgent(config)
    await agent.initialize()

    # Query database using natural language
    response = await agent.query_database(
        nl_query="Show me the top 10 customers by total purchase amount in the last 30 days",
        dataset_name="ecommerce",
        backend="bigquery",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nSQL Plan:\n{response.result}")

    await agent.cleanup()


async def example_data_analysis():
    """Example: Data analysis and visualization."""
    print("\n" + "=" * 80)
    print("Example 2: Data Analysis")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/data_science",
    )

    agent = DataScienceAgent(config)
    await agent.initialize()

    # Perform exploratory data analysis
    response = await agent.analyze_data(
        query="Analyze sales trends and identify key drivers of revenue growth",
        data_source="sales_data.csv",
        analysis_type="exploratory",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nAnalysis Plan:\n{response.result}")

    await agent.cleanup()


async def example_bqml_training():
    """Example: BigQuery ML model training."""
    print("\n" + "=" * 80)
    print("Example 3: BigQuery ML Training")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/data_science",
    )

    agent = DataScienceAgent(config)
    await agent.initialize()

    # Train ARIMA model for time series forecasting
    response = await agent.train_bqml_model(
        query="Forecast monthly sales for the next 6 months using historical data",
        model_type="arima",
        target_column="monthly_sales",
    )

    print(f"\nStatus: {response.status}")
    print(f"\nBQML Training Plan:\n{response.result}")

    # Get agent capabilities
    capabilities = agent.get_capabilities()
    print(f"\nSupported BQML Models: {capabilities.get('bqml_models', [])}")

    await agent.cleanup()


async def example_multi_step_workflow():
    """Example: Multi-step data science workflow."""
    print("\n" + "=" * 80)
    print("Example 4: Multi-Step Workflow")
    print("=" * 80)

    config = ADKConfig(
        backend=ADKBackend.LOCAL,
        workspace_dir="./workspace/adk_examples/data_science",
        enable_search=True,
    )

    agent = DataScienceAgent(config)
    await agent.initialize()

    # Step 1: Query data
    print("\nStep 1: Querying customer data...")
    query_response = await agent.query_database(
        nl_query="Get customer demographics and purchase history for active customers",
        dataset_name="customers",
    )
    print(f"Query plan generated: {query_response.status}")

    # Step 2: Analyze data
    print("\nStep 2: Analyzing customer segments...")
    analysis_response = await agent.analyze_data(
        query="Identify customer segments based on purchase behavior and demographics",
        data_source="customer_data_from_query",
        analysis_type="clustering",
    )
    print(f"Analysis plan generated: {analysis_response.status}")

    # Step 3: Train predictive model
    print("\nStep 3: Training customer lifetime value model...")
    model_response = await agent.train_bqml_model(
        query="Predict customer lifetime value based on identified segments",
        model_type="linear_reg",
        target_column="lifetime_value",
    )
    print(f"Model training plan generated: {model_response.status}")

    print("\nWorkflow complete!")

    await agent.cleanup()


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Google ADK Data Science Agent Examples")
    print("=" * 80)

    await example_nl2sql()
    await example_data_analysis()
    await example_bqml_training()
    await example_multi_step_workflow()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nTo use with real BigQuery:")
    print("1. Set GOOGLE_CLOUD_PROJECT in .env")
    print("2. Set BIGQUERY_PROJECT_ID and BIGQUERY_DATASET_ID")
    print("3. Set ADK_BACKEND=vertex_ai")
    print("4. Authenticate with: gcloud auth application-default login")


if __name__ == "__main__":
    asyncio.run(main())
