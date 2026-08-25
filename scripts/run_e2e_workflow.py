#!/usr/bin/env python3
"""
End-to-End E2E Workflow Runner with ADK Agents.

This script executes the full multi-agent workflow using the IntegratedFramework
infrastructure, including HRM, TRM, MCTS, and Google ADK agents.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm.anthropic_client import AnthropicClient
from src.adapters.llm.base import BaseLLMClient
from src.adapters.llm.lmstudio_client import LMStudioClient
from src.adapters.llm.openai_client import OpenAIClient
from src.framework.graph import GraphBuilder

# Import Demo Agents (Simulated behavior for E2E test)
try:
    from demo_src.agents_demo import HRMAgent, TRMAgent
except ImportError:
    # Fallback if demo_src not found (e.g. running from src root)
    sys.path.insert(0, str(Path(__file__).parent.parent / "demo_src"))
    from agents_demo import HRMAgent, TRMAgent

# Import ADK Integration
try:
    from src.integrations.google_adk.agents.data_science import DataScienceAgent  # noqa: F401
    from src.integrations.google_adk.agents.deep_search import DeepSearchAgent  # noqa: F401
    from src.integrations.google_adk.agents.ml_engineering import MLEngineeringAgent  # noqa: F401
    from src.integrations.google_adk.base import ADKAgentAdapter, ADKConfig  # noqa: F401
except ImportError:
    logger = logging.getLogger("E2E_Runner")
    logger.error("ADK dependencies not found. Cannot run E2E test without real ADK agents.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("E2E_Runner")


def get_real_llm_client() -> BaseLLMClient:
    """
    Get a real LLM client based on available environment variables.
    Prioritizes: OpenAI -> Anthropic -> LM Studio (Local)
    """
    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Using OpenAI Client")
        return OpenAIClient()
    elif os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic Client")
        return AnthropicClient()
    else:
        # Default to LM Studio if no keys, assuming local setup
        logger.info("No API keys found. Defaulting to LM Studio (Local)")
        # Ensure LM Studio URL is reachable or set env var if needed
        return LMStudioClient(base_url="http://localhost:1234/v1")


class LangGraphAgentAdapter:
    """Adapter to make demo agents compatible with GraphBuilder."""

    def __init__(self, agent, name: str):
        self.agent = agent
        self.name = name

    async def process(self, query: str, rag_context: str = None) -> dict[str, Any]:
        # Demo agents process(query) returns dict with response, confidence, etc.
        # GraphBuilder expects dict with "response", "metadata"

        try:
            result = await self.agent.process(query)

            return {
                "response": result.get("response", ""),
                "metadata": {
                    "decomposition_quality_score": result.get("confidence", 0.5),
                    "final_quality_score": result.get("confidence", 0.5),
                    "steps": result.get("steps", []),
                },
            }
        except Exception as e:
            logger.error(f"Agent {self.name} failed: {e}")
            return {"response": f"Error: {e}", "metadata": {}}


async def main():
    logger.info("Starting E2E Workflow with ADK Agents...")

    # Initialize Real Model Adapter
    try:
        model_adapter = get_real_llm_client()
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return 1

    # Initialize Demo Agents
    logger.info("Initializing Standard Agents...")
    # Note: demo_src agents take llm_client
    hrm_raw = HRMAgent(llm_client=model_adapter)
    trm_raw = TRMAgent(llm_client=model_adapter)

    # Wrap for GraphBuilder
    hrm_agent = LangGraphAgentAdapter(hrm_raw, "hrm")
    trm_agent = LangGraphAgentAdapter(trm_raw, "trm")

    # Import ADK Agents
    try:
        from src.integrations.google_adk.agents.data_science import DataScienceAgent
        from src.integrations.google_adk.agents.deep_search import DeepSearchAgent
        from src.integrations.google_adk.agents.ml_engineering import MLEngineeringAgent
        from src.integrations.google_adk.base import ADKConfig
    except ImportError:
        logger.error("Failed to import ADK agents. Ensure google-adk is installed.")
        sys.exit(1)

    # Initialize ADK Agents
    logger.info("Initializing ADK Agents...")
    adk_agents = {}

    # Check for credentials - either via env var (service account) or application default credentials
    app_default_creds_path = os.path.join(
        os.environ.get("APPDATA", ""), "gcloud", "application_default_credentials.json"
    )

    has_service_account = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    has_app_default = os.path.exists(app_default_creds_path)

    if not (has_service_account or has_app_default):
        logger.error("No Google Cloud credentials found.")
        logger.error("Either set GOOGLE_APPLICATION_CREDENTIALS (service account) or")
        logger.error(f"run 'gcloud auth application-default login' (creates {app_default_creds_path})")
        return 1

    if has_app_default:
        logger.info(f"Using application default credentials from: {app_default_creds_path}")
    else:
        logger.info(f"Using service account credentials from: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

    try:
        config = ADKConfig(workspace_dir="./workspace_test", enable_search=True)

        # Initialize all available ADK agents
        adk_agents["deep_search"] = DeepSearchAgent(config)
        adk_agents["ml_engineering"] = MLEngineeringAgent(config)
        adk_agents["data_science"] = DataScienceAgent(config)

        logger.info(f"Successfully initialized agents: {list(adk_agents.keys())}")
    except Exception as e:
        logger.error(f"Failed to init ADK agents: {e}")
        return 1

    # Initialize GraphBuilder
    logger.info("Building Workflow Graph...")
    builder = GraphBuilder(
        hrm_agent=hrm_agent,
        trm_agent=trm_agent,
        model_adapter=model_adapter,
        logger=logger,
        max_iterations=2,
        enable_parallel_agents=True,
        adk_agents=adk_agents,
    )

    # Build and Compile Graph
    try:
        workflow = builder.build_graph()
        app = workflow.compile()
    except ImportError as e:
        logger.error(f"Failed to build graph: {e}")
        logger.error("Please ensure 'langgraph' is installed: pip install langgraph")
        return 1

    # Define Queries for each agent
    queries = [
        ("Deep Search", "Conduct comprehensive research on the impact of quantum computing on cryptography."),
        ("ML Engineering", "Train a regression model to predict housing prices using the california_housing dataset."),
        ("Data Science", "Analyze the sales_data table in BigQuery to identify seasonal trends."),
        (
            "Market Research",
            "perform deep research on the trend of the market using advanced workflow offer bes suggestions for options withinn 3,000 and stocks to short with high return and safe",
        ),
    ]

    for agent_name, query in queries:
        print(f"\n{'='*20} Processing {agent_name} Query {'='*20}")
        print(f"Query: {query}\n")

        # Initial State
        initial_state = {
            "query": query,
            "use_rag": False,
            "use_mcts": True,
            "iteration": 0,
            "max_iterations": 2,
            "agent_outputs": [],
        }

        # Execute Workflow
        try:
            result = await app.ainvoke(initial_state)

            print(f"\n{'='*20} {agent_name} Workflow Complete {'='*20}")
            print(
                f"\n[RESPONSE] Final Response:\n{'-'*20}\n{result.get('final_response', 'No response generated')}\n{'-'*20}"
            )

            # Log details
            metadata = result.get("metadata", {})
            print("\n[METADATA] Execution Metadata:")
            print(f"  * Agents Used: {', '.join(metadata.get('agents_used', []))}")
            print(f"  * Consensus Score: {metadata.get('consensus_score', 0.0):.2f}")
            if "confidence_scores" in metadata:
                print("  * Confidence Scores:")
                for agent, score in metadata["confidence_scores"].items():
                    print(f"    - {agent}: {score:.2f}")

            # Check for ADK execution
            if "adk_results" in result.get("state", {}):
                print(f"\n[ADK] Integration Active: {list(result['state']['adk_results'].keys())}")

        except Exception as e:
            logger.error(f"Workflow execution failed for {agent_name}: {e}", exc_info=True)
            # Don't exit, try next query

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
