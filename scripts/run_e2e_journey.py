#!/usr/bin/env python3
"""
End-to-End RAG Journey Runner with W&B and LangSmith Integration.

This script executes a full RAG (Retrieval-Augmented Generation) user journey
with optional MCTS-guided reasoning, comprehensive telemetry, and experiment tracking.

Usage:
    python scripts/run_e2e_journey.py --query "What is X?" --mcts-enabled true
    python scripts/run_e2e_journey.py --query "Explain Y" --mcts-impl neural
    python scripts/run_e2e_journey.py --help

Features:
- Toggle MCTS on/off for A/B testing
- Support for baseline and neural MCTS implementations
- Weights & Biases experiment tracking
- LangSmith tracing for debugging
- Structured logging with metrics
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from src.config.settings import MCTSImplementation, get_settings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def init_wandb(settings: Any, run_config: dict[str, Any]) -> Any:
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        settings: Application settings
        run_config: Configuration for this run

    Returns:
        wandb run object or None if disabled
    """
    if settings.WANDB_MODE == "disabled":
        logger.info("W&B tracking disabled via WANDB_MODE=disabled")
        return None

    wandb_key = settings.get_wandb_api_key()
    if not wandb_key and settings.WANDB_MODE == "online":
        logger.warning("WANDB_API_KEY not set; W&B tracking disabled")
        return None

    try:
        import wandb

        # Set API key if provided
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key

        # Configure W&B
        wandb_settings = wandb.Settings(start_method="thread")

        run = wandb.init(
            project=settings.WANDB_PROJECT,
            entity=settings.WANDB_ENTITY,
            config=run_config,
            settings=wandb_settings,
            mode=settings.WANDB_MODE,
            tags=["e2e-journey", f"mcts-{run_config['mcts_enabled']}"],
        )

        logger.info(f"W&B run initialized: {run.url if run else 'N/A'}")
        return run

    except ImportError:
        logger.warning("wandb not installed; install with: pip install wandb")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None


def init_langsmith(settings: Any) -> None:
    """
    Initialize LangSmith tracing.

    Args:
        settings: Application settings
    """
    if not settings.LANGCHAIN_TRACING_V2:
        logger.info("LangSmith tracing disabled via LANGCHAIN_TRACING_V2=false")
        return

    langsmith_key = settings.get_langsmith_api_key()
    if not langsmith_key:
        logger.warning("LANGSMITH_API_KEY not set; LangSmith tracing disabled")
        return

    try:
        # Configure LangSmith environment
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT

        logger.info(f"LangSmith tracing enabled for project: {settings.LANGSMITH_PROJECT}")

    except Exception as e:
        logger.error(f"Failed to configure LangSmith: {e}")


async def run_rag_journey(
    query: str,
    mcts_enabled: bool,
    mcts_impl: MCTSImplementation,
    mcts_iterations: int,
) -> dict[str, Any]:
    """
    Execute a full RAG journey with the given configuration.

    This is a placeholder implementation. Replace with your actual RAG pipeline.

    Args:
        query: User query
        mcts_enabled: Whether to use MCTS
        mcts_impl: MCTS implementation variant
        mcts_iterations: Number of MCTS iterations

    Returns:
        Dictionary with results and metrics
    """
    logger.info(f"Starting RAG journey: query='{query}', mcts={mcts_enabled}, impl={mcts_impl}")

    start_time = time.time()

    # TODO: Replace with actual RAG pipeline
    # Example: agent = create_agent(mcts_enabled=mcts_enabled, ...)
    # result = await agent.run(query)

    # Placeholder logic
    await asyncio.sleep(0.5)  # Simulate processing

    response = f"[Placeholder] Processed query: {query}"
    if mcts_enabled:
        response += f" (with {mcts_impl} MCTS, {mcts_iterations} iterations)"

    elapsed = time.time() - start_time

    return {
        "query": query,
        "response": response,
        "mcts_enabled": mcts_enabled,
        "mcts_impl": mcts_impl.value,
        "mcts_iterations": mcts_iterations if mcts_enabled else 0,
        "latency_seconds": elapsed,
        "retrieval_docs": 5,  # Placeholder
        "tokens_used": 150,  # Placeholder
        "success": True,
    }


def log_metrics(wandb_run: Any, metrics: dict[str, Any]) -> None:
    """Log metrics to W&B."""
    if wandb_run:
        try:
            import wandb

            wandb.log(metrics)
            logger.info("Metrics logged to W&B")
        except Exception as e:
            logger.error(f"Failed to log to W&B: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG journey with MCTS and telemetry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query to process",
    )

    parser.add_argument(
        "--mcts-enabled",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Enable MCTS (overrides settings; true/false)",
    )

    parser.add_argument(
        "--mcts-impl",
        type=str,
        choices=["baseline", "neural"],
        default=None,
        help="MCTS implementation (overrides settings)",
    )

    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=None,
        help="Number of MCTS iterations (overrides settings)",
    )

    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="W&B mode (overrides settings)",
    )

    parser.add_argument(
        "--langsmith",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Enable LangSmith tracing (overrides settings; true/false)",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        # Load settings
        settings = get_settings()

        # Override settings from CLI args
        if args.mcts_enabled is not None:
            settings.MCTS_ENABLED = args.mcts_enabled
        if args.mcts_impl:
            settings.MCTS_IMPL = MCTSImplementation(args.mcts_impl)
        if args.mcts_iterations:
            settings.MCTS_ITERATIONS = args.mcts_iterations
        if args.wandb_mode:
            settings.WANDB_MODE = args.wandb_mode
        if args.langsmith is not None:
            settings.LANGCHAIN_TRACING_V2 = args.langsmith

        # Prepare run configuration
        run_config = {
            "query": args.query,
            "mcts_enabled": settings.MCTS_ENABLED,
            "mcts_impl": settings.MCTS_IMPL.value,
            "mcts_iterations": settings.MCTS_ITERATIONS,
            "llm_provider": settings.LLM_PROVIDER.value,
        }

        logger.info("=== E2E Journey Configuration ===")
        logger.info(f"Query: {args.query}")
        logger.info(f"MCTS Enabled: {settings.MCTS_ENABLED}")
        logger.info(f"MCTS Implementation: {settings.MCTS_IMPL.value}")
        logger.info(f"MCTS Iterations: {settings.MCTS_ITERATIONS}")
        logger.info(f"W&B Mode: {settings.WANDB_MODE}")
        logger.info(f"LangSmith Tracing: {settings.LANGCHAIN_TRACING_V2}")
        logger.info("=" * 35)

        # Initialize telemetry
        init_langsmith(settings)
        wandb_run = init_wandb(settings, run_config)

        # Execute RAG journey
        logger.info("Starting RAG journey...")
        result = await run_rag_journey(
            query=args.query,
            mcts_enabled=settings.MCTS_ENABLED,
            mcts_impl=settings.MCTS_IMPL,
            mcts_iterations=settings.MCTS_ITERATIONS,
        )

        # Log results
        logger.info("=== Journey Results ===")
        logger.info(f"Response: {result['response']}")
        logger.info(f"Latency: {result['latency_seconds']:.3f}s")
        logger.info(f"Tokens: {result['tokens_used']}")
        logger.info(f"Success: {result['success']}")
        logger.info("=" * 23)

        # Log metrics to W&B
        log_metrics(wandb_run, result)

        # Finalize W&B
        if wandb_run:
            import wandb

            wandb.finish()

        return 0 if result["success"] else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Journey failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
