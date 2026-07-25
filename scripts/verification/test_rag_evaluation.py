#!/usr/bin/env python3
"""
Test script to verify RAG evaluation setup.

This script performs a quick validation of:
- Environment configuration
- LangSmith dataset access
- RAG pipeline functionality
- Prometheus metrics collection
- RAGAS metric computation

Usage:
    python scripts/test_rag_evaluation.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check required environment variables."""
    logger.info("=== Checking Environment Configuration ===")

    required_vars = {
        "LANGSMITH_API_KEY": "LangSmith dataset access",
        "WANDB_API_KEY": "Weights & Biases logging (optional)",
    }

    optional_vars = {
        "OPENAI_API_KEY": "OpenAI LLM access",
        "ANTHROPIC_API_KEY": "Anthropic LLM access",
        "LLM_PROVIDER": "LLM provider selection",
    }

    issues = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            issues.append(f"✗ {var} not set ({description})")
            logger.error(f"{var} not set")
        else:
            logger.info(f"✓ {var} configured")

    for var, description in optional_vars.items():
        if os.getenv(var):
            logger.info(f"✓ {var} configured")
        else:
            logger.warning(f"○ {var} not set ({description})")

    if issues:
        logger.error("\nMissing required configuration:")
        for issue in issues:
            logger.error(f"  {issue}")
        return False

    logger.info("✓ Environment configuration valid\n")
    return True


def check_langsmith_dataset():
    """Check LangSmith dataset availability."""
    logger.info("=== Checking LangSmith Dataset ===")

    try:
        from langsmith import Client

        client = Client()
        datasets = list(client.list_datasets())
        dataset_names = [d.name for d in datasets]

        if "rag-eval-dataset" in dataset_names:
            logger.info("✓ Found 'rag-eval-dataset'")

            # Get dataset details
            dataset = client.read_dataset(dataset_name="rag-eval-dataset")
            examples = list(dataset.examples)
            logger.info(f"  - Contains {len(examples)} examples")
            return True
        else:
            logger.warning("✗ 'rag-eval-dataset' not found")
            logger.info(f"  Available datasets: {dataset_names}")
            logger.info("  Run: python scripts/create_rag_eval_datasets.py")
            return False

    except ImportError:
        logger.error("✗ langsmith not installed")
        logger.info("  Install with: pip install langsmith")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to access LangSmith: {e}")
        return False


def check_dependencies():
    """Check required Python packages."""
    logger.info("=== Checking Dependencies ===")

    packages = {
        "ragas": "RAGAS evaluation framework",
        "langsmith": "LangSmith client",
        "wandb": "Weights & Biases tracking (optional)",
        "pandas": "Data analysis",
        "prometheus_client": "Metrics collection",
    }

    all_installed = True
    for package, description in packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} installed")
        except ImportError:
            if package == "wandb":
                logger.warning(f"○ {package} not installed ({description})")
            else:
                logger.error(f"✗ {package} not installed ({description})")
                all_installed = False

    if not all_installed:
        logger.error("\nInstall missing packages:")
        logger.error("  pip install ragas langsmith wandb pandas prometheus-client")
        return False

    logger.info("✓ All required dependencies installed\n")
    return True


async def test_rag_pipeline():
    """Test RAG pipeline with a sample query."""
    logger.info("=== Testing RAG Pipeline ===")

    try:
        from src.adapters.llm import create_client
        from src.config.settings import get_settings
        from src.framework.graph import LangGraphMultiAgentFramework

        settings = get_settings()

        # Check LLM provider configuration
        if not settings.LLM_PROVIDER:
            logger.error("✗ LLM_PROVIDER not configured")
            return False

        logger.info(f"Using LLM provider: {settings.LLM_PROVIDER}")

        # Initialize framework
        model_adapter = create_client(
            provider=settings.LLM_PROVIDER,
            model=settings.DEFAULT_MODEL,
            timeout=30.0,
        )

        framework = LangGraphMultiAgentFramework(
            model_adapter=model_adapter,
            logger=logger,
            mcts_iterations=0,  # Disable MCTS for quick test
        )

        # Test query
        test_query = "What is Monte Carlo Tree Search?"
        logger.info(f"Processing test query: '{test_query}'")

        result = await framework.process(
            query=test_query,
            use_rag=True,
            use_mcts=False,
        )

        # Verify result structure
        if "response" not in result:
            logger.error("✗ Missing 'response' in result")
            return False

        if "metadata" not in result:
            logger.error("✗ Missing 'metadata' in result")
            return False

        # Check retrieved docs
        retrieved_docs = result.get("metadata", {}).get("retrieved_docs", [])
        if not retrieved_docs:
            logger.warning("○ No documents retrieved (RAG may not be configured)")
        else:
            logger.info(f"✓ Retrieved {len(retrieved_docs)} documents")

        logger.info(f"✓ Generated response: {result['response'][:100]}...")
        logger.info("✓ RAG pipeline functional\n")
        return True

    except Exception as e:
        logger.error(f"✗ RAG pipeline test failed: {e}", exc_info=True)
        return False


def test_prometheus_metrics():
    """Test Prometheus metrics collection."""
    logger.info("=== Testing Prometheus Metrics ===")

    try:
        from src.monitoring.prometheus_metrics import (
            RAG_QUERIES_TOTAL,
            RAG_RETRIEVAL_LATENCY,
            record_rag_retrieval,
        )

        # Test metric recording
        RAG_QUERIES_TOTAL.labels(status="test").inc()
        RAG_RETRIEVAL_LATENCY.observe(0.123)
        record_rag_retrieval(num_docs=5, relevance_scores=[0.8, 0.7, 0.6], latency=0.123)

        logger.info("✓ RAG metrics recording works")
        logger.info("  - RAG_QUERIES_TOTAL")
        logger.info("  - RAG_RETRIEVAL_LATENCY")
        logger.info("  - RAG_DOCUMENTS_RETRIEVED")
        logger.info("  - RAG_RELEVANCE_SCORES")
        logger.info("✓ Prometheus metrics functional\n")
        return True

    except Exception as e:
        logger.error(f"✗ Prometheus metrics test failed: {e}")
        return False


def test_ragas_import():
    """Test RAGAS package import and basic functionality."""
    logger.info("=== Testing RAGAS Package ===")

    try:
        from ragas import evaluate  # noqa: F401
        from ragas.metrics import (  # noqa: F401
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        logger.info("✓ RAGAS package imported successfully")
        logger.info("  - Available metrics: faithfulness, answer_relevancy, context_precision, context_recall")
        logger.info("✓ RAGAS functional\n")
        return True

    except ImportError:
        logger.error("✗ RAGAS not installed")
        logger.info("  Install with: pip install ragas")
        return False
    except Exception as e:
        logger.error(f"✗ RAGAS test failed: {e}")
        return False


async def main():
    """Run all validation checks."""
    logger.info("=" * 60)
    logger.info("RAG Evaluation Setup Validation")
    logger.info("=" * 60)
    logger.info("")

    checks = {
        "Environment": check_environment(),
        "Dependencies": check_dependencies(),
        "LangSmith Dataset": check_langsmith_dataset(),
        "RAGAS Package": test_ragas_import(),
        "Prometheus Metrics": test_prometheus_metrics(),
    }

    # Async checks
    checks["RAG Pipeline"] = await test_rag_pipeline()

    # Summary
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
        if not passed:
            all_passed = False

    logger.info("")
    if all_passed:
        logger.info("✓ All checks passed! RAG evaluation is ready to use.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run baseline evaluation:")
        logger.info("     python scripts/evaluate_rag.py --dataset rag-eval-dataset --mcts-enabled false --limit 10")
        logger.info("  2. Run MCTS evaluation:")
        logger.info("     python scripts/evaluate_rag.py --dataset rag-eval-dataset --mcts-enabled true --limit 10")
        logger.info("  3. View results in Grafana: http://localhost:3000/d/rag-evaluation")
        return 0
    else:
        logger.error("✗ Some checks failed. Please fix the issues above.")
        logger.info("")
        logger.info("Common fixes:")
        logger.info("  - Set environment variables: export LANGSMITH_API_KEY=your_key")
        logger.info("  - Install dependencies: pip install -r requirements.txt")
        logger.info("  - Create dataset: python scripts/create_rag_eval_datasets.py")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
