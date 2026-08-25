#!/usr/bin/env python3
"""
Extend RAG Evaluation Dataset with Synthetic Data.

This script extends the existing rag-eval-dataset in LangSmith
with synthetically generated Q&A pairs.

Usage:
    # Generate 500 new pairs and add to existing dataset
    python scripts/extend_rag_eval_dataset.py --num-samples 500

    # Create new dataset with synthetic + existing data
    python scripts/extend_rag_eval_dataset.py \\
        --num-samples 1000 \\
        --create-new-dataset \\
        --new-dataset-name "rag-eval-extended"
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.llm import create_client  # noqa: E402
from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator  # noqa: E402

logger = logging.getLogger(__name__)


def load_existing_rag_dataset() -> list[dict]:
    """Load existing RAG eval dataset from scripts/create_rag_eval_datasets.py."""
    try:
        # Import the dataset creation function
        sys.path.insert(0, str(project_root / "scripts"))

        # This returns the dataset in the expected format
        # We need to extract the examples instead
        logger.info("Loading existing RAG eval dataset from create_rag_eval_datasets.py")

        # Read the hardcoded examples from the file
        rag_script = project_root / "scripts" / "create_rag_eval_datasets.py"
        with open(rag_script) as f:
            content = f.read()

        # Parse to get examples count (this is a simplified approach)
        # In production, we'd query LangSmith API
        import re

        example_count = len(re.findall(r'"inputs":\s*{', content))

        logger.info(f"Found {example_count} examples in existing RAG eval dataset")

        # Return placeholder - in production, query LangSmith
        return []

    except Exception as e:
        logger.warning(f"Could not load existing dataset: {e}")
        return []


def merge_datasets(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Merge existing and new datasets, removing duplicates.

    Args:
        existing: Existing dataset examples
        new: New synthetic examples

    Returns:
        Merged dataset
    """
    # Create set of existing questions for deduplication
    existing_questions = {ex["inputs"]["question"] for ex in existing}

    # Add new examples that aren't duplicates
    merged = existing.copy()
    duplicates = 0

    for example in new:
        question = example["inputs"]["question"]
        if question not in existing_questions:
            merged.append(example)
            existing_questions.add(question)
        else:
            duplicates += 1

    logger.info(f"Merged {len(new)} new examples with {len(existing)} existing")
    logger.info(f"Removed {duplicates} duplicates")
    logger.info(f"Total merged dataset size: {len(merged)}")

    return merged


async def extend_dataset(args):
    """Main function to extend RAG eval dataset."""
    logger.info("=" * 70)
    logger.info("Extending RAG Evaluation Dataset")
    logger.info("=" * 70)

    # Load existing dataset
    existing_data = []
    if not args.create_new_dataset:
        logger.info("Loading existing rag-eval-dataset...")
        existing_data = load_existing_rag_dataset()

    # Create LLM client
    logger.info(f"Creating {args.provider} client...")
    llm_client = create_client(
        provider=args.provider,
        model=args.model,
        rate_limit_per_minute=args.rate_limit,
    )

    # Create generator with focus on RAG-relevant categories
    rag_categories = [
        "mcts_algorithms",
        "exploration_exploitation",
        "alphazero_neural",
        "advanced_mcts",
        "langgraph_workflows",
        "multi_agent_coordination",
    ]

    logger.info(f"Generating {args.num_samples} synthetic Q&A pairs...")
    logger.info(f"Categories: {rag_categories}")

    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/rag_extension",
        config={
            "min_question_length": 25,
            "min_answer_length": 150,
        },
    )

    # Generate
    pairs = await generator.generate_batch(
        num_samples=args.num_samples,
        categories=rag_categories,
        batch_size=args.batch_size,
    )

    # Filter for quality
    high_quality = generator.filter_by_quality(pairs, min_score=args.min_quality)

    logger.info(f"Generated {len(high_quality)} high-quality pairs")

    # Convert to LangSmith format
    new_examples = [pair.to_langsmith_format() for pair in high_quality]

    # Merge with existing
    if args.create_new_dataset:
        final_dataset = new_examples
        logger.info("Creating new dataset with synthetic data only")
    else:
        final_dataset = merge_datasets(existing_data, new_examples)

    # Save locally
    output_file = Path("training/synthetic_data/rag_extension") / "merged_rag_dataset.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(final_dataset, f, indent=2)

    logger.info(f"Saved merged dataset to {output_file}")

    # Upload to LangSmith if requested
    if args.upload_langsmith:
        if not os.getenv("LANGSMITH_API_KEY"):
            logger.warning("LANGSMITH_API_KEY not set. Skipping upload.")
        else:
            dataset_name = args.new_dataset_name or "rag-eval-extended"
            logger.info(f"Uploading to LangSmith dataset: {dataset_name}")

            try:
                from tests.utils.langsmith_tracing import create_test_dataset

                dataset_id = create_test_dataset(
                    dataset_name=dataset_name,
                    examples=final_dataset,
                    description=f"Extended RAG evaluation dataset with {len(final_dataset)} examples. "
                    f"Includes {len(existing_data)} original + {len(new_examples)} synthetic. "
                    f"Generated: {datetime.utcnow().isoformat()}",
                )

                logger.info(f"✓ Uploaded to LangSmith dataset ID: {dataset_id}")

            except Exception as e:
                logger.error(f"Failed to upload to LangSmith: {e}")

    # Statistics
    stats = generator.get_statistics()

    logger.info("\n" + "=" * 70)
    logger.info("Extension Summary")
    logger.info("=" * 70)
    logger.info(f"Existing examples: {len(existing_data)}")
    logger.info(f"New synthetic examples: {len(new_examples)}")
    logger.info(f"Total dataset size: {len(final_dataset)}")
    logger.info(f"Average quality score: {stats['avg_quality_score']:.3f}")
    logger.info(f"Estimated cost: ${stats['total_cost']:.2f}")
    logger.info("=" * 70)

    return final_dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extend RAG evaluation dataset with synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of synthetic samples to generate (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for generation (default: 10)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.6,
        help="Minimum quality score (default: 0.6)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "lmstudio"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: provider default)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Rate limit (requests/min, default: 60)",
    )
    parser.add_argument(
        "--create-new-dataset",
        action="store_true",
        help="Create new dataset instead of extending existing",
    )
    parser.add_argument(
        "--new-dataset-name",
        type=str,
        help="Name for new dataset (default: rag-eval-extended)",
    )
    parser.add_argument(
        "--upload-langsmith",
        action="store_true",
        help="Upload to LangSmith after generation",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check API key
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    elif args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Run extension
    try:
        asyncio.run(extend_dataset(args))
        logger.info("\n✓ RAG dataset extension completed successfully!")

    except KeyboardInterrupt:
        logger.info("\n⚠ Extension interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n✗ Extension failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
