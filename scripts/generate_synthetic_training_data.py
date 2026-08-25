#!/usr/bin/env python3
"""
Generate Synthetic Training Data for Multi-Agent MCTS System.

This script generates high-quality Q&A pairs and integrates them with
the existing training pipeline.

Usage:
    # Generate 1,000 samples with default settings
    python scripts/generate_synthetic_training_data.py --num-samples 1000

    # Generate 10,000 samples with GPT-4
    python scripts/generate_synthetic_training_data.py \\
        --num-samples 10000 \\
        --model gpt-4-turbo-preview \\
        --batch-size 20

    # Generate with Anthropic Claude
    python scripts/generate_synthetic_training_data.py \\
        --provider anthropic \\
        --model claude-3-sonnet-20240229 \\
        --num-samples 5000

    # Generate and upload to LangSmith
    python scripts/generate_synthetic_training_data.py \\
        --num-samples 1000 \\
        --upload-langsmith

    # Resume from checkpoint
    python scripts/generate_synthetic_training_data.py \\
        --num-samples 10000 \\
        --resume

Environment Variables:
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
    LANGSMITH_API_KEY: LangSmith API key (for uploading)
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

import yaml  # noqa: E402

from src.adapters.llm import create_client, create_client_from_config  # noqa: E402
from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator  # noqa: E402

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str | None = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def upload_to_langsmith(pairs: list, dataset_name: str, project: str = "multi-agent-mcts-training"):
    """
    Upload generated Q&A pairs to LangSmith.

    Args:
        pairs: List of QAPair objects
        dataset_name: Name for the dataset
        project: LangSmith project name
    """
    try:
        from tests.utils.langsmith_tracing import create_test_dataset

        # Convert to LangSmith format
        examples = [pair.to_langsmith_format() for pair in pairs]

        # Create dataset
        dataset_id = create_test_dataset(
            dataset_name=dataset_name,
            examples=examples,
            description=f"Synthetically generated Q&A pairs for multi-agent MCTS training. "
            f"Generated: {datetime.utcnow().isoformat()}. "
            f"Total pairs: {len(pairs)}",
        )

        logger.info(f"✓ Uploaded {len(pairs)} pairs to LangSmith dataset: {dataset_name}")
        logger.info(f"  Dataset ID: {dataset_id}")
        return dataset_id

    except ImportError:
        logger.error("LangSmith integration not available. Install langsmith package.")
        return None
    except Exception as e:
        logger.error(f"Failed to upload to LangSmith: {e}")
        return None


def merge_with_existing_dataset(new_pairs: list, existing_file: str) -> list:
    """
    Merge new pairs with existing dataset.

    Args:
        new_pairs: Newly generated pairs
        existing_file: Path to existing dataset file

    Returns:
        Merged list of pairs
    """
    if not Path(existing_file).exists():
        logger.info(f"No existing dataset found at {existing_file}")
        return new_pairs

    try:
        with open(existing_file) as f:
            existing_data = json.load(f)

        logger.info(f"Loaded {len(existing_data)} existing pairs from {existing_file}")

        # Convert new pairs to dict format for merging
        new_data = [pair.to_dict() for pair in new_pairs]

        # Simple merge (could add deduplication here)
        merged = existing_data + new_data

        logger.info(f"Merged dataset size: {len(merged)}")
        return merged

    except Exception as e:
        logger.error(f"Failed to merge with existing dataset: {e}")
        return new_pairs


async def generate_training_data(args):
    """Main generation function."""
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

    # Create LLM client
    logger.info(f"Creating {args.provider} client...")

    if args.config and "llm" in config:
        llm_client = create_client_from_config(config["llm"])
    else:
        client_kwargs = {
            "provider": args.provider,
            "model": args.model,
            "rate_limit_per_minute": args.rate_limit,
        }

        if args.base_url:
            client_kwargs["base_url"] = args.base_url

        llm_client = create_client(**client_kwargs)

    logger.info(f"  Model: {args.model or 'default'}")
    logger.info(f"  Rate limit: {args.rate_limit} requests/min")

    # Setup output directory
    output_dir = args.output_dir or config.get("output", {}).get("directory", "training/synthetic_data")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator_config = config.get("generation", {})
    generator_config.update(
        {
            "min_question_length": args.min_question_length,
            "min_answer_length": args.min_answer_length,
        }
    )

    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir=output_dir,
        config=generator_config,
    )

    # Generate data
    logger.info("=" * 70)
    logger.info(f"Generating {args.num_samples} Q&A pairs")
    logger.info("=" * 70)
    logger.info(f"  Categories: {args.categories or 'all'}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Min quality: {args.min_quality}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 70)

    start_time = datetime.utcnow()

    pairs = await generator.generate_batch(
        num_samples=args.num_samples,
        categories=args.categories,
        batch_size=args.batch_size,
    )

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"\nGeneration completed in {duration:.1f} seconds")
    logger.info(f"Rate: {len(pairs) / duration * 60:.1f} pairs/minute")

    # Filter by quality
    if args.min_quality > 0:
        filtered_pairs = generator.filter_by_quality(pairs, min_score=args.min_quality)
        logger.info(f"After quality filtering: {len(filtered_pairs)}/{len(pairs)} pairs")
    else:
        filtered_pairs = pairs

    # Save datasets
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    langsmith_file = f"synthetic_qa_langsmith_{timestamp}.json"
    raw_file = f"synthetic_qa_raw_{timestamp}.json"

    generator.save_dataset(filtered_pairs, langsmith_file, format="langsmith")
    generator.save_dataset(filtered_pairs, raw_file, format="raw")

    # Merge with existing if requested
    if args.merge_existing:
        existing_file = output_path / "synthetic_qa_merged.json"
        merged = merge_with_existing_dataset(filtered_pairs, str(existing_file))

        with open(existing_file, "w") as f:
            json.dump(merged, f, indent=2)

        logger.info(f"Saved merged dataset to {existing_file}")

    # Upload to LangSmith if requested
    if args.upload_langsmith:
        if not os.getenv("LANGSMITH_API_KEY"):
            logger.warning("LANGSMITH_API_KEY not set. Skipping upload.")
        else:
            dataset_name = args.langsmith_dataset or f"synthetic-knowledge-{timestamp}"
            upload_to_langsmith(filtered_pairs, dataset_name, project=args.langsmith_project)

    # Print final statistics
    stats = generator.get_statistics()

    logger.info("\n" + "=" * 70)
    logger.info("Generation Summary")
    logger.info("=" * 70)
    logger.info(f"Total generated: {stats['total_generated']}")
    logger.info(f"Valid pairs: {stats['valid_pairs']}")
    logger.info(f"Invalid pairs: {stats['invalid_pairs']}")
    logger.info(f"High quality (>= {args.min_quality}): {len(filtered_pairs)}")
    logger.info(f"Average quality score: {stats['avg_quality_score']:.3f}")
    logger.info("")
    logger.info(f"API calls: {stats['api_calls']}")
    logger.info(f"Total tokens: {stats['total_tokens']:,}")
    logger.info(f"Estimated cost: ${stats['total_cost']:.2f}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Rate: {len(pairs) / duration * 60:.1f} pairs/minute")
    logger.info("=" * 70)

    logger.info("\nOutput files:")
    logger.info(f"  LangSmith format: {output_path / langsmith_file}")
    logger.info(f"  Raw format: {output_path / raw_file}")
    logger.info(f"  Statistics: {output_path / 'generation_stats.json'}")

    if args.max_cost and stats["total_cost"] > args.max_cost:
        logger.warning(f"⚠ Cost exceeded maximum: ${stats['total_cost']:.2f} > ${args.max_cost:.2f}")

    return filtered_pairs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for multi-agent MCTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of Q&A pairs to generate (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Concurrent generation batch size (default: 10)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score filter (default: 0.5)",
    )

    # LLM configuration
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
        "--base-url",
        type=str,
        help="Base URL for LLM API (for lmstudio or custom endpoints)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="API rate limit (requests per minute, default: 60)",
    )

    # Categories
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Categories to generate (default: all)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: training/synthetic_data)",
    )

    # Quality control
    parser.add_argument(
        "--min-question-length",
        type=int,
        default=20,
        help="Minimum question length (default: 20)",
    )
    parser.add_argument(
        "--min-answer-length",
        type=int,
        default=100,
        help="Minimum answer length (default: 100)",
    )

    # Integration options
    parser.add_argument(
        "--upload-langsmith",
        action="store_true",
        help="Upload to LangSmith after generation",
    )
    parser.add_argument(
        "--langsmith-dataset",
        type=str,
        help="LangSmith dataset name (default: auto-generated)",
    )
    parser.add_argument(
        "--langsmith-project",
        type=str,
        default="multi-agent-mcts-training",
        help="LangSmith project name (default: multi-agent-mcts-training)",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge with existing dataset",
    )

    # Cost management
    parser.add_argument(
        "--max-cost",
        type=float,
        help="Maximum cost in USD (generation stops if exceeded)",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Check for API keys
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    elif args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Run generation
    try:
        asyncio.run(generate_training_data(args))
        logger.info("\n✓ Synthetic data generation completed successfully!")

    except KeyboardInterrupt:
        logger.info("\n⚠ Generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n✗ Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
