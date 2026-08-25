"""
Example: Synthetic Knowledge Generation for Training Data

This example demonstrates how to use the SyntheticKnowledgeGenerator
to create high-quality Q&A pairs for training multi-agent MCTS systems.

Usage:
    python examples/synthetic_data_generation_example.py

Requirements:
    - OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    - Or LM Studio running locally
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.llm import create_client  # noqa: E402
from training.synthetic_knowledge_generator import (  # noqa: E402
    QUESTION_TEMPLATES,
    SyntheticKnowledgeGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def example_basic_generation():
    """Example: Basic Q&A generation."""
    logger.info("=" * 70)
    logger.info("Example 1: Basic Q&A Generation")
    logger.info("=" * 70)

    # Create LLM client (OpenAI)
    llm_client = create_client(
        provider="openai",
        model="gpt-3.5-turbo",  # Using cheaper model for example
        rate_limit_per_minute=60,
    )

    # Create generator
    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/examples",
    )

    # Generate small batch
    logger.info("Generating 10 Q&A pairs...")
    pairs = await generator.generate_batch(
        num_samples=10,
        categories=["mcts_algorithms", "exploration_exploitation"],
        batch_size=5,
    )

    # Show results
    logger.info(f"\nGenerated {len(pairs)} Q&A pairs")
    logger.info("\nSample Q&A:")
    if pairs:
        sample = pairs[0]
        logger.info(f"Question: {sample.question}")
        logger.info(f"Answer (first 200 chars): {sample.answer[:200]}...")
        logger.info(f"Quality Score: {sample.quality_score:.2f}")
        logger.info(f"Category: {sample.metadata['category']}")
        logger.info(f"Difficulty: {sample.metadata['difficulty']}")

    # Save
    generator.save_dataset(pairs, "example_basic.json", format="langsmith")

    # Statistics
    stats = generator.get_statistics()
    logger.info("\nStatistics:")
    logger.info(f"  Total API calls: {stats['api_calls']}")
    logger.info(f"  Total tokens: {stats['total_tokens']:,}")
    logger.info(f"  Estimated cost: ${stats['total_cost']:.4f}")

    return pairs


async def example_category_specific():
    """Example: Generate for specific categories."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Category-Specific Generation")
    logger.info("=" * 70)

    llm_client = create_client(
        provider="openai",
        model="gpt-3.5-turbo",
        rate_limit_per_minute=60,
    )

    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/examples",
    )

    # Generate for specific categories
    categories_to_generate = [
        "langgraph_workflows",
        "multi_agent_coordination",
        "system_design",
    ]

    logger.info(f"Generating for categories: {categories_to_generate}")

    pairs = await generator.generate_batch(
        num_samples=15,
        categories=categories_to_generate,
        batch_size=5,
    )

    # Show category distribution
    from collections import Counter

    category_counts = Counter(pair.metadata["category"] for pair in pairs)
    logger.info("\nCategory distribution:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count}")

    generator.save_dataset(pairs, "example_categories.json", format="langsmith")

    return pairs


async def example_high_quality_filtering():
    """Example: Generate and filter for high quality."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: High-Quality Filtering")
    logger.info("=" * 70)

    llm_client = create_client(
        provider="openai",
        model="gpt-4-turbo-preview",  # Better model for higher quality
        rate_limit_per_minute=30,
    )

    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/examples",
    )

    # Generate
    logger.info("Generating Q&A pairs with quality focus...")
    pairs = await generator.generate_batch(
        num_samples=20,
        categories=["alphazero_neural", "advanced_mcts"],
        batch_size=5,
    )

    # Show quality distribution
    quality_scores = [pair.quality_score for pair in pairs]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    logger.info("\nQuality distribution:")
    logger.info(f"  Average: {avg_quality:.3f}")
    logger.info(f"  Min: {min(quality_scores):.3f}")
    logger.info(f"  Max: {max(quality_scores):.3f}")

    # Filter for high quality
    high_quality = generator.filter_by_quality(pairs, min_score=0.7)
    logger.info(f"\nHigh quality pairs (>= 0.7): {len(high_quality)}/{len(pairs)}")

    # Save both versions
    generator.save_dataset(pairs, "example_all_quality.json", format="langsmith")
    generator.save_dataset(high_quality, "example_high_quality.json", format="langsmith")

    return high_quality


async def example_with_reasoning_paths():
    """Example: Generate with multiple reasoning paths."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Multiple Reasoning Paths")
    logger.info("=" * 70)

    llm_client = create_client(
        provider="openai",
        model="gpt-3.5-turbo",
        rate_limit_per_minute=60,
    )

    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/examples",
    )

    # Generate single high-quality pair with reasoning
    pairs = await generator.generate_batch(
        num_samples=5,
        categories=["mcts_algorithms"],
        batch_size=1,
    )

    # Show reasoning paths
    if pairs:
        pair = pairs[0]
        logger.info(f"\nQuestion: {pair.question}")
        logger.info(f"Answer: {pair.answer[:300]}...")
        logger.info(f"\nReasoning Paths ({len(pair.reasoning_paths)}):")
        for i, path in enumerate(pair.reasoning_paths, 1):
            logger.info(f"\n  Path {i}: {path[:200]}...")

    generator.save_dataset(pairs, "example_reasoning.json", format="raw")

    return pairs


async def example_local_llm():
    """Example: Use local LLM (LM Studio)."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Local LLM Generation")
    logger.info("=" * 70)

    # Check if LM Studio is available
    try:
        llm_client = create_client(
            provider="lmstudio",
            base_url="http://localhost:1234/v1",
            timeout=180.0,
        )

        generator = SyntheticKnowledgeGenerator(
            llm_client=llm_client,
            output_dir="training/synthetic_data/examples",
        )

        logger.info("Generating with local LLM...")
        pairs = await generator.generate_batch(
            num_samples=5,
            categories=["mcts_algorithms"],
            batch_size=2,
        )

        logger.info(f"Generated {len(pairs)} pairs using local LLM")
        generator.save_dataset(pairs, "example_local.json", format="langsmith")

        return pairs

    except Exception as e:
        logger.warning(f"Local LLM not available: {e}")
        logger.info("Start LM Studio and try again")
        return []


async def example_resume_from_checkpoint():
    """Example: Resume generation from checkpoint."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 6: Resume from Checkpoint")
    logger.info("=" * 70)

    llm_client = create_client(
        provider="openai",
        model="gpt-3.5-turbo",
        rate_limit_per_minute=60,
    )

    # First generation
    generator1 = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/examples/resume_test",
    )

    logger.info("Phase 1: Generating initial batch...")
    await generator1.generate_batch(num_samples=5, batch_size=2)
    generator1._save_checkpoint()

    initial_count = generator1.stats["valid_pairs"]
    logger.info(f"Generated {initial_count} pairs in phase 1")

    # Resume with new generator instance
    logger.info("\nPhase 2: Resuming from checkpoint...")
    generator2 = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/examples/resume_test",
    )

    logger.info(f"Loaded checkpoint: {generator2.stats['valid_pairs']} existing pairs")

    pairs2 = await generator2.generate_batch(num_samples=5, batch_size=2)

    total_count = generator2.stats["valid_pairs"]
    logger.info(f"\nTotal pairs after resume: {total_count}")
    logger.info(f"New pairs generated: {total_count - initial_count}")

    return pairs2


async def example_all_categories():
    """Example: Generate samples from all categories."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 7: All Categories")
    logger.info("=" * 70)

    llm_client = create_client(
        provider="openai",
        model="gpt-3.5-turbo",
        rate_limit_per_minute=60,
    )

    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data/examples",
    )

    # Show available categories
    categories = list(QUESTION_TEMPLATES.keys())
    logger.info(f"Available categories ({len(categories)}):")
    for cat in categories:
        num_templates = len(QUESTION_TEMPLATES[cat])
        logger.info(f"  - {cat}: {num_templates} templates")

    # Generate from all
    logger.info(f"\nGenerating from all {len(categories)} categories...")
    pairs = await generator.generate_batch(
        num_samples=30,  # ~3-4 per category
        categories=None,  # None = all categories
        batch_size=10,
    )

    # Category distribution
    from collections import Counter

    category_counts = Counter(pair.metadata["category"] for pair in pairs)

    logger.info("\nGenerated distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"  {category}: {count}")

    generator.save_dataset(pairs, "example_all_categories.json", format="langsmith")

    stats = generator.get_statistics()
    logger.info(f"\nTotal cost: ${stats['total_cost']:.4f}")

    return pairs


async def main():
    """Run all examples."""
    logger.info("Synthetic Knowledge Generation Examples")
    logger.info("=" * 70)

    # Check for API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        logger.error("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    try:
        # Run examples
        await example_basic_generation()
        await example_category_specific()
        await example_high_quality_filtering()
        await example_with_reasoning_paths()
        await example_resume_from_checkpoint()
        await example_all_categories()

        # Optional: local LLM example
        # await example_local_llm()

        logger.info("\n" + "=" * 70)
        logger.info("All examples completed successfully!")
        logger.info("=" * 70)
        logger.info("\nGenerated files saved to: training/synthetic_data/examples/")
        logger.info("\nNext steps:")
        logger.info("  1. Review generated Q&A pairs")
        logger.info("  2. Adjust quality thresholds as needed")
        logger.info("  3. Run full generation with desired sample count")
        logger.info("  4. Upload to LangSmith for evaluation")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
