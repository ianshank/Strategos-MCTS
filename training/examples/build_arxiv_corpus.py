#!/usr/bin/env python3
"""
Example: Build arXiv Research Corpus and Integrate with RAG Pipeline

This script demonstrates how to:
1. Fetch AI/ML papers from arXiv
2. Process them into document chunks
3. Index them in Pinecone for RAG retrieval
4. Search and query the indexed papers

Usage:
    python training/examples/build_arxiv_corpus.py --mode keywords --max-papers 100
    python training/examples/build_arxiv_corpus.py --mode categories --max-papers 500
    python training/examples/build_arxiv_corpus.py --config training/config.yaml
"""

import argparse
import logging
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rag_builder import VectorIndexBuilder
from training.research_corpus_builder import (
    ResearchCorpusBuilder,
    integrate_with_rag_pipeline,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("arxiv_corpus_build.log"),
        ],
    )


def build_corpus_only(args: argparse.Namespace) -> None:
    """Build corpus without indexing."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Building arXiv Research Corpus (No Indexing)")
    logger.info("=" * 80)

    # Initialize builder
    if args.config:
        builder = ResearchCorpusBuilder(config_path=args.config)
    else:
        config = {
            "categories": args.categories.split(",") if args.categories else ["cs.AI", "cs.LG"],
            "keywords": args.keywords.split(",") if args.keywords else ["MCTS", "reinforcement learning"],
            "date_start": args.date_start,
            "date_end": args.date_end,
            "max_results": args.max_papers,
            "cache_dir": args.cache_dir,
            "chunk_size": 512,
            "chunk_overlap": 50,
        }
        builder = ResearchCorpusBuilder(config=config)

    # Build corpus
    logger.info(f"Fetching papers in '{args.mode}' mode...")
    logger.info(f"Max papers: {args.max_papers}")
    logger.info(f"Skip cached: {not args.no_cache}")

    chunk_count = 0
    sample_chunks = []

    try:
        for chunk in builder.build_corpus(
            mode=args.mode,
            max_papers=args.max_papers,
            skip_cached=not args.no_cache,
        ):
            chunk_count += 1

            # Collect samples
            if len(sample_chunks) < 5:
                sample_chunks.append(chunk)

            # Progress updates
            if chunk_count % 100 == 0:
                logger.info(f"Processed {chunk_count} chunks...")

    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
    except Exception as e:
        logger.error(f"Error during build: {e}", exc_info=True)
        return

    # Display statistics
    stats = builder.get_statistics()
    logger.info("\n" + "=" * 80)
    logger.info("CORPUS BUILDING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Papers fetched:    {stats.total_papers_fetched}")
    logger.info(f"Papers processed:  {stats.total_papers_processed}")
    logger.info(f"Chunks created:    {stats.total_chunks_created}")
    logger.info(f"Papers cached:     {stats.papers_cached}")
    logger.info(f"Papers skipped:    {stats.papers_skipped}")
    logger.info(f"Errors:            {stats.errors}")
    logger.info("\nCategory breakdown:")
    for cat, count in sorted(stats.categories_breakdown.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat}: {count}")

    # Show sample chunks
    if sample_chunks:
        logger.info("\n" + "=" * 80)
        logger.info("SAMPLE CHUNKS")
        logger.info("=" * 80)
        for i, chunk in enumerate(sample_chunks[:3], 1):
            logger.info(f"\nSample {i}:")
            logger.info(f"  Doc ID: {chunk.doc_id}")
            logger.info(f"  Section: {chunk.metadata.get('section', 'N/A')}")
            logger.info(f"  Title: {chunk.metadata.get('title', 'N/A')[:80]}...")
            logger.info(f"  Text preview: {chunk.text[:200]}...")

    # Export metadata if requested
    if args.export_metadata:
        export_path = Path(args.export_metadata)
        builder.export_metadata(export_path)
        logger.info(f"\nMetadata exported to: {export_path}")


def build_and_index(args: argparse.Namespace) -> None:
    """Build corpus and index in Pinecone."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Building arXiv Corpus and Indexing in Pinecone")
    logger.info("=" * 80)

    # Check for Pinecone API key
    if not os.environ.get("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY environment variable not set!")
        logger.error("Please set it with: export PINECONE_API_KEY='your-api-key'")
        return

    # Initialize builder
    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)

        corpus_config = config.get("research_corpus", {})
        rag_config = config.get("rag", {})

        # Override with research corpus namespace
        if "pinecone_namespace" in corpus_config:
            rag_config["namespace"] = corpus_config["pinecone_namespace"]

        builder = ResearchCorpusBuilder(config=corpus_config)
        index_builder = VectorIndexBuilder(rag_config)

    else:
        # Use command-line args
        corpus_config = {
            "categories": args.categories.split(",") if args.categories else ["cs.AI", "cs.LG"],
            "keywords": args.keywords.split(",") if args.keywords else ["MCTS", "reinforcement learning"],
            "date_start": args.date_start,
            "date_end": args.date_end,
            "max_results": args.max_papers,
            "cache_dir": args.cache_dir,
            "chunk_size": 512,
            "chunk_overlap": 50,
        }

        rag_config = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "namespace": "arxiv_research",
            "pinecone": {
                "api_key": os.environ.get("PINECONE_API_KEY"),
                "index_name": "multi-agent-mcts-rag",
                "environment": "us-east-1",
                "cloud": "aws",
            },
        }

        builder = ResearchCorpusBuilder(config=corpus_config)
        index_builder = VectorIndexBuilder(rag_config)

    # Build and index
    logger.info("Starting integrated corpus building and indexing...")

    try:
        combined_stats = integrate_with_rag_pipeline(
            corpus_builder=builder,
            rag_index_builder=index_builder,
            batch_size=args.batch_size,
        )

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("BUILD AND INDEX COMPLETE")
        logger.info("=" * 80)
        logger.info("\nCorpus Statistics:")
        logger.info(f"  Papers fetched:   {combined_stats['corpus']['papers_fetched']}")
        logger.info(f"  Papers processed: {combined_stats['corpus']['papers_processed']}")
        logger.info(f"  Chunks created:   {combined_stats['corpus']['chunks_created']}")

        logger.info("\nIndex Statistics:")
        logger.info(f"  Total documents:  {combined_stats['index']['total_documents']}")
        logger.info(f"  Total chunks:     {combined_stats['index']['total_chunks']}")
        logger.info(f"  Index size:       {combined_stats['index']['index_size_mb']:.2f} MB")
        logger.info(f"  Avg chunk length: {combined_stats['index']['avg_chunk_length']:.1f} chars")

        # Test search if requested
        if args.test_search:
            logger.info("\n" + "=" * 80)
            logger.info("TESTING SEARCH")
            logger.info("=" * 80)

            test_queries = [
                "Monte Carlo Tree Search in reinforcement learning",
                "AlphaZero and self-play training",
                "Chain-of-thought reasoning with language models",
            ]

            for query in test_queries:
                logger.info(f"\nQuery: {query}")
                results = index_builder.search(query, k=3)

                for i, result in enumerate(results, 1):
                    logger.info(f"\n  Result {i} (score: {result.score:.4f}):")
                    logger.info(f"    Doc: {result.doc_id}")
                    logger.info(f"    Title: {result.metadata.get('title', 'N/A')[:80]}...")
                    logger.info(f"    Text: {result.text[:150]}...")

        # Save index if requested
        if args.save_index:
            index_builder.save_index()
            logger.info("\nIndex metadata saved to cache directory")

    except Exception as e:
        logger.error(f"Error during build and index: {e}", exc_info=True)
        return


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build arXiv research corpus and integrate with RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build corpus from keywords (no indexing)
  python %(prog)s --mode keywords --max-papers 100

  # Build and index with config file
  python %(prog)s --config training/config.yaml --index --test-search

  # Build specific categories
  python %(prog)s --categories "cs.AI,cs.LG" --keywords "MCTS,AlphaZero" --max-papers 200

  # Resume from cache
  python %(prog)s --mode keywords --max-papers 500

  # Clear cache and rebuild
  python %(prog)s --mode keywords --max-papers 100 --no-cache --clear-cache
        """,
    )

    # Main operation mode
    parser.add_argument(
        "--mode",
        type=str,
        default="keywords",
        choices=["keywords", "categories", "all"],
        help="Fetching mode: keywords, categories, or all (default: keywords)",
    )

    parser.add_argument(
        "--index",
        action="store_true",
        help="Index papers in Pinecone (requires PINECONE_API_KEY)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (default: training/config.yaml)",
    )

    # Fetching parameters
    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of arXiv categories (e.g., 'cs.AI,cs.LG')",
    )

    parser.add_argument(
        "--keywords",
        type=str,
        help="Comma-separated list of keywords to search",
    )

    parser.add_argument(
        "--date-start",
        type=str,
        default="2020-01-01",
        help="Start date for papers (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--date-end",
        type=str,
        default="2025-12-31",
        help="End date for papers (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--max-papers",
        type=int,
        default=100,
        help="Maximum number of papers to process (default: 100)",
    )

    # Cache options
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/research_corpus",
        help="Directory for caching papers",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache and reprocess all papers",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before starting",
    )

    # Indexing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for vector indexing (default: 100)",
    )

    parser.add_argument(
        "--save-index",
        action="store_true",
        help="Save index metadata to disk",
    )

    parser.add_argument(
        "--test-search",
        action="store_true",
        help="Run test searches after indexing",
    )

    # Output options
    parser.add_argument(
        "--export-metadata",
        type=str,
        help="Export paper metadata to JSON file",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Clear cache if requested
    if args.clear_cache:
        logger.info("Clearing cache...")
        if args.config:
            builder = ResearchCorpusBuilder(config_path=args.config)
        else:
            config = {"cache_dir": args.cache_dir}
            builder = ResearchCorpusBuilder(config=config)
        builder.clear_cache()
        logger.info("Cache cleared")

    # Execute based on mode
    try:
        if args.index:
            build_and_index(args)
        else:
            build_corpus_only(args)

    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
