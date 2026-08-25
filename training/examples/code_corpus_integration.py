"""
Code Corpus Integration Example

Demonstrates how to build a code corpus from repositories and integrate it
with the RAG system for searchable code knowledge base.
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.code_corpus_builder import REPOSITORIES, CodeCorpusBuilder
from training.data_pipeline import DataOrchestrator
from training.rag_builder import VectorIndexBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_and_index_code_corpus(max_repos: int = 4):
    """
    Build code corpus and index in Pinecone for RAG.

    Args:
        max_repos: Maximum number of repositories to process
    """
    logger.info("=" * 80)
    logger.info("CODE CORPUS BUILDER - RAG INTEGRATION EXAMPLE")
    logger.info("=" * 80)

    # Step 1: Build code corpus
    logger.info("\nStep 1: Building code corpus from repositories...")
    builder = CodeCorpusBuilder("training/config.yaml")

    # Use high-priority repositories
    high_priority_repos = [r for r in REPOSITORIES if r.get("priority") == "high"][:max_repos]

    logger.info(f"Processing {len(high_priority_repos)} repositories:")
    for repo in high_priority_repos:
        logger.info(f"  - {repo['name']}: {repo['description']}")

    builder.build_corpus(repositories=high_priority_repos)

    # Step 2: Save corpus to disk
    logger.info("\nStep 2: Saving corpus to disk...")
    builder.save_corpus()

    # Step 3: Get statistics
    logger.info("\nStep 3: Corpus statistics...")
    stats = builder.get_corpus_statistics()
    print("\n" + "=" * 80)
    print("CODE CORPUS STATISTICS")
    print("=" * 80)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total repositories: {stats['total_repositories']}")
    print(f"Repositories: {', '.join(stats['repositories'])}")
    print(f"\nChunk types: {stats['chunk_types']}")
    print(f"Avg code length: {stats['avg_code_length']:.1f} chars")
    print(f"Avg complexity: {stats['avg_complexity']:.1f}")
    print("\nQuality metrics:")
    print(
        f"  - Chunks with docstrings: {stats['chunks_with_docstrings']} ({100 * stats['chunks_with_docstrings'] / stats['total_chunks']:.1f}%)"
    )
    print(
        f"  - Chunks with examples: {stats['chunks_with_examples']} ({100 * stats['chunks_with_examples'] / stats['total_chunks']:.1f}%)"
    )
    print(
        f"  - Chunks with tests: {stats['chunks_with_tests']} ({100 * stats['chunks_with_tests'] / stats['total_chunks']:.1f}%)"
    )
    print(f"  - Avg quality score: {stats['avg_quality_score']:.2f}")
    print("=" * 80)

    # Step 4: Index in Pinecone for RAG
    logger.info("\nStep 4: Indexing code corpus in Pinecone...")

    # Initialize RAG index builder with code corpus namespace
    import yaml

    with open("training/config.yaml") as f:
        config = yaml.safe_load(f)

    rag_config = config["rag"].copy()
    rag_config["namespace"] = config["code_corpus"]["pinecone_namespace"]

    index_builder = VectorIndexBuilder(rag_config)

    if index_builder.is_available:
        # Convert code chunks to document chunks
        document_chunks = list(builder.stream_document_chunks())
        logger.info(f"Converting {len(document_chunks)} code chunks to document chunks...")

        # Build index
        index_stats = index_builder.build_index(iter(document_chunks), batch_size=50)

        logger.info("\nIndex built successfully:")
        logger.info(f"  - Total documents: {index_stats.total_documents}")
        logger.info(f"  - Total chunks: {index_stats.total_chunks}")
        logger.info(f"  - Index size: {index_stats.index_size_mb:.2f} MB")
        logger.info(f"  - Avg chunk length: {index_stats.avg_chunk_length:.1f}")

        # Save index metadata
        index_builder.save_index()
        logger.info("Index metadata saved")

    else:
        logger.warning("Pinecone not available. Skipping indexing.")
        logger.warning("To enable Pinecone, set PINECONE_API_KEY environment variable")

    return builder, index_builder if index_builder.is_available else None


def search_code_examples(index_builder: VectorIndexBuilder):
    """
    Demonstrate code search functionality.

    Args:
        index_builder: Vector index builder with code corpus
    """
    logger.info("\n" + "=" * 80)
    logger.info("CODE SEARCH EXAMPLES")
    logger.info("=" * 80)

    queries = [
        "How to implement UCB1 in Python?",
        "LangGraph state machine example",
        "MCTS with neural network evaluation",
        "Reinforcement learning environment setup",
        "JAX neural network training",
    ]

    for i, query in enumerate(queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        logger.info("-" * 80)

        results = index_builder.search(query, k=3)

        if results:
            for j, result in enumerate(results, 1):
                print(f"\nResult {j} (score: {result.score:.3f}):")
                print(f"  Repository: {result.metadata.get('repo_name', 'N/A')}")
                print(f"  Function: {result.metadata.get('function_name', 'N/A')}")
                print(f"  File: {result.metadata.get('file_path', 'N/A')}")
                print(f"  Lines: {result.metadata.get('start_line', 'N/A')}-{result.metadata.get('end_line', 'N/A')}")
                print(f"  Quality: {result.metadata.get('quality_score', 0.0):.2f}")

                # Show preview of text
                text_preview = result.text[:200].replace("\n", " ")
                print(f"  Preview: {text_preview}...")
        else:
            print("  No results found")


def load_and_search_existing_corpus():
    """Load existing corpus and perform searches without rebuilding."""
    logger.info("\n" + "=" * 80)
    logger.info("LOADING EXISTING CODE CORPUS")
    logger.info("=" * 80)

    builder = CodeCorpusBuilder("training/config.yaml")

    try:
        builder.load_corpus()
        logger.info(f"Loaded {len(builder.all_chunks)} code chunks")

        # Perform simple keyword search
        logger.info("\nPerforming keyword search...")
        results = builder.search_code("MCTS tree search", top_k=5)

        print(f"\nFound {len(results)} results for 'MCTS tree search':")
        for i, chunk in enumerate(results, 1):
            print(f"\n{i}. {chunk.repo_name} - {chunk.function_name}")
            print(f"   File: {chunk.file_path}")
            print(f"   Lines: {chunk.start_line}-{chunk.end_line}")
            if chunk.docstring:
                print(f"   Doc: {chunk.docstring[:100]}...")

    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        logger.info("Run build_and_index_code_corpus() first to create the corpus")


def integrate_with_training_pipeline():
    """
    Show how code corpus integrates with the full training pipeline.
    """
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION WITH TRAINING PIPELINE")
    logger.info("=" * 80)

    # Initialize data orchestrator
    DataOrchestrator("training/config.yaml")

    # Build code corpus
    CodeCorpusBuilder("training/config.yaml")

    logger.info("\nData sources available:")
    logger.info("  1. DABStep multi-step reasoning tasks")
    logger.info("  2. PRIMUS cybersecurity documents")
    logger.info("  3. Code corpus from GitHub repositories")

    # In a real training scenario, you would:
    # 1. Build code corpus and add to RAG index
    # 2. Use RAG to retrieve relevant code examples during training
    # 3. Augment training data with code patterns and implementations
    # 4. Enable agents to reference code during inference

    logger.info("\nExample use cases:")
    logger.info("  - HRM agent learns decomposition patterns from MCTS implementations")
    logger.info("  - TRM agent references LangGraph state machine patterns")
    logger.info("  - MCTS agent learns from AlphaZero-style implementations")
    logger.info("  - All agents can query code examples during reasoning")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Code corpus builder integration example")
    parser.add_argument(
        "--mode", choices=["build", "search", "load", "integrate"], default="build", help="Operation mode"
    )
    parser.add_argument("--max-repos", type=int, default=2, help="Maximum repositories to process (for build mode)")

    args = parser.parse_args()

    if args.mode == "build":
        builder, index_builder = build_and_index_code_corpus(max_repos=args.max_repos)
        if index_builder:
            search_code_examples(index_builder)

    elif args.mode == "search":
        # Load existing index and search
        import yaml

        with open("training/config.yaml") as f:
            config = yaml.safe_load(f)

        rag_config = config["rag"].copy()
        rag_config["namespace"] = config["code_corpus"]["pinecone_namespace"]

        index_builder = VectorIndexBuilder(rag_config)

        if index_builder.is_available:
            index_builder.load_index()
            search_code_examples(index_builder)
        else:
            logger.error("Pinecone not available")

    elif args.mode == "load":
        load_and_search_existing_corpus()

    elif args.mode == "integrate":
        integrate_with_training_pipeline()


if __name__ == "__main__":
    main()
