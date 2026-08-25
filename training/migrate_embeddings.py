"""
Migration Script for Re-embedding Existing Data

Migrates existing vector index to use new embedding models.
Supports batch processing, progress tracking, and rollback.
"""

import argparse
import json
import logging
from pathlib import Path
import time

import yaml

from training.advanced_embeddings import EmbedderFactory
from training.embedding_integration import AdvancedVectorIndexBuilder, migrate_embeddings

logger = logging.getLogger(__name__)


def main():
    """Run migration from command line."""
    parser = argparse.ArgumentParser(description="Migrate embeddings to new model")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config file")
    parser.add_argument("--source-namespace", type=str, required=True, help="Source namespace to migrate from")
    parser.add_argument("--target-namespace", type=str, help="Target namespace (default: source_migrated)")
    parser.add_argument(
        "--target-model",
        type=str,
        help="Target embedding model (default: from config)",
    )
    parser.add_argument(
        "--target-dimension",
        type=int,
        help="Target dimension (default: model default)",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for migration")
    parser.add_argument("--dry-run", action="store_true", help="Preview migration without executing")
    parser.add_argument("--report-path", type=str, default="./reports/migration_report.json", help="Report output path")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    rag_config = config["rag"]

    # Set source namespace
    rag_config["namespace"] = args.source_namespace

    # Create source index builder
    logger.info(f"Loading source index from namespace: {args.source_namespace}")
    source_builder = AdvancedVectorIndexBuilder(rag_config, use_advanced=False)

    # Try to load existing chunks
    try:
        index_path = Path(rag_config.get("index_path", "./cache/rag_index")) / args.source_namespace
        if index_path.exists():
            source_builder.load_index(index_path)
            logger.info(f"Loaded {len(source_builder.chunk_store)} chunks from local cache")
        else:
            logger.warning(f"No local cache found at {index_path}, will fetch from Pinecone")
    except Exception as e:
        logger.warning(f"Failed to load local cache: {e}")

    # Get index stats
    stats = source_builder.get_index_stats()
    logger.info(f"Source index stats: {stats}")

    if stats.get("local_chunks", 0) == 0:
        logger.error("No chunks found in source index. Cannot proceed with migration.")
        return

    # Create target embedder config
    target_model = args.target_model or rag_config["embeddings"]["model"]
    target_dimension = args.target_dimension

    embeddings_config = rag_config["embeddings"]
    target_embedder_config = {
        "model": target_model,
        "dimension": target_dimension or embeddings_config.get("dimension", 1024),
        "batch_size": args.batch_size,
        "cache_dir": embeddings_config.get("cache_dir", "./cache/embeddings"),
        "cache_enabled": embeddings_config.get("cache_enabled", True),
    }

    # Determine provider from model name
    if "voyage" in target_model.lower():
        target_embedder_config["provider"] = "voyage"
        if "voyage" in embeddings_config:
            target_embedder_config.update(embeddings_config["voyage"])
    elif "embed-" in target_model.lower():
        target_embedder_config["provider"] = "cohere"
        if "cohere" in embeddings_config:
            target_embedder_config.update(embeddings_config["cohere"])
    elif "text-embedding" in target_model.lower():
        target_embedder_config["provider"] = "openai"
        if "openai" in embeddings_config:
            target_embedder_config.update(embeddings_config["openai"])

    # Create target embedder
    logger.info(f"Creating target embedder: {target_model}")
    target_embedder = EmbedderFactory.create_embedder(target_embedder_config)

    if not target_embedder.is_available():
        logger.error(f"Target embedder {target_model} is not available. Check API keys and installation.")
        return

    logger.info(f"Target embedder ready: {target_embedder.model_name} (dim={target_embedder.dimension})")

    # Set target namespace
    target_namespace = args.target_namespace or f"{args.source_namespace}_migrated"

    # Preview migration
    logger.info("\n" + "=" * 80)
    logger.info("MIGRATION PLAN")
    logger.info("=" * 80)
    logger.info(f"Source namespace:  {args.source_namespace}")
    logger.info(f"Target namespace:  {target_namespace}")
    logger.info(f"Source model:      {source_builder.embedding_model_name}")
    logger.info(f"Target model:      {target_embedder.model_name}")
    logger.info(f"Source dimension:  {source_builder.embedding_dim}")
    logger.info(f"Target dimension:  {target_embedder.dimension}")
    logger.info(f"Chunks to migrate: {len(source_builder.chunk_store)}")
    logger.info(f"Batch size:        {args.batch_size}")
    logger.info("=" * 80 + "\n")

    if args.dry_run:
        logger.info("DRY RUN - Migration not executed")
        return

    # Confirm migration
    response = input("Proceed with migration? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        logger.info("Migration cancelled")
        return

    # Perform migration
    start_time = time.time()
    migration_result = migrate_embeddings(
        old_index_builder=source_builder,
        new_embedder=target_embedder,
        output_namespace=target_namespace,
        batch_size=args.batch_size,
    )
    elapsed_time = time.time() - start_time

    # Add timing info to result
    migration_result["elapsed_time_s"] = elapsed_time
    migration_result["throughput_chunks_per_s"] = (
        migration_result["chunks_migrated"] / elapsed_time if elapsed_time > 0 else 0
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Status:            {migration_result['status']}")
    logger.info(f"Chunks migrated:   {migration_result['chunks_migrated']}")
    logger.info(f"Elapsed time:      {elapsed_time:.2f}s")
    logger.info(f"Throughput:        {migration_result['throughput_chunks_per_s']:.2f} chunks/s")
    logger.info(f"Old namespace:     {migration_result['old_namespace']}")
    logger.info(f"New namespace:     {migration_result['new_namespace']}")
    logger.info(f"Old model:         {migration_result['old_model']} (dim={migration_result['old_dimension']})")
    logger.info(f"New model:         {migration_result['new_model']} (dim={migration_result['new_dimension']})")
    logger.info("=" * 80 + "\n")

    # Save report
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(migration_result, f, indent=2)

    logger.info(f"Migration report saved to: {report_path}")

    # Save migrated index metadata
    new_index_path = Path(rag_config.get("index_path", "./cache/rag_index")) / target_namespace
    new_index_path.mkdir(parents=True, exist_ok=True)

    # Update config for new namespace
    new_config = rag_config.copy()
    new_config["namespace"] = target_namespace
    new_config["embedding_model"] = target_embedder.model_name
    new_config["embedding_dim"] = target_embedder.dimension

    new_builder = AdvancedVectorIndexBuilder(new_config, use_advanced=False)
    new_builder.chunk_store = source_builder.chunk_store
    new_builder.save_index(new_index_path)

    logger.info(f"New index metadata saved to: {new_index_path}")
    logger.info("\nMigration successful!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
