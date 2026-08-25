"""
Multi-Modal Knowledge Base - Integration Example

This example shows how to integrate the multi-modal knowledge base
with the existing research corpus builder to create a complete
pipeline that:

1. Fetches papers from arXiv
2. Extracts images, code, and text
3. Generates descriptions using vision models
4. Builds multi-modal vector index
5. Enables cross-modal search and retrieval

This is the recommended way to use the multimodal system in production.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.multimodal_knowledge_base import MultiModalRAG
from training.research_corpus_builder import ResearchCorpusBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def download_pdf(pdf_url: str, save_dir: Path) -> Path | None:
    """
    Download PDF from URL.

    Args:
        pdf_url: URL to PDF
        save_dir: Directory to save PDF

    Returns:
        Path to downloaded PDF or None if failed
    """
    import httpx

    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = pdf_url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        save_path = save_dir / filename

        # Download if not exists
        if not save_path.exists():
            logger.info(f"Downloading: {pdf_url}")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                with open(save_path, "wb") as f:
                    f.write(response.content)

            logger.info(f"Saved to: {save_path}")

        return save_path

    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        return None


async def process_arxiv_papers_with_multimodal(
    max_papers: int = 10,
    keywords: list[str] | None = None,
):
    """
    Complete pipeline: Fetch papers from arXiv and process with multi-modal RAG.

    Args:
        max_papers: Maximum number of papers to process
        keywords: Optional list of keywords to filter papers
    """
    logger.info("=" * 80)
    logger.info("Multi-Modal ArXiv Paper Processing Pipeline")
    logger.info("=" * 80)

    # Step 1: Initialize research corpus builder
    logger.info("\n[1/5] Initializing research corpus builder...")
    corpus_config = {
        "categories": ["cs.AI", "cs.LG"],
        "keywords": keywords
        or [
            "MCTS",
            "Monte Carlo Tree Search",
            "AlphaZero",
            "reinforcement learning",
        ],
        "date_start": "2023-01-01",
        "date_end": "2024-12-31",
        "max_results": max_papers,
        "cache_dir": "./cache/research_corpus",
        "chunk_size": 512,
    }

    corpus_builder = ResearchCorpusBuilder(config=corpus_config)

    # Step 2: Initialize multi-modal RAG
    logger.info("\n[2/5] Initializing multi-modal RAG system...")
    multimodal_rag = MultiModalRAG(config_path="../config.yaml")

    # Step 3: Fetch and process papers
    logger.info(f"\n[3/5] Fetching papers from arXiv (max {max_papers})...")
    papers = list(
        corpus_builder.fetcher.fetch_papers_by_keywords(max_per_keyword=max_papers // len(corpus_config["keywords"]))
    )

    logger.info(f"Fetched {len(papers)} papers")

    # Step 4: Process each paper
    logger.info("\n[4/5] Processing papers...")

    pdf_dir = Path("./cache/arxiv_pdfs")
    results = []

    for i, paper in enumerate(papers, 1):
        logger.info(f"\n--- Paper {i}/{len(papers)} ---")
        logger.info(f"Title: {paper.title}")
        logger.info(f"Authors: {', '.join(paper.authors[:3])}...")
        logger.info(f"ArXiv ID: {paper.arxiv_id}")

        try:
            # Download PDF
            pdf_path = await download_pdf(paper.pdf_url, pdf_dir)

            if not pdf_path:
                logger.warning("Skipping paper (PDF download failed)")
                continue

            # Prepare metadata
            doc_metadata = {
                "title": paper.title,
                "abstract": paper.abstract,
                "arxiv_id": paper.arxiv_id,
                "authors": paper.authors,
                "categories": paper.categories,
                "published": paper.published.isoformat(),
                "pdf_url": paper.pdf_url,
                "keywords": paper.keywords,
            }

            # Process with multi-modal RAG
            logger.info("Processing multi-modal content...")
            stats = await multimodal_rag.process_document(pdf_path, doc_metadata)

            # Log results
            logger.info(f"✓ Images extracted: {stats['images_extracted']}")
            logger.info(f"✓ Images described: {stats['images_described']}")
            logger.info(f"✓ Code blocks: {stats.get('code_blocks_extracted', 0)}")

            results.append(
                {
                    "paper": paper,
                    "pdf_path": pdf_path,
                    "stats": stats,
                }
            )

            # Add text chunks to RAG index
            text_chunks = corpus_builder.processor.process_paper(paper)
            multimodal_rag.text_index.add_documents(iter(text_chunks))

        except Exception as e:
            logger.error(f"Error processing paper: {e}", exc_info=True)
            continue

    # Step 5: Summary and search examples
    logger.info("\n[5/5] Processing complete!")
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info(f"  Papers processed: {len(results)}")
    logger.info(f"  Total images: {sum(r['stats']['images_extracted'] for r in results)}")
    logger.info(f"  Total code blocks: {sum(r['stats'].get('code_blocks_extracted', 0) for r in results)}")

    # Example queries
    logger.info("\n" + "=" * 80)
    logger.info("Example Multi-Modal Queries")
    logger.info("=" * 80)

    await demo_queries(multimodal_rag)

    return results


async def demo_queries(multimodal_rag: MultiModalRAG):
    """
    Demonstrate various multi-modal queries.

    Args:
        multimodal_rag: Initialized MultiModalRAG instance
    """
    queries = [
        {
            "query": "How does MCTS balance exploration and exploitation?",
            "modalities": ["text", "code"],
            "description": "Text + Code search",
        },
        {
            "query": "MCTS tree visualization",
            "modalities": ["image"],
            "description": "Image-only search",
        },
        {
            "query": "Neural network architecture for AlphaZero",
            "modalities": ["text", "image"],
            "description": "Text + Image search",
        },
        {
            "query": "UCB1 formula implementation",
            "modalities": ["code", "text"],
            "description": "Code + Text search",
        },
    ]

    for i, query_info in enumerate(queries, 1):
        logger.info(f"\n--- Query {i}: {query_info['description']} ---")
        logger.info(f"Query: {query_info['query']}")
        logger.info(f"Modalities: {', '.join(query_info['modalities'])}")

        try:
            # Retrieve results
            results = await multimodal_rag.retrieve(
                query=query_info["query"],
                k=5,
                modalities=query_info["modalities"],
            )

            # Display results
            for modality, items in results.items():
                if items:
                    logger.info(f"\n{modality.upper()} Results ({len(items)}):")
                    for j, result in enumerate(items[:3], 1):
                        logger.info(f"  {j}. Score: {result.score:.4f}")

                        if modality == "image" and hasattr(result.content, "description"):
                            logger.info(f"     Type: {result.content.image_type.value}")
                            logger.info(f"     Description: {result.content.description[:100]}...")
                        elif modality == "code":
                            logger.info(f"     {result.content[:150]}...")
                        else:
                            logger.info(f"     {result.content[:150]}...")

            # Generate response for first query
            if i == 1:
                logger.info("\n--- Generated Response ---")
                response = await multimodal_rag.generate_response(
                    query=query_info["query"],
                    context=results,
                    max_tokens=512,
                )
                logger.info(response[:500] + "...")

        except Exception as e:
            logger.error(f"Error executing query: {e}")


async def analyze_paper_images(multimodal_rag: MultiModalRAG, pdf_path: Path):
    """
    Detailed analysis of images in a single paper.

    Args:
        multimodal_rag: MultiModalRAG instance
        pdf_path: Path to PDF
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"Analyzing Images in: {pdf_path.name}")
    logger.info("=" * 80)

    # Extract images
    images = multimodal_rag.image_processor.extract_images_from_pdf(pdf_path)
    logger.info(f"\nExtracted {len(images)} images")

    # Analyze each image
    for i, image in enumerate(images[:5], 1):  # First 5 images
        logger.info(f"\n--- Image {i} ---")
        logger.info(f"Size: {image.width}x{image.height}")
        logger.info(f"Page: {image.page_number}")

        if image.caption:
            logger.info(f"Caption: {image.caption}")

        # Generate description
        description = await multimodal_rag.vision_adapter.generate_image_description(
            image, context=f"From paper: {pdf_path.stem}"
        )

        logger.info(f"Description: {description}")

        # Classify type
        image_type = multimodal_rag.vision_adapter.classify_image_type(description, image.caption)
        logger.info(f"Type: {image_type.value}")


async def search_by_image_type(multimodal_rag: MultiModalRAG):
    """
    Search for images by specific types.

    Args:
        multimodal_rag: MultiModalRAG instance
    """
    from training.multimodal_knowledge_base import ImageType

    logger.info("\n" + "=" * 80)
    logger.info("Searching by Image Type")
    logger.info("=" * 80)

    image_types = [
        (ImageType.ARCHITECTURE_DIAGRAM, "system architecture"),
        (ImageType.TREE_VISUALIZATION, "tree structure"),
        (ImageType.PLOT_CHART, "performance comparison"),
        (ImageType.NEURAL_NETWORK, "neural network layers"),
    ]

    for img_type, query in image_types:
        logger.info(f"\n--- Searching for {img_type.value} ---")
        logger.info(f"Query: {query}")

        results = multimodal_rag.visual_index.search_by_text(
            query=query,
            k=3,
            filter_type=img_type,
        )

        logger.info(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. Score: {result.score:.4f}")
            if hasattr(result.content, "description"):
                logger.info(f"     {result.content.description[:100]}...")


async def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-modal ArXiv paper processing")
    parser.add_argument("--max-papers", type=int, default=5, help="Maximum number of papers to process")
    parser.add_argument("--keywords", nargs="+", default=None, help="Keywords to search for")
    parser.add_argument("--analyze-images", action="store_true", help="Perform detailed image analysis")
    parser.add_argument("--search-by-type", action="store_true", help="Search images by type")

    args = parser.parse_args()

    # Run main pipeline
    results = await process_arxiv_papers_with_multimodal(
        max_papers=args.max_papers,
        keywords=args.keywords,
    )

    # Optional: Additional analysis
    if results:
        multimodal_rag = MultiModalRAG(config_path="../config.yaml")

        if args.analyze_images and results:
            pdf_path = results[0]["pdf_path"]
            await analyze_paper_images(multimodal_rag, pdf_path)

        if args.search_by_type:
            await search_by_image_type(multimodal_rag)

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
