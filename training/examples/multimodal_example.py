"""
Example: Using the Multi-Modal Knowledge Base

This example demonstrates how to:
1. Extract images from research papers
2. Generate descriptions using vision models
3. Extract code blocks from papers
4. Build multi-modal index
5. Perform cross-modal search (text → images, images → text)
6. Generate responses using multi-modal RAG
"""

import asyncio
import logging
from pathlib import Path

# Add parent directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.multimodal_knowledge_base import (
    CodeBlockExtractor,
    ExtractedImage,
    ImageProcessor,
    MultiModalEmbedder,
    MultiModalRAG,
    VisionModelAdapter,
    VisualIndexBuilder,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def example_1_extract_images_from_pdf():
    """Example 1: Extract images from a research paper PDF."""
    logger.info("=== Example 1: Extract Images from PDF ===")

    config = {
        "min_size": [100, 100],
        "max_size": [4096, 4096],
        "formats": ["png", "jpg"],
        "images_dir": "./cache/example_images",
    }

    processor = ImageProcessor(config)

    # Replace with path to your PDF
    pdf_path = "./cache/research_corpus/sample_paper.pdf"

    if not Path(pdf_path).exists():
        logger.warning(f"PDF not found: {pdf_path}")
        logger.info("Skipping image extraction example")
        return []

    # Extract images
    images = processor.extract_images_from_pdf(pdf_path)
    logger.info(f"Extracted {len(images)} images")

    # Save first few images
    for i, image in enumerate(images[:3]):
        save_path = processor.save_image(image)
        logger.info(f"Image {i + 1}: {image.width}x{image.height} saved to {save_path}")
        if image.caption:
            logger.info(f"  Caption: {image.caption}")

    return images


async def example_2_generate_image_descriptions(images: list[ExtractedImage]):
    """Example 2: Generate descriptions for images using vision model."""
    logger.info("=== Example 2: Generate Image Descriptions ===")

    if not images:
        logger.warning("No images provided, skipping")
        return

    config = {
        "model": "claude-3-5-sonnet-20241022",  # or "gpt-4-vision-preview"
        "max_tokens": 512,
        "temperature": 0.3,
    }

    adapter = VisionModelAdapter(config)

    # Describe first few images
    for i, image in enumerate(images[:3]):
        logger.info(f"\nProcessing image {i + 1}...")

        # Add context about the paper
        context = "This image is from a research paper about Monte Carlo Tree Search and reinforcement learning."

        description = await adapter.generate_image_description(image, context)
        image.description = description

        logger.info(f"Description: {description}")

        # Classify image type
        image_type = adapter.classify_image_type(description, image.caption)
        image.image_type = image_type
        logger.info(f"Image type: {image_type.value}")


async def example_3_extract_code_blocks():
    """Example 3: Extract code blocks from text."""
    logger.info("\n=== Example 3: Extract Code Blocks ===")

    config = {
        "min_code_lines": 3,
        "context_window": 200,
    }

    extractor = CodeBlockExtractor(config)

    # Sample text with code blocks
    sample_text = """
    # MCTS Implementation

    Here is a basic implementation of Monte Carlo Tree Search:

    ```python
    def mcts_search(root, num_simulations=1000):
        '''
        Perform MCTS from root node.
        '''
        for _ in range(num_simulations):
            node = select(root)
            reward = simulate(node)
            backpropagate(node, reward)
        return best_action(root)

    def select(node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return expand(node)
            else:
                node = best_uct(node)
        return node
    ```

    The algorithm consists of four phases: selection, expansion, simulation, and backpropagation.

    We can also implement UCB1 for node selection:

    ```python
    def ucb1_score(node, parent, c=1.414):
        exploitation = node.value / node.visits
        exploration = c * sqrt(log(parent.visits) / node.visits)
        return exploitation + exploration
    ```

    This is the standard UCB1 formula used in MCTS.
    """

    # Extract code blocks
    code_blocks = extractor.extract_code_blocks(sample_text, "mcts_tutorial")

    logger.info(f"Extracted {len(code_blocks)} code blocks")

    for i, code in enumerate(code_blocks):
        logger.info(f"\nCode block {i + 1}:")
        logger.info(f"  Language: {code.language}")
        logger.info(f"  Lines: {code.code.count(chr(10)) + 1}")
        logger.info(f"  Context: {code.context[:100]}...")
        logger.info(f"  Code preview:\n{code.code[:200]}...")


async def example_4_cross_modal_embeddings():
    """Example 4: Cross-modal embeddings with CLIP."""
    logger.info("\n=== Example 4: Cross-Modal Embeddings ===")

    config = {
        "clip_model": "openai/clip-vit-base-patch32",
        "text_model": "sentence-transformers/all-MiniLM-L6-v2",
        "use_clip": True,
    }

    embedder = MultiModalEmbedder(config)

    # Test text embeddings
    texts = [
        "MCTS tree search visualization",
        "Neural network architecture diagram",
        "Performance comparison plot",
    ]

    logger.info("Generating text embeddings...")
    for text in texts:
        embedding = embedder.embed_text(text, use_clip=True)
        logger.info(f"  '{text}': shape={embedding.shape}, norm={embedding.dot(embedding):.4f}")

    # Note: Image embedding would require actual image data
    logger.info("\nFor image embeddings, pass ExtractedImage or PIL Image objects to embed_image()")


async def example_5_multi_modal_search(images: list[ExtractedImage]):
    """Example 5: Multi-modal search (text → images)."""
    logger.info("\n=== Example 5: Multi-Modal Search ===")

    if not images:
        logger.warning("No images provided, skipping")
        return

    config = {
        "embeddings": {
            "clip_model": "openai/clip-vit-base-patch32",
            "text_model": "sentence-transformers/all-MiniLM-L6-v2",
            "use_clip": True,
        },
        "index_name": "multimodal-example",
        "namespace": "images",
        "api_key": None,  # Will use env var PINECONE_API_KEY
    }

    builder = VisualIndexBuilder(config)

    # Build index
    logger.info("Building visual index...")
    stats = builder.build_index(images)
    logger.info(f"Index stats: {stats}")

    # Search by text
    queries = [
        "tree diagram",
        "algorithm flowchart",
        "neural network architecture",
        "performance plot",
    ]

    for query in queries:
        logger.info(f"\nSearching for: '{query}'")
        results = builder.search_by_text(query, k=3)

        for i, result in enumerate(results):
            logger.info(f"  Result {i + 1}: score={result.score:.4f}")
            if hasattr(result.content, "description"):
                logger.info(f"    Description: {result.content.description[:100]}...")
            logger.info(f"    Type: {result.metadata.get('image_type', 'unknown')}")


async def example_6_full_multimodal_rag():
    """Example 6: Full multi-modal RAG pipeline."""
    logger.info("\n=== Example 6: Full Multi-Modal RAG ===")

    # Initialize multi-modal RAG
    rag = MultiModalRAG(config_path="../config.yaml")

    # Process a document (if PDF exists)
    pdf_path = "./cache/research_corpus/sample_paper.pdf"

    if Path(pdf_path).exists():
        logger.info(f"Processing document: {pdf_path}")

        doc_metadata = {
            "title": "Monte Carlo Tree Search: A Survey",
            "abstract": "This paper provides a comprehensive survey of MCTS algorithms...",
            "text": "Full text would be extracted here...",
        }

        stats = await rag.process_document(pdf_path, doc_metadata)
        logger.info(f"Processing stats: {stats}")
    else:
        logger.warning(f"PDF not found: {pdf_path}")

    # Perform multi-modal retrieval
    query = "How does MCTS balance exploration and exploitation?"

    logger.info(f"\nQuery: {query}")
    logger.info("Retrieving multi-modal context...")

    results = await rag.retrieve(query, k=5, modalities=["text", "image", "code"])

    logger.info("\n=== Retrieved Context ===")

    # Display text results
    if "text" in results:
        logger.info(f"\nText results: {len(results['text'])}")
        for i, result in enumerate(results["text"][:2]):
            logger.info(f"  {i + 1}. (score={result.score:.4f}) {result.content[:150]}...")

    # Display image results
    if "image" in results:
        logger.info(f"\nImage results: {len(results['image'])}")
        for i, result in enumerate(results["image"][:2]):
            logger.info(f"  {i + 1}. (score={result.score:.4f})")
            if hasattr(result.content, "description"):
                logger.info(f"     {result.content.description[:150]}...")

    # Display code results
    if "code" in results:
        logger.info(f"\nCode results: {len(results['code'])}")
        for i, result in enumerate(results["code"][:2]):
            logger.info(f"  {i + 1}. (score={result.score:.4f}) {result.content[:150]}...")

    # Generate response
    logger.info("\n=== Generating Response ===")
    response = await rag.generate_response(query, results, max_tokens=512)
    logger.info(f"\nResponse:\n{response}")


async def main():
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("Multi-Modal Knowledge Base Examples")
    logger.info("=" * 80)

    # Example 1: Extract images
    images = await example_1_extract_images_from_pdf()

    # Example 2: Generate descriptions (requires API key)
    if images:
        await example_2_generate_image_descriptions(images)

    # Example 3: Extract code
    await example_3_extract_code_blocks()

    # Example 4: Cross-modal embeddings
    await example_4_cross_modal_embeddings()

    # Example 5: Multi-modal search
    if images:
        await example_5_multi_modal_search(images)

    # Example 6: Full RAG pipeline
    await example_6_full_multimodal_rag()

    logger.info("\n" + "=" * 80)
    logger.info("Examples complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
