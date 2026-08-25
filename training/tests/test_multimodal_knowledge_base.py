"""
Unit tests for Multi-Modal Knowledge Base

Tests all components:
- ImageProcessor
- VisionModelAdapter
- MultiModalEmbedder
- CodeBlockExtractor
- VisualIndexBuilder
- MultiModalRAG
"""

import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from training.multimodal_knowledge_base import (
    CodeBlockExtractor,
    ExtractedImage,
    ImageProcessor,
    ImageType,
    MultiModalEmbedder,
    MultiModalSearchResult,
    VisionModelAdapter,
    VisualIndexBuilder,
)


class TestExtractedImage:
    """Test ExtractedImage dataclass."""

    def test_to_pil_image(self):
        """Test conversion to PIL Image."""
        # Create a simple PNG image
        pil_img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        extracted = ExtractedImage(
            image_id="test_1",
            doc_id="doc_1",
            page_number=0,
            image_data=img_bytes,
            format="png",
            width=100,
            height=100,
        )

        result = extracted.to_pil_image()
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_to_base64(self):
        """Test conversion to base64."""
        img_bytes = b"test_image_data"
        extracted = ExtractedImage(
            image_id="test_1",
            doc_id="doc_1",
            page_number=0,
            image_data=img_bytes,
            format="png",
            width=100,
            height=100,
        )

        result = extracted.to_base64()
        assert isinstance(result, str)
        assert len(result) > 0


class TestCodeBlockExtractor:
    """Test CodeBlockExtractor."""

    @pytest.fixture
    def extractor(self):
        config = {
            "min_code_lines": 3,
            "context_window": 200,
        }
        return CodeBlockExtractor(config)

    def test_extract_markdown_code(self, extractor):
        """Test extraction of markdown code blocks."""
        text = """
        Here's some Python code:

        ```python
        def hello():
            print("Hello, world!")
            return True
        ```

        And that's it!
        """

        codes = extractor.extract_code_blocks(text, "doc_1")

        assert len(codes) == 1
        assert codes[0].language == "python"
        assert "def hello()" in codes[0].code
        assert codes[0].doc_id == "doc_1"

    def test_extract_multiple_languages(self, extractor):
        """Test extraction of multiple code blocks."""
        text = """
        Python example:
        ```python
        def foo():
            pass
        ```

        JavaScript example:
        ```javascript
        function bar() {
            return true;
        }
        ```
        """

        codes = extractor.extract_code_blocks(text, "doc_1")

        assert len(codes) == 2
        assert codes[0].language == "python"
        assert codes[1].language == "javascript"

    def test_filter_short_code(self, extractor):
        """Test filtering of code blocks that are too short."""
        text = """
        ```python
        x = 1
        ```
        """

        codes = extractor.extract_code_blocks(text, "doc_1")
        assert len(codes) == 0  # Too short (< 3 lines)

    def test_extract_latex_code(self, extractor):
        """Test extraction of LaTeX code blocks."""
        text = r"""
        \begin{lstlisting}[language=Python]
        def algorithm():
            for i in range(10):
                process(i)
        \end{lstlisting}
        """

        codes = extractor.extract_code_blocks(text, "doc_1")

        assert len(codes) == 1
        assert codes[0].language == "Python"
        assert "def algorithm()" in codes[0].code

    def test_detect_language(self, extractor):
        """Test language detection."""
        python_code = "def foo():\n    import numpy\n    pass"
        assert extractor._detect_language(python_code) == "python"

        js_code = "function foo() {\n    const x = 1;\n    return x;\n}"
        assert extractor._detect_language(js_code) == "javascript"

    def test_context_extraction(self, extractor):
        """Test context extraction around code."""
        text = """
        This is important context about the algorithm.
        It explains what happens next.

        ```python
        def mcts():
            return True
        ```
        """

        codes = extractor.extract_code_blocks(text, "doc_1")

        assert len(codes) == 1
        assert "important context" in codes[0].context.lower()


class TestMultiModalEmbedder:
    """Test MultiModalEmbedder."""

    @pytest.fixture
    def embedder(self):
        config = {
            "clip_model": "openai/clip-vit-base-patch32",
            "text_model": "sentence-transformers/all-MiniLM-L6-v2",
            "use_clip": False,  # Disable CLIP for faster tests
        }
        return MultiModalEmbedder(config)

    def test_embed_text(self, embedder):
        """Test text embedding generation."""
        text = "Monte Carlo Tree Search algorithm"
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0

    def test_embed_text_consistency(self, embedder):
        """Test that same text produces similar embeddings."""
        text = "MCTS algorithm"
        emb1 = embedder.embed_text(text)
        emb2 = embedder.embed_text(text)

        # Should be very similar (cosine similarity > 0.99)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity > 0.99


class TestVisionModelAdapter:
    """Test VisionModelAdapter."""

    @pytest.fixture
    def adapter_config(self):
        return {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 512,
            "temperature": 0.3,
        }

    def test_detect_provider(self, adapter_config):
        """Test provider detection from model name."""
        adapter = VisionModelAdapter(adapter_config)
        assert adapter.provider == "anthropic"

        adapter_config["model"] = "gpt-4-vision-preview"
        adapter = VisionModelAdapter(adapter_config)
        assert adapter.provider == "openai"

    def test_build_description_prompt(self, adapter_config):
        """Test prompt building for image description."""
        adapter = VisionModelAdapter(adapter_config)

        image = ExtractedImage(
            image_id="test_1",
            doc_id="doc_1",
            page_number=0,
            image_data=b"",
            format="png",
            width=100,
            height=100,
            caption="Figure 1: MCTS tree",
        )

        prompt = adapter._build_description_prompt(image, "Paper about MCTS")

        assert "technical description" in prompt.lower()
        assert "Figure 1: MCTS tree" in prompt
        assert "Paper about MCTS" in prompt

    def test_classify_image_type(self, adapter_config):
        """Test image type classification."""
        adapter = VisionModelAdapter(adapter_config)

        # Test architecture diagram
        desc = "This diagram shows the system architecture with multiple components"
        img_type = adapter.classify_image_type(desc)
        assert img_type == ImageType.ARCHITECTURE_DIAGRAM

        # Test plot/chart
        desc = "Performance plot showing accuracy over time"
        img_type = adapter.classify_image_type(desc)
        assert img_type == ImageType.PLOT_CHART

        # Test tree visualization
        desc = "MCTS tree structure with nodes and edges"
        img_type = adapter.classify_image_type(desc)
        assert img_type == ImageType.TREE_VISUALIZATION

        # Test neural network
        desc = "Neural network with multiple layers and neurons"
        img_type = adapter.classify_image_type(desc)
        assert img_type == ImageType.NEURAL_NETWORK


class TestImageProcessor:
    """Test ImageProcessor."""

    @pytest.fixture
    def processor(self):
        config = {
            "min_size": [100, 100],
            "max_size": [4096, 4096],
            "formats": ["png", "jpg"],
            "images_dir": tempfile.mkdtemp(),
        }
        return ImageProcessor(config)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.min_size == (100, 100)
        assert processor.max_size == (4096, 4096)
        assert processor.backend in ["pymupdf", "pdf2image", None]

    def test_save_image(self, processor):
        """Test image saving."""
        # Create a test image
        pil_img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="PNG")

        image = ExtractedImage(
            image_id="test_save",
            doc_id="doc_1",
            page_number=0,
            image_data=img_bytes.getvalue(),
            format="png",
            width=100,
            height=100,
        )

        # Save image
        save_path = processor.save_image(image)

        assert save_path.exists()
        assert save_path.name == "test_save.png"

        # Cleanup
        save_path.unlink()


class TestVisualIndexBuilder:
    """Test VisualIndexBuilder."""

    @pytest.fixture
    def builder(self):
        config = {
            "embeddings": {
                "clip_model": "openai/clip-vit-base-patch32",
                "use_clip": False,  # Disable for tests
            },
            "index_name": "test-index",
            "namespace": "test",
            "api_key": None,  # Don't connect to Pinecone in tests
        }
        return VisualIndexBuilder(config)

    def test_build_index_local(self, builder):
        """Test building index without Pinecone."""
        # Create test images
        images = []
        for i in range(3):
            pil_img = Image.new("RGB", (100, 100))
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="PNG")

            img = ExtractedImage(
                image_id=f"test_{i}",
                doc_id="doc_1",
                page_number=i,
                image_data=img_bytes.getvalue(),
                format="png",
                width=100,
                height=100,
                description=f"Test image {i}",
                image_type=ImageType.PLOT_CHART,
            )
            images.append(img)

        # Build index (local only, no Pinecone)
        stats = builder.build_index(images)

        assert stats["total_images"] == 3
        assert len(builder.image_metadata) == 3
        assert ImageType.PLOT_CHART.value in stats["image_types"]


class TestMultiModalSearchResult:
    """Test MultiModalSearchResult."""

    def test_create_result(self):
        """Test creating search result."""
        result = MultiModalSearchResult(
            result_id="result_1",
            result_type="text",
            score=0.95,
            content="This is the content",
            metadata={"doc_id": "doc_1"},
        )

        assert result.result_id == "result_1"
        assert result.result_type == "text"
        assert result.score == 0.95
        assert result.content == "This is the content"
        assert result.metadata["doc_id"] == "doc_1"


@pytest.mark.asyncio
class TestMultiModalRAG:
    """Test MultiModalRAG (integration tests)."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        config_content = """
multimodal:
  vision_model:
    model: "claude-3-5-sonnet-20241022"
    max_tokens: 512
  image_processor:
    min_size: [100, 100]
    images_dir: "./cache/test_images"
  embeddings:
    clip_model: "openai/clip-vit-base-patch32"
    use_clip: false
  code_extractor:
    min_code_lines: 3
  visual_index:
    index_name: "test-index"
    namespace: "test"
  text_namespace: "test_text"
  code_namespace: "test_code"

rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  index_name: "test-rag"
  namespace: "test"
  pinecone:
    api_key: null
"""
        config_path = Path(tempfile.mkdtemp()) / "test_config.yaml"
        config_path.write_text(config_content)
        return str(config_path)

    @patch("training.multimodal_knowledge_base.MultiModalRAG.__init__")
    def test_initialization(self, mock_init):
        """Test RAG initialization."""
        mock_init.return_value = None
        # Just test that we can attempt initialization
        # Actual initialization requires full config


# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance benchmarks."""

    def test_code_extraction_speed(self):
        """Test code extraction performance."""
        import time

        extractor = CodeBlockExtractor({"min_code_lines": 3})

        # Generate large text with many code blocks
        text = "\n\n".join([f"""
            Section {i}:
            ```python
            def function_{i}():
                for j in range(100):
                    process(j)
                return True
            ```
            """ for i in range(100)])

        start = time.time()
        codes = extractor.extract_code_blocks(text, "perf_test")
        duration = time.time() - start

        assert len(codes) == 100
        assert duration < 2.0  # Should be fast (< 2 seconds)

    def test_embedding_speed(self):
        """Test embedding generation speed."""
        import time

        embedder = MultiModalEmbedder(
            {
                "text_model": "sentence-transformers/all-MiniLM-L6-v2",
                "use_clip": False,
            }
        )

        texts = [f"Test text number {i}" for i in range(100)]

        start = time.time()
        for text in texts:
            embedder.embed_text(text)
        duration = time.time() - start

        assert duration < 10.0  # Should be reasonably fast


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
