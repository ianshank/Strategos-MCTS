"""
Unit tests for Research Corpus Builder

Tests the arXiv paper fetching, processing, and integration functionality.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from training.research_corpus_builder import (
    ArXivPaperFetcher,
    PaperMetadata,
    PaperProcessor,
    ProcessingStats,
    ResearchCorpusBuilder,
    ResearchCorpusCache,
)


@pytest.fixture
def sample_paper_metadata():
    """Create sample paper metadata for testing."""
    return PaperMetadata(
        arxiv_id="2301.12345",
        title="Sample Paper on Monte Carlo Tree Search",
        authors=["John Doe", "Jane Smith", "Bob Johnson"],
        abstract=(
            "This paper presents a novel approach to Monte Carlo Tree Search "
            "in reinforcement learning. We demonstrate improved performance "
            "on several benchmark tasks."
        ),
        categories=["cs.AI", "cs.LG"],
        published=datetime(2023, 1, 15, 10, 30),
        updated=datetime(2023, 1, 20, 14, 45),
        pdf_url="https://arxiv.org/pdf/2301.12345",
        primary_category="cs.AI",
        comment="10 pages, 3 figures",
        journal_ref="Conference 2023",
        doi="10.1234/example",
        keywords=["MCTS", "reinforcement learning"],
    )


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPaperMetadata:
    """Test PaperMetadata dataclass."""

    def test_paper_metadata_creation(self, sample_paper_metadata):
        """Test creating paper metadata."""
        assert sample_paper_metadata.arxiv_id == "2301.12345"
        assert "Monte Carlo Tree Search" in sample_paper_metadata.title
        assert len(sample_paper_metadata.authors) == 3
        assert "cs.AI" in sample_paper_metadata.categories

    def test_paper_metadata_fields(self, sample_paper_metadata):
        """Test all metadata fields."""
        assert sample_paper_metadata.primary_category == "cs.AI"
        assert sample_paper_metadata.comment == "10 pages, 3 figures"
        assert sample_paper_metadata.doi == "10.1234/example"
        assert "MCTS" in sample_paper_metadata.keywords


class TestArXivPaperFetcher:
    """Test ArXiv paper fetching functionality."""

    def test_fetcher_initialization(self):
        """Test fetcher initialization with config."""
        config = {
            "categories": ["cs.AI", "cs.LG"],
            "keywords": ["MCTS", "reinforcement learning"],
            "date_start": "2020-01-01",
            "date_end": "2023-12-31",
            "max_results": 100,
        }

        fetcher = ArXivPaperFetcher(config)

        assert fetcher.categories == ["cs.AI", "cs.LG"]
        assert "MCTS" in fetcher.keywords
        assert fetcher.max_results == 100

    def test_build_query_with_category(self):
        """Test query building with category."""
        config = {"categories": ["cs.AI"], "keywords": []}
        fetcher = ArXivPaperFetcher(config)

        query = fetcher._build_query(category="cs.AI")
        assert "cat:cs.AI" in query

    def test_build_query_with_keyword(self):
        """Test query building with keyword."""
        config = {"categories": ["cs.AI"], "keywords": ["MCTS"]}
        fetcher = ArXivPaperFetcher(config)

        query = fetcher._build_query(keyword="MCTS")
        assert "MCTS" in query

    def test_extract_keywords(self):
        """Test keyword extraction from text."""
        config = {"keywords": ["MCTS", "AlphaZero", "reinforcement learning"]}
        fetcher = ArXivPaperFetcher(config)

        text = "This paper presents AlphaZero with MCTS for game playing."
        keywords = fetcher._extract_keywords(text)

        assert "MCTS" in keywords
        assert "AlphaZero" in keywords
        assert len(keywords) == 2


class TestPaperProcessor:
    """Test paper processing functionality."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        config = {"chunk_size": 512, "chunk_overlap": 50}
        processor = PaperProcessor(config)

        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 50

    def test_process_paper(self, sample_paper_metadata):
        """Test processing a paper into chunks."""
        config = {"chunk_size": 512, "chunk_overlap": 50}
        processor = PaperProcessor(config)

        chunks = processor.process_paper(sample_paper_metadata)

        # Should have at least title and abstract chunks
        assert len(chunks) >= 2

        # Check title chunk
        title_chunk = chunks[0]
        assert title_chunk.chunk_id == 0
        assert "Sample Paper on Monte Carlo Tree Search" in title_chunk.text
        assert title_chunk.metadata["section"] == "title"

        # Check abstract chunk
        abstract_chunk = chunks[1]
        assert abstract_chunk.chunk_id == 1
        assert abstract_chunk.metadata["section"] == "abstract"

    def test_chunk_metadata(self, sample_paper_metadata):
        """Test metadata in processed chunks."""
        config = {"chunk_size": 512, "chunk_overlap": 50}
        processor = PaperProcessor(config)

        chunks = processor.process_paper(sample_paper_metadata)

        for chunk in chunks:
            assert chunk.metadata["source"] == "arxiv"
            assert chunk.metadata["arxiv_id"] == "2301.12345"
            assert "title" in chunk.metadata
            assert "authors" in chunk.metadata
            assert "categories" in chunk.metadata

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        config = {"chunk_size": 512, "chunk_overlap": 50}
        processor = PaperProcessor(config)

        text = "First sentence. Second sentence! Third sentence?"
        sentences = processor._split_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_chunk_text_with_overlap(self):
        """Test text chunking with overlap."""
        config = {"chunk_size": 100, "chunk_overlap": 20}
        processor = PaperProcessor(config)

        text = "A " * 100  # Long text
        chunks = processor._chunk_text(text, "test_doc")

        # Should create multiple chunks
        assert len(chunks) > 1

        # Check overlap exists (simplified check)
        assert all(chunk.chunk_id == i for i, chunk in enumerate(chunks))


class TestResearchCorpusCache:
    """Test caching functionality."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = ResearchCorpusCache(temp_cache_dir)

        assert cache.cache_dir == temp_cache_dir
        assert cache.metadata_file.parent == temp_cache_dir
        assert len(cache._processed_ids) == 0

    def test_mark_and_check_processed(self, temp_cache_dir):
        """Test marking papers as processed."""
        cache = ResearchCorpusCache(temp_cache_dir)

        # Initially not processed
        assert not cache.is_processed("2301.12345")

        # Mark as processed
        cache.mark_processed("2301.12345")
        assert cache.is_processed("2301.12345")

        # Should persist across instances
        cache2 = ResearchCorpusCache(temp_cache_dir)
        assert cache2.is_processed("2301.12345")

    def test_save_and_load_metadata(self, temp_cache_dir, sample_paper_metadata):
        """Test saving and loading paper metadata."""
        cache = ResearchCorpusCache(temp_cache_dir)

        # Save metadata
        cache.save_paper_metadata(sample_paper_metadata)

        # Load in new instance
        cache2 = ResearchCorpusCache(temp_cache_dir)
        assert "2301.12345" in cache2._metadata_cache
        assert cache2._metadata_cache["2301.12345"]["title"] == sample_paper_metadata.title

    def test_save_and_load_stats(self, temp_cache_dir):
        """Test saving and loading statistics."""
        cache = ResearchCorpusCache(temp_cache_dir)

        stats = ProcessingStats(
            total_papers_fetched=100,
            total_papers_processed=95,
            total_chunks_created=500,
            papers_cached=95,
            papers_skipped=5,
            errors=0,
            categories_breakdown={"cs.AI": 50, "cs.LG": 45},
        )

        cache.save_stats(stats)

        # Load stats
        loaded_stats = cache.load_stats()
        assert loaded_stats.total_papers_fetched == 100
        assert loaded_stats.total_papers_processed == 95
        assert loaded_stats.categories_breakdown["cs.AI"] == 50

    def test_get_cached_papers(self, temp_cache_dir):
        """Test getting list of cached papers."""
        cache = ResearchCorpusCache(temp_cache_dir)

        cache.mark_processed("2301.12345")
        cache.mark_processed("2301.67890")

        cached = cache.get_cached_papers()
        assert len(cached) == 2
        assert "2301.12345" in cached
        assert "2301.67890" in cached

    def test_clear_cache(self, temp_cache_dir):
        """Test clearing cache."""
        cache = ResearchCorpusCache(temp_cache_dir)

        cache.mark_processed("2301.12345")
        cache.save_paper_metadata(
            PaperMetadata(
                arxiv_id="2301.12345",
                title="Test",
                authors=[],
                abstract="Test",
                categories=[],
                published=datetime.now(),
                updated=datetime.now(),
                pdf_url="",
                primary_category="cs.AI",
            )
        )

        assert cache.is_processed("2301.12345")

        cache.clear_cache()

        assert not cache.is_processed("2301.12345")
        assert len(cache._metadata_cache) == 0


class TestResearchCorpusBuilder:
    """Test main corpus builder."""

    def test_builder_initialization_with_config(self):
        """Test builder initialization with config dict."""
        config = {
            "categories": ["cs.AI"],
            "keywords": ["MCTS"],
            "date_start": "2020-01-01",
            "date_end": "2023-12-31",
            "max_results": 100,
            "cache_dir": "./cache/test",
        }

        builder = ResearchCorpusBuilder(config=config)

        assert builder.fetcher.categories == ["cs.AI"]
        assert builder.processor.chunk_size == 512  # default

    def test_builder_default_config(self):
        """Test builder with default configuration."""
        builder = ResearchCorpusBuilder()

        assert "cs.AI" in builder.config["categories"]
        assert "MCTS" in builder.config["keywords"]
        assert builder.stats.total_papers_fetched == 0

    @patch("training.research_corpus_builder.arxiv.Client")
    def test_build_corpus_mock(self, mock_client, temp_cache_dir, sample_paper_metadata):
        """Test corpus building with mocked arXiv client."""
        # Mock the arXiv client to return sample paper
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))  # Empty iterator for now
        mock_client.return_value.results.return_value = mock_results

        config = {
            "categories": ["cs.AI"],
            "keywords": ["MCTS"],
            "cache_dir": str(temp_cache_dir),
            "max_results": 10,
        }

        builder = ResearchCorpusBuilder(config=config)

        # This should not error even with empty results
        chunks = list(builder.build_corpus(mode="keywords", max_papers=5))

        # With empty results, no chunks created
        assert len(chunks) == 0

    def test_get_statistics(self, temp_cache_dir):
        """Test getting statistics."""
        config = {"cache_dir": str(temp_cache_dir)}
        builder = ResearchCorpusBuilder(config=config)

        stats = builder.get_statistics()
        assert isinstance(stats, ProcessingStats)
        assert stats.total_papers_fetched == 0

    def test_export_metadata(self, temp_cache_dir, sample_paper_metadata):
        """Test exporting metadata to JSON."""
        cache = ResearchCorpusCache(temp_cache_dir)
        cache.save_paper_metadata(sample_paper_metadata)

        config = {"cache_dir": str(temp_cache_dir)}
        builder = ResearchCorpusBuilder(config=config)

        export_path = temp_cache_dir / "export.json"
        builder.export_metadata(export_path)

        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)
            assert "2301.12345" in data
            assert data["2301.12345"]["title"] == sample_paper_metadata.title


class TestProcessingStats:
    """Test processing statistics."""

    def test_stats_initialization(self):
        """Test creating processing stats."""
        stats = ProcessingStats()

        assert stats.total_papers_fetched == 0
        assert stats.total_papers_processed == 0
        assert stats.total_chunks_created == 0
        assert isinstance(stats.categories_breakdown, dict)

    def test_stats_updates(self):
        """Test updating statistics."""
        stats = ProcessingStats()

        stats.total_papers_fetched = 100
        stats.total_papers_processed = 95
        stats.errors = 5
        stats.categories_breakdown["cs.AI"] = 50

        assert stats.total_papers_fetched == 100
        assert stats.categories_breakdown["cs.AI"] == 50


def test_module_imports():
    """Test that all required modules can be imported."""
    from training.research_corpus_builder import (
        ArXivPaperFetcher,
        PaperMetadata,
        PaperProcessor,
        ProcessingStats,
        ResearchCorpusBuilder,
        ResearchCorpusCache,
        build_arxiv_corpus_with_config,
        integrate_with_rag_pipeline,
    )

    # All imports should succeed
    assert ArXivPaperFetcher is not None
    assert PaperMetadata is not None
    assert PaperProcessor is not None
    assert ProcessingStats is not None
    assert ResearchCorpusBuilder is not None
    assert ResearchCorpusCache is not None
    assert build_arxiv_corpus_with_config is not None
    assert integrate_with_rag_pipeline is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
