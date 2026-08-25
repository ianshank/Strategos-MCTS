"""
Research Corpus Builder for AI/ML Papers from arXiv

Ingests academic papers from arXiv.org and processes them into DocumentChunk format
for integration with the RAG pipeline and Pinecone vector storage.

Features:
- Fetch papers from arXiv API with advanced filtering
- Process paper metadata and abstracts
- Extract full-text when available
- Generate section-aware chunks
- Deduplicate based on arXiv IDs
- Cache papers locally for resumability
- Stream processing for memory efficiency
- Comprehensive error handling and rate limiting
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import yaml

try:
    import arxiv

    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

try:
    import requests  # noqa: F401 - reserved for future use

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from training.data_pipeline import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata for an arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    primary_category: str
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None
    citations: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class ProcessingStats:
    """Statistics about corpus building process."""

    total_papers_fetched: int = 0
    total_papers_processed: int = 0
    total_chunks_created: int = 0
    papers_cached: int = 0
    papers_skipped: int = 0
    errors: int = 0
    categories_breakdown: dict[str, int] = field(default_factory=dict)
    date_range: tuple[datetime | None, datetime | None] = (None, None)


class ArXivPaperFetcher:
    """Fetch papers from arXiv API with rate limiting and error handling."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize arXiv paper fetcher.

        Args:
            config: Configuration dictionary with arXiv settings
        """
        if not HAS_ARXIV:
            raise ImportError("arxiv library required. Install with: pip install arxiv")

        self.config = config
        self.categories = config.get("categories", ["cs.AI", "cs.LG", "cs.CL", "cs.NE"])
        self.keywords = config.get(
            "keywords",
            [
                "MCTS",
                "AlphaZero",
                "MuZero",
                "reinforcement learning",
                "multi-agent",
                "LLM reasoning",
                "chain-of-thought",
                "tree-of-thought",
                "self-improvement",
                "Constitutional AI",
                "RLHF",
                "DPO",
            ],
        )
        self.date_start = self._parse_date(config.get("date_start", "2020-01-01"))
        self.date_end = self._parse_date(config.get("date_end", datetime.now().strftime("%Y-%m-%d")))
        self.max_results = config.get("max_results", 1000)
        self.rate_limit_delay = config.get("rate_limit_delay", 3.0)  # seconds between requests
        self.retry_attempts = config.get("retry_attempts", 3)

        self._last_request_time = 0.0

        logger.info(
            f"ArXivPaperFetcher initialized: categories={self.categories}, "
            f"keywords={len(self.keywords)}, date_range={self.date_start.date()} to {self.date_end.date()}"
        )

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}, using default")
            return datetime.now()

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _build_query(self, category: str | None = None, keyword: str | None = None) -> str:
        """
        Build arXiv API search query.

        Args:
            category: Specific category to search (e.g., 'cs.AI')
            keyword: Specific keyword to search

        Returns:
            Query string for arXiv API
        """
        query_parts = []

        # Category filter
        if category:
            query_parts.append(f"cat:{category}")
        elif self.categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
            query_parts.append(f"({cat_query})")

        # Keyword filter
        if keyword:
            # Search in title, abstract, and comments
            keyword_query = f'(ti:"{keyword}" OR abs:"{keyword}")'
            query_parts.append(keyword_query)

        # Combine with AND
        if query_parts:
            return " AND ".join(query_parts)
        else:
            # Default to all papers in specified categories
            return " OR ".join([f"cat:{cat}" for cat in self.categories])

    def fetch_papers(
        self,
        category: str | None = None,
        keyword: str | None = None,
        max_results: int | None = None,
    ) -> Iterator[PaperMetadata]:
        """
        Fetch papers from arXiv matching criteria.

        Args:
            category: Specific category to search
            keyword: Specific keyword to search
            max_results: Maximum number of results to return

        Yields:
            PaperMetadata objects
        """
        if max_results is None:
            max_results = self.max_results

        query = self._build_query(category, keyword)
        logger.info(f"Fetching papers with query: {query[:100]}...")

        # Create arXiv client with custom settings
        client = arxiv.Client(
            page_size=100,  # Fetch 100 papers per page
            delay_seconds=self.rate_limit_delay,
            num_retries=self.retry_attempts,
        )

        # Build search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers_fetched = 0
        papers_filtered = 0

        try:
            for result in client.results(search):
                self._apply_rate_limit()

                # Filter by date range
                if result.published < self.date_start or result.published > self.date_end:
                    papers_filtered += 1
                    continue

                # Extract metadata
                metadata = PaperMetadata(
                    arxiv_id=result.entry_id.split("/")[-1],  # Extract ID from URL
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    categories=result.categories,
                    published=result.published,
                    updated=result.updated,
                    pdf_url=result.pdf_url,
                    primary_category=result.primary_category,
                    comment=result.comment,
                    journal_ref=result.journal_ref,
                    doi=result.doi,
                    keywords=self._extract_keywords(result.title + " " + result.summary),
                )

                papers_fetched += 1
                yield metadata

                if papers_fetched % 50 == 0:
                    logger.info(f"Fetched {papers_fetched} papers (filtered {papers_filtered})")

        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            raise

        logger.info(f"Fetch complete: {papers_fetched} papers fetched, {papers_filtered} filtered by date")

    def fetch_papers_by_keywords(self, max_per_keyword: int = 100) -> Iterator[PaperMetadata]:
        """
        Fetch papers for each configured keyword.

        Args:
            max_per_keyword: Maximum papers per keyword

        Yields:
            PaperMetadata objects
        """
        seen_ids = set()

        for keyword in self.keywords:
            logger.info(f"Fetching papers for keyword: {keyword}")

            try:
                for paper in self.fetch_papers(keyword=keyword, max_results=max_per_keyword):
                    # Deduplicate
                    if paper.arxiv_id not in seen_ids:
                        seen_ids.add(paper.arxiv_id)
                        yield paper

            except Exception as e:
                logger.error(f"Error fetching papers for keyword '{keyword}': {e}")
                continue

        logger.info(f"Fetched {len(seen_ids)} unique papers across {len(self.keywords)} keywords")

    def fetch_papers_by_categories(self, max_per_category: int = 500) -> Iterator[PaperMetadata]:
        """
        Fetch papers for each configured category.

        Args:
            max_per_category: Maximum papers per category

        Yields:
            PaperMetadata objects
        """
        seen_ids = set()

        for category in self.categories:
            logger.info(f"Fetching papers for category: {category}")

            try:
                for paper in self.fetch_papers(category=category, max_results=max_per_category):
                    # Deduplicate
                    if paper.arxiv_id not in seen_ids:
                        seen_ids.add(paper.arxiv_id)
                        yield paper

            except Exception as e:
                logger.error(f"Error fetching papers for category '{category}': {e}")
                continue

        logger.info(f"Fetched {len(seen_ids)} unique papers across {len(self.categories)} categories")

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        found_keywords = []

        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)

        return found_keywords


class PaperProcessor:
    """Process arXiv papers into structured chunks."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize paper processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.include_citations = config.get("include_citations", True)
        self.extract_sections = config.get("extract_sections", True)

        logger.info(f"PaperProcessor initialized: chunk_size={self.chunk_size}")

    def process_paper(self, paper: PaperMetadata) -> list[DocumentChunk]:
        """
        Process a paper into document chunks.

        Args:
            paper: Paper metadata

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Create metadata for all chunks
        base_metadata = {
            "source": "arxiv",
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors[:5],  # Limit to first 5 authors
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "published_date": paper.published.isoformat(),
            "updated_date": paper.updated.isoformat(),
            "pdf_url": paper.pdf_url,
            "keywords": paper.keywords,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
        }

        # Chunk 1: Title + metadata
        title_chunk = DocumentChunk(
            doc_id=f"arxiv_{paper.arxiv_id}",
            chunk_id=0,
            text=f"Title: {paper.title}\n\nAuthors: {', '.join(paper.authors[:5])}\n\n"
            f"Categories: {', '.join(paper.categories)}\n\n"
            f"Published: {paper.published.strftime('%Y-%m-%d')}",
            metadata={**base_metadata, "section": "title", "chunk_type": "metadata"},
        )
        chunks.append(title_chunk)

        # Chunk 2+: Abstract (may be split if long)
        abstract_chunks = self._chunk_text(paper.abstract, paper.arxiv_id, start_chunk_id=1)
        for chunk in abstract_chunks:
            chunk.metadata.update({**base_metadata, "section": "abstract", "chunk_type": "abstract"})
        chunks.extend(abstract_chunks)

        # Add paper comment if available
        if paper.comment:
            comment_chunk = DocumentChunk(
                doc_id=f"arxiv_{paper.arxiv_id}",
                chunk_id=len(chunks),
                text=f"Comments: {paper.comment}",
                metadata={**base_metadata, "section": "comments", "chunk_type": "metadata"},
            )
            chunks.append(comment_chunk)

        logger.debug(f"Processed paper {paper.arxiv_id} into {len(chunks)} chunks")
        return chunks

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        start_chunk_id: int = 0,
    ) -> list[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            start_chunk_id: Starting chunk ID

        Returns:
            List of document chunks
        """
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_length = 0
        chunk_id = start_chunk_id

        for sentence in sentences:
            sentence_len = len(sentence)

            # Start new chunk if size exceeded
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    DocumentChunk(
                        doc_id=f"arxiv_{doc_id}",
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata={},
                    )
                )
                chunk_id += 1

                # Keep overlap
                if len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-1:]
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_len

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                DocumentChunk(
                    doc_id=f"arxiv_{doc_id}",
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={},
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_citations(self, text: str) -> list[str]:
        """Extract citation patterns from text."""
        # Match common citation patterns like [1], (Author et al., 2020)
        citations = []

        # Numbered citations
        numbered = re.findall(r"\[(\d+)\]", text)
        citations.extend([f"[{n}]" for n in numbered])

        # Author-year citations
        author_year = re.findall(r"\(([A-Z][a-z]+ et al\., \d{4})\)", text)
        citations.extend(author_year)

        return list(set(citations))[:10]  # Limit to 10 citations


class ResearchCorpusCache:
    """Manage caching and deduplication of papers."""

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "papers_metadata.json"
        self.processed_ids_file = self.cache_dir / "processed_ids.txt"
        self.stats_file = self.cache_dir / "processing_stats.json"

        self._processed_ids = self._load_processed_ids()
        self._metadata_cache = self._load_metadata_cache()

        logger.info(f"Cache initialized at {self.cache_dir}, {len(self._processed_ids)} papers cached")

    def _load_processed_ids(self) -> set[str]:
        """Load set of already processed paper IDs."""
        if self.processed_ids_file.exists():
            with open(self.processed_ids_file) as f:
                return {line.strip() for line in f if line.strip()}
        return set()

    def _load_metadata_cache(self) -> dict[str, Any]:
        """Load cached paper metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata cache: {e}")
        return {}

    def is_processed(self, arxiv_id: str) -> bool:
        """Check if paper has been processed."""
        return arxiv_id in self._processed_ids

    def mark_processed(self, arxiv_id: str) -> None:
        """Mark paper as processed."""
        self._processed_ids.add(arxiv_id)
        with open(self.processed_ids_file, "a") as f:
            f.write(f"{arxiv_id}\n")

    def save_paper_metadata(self, paper: PaperMetadata) -> None:
        """Save paper metadata to cache."""
        self._metadata_cache[paper.arxiv_id] = {
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published.isoformat(),
            "keywords": paper.keywords,
        }

        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata_cache, f, indent=2)

    def save_stats(self, stats: ProcessingStats) -> None:
        """Save processing statistics."""
        stats_dict = {
            "total_papers_fetched": stats.total_papers_fetched,
            "total_papers_processed": stats.total_papers_processed,
            "total_chunks_created": stats.total_chunks_created,
            "papers_cached": stats.papers_cached,
            "papers_skipped": stats.papers_skipped,
            "errors": stats.errors,
            "categories_breakdown": stats.categories_breakdown,
            "date_range": [
                stats.date_range[0].isoformat() if stats.date_range[0] else None,
                stats.date_range[1].isoformat() if stats.date_range[1] else None,
            ],
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.stats_file, "w") as f:
            json.dump(stats_dict, f, indent=2)

    def load_stats(self) -> ProcessingStats | None:
        """Load processing statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    data = json.load(f)
                    return ProcessingStats(
                        total_papers_fetched=data.get("total_papers_fetched", 0),
                        total_papers_processed=data.get("total_papers_processed", 0),
                        total_chunks_created=data.get("total_chunks_created", 0),
                        papers_cached=data.get("papers_cached", 0),
                        papers_skipped=data.get("papers_skipped", 0),
                        errors=data.get("errors", 0),
                        categories_breakdown=data.get("categories_breakdown", {}),
                    )
            except Exception as e:
                logger.warning(f"Could not load stats: {e}")
        return None

    def get_cached_papers(self) -> list[str]:
        """Get list of cached paper IDs."""
        return list(self._processed_ids)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._processed_ids.clear()
        self._metadata_cache.clear()

        for file in [self.metadata_file, self.processed_ids_file, self.stats_file]:
            if file.exists():
                file.unlink()

        logger.info("Cache cleared")


class ResearchCorpusBuilder:
    """Main class for building research corpus from arXiv papers."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | None = None):
        """
        Initialize research corpus builder.

        Args:
            config: Configuration dictionary
            config_path: Path to YAML configuration file
        """
        if config_path:
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get("research_corpus", {})
        elif config:
            self.config = config
        else:
            # Default configuration
            self.config = {
                "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.NE"],
                "keywords": [
                    "MCTS",
                    "AlphaZero",
                    "MuZero",
                    "reinforcement learning",
                    "multi-agent",
                    "LLM reasoning",
                    "chain-of-thought",
                    "tree-of-thought",
                    "self-improvement",
                    "Constitutional AI",
                    "RLHF",
                    "DPO",
                ],
                "date_start": "2020-01-01",
                "date_end": datetime.now().strftime("%Y-%m-%d"),
                "max_results": 1000,
                "cache_dir": "./cache/research_corpus",
                "chunk_size": 512,
                "chunk_overlap": 50,
            }

        self.cache_dir = Path(self.config.get("cache_dir", "./cache/research_corpus"))
        self.cache = ResearchCorpusCache(self.cache_dir)

        self.fetcher = ArXivPaperFetcher(self.config)
        self.processor = PaperProcessor(self.config)

        self.stats = self.cache.load_stats() or ProcessingStats()

        logger.info("ResearchCorpusBuilder initialized")

    def build_corpus(
        self,
        mode: str = "keywords",
        max_papers: int | None = None,
        skip_cached: bool = True,
    ) -> Iterator[DocumentChunk]:
        """
        Build research corpus by fetching and processing papers.

        Args:
            mode: Fetching mode ('keywords', 'categories', or 'all')
            max_papers: Maximum number of papers to process
            skip_cached: Whether to skip already processed papers

        Yields:
            DocumentChunk objects
        """
        logger.info(f"Building corpus in '{mode}' mode, max_papers={max_papers}, skip_cached={skip_cached}")

        # Select fetching strategy
        if mode == "keywords":
            papers_iterator = self.fetcher.fetch_papers_by_keywords(max_per_keyword=max_papers or 100)
        elif mode == "categories":
            papers_iterator = self.fetcher.fetch_papers_by_categories(max_per_category=max_papers or 500)
        else:  # all
            papers_iterator = self.fetcher.fetch_papers(max_results=max_papers)

        papers_processed = 0

        try:
            for paper in papers_iterator:
                self.stats.total_papers_fetched += 1

                # Skip if already processed
                if skip_cached and self.cache.is_processed(paper.arxiv_id):
                    self.stats.papers_skipped += 1
                    logger.debug(f"Skipping cached paper: {paper.arxiv_id}")
                    continue

                # Update date range
                if self.stats.date_range[0] is None or paper.published < self.stats.date_range[0]:
                    self.stats.date_range = (paper.published, self.stats.date_range[1])
                if self.stats.date_range[1] is None or paper.published > self.stats.date_range[1]:
                    self.stats.date_range = (self.stats.date_range[0], paper.published)

                # Update category breakdown
                for cat in paper.categories:
                    self.stats.categories_breakdown[cat] = self.stats.categories_breakdown.get(cat, 0) + 1

                # Process paper
                try:
                    chunks = self.processor.process_paper(paper)

                    for chunk in chunks:
                        yield chunk
                        self.stats.total_chunks_created += 1

                    # Mark as processed and cache metadata
                    self.cache.mark_processed(paper.arxiv_id)
                    self.cache.save_paper_metadata(paper)

                    papers_processed += 1
                    self.stats.total_papers_processed += 1
                    self.stats.papers_cached += 1

                    # Log progress
                    if papers_processed % 10 == 0:
                        logger.info(
                            f"Progress: {papers_processed} papers processed, "
                            f"{self.stats.total_chunks_created} chunks created"
                        )

                    # Save stats periodically
                    if papers_processed % 50 == 0:
                        self.cache.save_stats(self.stats)

                except Exception as e:
                    logger.error(f"Error processing paper {paper.arxiv_id}: {e}")
                    self.stats.errors += 1
                    continue

                # Check max papers limit
                if max_papers and papers_processed >= max_papers:
                    logger.info(f"Reached max_papers limit: {max_papers}")
                    break

        except KeyboardInterrupt:
            logger.info("Corpus building interrupted by user")
        except Exception as e:
            logger.error(f"Error during corpus building: {e}")
            raise
        finally:
            # Save final stats
            self.cache.save_stats(self.stats)
            logger.info(f"Corpus building complete: {self.stats}")

    def stream_chunks_from_cache(self) -> Iterator[DocumentChunk]:
        """
        Stream chunks from cached papers (for re-processing).

        Yields:
            DocumentChunk objects
        """
        cached_ids = self.cache.get_cached_papers()
        logger.info(f"Streaming chunks from {len(cached_ids)} cached papers")

        # Note: This would require storing processed chunks in cache
        # For now, it's a placeholder for future enhancement
        logger.warning("Streaming from cache not yet implemented")
        return iter([])

    def get_statistics(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.stats

    def export_metadata(self, output_path: Path) -> None:
        """
        Export paper metadata to JSON file.

        Args:
            output_path: Path to output file
        """
        metadata = self.cache._metadata_cache
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Exported metadata for {len(metadata)} papers to {output_path}")

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear_cache()
        self.stats = ProcessingStats()


def build_arxiv_corpus_with_config(config_path: str = "training/config.yaml") -> Iterator[DocumentChunk]:
    """
    Convenience function to build corpus using config file.

    Args:
        config_path: Path to configuration file

    Yields:
        DocumentChunk objects
    """
    builder = ResearchCorpusBuilder(config_path=config_path)
    yield from builder.build_corpus(mode="keywords", max_papers=None, skip_cached=True)


def integrate_with_rag_pipeline(
    corpus_builder: ResearchCorpusBuilder,
    rag_index_builder: Any,
    batch_size: int = 100,
) -> dict[str, Any]:
    """
    Integrate corpus builder with RAG pipeline.

    Args:
        corpus_builder: ResearchCorpusBuilder instance
        rag_index_builder: VectorIndexBuilder from rag_builder.py
        batch_size: Batch size for vector indexing

    Returns:
        Statistics dictionary
    """
    logger.info("Integrating arXiv corpus with RAG pipeline")

    chunks_iterator = corpus_builder.build_corpus(mode="keywords", skip_cached=True)

    # Build index
    index_stats = rag_index_builder.build_index(chunks_iterator, batch_size=batch_size)

    corpus_stats = corpus_builder.get_statistics()

    combined_stats = {
        "corpus": {
            "papers_fetched": corpus_stats.total_papers_fetched,
            "papers_processed": corpus_stats.total_papers_processed,
            "chunks_created": corpus_stats.total_chunks_created,
            "categories": corpus_stats.categories_breakdown,
        },
        "index": {
            "total_documents": index_stats.total_documents,
            "total_chunks": index_stats.total_chunks,
            "index_size_mb": index_stats.index_size_mb,
            "avg_chunk_length": index_stats.avg_chunk_length,
        },
    }

    logger.info(f"Integration complete: {combined_stats}")
    return combined_stats


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Testing Research Corpus Builder")

    # Test configuration
    test_config = {
        "categories": ["cs.AI", "cs.LG"],
        "keywords": ["MCTS", "reinforcement learning", "tree search"],
        "date_start": "2023-01-01",
        "date_end": "2024-12-31",
        "max_results": 50,  # Small number for testing
        "cache_dir": "./cache/research_corpus_test",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "rate_limit_delay": 3.0,
    }

    # Initialize builder
    builder = ResearchCorpusBuilder(config=test_config)

    # Test corpus building
    logger.info("Building test corpus...")
    chunk_count = 0

    try:
        for chunk in builder.build_corpus(mode="keywords", max_papers=10, skip_cached=True):
            chunk_count += 1
            if chunk_count <= 3:  # Print first 3 chunks as examples
                logger.info(f"\nChunk {chunk_count}:")
                logger.info(f"  Doc ID: {chunk.doc_id}")
                logger.info(f"  Chunk ID: {chunk.chunk_id}")
                logger.info(f"  Text length: {len(chunk.text)}")
                logger.info(f"  Text preview: {chunk.text[:150]}...")
                logger.info(f"  Metadata: {chunk.metadata.get('section', 'N/A')}")

            if chunk_count >= 20:  # Limit for testing
                logger.info("Reached test limit, stopping...")
                break

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()

    # Print statistics
    stats = builder.get_statistics()
    logger.info("\n=== Processing Statistics ===")
    logger.info(f"Papers fetched: {stats.total_papers_fetched}")
    logger.info(f"Papers processed: {stats.total_papers_processed}")
    logger.info(f"Chunks created: {stats.total_chunks_created}")
    logger.info(f"Papers cached: {stats.papers_cached}")
    logger.info(f"Papers skipped: {stats.papers_skipped}")
    logger.info(f"Errors: {stats.errors}")
    logger.info(f"Categories breakdown: {stats.categories_breakdown}")

    # Test with config file (if exists)
    config_path = Path("training/config.yaml")
    if config_path.exists():
        logger.info("\n=== Testing with config file ===")
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)

            # Add research_corpus section if not present
            if "research_corpus" not in yaml_config:
                logger.info("Adding research_corpus section to config for testing")
                yaml_config["research_corpus"] = test_config

                # Save updated config
                with open(config_path, "w") as f:
                    yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

            builder_from_config = ResearchCorpusBuilder(config_path=str(config_path))
            logger.info("Successfully initialized builder from config file")

        except Exception as e:
            logger.error(f"Error testing with config file: {e}")

    logger.info("\nResearch Corpus Builder test complete!")
