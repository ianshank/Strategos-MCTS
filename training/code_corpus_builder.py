"""
Code Corpus Builder - Repository Ingestion for Searchable Code Knowledge Base

Clones target repositories, parses code with AST, extracts functions/classes/patterns,
and builds a searchable index compatible with the RAG system.
"""

import ast
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml

try:
    from github import Github

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False

from training.data_pipeline import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================================
# Target Repositories Configuration
# ============================================================================

REPOSITORIES = [
    # MCTS and RL implementations
    {
        "name": "deepmind/mctx",
        "description": "JAX-based MCTS library from DeepMind",
        "priority": "high",
        "topics": ["mcts", "jax", "reinforcement-learning"],
    },
    {
        "name": "openai/gym",
        "description": "RL environments and benchmarks",
        "priority": "high",
        "topics": ["reinforcement-learning", "environments"],
    },
    {
        "name": "facebookresearch/ReAgent",
        "description": "Production RL systems from Facebook",
        "priority": "medium",
        "topics": ["reinforcement-learning", "production"],
    },
    {
        "name": "google-deepmind/alphatensor",
        "description": "AlphaTensor research code",
        "priority": "medium",
        "topics": ["mcts", "deep-learning"],
    },
    # LangGraph and multi-agent systems
    {
        "name": "langchain-ai/langgraph",
        "description": "LangGraph framework for multi-agent systems",
        "priority": "high",
        "topics": ["langgraph", "multi-agent", "state-machines"],
    },
    # ML infrastructure
    {
        "name": "karpathy/nanoGPT",
        "description": "Minimal GPT implementation",
        "priority": "high",
        "topics": ["transformers", "gpt", "training"],
    },
    {
        "name": "microsoft/DeepSpeed",
        "description": "Deep learning optimization library",
        "priority": "medium",
        "topics": ["optimization", "distributed-training"],
    },
    {
        "name": "huggingface/transformers",
        "description": "State-of-the-art NLP models",
        "priority": "low",  # Large repo, sample selectively
        "topics": ["transformers", "nlp", "models"],
    },
]


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CodeChunk:
    """Represents an extracted code snippet with metadata."""

    repo_name: str
    file_path: str
    function_name: str
    code: str
    docstring: str
    imports: list[str]
    usage_examples: list[str]
    related_papers: list[str]
    metadata: dict[str, Any]

    # Additional fields
    start_line: int = 0
    end_line: int = 0
    complexity_score: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_document_chunk(self, chunk_id: int = 0) -> DocumentChunk:
        """Convert to DocumentChunk for RAG integration."""
        # Create searchable text representation
        text_parts = []

        # Header with context
        text_parts.append(f"# Repository: {self.repo_name}")
        text_parts.append(f"# File: {self.file_path}")
        text_parts.append(f"# Function: {self.function_name}")
        text_parts.append("")

        # Docstring if available
        if self.docstring:
            text_parts.append("## Documentation")
            text_parts.append(self.docstring)
            text_parts.append("")

        # Code
        text_parts.append("## Code")
        text_parts.append("```python")
        text_parts.append(self.code)
        text_parts.append("```")
        text_parts.append("")

        # Usage examples
        if self.usage_examples:
            text_parts.append("## Usage Examples")
            for i, example in enumerate(self.usage_examples, 1):
                text_parts.append(f"### Example {i}")
                text_parts.append("```python")
                text_parts.append(example)
                text_parts.append("```")
            text_parts.append("")

        # Related papers
        if self.related_papers:
            text_parts.append("## Related Papers")
            for paper in self.related_papers:
                text_parts.append(f"- {paper}")
            text_parts.append("")

        # Create unique doc_id
        doc_id = hashlib.md5(f"{self.repo_name}:{self.file_path}:{self.function_name}".encode()).hexdigest()

        # Enhanced metadata
        metadata = {
            "type": "code",
            "repo_name": self.repo_name,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "complexity_score": self.complexity_score,
            "imports": self.imports,
            "dependencies": self.dependencies,
            "has_docstring": bool(self.docstring),
            "has_examples": bool(self.usage_examples),
            "has_tests": bool(self.test_files),
            "code_length": len(self.code),
            **self.metadata,
        }

        return DocumentChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text="\n".join(text_parts),
            metadata=metadata,
        )


@dataclass
class RepositoryMetadata:
    """Metadata about a processed repository."""

    name: str
    description: str
    clone_url: str
    stars: int
    language: str
    license: str
    topics: list[str]
    total_files: int
    total_functions: int
    total_classes: int
    processed_at: str


# ============================================================================
# Repository Fetcher
# ============================================================================


class RepositoryFetcher:
    """Handles repository cloning and fetching via Git or GitHub API."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize repository fetcher.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "./cache/code_repos"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_github_api = config.get("use_github_api", True)
        self.github_token = config.get("github_token") or os.environ.get("GITHUB_TOKEN")

        if self.use_github_api and HAS_GITHUB and self.github_token:
            self.github_client = Github(self.github_token)
            logger.info("Initialized GitHub API client")
        else:
            self.github_client = None
            if self.use_github_api:
                logger.warning("GitHub API not available, falling back to git clone")

        self.shallow_clone = config.get("shallow_clone", True)

    def fetch_repository(self, repo_info: dict[str, Any]) -> Path:
        """
        Fetch repository code (clone or use API).

        Args:
            repo_info: Repository information dictionary

        Returns:
            Path to local repository directory
        """
        repo_name = repo_info["name"]
        logger.info(f"Fetching repository: {repo_name}")

        # Check if already cached
        repo_path = self.cache_dir / repo_name.replace("/", "_")
        if repo_path.exists() and (repo_path / ".git").exists():
            logger.info(f"Repository already cached at {repo_path}")
            return repo_path

        # Clone repository
        repo_url = f"https://github.com/{repo_name}.git"

        try:
            cmd = ["git", "clone"]
            if self.shallow_clone:
                cmd.extend(["--depth", "1"])
            cmd.extend([repo_url, str(repo_path)])

            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )

            logger.info(f"Successfully cloned {repo_name}")
            return repo_path

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout cloning {repo_name}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {repo_name}: {e.stderr}")
            raise

    def get_repository_metadata(self, repo_info: dict[str, Any]) -> RepositoryMetadata:
        """Get repository metadata via GitHub API."""
        repo_name = repo_info["name"]

        if self.github_client:
            try:
                repo = self.github_client.get_repo(repo_name)
                return RepositoryMetadata(
                    name=repo.full_name,
                    description=repo.description or "",
                    clone_url=repo.clone_url,
                    stars=repo.stargazers_count,
                    language=repo.language or "Unknown",
                    license=repo.license.name if repo.license else "Unknown",
                    topics=repo.get_topics(),
                    total_files=0,  # Will be updated during processing
                    total_functions=0,
                    total_classes=0,
                    processed_at="",
                )
            except Exception as e:
                logger.warning(f"Failed to fetch metadata for {repo_name}: {e}")

        # Fallback metadata
        return RepositoryMetadata(
            name=repo_name,
            description=repo_info.get("description", ""),
            clone_url=f"https://github.com/{repo_name}.git",
            stars=0,
            language="Python",
            license="Unknown",
            topics=repo_info.get("topics", []),
            total_files=0,
            total_functions=0,
            total_classes=0,
            processed_at="",
        )

    def check_license_compliance(self, repo_path: Path) -> dict[str, Any]:
        """Check repository license for compliance."""
        license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"]

        for license_file in license_files:
            license_path = repo_path / license_file
            if license_path.exists():
                try:
                    license_text = license_path.read_text(encoding="utf-8")

                    # Simple license detection
                    license_type = "Unknown"
                    if "MIT" in license_text:
                        license_type = "MIT"
                    elif "Apache" in license_text:
                        license_type = "Apache-2.0"
                    elif "BSD" in license_text:
                        license_type = "BSD"
                    elif "GPL" in license_text:
                        license_type = "GPL"

                    return {
                        "found": True,
                        "type": license_type,
                        "file": license_file,
                        "compliant": license_type in ["MIT", "Apache-2.0", "BSD"],
                    }
                except Exception as e:
                    logger.warning(f"Error reading license file: {e}")

        return {
            "found": False,
            "type": "Unknown",
            "file": None,
            "compliant": False,
        }


# ============================================================================
# Code Parser
# ============================================================================


class PythonCodeParser:
    """Parse Python files and extract code structures using AST."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize code parser.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_function_lines = config.get("min_function_lines", 3)
        self.max_function_lines = config.get("max_function_lines", 200)
        self.extract_tests = config.get("extract_tests", True)
        self.extract_examples = config.get("extract_examples", True)

    def parse_file(self, file_path: Path, repo_name: str) -> list[CodeChunk]:
        """
        Parse a Python file and extract code chunks.

        Args:
            file_path: Path to Python file
            repo_name: Repository name

        Returns:
            List of extracted code chunks
        """
        try:
            source_code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        try:
            tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []

        chunks = []

        # Extract imports
        imports = self._extract_imports(tree)

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = self._extract_function(node, source_code, file_path, repo_name, imports)
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                # Extract class and its methods
                class_chunks = self._extract_class(node, source_code, file_path, repo_name, imports)
                chunks.extend(class_chunks)

        return chunks

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        return imports

    def _extract_function(
        self,
        node: ast.FunctionDef,
        source_code: str,
        file_path: Path,
        repo_name: str,
        imports: list[str],
    ) -> CodeChunk | None:
        """Extract function as a code chunk."""
        # Get function source code
        try:
            func_lines = source_code.split("\n")[node.lineno - 1 : node.end_lineno]
            func_code = "\n".join(func_lines)
        except Exception:
            return None

        # Filter by size
        num_lines = len(func_lines)
        if num_lines < self.min_function_lines or num_lines > self.max_function_lines:
            return None

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Calculate complexity (simple metric: number of nodes)
        complexity = len(list(ast.walk(node)))

        # Extract related papers from docstring
        papers = self._extract_paper_references(docstring)

        # Get function metadata
        metadata = {
            "type": "function",
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "returns": self._get_return_type(node),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
        }

        return CodeChunk(
            repo_name=repo_name,
            file_path=str(file_path.relative_to(file_path.parent.parent.parent)),
            function_name=node.name,
            code=func_code,
            docstring=docstring,
            imports=imports,
            usage_examples=[],  # Will be filled by example extractor
            related_papers=papers,
            metadata=metadata,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            complexity_score=float(complexity),
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        source_code: str,
        file_path: Path,
        repo_name: str,
        imports: list[str],
    ) -> list[CodeChunk]:
        """Extract class and its methods as code chunks."""
        chunks = []

        # Extract class docstring
        class_docstring = ast.get_docstring(node) or ""

        # Extract class-level code (without methods)
        try:
            class_start = node.lineno - 1
            class_end = node.end_lineno

            # Get class definition and __init__ if exists
            lines = source_code.split("\n")

            # Find class header and docstring
            class_header_lines = []
            current_line = class_start

            # Add class definition line
            class_header_lines.append(lines[current_line])
            current_line += 1

            # Add docstring if present
            if class_docstring:
                while current_line < class_end:
                    line = lines[current_line]
                    class_header_lines.append(line)
                    if '"""' in line or "'''" in line:
                        if line.count('"""') == 2 or line.count("'''") == 2:
                            break
                        current_line += 1
                        if current_line < class_end and ('"""' in lines[current_line] or "'''" in lines[current_line]):
                            class_header_lines.append(lines[current_line])
                            break
                    current_line += 1

            class_code = "\n".join(class_header_lines)

        except Exception:
            class_code = f"class {node.name}:\n    ..."

        # Create chunk for class itself
        papers = self._extract_paper_references(class_docstring)

        metadata = {
            "type": "class",
            "name": node.name,
            "bases": [self._get_base_name(b) for b in node.bases],
            "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
        }

        class_chunk = CodeChunk(
            repo_name=repo_name,
            file_path=str(file_path.relative_to(file_path.parent.parent.parent)),
            function_name=f"class {node.name}",
            code=class_code,
            docstring=class_docstring,
            imports=imports,
            usage_examples=[],
            related_papers=papers,
            metadata=metadata,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            complexity_score=float(len(list(ast.walk(node)))),
        )

        chunks.append(class_chunk)

        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_chunk = self._extract_function(item, source_code, file_path, repo_name, imports)
                if method_chunk:
                    # Update method metadata to include class info
                    method_chunk.metadata["class"] = node.name
                    method_chunk.metadata["method_type"] = "instance"
                    if item.args.args and item.args.args[0].arg == "self":
                        method_chunk.metadata["method_type"] = "instance"
                    elif item.args.args and item.args.args[0].arg == "cls":
                        method_chunk.metadata["method_type"] = "class"

                    # Update function name to include class
                    method_chunk.function_name = f"{node.name}.{item.name}"

                    chunks.append(method_chunk)

        return chunks

    def _extract_paper_references(self, text: str) -> list[str]:
        """Extract paper references from docstring."""
        papers = []

        # Pattern for arXiv links
        arxiv_pattern = r"arxiv\.org/abs/(\d+\.\d+)"
        for match in re.finditer(arxiv_pattern, text):
            papers.append(f"arXiv:{match.group(1)}")

        # Pattern for DOI
        doi_pattern = r"doi\.org/(10\.\d+/[^\s]+)"
        for match in re.finditer(doi_pattern, text):
            papers.append(f"DOI:{match.group(1)}")

        # Pattern for paper citations (Author et al. YYYY)
        citation_pattern = r"([A-Z][a-z]+(?:\s+et\s+al\.?)?(?:\s+\(\d{4}\)|\s+\d{4}))"
        for match in re.finditer(citation_pattern, text):
            papers.append(match.group(1))

        return papers[:5]  # Limit to 5 references

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """Get return type annotation if available."""
        if node.returns:
            return ast.unparse(node.returns)
        return "Any"

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            return ast.unparse(decorator.func)
        return ast.unparse(decorator)

    def _get_base_name(self, base: ast.expr) -> str:
        """Get base class name."""
        if isinstance(base, ast.Name):
            return base.id
        return ast.unparse(base)


# ============================================================================
# Example and Test Extractor
# ============================================================================


class ExampleExtractor:
    """Extract usage examples and test cases from code."""

    def __init__(self, config: dict[str, Any]):
        """Initialize example extractor."""
        self.config = config

    def extract_examples_from_repo(self, repo_path: Path, chunks: list[CodeChunk]) -> dict[str, list[str]]:
        """
        Extract usage examples from examples/ and tests/ directories.

        Args:
            repo_path: Repository path
            chunks: List of code chunks

        Returns:
            Dictionary mapping function names to examples
        """
        examples_map = defaultdict(list)

        # Find example and test directories
        example_dirs = [
            repo_path / "examples",
            repo_path / "example",
            repo_path / "demos",
            repo_path / "tests",
            repo_path / "test",
        ]

        for example_dir in example_dirs:
            if not example_dir.exists():
                continue

            for py_file in example_dir.rglob("*.py"):
                try:
                    source = py_file.read_text(encoding="utf-8")

                    # Match function calls to our extracted functions
                    for chunk in chunks:
                        func_name = chunk.function_name.split(".")[-1]  # Get method name

                        # Look for function calls
                        pattern = rf"{func_name}\s*\("
                        if re.search(pattern, source):
                            # Extract context around the call
                            examples = self._extract_call_context(source, func_name)
                            examples_map[chunk.function_name].extend(examples)

                except Exception as e:
                    logger.debug(f"Failed to process example file {py_file}: {e}")

        return examples_map

    def _extract_call_context(self, source: str, func_name: str) -> list[str]:
        """Extract code context around function calls."""
        examples = []
        lines = source.split("\n")

        for i, line in enumerate(lines):
            if re.search(rf"\b{func_name}\s*\(", line):
                # Extract surrounding context (up to 10 lines)
                start = max(0, i - 3)
                end = min(len(lines), i + 7)
                context = "\n".join(lines[start:end])

                if len(context) < 500:  # Limit example size
                    examples.append(context)

        return examples[:3]  # Limit to 3 examples

    def find_test_files(self, repo_path: Path, chunk: CodeChunk) -> list[str]:
        """Find test files that test a given function."""
        test_files = []

        test_dirs = [
            repo_path / "tests",
            repo_path / "test",
        ]

        func_name = chunk.function_name.split(".")[-1]

        for test_dir in test_dirs:
            if not test_dir.exists():
                continue

            for test_file in test_dir.rglob("test_*.py"):
                try:
                    source = test_file.read_text(encoding="utf-8")
                    if func_name in source:
                        test_files.append(str(test_file.relative_to(repo_path)))
                except Exception:
                    pass

        return test_files


# ============================================================================
# Quality Filter
# ============================================================================


class CodeQualityFilter:
    """Filter code chunks based on quality metrics."""

    def __init__(self, config: dict[str, Any]):
        """Initialize quality filter."""
        self.config = config
        self.min_quality_score = config.get("min_quality_score", 0.5)

    def calculate_quality_score(self, chunk: CodeChunk) -> float:
        """
        Calculate quality score for a code chunk.

        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        weights = []

        # Has docstring (30%)
        if chunk.docstring:
            score += 0.3
            weights.append(0.3)

        # Has usage examples (20%)
        if chunk.usage_examples:
            score += 0.2
            weights.append(0.2)

        # Has type hints (15%)
        if chunk.metadata.get("returns") != "Any":
            score += 0.15
            weights.append(0.15)

        # Reasonable complexity (15%)
        if 5 <= chunk.complexity_score <= 100:
            score += 0.15
            weights.append(0.15)

        # Has tests (10%)
        if chunk.test_files:
            score += 0.1
            weights.append(0.1)

        # Code length (10%)
        code_lines = len(chunk.code.split("\n"))
        if 10 <= code_lines <= 150:
            score += 0.1
            weights.append(0.1)

        # Normalize by actual weights
        total_weight = sum(weights) or 1.0
        return score / total_weight

    def filter_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Filter chunks by quality score."""
        filtered = []

        for chunk in chunks:
            quality_score = self.calculate_quality_score(chunk)
            chunk.metadata["quality_score"] = quality_score

            if quality_score >= self.min_quality_score:
                filtered.append(chunk)

        logger.info(
            f"Quality filter: {len(filtered)}/{len(chunks)} chunks passed (threshold: {self.min_quality_score})"
        )

        return filtered

    def deduplicate_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Remove duplicate code chunks based on content hash."""
        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            # Hash based on code content
            code_hash = hashlib.md5(chunk.code.encode()).hexdigest()

            if code_hash not in seen_hashes:
                seen_hashes.add(code_hash)
                unique_chunks.append(chunk)

        logger.info(f"Deduplication: {len(unique_chunks)}/{len(chunks)} unique chunks")

        return unique_chunks


# ============================================================================
# Main Code Corpus Builder
# ============================================================================


class CodeCorpusBuilder:
    """Main orchestrator for building code corpus."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize code corpus builder.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.config = config.get("code_corpus", {})

        # Initialize components
        self.fetcher = RepositoryFetcher(self.config)
        self.parser = PythonCodeParser(self.config)
        self.example_extractor = ExampleExtractor(self.config)
        self.quality_filter = CodeQualityFilter(self.config)

        # Storage
        self.all_chunks: list[CodeChunk] = []
        self.repo_metadata: dict[str, RepositoryMetadata] = {}

        # Output directory
        self.output_dir = Path(self.config.get("output_dir", "./cache/code_corpus"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("CodeCorpusBuilder initialized")

    def build_corpus(
        self,
        repositories: list[dict[str, Any]] | None = None,
        max_repos: int | None = None,
    ) -> list[CodeChunk]:
        """
        Build code corpus from target repositories.

        Args:
            repositories: List of repository info dicts (uses REPOSITORIES if None)
            max_repos: Maximum number of repositories to process

        Returns:
            List of extracted code chunks
        """
        if repositories is None:
            repositories = REPOSITORIES

        if max_repos:
            repositories = repositories[:max_repos]

        logger.info(f"Building code corpus from {len(repositories)} repositories")

        all_chunks = []

        for i, repo_info in enumerate(repositories, 1):
            logger.info(f"Processing repository {i}/{len(repositories)}: {repo_info['name']}")

            try:
                chunks = self.process_repository(repo_info)
                all_chunks.extend(chunks)
                logger.info(f"Extracted {len(chunks)} chunks from {repo_info['name']}")

            except Exception as e:
                logger.error(f"Failed to process {repo_info['name']}: {e}", exc_info=True)
                continue

        logger.info(f"Total chunks extracted: {len(all_chunks)}")

        # Deduplicate
        all_chunks = self.quality_filter.deduplicate_chunks(all_chunks)

        # Filter by quality
        all_chunks = self.quality_filter.filter_chunks(all_chunks)

        self.all_chunks = all_chunks

        logger.info(f"Final corpus size: {len(all_chunks)} chunks")

        return all_chunks

    def process_repository(self, repo_info: dict[str, Any]) -> list[CodeChunk]:
        """Process a single repository."""
        repo_name = repo_info["name"]

        # Check license compliance
        logger.info(f"Checking license compliance for {repo_name}")

        # Fetch repository
        try:
            repo_path = self.fetcher.fetch_repository(repo_info)
        except Exception as e:
            logger.error(f"Failed to fetch {repo_name}: {e}")
            return []

        # Check license
        license_info = self.fetcher.check_license_compliance(repo_path)
        if not license_info["compliant"]:
            logger.warning(f"Repository {repo_name} may have licensing issues: {license_info['type']}")

        # Get metadata
        metadata = self.fetcher.get_repository_metadata(repo_info)

        # Find Python files
        python_files = list(repo_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files in {repo_name}")

        # Parse files
        all_chunks = []
        for py_file in python_files:
            # Skip test files in initial pass
            if "test" in str(py_file).lower() and not self.config.get("include_tests", True):
                continue

            chunks = self.parser.parse_file(py_file, repo_name)
            all_chunks.extend(chunks)

        logger.info(f"Extracted {len(all_chunks)} chunks from {repo_name}")

        # Extract usage examples
        if self.config.get("extract_examples", True):
            examples_map = self.example_extractor.extract_examples_from_repo(repo_path, all_chunks)

            # Add examples to chunks
            for chunk in all_chunks:
                if chunk.function_name in examples_map:
                    chunk.usage_examples = examples_map[chunk.function_name]

        # Find test files
        if self.config.get("find_tests", True):
            for chunk in all_chunks:
                chunk.test_files = self.example_extractor.find_test_files(repo_path, chunk)

        # Update metadata
        metadata.total_files = len(python_files)
        metadata.total_functions = len([c for c in all_chunks if c.metadata["type"] == "function"])
        metadata.total_classes = len([c for c in all_chunks if c.metadata["type"] == "class"])
        metadata.processed_at = str(Path.ctime)

        self.repo_metadata[repo_name] = metadata

        return all_chunks

    def stream_document_chunks(self) -> Iterator[DocumentChunk]:
        """
        Stream code chunks as DocumentChunk objects for RAG integration.

        Yields:
            DocumentChunk objects
        """
        for i, code_chunk in enumerate(self.all_chunks):
            yield code_chunk.to_document_chunk(chunk_id=i)

    def save_corpus(self, output_path: Path | None = None) -> None:
        """Save corpus to disk."""
        if output_path is None:
            output_path = self.output_dir

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save code chunks
        chunks_file = output_path / "code_chunks.json"
        with open(chunks_file, "w") as f:
            json.dump([chunk.to_dict() for chunk in self.all_chunks], f, indent=2)

        logger.info(f"Saved {len(self.all_chunks)} chunks to {chunks_file}")

        # Save metadata
        metadata_file = output_path / "repo_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(
                {name: asdict(meta) for name, meta in self.repo_metadata.items()},
                f,
                indent=2,
            )

        logger.info(f"Saved metadata to {metadata_file}")

        # Save statistics
        stats = self.get_corpus_statistics()
        stats_file = output_path / "corpus_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

    def load_corpus(self, input_path: Path | None = None) -> list[CodeChunk]:
        """Load corpus from disk."""
        if input_path is None:
            input_path = self.output_dir

        chunks_file = input_path / "code_chunks.json"

        if not chunks_file.exists():
            logger.warning(f"Corpus file not found: {chunks_file}")
            return []

        with open(chunks_file) as f:
            chunks_data = json.load(f)

        self.all_chunks = [CodeChunk(**chunk_data) for chunk_data in chunks_data]

        logger.info(f"Loaded {len(self.all_chunks)} chunks from {chunks_file}")

        return self.all_chunks

    def get_corpus_statistics(self) -> dict[str, Any]:
        """Get corpus statistics."""
        if not self.all_chunks:
            return {}

        stats = {
            "total_chunks": len(self.all_chunks),
            "total_repositories": len(self.repo_metadata),
            "repositories": list(self.repo_metadata.keys()),
            "chunk_types": defaultdict(int),
            "avg_code_length": 0,
            "avg_complexity": 0,
            "chunks_with_docstrings": 0,
            "chunks_with_examples": 0,
            "chunks_with_tests": 0,
            "avg_quality_score": 0,
        }

        for chunk in self.all_chunks:
            stats["chunk_types"][chunk.metadata["type"]] += 1
            stats["avg_code_length"] += len(chunk.code)
            stats["avg_complexity"] += chunk.complexity_score

            if chunk.docstring:
                stats["chunks_with_docstrings"] += 1
            if chunk.usage_examples:
                stats["chunks_with_examples"] += 1
            if chunk.test_files:
                stats["chunks_with_tests"] += 1

            stats["avg_quality_score"] += chunk.metadata.get("quality_score", 0)

        n = len(self.all_chunks)
        stats["avg_code_length"] /= n
        stats["avg_complexity"] /= n
        stats["avg_quality_score"] /= n
        stats["chunk_types"] = dict(stats["chunk_types"])

        return stats

    def search_code(self, query: str, top_k: int = 10) -> list[CodeChunk]:
        """
        Simple keyword-based search (for testing without RAG).

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matching code chunks
        """
        query_lower = query.lower()
        scored_chunks = []

        for chunk in self.all_chunks:
            score = 0.0

            # Match in function name
            if query_lower in chunk.function_name.lower():
                score += 10.0

            # Match in docstring
            if query_lower in chunk.docstring.lower():
                score += 5.0

            # Match in code
            if query_lower in chunk.code.lower():
                score += 2.0

            # Match in topics
            for topic in chunk.metadata.get("topics", []):
                if query_lower in topic.lower():
                    score += 3.0

            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score
        scored_chunks.sort(reverse=True, key=lambda x: x[0])

        return [chunk for _, chunk in scored_chunks[:top_k]]


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Build code corpus from repositories")
    parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    parser.add_argument("--max-repos", type=int, help="Maximum number of repos to process")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--load", action="store_true", help="Load existing corpus")
    parser.add_argument("--search", help="Search query (requires --load)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    builder = CodeCorpusBuilder(args.config)

    if args.load:
        logger.info("Loading existing corpus...")
        builder.load_corpus(Path(args.output) if args.output else None)

        if args.search:
            results = builder.search_code(args.search)
            logger.info(f"Found {len(results)} results for '{args.search}':")
            for i, chunk in enumerate(results, 1):
                print(f"\n{i}. {chunk.repo_name} - {chunk.function_name}")
                print(f"   {chunk.file_path}:{chunk.start_line}")
                if chunk.docstring:
                    print(f"   {chunk.docstring[:100]}...")

    else:
        logger.info("Building code corpus...")
        builder.build_corpus(max_repos=args.max_repos)

        # Save corpus
        output_path = Path(args.output) if args.output else None
        builder.save_corpus(output_path)

        # Print statistics
        stats = builder.get_corpus_statistics()
        print("\n" + "=" * 60)
        print("CODE CORPUS STATISTICS")
        print("=" * 60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total repositories: {stats['total_repositories']}")
        print(f"Chunk types: {stats['chunk_types']}")
        print(f"Avg code length: {stats['avg_code_length']:.1f} chars")
        print(f"Avg complexity: {stats['avg_complexity']:.1f}")
        print(f"Chunks with docstrings: {stats['chunks_with_docstrings']}")
        print(f"Chunks with examples: {stats['chunks_with_examples']}")
        print(f"Chunks with tests: {stats['chunks_with_tests']}")
        print(f"Avg quality score: {stats['avg_quality_score']:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
