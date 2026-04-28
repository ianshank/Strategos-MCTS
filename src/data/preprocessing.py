"""
Text Preprocessing Module for Training Data.

Provides utilities for:
- Text cleaning and normalization
- Tokenization with various backends
- Feature extraction for meta-controller training
"""

import re
from dataclasses import dataclass
from typing import Any

from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessedText:
    """Preprocessed text with metadata."""

    original: str
    cleaned: str
    tokens: list[str]
    token_ids: list[int] | None = None
    features: dict[str, Any] | None = None


class TextPreprocessor:
    """
    Text preprocessing pipeline for multi-agent training data.

    Handles:
    - HTML/XML tag removal
    - Special character normalization
    - Whitespace cleanup
    - Domain-specific preprocessing (cyber, military, etc.)
    """

    # Patterns for cleaning
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    MULTIPLE_SPACES = re.compile(r"\s+")
    SPECIAL_CHARS = re.compile(r"[^\w\s\-.,!?;:()[\]{}\"'/]")

    # Domain-specific patterns
    IP_ADDRESS_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}")
    MITRE_TECHNIQUE_PATTERN = re.compile(r"T\d{4}(?:\.\d{3})?")

    def __init__(
        self,
        remove_html: bool = True,
        normalize_urls: bool = True,
        lowercase: bool = False,
        preserve_domain_patterns: bool = True,
    ):
        """
        Initialize preprocessor.

        Args:
            remove_html: Remove HTML/XML tags
            normalize_urls: Replace URLs with placeholder
            lowercase: Convert to lowercase
            preserve_domain_patterns: Keep domain-specific patterns (IPs, CVEs, etc.)
        """
        self.remove_html = remove_html
        self.normalize_urls = normalize_urls
        self.lowercase = lowercase
        self.preserve_domain_patterns = preserve_domain_patterns

    def clean(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        result = text

        # Remove HTML tags
        if self.remove_html:
            result = self.HTML_TAG_PATTERN.sub(" ", result)

        # Preserve or normalize URLs
        if self.normalize_urls:
            if self.preserve_domain_patterns:
                result = self.URL_PATTERN.sub("[URL]", result)
            else:
                result = self.URL_PATTERN.sub("", result)

        # Normalize whitespace
        result = self.MULTIPLE_SPACES.sub(" ", result)

        # Lowercase if requested
        if self.lowercase:
            result = result.lower()

        # Strip leading/trailing whitespace
        result = result.strip()

        return result

    def extract_domain_features(self, text: str) -> dict[str, Any]:
        """
        Extract domain-specific features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of extracted features
        """
        features = {
            "has_ip_addresses": bool(self.IP_ADDRESS_PATTERN.search(text)),
            "ip_count": len(self.IP_ADDRESS_PATTERN.findall(text)),
            "has_cve": bool(self.CVE_PATTERN.search(text)),
            "cve_ids": self.CVE_PATTERN.findall(text),
            "has_mitre_techniques": bool(self.MITRE_TECHNIQUE_PATTERN.search(text)),
            "mitre_techniques": self.MITRE_TECHNIQUE_PATTERN.findall(text),
            "text_length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.findall(r"[.!?]+", text)),
        }

        # Detect domain indicators
        domain_keywords = {
            "cybersecurity": ["attack", "vulnerability", "exploit", "malware", "threat"],
            "military": ["tactical", "reconnaissance", "deployment", "terrain", "objective"],
            "data_analysis": ["dataset", "analysis", "correlation", "statistics", "visualization"],
        }

        for domain, keywords in domain_keywords.items():
            features[f"is_{domain}"] = any(kw in text.lower() for kw in keywords)

        return features

    def preprocess(self, text: str) -> PreprocessedText:
        """
        Full preprocessing pipeline.

        Args:
            text: Raw input text

        Returns:
            PreprocessedText object with all preprocessing results
        """
        cleaned = self.clean(text)
        tokens = cleaned.split()  # Simple whitespace tokenization
        features = self.extract_domain_features(text)

        return PreprocessedText(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            features=features,
        )

    def batch_preprocess(self, texts: list[str]) -> list[PreprocessedText]:
        """
        Preprocess multiple texts.

        Args:
            texts: List of raw texts

        Returns:
            List of PreprocessedText objects
        """
        return [self.preprocess(text) for text in texts]


class TokenizerWrapper:
    """
    Wrapper for various tokenization backends.

    Supports:
    - Simple whitespace tokenization
    - HuggingFace tokenizers
    - Custom vocabularies
    """

    def __init__(
        self,
        backend: str = "simple",
        model_name: str | None = None,
        max_length: int = 512,
    ):
        """
        Initialize tokenizer.

        Args:
            backend: Tokenizer backend ('simple', 'huggingface', 'custom')
            model_name: Model name for HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.backend = backend
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None

        if backend == "huggingface" and model_name:
            self._load_huggingface_tokenizer()

    def _load_huggingface_tokenizer(self):
        """Load HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.max_length,
            )
            logger.info(f"Loaded HuggingFace tokenizer: {self.model_name}")
        except ImportError:
            logger.error("transformers library not installed. Run: pip install transformers")
            raise

    def tokenize(self, text: str) -> tuple[list[str], list[int] | None]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            Tuple of (tokens, token_ids)
        """
        if self.backend == "simple":
            tokens = text.split()[: self.max_length]
            return tokens, None

        elif self.backend == "huggingface" and self._tokenizer:
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            tokens = self._tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            token_ids = encoded["input_ids"]
            return tokens, token_ids

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def batch_tokenize(self, texts: list[str]) -> list[tuple[list[str], list[int] | None]]:
        """
        Tokenize multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of (tokens, token_ids) tuples
        """
        return [self.tokenize(text) for text in texts]

    def encode_for_training(self, texts: list[str]) -> dict[str, Any]:
        """
        Encode texts for model training.

        Args:
            texts: List of input texts

        Returns:
            Dictionary with encoded data ready for training
        """
        if self.backend != "huggingface" or not self._tokenizer:
            raise ValueError("encode_for_training requires HuggingFace backend")

        encoded = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return encoded


class MetaControllerFeatureExtractor:
    """
    Extract features for meta-controller training.

    Converts text and agent state information into numerical features
    suitable for RNN/BERT routing decisions.
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.preprocessor = TextPreprocessor()

    def extract_query_features(self, query: str) -> dict[str, float]:
        """
        Extract numerical features from query text.

        Args:
            query: User query text

        Returns:
            Dictionary of numerical features
        """
        domain_features = self.preprocessor.extract_domain_features(query)

        features = {
            "query_length": domain_features["text_length"] / 10000,  # Normalize
            "word_count": domain_features["word_count"] / 500,
            "sentence_count": domain_features["sentence_count"] / 50,
            "has_technical_terms": float(
                domain_features["has_ip_addresses"]
                or domain_features["has_cve"]
                or domain_features["has_mitre_techniques"]
            ),
            "is_cybersecurity": float(domain_features["is_cybersecurity"]),
            "is_military": float(domain_features["is_military"]),
            "is_data_analysis": float(domain_features["is_data_analysis"]),
            "complexity_score": self._estimate_complexity(query),
        }

        return features

    def _estimate_complexity(self, text: str) -> float:
        """
        Estimate query complexity (0-1 scale).

        Args:
            text: Input text

        Returns:
            Complexity score
        """
        # Simple heuristic based on length, technical terms, etc.
        score = 0.0

        # Length factor
        word_count = len(text.split())
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.1

        # Technical term factor
        technical_indicators = [
            "analyze",
            "compare",
            "evaluate",
            "synthesize",
            "strategic",
            "tactical",
            "multi-step",
            "consider",
        ]
        for term in technical_indicators:
            if term in text.lower():
                score += 0.1

        # Question complexity
        if "?" in text:
            if any(kw in text.lower() for kw in ["why", "how", "what if"]):
                score += 0.2
            else:
                score += 0.1

        return min(score, 1.0)

    def extract_agent_state_features(
        self,
        hrm_confidence: float = 0.0,
        trm_confidence: float = 0.0,
        mcts_iterations: int = 0,
        consensus_score: float = 0.0,
        rag_retrieved: int = 0,
    ) -> list[float]:
        """
        Extract features from current agent state.

        Args:
            hrm_confidence: HRM agent confidence
            trm_confidence: TRM agent confidence
            mcts_iterations: MCTS iterations completed
            consensus_score: Inter-agent consensus
            rag_retrieved: Number of RAG documents retrieved

        Returns:
            List of normalized features (10-dimensional)
        """
        return [
            hrm_confidence,
            trm_confidence,
            min(mcts_iterations / 1000, 1.0),
            consensus_score,
            min(rag_retrieved / 20, 1.0),
            # Derived features
            abs(hrm_confidence - trm_confidence),  # Disagreement
            (hrm_confidence + trm_confidence) / 2,  # Average confidence
            float(mcts_iterations > 0),  # MCTS active
            float(consensus_score > 0.7),  # High consensus
            float(rag_retrieved > 0),  # RAG used
        ]
