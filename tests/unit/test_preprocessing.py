"""Unit tests for src/data/preprocessing.py."""

from __future__ import annotations

import pytest

from src.data.preprocessing import (
    MetaControllerFeatureExtractor,
    PreprocessedText,
    TextPreprocessor,
    TokenizerWrapper,
)


@pytest.mark.unit
class TestPreprocessedText:
    """Tests for PreprocessedText dataclass."""

    def test_creation(self):
        result = PreprocessedText(original="hello", cleaned="hello", tokens=["hello"])
        assert result.original == "hello"
        assert result.cleaned == "hello"
        assert result.tokens == ["hello"]
        assert result.token_ids is None
        assert result.features is None

    def test_creation_with_all_fields(self):
        result = PreprocessedText(
            original="raw",
            cleaned="clean",
            tokens=["clean"],
            token_ids=[1],
            features={"length": 3},
        )
        assert result.token_ids == [1]
        assert result.features == {"length": 3}


@pytest.mark.unit
class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_default_init(self):
        pp = TextPreprocessor()
        assert pp.remove_html is True
        assert pp.normalize_urls is True
        assert pp.lowercase is False
        assert pp.preserve_domain_patterns is True

    def test_custom_init(self):
        pp = TextPreprocessor(remove_html=False, lowercase=True)
        assert pp.remove_html is False
        assert pp.lowercase is True

    # -- clean() tests --

    def test_clean_empty_string(self):
        pp = TextPreprocessor()
        assert pp.clean("") == ""

    def test_clean_removes_html_tags(self):
        pp = TextPreprocessor()
        result = pp.clean("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_clean_preserves_html_when_disabled(self):
        pp = TextPreprocessor(remove_html=False)
        text = "<p>Hello</p>"
        result = pp.clean(text)
        assert "<p>" in result

    def test_clean_normalizes_urls_with_placeholder(self):
        pp = TextPreprocessor(normalize_urls=True, preserve_domain_patterns=True)
        result = pp.clean("Visit https://example.com for info")
        assert "[URL]" in result
        assert "https://example.com" not in result

    def test_clean_removes_urls_without_preserve(self):
        pp = TextPreprocessor(normalize_urls=True, preserve_domain_patterns=False)
        result = pp.clean("Visit https://example.com for info")
        assert "[URL]" not in result
        assert "https://example.com" not in result

    def test_clean_preserves_urls_when_disabled(self):
        pp = TextPreprocessor(normalize_urls=False)
        text = "Visit https://example.com"
        result = pp.clean(text)
        assert "https://example.com" in result

    def test_clean_normalizes_whitespace(self):
        pp = TextPreprocessor()
        result = pp.clean("hello   world\t\nfoo")
        assert result == "hello world foo"

    def test_clean_lowercase(self):
        pp = TextPreprocessor(lowercase=True)
        result = pp.clean("Hello WORLD")
        assert result == "hello world"

    def test_clean_strips_whitespace(self):
        pp = TextPreprocessor()
        result = pp.clean("  hello  ")
        assert result == "hello"

    # -- extract_domain_features() tests --

    def test_extract_domain_features_ip_addresses(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("Server at 192.168.1.1 and 10.0.0.1")
        assert features["has_ip_addresses"] is True
        assert features["ip_count"] == 2

    def test_extract_domain_features_no_ip(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("No addresses here")
        assert features["has_ip_addresses"] is False
        assert features["ip_count"] == 0

    def test_extract_domain_features_cve(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("Patch CVE-2024-12345 immediately")
        assert features["has_cve"] is True
        assert "CVE-2024-12345" in features["cve_ids"]

    def test_extract_domain_features_mitre(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("Uses technique T1059.001")
        assert features["has_mitre_techniques"] is True
        assert "T1059.001" in features["mitre_techniques"]

    def test_extract_domain_features_text_stats(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("Hello world. How are you?")
        assert features["text_length"] == 25
        assert features["word_count"] == 5
        assert features["sentence_count"] == 2  # . and ?

    def test_extract_domain_features_cybersecurity(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("The malware exploited a vulnerability")
        assert features["is_cybersecurity"] is True
        assert features["is_military"] is False

    def test_extract_domain_features_military(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("Tactical reconnaissance of the terrain")
        assert features["is_military"] is True

    def test_extract_domain_features_data_analysis(self):
        pp = TextPreprocessor()
        features = pp.extract_domain_features("Run analysis on the dataset with correlation")
        assert features["is_data_analysis"] is True

    # -- preprocess() tests --

    def test_preprocess_full_pipeline(self):
        pp = TextPreprocessor()
        result = pp.preprocess("<p>Check CVE-2024-1234 at 10.0.0.1</p>")
        assert isinstance(result, PreprocessedText)
        assert "<p>" not in result.cleaned
        assert len(result.tokens) > 0
        assert result.features is not None
        assert result.features["has_cve"] is True
        assert result.original == "<p>Check CVE-2024-1234 at 10.0.0.1</p>"

    def test_preprocess_empty(self):
        pp = TextPreprocessor()
        result = pp.preprocess("")
        assert result.cleaned == ""
        assert result.tokens == []

    # -- batch_preprocess() tests --

    def test_batch_preprocess(self):
        pp = TextPreprocessor()
        results = pp.batch_preprocess(["hello world", "foo bar"])
        assert len(results) == 2
        assert all(isinstance(r, PreprocessedText) for r in results)

    def test_batch_preprocess_empty_list(self):
        pp = TextPreprocessor()
        results = pp.batch_preprocess([])
        assert results == []


@pytest.mark.unit
class TestTokenizerWrapper:
    """Tests for TokenizerWrapper class."""

    def test_simple_backend_init(self):
        tok = TokenizerWrapper(backend="simple", max_length=128)
        assert tok.backend == "simple"
        assert tok.max_length == 128
        assert tok._tokenizer is None

    def test_simple_tokenize(self):
        tok = TokenizerWrapper(backend="simple", max_length=5)
        tokens, ids = tok.tokenize("one two three four five six seven")
        assert tokens == ["one", "two", "three", "four", "five"]
        assert ids is None

    def test_simple_tokenize_empty(self):
        tok = TokenizerWrapper(backend="simple")
        tokens, ids = tok.tokenize("")
        assert tokens == []
        assert ids is None

    def test_unsupported_backend_raises(self):
        tok = TokenizerWrapper(backend="unknown")
        with pytest.raises(ValueError, match="Unsupported backend"):
            tok.tokenize("hello")

    def test_batch_tokenize(self):
        tok = TokenizerWrapper(backend="simple")
        results = tok.batch_tokenize(["hello world", "foo"])
        assert len(results) == 2
        assert results[0] == (["hello", "world"], None)
        assert results[1] == (["foo"], None)

    def test_encode_for_training_requires_huggingface(self):
        tok = TokenizerWrapper(backend="simple")
        with pytest.raises(ValueError, match="encode_for_training requires HuggingFace"):
            tok.encode_for_training(["hello"])


@pytest.mark.unit
class TestMetaControllerFeatureExtractor:
    """Tests for MetaControllerFeatureExtractor class."""

    def test_init(self):
        extractor = MetaControllerFeatureExtractor()
        assert extractor.preprocessor is not None

    def test_extract_query_features_basic(self):
        extractor = MetaControllerFeatureExtractor()
        features = extractor.extract_query_features("Hello world")
        assert "query_length" in features
        assert "word_count" in features
        assert "complexity_score" in features
        assert "is_cybersecurity" in features
        assert all(isinstance(v, float) for v in features.values())

    def test_extract_query_features_technical(self):
        extractor = MetaControllerFeatureExtractor()
        features = extractor.extract_query_features("Analyze CVE-2024-1234 at 10.0.0.1")
        assert features["has_technical_terms"] == 1.0
        assert features["is_cybersecurity"] == 0.0 or features["is_cybersecurity"] == 1.0

    def test_extract_query_features_normalized(self):
        extractor = MetaControllerFeatureExtractor()
        features = extractor.extract_query_features("short")
        assert features["query_length"] < 1.0
        assert features["word_count"] < 1.0

    def test_estimate_complexity_simple(self):
        extractor = MetaControllerFeatureExtractor()
        score = extractor._estimate_complexity("hello")
        assert score == 0.0

    def test_estimate_complexity_complex(self):
        extractor = MetaControllerFeatureExtractor()
        text = "How can we analyze and evaluate the strategic tactical deployment? " * 5
        score = extractor._estimate_complexity(text)
        assert score > 0.0
        assert score <= 1.0

    def test_estimate_complexity_capped_at_one(self):
        extractor = MetaControllerFeatureExtractor()
        # Include many technical indicators + long text + question words
        text = (
            "How do we analyze compare evaluate synthesize strategic tactical " "multi-step consider why what if? " * 10
        )
        score = extractor._estimate_complexity(text)
        assert score == 1.0

    def test_extract_agent_state_features_defaults(self):
        extractor = MetaControllerFeatureExtractor()
        features = extractor.extract_agent_state_features()
        assert len(features) == 10
        assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_extract_agent_state_features_values(self):
        extractor = MetaControllerFeatureExtractor()
        features = extractor.extract_agent_state_features(
            hrm_confidence=0.8,
            trm_confidence=0.6,
            mcts_iterations=500,
            consensus_score=0.9,
            rag_retrieved=5,
        )
        assert len(features) == 10
        assert features[0] == 0.8  # hrm
        assert features[1] == 0.6  # trm
        assert features[2] == 0.5  # mcts_iterations / 1000
        assert features[3] == 0.9  # consensus
        assert features[4] == 0.25  # rag / 20
        assert features[5] == pytest.approx(0.2)  # disagreement
        assert features[6] == pytest.approx(0.7)  # avg confidence
        assert features[7] == 1.0  # mcts active
        assert features[8] == 1.0  # high consensus
        assert features[9] == 1.0  # rag used

    def test_extract_agent_state_features_normalization(self):
        extractor = MetaControllerFeatureExtractor()
        features = extractor.extract_agent_state_features(mcts_iterations=5000, rag_retrieved=100)
        # Capped at 1.0
        assert features[2] == 1.0
        assert features[4] == 1.0
