"""
Integration tests for Synthetic Knowledge Generator.

These tests validate the end-to-end generation pipeline.
"""

import json
from pathlib import Path
import tempfile

import pytest

from src.adapters.llm.base import LLMResponse
from training.synthetic_knowledge_generator import (
    DOMAIN_VOCABULARIES,
    QUESTION_TEMPLATES,
    QAPair,
    QualityValidator,
    SyntheticKnowledgeGenerator,
)


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0

    async def generate(self, prompt: str, **kwargs):
        """Mock generate method."""
        self.call_count += 1

        # Simulate different responses based on prompt
        if "Reasoning Path" in prompt:
            answer = "Step 1: Analyze the problem. Step 2: Apply MCTS principles. Step 3: Validate results."
            tokens = 50
        else:
            answer = (
                "Monte Carlo Tree Search (MCTS) is a heuristic search algorithm for decision-making. "
                "It combines tree search with random sampling to find optimal moves. The algorithm has four phases: "
                "1) Selection - traverse tree using UCB1, 2) Expansion - add new nodes, "
                "3) Simulation - random playout, 4) Backpropagation - update statistics. "
                "MCTS is widely used in game playing AI, especially for games with large state spaces like Go."
            )
            tokens = 150

        self.total_tokens += tokens

        return LLMResponse(
            text=answer,
            usage={"total_tokens": tokens, "prompt_tokens": 50, "completion_tokens": tokens - 50},
            model="mock-model",
            raw_response=None,
        )


@pytest.fixture
def mock_llm_client():
    """Fixture for mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def temp_output_dir():
    """Fixture for temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestQAPair:
    """Test QAPair data structure."""

    def test_qa_pair_creation(self):
        """Test creating a QAPair."""
        pair = QAPair(
            question="What is MCTS?",
            answer="Monte Carlo Tree Search is...",
            contexts=["Context 1", "Context 2"],
            metadata={"category": "mcts_algorithms", "difficulty": "easy"},
        )

        assert pair.question == "What is MCTS?"
        assert len(pair.contexts) == 2
        assert pair.metadata["category"] == "mcts_algorithms"
        assert 0.0 <= pair.quality_score <= 1.0

    def test_to_langsmith_format(self):
        """Test conversion to LangSmith format."""
        pair = QAPair(
            question="What is UCB1?",
            answer="Upper Confidence Bound 1...",
            contexts=["UCB1 is a formula...", "It balances exploration..."],
            metadata={"category": "exploration_exploitation"},
            quality_score=0.85,
        )

        langsmith = pair.to_langsmith_format()

        assert "inputs" in langsmith
        assert "outputs" in langsmith
        assert "metadata" in langsmith
        assert langsmith["inputs"]["question"] == "What is UCB1?"
        assert langsmith["outputs"]["ground_truth"] == "Upper Confidence Bound 1..."
        assert langsmith["metadata"]["quality_score"] == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        pair = QAPair(
            question="Test?",
            answer="Answer",
            contexts=["C1"],
            metadata={"test": True},
        )

        d = pair.to_dict()

        assert d["question"] == "Test?"
        assert d["answer"] == "Answer"
        assert "generated_at" in d


class TestQualityValidator:
    """Test quality validation."""

    def test_valid_qa_pair(self):
        """Test validation of valid Q&A pair."""
        validator = QualityValidator(min_question_length=20, min_answer_length=100)

        pair = QAPair(
            question="How does UCB1 work in MCTS?",
            answer="UCB1 (Upper Confidence Bound 1) is the formula used in MCTS to select which child node to visit. "
            "It balances exploration of less-visited nodes with exploitation of high-value nodes. "
            "The formula is Q/N + C*sqrt(ln(N_parent)/N).",
            contexts=["UCB1 is a selection policy", "It has an exploration term"],
            metadata={},
        )

        is_valid, errors = validator.validate(pair)

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_short_question(self):
        """Test validation fails for short question."""
        validator = QualityValidator(min_question_length=20, min_answer_length=100)

        pair = QAPair(
            question="What is MCTS?",  # Too short
            answer="A" * 150,  # Long enough
            contexts=["Context"],
            metadata={},
        )

        is_valid, errors = validator.validate(pair)

        assert is_valid is False
        assert any("too short" in err.lower() for err in errors)

    def test_invalid_short_answer(self):
        """Test validation fails for short answer."""
        validator = QualityValidator(min_question_length=20, min_answer_length=100)

        pair = QAPair(
            question="How does MCTS work?",  # Long enough
            answer="Short answer",  # Too short
            contexts=["Context"],
            metadata={},
        )

        is_valid, errors = validator.validate(pair)

        assert is_valid is False
        assert any("too short" in err.lower() for err in errors)

    def test_invalid_no_question_mark(self):
        """Test validation checks for question mark."""
        validator = QualityValidator()

        pair = QAPair(
            question="Explain MCTS algorithm",  # No question mark
            answer="A" * 150,
            contexts=["Context"],
            metadata={},
        )

        is_valid, errors = validator.validate(pair)

        assert is_valid is False
        assert any("?" in err for err in errors)

    def test_invalid_placeholder_text(self):
        """Test validation rejects placeholder text."""
        validator = QualityValidator()

        pair = QAPair(
            question="What is {algorithm}?",  # Placeholder
            answer="A" * 150,
            contexts=["Context"],
            metadata={},
        )

        is_valid, errors = validator.validate(pair)

        assert is_valid is False
        assert any("placeholder" in err.lower() for err in errors)

    def test_quality_scoring(self):
        """Test quality score calculation."""
        validator = QualityValidator()

        # High quality pair with code, examples, structure
        high_quality = QAPair(
            question="How does MCTS work?",
            answer="""MCTS works through four phases:

1. Selection - traverse using UCB1
2. Expansion - add new nodes
3. Simulation - random playout
4. Backpropagation - update stats

For example, in game playing:

```python
def mcts_search(root, iterations):
    for _ in range(iterations):
        node = select(root)
        result = simulate(node)
        backpropagate(node, result)
    return best_move(root)
```

The algorithm provides strong performance in complex domains with high branching factors.
""",
            contexts=["C1", "C2", "C3"],
            metadata={},
        )

        # Low quality pair
        low_quality = QAPair(
            question="What is MCTS?",
            answer="MCTS is a search algorithm.",
            contexts=["C1"],
            metadata={},
        )

        high_score = validator.score_quality(high_quality)
        low_score = validator.score_quality(low_quality)

        assert high_score > low_score
        assert high_score >= 0.5  # Should be fairly high
        assert low_score < 0.3  # Should be low


class TestSyntheticKnowledgeGenerator:
    """Test synthetic knowledge generator."""

    @pytest.mark.asyncio
    async def test_generator_initialization(self, mock_llm_client, temp_output_dir):
        """Test generator initialization."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        assert generator.llm_client == mock_llm_client
        assert Path(generator.output_dir).exists()
        assert generator.stats["total_generated"] == 0

    @pytest.mark.asyncio
    async def test_fill_template(self, mock_llm_client, temp_output_dir):
        """Test template filling."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        template = "Explain {algorithm} step by step with examples"
        filled = generator._fill_template(template)

        assert "{algorithm}" not in filled
        assert filled.startswith("Explain ")
        assert "step by step with examples" in filled

    @pytest.mark.asyncio
    async def test_generate_question_answer(self, mock_llm_client, temp_output_dir):
        """Test generating a single Q&A pair."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        template = "What is {algorithm} and how does it work in practice?"
        qa_pair = await generator._generate_question_answer(template, "mcts_algorithms", "medium")

        assert qa_pair is not None
        assert len(qa_pair.question) > 0
        assert len(qa_pair.answer) > 0
        assert len(qa_pair.contexts) > 0
        assert qa_pair.metadata["category"] == "mcts_algorithms"
        assert qa_pair.metadata["difficulty"] == "medium"

    @pytest.mark.asyncio
    async def test_generate_contexts(self, mock_llm_client, temp_output_dir):
        """Test context extraction."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        question = "What is MCTS?"
        answer = """
MCTS is a search algorithm. It has four phases: Selection, Expansion, Simulation, Backpropagation.

Key benefits:
- Handles large state spaces
- Anytime algorithm
- No evaluation function needed

Example code:
```python
def mcts():
    pass
```
"""

        contexts = generator._generate_contexts(question, answer)

        assert len(contexts) > 0
        assert all(len(ctx) > 0 for ctx in contexts)

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, mock_llm_client, temp_output_dir):
        """Test duplicate question detection."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        # Generate first pair with a longer template to meet minimum length
        template = "What is UCB1 and how does it balance exploration versus exploitation?"
        qa1 = await generator._generate_question_answer(template, "mcts_algorithms")

        assert qa1 is not None

        # Try to generate same question again
        qa2 = await generator._generate_question_answer(template, "mcts_algorithms")

        # Should be rejected as duplicate
        assert qa2 is None

    @pytest.mark.asyncio
    async def test_generate_batch(self, mock_llm_client, temp_output_dir):
        """Test batch generation."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        pairs = await generator.generate_batch(
            num_samples=5,
            categories=["mcts_algorithms"],
            batch_size=2,
        )

        assert len(pairs) > 0
        assert all(isinstance(pair, QAPair) for pair in pairs)
        assert generator.stats["api_calls"] > 0

    @pytest.mark.asyncio
    async def test_save_dataset(self, mock_llm_client, temp_output_dir):
        """Test saving dataset."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        pairs = [
            QAPair(
                question="Test question?",
                answer="Test answer",
                contexts=["C1", "C2"],
                metadata={"test": True},
            )
        ]

        # Save in LangSmith format
        generator.save_dataset(pairs, "test_langsmith.json", format="langsmith")

        # Check file exists
        output_file = Path(temp_output_dir) / "test_langsmith.json"
        assert output_file.exists()

        # Check content
        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 1
        assert "inputs" in data[0]
        assert "outputs" in data[0]

    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, mock_llm_client, temp_output_dir):
        """Test checkpoint save and load."""
        # Create generator and generate some data
        generator1 = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        await generator1.generate_batch(num_samples=3, batch_size=2)
        generator1._save_checkpoint()

        initial_count = generator1.stats["valid_pairs"]

        # Create new generator (should load checkpoint)
        generator2 = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        # Should have loaded previous stats
        assert generator2.stats["valid_pairs"] == initial_count

    @pytest.mark.asyncio
    async def test_filter_by_quality(self, mock_llm_client, temp_output_dir):
        """Test quality filtering."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        pairs = [
            QAPair("Q1?", "A1", ["C"], {}, quality_score=0.3),
            QAPair("Q2?", "A2", ["C"], {}, quality_score=0.7),
            QAPair("Q3?", "A3", ["C"], {}, quality_score=0.9),
        ]

        filtered = generator.filter_by_quality(pairs, min_score=0.6)

        assert len(filtered) == 2
        assert all(p.quality_score >= 0.6 for p in filtered)

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, mock_llm_client, temp_output_dir):
        """Test statistics tracking."""
        generator = SyntheticKnowledgeGenerator(
            llm_client=mock_llm_client,
            output_dir=temp_output_dir,
        )

        await generator.generate_batch(num_samples=3, batch_size=2)

        stats = generator.get_statistics()

        assert stats["total_generated"] > 0
        assert stats["api_calls"] > 0
        assert stats["total_tokens"] > 0
        assert "avg_quality_score" in stats


class TestQuestionTemplates:
    """Test question template structure."""

    def test_all_categories_exist(self):
        """Test that all template categories are defined."""
        assert len(QUESTION_TEMPLATES) > 0
        assert "mcts_algorithms" in QUESTION_TEMPLATES
        assert "exploration_exploitation" in QUESTION_TEMPLATES
        assert "langgraph_workflows" in QUESTION_TEMPLATES

    def test_all_templates_valid(self):
        """Test that all templates are non-empty strings."""
        for _category, templates in QUESTION_TEMPLATES.items():
            assert isinstance(templates, list)
            assert len(templates) > 0
            for template in templates:
                assert isinstance(template, str)
                assert len(template) > 0

    def test_vocabulary_coverage(self):
        """Test that vocabulary covers template placeholders."""
        # Extract all placeholders from templates
        import re

        all_placeholders = set()
        for templates in QUESTION_TEMPLATES.values():
            for template in templates:
                placeholders = re.findall(r"\{(\w+)\}", template)
                all_placeholders.update(placeholders)

        # Check that all have vocabulary
        for placeholder in all_placeholders:
            assert placeholder in DOMAIN_VOCABULARIES, f"Missing vocabulary for {placeholder}"


@pytest.mark.asyncio
async def test_end_to_end_generation(tmp_path):
    """Test complete generation pipeline."""
    # Create mock client
    mock_client = MockLLMClient()

    # Create generator
    generator = SyntheticKnowledgeGenerator(
        llm_client=mock_client,
        output_dir=str(tmp_path),
        config={"min_question_length": 10, "min_answer_length": 50},
    )

    # Generate batch
    pairs = await generator.generate_batch(
        num_samples=5,
        categories=["mcts_algorithms"],
        batch_size=2,
    )

    # Verify results
    assert len(pairs) > 0

    # Filter and save
    filtered = generator.filter_by_quality(pairs, min_score=0.3)
    generator.save_dataset(filtered, "test_output.json", format="langsmith")

    # Verify file
    output_file = tmp_path / "test_output.json"
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)
        assert len(data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
