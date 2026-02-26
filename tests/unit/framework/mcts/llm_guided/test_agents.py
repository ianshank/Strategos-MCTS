"""Tests for LLM Agents."""

import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("numpy", reason="numpy required for MCTS framework")

from src.framework.mcts.llm_guided.agents import (
    CodeVariant,
    GeneratorAgent,
    GeneratorOutput,
    ReflectorAgent,
    ReflectorOutput,
    create_generator_from_config,
    create_reflector_from_config,
)
from src.framework.mcts.llm_guided.config import (
    GeneratorConfig,
    LLMGuidedMCTSConfig,
    ReflectorConfig,
)
from src.framework.mcts.llm_guided.node import NodeState


class TestCodeVariant:
    """Tests for CodeVariant model."""

    def test_creation(self):
        """Test basic creation."""
        variant = CodeVariant(
            code="def foo(): return 1",
            confidence=0.8,
            reasoning="Simple approach",
        )

        assert variant.code == "def foo(): return 1"
        assert variant.confidence == 0.8
        assert variant.reasoning == "Simple approach"

    def test_confidence_bounds(self):
        """Test confidence is bounded to [0, 1]."""
        variant = CodeVariant(code="code", confidence=1.5)
        assert variant.confidence == 1.0

        variant = CodeVariant(code="code", confidence=-0.5)
        assert variant.confidence == 0.0

    def test_default_reasoning(self):
        """Test default reasoning is empty string."""
        variant = CodeVariant(code="code", confidence=0.5)
        assert variant.reasoning == ""


class TestGeneratorOutput:
    """Tests for GeneratorOutput model."""

    def test_creation(self):
        """Test basic creation."""
        output = GeneratorOutput(
            variants=[
                CodeVariant(code="code1", confidence=0.6),
                CodeVariant(code="code2", confidence=0.4),
            ]
        )

        assert len(output.variants) == 2

    def test_action_probs(self):
        """Test action probability computation."""
        output = GeneratorOutput(
            variants=[
                CodeVariant(code="code1", confidence=0.6),
                CodeVariant(code="code2", confidence=0.4),
            ]
        )

        probs = output.action_probs

        assert "variant_0" in probs
        assert "variant_1" in probs
        assert probs["variant_0"] == 0.6
        assert probs["variant_1"] == 0.4

    def test_action_probs_zero_confidence(self):
        """Test action probs when all confidences are zero."""
        output = GeneratorOutput(
            variants=[
                CodeVariant(code="code1", confidence=0.0),
                CodeVariant(code="code2", confidence=0.0),
            ]
        )

        probs = output.action_probs

        # Should be equal probability
        assert probs["variant_0"] == 0.5
        assert probs["variant_1"] == 0.5

    def test_action_probs_empty(self):
        """Test action probs with no variants."""
        output = GeneratorOutput(variants=[])
        assert output.action_probs == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = GeneratorOutput(
            variants=[CodeVariant(code="code", confidence=0.5)],
            total_tokens=100,
        )

        d = output.to_dict()
        assert "variants" in d
        assert "action_probs" in d
        assert d["total_tokens"] == 100


class TestReflectorOutput:
    """Tests for ReflectorOutput model."""

    def test_creation(self):
        """Test basic creation."""
        output = ReflectorOutput(
            value=0.8,
            reflection="Good code",
            is_solution=True,
            issues=["minor issue"],
            suggestions=["use typing"],
        )

        assert output.value == 0.8
        assert output.reflection == "Good code"
        assert output.is_solution is True
        assert len(output.issues) == 1
        assert len(output.suggestions) == 1

    def test_value_bounds(self):
        """Test value is bounded to [0, 1]."""
        output = ReflectorOutput(value=1.5, reflection="test")
        assert output.value == 1.0

        output = ReflectorOutput(value=-0.5, reflection="test")
        assert output.value == 0.0

    def test_default_values(self):
        """Test default values."""
        output = ReflectorOutput(value=0.5, reflection="test")
        assert output.is_solution is False
        assert output.issues == []
        assert output.suggestions == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = ReflectorOutput(
            value=0.7,
            reflection="analysis",
            is_solution=False,
            issues=["issue1"],
            suggestions=["suggestion1"],
            total_tokens=50,
        )

        d = output.to_dict()
        assert d["value"] == 0.7
        assert d["reflection"] == "analysis"
        assert d["is_solution"] is False
        assert d["issues"] == ["issue1"]
        assert d["total_tokens"] == 50


class TestGeneratorAgent:
    """Tests for GeneratorAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock()
        return client

    @pytest.fixture
    def generator_config(self):
        """Create generator config."""
        return GeneratorConfig(
            model="gpt-4o",
            temperature=0.7,
            num_variants=3,
        )

    def test_initialization(self, mock_llm_client, generator_config):
        """Test agent initialization."""
        agent = GeneratorAgent(mock_llm_client, generator_config)

        assert agent.config == generator_config
        assert agent.total_calls == 0
        assert agent.total_tokens == 0

    def test_initialization_default_config(self, mock_llm_client):
        """Test agent initialization with default config."""
        agent = GeneratorAgent(mock_llm_client)
        assert agent.config is not None

    @pytest.mark.asyncio
    async def test_run_success(self, mock_llm_client, generator_config):
        """Test successful code generation."""
        mock_llm_client.complete.return_value = json.dumps(
            {
                "variants": [
                    {"code": "def foo(): return 1", "confidence": 0.8, "reasoning": "simple"},
                    {"code": "def foo(): return int(1)", "confidence": 0.2, "reasoning": "explicit"},
                ]
            }
        )

        agent = GeneratorAgent(mock_llm_client, generator_config)
        state = NodeState(code="", problem="Return 1", test_cases=["assert foo() == 1"])

        output = await agent.run(state)

        assert len(output.variants) == 2
        assert output.variants[0].code == "def foo(): return 1"
        assert output.variants[0].confidence == 0.8
        assert agent.total_calls == 1

    @pytest.mark.asyncio
    async def test_run_with_code_blocks(self, mock_llm_client, generator_config):
        """Test parsing response with code blocks."""
        mock_llm_client.complete.return_value = """
Here are some solutions:

```python
def foo():
    return 1
```

```python
def foo():
    x = 1
    return x
```
"""

        agent = GeneratorAgent(mock_llm_client, generator_config)
        state = NodeState(code="", problem="Return 1")

        output = await agent.run(state)

        assert len(output.variants) >= 1

    @pytest.mark.asyncio
    async def test_run_failure(self, mock_llm_client, generator_config):
        """Test handling of LLM failure."""
        mock_llm_client.complete.side_effect = Exception("API error")

        agent = GeneratorAgent(mock_llm_client, generator_config)
        state = NodeState(code="", problem="test")

        output = await agent.run(state)

        assert len(output.variants) == 0

    def test_reset_stats(self, mock_llm_client, generator_config):
        """Test statistics reset."""
        agent = GeneratorAgent(mock_llm_client, generator_config)
        agent._total_calls = 10
        agent._total_tokens = 1000

        agent.reset_stats()

        assert agent.total_calls == 0
        assert agent.total_tokens == 0


class TestReflectorAgent:
    """Tests for ReflectorAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock()
        return client

    @pytest.fixture
    def reflector_config(self):
        """Create reflector config."""
        return ReflectorConfig(
            model="gpt-4o",
            temperature=0.3,
        )

    def test_initialization(self, mock_llm_client, reflector_config):
        """Test agent initialization."""
        agent = ReflectorAgent(mock_llm_client, reflector_config)

        assert agent.config == reflector_config
        assert agent.total_calls == 0

    @pytest.mark.asyncio
    async def test_run_success(self, mock_llm_client, reflector_config):
        """Test successful code evaluation."""
        mock_llm_client.complete.return_value = json.dumps(
            {
                "value": 0.85,
                "reflection": "Good implementation",
                "is_solution": True,
                "issues": [],
                "suggestions": ["add type hints"],
            }
        )

        agent = ReflectorAgent(mock_llm_client, reflector_config)
        state = NodeState(code="def foo(): return 1", problem="Return 1")

        output = await agent.run(state)

        assert output.value == 0.85
        assert output.reflection == "Good implementation"
        assert output.is_solution is True
        assert agent.total_calls == 1

    @pytest.mark.asyncio
    async def test_run_with_test_results(self, mock_llm_client, reflector_config):
        """Test evaluation with test results."""
        mock_llm_client.complete.return_value = json.dumps(
            {
                "value": 0.3,
                "reflection": "Tests failing",
                "is_solution": False,
                "issues": ["assertion failed"],
            }
        )

        agent = ReflectorAgent(mock_llm_client, reflector_config)
        state = NodeState(code="def foo(): return 0", problem="Return 1")
        test_results = {"passed": False, "errors": ["AssertionError"]}

        output = await agent.run(state, test_results)

        assert output.value == 0.3
        assert output.is_solution is False

    @pytest.mark.asyncio
    async def test_run_parse_fallback(self, mock_llm_client, reflector_config):
        """Test fallback parsing when JSON fails."""
        mock_llm_client.complete.return_value = """
The code looks okay. Value: 0.6
There are some minor issues but overall it should work.
"""

        agent = ReflectorAgent(mock_llm_client, reflector_config)
        state = NodeState(code="def foo(): return 1", problem="test")

        output = await agent.run(state)

        # Should extract value from text
        assert output.value == 0.6

    @pytest.mark.asyncio
    async def test_run_failure(self, mock_llm_client, reflector_config):
        """Test handling of LLM failure."""
        mock_llm_client.complete.side_effect = Exception("API error")

        agent = ReflectorAgent(mock_llm_client, reflector_config)
        state = NodeState(code="def foo(): pass", problem="test")

        output = await agent.run(state)

        # Should return neutral output
        assert output.value == 0.5
        assert output.is_solution is False


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_generator_from_config(self):
        """Test creating generator from full config."""
        mock_client = AsyncMock()
        config = LLMGuidedMCTSConfig(generator_config=GeneratorConfig(model="test-model", temperature=0.5))

        generator = create_generator_from_config(mock_client, config)

        assert generator.config.model == "test-model"
        assert generator.config.temperature == 0.5

    def test_create_reflector_from_config(self):
        """Test creating reflector from full config."""
        mock_client = AsyncMock()
        config = LLMGuidedMCTSConfig(reflector_config=ReflectorConfig(model="test-model", temperature=0.2))

        reflector = create_reflector_from_config(mock_client, config)

        assert reflector.config.model == "test-model"
        assert reflector.config.temperature == 0.2
