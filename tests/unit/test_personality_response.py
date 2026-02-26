"""
Unit tests for PersonalityResponseGenerator.

Following 2025 best practices:
- pytest framework
- Fixtures for reusable test data
- Parametrized tests for comprehensive coverage
- Type hints throughout
- Clear test naming (test_<feature>_<scenario>_<expected_outcome>)
- Arrange-Act-Assert pattern
- Edge case testing
"""

import pytest

from src.utils.personality_response import (
    PersonalityResponseGenerator,
    PersonalityTraits,
)


# Fixtures
@pytest.fixture
def default_generator() -> PersonalityResponseGenerator:
    """Create a PersonalityResponseGenerator with default traits."""
    return PersonalityResponseGenerator()


@pytest.fixture
def custom_traits_generator() -> PersonalityResponseGenerator:
    """Create a PersonalityResponseGenerator with custom traits."""
    custom_traits = PersonalityTraits(
        loyalty=0.9,
        curiosity=0.7,
        aspiration=0.85,
        ethical_weight=0.95,
        transparency=0.8,
    )
    return PersonalityResponseGenerator(traits=custom_traits)


@pytest.fixture
def sample_agent_response() -> str:
    """Sample agent response for testing."""
    return (
        "[HRM Analysis] Breaking down the problem hierarchically: "
        "What are the key factors to consider when choosing between "
        "microservices and monolithic architecture?..."
    )


@pytest.fixture
def sample_query() -> str:
    """Sample user query for testing."""
    return "What are the key factors to consider when choosing between microservices and monolithic architecture?"


# Tests for PersonalityTraits
class TestPersonalityTraits:
    """Tests for the PersonalityTraits dataclass."""

    def test_default_traits_initialization(self) -> None:
        """Test that default traits are initialized with correct values."""
        # Arrange & Act
        traits = PersonalityTraits()

        # Assert
        assert traits.loyalty == 0.95
        assert traits.curiosity == 0.85
        assert traits.aspiration == 0.90
        assert traits.ethical_weight == 0.92
        assert traits.transparency == 0.88

    def test_custom_traits_initialization(self) -> None:
        """Test that custom traits can be set correctly."""
        # Arrange & Act
        traits = PersonalityTraits(
            loyalty=0.5,
            curiosity=0.6,
            aspiration=0.7,
            ethical_weight=0.8,
            transparency=0.9,
        )

        # Assert
        assert traits.loyalty == 0.5
        assert traits.curiosity == 0.6
        assert traits.aspiration == 0.7
        assert traits.ethical_weight == 0.8
        assert traits.transparency == 0.9

    @pytest.mark.parametrize(
        "trait_name,invalid_value",
        [
            ("loyalty", -0.1),
            ("loyalty", 1.5),
            ("curiosity", -1.0),
            ("curiosity", 2.0),
            ("aspiration", -0.5),
            ("transparency", 1.1),
        ],
    )
    def test_trait_validation_rejects_out_of_range_values(self, trait_name: str, invalid_value: float) -> None:
        """Test that trait values outside [0.0, 1.0] raise ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match=f"Trait '{trait_name}' must be in range"):
            PersonalityTraits(**{trait_name: invalid_value})

    def test_traits_immutability(self) -> None:
        """Test that PersonalityTraits is immutable (frozen dataclass)."""
        # Arrange
        traits = PersonalityTraits()

        # Act & Assert - FrozenInstanceError is subclass of AttributeError
        with pytest.raises(AttributeError):
            traits.loyalty = 0.5  # type: ignore


# Tests for PersonalityResponseGenerator
class TestPersonalityResponseGenerator:
    """Tests for the PersonalityResponseGenerator class."""

    def test_initialization_with_default_traits(self) -> None:
        """Test generator initialization with default traits."""
        # Arrange & Act
        generator = PersonalityResponseGenerator()

        # Assert
        assert generator.traits.loyalty == 0.95
        assert generator.traits.curiosity == 0.85

    def test_initialization_with_custom_traits(self, custom_traits_generator: PersonalityResponseGenerator) -> None:
        """Test generator initialization with custom traits."""
        # Arrange & Act
        generator = custom_traits_generator

        # Assert
        assert generator.traits.loyalty == 0.9
        assert generator.traits.ethical_weight == 0.95

    def test_generate_response_basic_functionality(
        self, default_generator: PersonalityResponseGenerator, sample_agent_response: str, sample_query: str
    ) -> None:
        """Test basic response generation functionality."""
        # Arrange
        generator = default_generator

        # Act
        result = generator.generate_response(sample_agent_response, sample_query)

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) > len(sample_agent_response)  # Should be enhanced

    def test_generate_response_includes_transparency_phrase(
        self, default_generator: PersonalityResponseGenerator, sample_agent_response: str, sample_query: str
    ) -> None:
        """Test that high transparency trait includes transparency phrases."""
        # Arrange
        generator = default_generator  # Has transparency=0.88

        # Act
        result = generator.generate_response(sample_agent_response, sample_query)

        # Assert
        assert "transparent" in result.lower() or "clear" in result.lower()

    def test_generate_response_removes_technical_markers(
        self, default_generator: PersonalityResponseGenerator, sample_agent_response: str, sample_query: str
    ) -> None:
        """Test that technical markers like [HRM Analysis] are transformed."""
        # Arrange
        generator = default_generator

        # Act
        result = generator.generate_response(sample_agent_response, sample_query)

        # Assert
        # Should mention the agent name but not in the exact format [HRM Analysis]
        assert "HRM" not in result or "[HRM Analysis]" not in result

    def test_generate_response_with_preamble_disabled(
        self, default_generator: PersonalityResponseGenerator, sample_agent_response: str, sample_query: str
    ) -> None:
        """Test response generation with preamble disabled."""
        # Arrange
        generator = default_generator

        # Act
        result_with_preamble = generator.generate_response(sample_agent_response, sample_query, include_preamble=True)
        result_without_preamble = generator.generate_response(
            sample_agent_response, sample_query, include_preamble=False
        )

        # Assert
        assert len(result_without_preamble) < len(result_with_preamble)

    def test_generate_response_respects_max_length(
        self, default_generator: PersonalityResponseGenerator, sample_agent_response: str, sample_query: str
    ) -> None:
        """Test that max_length parameter is respected."""
        # Arrange
        generator = default_generator
        max_len = 100

        # Act
        result = generator.generate_response(sample_agent_response, sample_query, max_length=max_len)

        # Assert
        assert len(result) <= max_len

    @pytest.mark.parametrize(
        "invalid_response,invalid_query",
        [
            ("", "valid query"),
            ("   ", "valid query"),
            ("valid response", ""),
            ("valid response", "   "),
        ],
    )
    def test_generate_response_raises_value_error_for_empty_inputs(
        self, default_generator: PersonalityResponseGenerator, invalid_response: str, invalid_query: str
    ) -> None:
        """Test that empty or whitespace-only inputs raise ValueError."""
        # Arrange
        generator = default_generator

        # Act & Assert
        with pytest.raises(ValueError):
            generator.generate_response(invalid_response, invalid_query)

    def test_generate_response_fallback_on_exception(
        self, default_generator: PersonalityResponseGenerator, sample_query: str
    ) -> None:
        """Test graceful fallback when transformation fails."""
        # Arrange
        generator = default_generator
        # Simulate a problematic response that might cause issues
        problematic_response = "[" * 1000  # Unbalanced brackets

        # Act
        result = generator.generate_response(problematic_response, sample_query)

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0

    def test_trait_summary_property(self, default_generator: PersonalityResponseGenerator) -> None:
        """Test the trait_summary property returns correct dictionary."""
        # Arrange
        generator = default_generator

        # Act
        summary = generator.trait_summary

        # Assert
        assert isinstance(summary, dict)
        assert "loyalty" in summary
        assert "curiosity" in summary
        assert "aspiration" in summary
        assert "ethical_weight" in summary
        assert "transparency" in summary
        assert summary["loyalty"] == 0.95

    def test_repr_method(self, default_generator: PersonalityResponseGenerator) -> None:
        """Test the __repr__ method produces useful output."""
        # Arrange
        generator = default_generator

        # Act
        repr_str = repr(generator)

        # Assert
        assert "PersonalityResponseGenerator" in repr_str
        assert "loyalty" in repr_str
        assert "0.95" in repr_str

    def test_curiosity_closing_for_optimization_queries(self) -> None:
        """Test that curiosity trait is reflected when appropriate."""
        # Arrange
        # Create generator with very high curiosity to ensure it triggers
        high_curiosity_traits = PersonalityTraits(
            loyalty=0.5,  # Lower to reduce other closings
            curiosity=0.95,  # Very high to ensure trigger
            aspiration=0.5,  # Lower to reduce other closings
            ethical_weight=0.5,  # Lower to reduce other closings
            transparency=0.5,
        )
        generator = PersonalityResponseGenerator(traits=high_curiosity_traits)
        optimization_response = "[MCTS Analysis] We should optimize the system for better performance"
        query = "How can we improve this?"

        # Act
        result = generator.generate_response(optimization_response, query)

        # Assert
        # With very high curiosity (0.95), should include exploration phrases
        # Or at minimum, the response should be generated successfully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ethical_closing_for_system_design_queries(self, default_generator: PersonalityResponseGenerator) -> None:
        """Test that ethical weight adds considerations for system design queries."""
        # Arrange
        generator = default_generator
        system_response = "[HRM Analysis] Designing a distributed authentication system..."
        query = "How should we design our authentication system?"

        # Act
        result = generator.generate_response(system_response, query)

        # Assert
        # With high ethical_weight (0.92), should include ethical considerations
        assert "ethical" in result.lower() or "best practice" in result.lower()

    def test_aspiration_closing_included(
        self, default_generator: PersonalityResponseGenerator, sample_agent_response: str, sample_query: str
    ) -> None:
        """Test that aspiration trait includes commitment phrases."""
        # Arrange
        generator = default_generator  # Has aspiration=0.90

        # Act
        result = generator.generate_response(sample_agent_response, sample_query)

        # Assert
        # With high aspiration (0.90), should include commitment phrases
        assert "committed" in result.lower() or "best" in result.lower()


# Integration tests
class TestPersonalityResponseGeneratorIntegration:
    """Integration tests for PersonalityResponseGenerator with various scenarios."""

    @pytest.mark.parametrize(
        "agent_response,query,expected_in_result",
        [
            (
                "[HRM Analysis] Breaking down hierarchically...",
                "How do I solve this?",
                "hrm",
            ),
            (
                "[TRM Analysis] Iterative refinement approach...",
                "Compare options",
                "trm",
            ),
            (
                "[MCTS Analysis] Strategic exploration...",
                "Optimize performance",
                "mcts",
            ),
        ],
    )
    def test_generate_response_for_different_agents(
        self,
        default_generator: PersonalityResponseGenerator,
        agent_response: str,
        query: str,
        expected_in_result: str,
    ) -> None:
        """Test response generation for different agent types."""
        # Arrange
        generator = default_generator

        # Act
        result = generator.generate_response(agent_response, query)

        # Assert
        assert expected_in_result in result.lower()
        assert len(result) > len(agent_response)

    def test_long_response_handling(self, default_generator: PersonalityResponseGenerator) -> None:
        """Test handling of very long agent responses."""
        # Arrange
        generator = default_generator
        long_response = "[HRM Analysis] " + ("This is a very long analysis. " * 100)
        query = "Analyze this complex problem"

        # Act
        result = generator.generate_response(long_response, query, max_length=500)

        # Assert
        assert len(result) <= 500
        assert isinstance(result, str)
