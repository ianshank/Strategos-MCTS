"""
Personality Response Generator for LangGraph Multi-Agent MCTS Framework.

This module provides a conversational personality layer that transforms
technical agent responses into friendly, balanced advisor outputs while
maintaining transparency and ethical considerations.

Following 2025 best practices:
- Type hints throughout
- Comprehensive docstrings (Google style)
- Dataclasses for configuration
- Property-based encapsulation
- Exception handling
- Logging for observability
"""

import logging
import re
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonalityTraits:
    """
    Immutable configuration for personality traits.

    Attributes:
        loyalty: Commitment to user's goals (0.0-1.0)
        curiosity: Tendency to explore alternatives (0.0-1.0)
        aspiration: Drive toward optimal solutions (0.0-1.0)
        ethical_weight: Consideration of ethical implications (0.0-1.0)
        transparency: Openness about reasoning and limitations (0.0-1.0)
    """

    loyalty: float = 0.95
    curiosity: float = 0.85
    aspiration: float = 0.90
    ethical_weight: float = 0.92
    transparency: float = 0.88

    def __post_init__(self) -> None:
        """Validate trait values are in range [0.0, 1.0]."""
        for trait_name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Trait '{trait_name}' must be in range [0.0, 1.0], got {value}")


@dataclass
class PersonalityResponseGenerator:
    """
    Generates personality-infused responses based on configurable traits.

    This class transforms technical agent outputs into conversational,
    balanced advisor responses that maintain transparency while being
    approachable and user-friendly.

    Attributes:
        traits: PersonalityTraits configuration

    Example:
        >>> generator = PersonalityResponseGenerator()
        >>> response = generator.generate_response(
        ...     agent_response="Technical analysis complete.",
        ...     query="How do I optimize my code?"
        ... )
        >>> print(response)
        Let me be transparent about my approach...
    """

    traits: PersonalityTraits = field(default_factory=PersonalityTraits)

    # Class-level constants for phrase templates
    TRANSPARENCY_PHRASES: ClassVar[list[str]] = [
        "Let me be transparent about",
        "I want to be clear that",
        "To be honest",
        "Let me share openly",
    ]

    CURIOSITY_PHRASES: ClassVar[list[str]] = [
        "I'm curious about exploring",
        "There are interesting alternatives worth considering",
        "It might be valuable to also look at",
        "I wonder if we could also approach this by",
    ]

    ASPIRATION_PHRASES: ClassVar[list[str]] = [
        "I'm committed to helping you find the best solution",
        "Let's aim for the optimal approach",
        "I believe we can achieve even better results by",
        "Striving for excellence",
    ]

    LOYALTY_PHRASES: ClassVar[list[str]] = [
        "I'm here to support your goals",
        "Your success is my priority",
        "I'm committed to helping you succeed",
        "Working together toward your objectives",
    ]

    ETHICAL_PHRASES: ClassVar[list[str]] = [
        "It's important to consider the ethical implications",
        "Let's ensure this aligns with best practices",
        "We should be mindful of",
        "From an ethical standpoint",
    ]

    def generate_response(
        self,
        agent_response: str,
        query: str,
        include_preamble: bool = True,
        max_length: int = 1000,
    ) -> str:
        """
        Generate a personality-infused response from technical agent output.

        Args:
            agent_response: The original technical response from the agent
            query: The original user query for context
            include_preamble: Whether to include personality preamble
            max_length: Maximum length of the generated response

        Returns:
            A conversational, personality-infused version of the response

        Raises:
            ValueError: If agent_response or query is empty

        Example:
            >>> gen = PersonalityResponseGenerator()
            >>> response = gen.generate_response(
            ...     "[HRM Analysis] Breaking down hierarchically...",
            ...     "How do I solve this problem?"
            ... )
            >>> "transparent" in response.lower()
            True
        """
        # Input validation
        if not agent_response or not agent_response.strip():
            raise ValueError("agent_response cannot be empty")
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        try:
            # Build the personality-infused response
            parts = []

            # Add preamble based on traits
            if include_preamble:
                preamble = self._generate_preamble(query)
                parts.append(preamble)

            # Transform the technical response
            transformed_response = self._transform_response(agent_response, query)
            parts.append(transformed_response)

            # Add trait-based closing
            closing = self._generate_closing(agent_response)
            if closing:
                parts.append(closing)

            # Combine and truncate if needed
            full_response = "\n\n".join(parts)

            if len(full_response) > max_length:
                full_response = full_response[: max_length - 3] + "..."
                logger.warning(f"Response truncated to {max_length} characters")

            return full_response

        except Exception as e:
            logger.error(f"Error generating personality response: {e}", exc_info=True)
            # Fallback to original response with simple wrapper
            return f"Here's what I found:\n\n{agent_response}"

    def _generate_preamble(self, query: str) -> str:
        """
        Generate an opening preamble based on personality traits.

        Args:
            query: The user's query

        Returns:
            A personalized preamble
        """
        preamble_parts = []

        # Transparency (highest weight)
        if self.traits.transparency >= 0.8:
            preamble_parts.append(f"{self.TRANSPARENCY_PHRASES[0]} my approach to your query. ")

        # Loyalty
        if self.traits.loyalty >= 0.9:
            preamble_parts.append(f"{self.LOYALTY_PHRASES[0]}, and I've carefully analyzed your question. ")

        return "".join(preamble_parts).strip()

    def _transform_response(self, agent_response: str, query: str) -> str:
        """
        Transform technical agent response into conversational tone.

        Args:
            agent_response: Original technical response
            query: User query for context

        Returns:
            Conversational version of the response
        """
        # Extract agent name from response if present
        agent_match = re.search(r"\[(.*?)\]", agent_response)
        agent_name = agent_match.group(1) if agent_match else "the agent"

        # Remove technical markers like [HRM Analysis], [TRM Analysis], etc.
        cleaned_response = re.sub(r"\[.*?\]\s*", "", agent_response)

        # Create conversational wrapper
        conversational = (
            f"Based on my analysis using {agent_name.lower()}, "
            f"I've identified the following approach:\n\n{cleaned_response}"
        )

        return conversational

    def _generate_closing(self, agent_response: str) -> str:
        """
        Generate a closing statement based on traits.

        Args:
            agent_response: The agent response (to check for certain keywords)

        Returns:
            A closing statement or empty string
        """
        closing_parts = []

        # Aspiration - offer to go further
        if self.traits.aspiration >= 0.85:
            closing_parts.append("I'm committed to helping you achieve the best possible outcome. ")

        # Curiosity - suggest alternatives
        if self.traits.curiosity >= 0.8 and any(
            keyword in agent_response.lower() for keyword in ["optimize", "improve", "compare"]
        ):
            closing_parts.append("I'm curious if you'd like to explore alternative approaches as well. ")

        # Ethical considerations for certain technical queries
        if self.traits.ethical_weight >= 0.9 and any(
            keyword in agent_response.lower() for keyword in ["system", "design", "architecture", "security"]
        ):
            closing_parts.append(
                "As we proceed, let's ensure our approach aligns with best practices and ethical considerations. "
            )

        return "".join(closing_parts).strip()

    @property
    def trait_summary(self) -> dict[str, float]:
        """
        Get a summary of current personality traits.

        Returns:
            Dictionary mapping trait names to their values
        """
        return {
            "loyalty": self.traits.loyalty,
            "curiosity": self.traits.curiosity,
            "aspiration": self.traits.aspiration,
            "ethical_weight": self.traits.ethical_weight,
            "transparency": self.traits.transparency,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PersonalityResponseGenerator("
            f"loyalty={self.traits.loyalty:.2f}, "
            f"curiosity={self.traits.curiosity:.2f}, "
            f"aspiration={self.traits.aspiration:.2f}, "
            f"ethical_weight={self.traits.ethical_weight:.2f}, "
            f"transparency={self.traits.transparency:.2f})"
        )


# Example usage
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)

    # Create generator with default traits
    generator = PersonalityResponseGenerator()

    # Example technical response
    agent_response = (
        "[HRM Analysis] Breaking down the problem hierarchically: "
        "What are the key factors to consider when choosing between "
        "microservices and monolithic architecture?..."
    )
    query = "What are the key factors to consider when choosing between microservices and monolithic architecture?"

    # Generate personality response
    personality_response = generator.generate_response(agent_response, query)

    print("=" * 80)
    print("ORIGINAL RESPONSE:")
    print("=" * 80)
    print(agent_response)
    print("\n" + "=" * 80)
    print("PERSONALITY-INFUSED RESPONSE:")
    print("=" * 80)
    print(personality_response)
    print("\n" + "=" * 80)
    print(f"Trait Summary: {generator.trait_summary}")
