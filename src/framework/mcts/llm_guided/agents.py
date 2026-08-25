"""
LLM Agents for LLM-Guided MCTS.

Provides:
- GeneratorAgent: Generates code variants with confidence scores
- ReflectorAgent: Evaluates code and provides value estimates
- Structured output parsing with Pydantic models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field, field_validator

from src.observability.logging import get_structured_logger

if TYPE_CHECKING:
    from .config import GeneratorConfig, LLMGuidedMCTSConfig, ReflectorConfig
    from .node import NodeState

logger = get_structured_logger(__name__)


# ============================================================================
# Structured Output Models
# ============================================================================


class CodeVariant(BaseModel):
    """A single code variant with confidence score."""

    code: str = Field(..., description="The generated code")
    confidence: float = Field(..., description="Confidence score between 0 and 1 (will be clamped)")
    reasoning: str = Field(default="", description="Brief explanation of the approach")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class GeneratorOutput(BaseModel):
    """Output from the Generator agent."""

    variants: list[CodeVariant] = Field(..., description="List of code variants with confidence scores")
    total_tokens: int = Field(default=0, description="Total tokens used")

    @property
    def action_probs(self) -> dict[str, float]:
        """Convert variants to action probability distribution."""
        if not self.variants:
            return {}

        # Normalize confidences to sum to 1
        total_conf = sum(v.confidence for v in self.variants)
        if total_conf == 0:
            # Equal probability if all confidences are 0
            n = len(self.variants)
            return {f"variant_{i}": 1.0 / n for i in range(n)}

        return {f"variant_{i}": v.confidence / total_conf for i, v in enumerate(self.variants)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "variants": [v.model_dump() for v in self.variants],
            "action_probs": self.action_probs,
            "total_tokens": self.total_tokens,
        }


class ReflectorOutput(BaseModel):
    """Output from the Reflector agent."""

    value: float = Field(..., description="Estimated probability of success (will be clamped to [0, 1])")
    reflection: str = Field(..., description="Analysis of the code")
    is_solution: bool = Field(default=False, description="Whether this code is a valid solution")
    issues: list[str] = Field(default_factory=list, description="Identified issues with the code")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions for improvement")
    total_tokens: int = Field(default=0, description="Total tokens used")

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "reflection": self.reflection,
            "is_solution": self.is_solution,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "total_tokens": self.total_tokens,
        }


# ============================================================================
# LLM Client Protocol
# ============================================================================


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate completion from prompt."""
        ...


# ============================================================================
# Base Agent Class
# ============================================================================


class BaseAgent(ABC):
    """Base class for LLM agents."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        logger_name: str | None = None,
    ):
        """
        Initialize agent.

        Args:
            llm_client: LLM client for API calls
            logger_name: Optional logger name
        """
        self._llm_client = llm_client
        self._logger = get_structured_logger(logger_name or __name__)
        self._total_tokens = 0
        self._total_calls = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used by this agent."""
        return self._total_tokens

    @property
    def total_calls(self) -> int:
        """Total API calls made by this agent."""
        return self._total_calls

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_tokens = 0
        self._total_calls = 0

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent."""
        pass


# ============================================================================
# Generator Agent
# ============================================================================


GENERATOR_SYSTEM_PROMPT = """You are an expert Python programmer. Your task is to generate multiple code variants to solve programming problems.

Given a problem description and optionally the current code with errors, generate {num_variants} different approaches to solve the problem.

For each variant, provide:
1. The complete Python code (not just modifications)
2. A confidence score (0.0 to 1.0) indicating how likely the code is to work
3. Brief reasoning explaining your approach

Output format (JSON):
{{
  "variants": [
    {{
      "code": "def solution(...):\\n    ...",
      "confidence": 0.8,
      "reasoning": "Using iterative approach for efficiency"
    }},
    ...
  ]
}}

Important:
- Each variant should be a COMPLETE, runnable solution
- Confidence scores should reflect actual likelihood of correctness
- Higher confidence = more likely to pass all tests
- Diversify your approaches (e.g., iterative vs recursive, different algorithms)
- If test errors are provided, focus on fixing those specific issues"""


class GeneratorAgent(BaseAgent):
    """
    Generator agent that produces code variants.

    Returns both code AND action probability distribution for training.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: GeneratorConfig | None = None,
    ):
        """
        Initialize generator agent.

        Args:
            llm_client: LLM client for API calls
            config: Generator configuration
        """
        super().__init__(llm_client, "mcts.agents.generator")
        self._config = config

    @property
    def config(self) -> GeneratorConfig:
        """Get configuration, creating default if needed."""
        if self._config is None:
            from .config import GeneratorConfig

            self._config = GeneratorConfig()
        return self._config

    def _build_prompt(self, state: NodeState) -> str:
        """Build the prompt for the LLM."""
        system = GENERATOR_SYSTEM_PROMPT.format(num_variants=self.config.num_variants)

        user_parts = [
            f"## Problem\n{state.problem}",
        ]

        if state.test_cases:
            tests_str = "\n".join(state.test_cases)
            user_parts.append(f"## Test Cases\n```python\n{tests_str}\n```")

        if state.code and state.code.strip():
            user_parts.append(f"## Current Code\n```python\n{state.code}\n```")

        if state.errors and self.config.include_test_errors:
            errors_str = "\n".join(state.errors)
            user_parts.append(f"## Errors from Previous Attempt\n{errors_str}")

        if state.attempt_history and self.config.include_previous_attempts:
            max_attempts = self.config.max_previous_attempts
            recent = state.attempt_history[-max_attempts:]
            for i, attempt in enumerate(recent):
                user_parts.append(f"## Previous Attempt {i + 1}\n```python\n{attempt}\n```")

        separator = "\n\n"
        user_section = separator.join(user_parts)
        return f"{system}\n\n{user_section}"

    def _parse_response(self, response: str) -> GeneratorOutput:
        """Parse LLM response into structured output."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                variants = []
                for v in data.get("variants", []):
                    variants.append(
                        CodeVariant(
                            code=v.get("code", ""),
                            confidence=float(v.get("confidence", 0.5)),
                            reasoning=v.get("reasoning", ""),
                        )
                    )
                return GeneratorOutput(variants=variants)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._logger.warning(
                "Failed to parse JSON response",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Fallback: try to extract code blocks
        code_blocks = re.findall(r"```python\n([\s\S]*?)```", response)
        if code_blocks:
            variants = [
                CodeVariant(
                    code=code.strip(),
                    confidence=0.5,
                    reasoning="Extracted from response",
                )
                for code in code_blocks[: self.config.num_variants]
            ]
            return GeneratorOutput(variants=variants)

        # Last resort: use entire response as code
        self._logger.warning("Could not parse response, using fallback")
        return GeneratorOutput(variants=[CodeVariant(code=response.strip(), confidence=0.3, reasoning="Fallback")])

    async def run(self, state: NodeState) -> GeneratorOutput:
        """
        Generate code variants for the given state.

        Args:
            state: Current node state with problem and code

        Returns:
            GeneratorOutput with variants and action probabilities
        """
        prompt = self._build_prompt(state)

        self._logger.debug(
            "Generating code variants",
            problem_length=len(state.problem),
            has_code=bool(state.code),
            num_errors=len(state.errors),
        )

        try:
            response = await self._llm_client.complete(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            self._total_calls += 1
            output = self._parse_response(response)

            self._logger.info(
                "Generated variants",
                num_variants=len(output.variants),
                action_probs=output.action_probs,
            )

            return output

        except Exception as e:
            self._logger.error(
                "Generator failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return empty output on failure
            return GeneratorOutput(variants=[])


# ============================================================================
# Reflector Agent
# ============================================================================


REFLECTOR_SYSTEM_PROMPT = """You are a code reviewer and evaluator. Analyze the given code and predict its likelihood of passing all tests.

Evaluate:
1. Correctness: Does the logic appear correct?
2. Edge cases: Are edge cases handled?
3. Test alignment: Does the code match the test expectations?
4. Code quality: Is the code well-structured?

Output format (JSON):
{{
  "value": 0.8,
  "reflection": "Brief analysis of the code's strengths and weaknesses",
  "is_solution": false,
  "issues": ["List of identified issues"],
  "suggestions": ["List of improvement suggestions"]
}}

The "value" should be a float between 0.0 and 1.0:
- 0.0-0.3: Code has significant issues, unlikely to work
- 0.3-0.6: Code has some issues, might partially work
- 0.6-0.8: Code looks mostly correct, likely to work
- 0.8-1.0: Code appears correct, high confidence in success

Set "is_solution" to true ONLY if the code clearly passes all tests."""


class ReflectorAgent(BaseAgent):
    """
    Reflector agent that evaluates code quality.

    Returns value estimate for training the value network.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: ReflectorConfig | None = None,
    ):
        """
        Initialize reflector agent.

        Args:
            llm_client: LLM client for API calls
            config: Reflector configuration
        """
        super().__init__(llm_client, "mcts.agents.reflector")
        self._config = config

    @property
    def config(self) -> ReflectorConfig:
        """Get configuration, creating default if needed."""
        if self._config is None:
            from .config import ReflectorConfig

            self._config = ReflectorConfig()
        return self._config

    def _build_prompt(
        self,
        state: NodeState,
        test_results: dict[str, Any] | None = None,
    ) -> str:
        """Build the prompt for the LLM."""
        parts = [
            REFLECTOR_SYSTEM_PROMPT,
            f"## Problem\n{state.problem}",
        ]

        if state.test_cases and self.config.include_code_context:
            tests_str = "\n".join(state.test_cases)
            parts.append(f"## Test Cases\n```python\n{tests_str}\n```")

        parts.append(f"## Code to Evaluate\n```python\n{state.code}\n```")

        if test_results and self.config.include_test_results:
            parts.append(f"## Test Execution Results\n{json.dumps(test_results, indent=2)}")

        if state.errors:
            errors_str = "\n".join(state.errors)
            parts.append(f"## Errors\n{errors_str}")

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> ReflectorOutput:
        """Parse LLM response into structured output."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                return ReflectorOutput(
                    value=float(data.get("value", 0.5)),
                    reflection=data.get("reflection", ""),
                    is_solution=bool(data.get("is_solution", False)),
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._logger.warning(
                "Failed to parse JSON response",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Fallback: try to extract value from text
        value = 0.5
        value_match = re.search(r"value[:\s]+([0-9.]+)", response.lower())
        if value_match:
            import contextlib

            with contextlib.suppress(ValueError):
                value = float(value_match.group(1))

        is_solution = "is_solution: true" in response.lower() or 'is_solution":true' in response.lower()

        return ReflectorOutput(
            value=value,
            reflection=response[:500],  # Use first 500 chars as reflection
            is_solution=is_solution,
        )

    async def run(
        self,
        state: NodeState,
        test_results: dict[str, Any] | None = None,
    ) -> ReflectorOutput:
        """
        Evaluate code and predict success probability.

        Args:
            state: Node state with code to evaluate
            test_results: Optional test execution results

        Returns:
            ReflectorOutput with value estimate and analysis
        """
        prompt = self._build_prompt(state, test_results)

        self._logger.debug(
            "Evaluating code",
            code_length=len(state.code),
            has_test_results=test_results is not None,
        )

        try:
            response = await self._llm_client.complete(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            self._total_calls += 1
            output = self._parse_response(response)

            self._logger.info(
                "Code evaluation complete",
                value=output.value,
                is_solution=output.is_solution,
                num_issues=len(output.issues),
            )

            return output

        except Exception as e:
            self._logger.error(
                "Reflector failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return neutral output on failure
            return ReflectorOutput(
                value=0.5,
                reflection=f"Evaluation failed: {e}",
                is_solution=False,
            )


# ============================================================================
# Factory Functions
# ============================================================================


def create_generator_from_config(
    llm_client: LLMClientProtocol,
    config: LLMGuidedMCTSConfig,
) -> GeneratorAgent:
    """Create a GeneratorAgent from configuration."""
    return GeneratorAgent(llm_client, config.generator_config)


def create_reflector_from_config(
    llm_client: LLMClientProtocol,
    config: LLMGuidedMCTSConfig,
) -> ReflectorAgent:
    """Create a ReflectorAgent from configuration."""
    return ReflectorAgent(llm_client, config.reflector_config)
