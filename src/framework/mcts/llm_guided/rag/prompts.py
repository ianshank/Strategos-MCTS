"""
RAG-Enhanced Prompts for Code Generation.

Provides prompt builders that incorporate retrieved context
into generator and reflector agent prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import RAGContext


@dataclass
class RAGPromptBuilder:
    """
    Builder for RAG-enhanced prompts.

    Combines problem description, current code state, and retrieved
    context into prompts for LLM agents.
    """

    # Context limits
    max_context_length: int = 4000
    """Maximum length of RAG context in characters."""

    max_examples: int = 3
    """Maximum number of examples to include."""

    include_patterns: bool = True
    """Include code patterns in context."""

    include_docs: bool = True
    """Include API documentation."""

    def build_generator_prompt(
        self,
        problem: str,
        current_code: str | None,
        rag_context: RAGContext | None,
        num_variants: int = 3,
        iteration: int = 0,
        feedback: str | None = None,
    ) -> str:
        """
        Build prompt for code generation with RAG context.

        Args:
            problem: Problem description
            current_code: Current code state
            rag_context: Retrieved context
            num_variants: Number of variants to generate
            iteration: Current MCTS iteration
            feedback: Feedback from previous attempts

        Returns:
            Complete prompt string
        """
        sections = []

        # System context
        sections.append(self._system_section())

        # RAG context
        if rag_context and not rag_context.is_empty():
            sections.append(self._rag_context_section(rag_context))

        # Problem
        sections.append(f"## Problem\n\n{problem}")

        # Current code
        if current_code:
            sections.append(f"## Current Code\n\n```python\n{current_code}\n```")

        # Feedback
        if feedback:
            sections.append(f"## Feedback from Previous Attempts\n\n{feedback}")

        # Instructions
        sections.append(self._generator_instructions(num_variants, iteration))

        return "\n\n".join(sections)

    def build_reflector_prompt(
        self,
        problem: str,
        code: str,
        test_results: dict[str, Any] | None,
        rag_context: RAGContext | None,
    ) -> str:
        """
        Build prompt for code evaluation with RAG context.

        Args:
            problem: Problem description
            code: Code to evaluate
            test_results: Results from test execution
            rag_context: Retrieved context

        Returns:
            Complete prompt string
        """
        sections = []

        # System context
        sections.append(
            """You are an expert code reviewer evaluating Python solutions.
Your task is to analyze the code's correctness, efficiency, and style,
then estimate its probability of passing all test cases."""
        )

        # RAG context (brief)
        if rag_context and not rag_context.is_empty():
            sections.append(self._rag_context_section(rag_context, brief=True))

        # Problem
        sections.append(f"## Problem\n\n{problem}")

        # Code to evaluate
        sections.append(f"## Solution to Evaluate\n\n```python\n{code}\n```")

        # Test results
        if test_results:
            sections.append(self._test_results_section(test_results))

        # Instructions
        sections.append(self._reflector_instructions())

        return "\n\n".join(sections)

    def _system_section(self) -> str:
        """Build system context section."""
        return """You are an expert Python programmer participating in a code generation task.
Your goal is to generate correct, efficient, and readable solutions.
Use the provided context and examples to guide your implementation."""

    def _rag_context_section(self, context: RAGContext, brief: bool = False) -> str:
        """Build RAG context section."""
        sections = ["## Relevant Context"]

        if context.similar_solutions:
            sections.append("### Similar Solutions")
            limit = 1 if brief else self.max_examples
            for i, sol in enumerate(context.similar_solutions[:limit], 1):
                code = sol.get("code", "")
                desc = sol.get("description", "")
                if brief:
                    # Truncate code for brief mode
                    code_lines = code.split("\n")[:10]
                    code = "\n".join(code_lines)
                    if len(code.split("\n")) < len(sol.get("code", "").split("\n")):
                        code += "\n# ..."

                sections.append(f"**Example {i}:**")
                if desc:
                    sections.append(f"*{desc}*")
                sections.append(f"```python\n{code}\n```")

        if self.include_patterns and context.code_patterns and not brief:
            sections.append("### Useful Patterns")
            for pattern in context.code_patterns[: self.max_examples]:
                name = pattern.get("name", "Pattern")
                code = pattern.get("code", "")
                sections.append(f"**{name}:**")
                sections.append(f"```python\n{code}\n```")

        if self.include_docs and context.api_docs and not brief:
            sections.append("### API Reference")
            for doc in context.api_docs[:2]:
                name = doc.get("name", "API")
                content = doc.get("content", "")
                # Truncate long docs
                if len(content) > 500:
                    content = content[:500] + "..."
                sections.append(f"**{name}:** {content}")

        result = "\n\n".join(sections)

        # Enforce max length
        if len(result) > self.max_context_length:
            result = result[: self.max_context_length] + "\n\n[Context truncated...]"

        return result

    def _test_results_section(self, results: dict[str, Any]) -> str:
        """Build test results section."""
        sections = ["## Test Results"]

        if results.get("passed"):
            sections.append("**Status:** All tests passed!")
        else:
            sections.append(f"**Status:** {results.get('num_passed', 0)}/{results.get('num_total', 0)} tests passed")

            if results.get("errors"):
                sections.append("\n**Errors:**")
                for error in results["errors"][:3]:
                    sections.append(f"- {error}")

            if results.get("stdout"):
                sections.append(f"\n**Output:**\n```\n{results['stdout'][:500]}\n```")

        return "\n".join(sections)

    def _generator_instructions(self, num_variants: int, iteration: int) -> str:
        """Build generator instructions."""
        return f"""## Instructions

Generate {num_variants} different solution variants for the problem above.
{"This is iteration " + str(iteration) + " of the search. Consider improvements based on any feedback." if iteration > 0 else ""}

For each variant, provide:
1. A complete, working Python implementation
2. A confidence score (0.0-1.0) indicating your belief it will pass all tests
3. A brief explanation of the approach

Respond in JSON format:
```json
{{
  "variants": [
    {{
      "code": "def solution(...): ...",
      "confidence": 0.85,
      "explanation": "Brief explanation of the approach"
    }},
    ...
  ]
}}
```

Focus on:
- Correctness: Handle all edge cases
- Efficiency: Choose appropriate algorithms
- Clarity: Write readable, well-structured code"""

    def _reflector_instructions(self) -> str:
        """Build reflector instructions."""
        return """## Instructions

Analyze the solution above and provide:
1. **Correctness Assessment**: Will it handle all test cases?
2. **Edge Cases**: What edge cases might fail?
3. **Efficiency**: Is the time/space complexity appropriate?
4. **Code Quality**: Is it readable and well-structured?
5. **Value Estimate**: Probability of passing all tests (0.0-1.0)

Respond in JSON format:
```json
{
  "correctness": "Assessment of correctness",
  "edge_cases": ["List of potential edge case issues"],
  "efficiency": "Time and space complexity analysis",
  "code_quality": "Assessment of code quality",
  "value_estimate": 0.75,
  "suggestions": ["List of specific improvement suggestions"]
}
```"""


def build_generator_prompt_with_rag(
    problem: str,
    current_code: str | None = None,
    rag_context: RAGContext | None = None,
    num_variants: int = 3,
    iteration: int = 0,
    feedback: str | None = None,
) -> str:
    """
    Build a generator prompt with RAG context.

    Convenience function that uses default RAGPromptBuilder settings.

    Args:
        problem: Problem description
        current_code: Current code state
        rag_context: Retrieved context
        num_variants: Number of variants to generate
        iteration: Current MCTS iteration
        feedback: Feedback from previous attempts

    Returns:
        Complete prompt string
    """
    builder = RAGPromptBuilder()
    return builder.build_generator_prompt(
        problem=problem,
        current_code=current_code,
        rag_context=rag_context,
        num_variants=num_variants,
        iteration=iteration,
        feedback=feedback,
    )


def build_reflector_prompt_with_rag(
    problem: str,
    code: str,
    test_results: dict[str, Any] | None = None,
    rag_context: RAGContext | None = None,
) -> str:
    """
    Build a reflector prompt with RAG context.

    Convenience function that uses default RAGPromptBuilder settings.

    Args:
        problem: Problem description
        code: Code to evaluate
        test_results: Results from test execution
        rag_context: Retrieved context

    Returns:
        Complete prompt string
    """
    builder = RAGPromptBuilder()
    return builder.build_reflector_prompt(
        problem=problem,
        code=code,
        test_results=test_results,
        rag_context=rag_context,
    )
