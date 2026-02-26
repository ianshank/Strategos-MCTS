"""
LLM-backed Task Refinement Module (TRM) agent.

Replaces the PyTorch-based TRM with an LLM-powered agent that iteratively
refines an answer by generating an initial response, critiquing it, and
producing a strengthened final answer.

Implements the AsyncAgentBase interface so it can be used as a drop-in
replacement in the LangGraph graph nodes.
"""

from __future__ import annotations

import logging
from typing import Any

from src.adapters.llm.base import LLMClient
from src.framework.agents.base import AgentContext, AgentResult, AsyncAgentBase

logger = logging.getLogger(__name__)

# Configurable defaults (no hardcoded magic numbers)
DEFAULT_TRM_TEMPERATURE = 0.5
DEFAULT_TRM_MAX_TOKENS = 1500
QUALITY_BASELINE = 0.5
QUALITY_INITIAL_BONUS = 0.1
QUALITY_REVIEW_BONUS = 0.2
QUALITY_REFINED_BONUS = 0.15
QUALITY_LENGTH_BONUS = 0.05
QUALITY_LENGTH_THRESHOLD = 300

REFINEMENT_PROMPT = """\
You are a Task Refinement Module (TRM). Your job is to produce a high-quality
answer through iterative self-refinement.

## Instructions
1. Write an initial answer to the query
2. Critically review your answer for gaps, errors, and unclear points
3. Produce a refined, strengthened final answer

## Query
{query}

{rag_section}

## Response Format
### Initial Answer
<your first-pass answer>

### Critical Review
- <weakness or gap 1>
- <weakness or gap 2>

### Refined Answer
<improved final answer addressing the weaknesses>
"""


class LLMTRMAgent(AsyncAgentBase):
    """
    LLM-powered Task Refinement Module.

    Generates an initial answer, self-critiques, and refines it into a
    stronger response. Compatible with LangGraph graph nodes via the
    AsyncAgentBase.process() interface.
    """

    def __init__(
        self,
        model_adapter: LLMClient,
        logger: Any = None,
        name: str = "LLM_TRM",
        temperature: float = DEFAULT_TRM_TEMPERATURE,
        max_tokens: int = DEFAULT_TRM_MAX_TOKENS,
        **config: Any,
    ):
        super().__init__(model_adapter, logger, name, **config)
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Generate, critique, and refine an answer."""
        query = context.query
        self.logger.info("TRM processing query: %s", query[:100])

        # Build RAG section if available
        rag_section = ""
        if context.rag_context:
            rag_section = f"## Additional Context\n{context.rag_context}\n"

        prompt = REFINEMENT_PROMPT.format(query=query, rag_section=rag_section)

        response = await self.generate_llm_response(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        text = response.text
        quality_score = self._compute_quality_score(text)

        has_initial = "initial" in text.lower()
        has_review = "review" in text.lower() or "weakness" in text.lower()
        has_refined = "refined" in text.lower() or "improved" in text.lower()

        self.logger.debug(
            "TRM result: quality=%.3f, has_initial=%s, has_review=%s, has_refined=%s",
            quality_score,
            has_initial,
            has_review,
            has_refined,
        )

        return AgentResult(
            response=text,
            confidence=quality_score,
            metadata={
                "final_quality_score": quality_score,
                "has_initial_answer": has_initial,
                "has_critical_review": has_review,
                "has_refined_answer": has_refined,
                "strategy": "iterative_refinement",
            },
            token_usage=response.usage,
        )

    @staticmethod
    def _compute_quality_score(text: str) -> float:
        """Compute a quality score from response structure."""
        score = QUALITY_BASELINE
        text_lower = text.lower()
        if "initial" in text_lower:
            score += QUALITY_INITIAL_BONUS
        if "review" in text_lower or "weakness" in text_lower:
            score += QUALITY_REVIEW_BONUS
        if "refined" in text_lower or "improved" in text_lower:
            score += QUALITY_REFINED_BONUS
        if len(text) > QUALITY_LENGTH_THRESHOLD:
            score += QUALITY_LENGTH_BONUS
        return min(score, 1.0)
