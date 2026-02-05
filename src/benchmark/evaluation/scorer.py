"""
LLM-as-judge scorer for benchmark evaluation.

Uses a configurable LLM to score benchmark results on multiple
quality dimensions, avoiding manual scoring bias.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from src.benchmark.config.benchmark_settings import ScoringConfig
from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult
from src.benchmark.tasks.models import BenchmarkTask
from src.observability.logging import get_correlation_id

logger = logging.getLogger(__name__)


class ScorerProtocol(Protocol):
    """Protocol for benchmark result scorers."""

    async def score(self, result: BenchmarkResult, task: BenchmarkTask) -> ScoringResult:
        """Score a benchmark result against its task."""
        ...


SCORING_PROMPT_TEMPLATE = """You are evaluating an AI multi-agent system's response.

Task: {task_description}
Task Category: {task_category}
Task Complexity: {task_complexity}

Input given to the system:
{task_input}

Expected elements in a good response:
{expected_outputs}

System response:
{system_response}

Score each dimension on a scale of {min_score} to {max_score} ({max_score} = excellent):
1. task_completion: Did it address all aspects of the task?
2. reasoning_depth: How thorough and multi-layered is the reasoning?
3. accuracy: Are the claims and recommendations correct?
4. coherence: Is the response well-organized and consistent?
5. delegation_appropriateness: If multi-agent, were subtasks delegated well?

Respond in JSON only (no markdown, no explanation outside JSON):
{{"task_completion": N, "reasoning_depth": N, "accuracy": N, "coherence": N, "delegation_appropriateness": N, "explanation": "brief justification"}}"""


class LLMJudgeScorer:
    """
    LLM-as-judge scorer using configurable LLM provider.

    Sends structured prompts to an LLM and parses scoring responses.
    Supports retry with exponential backoff on failures.

    Example:
        >>> scorer = LLMJudgeScorer(config=scoring_config, llm_client=client)
        >>> scoring = await scorer.score(result, task)
        >>> print(scoring.average_score)
    """

    def __init__(
        self,
        config: ScoringConfig,
        llm_client: Any | None = None,
    ) -> None:
        """
        Initialize the scorer.

        Args:
            config: Scoring configuration
            llm_client: LLM client for scoring (Protocol-compatible)
        """
        self._config = config
        self._llm_client = llm_client
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def score(self, result: BenchmarkResult, task: BenchmarkTask) -> ScoringResult:
        """
        Score a benchmark result using LLM-as-judge.

        Args:
            result: The benchmark result to score
            task: The original task definition

        Returns:
            ScoringResult with scores and explanation
        """
        if not self._config.enabled:
            self._logger.debug("Scoring disabled, returning empty result")
            return ScoringResult()

        if result.has_error:
            self._logger.warning("Skipping scoring for errored result: %s", result.task_id)
            return ScoringResult(judge_explanation="Skipped: result contains error")

        correlation_id = get_correlation_id()
        self._logger.info(
            "Scoring result for task %s (system: %s)",
            result.task_id,
            result.system,
            extra={"correlation_id": correlation_id},
        )

        prompt = self._build_scoring_prompt(result, task)

        for attempt in range(self._config.max_retries + 1):
            try:
                scoring = await self._call_judge(prompt)
                scoring.judge_model = self._config.model
                return scoring
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._logger.warning(
                    "Scoring attempt %d failed (parse error): %s",
                    attempt + 1,
                    e,
                )
                if attempt < self._config.max_retries:
                    import asyncio

                    backoff = self._config.retry_backoff_base_seconds * (2**attempt)
                    await asyncio.sleep(backoff)
            except Exception as e:
                self._logger.error("Scoring failed with unexpected error: %s", e, exc_info=True)
                break

        return ScoringResult(judge_explanation=f"Scoring failed after {self._config.max_retries + 1} attempts")

    async def score_batch(
        self,
        results: list[tuple[BenchmarkResult, BenchmarkTask]],
    ) -> list[ScoringResult]:
        """
        Score multiple results sequentially.

        Args:
            results: List of (result, task) tuples

        Returns:
            List of ScoringResult instances
        """
        scorings: list[ScoringResult] = []
        for result, task in results:
            scoring = await self.score(result, task)
            scorings.append(scoring)
        return scorings

    def _build_scoring_prompt(self, result: BenchmarkResult, task: BenchmarkTask) -> str:
        """Build the scoring prompt from template and data."""
        return SCORING_PROMPT_TEMPLATE.format(
            task_description=task.description,
            task_category=task.category_label,
            task_complexity=task.complexity.value,
            task_input=task.input_data[:2000],  # Truncate very long inputs
            expected_outputs=json.dumps(list(task.expected_outputs)),
            system_response=result.raw_response[:3000],  # Truncate very long responses
            min_score=self._config.min_score,
            max_score=self._config.max_score,
        )

    async def _call_judge(self, prompt: str) -> ScoringResult:
        """
        Call the LLM judge and parse the response.

        Args:
            prompt: The scoring prompt

        Returns:
            Parsed ScoringResult

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            ValueError: If scores are out of range
        """
        if self._llm_client is None:
            raise RuntimeError("LLM client not configured for scoring")

        response = await self._llm_client.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        return self._parse_scoring_response(response.text)

    def _parse_scoring_response(self, response_text: str) -> ScoringResult:
        """
        Parse LLM judge response into ScoringResult.

        Args:
            response_text: Raw JSON response from judge

        Returns:
            Validated ScoringResult

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            ValueError: If scores are out of range
        """
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        scores = json.loads(text)

        # Validate score ranges
        min_s = self._config.min_score
        max_s = self._config.max_score

        def clamp(value: Any) -> float:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return min_s
            return max(min_s, min(max_s, v))

        return ScoringResult(
            task_completion=clamp(scores.get("task_completion", 0)),
            reasoning_depth=clamp(scores.get("reasoning_depth", 0)),
            accuracy=clamp(scores.get("accuracy", 0)),
            coherence=clamp(scores.get("coherence", 0)),
            delegation_appropriateness=clamp(scores.get("delegation_appropriateness", 0)),
            judge_explanation=str(scores.get("explanation", "")),
        )
