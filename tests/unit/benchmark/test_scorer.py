"""
Tests for the LLM-as-judge scorer.

Validates scoring prompt construction, response parsing,
retry logic, and edge case handling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.config.benchmark_settings import ScoringConfig
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.evaluation.scorer import LLMJudgeScorer
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory, TaskComplexity


@dataclass
class MockLLMResponse:
    text: str
    usage: dict


def _make_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="T1",
        category=TaskCategory.QE,
        description="Test task",
        input_data="Test input data",
        expected_outputs=("output1", "output2"),
        complexity=TaskComplexity.MEDIUM,
    )


def _make_result(response: str = "Good analysis") -> BenchmarkResult:
    return BenchmarkResult(
        task_id="T1",
        system="test_system",
        task_description="Test task",
        raw_response=response,
    )


def _make_llm_response(scores: dict) -> MockLLMResponse:
    return MockLLMResponse(
        text=json.dumps(scores),
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )


@pytest.mark.unit
class TestLLMJudgeScorer:
    """Test LLM-as-judge scoring."""

    def test_init(self) -> None:
        config = ScoringConfig()
        scorer = LLMJudgeScorer(config=config)
        assert scorer._config is config

    @pytest.mark.asyncio
    async def test_score_disabled(self) -> None:
        config = ScoringConfig(enabled=False)
        scorer = LLMJudgeScorer(config=config)
        result = await scorer.score(_make_result(), _make_task())
        assert result.task_completion == 0.0

    @pytest.mark.asyncio
    async def test_score_errored_result(self) -> None:
        config = ScoringConfig()
        scorer = LLMJudgeScorer(config=config)
        result = _make_result("Error: TimeoutError: timed out")
        scoring = await scorer.score(result, _make_task())
        assert "error" in scoring.judge_explanation.lower()

    @pytest.mark.asyncio
    async def test_score_success(self) -> None:
        config = ScoringConfig()
        mock_client = AsyncMock()
        mock_client.generate.return_value = _make_llm_response(
            {
                "task_completion": 4.0,
                "reasoning_depth": 3.5,
                "accuracy": 4.5,
                "coherence": 4.0,
                "delegation_appropriateness": 3.0,
                "explanation": "Good analysis",
            }
        )

        scorer = LLMJudgeScorer(config=config, llm_client=mock_client)
        scoring = await scorer.score(_make_result(), _make_task())

        assert scoring.task_completion == 4.0
        assert scoring.reasoning_depth == 3.5
        assert scoring.accuracy == 4.5
        assert scoring.coherence == 4.0
        assert scoring.judge_model == config.model

    @pytest.mark.asyncio
    async def test_score_clamps_values(self) -> None:
        config = ScoringConfig(min_score=1.0, max_score=5.0)
        mock_client = AsyncMock()
        mock_client.generate.return_value = _make_llm_response(
            {
                "task_completion": 10.0,  # Above max
                "reasoning_depth": -1.0,  # Below min
                "accuracy": 3.0,
                "coherence": 3.0,
                "explanation": "test",
            }
        )

        scorer = LLMJudgeScorer(config=config, llm_client=mock_client)
        scoring = await scorer.score(_make_result(), _make_task())

        assert scoring.task_completion == 5.0  # Clamped to max
        assert scoring.reasoning_depth == 1.0  # Clamped to min

    @pytest.mark.asyncio
    async def test_score_handles_markdown_json(self) -> None:
        """Test that scorer handles JSON wrapped in markdown code fences."""
        config = ScoringConfig()
        mock_client = AsyncMock()

        # Response wrapped in markdown code fences
        json_response = json.dumps(
            {
                "task_completion": 4.0,
                "reasoning_depth": 3.0,
                "accuracy": 4.0,
                "coherence": 3.5,
                "explanation": "Adequate",
            }
        )
        mock_client.generate.return_value = MockLLMResponse(
            text=f"```json\n{json_response}\n```",
            usage={},
        )

        scorer = LLMJudgeScorer(config=config, llm_client=mock_client)
        scoring = await scorer.score(_make_result(), _make_task())
        assert scoring.task_completion == 4.0

    @pytest.mark.asyncio
    async def test_score_retry_on_parse_error(self) -> None:
        config = ScoringConfig(max_retries=2, retry_backoff_base_seconds=0.01)
        mock_client = AsyncMock()

        # First two calls return invalid JSON, third succeeds
        mock_client.generate.side_effect = [
            MockLLMResponse(text="not json", usage={}),
            MockLLMResponse(text="still not json", usage={}),
            _make_llm_response(
                {
                    "task_completion": 4.0,
                    "reasoning_depth": 3.0,
                    "accuracy": 4.0,
                    "coherence": 3.0,
                    "explanation": "ok",
                }
            ),
        ]

        scorer = LLMJudgeScorer(config=config, llm_client=mock_client)
        scoring = await scorer.score(_make_result(), _make_task())
        assert scoring.task_completion == 4.0
        assert mock_client.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_score_all_retries_fail(self) -> None:
        config = ScoringConfig(max_retries=1, retry_backoff_base_seconds=0.01)
        mock_client = AsyncMock()
        mock_client.generate.return_value = MockLLMResponse(text="bad", usage={})

        scorer = LLMJudgeScorer(config=config, llm_client=mock_client)
        scoring = await scorer.score(_make_result(), _make_task())
        assert "failed" in scoring.judge_explanation.lower()

    @pytest.mark.asyncio
    async def test_score_no_llm_client(self) -> None:
        config = ScoringConfig(max_retries=0)
        scorer = LLMJudgeScorer(config=config, llm_client=None)
        scoring = await scorer.score(_make_result(), _make_task())
        assert scoring.task_completion == 0.0

    @pytest.mark.asyncio
    async def test_score_batch(self) -> None:
        config = ScoringConfig()
        mock_client = AsyncMock()
        mock_client.generate.return_value = _make_llm_response(
            {
                "task_completion": 4.0,
                "reasoning_depth": 3.0,
                "accuracy": 4.0,
                "coherence": 3.0,
                "explanation": "ok",
            }
        )

        scorer = LLMJudgeScorer(config=config, llm_client=mock_client)
        pairs = [(_make_result(), _make_task()) for _ in range(3)]
        scorings = await scorer.score_batch(pairs)

        assert len(scorings) == 3
        assert all(s.task_completion == 4.0 for s in scorings)

    def test_build_scoring_prompt(self) -> None:
        config = ScoringConfig()
        scorer = LLMJudgeScorer(config=config)
        prompt = scorer._build_scoring_prompt(_make_result(), _make_task())

        assert "Test task" in prompt
        assert "Test input data" in prompt
        assert "Good analysis" in prompt
        assert "output1" in prompt

    def test_parse_scoring_response_valid(self) -> None:
        config = ScoringConfig()
        scorer = LLMJudgeScorer(config=config)
        response = json.dumps(
            {
                "task_completion": 4.0,
                "reasoning_depth": 3.0,
                "accuracy": 4.5,
                "coherence": 4.0,
                "delegation_appropriateness": 3.5,
                "explanation": "Good work",
            }
        )
        scoring = scorer._parse_scoring_response(response)
        assert scoring.task_completion == 4.0
        assert scoring.judge_explanation == "Good work"

    def test_parse_scoring_response_invalid_json(self) -> None:
        config = ScoringConfig()
        scorer = LLMJudgeScorer(config=config)
        with pytest.raises(json.JSONDecodeError):
            scorer._parse_scoring_response("not json at all")
