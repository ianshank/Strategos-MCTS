"""
Tests for benchmark evaluation data models.

Validates BenchmarkResult and ScoringResult creation,
serialization, and computed properties.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult


@pytest.mark.unit
class TestScoringResult:
    """Test ScoringResult dataclass."""

    def test_default_creation(self) -> None:
        scoring = ScoringResult()
        assert scoring.task_completion == 0.0
        assert scoring.reasoning_depth == 0.0
        assert scoring.accuracy == 0.0
        assert scoring.coherence == 0.0
        assert scoring.judge_model == ""

    def test_average_score(self) -> None:
        scoring = ScoringResult(
            task_completion=4.0,
            reasoning_depth=3.0,
            accuracy=5.0,
            coherence=4.0,
        )
        assert scoring.average_score == pytest.approx(4.0)

    def test_average_score_with_zeros(self) -> None:
        scoring = ScoringResult(
            task_completion=4.0,
            reasoning_depth=0.0,
            accuracy=2.0,
            coherence=0.0,
        )
        # Averages all dimensions including zeros: (4+0+2+0)/4 = 1.5
        assert scoring.average_score == pytest.approx(1.5)

    def test_average_score_all_zero(self) -> None:
        scoring = ScoringResult()
        assert scoring.average_score == 0.0

    def test_to_dict(self) -> None:
        scoring = ScoringResult(task_completion=4.5, judge_model="gpt-4")
        data = scoring.to_dict()
        assert data["task_completion"] == 4.5
        assert data["judge_model"] == "gpt-4"

    def test_from_dict(self) -> None:
        data = {"task_completion": 4.0, "accuracy": 3.5, "judge_model": "gpt-4"}
        scoring = ScoringResult.from_dict(data)
        assert scoring.task_completion == 4.0
        assert scoring.accuracy == 3.5

    def test_from_dict_ignores_extra_fields(self) -> None:
        data = {"task_completion": 4.0, "unknown_field": "ignored"}
        scoring = ScoringResult.from_dict(data)
        assert scoring.task_completion == 4.0


@pytest.mark.unit
class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_default_creation(self) -> None:
        result = BenchmarkResult()
        assert result.task_id == ""
        assert result.system == ""
        assert result.total_latency_ms == 0.0
        assert result.num_agent_calls == 0
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.raw_response == ""
        assert isinstance(result.scoring, ScoringResult)

    def test_create_with_values(self) -> None:
        result = BenchmarkResult(
            task_id="A1",
            system="langgraph_mcts",
            task_description="Code review",
            total_latency_ms=1500.0,
            input_tokens=1000,
            output_tokens=500,
            raw_response="Found 3 bugs",
        )
        assert result.task_id == "A1"
        assert result.system == "langgraph_mcts"
        assert result.total_latency_ms == 1500.0

    def test_total_tokens(self) -> None:
        result = BenchmarkResult(input_tokens=1000, output_tokens=500)
        assert result.total_tokens == 1500

    def test_has_error_true(self) -> None:
        result = BenchmarkResult(raw_response="Error: TimeoutError: timed out")
        assert result.has_error is True

    def test_has_error_false(self) -> None:
        result = BenchmarkResult(raw_response="Found 3 bugs in the code")
        assert result.has_error is False

    def test_to_dict(self) -> None:
        result = BenchmarkResult(
            task_id="A1",
            system="test",
            raw_response="short response",
            agent_trace=[{"event": "test"}],
        )
        data = result.to_dict()
        assert data["task_id"] == "A1"
        assert "agent_trace" not in data  # Excluded by default

    def test_to_full_dict_includes_trace(self) -> None:
        result = BenchmarkResult(
            task_id="A1",
            agent_trace=[{"event": "test"}],
        )
        data = result.to_full_dict()
        assert "agent_trace" in data
        assert len(data["agent_trace"]) == 1

    def test_to_dict_truncates_long_response(self) -> None:
        result = BenchmarkResult(raw_response="x" * 2000)
        data = result.to_dict()
        assert "raw_response_preview" in data
        assert len(data["raw_response_preview"]) < 2000

    def test_from_dict(self) -> None:
        data = {
            "task_id": "B1",
            "system": "vertex_adk",
            "total_latency_ms": 2000.0,
            "scoring": {"task_completion": 4.0, "accuracy": 3.5},
        }
        result = BenchmarkResult.from_dict(data)
        assert result.task_id == "B1"
        assert result.system == "vertex_adk"
        assert result.scoring.task_completion == 4.0

    def test_from_dict_does_not_mutate_input(self) -> None:
        data = {
            "task_id": "B1",
            "system": "vertex_adk",
            "scoring": {"task_completion": 4.0},
        }
        original_data = dict(data)
        BenchmarkResult.from_dict(data)
        # Input dict should not be mutated
        assert data == original_data

    def test_to_json(self) -> None:
        result = BenchmarkResult(task_id="A1", system="test")
        json_str = result.to_json()
        assert '"task_id": "A1"' in json_str
        assert '"system": "test"' in json_str

    def test_scoring_embedded(self) -> None:
        scoring = ScoringResult(task_completion=5.0, accuracy=4.0)
        result = BenchmarkResult(task_id="A1", scoring=scoring)
        assert result.scoring.task_completion == 5.0
        assert result.scoring.average_score > 0
