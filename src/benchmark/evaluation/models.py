"""
Benchmark evaluation data models.

Defines BenchmarkResult and ScoringResult for structured
collection and analysis of benchmark outcomes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ScoringResult:
    """
    LLM-as-judge scoring result for a single benchmark result.

    Scores are on a configurable scale (default 1-5).
    """

    task_completion: float = 0.0
    reasoning_depth: float = 0.0
    accuracy: float = 0.0
    coherence: float = 0.0
    delegation_appropriateness: float = 0.0
    judge_model: str = ""
    judge_explanation: str = ""
    scored_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def average_score(self) -> float:
        """Calculate average across all quality dimensions."""
        scores = [
            self.task_completion,
            self.reasoning_depth,
            self.accuracy,
            self.coherence,
        ]
        non_zero = [s for s in scores if s > 0]
        return sum(non_zero) / len(non_zero) if non_zero else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringResult:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class BenchmarkResult:
    """
    Complete result for a single benchmark task execution.

    Captures timing, token usage, quality scores, and response data.
    """

    task_id: str = ""
    system: str = ""
    task_description: str = ""

    # Timing
    total_latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0

    # Quality scores (populated by scorer)
    scoring: ScoringResult = field(default_factory=ScoringResult)

    # Agent coordination metrics
    num_agent_calls: int = 0
    num_tool_calls: int = 0

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Response data
    raw_response: str = ""
    agent_trace: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    run_id: str = ""
    iteration: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_error(self) -> bool:
        return self.raw_response.startswith("Error:")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, excluding large trace data by default."""
        data = asdict(self)
        # Truncate large fields for summary views
        if len(data.get("raw_response", "")) > 1000:
            data["raw_response_preview"] = data["raw_response"][:1000] + "..."
        data.pop("agent_trace", None)
        return data

    def to_full_dict(self) -> dict[str, Any]:
        """Serialize to dictionary including all data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Deserialize from dictionary."""
        scoring_data = data.pop("scoring", {})
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        result = cls(**filtered)
        if scoring_data:
            result.scoring = ScoringResult.from_dict(scoring_data)
        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
