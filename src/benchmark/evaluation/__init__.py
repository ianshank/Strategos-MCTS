"""Benchmark evaluation engine."""

from src.benchmark.evaluation.cost_calculator import CostCalculator
from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult
from src.benchmark.evaluation.scorer import LLMJudgeScorer

# EvaluationHarness is not imported here to avoid circular imports
# (harness -> adapters.protocol -> evaluation.models -> this __init__ -> harness).
# Import directly from src.benchmark.evaluation.harness when needed.

__all__ = [
    "BenchmarkResult",
    "CostCalculator",
    "LLMJudgeScorer",
    "ScoringResult",
]
