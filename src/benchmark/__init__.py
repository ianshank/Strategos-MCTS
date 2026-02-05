"""
Benchmark framework for comparing multi-agent systems.

Provides a structured evaluation harness for comparing LangGraph MCTS
against Google ADK Agent Builder across equivalent coordination tasks.

Public API:
    - BenchmarkSettings: Configuration for benchmark runs
    - BenchmarkFactory: Master factory for creating wired pipeline components
    - BenchmarkTask: Task definition model
    - BenchmarkResult: Execution result model
    - ScoringResult: LLM-as-judge scoring model
    - BenchmarkTaskRegistry: Data-driven task management
    - BenchmarkAdapterFactory: System adapter creation
    - EvaluationHarness: Orchestrates benchmark runs
    - ReportGenerator: Produces comparison reports
"""

from src.benchmark.adapters.factory import BenchmarkAdapterFactory
from src.benchmark.config.benchmark_settings import BenchmarkSettings, get_benchmark_settings
from src.benchmark.evaluation.harness import EvaluationHarness
from src.benchmark.evaluation.models import BenchmarkResult, ScoringResult
from src.benchmark.factory import BenchmarkFactory
from src.benchmark.reporting.report_generator import ReportGenerator
from src.benchmark.tasks.models import BenchmarkTask, TaskCategory, TaskComplexity
from src.benchmark.tasks.registry import BenchmarkTaskRegistry

__all__ = [
    "BenchmarkAdapterFactory",
    "BenchmarkFactory",
    "BenchmarkResult",
    "BenchmarkSettings",
    "BenchmarkTask",
    "BenchmarkTaskRegistry",
    "EvaluationHarness",
    "ReportGenerator",
    "ScoringResult",
    "TaskCategory",
    "TaskComplexity",
    "get_benchmark_settings",
]
