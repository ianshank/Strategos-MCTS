"""
Benchmark system protocol definition.

Defines the contract that all benchmark system adapters must implement.
Uses Protocol (structural subtyping) consistent with codebase patterns.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.tasks.models import BenchmarkTask


@runtime_checkable
class BenchmarkSystemProtocol(Protocol):
    """
    Protocol for benchmark system adapters.

    Every system under benchmark (LangGraph MCTS, Google ADK, etc.)
    must implement this interface for uniform evaluation.
    """

    @property
    def name(self) -> str:
        """Unique system identifier (e.g., 'langgraph_mcts', 'vertex_adk')."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if the system is available for benchmarking (deps installed, keys set)."""
        ...

    async def execute(self, task: BenchmarkTask) -> BenchmarkResult:
        """
        Execute a benchmark task and return structured results.

        Args:
            task: The benchmark task to execute

        Returns:
            BenchmarkResult with timing, token usage, and response data

        Raises:
            RuntimeError: If system is not available
            TimeoutError: If execution exceeds configured timeout
        """
        ...

    async def health_check(self) -> bool:
        """
        Verify system is operational.

        Returns:
            True if system can accept and process tasks
        """
        ...
