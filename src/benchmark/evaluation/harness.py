"""
Evaluation harness for orchestrating benchmark runs.

Coordinates task execution across multiple systems, scoring,
cost calculation, and result aggregation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from src.benchmark.adapters.protocol import BenchmarkSystemProtocol
from src.benchmark.config.benchmark_settings import BenchmarkSettings, get_benchmark_settings
from src.benchmark.evaluation.cost_calculator import CostCalculator
from src.benchmark.evaluation.models import BenchmarkResult
from src.benchmark.evaluation.scorer import ScorerProtocol
from src.benchmark.tasks.models import BenchmarkTask
from src.benchmark.tasks.registry import BenchmarkTaskRegistry
from src.observability.logging import set_correlation_id

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """
    Orchestrates benchmark evaluation across multiple systems and tasks.

    Manages the full lifecycle: task execution, scoring, cost calculation,
    and result persistence.

    Example:
        >>> harness = EvaluationHarness(
        ...     adapters=[lg_adapter, adk_adapter],
        ...     registry=registry,
        ...     scorer=scorer,
        ... )
        >>> results = await harness.run()
    """

    def __init__(
        self,
        adapters: list[BenchmarkSystemProtocol],
        registry: BenchmarkTaskRegistry,
        scorer: ScorerProtocol | None = None,
        cost_calculator: CostCalculator | None = None,
        settings: BenchmarkSettings | None = None,
    ) -> None:
        """
        Initialize the evaluation harness.

        Args:
            adapters: System adapters to benchmark
            registry: Task registry with loaded tasks
            scorer: Optional scorer for quality evaluation
            cost_calculator: Optional cost calculator
            settings: Benchmark settings
        """
        self._adapters = adapters
        self._registry = registry
        self._scorer = scorer
        self._cost_calculator = cost_calculator
        self._settings = settings or get_benchmark_settings()
        self._results: list[BenchmarkResult] = []
        self._run_id = ""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def results(self) -> list[BenchmarkResult]:
        """Return collected results."""
        return list(self._results)

    @property
    def run_id(self) -> str:
        """Return the current run ID."""
        return self._run_id

    async def run(
        self,
        task_ids: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """
        Execute the full benchmark suite.

        Args:
            task_ids: Optional list of specific task IDs to run.
                     If None, runs all registered tasks.

        Returns:
            List of all BenchmarkResult instances
        """
        self._run_id = str(uuid.uuid4())[:8]
        set_correlation_id(f"benchmark-{self._run_id}")

        self._logger.info(
            "Starting benchmark run %s with %d adapters and %d tasks",
            self._run_id,
            len(self._adapters),
            self._registry.task_count if task_ids is None else len(task_ids),
            extra={"run_id": self._run_id},
        )

        # Get tasks to run
        if task_ids:
            tasks = [self._registry.get(tid) for tid in task_ids]
        else:
            tasks = self._registry.get_all()

        if not tasks:
            self._logger.warning("No tasks to run")
            return []

        # Run health checks
        available_adapters = await self._health_check_adapters()
        if not available_adapters:
            self._logger.error("No adapters available, aborting benchmark")
            return []

        # Execute benchmark
        self._results = []
        run_config = self._settings.run

        for iteration in range(run_config.num_iterations):
            self._logger.info("Starting iteration %d/%d", iteration + 1, run_config.num_iterations)

            for task in tasks:
                for adapter in available_adapters:
                    result = await self._execute_with_timeout(adapter, task, iteration)
                    result.run_id = self._run_id
                    result.iteration = iteration

                    # Score the result
                    if self._scorer is not None:
                        result.scoring = await self._scorer.score(result, task)

                    self._results.append(result)

        # Apply cost estimates
        if self._cost_calculator:
            self._apply_costs()

        self._logger.info(
            "Benchmark run %s complete: %d results collected",
            self._run_id,
            len(self._results),
            extra={"run_id": self._run_id},
        )

        return self._results

    async def run_single(
        self,
        adapter: BenchmarkSystemProtocol,
        task: BenchmarkTask,
    ) -> BenchmarkResult:
        """
        Run a single task on a single adapter.

        Useful for debugging and targeted testing.

        Args:
            adapter: System adapter to use
            task: Task to execute

        Returns:
            BenchmarkResult
        """
        result = await self._execute_with_timeout(adapter, task, iteration=0)
        if self._scorer:
            result.scoring = await self._scorer.score(result, task)
        return result

    def save_results(self, output_dir: str | Path | None = None) -> Path:
        """
        Save results to JSON file.

        Args:
            output_dir: Output directory (uses config default if not provided)

        Returns:
            Path to the saved results file
        """
        out_dir = Path(output_dir or self._settings.report.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_path = out_dir / self._settings.report.results_filename
        data = {
            "run_id": self._run_id,
            "settings": self._settings.safe_dict(),
            "results": [r.to_dict() for r in self._results],
        }

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._logger.info("Results saved to %s", results_path)
        return results_path

    def get_results_by_system(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by system name."""
        by_system: dict[str, list[BenchmarkResult]] = {}
        for result in self._results:
            by_system.setdefault(result.system, []).append(result)
        return by_system

    def get_results_by_task(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by task ID."""
        by_task: dict[str, list[BenchmarkResult]] = {}
        for result in self._results:
            by_task.setdefault(result.task_id, []).append(result)
        return by_task

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the benchmark run."""
        by_system = self.get_results_by_system()

        system_summaries = {}
        for system, results in by_system.items():
            valid = [r for r in results if not r.has_error]
            system_summaries[system] = {
                "total_tasks": len(results),
                "successful": len(valid),
                "errors": len(results) - len(valid),
                "avg_latency_ms": (sum(r.total_latency_ms for r in valid) / len(valid)) if valid else 0,
                "avg_score": (sum(r.scoring.average_score for r in valid) / len(valid)) if valid else 0,
                "total_cost_usd": sum(r.estimated_cost_usd for r in results),
            }

        return {
            "run_id": self._run_id,
            "total_results": len(self._results),
            "systems": system_summaries,
        }

    async def _health_check_adapters(self) -> list[BenchmarkSystemProtocol]:
        """Run health checks on all adapters, return available ones."""
        available: list[BenchmarkSystemProtocol] = []
        for adapter in self._adapters:
            try:
                healthy = await adapter.health_check()
                if healthy:
                    available.append(adapter)
                    self._logger.info("Adapter '%s' health check passed", adapter.name)
                else:
                    self._logger.warning("Adapter '%s' health check failed", adapter.name)
            except Exception as e:
                self._logger.warning("Adapter '%s' health check error: %s", adapter.name, e)
        return available

    async def _execute_with_timeout(
        self,
        adapter: BenchmarkSystemProtocol,
        task: BenchmarkTask,
        iteration: int,
    ) -> BenchmarkResult:
        """Execute a task with timeout and retry handling."""
        timeout = self._settings.run.task_timeout_seconds
        max_retries = self._settings.run.max_retries if self._settings.run.retry_on_failure else 0

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    adapter.execute(task),
                    timeout=timeout,
                )
                return result
            except TimeoutError:
                last_error = TimeoutError(f"Task {task.task_id} timed out after {timeout}s")
                self._logger.warning(
                    "Task %s attempt %d timed out on %s",
                    task.task_id,
                    attempt + 1,
                    adapter.name,
                )
            except Exception as e:
                last_error = e
                self._logger.warning(
                    "Task %s attempt %d failed on %s: %s",
                    task.task_id,
                    attempt + 1,
                    adapter.name,
                    e,
                )

            if attempt < max_retries:
                backoff = self._settings.run.retry_backoff_base_seconds * (2**attempt)
                await asyncio.sleep(backoff)

        # All retries exhausted
        error_msg = f"Error: {type(last_error).__name__}: {last_error}" if last_error else "Error: Unknown"
        return BenchmarkResult(
            task_id=task.task_id,
            system=adapter.name,
            task_description=task.description,
            raw_response=error_msg,
        )

    def _apply_costs(self) -> None:
        """Apply cost estimates based on system-provider mapping."""
        system_providers: dict[str, tuple[str, str]] = {
            "langgraph_mcts": ("openai", "gpt-4-turbo-preview"),
            "vertex_adk": ("google", self._settings.adk.coordinator_model),
        }
        if self._cost_calculator:
            self._cost_calculator.apply_costs(self._results, system_providers)
