"""
Benchmark Runner for LLM-Guided MCTS.

Provides:
- BenchmarkRunner: Run MCTS on benchmark problems
- BenchmarkReport: Comprehensive benchmark report
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.observability.logging import get_structured_logger

from .humaneval import HumanEvalBenchmark, HumanEvalProblem
from .metrics import BenchmarkMetrics, ProblemResult, aggregate_metrics

logger = get_structured_logger(__name__)


@dataclass
class BenchmarkRunnerConfig:
    """Configuration for benchmark runner."""

    # Problem selection
    num_problems: int | None = None
    """Number of problems to run (None = all)."""

    problem_ids: list[str] | None = None
    """Specific problem IDs to run."""

    difficulty_filter: str | None = None
    """Filter by difficulty: 'easy', 'medium', 'hard'."""

    seed: int = 42
    """Random seed for problem sampling."""

    # Execution
    num_samples_per_problem: int = 1
    """Number of solution samples per problem."""

    timeout_per_problem_seconds: float = 60.0
    """Timeout for each problem."""

    max_concurrent: int = 1
    """Maximum concurrent problems (1 = sequential)."""

    # Output
    output_dir: str | Path = "./benchmark_results"
    """Directory for saving results."""

    save_solutions: bool = True
    """Save generated solutions."""

    save_training_data: bool = True
    """Save MCTS training data."""

    verbose: bool = True
    """Print progress during run."""

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        if self.num_samples_per_problem < 1:
            errors.append("num_samples_per_problem must be >= 1")
        if self.timeout_per_problem_seconds <= 0:
            errors.append("timeout_per_problem_seconds must be > 0")
        if self.max_concurrent < 1:
            errors.append("max_concurrent must be >= 1")

        if errors:
            raise ValueError("Invalid BenchmarkRunnerConfig:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""

    # Identification
    run_id: str
    """Unique run identifier."""

    timestamp: str
    """Run timestamp."""

    # Configuration
    mcts_config_name: str
    """MCTS configuration used."""

    runner_config: dict[str, Any]
    """Runner configuration."""

    # Results
    metrics: BenchmarkMetrics
    """Aggregated metrics."""

    # Environment
    python_version: str = ""
    """Python version."""

    model_name: str = ""
    """LLM model name."""

    hardware_info: dict[str, str] = field(default_factory=dict)
    """Hardware information."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mcts_config_name": self.mcts_config_name,
            "runner_config": self.runner_config,
            "metrics": self.metrics.to_dict(),
            "python_version": self.python_version,
            "model_name": self.model_name,
            "hardware_info": self.hardware_info,
        }

    def save(self, filepath: str | Path) -> None:
        """Save report to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info("Saved benchmark report", filepath=str(filepath))

    @classmethod
    def load(cls, filepath: str | Path) -> BenchmarkReport:
        """Load report from JSON file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in benchmark report file: {e}") from e
        except OSError as e:
            raise ValueError(f"Failed to read benchmark report file: {e}") from e

        # Validate required fields
        required_fields = ["run_id", "timestamp", "mcts_config_name", "runner_config", "metrics"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields in benchmark report: {missing_fields}")

        if "problem_results" not in data.get("metrics", {}):
            raise ValueError("Missing 'problem_results' in metrics")

        try:
            # Reconstruct metrics and results
            problem_results = [ProblemResult(**r) for r in data["metrics"]["problem_results"]]
            metrics_data = data["metrics"].copy()
            metrics_data["problem_results"] = problem_results
            metrics = BenchmarkMetrics(**metrics_data)
        except (TypeError, KeyError) as e:
            raise ValueError(f"Invalid metrics structure in benchmark report: {e}") from e

        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            mcts_config_name=data["mcts_config_name"],
            runner_config=data["runner_config"],
            metrics=metrics,
            python_version=data.get("python_version", ""),
            model_name=data.get("model_name", ""),
            hardware_info=data.get("hardware_info", {}),
        )


class MCTSEngineProtocol(Protocol):
    """Protocol for MCTS engine to allow dependency injection."""

    async def search(
        self,
        problem: str,
        test_cases: list[str],
        initial_code: str | None = None,
    ) -> Any:
        """Run MCTS search."""
        ...


class BenchmarkRunner:
    """
    Runner for benchmarking LLM-Guided MCTS on code generation problems.

    Supports multiple benchmark types and comprehensive metrics collection.
    """

    def __init__(
        self,
        mcts_engine: MCTSEngineProtocol,
        config: BenchmarkRunnerConfig | None = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            mcts_engine: MCTS engine to use for solving problems
            config: Runner configuration
        """
        self._engine = mcts_engine
        self._config = config or BenchmarkRunnerConfig()
        self._config.validate()

        # Results storage
        self._results: list[ProblemResult] = []
        self._start_time: float = 0.0

        logger.info(
            "Initialized BenchmarkRunner",
            num_samples=self._config.num_samples_per_problem,
            timeout=self._config.timeout_per_problem_seconds,
        )

    async def run_humaneval(
        self,
        benchmark: HumanEvalBenchmark | None = None,
    ) -> BenchmarkReport:
        """
        Run benchmark on HumanEval problems.

        Args:
            benchmark: HumanEval benchmark instance (or create default)

        Returns:
            BenchmarkReport with results
        """
        if benchmark is None:
            benchmark = HumanEvalBenchmark()

        # Select problems
        problems = self._select_problems(list(benchmark))

        logger.info(
            "Running HumanEval benchmark",
            num_problems=len(problems),
        )

        self._start_time = time.perf_counter()
        self._results = []

        # Run problems
        if self._config.max_concurrent == 1:
            # Sequential execution
            for i, problem in enumerate(problems):
                if self._config.verbose:
                    logger.info(
                        "Solving problem",
                        progress=f"{i + 1}/{len(problems)}",
                        task_id=problem.task_id,
                    )

                result = await self._solve_problem(problem)
                self._results.append(result)

                if self._config.verbose:
                    status = "SOLVED" if result.solved else "FAILED"
                    logger.info(
                        "Problem complete",
                        status=status,
                        time_ms=round(result.total_time_ms),
                    )
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(self._config.max_concurrent)

            async def solve_with_limit(problem: HumanEvalProblem) -> ProblemResult:
                async with semaphore:
                    return await self._solve_problem(problem)

            self._results = await asyncio.gather(*[solve_with_limit(p) for p in problems])

        # Aggregate metrics
        metrics = aggregate_metrics(self._results)
        metrics.total_time_ms = (time.perf_counter() - self._start_time) * 1000

        # Create report
        report = self._create_report(metrics, "humaneval")

        # Save results
        self._save_results(report)

        return report

    def _select_problems(
        self,
        all_problems: list[HumanEvalProblem],
    ) -> list[HumanEvalProblem]:
        """Select problems to run based on configuration."""
        problems = all_problems

        # Filter by specific IDs
        if self._config.problem_ids:
            problem_set = set(self._config.problem_ids)
            problems = [p for p in problems if p.task_id in problem_set]

        # Filter by difficulty
        if self._config.difficulty_filter:
            problems = [p for p in problems if p.difficulty == self._config.difficulty_filter]

        # Sample if needed
        if self._config.num_problems and len(problems) > self._config.num_problems:
            import random

            random.seed(self._config.seed)
            problems = random.sample(problems, self._config.num_problems)

        return problems

    async def _solve_problem(self, problem: HumanEvalProblem) -> ProblemResult:
        """Solve a single problem."""
        start_time = time.perf_counter()

        result = ProblemResult(
            task_id=problem.task_id,
            solved=False,
            num_attempts=0,
            num_passed=0,
            num_test_cases=len(problem.test_cases),
        )

        # Extract test cases for execution
        test_cases = problem.test_cases

        first_solution_found = False

        for sample_idx in range(self._config.num_samples_per_problem):
            result.num_attempts += 1

            try:
                # Run MCTS with timeout
                search_result = await asyncio.wait_for(
                    self._engine.search(
                        problem=problem.get_problem_description(),
                        test_cases=test_cases,
                        initial_code=problem.prompt,
                    ),
                    timeout=self._config.timeout_per_problem_seconds,
                )

                # Check result
                if hasattr(search_result, "solution_found") and search_result.solution_found:
                    result.num_passed += 1
                    result.solved = True

                    if not first_solution_found:
                        first_solution_found = True
                        result.first_solution_time_ms = (time.perf_counter() - start_time) * 1000
                        result.best_solution = getattr(search_result, "best_code", None)

                # Collect MCTS stats
                if hasattr(search_result, "num_iterations"):
                    result.num_iterations = max(result.num_iterations, search_result.num_iterations)
                if hasattr(search_result, "tree_size"):
                    result.tree_size = max(result.tree_size, search_result.tree_size)
                if hasattr(search_result, "max_depth"):
                    result.max_depth_reached = max(result.max_depth_reached, search_result.max_depth)

            except TimeoutError:
                result.timeout_errors += 1
                result.error_messages.append(f"Sample {sample_idx + 1} timed out")

            except SyntaxError as e:
                result.syntax_errors += 1
                result.error_messages.append(f"Sample {sample_idx + 1} syntax error: {e}")

            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError, KeyError) as e:
                # Specific runtime errors from code execution
                result.runtime_errors += 1
                result.error_messages.append(f"Sample {sample_idx + 1} runtime error: {type(e).__name__}: {e}")

            except (MemoryError, RecursionError) as e:
                # Resource exhaustion - count as runtime error but log prominently
                result.runtime_errors += 1
                result.error_messages.append(f"Sample {sample_idx + 1} resource error: {type(e).__name__}: {e}")
                logger.warning(
                    "Resource exhaustion during benchmark",
                    task_id=problem.task_id,
                    error=str(e),
                )

            except (KeyboardInterrupt, SystemExit):
                # Re-raise system exceptions - don't swallow these
                raise

            except Exception as e:
                # Truly unexpected errors - log for debugging but continue
                result.runtime_errors += 1
                result.error_messages.append(f"Sample {sample_idx + 1} unexpected error: {type(e).__name__}: {e}")
                logger.error(
                    "Unexpected error during benchmark sample",
                    task_id=problem.task_id,
                    error_type=type(e).__name__,
                    error=str(e),
                )

        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        if result.best_solution:
            result.solution_length = len(result.best_solution)

        return result

    def _create_report(self, metrics: BenchmarkMetrics, benchmark_name: str) -> BenchmarkReport:
        """Create benchmark report."""
        import platform
        import sys
        import uuid

        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Get engine config name if available
        config_name = "default"
        if hasattr(self._engine, "_config") and hasattr(self._engine._config, "name"):
            config_name = self._engine._config.name

        # Get model name if available
        model_name = "unknown"
        if hasattr(self._engine, "_llm_client"):
            model_name = getattr(self._engine._llm_client, "model", "unknown")

        return BenchmarkReport(
            run_id=f"{benchmark_name}_{run_id}",
            timestamp=timestamp,
            mcts_config_name=config_name,
            runner_config={
                "num_samples_per_problem": self._config.num_samples_per_problem,
                "timeout_per_problem_seconds": self._config.timeout_per_problem_seconds,
                "num_problems": self._config.num_problems,
                "difficulty_filter": self._config.difficulty_filter,
            },
            metrics=metrics,
            python_version=sys.version,
            model_name=model_name,
            hardware_info={
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
        )

    def _save_results(self, report: BenchmarkReport) -> None:
        """Save benchmark results."""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        report_path = output_dir / f"{report.run_id}_report.json"
        report.save(report_path)

        # Save solutions if configured
        if self._config.save_solutions:
            solutions_path = output_dir / f"{report.run_id}_solutions.jsonl"
            with open(solutions_path, "w") as f:
                for result in self._results:
                    if result.best_solution:
                        f.write(
                            json.dumps(
                                {
                                    "task_id": result.task_id,
                                    "solution": result.best_solution,
                                    "solved": result.solved,
                                }
                            )
                            + "\n"
                        )

        # Log summary
        if self._config.verbose:
            logger.info("Benchmark complete", summary=report.metrics.summary())


async def run_benchmark(
    mcts_engine: MCTSEngineProtocol,
    benchmark_type: str = "humaneval",
    num_problems: int | None = None,
    num_samples: int = 1,
    output_dir: str | Path = "./benchmark_results",
    verbose: bool = True,
) -> BenchmarkReport:
    """
    Convenience function to run a benchmark.

    Args:
        mcts_engine: MCTS engine to use
        benchmark_type: Type of benchmark ('humaneval')
        num_problems: Number of problems (None = all)
        num_samples: Number of samples per problem
        output_dir: Output directory
        verbose: Print progress

    Returns:
        BenchmarkReport with results
    """
    config = BenchmarkRunnerConfig(
        num_problems=num_problems,
        num_samples_per_problem=num_samples,
        output_dir=output_dir,
        verbose=verbose,
    )

    runner = BenchmarkRunner(mcts_engine, config)

    if benchmark_type == "humaneval":
        return await runner.run_humaneval()
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
