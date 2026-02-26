"""
Benchmark CLI runner entry point.

Provides a command-line interface for running benchmark comparisons
between LangGraph MCTS and Google ADK Agent Builder.

Usage:
    python -m src.benchmark.cli [OPTIONS]

Examples:
    # Run full benchmark with defaults
    python -m src.benchmark.cli

    # Run specific systems
    python -m src.benchmark.cli --systems langgraph_mcts vertex_adk

    # Run specific tasks
    python -m src.benchmark.cli --tasks A1 A2 B1

    # Custom output directory
    python -m src.benchmark.cli --output-dir ./results

    # Run with 3 iterations for statistical significance
    python -m src.benchmark.cli --iterations 3

    # Dry run (list tasks and systems without executing)
    python -m src.benchmark.cli --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from src.benchmark.config.benchmark_settings import BenchmarkSettings, get_benchmark_settings, reset_benchmark_settings
from src.benchmark.factory import BenchmarkFactory
from src.observability.logging import get_correlation_id, set_correlation_id

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Run benchmark comparisons between multi-agent systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  BENCHMARK_ENABLED         Enable/disable benchmark (default: true)\n"
            "  BENCHMARK_RUN_NUM_ITERATIONS  Number of iterations (default: 1)\n"
            "  BENCHMARK_SCORING_ENABLED Enable LLM scoring (default: true)\n"
            "  BENCHMARK_REPORT_OUTPUT_DIR  Output directory\n"
            "\n"
            "See .env.example for full configuration reference."
        ),
    )

    parser.add_argument(
        "--systems",
        nargs="+",
        metavar="NAME",
        help="Systems to benchmark (default: all available). Options: langgraph_mcts, vertex_adk",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="ID",
        help="Specific task IDs to run (default: all). Example: A1 A2 B1 C1",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        metavar="N",
        help="Number of benchmark iterations (overrides BENCHMARK_RUN_NUM_ITERATIONS)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="DIR",
        help="Output directory for results and report (overrides BENCHMARK_REPORT_OUTPUT_DIR)",
    )

    parser.add_argument(
        "--no-scoring",
        action="store_true",
        help="Disable LLM-as-judge scoring",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List systems and tasks without executing",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Skip saving raw results to JSON",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report generation",
    )

    return parser


def configure_logging(level: str) -> None:
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def apply_cli_overrides(settings: BenchmarkSettings, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to settings."""
    if args.iterations is not None:
        settings._run = None  # Reset lazy cache
        # Create new config with override
        from src.benchmark.config.benchmark_settings import BenchmarkRunConfig

        settings._run = BenchmarkRunConfig(num_iterations=args.iterations)

    if args.output_dir:
        settings._report = None
        from src.benchmark.config.benchmark_settings import ReportConfig

        settings._report = ReportConfig(output_dir=args.output_dir)

    if args.no_scoring:
        settings._scoring = None
        from src.benchmark.config.benchmark_settings import ScoringConfig

        settings._scoring = ScoringConfig(enabled=False)


async def run_benchmark(args: argparse.Namespace) -> int:
    """
    Execute the benchmark run.

    Args:
        args: Parsed CLI arguments

    Returns:
        Exit code (0 = success, 1 = error, 2 = no results)
    """
    set_correlation_id(f"cli-benchmark-{uuid.uuid4().hex[:8]}")

    # Load and configure settings
    reset_benchmark_settings()
    settings = get_benchmark_settings()
    apply_cli_overrides(settings, args)

    if not settings.benchmark_enabled:
        logger.error("Benchmarking is disabled (BENCHMARK_ENABLED=false)")
        return 1

    # Create factory
    factory = BenchmarkFactory(settings=settings)

    # Dry run mode
    if args.dry_run:
        return _dry_run(factory, args)

    logger.info(
        "Starting benchmark run",
        extra={"correlation_id": get_correlation_id()},
    )

    # Create and run harness
    try:
        harness = factory.create_harness(systems=args.systems)
    except Exception as e:
        logger.error("Failed to create harness: %s", e, exc_info=True)
        return 1

    # Execute
    try:
        results = await harness.run(task_ids=args.tasks)
    except Exception as e:
        logger.error("Benchmark execution failed: %s", e, exc_info=True)
        return 1

    if not results:
        logger.warning("No results collected")
        return 2

    # Print summary
    summary = harness.summary()
    _print_summary(summary)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(settings.report.output_dir)

    if not args.no_save_results:
        try:
            results_path = harness.save_results(output_dir)
            logger.info("Results saved to %s", results_path)
        except Exception as e:
            logger.error("Failed to save results: %s", e)

    # Generate report
    if not args.no_report:
        try:
            report_gen = factory.create_report_generator()
            report = report_gen.generate(
                results,
                title=f"Benchmark: {settings.benchmark_name}",
            )
            report_path = report_gen.save(report, str(output_dir))
            logger.info("Report saved to %s", report_path)
        except Exception as e:
            logger.error("Failed to generate report: %s", e)

    return 0


def _dry_run(factory: BenchmarkFactory, args: argparse.Namespace) -> int:
    """Print what would be run without executing."""
    print("\n=== Benchmark Dry Run ===\n")

    # Show adapters
    adapter_factory = factory.create_adapter_factory()
    all_systems = adapter_factory.get_available_systems()

    if args.systems:
        systems = args.systems
    else:
        systems = all_systems

    print("Systems:")
    for name in systems:
        available = name in all_systems
        status = "available" if available else "not registered"
        print(f"  - {name} ({status})")

    # Show tasks
    registry = factory.create_task_registry()

    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = [t.task_id for t in registry.get_all()]

    print(f"\nTasks ({len(task_ids)}):")
    for tid in task_ids:
        try:
            task = registry.get(tid)
            print(f"  - {tid}: {task.description} [{task.category.value}]")
        except KeyError:
            print(f"  - {tid}: NOT FOUND")

    # Show config
    settings = factory.settings
    print(f"\nIterations: {settings.run.num_iterations}")
    print(f"Scoring: {'enabled' if settings.scoring.enabled else 'disabled'}")
    print(f"Output: {settings.report.output_dir}")

    total_runs = len(systems) * len(task_ids) * settings.run.num_iterations
    print(f"\nTotal executions: {total_runs}")

    return 0


def _print_summary(summary: dict[str, Any]) -> None:
    """Print benchmark summary to stdout."""
    print("\n" + "=" * 60)
    print(f"  Benchmark Run: {summary['run_id']}")
    print(f"  Total Results: {summary['total_results']}")
    print("=" * 60)

    for system, stats in summary.get("systems", {}).items():
        print(f"\n  System: {system}")
        print(f"    Tasks: {stats['successful']}/{stats['total_tasks']} successful")
        print(f"    Avg Latency: {stats['avg_latency_ms']:.0f}ms")
        print(f"    Avg Score: {stats['avg_score']:.2f}")
        print(f"    Est. Cost: ${stats['total_cost_usd']:.4f}")

    print("\n" + "=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    exit_code = asyncio.run(run_benchmark(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
