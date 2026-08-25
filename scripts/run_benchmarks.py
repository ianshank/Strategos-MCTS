#!/usr/bin/env python3
"""
CLI Script for Running Benchmark Suite

This script provides a command-line interface for running comprehensive
benchmarks on the RAG system and knowledge base.

Usage:
    # Run all benchmarks
    python scripts/run_benchmarks.py --all

    # Run specific benchmark
    python scripts/run_benchmarks.py --benchmark rag_retrieval

    # Run with custom config
    python scripts/run_benchmarks.py --config custom_benchmark_config.yaml

    # Compare with baseline
    python scripts/run_benchmarks.py --all --compare-with baseline_run_id

    # Generate only reports from existing runs
    python scripts/run_benchmarks.py --report-only --run-id 2024-01-15T10:30:00
"""

import argparse
import logging
from pathlib import Path
import sys

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.benchmark_suite import BenchmarkSuite

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: Path | None = None) -> dict:
    """Load benchmark configuration."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Try default config
    default_config = Path("training/benchmark_config.yaml")
    if default_config.exists():
        with open(default_config) as f:
            return yaml.safe_load(f)

    logger.warning("No config file found, using defaults")
    return {}


def run_all_benchmarks(suite: BenchmarkSuite, config: dict, model_config: dict) -> list:
    """Run all enabled benchmarks."""
    runs = []

    benchmarks_config = config.get("benchmarks", {})

    # Retrieval benchmark
    if benchmarks_config.get("rag_retrieval", {}).get("enabled", True):
        logger.info("Running RAG retrieval benchmarks...")
        from training.benchmark_suite import create_example_retrieval_function

        retrieval_fn = create_example_retrieval_function()

        for dataset in benchmarks_config.get("rag_retrieval", {}).get("datasets", []):
            dataset_name = dataset.get("name", "unknown")
            try:
                run = suite.run_retrieval_benchmark(
                    dataset_name=dataset_name,
                    retrieval_fn=retrieval_fn,
                    model_config=model_config,
                    k_values=benchmarks_config.get("rag_retrieval", {}).get("k_values", [10]),
                )
                runs.append(run)
                logger.info(f"Completed retrieval benchmark: {dataset_name}")
            except Exception as e:
                logger.error(f"Failed retrieval benchmark {dataset_name}: {e}", exc_info=True)

    # Reasoning benchmark
    if benchmarks_config.get("reasoning", {}).get("enabled", True):
        logger.info("Running reasoning benchmarks...")
        from training.benchmark_suite import create_example_reasoning_function

        reasoning_fn = create_example_reasoning_function()

        for dataset in benchmarks_config.get("reasoning", {}).get("datasets", []):
            dataset_name = dataset.get("name", "unknown")
            try:
                run = suite.run_reasoning_benchmark(
                    dataset_name=dataset_name,
                    reasoning_fn=reasoning_fn,
                    model_config=model_config,
                    use_llm_judge=benchmarks_config.get("reasoning", {}).get("use_llm_judge", False),
                )
                runs.append(run)
                logger.info(f"Completed reasoning benchmark: {dataset_name}")
            except Exception as e:
                logger.error(f"Failed reasoning benchmark {dataset_name}: {e}", exc_info=True)

    # Code generation benchmark
    if benchmarks_config.get("code_generation", {}).get("enabled", True):
        logger.info("Running code generation benchmarks...")
        from training.benchmark_suite import create_example_code_gen_function

        code_gen_fn = create_example_code_gen_function()

        for dataset in benchmarks_config.get("code_generation", {}).get("datasets", []):
            dataset_name = dataset.get("name", "unknown")
            try:
                run = suite.run_code_generation_benchmark(
                    dataset_name=dataset_name,
                    code_gen_fn=code_gen_fn,
                    model_config=model_config,
                    k_values=benchmarks_config.get("code_generation", {}).get("k_values", [1]),
                )
                runs.append(run)
                logger.info(f"Completed code generation benchmark: {dataset_name}")
            except Exception as e:
                logger.error(f"Failed code generation benchmark {dataset_name}: {e}", exc_info=True)

    return runs


def generate_reports(suite: BenchmarkSuite, runs: list, output_dir: Path) -> None:
    """Generate reports for benchmark runs."""
    logger.info("Generating reports...")

    for run in runs:
        run_dir = output_dir / run.benchmark_name / run.dataset_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate all formats
        suite.generate_report(run, output_format="json", output_file=run_dir / "report.json")
        suite.generate_report(run, output_format="markdown", output_file=run_dir / "report.md")
        suite.generate_report(run, output_format="html", output_file=run_dir / "report.html")

        # Export CSV
        suite.export_to_csv(run, run_dir / "metrics.csv")

    logger.info(f"Reports generated in {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks for RAG system evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to benchmark configuration YAML file",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all enabled benchmarks",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["rag_retrieval", "reasoning", "code_generation"],
        help="Run specific benchmark type",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to run (used with --benchmark)",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="baseline",
        help="Model configuration name from config file",
    )

    parser.add_argument(
        "--compare-with",
        type=str,
        help="Baseline run ID to compare against",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmarks"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate reports from existing runs only",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=== Benchmark Suite CLI ===")

    # Load configuration
    config = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_configs = config.get("model_configs", {})
    model_config = model_configs.get(args.model_config, {"name": args.model_config})

    # Initialize benchmark suite
    suite = BenchmarkSuite(config=config, output_dir=output_dir)
    suite.initialize_integrations()

    if args.report_only:
        logger.info("Report-only mode: Generating reports from existing runs")
        # Load existing runs and generate reports
        if suite.runs:
            generate_reports(suite, suite.runs, output_dir)
        else:
            logger.warning("No existing runs found")
        return 0

    # Run benchmarks
    runs = []

    if args.all:
        logger.info("Running all benchmarks...")
        runs = run_all_benchmarks(suite, config, model_config)

    elif args.benchmark:
        logger.info(f"Running {args.benchmark} benchmark...")

        if args.benchmark == "rag_retrieval":
            from training.benchmark_suite import create_example_retrieval_function

            retrieval_fn = create_example_retrieval_function()
            dataset_name = args.dataset or "custom_mcts"

            run = suite.run_retrieval_benchmark(
                dataset_name=dataset_name, retrieval_fn=retrieval_fn, model_config=model_config
            )
            runs.append(run)

        elif args.benchmark == "reasoning":
            from training.benchmark_suite import create_example_reasoning_function

            reasoning_fn = create_example_reasoning_function()
            dataset_name = args.dataset or "gsm8k_subset"

            run = suite.run_reasoning_benchmark(
                dataset_name=dataset_name, reasoning_fn=reasoning_fn, model_config=model_config
            )
            runs.append(run)

        elif args.benchmark == "code_generation":
            from training.benchmark_suite import create_example_code_gen_function

            code_gen_fn = create_example_code_gen_function()
            dataset_name = args.dataset or "humaneval_subset"

            run = suite.run_code_generation_benchmark(
                dataset_name=dataset_name, code_gen_fn=code_gen_fn, model_config=model_config
            )
            runs.append(run)

    else:
        logger.error("Must specify --all or --benchmark")
        parser.print_help()
        return 1

    # Generate reports
    if runs:
        generate_reports(suite, runs, output_dir)

        # Generate visualizations
        if args.visualize or config.get("visualization", {}).get("enabled", True):
            logger.info("Generating visualizations...")
            suite.visualize_results(runs=runs)

        # Compare with baseline if requested
        if args.compare_with and len(runs) > 0:
            logger.info(f"Comparing with baseline: {args.compare_with}")
            comparison = suite.compare_runs(
                baseline_run_id=args.compare_with,
                comparison_run_ids=[run.timestamp for run in runs],
                output_file=output_dir / "comparison.json",
            )

            logger.info("\n=== Comparison Results ===")
            for rec in comparison.recommendations:
                logger.info(f"  {rec}")

        # Summary
        logger.info("\n=== Benchmark Summary ===")
        for run in runs:
            logger.info(f"\n{run.benchmark_name} - {run.dataset_name}:")
            for metric_name, metric in run.metrics.items():
                logger.info(f"  {metric_name}: {metric.value:.4f}")

        logger.info(f"\nResults saved to: {output_dir}")

    # Close integrations
    if suite.wandb_run:
        import wandb

        wandb.finish()

    logger.info("=== Benchmark Suite Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
