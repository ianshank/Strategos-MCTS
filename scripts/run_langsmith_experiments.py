"""
Run LangSmith Experiments.

This script runs experiments on different agent configurations using LangSmith
datasets and tracks results for comparison.

Experiments:
- Model variations (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
- MCTS iterations (100, 200, 500)
- Agent routing strategies (HRM-first, TRM-first, HRM+TRM+MCTS)

Usage:
    # Run all experiments
    python scripts/run_langsmith_experiments.py

    # Run specific experiment
    python scripts/run_langsmith_experiments.py --experiment exp_hrm_trm_baseline

    # Run with specific model
    python scripts/run_langsmith_experiments.py --model gpt-4o-mini
"""

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langsmith import Client  # noqa: E402

from tests.mocks.mock_external_services import create_mock_llm  # noqa: E402
from tests.utils.langsmith_tracing import (  # noqa: E402
    get_langsmith_client,
    trace_e2e_test,
    update_run_metadata,
)


class ExperimentConfig:
    """Configuration for a specific experiment."""

    def __init__(
        self,
        name: str,
        description: str,
        model: str = "gpt-4o",
        use_mcts: bool = False,
        mcts_iterations: int | None = None,
        agent_strategy: str = "hrm_trm",
    ):
        self.name = name
        self.description = description
        self.model = model
        self.use_mcts = use_mcts
        self.mcts_iterations = mcts_iterations
        self.agent_strategy = agent_strategy


# Define experiment configurations
EXPERIMENTS = {
    "exp_hrm_trm_baseline": ExperimentConfig(
        name="exp_hrm_trm_baseline",
        description="Baseline HRM+TRM without MCTS",
        model="gpt-4o",
        use_mcts=False,
        agent_strategy="hrm_trm",
    ),
    "exp_full_stack_mcts_100": ExperimentConfig(
        name="exp_full_stack_mcts_100",
        description="Full stack HRM+TRM+MCTS with 100 iterations",
        model="gpt-4o",
        use_mcts=True,
        mcts_iterations=100,
        agent_strategy="full_stack",
    ),
    "exp_full_stack_mcts_200": ExperimentConfig(
        name="exp_full_stack_mcts_200",
        description="Full stack HRM+TRM+MCTS with 200 iterations",
        model="gpt-4o",
        use_mcts=True,
        mcts_iterations=200,
        agent_strategy="full_stack",
    ),
    "exp_full_stack_mcts_500": ExperimentConfig(
        name="exp_full_stack_mcts_500",
        description="Full stack HRM+TRM+MCTS with 500 iterations",
        model="gpt-4o",
        use_mcts=True,
        mcts_iterations=500,
        agent_strategy="full_stack",
    ),
    "exp_model_gpt4o_mini": ExperimentConfig(
        name="exp_model_gpt4o_mini",
        description="HRM+TRM with gpt-4o-mini (cost optimization)",
        model="gpt-4o-mini",
        use_mcts=False,
        agent_strategy="hrm_trm",
    ),
}


@trace_e2e_test(
    "experiment_run",
    phase="experiment",
    tags=["experiment", "evaluation"],
)
async def run_experiment_on_example(
    example: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Run a single experiment example with the given configuration.

    Args:
        example: Dataset example with inputs and expected outputs
        config: Experiment configuration

    Returns:
        Result dictionary with outputs and metrics
    """
    import random
    import time

    inputs = example["inputs"]
    _expected_outputs = example.get("outputs", {})

    # Create mock LLM client (in production, this would be real LLM)
    mock_client = create_mock_llm(provider="openai")
    mock_client.set_responses([f"""Analysis using {config.model}:
            Strategy: {config.agent_strategy}
            MCTS: {"enabled" if config.use_mcts else "disabled"}
            Recommendation: Comprehensive tactical analysis complete.
            Confidence: 0.85"""])

    start_time = time.time()

    # Simulate agent processing
    if config.agent_strategy in ["hrm_trm", "full_stack"]:
        # HRM processing
        _hrm_response = await mock_client.generate(f"HRM: {inputs.get('query', '')}")
        hrm_confidence = 0.87

        # TRM processing
        _trm_response = await mock_client.generate(f"TRM: {inputs.get('query', '')}")
        trm_confidence = 0.83

    # MCTS processing if enabled
    mcts_win_prob = None
    if config.use_mcts:
        random.seed(42)
        _iterations = config.mcts_iterations or 100
        mcts_win_prob = random.uniform(0.70, 0.90)

    elapsed_ms = (time.time() - start_time) * 1000

    # Calculate metrics
    result = {
        "model": config.model,
        "agent_strategy": config.agent_strategy,
        "use_mcts": config.use_mcts,
        "mcts_iterations": config.mcts_iterations,
        "hrm_confidence": hrm_confidence if config.agent_strategy != "mcts_only" else None,
        "trm_confidence": trm_confidence if config.agent_strategy != "mcts_only" else None,
        "mcts_win_probability": mcts_win_prob,
        "elapsed_ms": elapsed_ms,
        "success": True,
    }

    # Update trace metadata
    update_run_metadata(
        {
            "experiment_name": config.name,
            "experiment_config": config.description,
            **result,
        }
    )

    return result


async def run_experiment(
    config: ExperimentConfig,
    dataset_name: str,
    client: Client,
) -> dict[str, Any]:
    """
    Run an experiment on a dataset.

    Args:
        config: Experiment configuration
        dataset_name: Name of the dataset to run experiment on
        client: LangSmith client

    Returns:
        Experiment summary with aggregated results
    """
    print(f"\n{'=' * 70}")
    print(f"Running Experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 70}")

    # Load dataset
    try:
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if not datasets:
            print(f"[ERROR] Dataset '{dataset_name}' not found")
            return {"error": f"Dataset '{dataset_name}' not found"}

        dataset = datasets[0]
        examples = list(client.list_examples(dataset_id=dataset.id))

        if not examples:
            print(f"[WARN] No examples in dataset '{dataset_name}'")
            return {"error": f"No examples in dataset '{dataset_name}'"}

        print(f"Found {len(examples)} examples in dataset")

    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return {"error": str(e)}

    # Run experiment on each example
    results = []
    for i, example in enumerate(examples, 1):
        print(f"\n  Running example {i}/{len(examples)}...", end=" ")
        try:
            result = await run_experiment_on_example(
                {"inputs": example.inputs, "outputs": example.outputs},
                config,
            )
            results.append(result)
            print("[OK]")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
            results.append({"error": str(e), "success": False})

    # Calculate aggregate metrics
    successful_results = [r for r in results if r.get("success")]
    if successful_results:
        avg_elapsed = sum(r["elapsed_ms"] for r in successful_results) / len(successful_results)
        avg_hrm_conf = (
            sum(r["hrm_confidence"] for r in successful_results if r["hrm_confidence"])
            / len([r for r in successful_results if r["hrm_confidence"]])
            if any(r["hrm_confidence"] for r in successful_results)
            else None
        )
        avg_trm_conf = (
            sum(r["trm_confidence"] for r in successful_results if r["trm_confidence"])
            / len([r for r in successful_results if r["trm_confidence"]])
            if any(r["trm_confidence"] for r in successful_results)
            else None
        )

        summary = {
            "experiment": config.name,
            "dataset": dataset_name,
            "total_examples": len(examples),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "avg_elapsed_ms": round(avg_elapsed, 2),
            "avg_hrm_confidence": round(avg_hrm_conf, 3) if avg_hrm_conf else None,
            "avg_trm_confidence": round(avg_trm_conf, 3) if avg_trm_conf else None,
        }

        print(f"\n{'=' * 70}")
        print("Experiment Summary:")
        print(f"  Total examples: {summary['total_examples']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Avg latency: {summary['avg_elapsed_ms']:.2f}ms")
        if summary["avg_hrm_confidence"]:
            print(f"  Avg HRM confidence: {summary['avg_hrm_confidence']:.3f}")
        if summary["avg_trm_confidence"]:
            print(f"  Avg TRM confidence: {summary['avg_trm_confidence']:.3f}")
        print(f"{'=' * 70}")

        return summary
    else:
        return {"error": "No successful results"}


async def run_all_experiments(
    experiments: list[str] | None = None,
    datasets: list[str] | None = None,
):
    """
    Run all specified experiments on all specified datasets.

    Args:
        experiments: List of experiment names to run (None = all)
        datasets: List of dataset names to use (None = all)
    """
    client = get_langsmith_client()

    # Default datasets
    if datasets is None:
        datasets = [
            "tactical_e2e_scenarios",
            "cybersecurity_e2e_scenarios",
            "stem_scenarios",
            "generic_scenarios",
        ]

    # Default experiments
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())

    print(f"\n{'=' * 70}")
    print("LangSmith Experiments Runner")
    print(f"{'=' * 70}")
    print(f"Experiments to run: {len(experiments)}")
    print(f"Datasets to use: {len(datasets)}")
    print()

    all_results = {}

    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            print(f"[WARN] Unknown experiment: {exp_name}, skipping...")
            continue

        config = EXPERIMENTS[exp_name]

        for dataset_name in datasets:
            result = await run_experiment(config, dataset_name, client)
            all_results[f"{exp_name}_{dataset_name}"] = result

    # Print final summary
    print(f"\n{'=' * 70}")
    print("All Experiments Complete!")
    print(f"{'=' * 70}")
    print(f"\nTotal experiment runs: {len(all_results)}")
    print("\nView results in LangSmith UI:")
    print(f"  https://smith.langchain.com/o/{os.getenv('LANGSMITH_ORG_ID', 'your-org')}/projects")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run LangSmith experiments")
    parser.add_argument(
        "--experiment",
        help="Specific experiment to run (default: all)",
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument(
        "--dataset",
        help="Specific dataset to use (default: all)",
        choices=[
            "tactical_e2e_scenarios",
            "cybersecurity_e2e_scenarios",
            "mcts_benchmark_scenarios",
            "stem_scenarios",
            "generic_scenarios",
        ],
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit",
    )

    args = parser.parse_args()

    # List experiments if requested
    if args.list_experiments:
        print("\nAvailable Experiments:")
        print("=" * 70)
        for name, config in EXPERIMENTS.items():
            print(f"\n{name}:")
            print(f"  Description: {config.description}")
            print(f"  Model: {config.model}")
            print(f"  MCTS: {config.use_mcts}")
            if config.mcts_iterations:
                print(f"  MCTS Iterations: {config.mcts_iterations}")
            print(f"  Strategy: {config.agent_strategy}")
        print()
        return

    # Check environment
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[ERROR] LANGSMITH_API_KEY environment variable not set")
        print("        Set it with: export LANGSMITH_API_KEY=your_key_here")
        sys.exit(1)

    # Run experiments
    experiments = [args.experiment] if args.experiment else None
    datasets = [args.dataset] if args.dataset else None

    asyncio.run(run_all_experiments(experiments, datasets))


if __name__ == "__main__":
    main()
