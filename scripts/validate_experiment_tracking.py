#!/usr/bin/env python3
"""
Experiment Tracking Validation Script.

Validates Braintrust and Weights & Biases integrations.

Usage:
    python scripts/validate_experiment_tracking.py

Expected outcomes:
- Braintrust initialization
- W&B initialization
- Metric logging
- Artifact tracking
"""

import logging
import os
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.experiment_tracker import (
    BraintrustTracker,
    TrainingMetrics,
    UnifiedExperimentTracker,
    WandBTracker,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_braintrust():
    """Validate Braintrust integration."""
    print("\n" + "=" * 60)
    print("VALIDATING BRAINTRUST EXPERIMENT TRACKING")
    print("=" * 60)

    api_key = os.getenv("BRAINTRUST_API_KEY")
    if not api_key:
        print("[INFO] BRAINTRUST_API_KEY not set")
        print("  Will use offline mode for validation")
    else:
        print("[OK] BRAINTRUST_API_KEY found")

    try:
        tracker = BraintrustTracker(project_name="mcts-training-validation")
        print(f"[OK] Tracker initialized (offline={tracker._offline_mode})")

        # Initialize experiment
        exp_id = tracker.init_experiment(
            name="validation_experiment",
            description="Testing Braintrust integration",
            tags=["validation", "testing"],
        )
        print(f"[OK] Experiment created: {exp_id}")

        # Log hyperparameters
        hyperparams = {
            "model_type": "rnn",
            "hidden_size": 64,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }
        tracker.log_hyperparameters(hyperparams)
        print("[OK] Hyperparameters logged")

        # Log training metrics
        for epoch in range(5):
            metrics = TrainingMetrics(
                epoch=epoch,
                step=epoch * 100,
                train_loss=1.0 / (epoch + 1),
                val_loss=1.1 / (epoch + 1),
                accuracy=0.5 + epoch * 0.1,
                custom_metrics={
                    "consensus_score": 0.7 + epoch * 0.05,
                    "mcts_iterations": 100 + epoch * 20,
                },
            )
            tracker.log_training_step(metrics)

        print("[OK] Training metrics logged (5 epochs)")

        # Log evaluation
        tracker.log_evaluation(
            input_data="Test tactical scenario",
            output="Recommended action: Alpha",
            expected="Recommended action: Alpha",
            scores={"accuracy": 1.0, "confidence": 0.85},
        )
        print("[OK] Evaluation logged")

        # Create and log temporary artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test model artifact")
            artifact_path = f.name

        tracker.log_artifact(artifact_path, "test_artifact")
        print(f"[OK] Artifact logged: {artifact_path}")

        # Get summary
        summary = tracker.get_summary()
        print("\nExperiment Summary:")
        print(f"  - ID: {summary['id']}")
        print(f"  - Metrics logged: {summary['metrics_count']}")
        print(f"  - Offline mode: {summary['offline']}")

        # End experiment
        tracker.end_experiment()
        print("[OK] Experiment ended successfully")

        # Cleanup
        os.unlink(artifact_path)

        print("\n[PASS] BRAINTRUST VALIDATION PASSED")
        return True

    except Exception as e:
        print(f"\n[FAIL] BRAINTRUST VALIDATION FAILED: {e}")
        logger.exception("Braintrust validation error")
        return False


def validate_wandb():
    """Validate Weights & Biases integration."""
    print("\n" + "=" * 60)
    print("VALIDATING WEIGHTS & BIASES INTEGRATION")
    print("=" * 60)

    api_key = os.getenv("WANDB_API_KEY")
    wandb_mode = os.getenv("WANDB_MODE", "offline")

    if not api_key:
        print("[INFO] WANDB_API_KEY not set")
        print("  Will use offline mode for validation")
        os.environ["WANDB_MODE"] = "offline"
    else:
        print("[OK] WANDB_API_KEY found")

    if wandb_mode == "offline":
        print("[INFO] WANDB_MODE=offline (CI/test environment)")

    try:
        tracker = WandBTracker(project_name="mcts-training-validation")
        print(f"[OK] W&B tracker initialized (offline={tracker._offline_mode})")

        # Initialize run
        config = {
            "model_type": "rnn",
            "hidden_size": 64,
            "learning_rate": 0.001,
        }
        tracker.init_run(
            name="validation_run",
            config=config,
            tags=["validation", "testing"],
            notes="Testing W&B integration",
        )
        print("[OK] Run initialized")

        # Log metrics
        for epoch in range(5):
            metrics = TrainingMetrics(
                epoch=epoch,
                step=epoch * 100,
                train_loss=1.0 / (epoch + 1),
                val_loss=1.1 / (epoch + 1),
                accuracy=0.5 + epoch * 0.1,
            )
            tracker.log_training_step(metrics)

        print("[OK] Training metrics logged (5 epochs)")

        # Update config
        tracker.update_config({"additional_param": "test_value"})
        print("[OK] Config updated")

        # Log custom metrics
        tracker.log(
            {
                "mcts_simulation_time": 1.5,
                "consensus_score": 0.83,
                "agent_routing_accuracy": 0.79,
            }
        )
        print("[OK] Custom metrics logged")

        # Finish run
        tracker.finish()
        print("[OK] Run finished successfully")

        print("\n[PASS] WANDB VALIDATION PASSED")
        return True

    except Exception as e:
        print(f"\n[FAIL] WANDB VALIDATION FAILED: {e}")
        logger.exception("W&B validation error")
        return False


def validate_unified_tracker():
    """Validate unified experiment tracker."""
    print("\n" + "=" * 60)
    print("VALIDATING UNIFIED EXPERIMENT TRACKER")
    print("=" * 60)

    try:
        tracker = UnifiedExperimentTracker(project_name="mcts-unified-validation")
        print("[OK] Unified tracker initialized")

        # Initialize experiment
        config = {
            "model_type": "rnn",
            "hidden_size": 64,
            "learning_rate": 0.001,
        }
        tracker.init_experiment(
            name="unified_validation",
            config=config,
            description="Testing unified tracker",
            tags=["unified", "validation"],
        )
        print("[OK] Experiment initialized on both platforms")

        # Log metrics to both
        for epoch in range(3):
            metrics = TrainingMetrics(
                epoch=epoch,
                step=epoch * 100,
                train_loss=1.0 / (epoch + 1),
                accuracy=0.6 + epoch * 0.1,
            )
            tracker.log_metrics(metrics)

        print("[OK] Metrics logged to both platforms")

        # Log evaluation
        tracker.log_evaluation(
            input_data="Test input",
            output="Test output",
            expected="Test expected",
            scores={"accuracy": 0.9},
        )
        print("[OK] Evaluation logged")

        # Finish
        summary = tracker.finish()
        print("[OK] Experiment ended on both platforms")
        print(f"\nSummary: {summary}")

        print("\n[PASS] UNIFIED TRACKER VALIDATION PASSED")
        return True

    except Exception as e:
        print(f"\n[FAIL] UNIFIED TRACKER VALIDATION FAILED: {e}")
        logger.exception("Unified tracker validation error")
        return False


def main():
    """Run all experiment tracking validations."""
    print("=" * 60)
    print("EXPERIMENT TRACKING VALIDATION")
    print("=" * 60)
    print("\nThis script validates experiment tracking integrations:")
    print("  - Braintrust experiment tracking")
    print("  - Weights & Biases logging")
    print("  - Unified tracking interface")
    print("\nNote: Without API keys, validations run in offline mode.")

    results = {}

    # Run validations
    results["braintrust"] = validate_braintrust()
    results["wandb"] = validate_wandb()
    results["unified"] = validate_unified_tracker()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name.upper()}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL VALIDATIONS PASSED - Experiment tracking ready!")
        print("=" * 60)

        # Print setup recommendations
        print("\nNext steps:")
        print("1. Set BRAINTRUST_API_KEY in .env for Braintrust integration")
        print("2. Set WANDB_API_KEY in .env for W&B integration")
        print("3. Import trackers in training scripts:")
        print("   from src.training import BraintrustTracker, WandBTracker")
        print("4. View experiments at:")
        print("   - Braintrust: https://www.braintrust.dev/")
        print("   - W&B: https://wandb.ai/")
        return 0
    else:
        print("SOME VALIDATIONS FAILED - Check logs above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
