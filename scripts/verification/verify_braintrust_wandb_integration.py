"""
Verify Braintrust and Wandb integrations for the Multi-Agent MCTS Framework.

This script tests:
1. Package availability
2. Connection status
3. Logging functionality
4. Error handling and buffering
"""

import os
import sys
from datetime import datetime


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def verify_braintrust():
    """Verify Braintrust integration."""
    print_section("BRAINTRUST INTEGRATION VERIFICATION")

    # Check package installation
    print("\n1. Checking package installation...")
    try:
        import braintrust

        print("   [OK] braintrust package is installed")
        print(f"   Version: {braintrust.__version__ if hasattr(braintrust, '__version__') else 'Unknown'}")
        _BRAINTRUST_AVAILABLE = True  # noqa: F841 - variable used to track state
    except ImportError:
        print("   [INFO] braintrust package not installed")
        print("   To install: pip install braintrust")
        _BRAINTRUST_AVAILABLE = False  # noqa: F841 - variable used to track state
        return False

    # Check environment configuration
    print("\n2. Checking environment configuration...")
    api_key = os.environ.get("BRAINTRUST_API_KEY")

    if api_key:
        print(f"   [OK] BRAINTRUST_API_KEY is set (length: {len(api_key)})")
    else:
        print("   [INFO] BRAINTRUST_API_KEY not set")
        print("   Get your API key from: https://www.braintrust.dev/")

    # Test the integration module
    print("\n3. Testing BraintrustTracker...")
    try:
        from src.observability.braintrust_tracker import BRAINTRUST_AVAILABLE as BT_AVAILABLE
        from src.observability.braintrust_tracker import BraintrustTracker

        if not BT_AVAILABLE:
            print("   [FAIL] Braintrust not available in tracker module")
            return False

        # Create tracker instance
        tracker = BraintrustTracker(project_name="integration-test", auto_init=True)

        if tracker.is_available:
            print("   [OK] BraintrustTracker initialized successfully")

            # Test experiment creation
            print("\n   Testing experiment creation...")
            experiment = tracker.start_experiment(
                experiment_name="verification_test", metadata={"type": "integration_test"}
            )

            if experiment:
                print("   [OK] Experiment created successfully")

                # Test logging
                print("\n   Testing metric logging...")
                tracker.log_hyperparameters({"learning_rate": 0.001, "batch_size": 32, "model_type": "test"})

                tracker.log_epoch_summary(epoch=1, train_loss=0.5, val_loss=0.4, train_accuracy=0.85, val_accuracy=0.82)

                print("   [OK] Metrics logged successfully")

                # End experiment
                url = tracker.end_experiment()
                if url:
                    print(f"   [OK] Experiment ended. URL: {url}")
                else:
                    print("   [OK] Experiment ended")

                return True
            else:
                print("   [FAIL] Failed to create experiment")
                return False
        else:
            print("   [INFO] BraintrustTracker not available (check API key)")

            # Test buffering
            print("\n   Testing offline buffering...")
            test_tracker = BraintrustTracker(api_key=None, auto_init=False)
            test_tracker.log_hyperparameters({"test": "offline"})
            test_tracker.log_epoch_summary(1, train_loss=0.5, val_loss=0.4)

            buffered = test_tracker.get_buffered_metrics()
            print(f"   [OK] Buffered {len(buffered)} operations")

            return False

    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_wandb():
    """Verify Weights & Biases integration."""
    print_section("WEIGHTS & BIASES (WANDB) INTEGRATION VERIFICATION")

    # Check package installation
    print("\n1. Checking package installation...")
    try:
        import wandb

        print("   [OK] wandb package is installed")
        print(f"   Version: {wandb.__version__}")
        _WANDB_AVAILABLE = True  # noqa: F841 - variable used to track state
    except ImportError:
        print("   [INFO] wandb package not installed")
        print("   To install: pip install wandb")
        _WANDB_AVAILABLE = False  # noqa: F841 - variable used to track state
        return False

    # Check environment configuration
    print("\n2. Checking environment configuration...")
    api_key = os.environ.get("WANDB_API_KEY")

    if api_key:
        print(f"   [OK] WANDB_API_KEY is set (length: {len(api_key)})")
    else:
        print("   [INFO] WANDB_API_KEY not set")
        print("   Get your API key from: https://wandb.ai/settings")

    # Test basic functionality
    print("\n3. Testing Wandb functionality...")
    try:
        import wandb

        # Initialize in offline mode for testing
        print("   Initializing run in offline mode...")
        _run = wandb.init(  # noqa: F841 - run object managed by wandb.finish()
            project="langgraph-mcts-verification",
            name=f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={"test": "integration_verification", "framework": "langgraph-mcts", "model": "meta-controller"},
            mode="offline",  # Offline mode for testing
        )

        print("   [OK] Wandb run initialized")

        # Log some test metrics
        print("   Logging test metrics...")
        wandb.log({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "accuracy": 0.85})

        # Log a summary
        wandb.summary["best_accuracy"] = 0.85
        wandb.summary["final_loss"] = 0.4

        print("   [OK] Metrics logged successfully")

        # Finish the run
        wandb.finish()
        print("   [OK] Run finished successfully")
        print("   Note: Run was created in offline mode")
        print("   To sync: wandb sync [run_path]")

        return True

    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {e}")
        return False


def check_training_integration():
    """Check if experiment tracking is integrated into training scripts."""
    print_section("TRAINING INTEGRATION CHECK")

    print("\n1. Checking RNN training script...")
    try:
        with open("src/training/train_rnn.py") as f:
            content = f.read()

        braintrust_integrated = "BraintrustTracker" in content
        wandb_integrated = "wandb" in content

        if braintrust_integrated:
            print("   [OK] Braintrust is integrated in RNN training")
        else:
            print("   [INFO] Braintrust not found in RNN training")

        if wandb_integrated:
            print("   [OK] Wandb is integrated in RNN training")
        else:
            print("   [INFO] Wandb not found in RNN training")
    except Exception as e:
        print(f"   [ERROR] Could not check RNN training: {e}")

    print("\n2. Checking BERT training script...")
    try:
        with open("src/training/train_bert_lora.py") as f:
            content = f.read()

        braintrust_integrated = "braintrust" in content.lower()
        wandb_integrated = "wandb" in content.lower()

        if braintrust_integrated:
            print("   [OK] Braintrust mentioned in BERT training")
        else:
            print("   [INFO] Braintrust not found in BERT training")

        if wandb_integrated:
            print("   [OK] Wandb mentioned in BERT training")
            print("   Note: HuggingFace Trainer supports wandb natively")
        else:
            print("   [INFO] Wandb not found in BERT training")
    except Exception as e:
        print(f"   [ERROR] Could not check BERT training: {e}")


def show_integration_features():
    """Show features of each integration."""
    print_section("INTEGRATION FEATURES")

    print("\n### Braintrust Features (Fully Integrated):")
    print("- Experiment tracking with automatic versioning")
    print("- Hyperparameter logging")
    print("- Metric logging (loss, accuracy, per-class metrics)")
    print("- Model artifact tracking")
    print("- Training step logging")
    print("- Offline buffering when disconnected")
    print("- Context manager for easy usage")
    print("- Integrated into RNN training pipeline")

    print("\n### Wandb Features (Basic Support):")
    print("- Standard experiment tracking")
    print("- Metric visualization")
    print("- Hyperparameter tracking")
    print("- Model checkpointing")
    print("- Integration with HuggingFace Trainer")
    print("- Offline mode support")
    print("- Can be enabled in BERT training")

    print("\n### Usage Examples:")
    print("\n1. Braintrust in training:")
    print("   python src/training/train_rnn.py --use_braintrust")

    print("\n2. Wandb with HuggingFace:")
    print("   # Set WANDB_API_KEY environment variable")
    print("   # Modify train_bert_lora.py: report_to='wandb'")


def main():
    """Run all verification tests."""
    print("EXPERIMENT TRACKING INTEGRATION VERIFICATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Verify Braintrust
    braintrust_ok = verify_braintrust()

    # Verify Wandb
    wandb_ok = verify_wandb()

    # Check training integration
    check_training_integration()

    # Show features
    show_integration_features()

    # Summary
    print_section("SUMMARY")

    if braintrust_ok:
        print("[OK] Braintrust is fully configured and working!")
    elif os.environ.get("BRAINTRUST_API_KEY"):
        print("[WARN] Braintrust is installed but connection failed")
    else:
        print("[INFO] Braintrust is installed but not configured")
        print("      Set BRAINTRUST_API_KEY to enable")

    if wandb_ok:
        print("[OK] Wandb is installed and working!")
    else:
        print("[INFO] Wandb is not installed or configured")
        print("      Install with: pip install wandb")

    print("\nRecommendations:")
    if not braintrust_ok and not wandb_ok:
        print("- Consider setting up at least one tracking solution")
        print("- Braintrust is already integrated into the training pipeline")
        print("- Wandb can be easily enabled for HuggingFace training")
    elif braintrust_ok and not wandb_ok:
        print("- Braintrust is ready for experiment tracking")
        print("- Optionally install wandb for additional visualization")
    elif not braintrust_ok and wandb_ok:
        print("- Wandb is available for experiment tracking")
        print("- Consider setting up Braintrust for integrated tracking")
    else:
        print("- Both tracking solutions are available!")
        print("- Use Braintrust for RNN training")
        print("- Use Wandb for BERT/HuggingFace training")

    print("\n" + "=" * 60)

    return 0 if (braintrust_ok or wandb_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
