"""
Unified integration verification script for LangGraph Multi-Agent MCTS Framework.

Verifies all optional integrations:
- Pinecone (Vector Storage)
- Braintrust (Experiment Tracking)
- Weights & Biases (Experiment Tracking)

Run this script to check the status of all integrations at once.
"""

import os
import sys
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}")


def print_status(status: str, message: str, details: str = ""):
    """Print a status message with color."""
    if status == "OK":
        color = Colors.GREEN
        symbol = "[OK]"
    elif status == "WARN":
        color = Colors.YELLOW
        symbol = "[!]"
    elif status == "FAIL":
        color = Colors.RED
        symbol = "[FAIL]"
    elif status == "INFO":
        color = Colors.BLUE
        symbol = "[i]"
    else:
        color = ""
        symbol = "[-]"

    print(f"{color}{symbol} {message}{Colors.ENDC}")
    if details:
        print(f"    {details}")


def check_pinecone() -> tuple[bool, bool, dict[str, any]]:
    """Check Pinecone integration status."""
    print_header("PINECONE VECTOR STORAGE")

    results = {"installed": False, "configured": False, "connected": False, "features": []}

    # Check installation
    try:
        import pinecone

        results["installed"] = True
        print_status("OK", "Pinecone package installed", f"Version: {getattr(pinecone, '__version__', 'Unknown')}")
    except ImportError:
        print_status("INFO", "Pinecone not installed", "Install with: pip install pinecone")
        return False, False, results

    # Check configuration
    api_key = os.environ.get("PINECONE_API_KEY")
    host = os.environ.get("PINECONE_HOST")

    if api_key and host:
        results["configured"] = True
        print_status("OK", "Environment variables configured")
        print(f"    API Key: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else '****'}")
        print(f"    Host: {host}")
    else:
        if not api_key:
            print_status("WARN", "PINECONE_API_KEY not set")
        if not host:
            print_status("WARN", "PINECONE_HOST not set")
        return True, False, results

    # Test connection using SDK directly to avoid package import side-effects
    try:
        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(host=host)
        stats = index.describe_index_stats()
        results["connected"] = True
        print_status("OK", "Connected to Pinecone successfully")
        print(f"    Total vectors: {stats.get('total_vector_count', stats.get('total_vectors', 0))}")
        print(f"    Dimension: {stats.get('dimension', 'Unknown')}")

        results["features"] = [
            "10-dimensional feature vectors",
            "Semantic similarity search",
            "Agent distribution analysis",
            "Offline operation buffering",
            "Namespace isolation",
        ]
    except Exception as e:
        print_status("FAIL", f"Error testing connection: {type(e).__name__}", str(e))

    return results["installed"], results["configured"] and results["connected"], results


def check_braintrust() -> tuple[bool, bool, dict[str, any]]:
    """Check Braintrust integration status."""
    print_header("BRAINTRUST EXPERIMENT TRACKING")

    results = {"installed": False, "configured": False, "connected": False, "features": []}

    # Check installation
    try:
        import braintrust

        results["installed"] = True
        print_status("OK", "Braintrust package installed", f"Version: {getattr(braintrust, '__version__', 'Unknown')}")
    except ImportError:
        print_status("INFO", "Braintrust not installed", "Install with: pip install braintrust")
        return False, False, results

    # Check configuration
    api_key = os.environ.get("BRAINTRUST_API_KEY")

    if api_key:
        results["configured"] = True
        print_status("OK", "API key configured", f"Key: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else '****'}")
    else:
        print_status("WARN", "BRAINTRUST_API_KEY not set")
        print("    Get your API key from: https://www.braintrust.dev/")
        return True, False, results

    # Test connection using SDK directly to avoid package import side-effects
    try:
        braintrust.login(api_key=api_key)
        exp = braintrust.init(project="verification", experiment="verification_smoke")
        exp.log(input={"ping": True}, output={"pong": True}, scores={"ok": 1.0})
        results["connected"] = True
        print_status("OK", "Connected to Braintrust successfully")

        results["features"] = [
            "Experiment versioning",
            "Hyperparameter tracking",
            "Metric logging (loss, accuracy)",
            "Model artifact tracking",
            "Offline buffering",
            "RNN training integration",
        ]
    except Exception as e:
        print_status("FAIL", f"Error testing connection: {type(e).__name__}", str(e))

    # Check training integration
    try:
        with open("src/training/train_rnn.py") as f:
            if "BraintrustTracker" in f.read():
                print_status("OK", "Integrated in RNN training script", "Use --use_braintrust flag")
    except OSError:
        pass  # File may not exist in all environments

    return results["installed"], results["configured"] and results["connected"], results


def check_wandb() -> tuple[bool, bool, dict[str, any]]:
    """Check Weights & Biases integration status."""
    print_header("WEIGHTS & BIASES (WANDB)")

    results = {"installed": False, "configured": False, "connected": False, "features": []}

    # Check installation
    try:
        import wandb

        results["installed"] = True
        print_status("OK", "Wandb package installed", f"Version: {wandb.__version__}")
    except ImportError:
        print_status("INFO", "Wandb not installed", "Install with: pip install wandb")
        return False, False, results

    # Check configuration
    api_key = os.environ.get("WANDB_API_KEY")

    if api_key:
        results["configured"] = True
        print_status("OK", "API key configured", f"Key: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else '****'}")
    else:
        print_status("WARN", "WANDB_API_KEY not set")
        print("    Get your API key from: https://wandb.ai/settings")
        # Wandb can work without API key in offline mode
        results["configured"] = True  # Partial configuration

    # Test functionality (offline mode)
    try:
        import wandb

        print_status("INFO", "Testing in offline mode...")

        # Suppress wandb output temporarily
        import io
        from contextlib import redirect_stderr, redirect_stdout

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _run = wandb.init(  # noqa: F841 - run object managed by wandb.finish()
                project="verification-test", mode="offline", config={"test": True}
            )
            wandb.log({"test_metric": 1.0})
            wandb.finish()

        results["connected"] = True
        print_status("OK", "Wandb functional in offline mode")
        print("    Sync later with: wandb sync [run_directory]")

        results["features"] = [
            "Experiment tracking",
            "Metric visualization",
            "HuggingFace integration",
            "Model checkpointing",
            "Offline mode support",
            "Rich web UI",
        ]
    except Exception as e:
        print_status("FAIL", f"Error testing wandb: {type(e).__name__}")

    # Check training integration
    try:
        with open("src/training/train_bert_lora.py") as f:
            if "wandb" in f.read().lower():
                print_status("OK", "Compatible with BERT training", "Set report_to='wandb' in TrainingArguments")
    except OSError:
        pass  # File may not exist in all environments

    return results["installed"], results["configured"], results


def show_quick_start(pinecone_status, braintrust_status, wandb_status):
    """Show quick start commands based on status."""
    print_header("QUICK START GUIDE")

    if not all(s[0] for s in [pinecone_status, braintrust_status, wandb_status]):
        print(f"\n{Colors.BOLD}1. Install missing packages:{Colors.ENDC}")
        packages = []
        if not pinecone_status[0]:
            packages.append("pinecone-client")
        if not braintrust_status[0]:
            packages.append("braintrust")
        if not wandb_status[0]:
            packages.append("wandb")

        if packages:
            print(f"   pip install {' '.join(packages)}")

    if not all(s[1] for s in [pinecone_status, braintrust_status, wandb_status]):
        print(f"\n{Colors.BOLD}2. Configure API keys:{Colors.ENDC}")

        if not pinecone_status[1] and pinecone_status[0]:
            print("\n   # Pinecone")
            print("   export PINECONE_API_KEY='your-key'")
            print("   export PINECONE_HOST='https://your-index.svc.region.pinecone.io'")

        if not braintrust_status[1] and braintrust_status[0]:
            print("\n   # Braintrust")
            print("   export BRAINTRUST_API_KEY='your-key'")

        if not wandb_status[1] and wandb_status[0]:
            print("\n   # Wandb (optional)")
            print("   export WANDB_API_KEY='your-key'")

    print(f"\n{Colors.BOLD}3. Example usage:{Colors.ENDC}")

    if braintrust_status[1]:
        print("\n   # Train RNN with experiment tracking")
        print("   python src/training/train_rnn.py --use_braintrust")

    if wandb_status[0]:
        print("\n   # Use Wandb with BERT training")
        print("   # Edit train_bert_lora.py: report_to='wandb'")

    if pinecone_status[1]:
        print("\n   # Vector storage is automatically used by Meta-Controller")
        print("   # when Pinecone is configured")


def main():
    """Run all integration checks."""
    print(f"{Colors.BOLD}LANGGRAPH MULTI-AGENT MCTS - INTEGRATION VERIFICATION{Colors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check all integrations
    pinecone_status = check_pinecone()
    braintrust_status = check_braintrust()
    wandb_status = check_wandb()

    # Summary
    print_header("INTEGRATION SUMMARY")

    # Create summary table
    integrations = [
        ("Pinecone", pinecone_status, "Vector storage for agent decisions"),
        ("Braintrust", braintrust_status, "Experiment tracking (RNN)"),
        ("Wandb", wandb_status, "Experiment tracking (BERT/General)"),
    ]

    print(f"\n{'Integration':<15} {'Installed':<12} {'Configured':<12} {'Purpose':<35}")
    print("-" * 75)

    for name, (installed, configured, _results), purpose in integrations:
        inst_status = f"{Colors.GREEN}Yes{Colors.ENDC}" if installed else f"{Colors.YELLOW}No{Colors.ENDC}"
        conf_status = f"{Colors.GREEN}Yes{Colors.ENDC}" if configured else f"{Colors.YELLOW}No{Colors.ENDC}"
        print(f"{name:<15} {inst_status:<21} {conf_status:<21} {purpose:<35}")

    # Show features if available
    print_header("AVAILABLE FEATURES")

    for name, (_installed, _configured, results) in [
        ("Pinecone", pinecone_status),
        ("Braintrust", braintrust_status),
        ("Wandb", wandb_status),
    ]:
        if results.get("features"):
            print(f"\n{Colors.BOLD}{name}:{Colors.ENDC}")
            for feature in results["features"]:
                print(f"  - {feature}")

    # Quick start guide
    show_quick_start(pinecone_status, braintrust_status, wandb_status)

    # Final recommendations
    print_header("RECOMMENDATIONS")

    all_configured = all(s[1] for s in [pinecone_status, braintrust_status, wandb_status])
    any_configured = any(s[1] for s in [pinecone_status, braintrust_status, wandb_status])

    if all_configured:
        print_status("OK", "All integrations are configured and ready to use!")
        print("\nYou have access to:")
        print("  - Vector storage for improved agent routing (Pinecone)")
        print("  - Experiment tracking for RNN training (Braintrust)")
        print("  - General experiment tracking with rich UI (Wandb)")
    elif any_configured:
        print("Some integrations are ready:")
        if pinecone_status[1]:
            print_status("OK", "Pinecone vector storage is ready")
        if braintrust_status[1]:
            print_status("OK", "Braintrust experiment tracking is ready")
        if wandb_status[1]:
            print_status("OK", "Wandb experiment tracking is ready")
        print("\nConsider setting up the remaining integrations for full functionality.")
    else:
        print("No integrations are currently configured.")
        print("\nThe framework will work without these integrations, but you'll miss:")
        print("  - Vector-based agent routing improvements")
        print("  - Experiment tracking and comparison")
        print("  - Training visualization and monitoring")
        print("\nConsider setting up at least one integration.")

    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")

    # Return status code
    return 0 if any_configured else 1


if __name__ == "__main__":
    # Disable color on Windows if not supported
    if sys.platform == "win32":
        try:
            import colorama

            colorama.init()
        except ImportError:
            # Remove color codes if colorama not available
            for attr in dir(Colors):
                if not attr.startswith("_"):
                    setattr(Colors, attr, "")

    sys.exit(main())
