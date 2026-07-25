#!/usr/bin/env python
"""
Verification script for Continual Learning system installation.

Checks that all components are properly installed and can be imported.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check all imports work."""
    print("Checking imports...")

    try:
        from training.continual_learning import (
            ABTestFramework,
            ActiveLearningCandidate,
            ActiveLearningSelector,
            # Core components
            DataQualityValidator,
            DriftDetector,
            DriftReport,
            FailurePattern,
            FailurePatternAnalyzer,
            # Legacy components
            FeedbackCollector,
            FeedbackSample,
            IncrementalRetrainingPipeline,
            IncrementalTrainer,
            # Data classes
            ProductionInteraction,
            ProductionInteractionLogger,
        )

        # Verify all classes are importable
        classes_to_verify = [
            ABTestFramework,
            ActiveLearningCandidate,
            ActiveLearningSelector,
            DataQualityValidator,
            DriftDetector,
            DriftReport,
            FailurePattern,
            FailurePatternAnalyzer,
            FeedbackCollector,
            FeedbackSample,
            IncrementalRetrainingPipeline,
            IncrementalTrainer,
            ProductionInteraction,
            ProductionInteractionLogger,
        ]
        print(f"✓ All {len(classes_to_verify)} imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def check_config():
    """Check configuration file."""
    print("\nChecking configuration...")

    try:
        import yaml

        with open("training/config.yaml") as f:
            config = yaml.safe_load(f)

        assert "continual_learning" in config, "Missing continual_learning section"
        cl_config = config["continual_learning"]

        required_sections = [
            "logging",
            "failure_analysis",
            "active_learning",
            "retraining",
            "drift_detection",
        ]

        for section in required_sections:
            assert section in cl_config, f"Missing {section} section"

        print("✓ Configuration valid")
        print(f"  Found sections: {', '.join(cl_config.keys())}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def check_files():
    """Check all files exist."""
    print("\nChecking files...")

    files = [
        "training/continual_learning.py",
        "training/tests/test_continual_learning.py",
        "training/examples/continual_learning_demo.py",
        "training/CONTINUAL_LEARNING.md",
        "training/QUICKSTART_CONTINUAL_LEARNING.md",
        "CONTINUAL_LEARNING_SUMMARY.md",
        "training/config.yaml",
    ]

    all_exist = True
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} MISSING")
            all_exist = False

    return all_exist


def check_syntax():
    """Check Python syntax."""
    print("\nChecking Python syntax...")

    import py_compile

    py_files = [
        "training/continual_learning.py",
        "training/tests/test_continual_learning.py",
        "training/examples/continual_learning_demo.py",
    ]

    all_valid = True
    for file_path in py_files:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"  ✓ {file_path}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {file_path}: {e}")
            all_valid = False

    return all_valid


def main():
    """Run all checks."""
    print("=" * 60)
    print("Continual Learning System Verification")
    print("=" * 60)

    results = {
        "imports": check_imports(),
        "config": check_config(),
        "files": check_files(),
        "syntax": check_syntax(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check.capitalize():20s} {status}")

    print("\n" + "=" * 60)

    if all_passed:
        print("✓ All checks passed!")
        print("\nNext steps:")
        print("  1. Run demo: python training/examples/continual_learning_demo.py")
        print("  2. Read quickstart: training/QUICKSTART_CONTINUAL_LEARNING.md")
        print("  3. Run tests: pytest training/tests/test_continual_learning.py")
        return 0
    else:
        print("✗ Some checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
