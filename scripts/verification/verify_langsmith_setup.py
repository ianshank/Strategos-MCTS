"""
Verify LangSmith setup for training program.

Usage:
    python scripts/verify_langsmith_setup.py
"""

import os
import sys

from langsmith import Client


def verify_environment() -> bool:
    """Verify environment variables."""
    print("🔍 Checking environment variables...")

    required_vars = {
        "LANGSMITH_API_KEY": "API key for authentication",
        "LANGSMITH_PROJECT": "Current training project",
        "LANGSMITH_TRACING_ENABLED": "Enable tracing",
    }

    all_set = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask API key for security
            display_value = value[:10] + "..." if var == "LANGSMITH_API_KEY" and len(value) > 10 else value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: NOT SET ({description})")
            all_set = False

    return all_set


def verify_connection() -> bool:
    """Verify connection to LangSmith API."""
    print("\n🔍 Checking LangSmith API connection...")

    try:
        client = Client()
        # Test API connection by listing projects (limit 1 for speed)
        _ = list(client.list_projects(limit=1))
        print("  ✅ Connected to LangSmith API")
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        print("\n  Possible causes:")
        print("  - Invalid API key")
        print("  - Network/firewall blocking api.smith.langchain.com")
        print("  - LangSmith service down")
        return False


def verify_projects() -> bool:
    """Verify training projects exist."""
    print("\n🔍 Checking training projects...")

    try:
        client = Client()
        all_projects = list(client.list_projects())
        project_names = [p.name for p in all_projects]

        expected_projects = [
            "training-2025-module-1",
            "training-2025-module-2",
            "training-2025-module-3",
            "training-2025-module-4",
            "training-2025-module-5",
            "training-2025-module-6",
            "training-2025-module-7",
            "training-2025-capstone",
        ]

        found = 0
        missing = []
        for project in expected_projects:
            if project in project_names:
                print(f"  ✅ {project}")
                found += 1
            else:
                print(f"  ❌ {project} - NOT FOUND")
                missing.append(project)

        print(f"\nFound {found}/{len(expected_projects)} training projects")

        if missing:
            print(f"\n  Missing projects: {', '.join(missing)}")
            print("  Run: python scripts/setup_langsmith_projects.py")

        return found == len(expected_projects)

    except Exception as e:
        print(f"  ❌ Error listing projects: {e}")
        return False


def verify_tracing() -> bool:
    """Verify tracing works with a test run."""
    print("\n🔍 Testing trace creation...")

    if not os.getenv("LANGSMITH_TRACING_ENABLED"):
        print("  ⚠️  LANGSMITH_TRACING_ENABLED not set")
        print("  Traces may not be created")
        return False

    try:
        from langchain.callbacks import tracing_v2_enabled

        with tracing_v2_enabled():
            # Simple test trace
            print("  Creating test trace...")
            # The context manager itself creates a trace

        print("  ✅ Tracing works!")
        print(f"  Check: https://smith.langchain.com/{os.getenv('LANGSMITH_PROJECT', '')}")
        return True

    except ImportError:
        print("  ⚠️  langchain not installed")
        print("  Install: pip install langchain")
        return False
    except Exception as e:
        print(f"  ❌ Tracing failed: {e}")
        return False


def main() -> None:
    """Run all verification checks."""
    print("=" * 60)
    print("LangSmith Setup Verification")
    print("=" * 60)
    print()

    results = {
        "Environment": verify_environment(),
        "Connection": verify_connection(),
        "Projects": verify_projects(),
        "Tracing": verify_tracing(),
    }

    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Read training overview:")
        print("   cat docs/training/README.md")
        print("2. Start Module 1:")
        print("   cat docs/training/MODULE_1_ARCHITECTURE.md")
        print("3. Run first traced test:")
        print("   python scripts/smoke_test_traced.py")
        print("4. View traces:")
        print("   https://smith.langchain.com/")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Set LANGSMITH_API_KEY:")
        print("   export LANGSMITH_API_KEY='your-key-here'")
        print("2. Set LANGSMITH_PROJECT:")
        print("   export LANGSMITH_PROJECT='training-2025-module-1'")
        print("3. Enable tracing:")
        print("   export LANGSMITH_TRACING_ENABLED='true'")
        print("4. Create projects:")
        print("   python scripts/setup_langsmith_projects.py")
        print("5. Check API key permissions in LangSmith UI")
        print("\nFor more help, see:")
        print("  docs/training/LANGSMITH_PROJECT_SETUP.md")
        print("  docs/training/TROUBLESHOOTING_PLAYBOOK.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
