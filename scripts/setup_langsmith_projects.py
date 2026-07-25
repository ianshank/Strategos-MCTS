"""
Create LangSmith projects for training program.

Usage:
    python scripts/setup_langsmith_projects.py
"""

import os
import sys
from typing import Any

from langsmith import Client


def setup_training_projects() -> None:
    """Create all training projects."""
    # Initialize client
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("❌ Error: LANGSMITH_API_KEY not set")
        print("\nSet your API key:")
        print("  export LANGSMITH_API_KEY='your-key-here'  # Linux/Mac")
        print("  $env:LANGSMITH_API_KEY='your-key-here'   # Windows PowerShell")
        sys.exit(1)

    try:
        client = Client(api_key=api_key, api_url="https://api.smith.langchain.com")
    except Exception as e:
        print(f"❌ Error initializing LangSmith client: {e}")
        sys.exit(1)

    # Project definitions
    projects: list[dict[str, Any]] = [
        {
            "name": "training-2025-module-1",
            "description": "Module 1: System & Architecture Deep Dive",
            "metadata": {"module": 1, "topic": "architecture"},
        },
        {
            "name": "training-2025-module-2",
            "description": "Module 2: Agents Deep Dive (HRM, TRM, MCTS)",
            "metadata": {"module": 2, "topic": "agents"},
        },
        {
            "name": "training-2025-module-3",
            "description": "Module 3: E2E Flows & LangGraph Orchestration",
            "metadata": {"module": 3, "topic": "e2e_flows"},
        },
        {
            "name": "training-2025-module-4",
            "description": "Module 4: LangSmith Tracing Utilities & Patterns",
            "metadata": {"module": 4, "topic": "tracing"},
        },
        {
            "name": "training-2025-module-5",
            "description": "Module 5: Experiments & Datasets in LangSmith",
            "metadata": {"module": 5, "topic": "experiments"},
        },
        {
            "name": "training-2025-module-6",
            "description": "Module 6: 2025 Python Coding & Testing Practices",
            "metadata": {"module": 6, "topic": "python_practices"},
        },
        {
            "name": "training-2025-module-7",
            "description": "Module 7: CI/CD & Observability Integration",
            "metadata": {"module": 7, "topic": "cicd"},
        },
        {
            "name": "training-2025-capstone",
            "description": "Capstone Projects (Week 8)",
            "metadata": {"module": 8, "topic": "capstone"},
        },
    ]

    print("Creating LangSmith training projects...")
    print("=" * 60)

    created = 0
    exists = 0
    failed = 0

    for project_info in projects:
        try:
            # Create project
            client.create_project(
                project_name=project_info["name"],
                description=project_info["description"],
                metadata=project_info["metadata"],
            )
            print(f"✅ Created: {project_info['name']}")
            created += 1
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                print(f"⚠️  Exists:  {project_info['name']}")
                exists += 1
            else:
                print(f"❌ Failed:  {project_info['name']}: {e}")
                failed += 1

    print("=" * 60)
    print("\nSummary:")
    print(f"  Created: {created}")
    print(f"  Already existed: {exists}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(projects)}")

    if failed > 0:
        print(f"\n⚠️  {failed} project(s) failed to create. Check errors above.")
        sys.exit(1)

    print("\n✅ Setup complete! Projects ready for training.")
    print("\nNext steps:")
    print("1. Set LANGSMITH_PROJECT environment variable:")
    print("   export LANGSMITH_PROJECT='training-2025-module-1'")
    print("2. Enable tracing:")
    print("   export LANGSMITH_TRACING_ENABLED='true'")
    print("3. Run verification:")
    print("   python scripts/verification/verify_langsmith_setup.py")
    print("4. Start Module 1:")
    print("   cat docs/training/MODULE_1_ARCHITECTURE.md")


if __name__ == "__main__":
    setup_training_projects()
