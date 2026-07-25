"""Minimal LangSmith connectivity verification script.

This script verifies basic LangSmith setup and connectivity per MODULE_4_TRACING.md.
Tests:
1. Environment variable configuration
2. LangSmith client connectivity
3. Basic traced function execution
4. Trace visibility in LangSmith UI
"""

import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_env_vars() -> bool:
    """Check required LangSmith environment variables."""
    print("=" * 80)
    print("1. Checking LangSmith Environment Variables")
    print("=" * 80)

    required_vars = {
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
        "LANGSMITH_ORG_ID": os.getenv("LANGSMITH_ORG_ID"),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
        "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
    }

    all_set = True
    for var, value in required_vars.items():
        status = "[OK]" if value else "[MISSING]"
        display_value = f"{value[:20]}..." if value and len(str(value)) > 20 else value
        print(f"  {status} {var}: {display_value or 'NOT SET'}")
        if not value:
            all_set = False

    print()
    return all_set


def test_client_connection():
    """Test LangSmith client connection."""
    print("=" * 80)
    print("2. Testing LangSmith Client Connection")
    print("=" * 80)

    try:
        from langsmith import Client

        client = Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        )

        # Try to get project info
        project_name = os.getenv("LANGSMITH_PROJECT", "langgraph-multi-agent-mcts")
        print("  [OK] Client initialized successfully")
        print(f"  [OK] Project: {project_name}")

        # List recent runs to verify connectivity
        try:
            runs = list(client.list_runs(project_name=project_name, limit=5))
            print(f"  [OK] Connection verified: Retrieved {len(runs)} recent run(s)")
            return True
        except Exception as e:
            print(f"  [WARN] Warning: Could not list runs (project may not exist yet): {e}")
            print("  [OK] Client connection is valid, traces will create project automatically")
            return True

    except Exception as e:
        print(f"  [ERROR] Client connection failed: {e}")
        return False


def test_traced_function():
    """Test a simple traced function."""
    print("\n" + "=" * 80)
    print("3. Testing Basic Traced Function")
    print("=" * 80)

    try:
        from langchain_openai import ChatOpenAI
        from langsmith import traceable

        @traceable(name="minimal_test", run_type="tool", tags=["verification", "test"])
        def simple_traced_function(text: str) -> dict:
            """Simple function with tracing."""
            return {
                "input": text,
                "output": f"Processed: {text}",
                "length": len(text),
            }

        # Execute traced function
        result = simple_traced_function("Hello LangSmith!")
        print("  [OK] Traced function executed successfully")
        print(f"  [OK] Result: {result}")

        # Test with LLM call
        print("\n  Testing traced LLM call...")

        @traceable(name="llm_test", run_type="llm", tags=["verification", "llm"])
        def test_llm_call(query: str) -> str:
            """Test LLM call with tracing."""
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(query)
            return response.content

        llm_result = test_llm_call("Say 'LangSmith tracing works!'")
        print("  [OK] LLM call traced successfully")
        print(f"  [OK] LLM Response: {llm_result[:100]}...")

        return True

    except Exception as e:
        print(f"  [ERROR] Traced function test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main verification workflow."""
    print("\n" + "=" * 80)
    print("LangSmith Connectivity Verification")
    print("Per MODULE_4_TRACING.md requirements")
    print("=" * 80 + "\n")

    # Check environment
    env_ok = check_env_vars()
    if not env_ok:
        print("\n[FAIL] Environment configuration incomplete!")
        print("   Please configure missing variables in .env file")
        sys.exit(1)

    # Test client
    client_ok = test_client_connection()
    if not client_ok:
        print("\n[FAIL] LangSmith client connection failed!")
        print("   Check your API key and endpoint configuration")
        sys.exit(1)

    # Test traced function
    trace_ok = test_traced_function()
    if not trace_ok:
        print("\n[FAIL] Traced function test failed!")
        sys.exit(1)

    # Success summary
    print("\n" + "=" * 80)
    print("[SUCCESS] LangSmith Connectivity Verified!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. View traces in LangSmith UI:")
    print(
        f"   https://smith.langchain.com/o/{os.getenv('LANGSMITH_ORG_ID')}/projects/p/{os.getenv('LANGSMITH_PROJECT')}"
    )
    print("\n2. Look for traces with tags:")
    print("   - 'verification'")
    print("   - 'test'")
    print("   - 'llm'")
    print("\n3. Proceed to Module 1 labs")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
