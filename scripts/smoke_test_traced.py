#!/usr/bin/env python3
"""
LangSmith-traced smoke test runner for MCTS Framework API.

Wraps the bash smoke test scenarios with LangSmith tracing for observability.

Usage:
    python scripts/smoke_test_traced.py [--port PORT] [--api-key API_KEY]

Environment:
    LANGSMITH_TRACING=true - Enable LangSmith tracing
    LANGSMITH_API_KEY - LangSmith API key
    LANGSMITH_PROJECT - Project name (default: langgraph-multi-agent-mcts)
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx  # noqa: E402
from langsmith import traceable  # noqa: E402

from tests.utils.langsmith_tracing import (  # noqa: E402
    get_test_metadata,
    update_run_metadata,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LangSmith-traced smoke tests for MCTS API")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    parser.add_argument(
        "--api-key",
        type=str,
        default="demo-api-key-replace-in-production",
        help="API key for authentication",
    )
    parser.add_argument("--base-url", type=str, help="Override base URL (default: http://localhost:PORT)")
    return parser.parse_args()


class SmokeTestRunner:
    """Smoke test runner with LangSmith tracing."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.Client(timeout=30.0)
        self.passed = 0
        self.failed = 0
        self.results: list[dict[str, Any]] = []

    @traceable(
        run_type="chain",
        name="smoke_test_endpoint",
        tags=["smoke", "api"],
    )
    def test_endpoint(
        self,
        name: str,
        method: str,
        path: str,
        expected_code: int,
        data: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """
        Test a single endpoint with tracing.

        Args:
            name: Test name
            method: HTTP method
            path: Endpoint path
            expected_code: Expected HTTP status code
            data: Request data (for POST)
            headers: Additional headers

        Returns:
            Test result dictionary
        """
        url = f"{self.base_url}{path}"
        request_headers = headers or {}

        print(f"Test: {name}... ", end="", flush=True)

        start_time = datetime.now()

        try:
            if method == "GET":
                response = self.client.get(url, headers=request_headers)
            else:
                response = self.client.post(url, json=data, headers=request_headers)

            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

            passed = response.status_code == expected_code

            result = {
                "test_name": name,
                "method": method,
                "path": path,
                "expected_code": expected_code,
                "actual_code": response.status_code,
                "passed": passed,
                "elapsed_ms": elapsed_ms,
            }

            if passed:
                print(f"PASS (HTTP {response.status_code}, {elapsed_ms:.0f}ms)")
                self.passed += 1
            else:
                print(f"FAIL (Expected {expected_code}, got {response.status_code})")
                self.failed += 1

            self.results.append(result)

            # Update trace with result
            update_run_metadata(
                {
                    "test_name": name,
                    "passed": passed,
                    "status_code": response.status_code,
                    "latency_ms": elapsed_ms,
                }
            )

            return result

        except Exception as e:
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            print(f"ERROR ({str(e)})")
            self.failed += 1

            result = {
                "test_name": name,
                "method": method,
                "path": path,
                "expected_code": expected_code,
                "actual_code": None,
                "passed": False,
                "error": str(e),
                "elapsed_ms": elapsed_ms,
            }

            self.results.append(result)
            return result

    @traceable(
        run_type="chain",
        name="e2e_smoke_test_suite",
        tags=["e2e", "smoke", "api", "full_suite"],
    )
    def run_all_tests(self) -> bool:
        """
        Run all smoke tests with tracing.

        Returns:
            True if all tests passed, False otherwise
        """
        metadata = get_test_metadata()
        metadata["test_suite"] = "smoke"
        metadata["base_url"] = self.base_url
        metadata["start_time"] = datetime.now().isoformat()

        print("=== MCTS Framework API Smoke Tests (with LangSmith Tracing) ===")
        print(f"Target: {self.base_url}")
        print(f"Started at: {datetime.now()}")
        print("")

        # Test 1: Health Check
        self.test_endpoint("Health Check", "GET", "/health", 200)

        # Test 2: Readiness Check
        self.test_endpoint("Readiness Check", "GET", "/ready", 200)

        # Test 3: OpenAPI Docs
        self.test_endpoint("OpenAPI Docs", "GET", "/docs", 200)

        # Test 4: Query with Valid API Key
        self.test_endpoint(
            "Query (Valid Key)",
            "POST",
            "/query",
            200,
            data={"query": "Test tactical scenario", "use_mcts": False},
            headers={"X-API-Key": self.api_key},
        )

        # Test 5: Query with MCTS
        self.test_endpoint(
            "Query (with MCTS)",
            "POST",
            "/query",
            200,
            data={"query": "Analyze defensive positions", "use_mcts": True, "mcts_iterations": 10},
            headers={"X-API-Key": self.api_key},
        )

        # Test 6: Authentication Failure
        self.test_endpoint(
            "Auth Failure",
            "POST",
            "/query",
            401,
            data={"query": "Test"},
            headers={"X-API-Key": "invalid-key"},
        )

        # Test 7: Validation Error (empty query)
        self.test_endpoint(
            "Validation Error",
            "POST",
            "/query",
            422,
            data={"query": ""},
            headers={"X-API-Key": self.api_key},
        )

        # Test 8: Metrics Endpoint
        result = self.test_endpoint("Metrics Endpoint", "GET", "/metrics", 200)
        # Accept 501 if Prometheus not configured
        if result.get("actual_code") == 501:
            print("  (Metrics not configured - acceptable)")

        # Summary
        print("")
        print("=== Results ===")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        print("")

        # Update final trace metadata
        update_run_metadata(
            {
                "total_tests": self.passed + self.failed,
                "tests_passed": self.passed,
                "tests_failed": self.failed,
                "success_rate": self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0,
                "end_time": datetime.now().isoformat(),
            }
        )

        success = self.failed == 0

        if success:
            print("SUCCESS: All smoke tests passed!")
        else:
            print(f"FAILURE: {self.failed} test(s) failed")

        return success

    def __del__(self):
        """Close HTTP client."""
        if hasattr(self, "client"):
            self.client.close()


def main():
    """Main entry point."""
    args = parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}"

    runner = SmokeTestRunner(base_url=base_url, api_key=args.api_key)

    try:
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
