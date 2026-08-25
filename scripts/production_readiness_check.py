#!/usr/bin/env python3
"""
Production Readiness Check Script for LangGraph Multi-Agent MCTS Framework.

This script verifies all critical components are in place before production deployment.
Run this before deploying to ensure compliance with production requirements.

Usage:
    python scripts/production_readiness_check.py
    python scripts/production_readiness_check.py --verbose
    python scripts/production_readiness_check.py --json-output
"""

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import sys


class Status(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    SKIP = "⏭️  SKIP"


class Priority(Enum):
    P0 = "P0-CRITICAL"
    P1 = "P1-HIGH"
    P2 = "P2-MEDIUM"
    P3 = "P3-LOW"


@dataclass
class CheckResult:
    name: str
    status: Status
    priority: Priority
    message: str
    details: str | None = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists."""
    return path.exists() and path.is_file()


def check_dir_exists(path: Path, description: str) -> bool:
    """Check if a directory exists."""
    return path.exists() and path.is_dir()


def run_security_checks(root: Path) -> list[CheckResult]:
    """Run security-related checks."""
    results = []

    # Check 1: Validation models integrated
    val_file = root / "src" / "models" / "validation.py"
    app_file = root / "app.py"
    builder_file = root / "src" / "framework" / "graph" / "builder.py"
    legacy_file = root / "langgraph_multi_agent_mcts.py"

    if (val_file.exists() and (app_file.exists() or builder_file.exists())) or legacy_file.exists():
        has_validation = False
        if val_file.exists():
            val_content = val_file.read_text(encoding="utf-8", errors="ignore")
            if "validate_query" in val_content:
                has_validation = True
        elif legacy_file.exists():
            leg_content = legacy_file.read_text(encoding="utf-8", errors="ignore")
            if "validate_query" in leg_content:
                has_validation = True

        if has_validation:
            results.append(
                CheckResult(
                    name="Validation Models Integrated",
                    status=Status.PASS,
                    priority=Priority.P0,
                    message="Input validation is integrated into main framework",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Validation Models Integrated",
                    status=Status.FAIL,
                    priority=Priority.P0,
                    message="Validation models NOT integrated into main framework",
                    details="Check src/models/validation.py for validation definitions",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Validation Models Integrated",
                status=Status.FAIL,
                priority=Priority.P0,
                message="Framework validation files not found",
            )
        )

    # Check 2: Authentication layer exists
    auth_file = root / "src" / "api" / "auth.py"
    if check_file_exists(auth_file, "Authentication module"):
        content = auth_file.read_text()
        if "APIKeyAuthenticator" in content and "RateLimitError" in content:
            results.append(
                CheckResult(
                    name="Authentication Layer",
                    status=Status.PASS,
                    priority=Priority.P0,
                    message="Authentication and rate limiting implemented",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Authentication Layer",
                    status=Status.WARN,
                    priority=Priority.P0,
                    message="Authentication module incomplete",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Authentication Layer",
                status=Status.FAIL,
                priority=Priority.P0,
                message="Authentication module NOT found at src/api/auth.py",
            )
        )

    # Check 3: Custom exceptions
    exceptions_file = root / "src" / "api" / "exceptions.py"
    if check_file_exists(exceptions_file, "Custom exceptions"):
        content = exceptions_file.read_text()
        if "sanitize_details" in content and "FrameworkError" in content:
            results.append(
                CheckResult(
                    name="Error Sanitization",
                    status=Status.PASS,
                    priority=Priority.P0,
                    message="Custom exceptions with error sanitization implemented",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Error Sanitization",
                    status=Status.WARN,
                    priority=Priority.P0,
                    message="Exception module exists but may lack sanitization",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Error Sanitization",
                status=Status.FAIL,
                priority=Priority.P0,
                message="Custom exceptions module NOT found",
            )
        )

    # Check 4: No hardcoded secrets
    env_example = root / ".env.example"
    if check_file_exists(env_example, "Environment example"):
        results.append(
            CheckResult(
                name="Environment Configuration",
                status=Status.PASS,
                priority=Priority.P0,
                message=".env.example file exists for secure configuration",
            )
        )
    else:
        results.append(
            CheckResult(
                name="Environment Configuration",
                status=Status.WARN,
                priority=Priority.P0,
                message=".env.example not found",
            )
        )

    return results


def run_infrastructure_checks(root: Path) -> list[CheckResult]:
    """Run infrastructure-related checks."""
    results = []

    # Check 1: Dockerfile exists
    dockerfile = root / "Dockerfile"
    if check_file_exists(dockerfile, "Dockerfile"):
        content = dockerfile.read_text()
        multi_stage = "FROM" in content and content.count("FROM") >= 2
        non_root = "USER" in content and "root" not in content.split("USER")[-1][:20]

        if multi_stage and non_root:
            results.append(
                CheckResult(
                    name="Containerization",
                    status=Status.PASS,
                    priority=Priority.P0,
                    message="Multi-stage Dockerfile with non-root user",
                )
            )
        elif multi_stage:
            results.append(
                CheckResult(
                    name="Containerization",
                    status=Status.WARN,
                    priority=Priority.P0,
                    message="Dockerfile exists but may not use non-root user",
                    details="Consider adding USER directive for security",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Containerization",
                    status=Status.WARN,
                    priority=Priority.P0,
                    message="Dockerfile exists but not multi-stage",
                    details="Consider multi-stage build for smaller images",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Containerization", status=Status.FAIL, priority=Priority.P0, message="Dockerfile NOT found"
            )
        )

    # Check 2: Docker Compose
    docker_compose = root / "docker-compose.yml"
    if check_file_exists(docker_compose, "Docker Compose"):
        content = docker_compose.read_text()
        has_monitoring = "prometheus" in content.lower() and "grafana" in content.lower()
        if has_monitoring:
            results.append(
                CheckResult(
                    name="Docker Compose Setup",
                    status=Status.PASS,
                    priority=Priority.P0,
                    message="Docker Compose with monitoring stack configured",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Docker Compose Setup",
                    status=Status.WARN,
                    priority=Priority.P0,
                    message="Docker Compose exists but may lack monitoring",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Docker Compose Setup",
                status=Status.FAIL,
                priority=Priority.P0,
                message="docker-compose.yml NOT found",
            )
        )

    # Check 3: Kubernetes manifests
    k8s_dir = root / "kubernetes"
    if check_dir_exists(k8s_dir, "Kubernetes directory"):
        deployment_file = k8s_dir / "deployment.yaml"
        if check_file_exists(deployment_file, "K8s deployment"):
            results.append(
                CheckResult(
                    name="Kubernetes Deployment",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message="Kubernetes deployment manifests configured",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Kubernetes Deployment",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message="Kubernetes directory exists but no deployment.yaml",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Kubernetes Deployment",
                status=Status.WARN,
                priority=Priority.P1,
                message="Kubernetes manifests not found (optional for non-K8s deployments)",
            )
        )

    # Check 4: Monitoring configuration
    monitoring_dir = root / "monitoring"
    if check_dir_exists(monitoring_dir, "Monitoring directory"):
        prometheus_config = monitoring_dir / "prometheus.yml"
        alerts_config = monitoring_dir / "alerts.yml"

        if check_file_exists(prometheus_config, "Prometheus config") and check_file_exists(
            alerts_config, "Alerts config"
        ):
            results.append(
                CheckResult(
                    name="Monitoring Configuration",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message="Prometheus and alerting configured",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Monitoring Configuration",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message="Monitoring directory exists but configs incomplete",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Monitoring Configuration",
                status=Status.FAIL,
                priority=Priority.P1,
                message="Monitoring configuration NOT found",
            )
        )

    return results


def run_testing_checks(root: Path) -> list[CheckResult]:
    """Run testing-related checks."""
    results = []

    tests_dir = root / "tests"

    # Check 1: Unit tests exist
    unit_tests = tests_dir / "unit"
    if check_dir_exists(unit_tests, "Unit tests directory"):
        test_files = list(unit_tests.glob("test_*.py"))
        if len(test_files) >= 2:
            results.append(
                CheckResult(
                    name="Unit Tests",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message=f"{len(test_files)} unit test files found",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Unit Tests",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message=f"Only {len(test_files)} unit test files found",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Unit Tests", status=Status.FAIL, priority=Priority.P1, message="Unit tests directory NOT found"
            )
        )

    # Check 2: Performance tests
    perf_tests = tests_dir / "performance"
    if check_dir_exists(perf_tests, "Performance tests"):
        test_files = list(perf_tests.glob("test_*.py"))
        if test_files:
            results.append(
                CheckResult(
                    name="Performance Tests",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message=f"{len(test_files)} performance test files found",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Performance Tests",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message="Performance test directory exists but no tests",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Performance Tests",
                status=Status.FAIL,
                priority=Priority.P1,
                message="Performance tests NOT found",
            )
        )

    # Check 3: Chaos tests
    chaos_tests = tests_dir / "chaos"
    if check_dir_exists(chaos_tests, "Chaos tests"):
        results.append(
            CheckResult(
                name="Chaos Engineering Tests",
                status=Status.PASS,
                priority=Priority.P1,
                message="Chaos engineering tests configured",
            )
        )
    else:
        results.append(
            CheckResult(
                name="Chaos Engineering Tests",
                status=Status.WARN,
                priority=Priority.P1,
                message="Chaos tests not found (recommended for resilience)",
            )
        )

    return results


def run_documentation_checks(root: Path) -> list[CheckResult]:
    """Run documentation-related checks."""
    results = []

    docs_dir = root / "docs"

    # Check 1: SLA documentation
    sla_doc = docs_dir / "SLA.md"
    if check_file_exists(sla_doc, "SLA documentation"):
        results.append(
            CheckResult(
                name="SLA Documentation",
                status=Status.PASS,
                priority=Priority.P2,
                message="Service Level Agreement documented",
            )
        )
    else:
        results.append(
            CheckResult(
                name="SLA Documentation",
                status=Status.WARN,
                priority=Priority.P2,
                message="SLA documentation not found",
            )
        )

    # Check 2: Runbooks
    runbooks_dir = docs_dir / "runbooks"
    if check_dir_exists(runbooks_dir, "Runbooks directory"):
        runbook_files = list(runbooks_dir.glob("*.md"))
        if runbook_files:
            results.append(
                CheckResult(
                    name="Operational Runbooks",
                    status=Status.PASS,
                    priority=Priority.P2,
                    message=f"{len(runbook_files)} runbook(s) documented",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Operational Runbooks",
                    status=Status.WARN,
                    priority=Priority.P2,
                    message="Runbooks directory exists but empty",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Operational Runbooks",
                status=Status.WARN,
                priority=Priority.P2,
                message="Operational runbooks not found",
            )
        )

    # Check 3: API documentation (FastAPI auto-generates)
    api_server = root / "src" / "api" / "rest_server.py"
    if check_file_exists(api_server, "REST API server"):
        content = api_server.read_text()
        if "FastAPI" in content and "/docs" in content:
            results.append(
                CheckResult(
                    name="API Documentation",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message="OpenAPI/Swagger documentation auto-generated via FastAPI",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="API Documentation",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message="API server exists but may lack documentation",
                )
            )
    else:
        results.append(
            CheckResult(
                name="API Documentation", status=Status.FAIL, priority=Priority.P1, message="REST API server NOT found"
            )
        )

    return results


def run_dependency_checks(root: Path) -> list[CheckResult]:
    """Run dependency management checks."""
    results = []

    # Accepted specifiers for production pinning: ``==`` is ideal; ``~=`` and
    # upper-bounded ranges (``>=X,<Y``) are acceptable.  Bare ``>=`` without an
    # upper cap is risky in production.

    # Check 1: Dependencies pinned
    requirements = root / "requirements.txt"
    if check_file_exists(requirements, "Requirements file"):
        content = requirements.read_text()
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
        pinned = sum(1 for line in lines if "==" in line)
        unbounded = sum(1 for line in lines if not any(op in line for op in ("==", ">=", "<=", "~=", "<", ">", "!=")))

        if unbounded > 0:
            results.append(
                CheckResult(
                    name="Dependencies Pinned",
                    status=Status.FAIL,
                    priority=Priority.P0,
                    message=f"{unbounded} dependencies have no version constraint",
                    details="Pin all production dependencies to avoid supply-chain drift",
                )
            )
        elif pinned == len(lines):
            results.append(
                CheckResult(
                    name="Dependencies Pinned",
                    status=Status.PASS,
                    priority=Priority.P0,
                    message=f"All {pinned} dependencies are pinned to specific versions",
                )
            )
        else:
            range_only = len(lines) - pinned
            results.append(
                CheckResult(
                    name="Dependencies Pinned",
                    status=Status.WARN,
                    priority=Priority.P0,
                    message=(
                        f"{len(lines)} dependencies version-bounded " f"({pinned} pinned, {range_only} range-bounded)"
                    ),
                    details="Consider pinning range-bounded deps with == for reproducibility",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Dependencies Pinned",
                status=Status.FAIL,
                priority=Priority.P0,
                message="requirements.txt NOT found",
            )
        )

    return results


def run_observability_checks(root: Path) -> list[CheckResult]:
    """Run observability and monitoring checks."""
    results = []

    # Check 1: Prometheus metrics module
    metrics_file = root / "src" / "monitoring" / "prometheus_metrics.py"
    if check_file_exists(metrics_file, "Prometheus metrics"):
        content = metrics_file.read_text()
        if "Counter" in content and "Histogram" in content and "Gauge" in content:
            results.append(
                CheckResult(
                    name="Prometheus Metrics",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message="Prometheus metrics module implemented",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Prometheus Metrics",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message="Metrics module exists but may be incomplete",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Prometheus Metrics",
                status=Status.FAIL,
                priority=Priority.P1,
                message="Prometheus metrics module NOT found",
                details="Create src/monitoring/prometheus_metrics.py",
            )
        )

    # Check 2: OpenTelemetry tracing
    tracing_file = root / "src" / "monitoring" / "otel_tracing.py"
    if check_file_exists(tracing_file, "OpenTelemetry tracing"):
        content = tracing_file.read_text()
        if "setup_tracing" in content and "trace_operation" in content:
            results.append(
                CheckResult(
                    name="OpenTelemetry Tracing",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message="Distributed tracing configured",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="OpenTelemetry Tracing",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message="Tracing module exists but may be incomplete",
                )
            )
    else:
        results.append(
            CheckResult(
                name="OpenTelemetry Tracing",
                status=Status.WARN,
                priority=Priority.P2,
                message="OpenTelemetry tracing not configured (optional)",
                details="Consider adding src/monitoring/otel_tracing.py",
            )
        )

    # Check 3: Grafana dashboards
    dashboards_dir = root / "monitoring" / "grafana" / "dashboards"
    if check_dir_exists(dashboards_dir, "Grafana dashboards"):
        dashboard_files = list(dashboards_dir.glob("*.json"))
        if len(dashboard_files) >= 2:
            results.append(
                CheckResult(
                    name="Grafana Dashboards",
                    status=Status.PASS,
                    priority=Priority.P2,
                    message=f"{len(dashboard_files)} Grafana dashboard(s) configured",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Grafana Dashboards",
                    status=Status.WARN,
                    priority=Priority.P2,
                    message=f"Only {len(dashboard_files)} dashboard(s) found",
                    details="Add more dashboards for comprehensive monitoring",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Grafana Dashboards",
                status=Status.WARN,
                priority=Priority.P2,
                message="Grafana dashboards not configured",
                details="Create monitoring/grafana/dashboards/",
            )
        )

    # Check 4: Alert rules
    alerts_file = root / "monitoring" / "alerts.yml"
    if check_file_exists(alerts_file, "Prometheus alerts"):
        content = alerts_file.read_text()
        alert_count = content.count("alert:")
        if alert_count >= 5:
            results.append(
                CheckResult(
                    name="Alert Rules",
                    status=Status.PASS,
                    priority=Priority.P1,
                    message=f"{alert_count} alert rules configured",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Alert Rules",
                    status=Status.WARN,
                    priority=Priority.P1,
                    message=f"Only {alert_count} alert rules found",
                    details="Add more comprehensive alerting",
                )
            )
    else:
        results.append(
            CheckResult(
                name="Alert Rules",
                status=Status.FAIL,
                priority=Priority.P1,
                message="Alert rules NOT configured",
                details="Create monitoring/alerts.yml",
            )
        )

    return results


def run_cicd_checks(root: Path) -> list[CheckResult]:
    """Run CI/CD pipeline checks."""
    results = []

    # Check 1: CI workflow exists
    ci_workflow = root / ".github" / "workflows" / "ci.yml"
    if check_file_exists(ci_workflow, "CI workflow"):
        content = ci_workflow.read_text()
        has_lint = "lint" in content.lower() or "ruff" in content.lower()
        has_test = "test" in content.lower() or "pytest" in content.lower()
        has_security = "bandit" in content.lower() or "security" in content.lower()
        has_docker = "docker" in content.lower()

        checks = {
            "linting": has_lint,
            "testing": has_test,
            "security_scan": has_security,
            "docker_build": has_docker,
        }

        passed = sum(checks.values())
        total = len(checks)

        if passed == total:
            results.append(
                CheckResult(
                    name="CI Pipeline",
                    status=Status.PASS,
                    priority=Priority.P0,
                    message=f"Comprehensive CI pipeline ({passed}/{total} checks)",
                )
            )
        elif passed >= total * 0.75:
            results.append(
                CheckResult(
                    name="CI Pipeline",
                    status=Status.WARN,
                    priority=Priority.P0,
                    message=f"CI pipeline partially complete ({passed}/{total} checks)",
                    details=f"Missing: {', '.join(k for k, v in checks.items() if not v)}",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="CI Pipeline",
                    status=Status.FAIL,
                    priority=Priority.P0,
                    message=f"CI pipeline incomplete ({passed}/{total} checks)",
                    details=f"Missing: {', '.join(k for k, v in checks.items() if not v)}",
                )
            )
    else:
        results.append(
            CheckResult(
                name="CI Pipeline",
                status=Status.FAIL,
                priority=Priority.P0,
                message="CI workflow NOT found",
                details="Create .github/workflows/ci.yml",
            )
        )

    # Check 2: Production readiness script exists
    readiness_script = root / "scripts" / "production_readiness_check.py"
    if check_file_exists(readiness_script, "Production readiness script"):
        results.append(
            CheckResult(
                name="Production Readiness Script",
                status=Status.PASS,
                priority=Priority.P1,
                message="Production readiness check script exists",
            )
        )
    else:
        results.append(
            CheckResult(
                name="Production Readiness Script",
                status=Status.WARN,
                priority=Priority.P1,
                message="Production readiness script not found",
            )
        )

    return results


def generate_report(results: list[CheckResult], verbose: bool = False) -> int:
    """Generate and print the readiness report."""
    print("\n" + "=" * 70)
    print(" 🚀 PRODUCTION READINESS REPORT - LangGraph Multi-Agent MCTS")
    print("=" * 70)
    print(f" Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 70 + "\n")

    # Group by priority
    by_priority = {p: [] for p in Priority}
    for result in results:
        by_priority[result.priority].append(result)

    # Print by priority
    for priority in Priority:
        if by_priority[priority]:
            print(f"\n{priority.value}:")
            print("-" * 70)
            for check in by_priority[priority]:
                print(f"  {check.status.value} {check.name}")
                print(f"      {check.message}")
                if verbose and check.details:
                    print(f"      Details: {check.details}")
            print()

    # Summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.status == Status.PASS)
    failed = sum(1 for r in results if r.status == Status.FAIL)
    warnings = sum(1 for r in results if r.status == Status.WARN)

    p0_failed = sum(1 for r in results if r.status == Status.FAIL and r.priority == Priority.P0)

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f" Total Checks: {total}")
    print(f" ✅ Passed: {passed}")
    print(f" ❌ Failed: {failed}")
    print(f" ⚠️  Warnings: {warnings}")
    print(f"\n Readiness Score: {passed}/{total} ({100 * passed / total:.1f}%)")
    print("=" * 70 + "\n")

    # Final verdict
    if p0_failed > 0:
        print(f"❌ BLOCKED: {p0_failed} critical (P0) check(s) failed")
        print("   These MUST be resolved before production deployment.\n")
        return 1
    elif failed > 0:
        print(f"⚠️  WARNING: {failed} non-critical check(s) failed")
        print("   Consider fixing these before deployment.\n")
        return 0
    elif warnings > 0:
        print(f"✅ READY with {warnings} warning(s)")
        print("   System is production-ready but improvements recommended.\n")
        return 0
    else:
        print("✅ FULLY READY FOR PRODUCTION")
        print("   All checks passed. System is production-ready.\n")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Check production readiness of MCTS Framework")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information for each check")
    parser.add_argument("--json-output", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Fix Windows console encoding
    if sys.platform == "win32":
        with contextlib.suppress(AttributeError):
            sys.stdout.reconfigure(encoding="utf-8")

    root = get_project_root()
    all_results = []

    # Run all checks
    print("Running production readiness checks...\n")

    print("[SECURITY] Security checks...")
    all_results.extend(run_security_checks(root))

    print("[INFRA] Infrastructure checks...")
    all_results.extend(run_infrastructure_checks(root))

    print("[TESTS] Testing checks...")
    all_results.extend(run_testing_checks(root))

    print("[DOCS] Documentation checks...")
    all_results.extend(run_documentation_checks(root))

    print("[DEPS] Dependency checks...")
    all_results.extend(run_dependency_checks(root))

    print("[OBSERVABILITY] Observability checks...")
    all_results.extend(run_observability_checks(root))

    print("[CI/CD] CI/CD pipeline checks...")
    all_results.extend(run_cicd_checks(root))

    # Output results
    if args.json_output:
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "status": r.status.name,
                    "priority": r.priority.name,
                    "message": r.message,
                    "details": r.details,
                }
                for r in all_results
            ],
            "summary": {
                "total": len(all_results),
                "passed": sum(1 for r in all_results if r.status == Status.PASS),
                "failed": sum(1 for r in all_results if r.status == Status.FAIL),
                "warnings": sum(1 for r in all_results if r.status == Status.WARN),
            },
        }
        print(json.dumps(output, indent=2))
        return 0
    else:
        return generate_report(all_results, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
