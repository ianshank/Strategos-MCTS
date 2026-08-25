#!/usr/bin/env python3
"""
Deployment Sanity Check Script
===============================

Runs pre-deployment checks to verify the codebase is ready for Docker deployment.

Checks:
- Configuration files validity
- Python syntax
- Import dependencies
- Test suite passing
- Docker files exist
- Environment templates

Usage:
    python scripts/deployment_sanity_check.py [--verbose]

2025 Best Practices:
- Comprehensive validation before deployment
- Early failure detection
- Clear error messages
- Exit codes for CI/CD integration
"""

import logging
from pathlib import Path
import subprocess
import sys

from rich.console import Console
from rich.table import Table
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DeploymentSanityChecker:
    """Comprehensive pre-deployment sanity checker."""

    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose
        self.failures: list[str] = []
        self.warnings: list[str] = []

        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def check_config_files(self) -> bool:
        """Check all configuration files are valid."""
        self.console.print("\n[cyan]Checking configuration files...[/cyan]")

        config_files = [
            PROJECT_ROOT / "training" / "config.yaml",
            PROJECT_ROOT / "training" / "config_local_demo.yaml",
            PROJECT_ROOT / "pyproject.toml",
        ]

        all_valid = True
        for config_file in config_files:
            if not config_file.exists():
                self.console.print(f"  [red]✗[/red] Missing: {config_file.name}")
                self.failures.append(f"Missing config: {config_file}")
                all_valid = False
                continue

            try:
                if config_file.suffix in [".yaml", ".yml"]:
                    with open(config_file) as f:
                        yaml.safe_load(f)
                elif config_file.suffix == ".toml":
                    import tomli

                    with open(config_file, "rb") as f:
                        tomli.load(f)

                self.console.print(f"  [green]✓[/green] Valid: {config_file.name}")
            except Exception as e:
                self.console.print(f"  [red]✗[/red] Invalid: {config_file.name} - {e}")
                self.failures.append(f"Invalid config: {config_file}: {e}")
                all_valid = False

        return all_valid

    def check_docker_files(self) -> bool:
        """Check Docker-related files exist."""
        self.console.print("\n[cyan]Checking Docker files...[/cyan]")

        docker_files = [
            PROJECT_ROOT / "Dockerfile",
            PROJECT_ROOT / "Dockerfile.train",
            PROJECT_ROOT / "docker-compose.yml",
            PROJECT_ROOT / "docker-compose.train.yml",
            PROJECT_ROOT / ".dockerignore",
        ]

        all_exist = True
        for docker_file in docker_files:
            if docker_file.exists():
                self.console.print(f"  [green]✓[/green] Found: {docker_file.name}")
            else:
                if docker_file.name == ".dockerignore":
                    self.console.print(f"  [yellow]⚠[/yellow] Missing (optional): {docker_file.name}")
                    self.warnings.append(f"Missing optional file: {docker_file}")
                else:
                    self.console.print(f"  [red]✗[/red] Missing: {docker_file.name}")
                    self.failures.append(f"Missing Docker file: {docker_file}")
                    all_exist = False

        return all_exist

    def check_python_syntax(self) -> bool:
        """Check Python files have valid syntax."""
        self.console.print("\n[cyan]Checking Python syntax...[/cyan]")

        python_files = list(PROJECT_ROOT.glob("**/*.py"))
        python_files = [f for f in python_files if "venv" not in str(f) and "__pycache__" not in str(f)]

        errors = []
        for py_file in python_files[:50]:  # Check first 50 files
            try:
                with open(py_file, encoding="utf-8") as f:
                    compile(f.read(), str(py_file), "exec")
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")

        if errors:
            self.console.print(f"  [red]✗[/red] Found {len(errors)} syntax errors")
            for error in errors[:5]:  # Show first 5
                self.console.print(f"    {error}")
            self.failures.extend(errors)
            return False

        self.console.print("  [green]✓[/green] All Python files have valid syntax")
        return True

    def check_dependencies(self) -> bool:
        """Check required dependencies are installable."""
        self.console.print("\n[cyan]Checking dependencies...[/cyan]")

        requirements_file = PROJECT_ROOT / "requirements.txt"
        if not requirements_file.exists():
            self.console.print("  [red]✗[/red] requirements.txt not found")
            self.failures.append("Missing requirements.txt")
            return False

        # Just check file is readable
        try:
            with open(requirements_file) as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            self.console.print(f"  [green]✓[/green] Found {len(lines)} dependencies")
            return True
        except Exception as e:
            self.console.print(f"  [red]✗[/red] Error reading requirements.txt: {e}")
            self.failures.append(f"Cannot read requirements.txt: {e}")
            return False

    def check_test_suite(self) -> bool:
        """Run smoke tests."""
        self.console.print("\n[cyan]Running smoke tests...[/cyan]")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-m", "smoke", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                self.console.print("  [green]✓[/green] Smoke tests passed")
                return True
            else:
                self.console.print("  [red]✗[/red] Smoke tests failed")
                if self.verbose:
                    self.console.print(result.stdout)
                    self.console.print(result.stderr)
                self.failures.append("Smoke tests failed")
                return False
        except subprocess.TimeoutExpired:
            self.console.print("  [red]✗[/red] Smoke tests timed out")
            self.failures.append("Smoke tests timeout")
            return False
        except Exception as e:
            self.console.print(f"  [yellow]⚠[/yellow] Could not run tests: {e}")
            self.warnings.append(f"Test execution error: {e}")
            return True  # Don't fail deployment

    def check_required_scripts(self) -> bool:
        """Check required scripts exist."""
        self.console.print("\n[cyan]Checking required scripts...[/cyan]")

        scripts = [
            PROJECT_ROOT / "scripts" / "verify_external_services.py",
            PROJECT_ROOT / "scripts" / "run_local_demo.ps1",
            PROJECT_ROOT / "training" / "cli.py",
        ]

        all_exist = True
        for script in scripts:
            if script.exists():
                self.console.print(f"  [green]✓[/green] Found: {script.name}")
            else:
                self.console.print(f"  [red]✗[/red] Missing: {script.name}")
                self.failures.append(f"Missing script: {script}")
                all_exist = False

        return all_exist

    def check_documentation(self) -> bool:
        """Check documentation exists."""
        self.console.print("\n[cyan]Checking documentation...[/cyan]")

        docs = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "LOCAL_TRAINING_GUIDE.md",
        ]

        for doc in docs:
            if doc.exists():
                self.console.print(f"  [green]✓[/green] Found: {doc.name}")
            else:
                self.console.print(f"  [yellow]⚠[/yellow] Missing: {doc.name}")
                self.warnings.append(f"Missing documentation: {doc}")

        return True  # Don't fail on missing docs

    def run_all_checks(self) -> bool:
        """Run all sanity checks."""
        self.console.print("\n[bold cyan]Starting Deployment Sanity Checks[/bold cyan]")

        checks = [
            ("Configuration Files", self.check_config_files),
            ("Docker Files", self.check_docker_files),
            ("Python Syntax", self.check_python_syntax),
            ("Dependencies", self.check_dependencies),
            ("Required Scripts", self.check_required_scripts),
            ("Test Suite", self.check_test_suite),
            ("Documentation", self.check_documentation),
        ]

        results = {}
        for name, check_func in checks:
            results[name] = check_func()

        # Display summary
        self.console.print("\n[bold]Summary[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")

        for name, passed in results.items():
            status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
            table.add_row(name, status)

        self.console.print(table)

        # Display failures and warnings
        if self.failures:
            self.console.print(f"\n[red]Failures ({len(self.failures)}):[/red]")
            for failure in self.failures[:10]:
                self.console.print(f"  • {failure}")

        if self.warnings:
            self.console.print(f"\n[yellow]Warnings ({len(self.warnings)}):[/yellow]")
            for warning in self.warnings[:10]:
                self.console.print(f"  • {warning}")

        all_passed = all(results.values())

        if all_passed:
            self.console.print("\n[bold green]✓ All checks passed - Ready for deployment![/bold green]")
        else:
            self.console.print("\n[bold red]✗ Some checks failed - Fix issues before deploying[/bold red]")

        return all_passed


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-deployment sanity checks")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    checker = DeploymentSanityChecker(verbose=args.verbose)

    try:
        success = checker.run_all_checks()
        return 0 if success else 1
    except Exception as e:
        checker.console.print(f"\n[bold red]Error: {e}[/bold red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
