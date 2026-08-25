#!/usr/bin/env python3
"""
Comprehensive linting and formatting script.
Run this before committing to ensure CI will pass.

Usage:
    python scripts/lint_and_format.py           # Check and fix everything
    python scripts/lint_and_format.py --check   # Check only (no fixes)
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], check: bool = True) -> tuple[int, str, str]:
    """Run a command from the repo root and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    return result.returncode, result.stdout, result.stderr


def main():
    """Main linting and formatting workflow."""
    check_only = "--check" in sys.argv

    print("🔍 Comprehensive Code Quality Check")
    print("=" * 60)

    errors = []

    # 1. Black Format (the project formatter; matches the CI lint job)
    print("\n📝 Step 1: Formatting with Black")
    if check_only:
        returncode, stdout, stderr = run_command(["black", "--check", "--line-length", "120", "."])
        if returncode != 0:
            errors.append("Formatting check failed (run without --check to fix)")
            dirty = [line for line in stderr.splitlines() if line.startswith("would reformat")]
            print(f"❌ {len(dirty)} files need formatting")
        else:
            print("✅ All files properly formatted")
    else:
        returncode, stdout, stderr = run_command(["black", "--line-length", "120", "."])
        if returncode == 0:
            print("✅ Auto-formatted all files")
        else:
            errors.append("Auto-formatting failed")
            print(f"❌ Formatting failed: {stderr}")

    # 2. Ruff Lint (with auto-fix)
    print("\n🔎 Step 2: Linting with Ruff")
    if check_only:
        returncode, stdout, stderr = run_command(["ruff", "check", "."])
    else:
        returncode, stdout, stderr = run_command(["ruff", "check", ".", "--fix"])

    if returncode != 0:
        print(f"⚠️  Some linting issues {'remain' if not check_only else 'found'}")
        print(stdout)
        if not check_only:
            errors.append("Some linting errors couldn't be auto-fixed")
    else:
        print("✅ No linting errors")

    # 3. Python Syntax Check
    print("\n🐍 Step 3: Python Syntax Validation")
    python_files = list(ROOT.rglob("*.py"))
    syntax_errors = []

    for py_file in python_files:
        if ".venv" in str(py_file) or "venv" in str(py_file):
            continue
        try:
            compile(py_file.read_text(encoding="utf-8"), str(py_file), "exec")
        except SyntaxError as e:
            syntax_errors.append(f"{py_file}:{e.lineno}: {e.msg}")

    if syntax_errors:
        print(f"❌ {len(syntax_errors)} syntax errors found:")
        for error in syntax_errors[:10]:  # Show first 10
            print(f"   {error}")
        errors.append("Syntax errors found")
    else:
        print(f"✅ All {len(python_files)} Python files have valid syntax")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ FAILED - Issues found:")
        for error in errors:
            print(f"   • {error}")
        print("\n💡 Run without --check to auto-fix most issues")
        sys.exit(1)
    else:
        print("✅ SUCCESS - All checks passed!")
        if not check_only:
            print("\n💡 Auto-fixed files have been modified.")
            print("   Review changes and commit: git add -u && git commit")
        sys.exit(0)


if __name__ == "__main__":
    main()
