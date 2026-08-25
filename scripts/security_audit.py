#!/usr/bin/env python3
"""
Security Audit Script.

Scans the codebase for potential security issues:
- Exposed API keys and secrets
- Hardcoded credentials
- Insecure configurations
- Missing .gitignore entries

CRITICAL: This script identifies sensitive data that should NOT be committed.
"""

import os
import re
import sys
from pathlib import Path

# Patterns for detecting sensitive data
SENSITIVE_PATTERNS = {
    "api_key_generic": r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    "secret_key": r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    "password": r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']',
    "token": r'(?i)(token|bearer)\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?',
    "openai_key": r"sk-[a-zA-Z0-9]{32,}",
    "anthropic_key": r"sk-ant-[a-zA-Z0-9\-]{32,}",
    "pinecone_key": r"pcsk_[a-zA-Z0-9_]{32,}",
    "wandb_key": r"wandb_[a-zA-Z0-9]{32,}",
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "aws_secret": r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*[a-zA-Z0-9/+]{40}",
    "private_key": r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----",
    "github_token": r"ghp_[a-zA-Z0-9]{36}",
    "hf_token": r"hf_[a-zA-Z0-9]{32,}",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "__pycache__",
    "node_modules",
    "wandb",
    "output",
    "outputs",
    "artifacts",
    "cache",
    "checkpoints",
    "models",
    "htmlcov",
    ".benchmarks",
    "langgraph_multi_agent_mcts.egg-info",
}

# Files/directories to skip
SKIP_PATTERNS = [
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env.example",
    ".env.template",
    "*.pyc",
    "*.so",
    "*.dll",
    "wandb",
    "output",
    "outputs",
    "htmlcov",
]

# Sensitive file patterns
SENSITIVE_FILES = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.test",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "credentials.json",
    "service_account.json",
    "*.sqlite",
    "*.db",
]


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    return any(pattern in path_str for pattern in SKIP_PATTERNS)


def scan_file_for_secrets(file_path: Path) -> list[tuple[str, str, int]]:
    """
    Scan a single file for potential secrets.

    Returns:
        List of (pattern_name, match, line_number) tuples
    """
    findings = []

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern in SENSITIVE_PATTERNS.items():
                matches = re.findall(pattern, line)
                if matches:
                    # Extract the actual secret value
                    for match in matches:
                        secret = (match[-1] if len(match) > 1 else match[0]) if isinstance(match, tuple) else match

                        # Mask the secret for reporting
                        masked = secret[:4] + "*" * (len(secret) - 8) + secret[-4:] if len(secret) > 8 else "***"
                        findings.append((pattern_name, masked, line_num))

    except Exception:
        pass  # Skip files that can't be read

    return findings


def scan_codebase(root_path: Path) -> dict[str, list[tuple[str, str, int]]]:
    """
    Scan entire codebase for secrets.

    Returns:
        Dictionary mapping file paths to findings
    """
    all_findings = {}

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            file_path = Path(root) / file
            if should_skip(file_path):
                continue

            # Skip binary files
            if file_path.suffix in [".pyc", ".so", ".dll", ".exe", ".bin", ".zip", ".tar", ".gz"]:
                continue

            findings = scan_file_for_secrets(file_path)
            if findings:
                rel_path = str(file_path.relative_to(root_path))
                all_findings[rel_path] = findings

    return all_findings


def check_sensitive_files(root_path: Path) -> list[str]:
    """Check for sensitive files that shouldn't be committed."""
    found_files = []

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            file_path = Path(root) / file
            for pattern in SENSITIVE_FILES:
                if "*" in pattern:
                    if file_path.match(pattern) and not should_skip(file_path):
                        found_files.append(str(file_path.relative_to(root_path)))
                else:
                    if file == pattern and not should_skip(file_path):
                        found_files.append(str(file_path.relative_to(root_path)))

    return found_files


def check_gitignore(root_path: Path) -> list[str]:
    """Check if .gitignore includes necessary entries."""
    gitignore_path = root_path / ".gitignore"
    missing_entries = []

    required_entries = [
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "credentials.json",
        "service_account.json",
        "__pycache__",
        "*.pyc",
        "wandb/",
        "output/",
    ]

    if not gitignore_path.exists():
        return required_entries

    with open(gitignore_path) as f:
        gitignore_content = f.read()

    for entry in required_entries:
        # Check if entry or similar pattern exists
        if entry not in gitignore_content:
            # Check for variations
            base = entry.replace("*", "").replace("/", "")
            if base not in gitignore_content:
                missing_entries.append(entry)

    return missing_entries


def generate_report(
    secrets_found: dict[str, list[tuple[str, str, int]]],
    sensitive_files: list[str],
    gitignore_missing: list[str],
    root_path: Path,
) -> str:
    """Generate security audit report."""
    report = []
    report.append("=" * 60)
    report.append("SECURITY AUDIT REPORT")
    report.append("=" * 60)
    report.append(f"\nScanned: {root_path}")
    report.append(f"Date: {__import__('datetime').datetime.now().isoformat()}")

    # Critical findings
    if secrets_found:
        report.append("\n" + "=" * 60)
        report.append("[CRITICAL] POTENTIAL SECRETS FOUND IN CODE")
        report.append("=" * 60)
        report.append("These files may contain exposed API keys or secrets:")
        report.append("")

        for file_path, findings in sorted(secrets_found.items()):
            report.append(f"File: {file_path}")
            for pattern_name, masked_value, line_num in findings:
                report.append(f"  Line {line_num}: {pattern_name} = {masked_value}")
            report.append("")

        report.append("ACTION REQUIRED:")
        report.append("1. Remove or rotate these credentials immediately")
        report.append("2. Add files to .gitignore if not already")
        report.append("3. Consider using environment variables or secret managers")
        report.append("4. If already committed, consider git history cleanup")

    # Sensitive files
    if sensitive_files:
        report.append("\n" + "=" * 60)
        report.append("[WARNING] SENSITIVE FILES DETECTED")
        report.append("=" * 60)
        for file in sensitive_files:
            report.append(f"  - {file}")

        report.append("\nACTION: Ensure these files are in .gitignore")

    # Gitignore gaps
    if gitignore_missing:
        report.append("\n" + "=" * 60)
        report.append("[RECOMMENDATION] GITIGNORE IMPROVEMENTS")
        report.append("=" * 60)
        report.append("Consider adding these entries to .gitignore:")
        for entry in gitignore_missing:
            report.append(f"  {entry}")

    # Summary
    report.append("\n" + "=" * 60)
    report.append("SUMMARY")
    report.append("=" * 60)

    if secrets_found:
        report.append(f"[CRITICAL] {len(secrets_found)} files with potential secrets")
    else:
        report.append("[OK] No secrets found in code")

    if sensitive_files:
        report.append(f"[WARNING] {len(sensitive_files)} sensitive files present")
    else:
        report.append("[OK] No sensitive files found")

    if gitignore_missing:
        report.append(f"[INFO] {len(gitignore_missing)} gitignore entries recommended")
    else:
        report.append("[OK] .gitignore looks comprehensive")

    return "\n".join(report)


def main():
    """Run security audit."""
    print("=" * 60)
    print("MULTI-AGENT MCTS SECURITY AUDIT")
    print("=" * 60)
    print("\nScanning codebase for security issues...")

    root_path = Path(__file__).parent.parent

    # Scan for secrets
    print("  - Scanning for exposed secrets...")
    secrets_found = scan_codebase(root_path)

    # Check for sensitive files
    print("  - Checking for sensitive files...")
    sensitive_files = check_sensitive_files(root_path)

    # Check gitignore
    print("  - Analyzing .gitignore...")
    gitignore_missing = check_gitignore(root_path)

    # Generate report
    report = generate_report(secrets_found, sensitive_files, gitignore_missing, root_path)

    print("\n" + report)

    # Save report
    report_path = root_path / "output" / "security_audit_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Exit code based on findings
    if secrets_found:
        print("\n[CRITICAL] Security issues found! Review report above.")
        return 1
    elif sensitive_files:
        print("\n[WARNING] Sensitive files detected. Ensure they're in .gitignore.")
        return 0
    else:
        print("\n[OK] No critical security issues found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
