---
description: Run the full pre-PR quality gate (format, lint, mypy, specs, docs, claims, status artifact, unit/integration tests)
allowed-tools: Bash, Read
---

# Pre-PR Validation Pipeline

Run the comprehensive pre-PR validation suite in CI order to ensure the branch is ready for submission:

1. **Static Analysis & Formatting**:

    ```bash
    black . --check --line-length 120
    ruff check .
    mypy src/
    ```

2. **Deterministic Context & Spec Invariants**:

    ```bash
    python scripts/validate_context_docs.py
    python -m src.framework.harness.cli validate-spec specs/*.SPEC.md
    python -m src.tools.claim_ledger
    python -m src.tools.action_pins
    ```

3. **Status Artifact Generation**:

    ```bash
    python -m src.tools.status_artifact --strict
    ```

4. **Unit & Integration Test Gates** (must match CI's `STRICT_OPTIONAL_DEPS=1`):

    ```bash
    STRICT_OPTIONAL_DEPS=1 pytest tests/unit/ -v --cov=src --cov-fail-under=85
    pytest tests/integration/ -v
    pytest tests/e2e/ -v
    ```

5. **Security & Readiness Checks**:

    ```bash
    python scripts/security_audit.py
    python scripts/production_readiness_check.py
    git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/ && echo "FAIL: hardcoded key material" && exit 1 || echo "OK: no key material"
    command -v gitleaks > /dev/null && gitleaks detect --config .gitleaks.toml --source . --no-git -v || echo "gitleaks not installed locally — that layer runs in CI"
    ```
