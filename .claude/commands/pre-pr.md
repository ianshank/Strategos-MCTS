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

4. **Unit & Integration Test Gates**:
   ```bash
   pytest tests/unit/ -v --cov=src --cov-fail-under=85
   pytest tests/integration/ -v
   pytest tests/e2e/ -v
   ```

5. **Security & Readiness Checks**:
   ```bash
   python scripts/security_audit.py
   python scripts/production_readiness_check.py
   ```
