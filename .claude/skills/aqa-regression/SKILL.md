---
name: aqa-regression
description: Executes full automated quality assurance suites, categorizes failures, and generates RCA reports.
---
# aqa-regression

Use this skill to execute the full test suite and verify no regressions exist.

## When to use
- After completing a complex refactoring or adding a new feature.
- To ensure no existing functionality is broken.
- To run the full matrix of tests (unit, integration, ui, e2e).

## Instructions

1. **Run Unit Tests & Mypy**:
   ```bash
   mypy src/
   pytest tests/unit -v
   ```
2. **Run Integration Tests**:
   ```bash
   pytest tests/integration -v
   ```
3. **Run E2E and UI Tests** (Ensure Docker or necessary services are running):
   ```bash
   pytest tests/e2e -m "not ui" -ra
   pytest tests/e2e/test_ui_e2e.py
   pytest tests/e2e/test_user_journeys.py
   ```
4. **Run Status Artifact and Claim Ledger Check**:
   ```bash
   python -m src.tools.claim_ledger
   python -m src.tools.status_artifact --strict
   ```
5. **Parse Results**:
   If any failures occur, classify them into:
   - Typing
   - Serialization
   - Logger Leaks
   - Numerical Tolerances
   - Other
   And provide a root-cause analysis (RCA) report.
