---
name: quality-gate
description: >-
  Run the full Strategos-MCTS local quality gate before any commit or push:
  format check, lint, type check, spec and doc validation, the evidence-chain
  checks (claim ledger, Actions pin ratchet, ruff rule ratchet), unit tests with
  branch coverage, the end-to-end suite, and
  the two secret-detection layers (a fast src/+kubernetes/ grep, and a repo-wide gitleaks
  scan if the binary is installed). Use whenever you are about to push, open a
  PR, or want CI/local parity.
---

# Quality Gate

Run the repo's CI-equivalent gate locally, in order. Stop at the first failure and fix before
continuing. Mirrors `.github/workflows/ci.yml` so green locally means green in CI.

```bash
# 1. Format (check only; run without --check to auto-fix) — repo-wide, matching CI
black . --check --line-length 120

# 2. Lint — repo-wide, matching CI (notebooks are excluded via pyproject)
ruff check .

# 3. Types (pinned mypy; see [dev] extra)
mypy src/

# 4. Specs and context docs — deterministic, no network
harness validate-spec specs/*.SPEC.md
python -m src.tools.context_docs

# 5. Evidence chain: every capability claim graded against the tree, the GitHub Actions
# pin ratchet, and the ruff rule ratchet. All fail closed; none has a bypass flag.
python -m src.tools.claim_ledger
python -m src.tools.action_pins
python -m src.tools.lint_ratchet

# 6. Tests with branch coverage (gate = fail_under 85.0 in pyproject.toml)
pytest tests/unit/ --cov=src --cov-report=term-missing --cov-fail-under=85

# 6b. End-to-end suite, outside the coverage run (evidence-program R3). Read the skip
# lines before reporting it green — see the `e2e-device-matrix` skill.
pytest tests/e2e -m "not ui" -ra

# 7. No hardcoded secrets in source or manifests (fast, narrow-scope layer).
# Use if/else with a subshell `(exit 1)` so a match returns a non-zero status
# (for CI / `set -e`) without the `A && B || C` pitfall and without closing an
# interactive shell.
if git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/; then echo "FAIL: hardcoded key"; (exit 1); else echo "OK: no keys"; fi

# 8. Repo-wide, pattern-agnostic secret scan (CI-enforced via secret-scan-gitleaks;
# run locally only if the gitleaks binary is installed — it is not part of the [dev] extra).
# Note the shape: step 7 warns about `A && B || C` and this line used to commit it, so a scan
# that FOUND something took the `||` branch and printed "not installed" while exiting 0.
if command -v gitleaks >/dev/null 2>&1; then gitleaks detect --config .gitleaks.toml --source . --no-git -v; else echo "gitleaks not installed locally — this layer runs in CI"; fi
```

Or run all of it in CI order with one command: `make gate`. The Makefile is pinned to the workflow
by `tests/unit/test_ci_workflow_invariants.py`, so the two cannot drift apart silently.

Notes:
- Steps 4–5 are the cheap ones and the ones most often skipped. Run them first when you have only
  touched documentation: a prose edit that over-claims fails `claim_ledger`, not `pytest`.
- Install deps first if missing: `pip install -e ".[dev,neural,api]"` — the exact set the CI
  test job installs. Without `api`, FastAPI is absent and 115 API-server tests are silently
  skipped, so the local number diverges from CI.
- `--strict-markers` is active: an unregistered `@pytest.mark.*` is a collection error, not a
  silent no-op. Register new markers in `pyproject.toml [tool.pytest.ini_options] markers`.
- All tunables must come from `src/config/settings.py` / `src/config/constants.py` — never hardcode.
- Unit tests must not make real network/API calls; mock all I/O.
- The step-7 and step-8 secret scans are complementary, not redundant: step 7 is instant and
  dependency-free but scoped to `src/`/`kubernetes/` and `sk-`-shaped keys only; step 8 is
  repo-wide and pattern-agnostic (see `docs/reviews/2026-07-31-charter-alignment-audit.md` F-17 for
  why both are needed — a real key sat in `docs/` for a while, invisible to step 7 alone).
