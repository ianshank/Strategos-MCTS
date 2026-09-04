# Strategos-MCTS — developer entry points.
#
# This file is deliberately THIN. It does not invent commands: every target runs
# exactly what `.github/workflows/ci.yml` runs, so "green locally" means "green in
# CI". `tests/unit/test_ci_workflow_invariants.py` asserts that the flags below
# still match the workflow, so this cannot silently drift into a third,
# contradictory source of truth alongside CLAUDE.md and the quality-gate skill.
#
# Run `make` or `make help` for the target list.

# Single source for the values CI also uses. Override on the command line, e.g.
#   make test PYTEST_ARGS="-k mcts"
#   make install EXTRAS="dev,neural"
PYTHON      ?= python
LINE_LENGTH ?= 120
COV_MIN     ?= 85
EXTRAS      ?= dev,neural,api
PYTEST_ARGS ?=
# Extra flags for `make status`, e.g. STATUS_ARGS="--coverage-json coverage.json"
# to stamp a measured coverage total into the artifact.
STATUS_ARGS ?=

# Mirrors the `test` job's env block in ci.yml. Unit tests must not touch the
# network (CHARTER.md INV-4), and settings validation rejects an empty API key.
TEST_ENV := WANDB_MODE=disabled \
            LANGCHAIN_TRACING_V2=false \
            HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            TOKENIZERS_PARALLELISM=false \
            OPENAI_API_KEY=sk-test-key-not-real \
            STRICT_OPTIONAL_DEPS=1

.DEFAULT_GOAL := help
.PHONY: help install format format-check lint lint-fix typecheck test test-e2e test-all \
        coverage specs docs claims claims-baseline status pins pins-baseline secrets gate clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the project with the CI extras
	$(PYTHON) -m pip install -e ".[$(EXTRAS)]"

format: ## Auto-format (black)
	black . --line-length $(LINE_LENGTH)

format-check: ## Check formatting without writing (CI step)
	black . --check --line-length $(LINE_LENGTH)

lint: ## Lint (ruff), no autofix (CI step)
	ruff check .

lint-fix: ## Lint with autofix
	ruff check . --fix

typecheck: ## Type check. NOT --strict; see CLAUDE.md for why (CI step)
	mypy src/

test: ## Unit tests with branch coverage — the gate CI enforces
	$(TEST_ENV) $(PYTHON) -m pytest tests/unit/ \
		--cov=src --cov-report=term-missing --cov-fail-under=$(COV_MIN) $(PYTEST_ARGS)

test-e2e: ## End-to-end suite, as the CI test job runs it (set E2E_DEVICES to pin the device matrix)
	$(TEST_ENV) $(PYTHON) -m pytest tests/e2e -m "not ui" -ra $(PYTEST_ARGS)

test-all: ## Full non-slow sweep (wider than the gate; expect env-dependent failures)
	$(TEST_ENV) $(PYTHON) -m pytest tests/ -m "not slow" $(PYTEST_ARGS)

coverage: ## Refresh the measured baseline in docs/STATUS.md
	$(TEST_ENV) $(PYTHON) -m pytest tests/unit/ \
		--cov=src --cov-report=term-missing --cov-report=html:htmlcov \
		--cov-fail-under=$(COV_MIN)
	@echo "HTML report: htmlcov/index.html"

specs: ## Validate every spec against schema v2 (CI step)
	harness validate-spec specs/*.SPEC.md

docs: ## Verify documentation claims resolve against the tree (CHARTER INV-10)
	$(PYTHON) -m src.tools.context_docs

claims: ## Validate the capability claim ledger (evidence-chain R1, CI step)
	$(PYTHON) -m src.tools.claim_ledger

claims-baseline: ## Re-tighten the claim-surface baseline after grading a new claim
	$(PYTHON) -m src.tools.claim_ledger --write-surface-baseline

status: ## Write the provenance-stamped status artifact (needs `make claims` green)
	$(PYTHON) -m src.tools.status_artifact --strict $(STATUS_ARGS)

pins: ## Check the GitHub Actions commit-SHA pin ratchet (CI step)
	$(PYTHON) -m src.tools.action_pins

pins-baseline: ## Re-tighten the pin baseline after pinning an action to a SHA
	$(PYTHON) -m src.tools.action_pins --write-baseline

secrets: ## Fast secret grep, matching the spec-validate CI step
	@if git grep -nE "sk-[A-Za-z0-9]{20,}" -- src/ kubernetes/; then \
		echo "FAIL: hardcoded key material"; exit 1; \
	else echo "OK: no key material"; fi
	@command -v gitleaks >/dev/null \
		&& gitleaks detect --config .gitleaks.toml --source . --no-git -v \
		|| echo "gitleaks not installed locally — that layer runs in CI"

gate: format-check lint typecheck specs docs claims status pins test secrets ## Full local gate, in CI order
	@echo "Quality gate passed."

clean: ## Remove build/test artifacts
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml junit.xml \
	       bandit-report.json pip-audit-report.json trivy-results.sarif \
	       .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
