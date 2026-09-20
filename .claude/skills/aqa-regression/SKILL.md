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

## Lanes

Run these in order. Do not treat a CPU-only green e2e run as a GPU claim (CHARTER NG-3).
Live LM Studio is adapter QA, not a distillation teacher.

1. **Hermetic unit** (`make test`): `tests/unit/` with `--cov=src --cov-fail-under=85`. Dummy `OPENAI_API_KEY`, `HF_HUB_OFFLINE=1`, `ALLOW_MOCK_LLM_FALLBACK` default false. Distillation units live under `tests/unit/scripts/local_distillation/` (also `make test-local-distillation`). Coverage gate does not measure `scripts/`.
2. **E2E device matrix**: `E2E_DEVICES=cpu make test-e2e` then `E2E_DEVICES=cuda,cpu make test-e2e`. CUDA must *fail* if missing when named. Distillation CLI is `tests/e2e/test_local_distillation_cli_e2e.py` (`python -m scripts.local_distillation`). Placement is `tests/e2e/test_local_distillation_device_e2e.py`. Same-device NeuralMCTS reproducibility is `tests/e2e/test_neural_mcts_device_e2e.py`.
3. **Regression / slow**: `make test-regression` then `pytest tests/ --runslow -ra`.
4. **UI**: `make test-ui` (record env skips; Gradio/LFS/offline-hub misses matching origin/main are RCA-ENV).
5. **Live LM Studio** (opt-in, never `tests/unit/`): `REQUIRE_LMSTUDIO=1`, `LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1`, `ALLOW_MOCK_LLM_FALLBACK=false`. Tests: `tests/e2e/test_lmstudio_live_e2e.py`. Fail-loud closed port: `tests/integration/test_lmstudio_fail_loud.py`. Health `GET /v1/models` must be 200 when required; skip unless `REQUIRE_LMSTUDIO=1`.
6. **Value-semantics CHARTER demo**: `pytest tests/unit/framework/mcts/test_value_semantics_regression.py -q` (CHARTER.md §2). Neural arm skips without torch; CI `[dev,neural,api]` must collect it. CL-1 is `FALSE` on this tree (`docs/CLAIM_LEDGER.md`) and only moves to PARTIAL after the #174 in-module fixes land; GraphBuilder `two_player=True` remains follow-up.
7. **Dockerfile perl-base pin**: run the perl-base unit lane when overlay #172 is present; otherwise record it as pending overlay evidence for this tree. Blocking Trivy CRITICAL is `.github/workflows/ci.yml` docker-build; `.github/workflows/docker-deployment.yml` scan is advisory. Do not ignore CVE-2026-13221/42496/8376.
8. **Deploy-sanity path pin**: run the deploy-sanity path unit lane when overlay #173 is present; otherwise record it as pending overlay evidence for this tree. In this tree today, `scripts/deployment_sanity_check.py` still runs `pytest tests/ -m smoke` with a 60s timeout. Container docker smoke stays on the Container Smoke Tests job.

## RCA ids

Classify every failure before patching. Env-identical-to-`origin/main` is not a #169 bug.

| Id | Meaning |
|---|---|
| RCA-DIST | Distillation π/z/sidecar/device placement |
| RCA-NUM | CUDA bitwise / floating-point |
| RCA-LLM | Adapter/factory (model id, timeout, omni `content`) |
| RCA-MOCK | Mock fallback when fail-loud was required |
| RCA-DEV | Device mismatch (CPU vs CUDA, NeuralMCTS.device) |
| RCA-URL | Missing `/v1`, `localhost` vs `127.0.0.1` |
| RCA-ENV | Gradio/LFS/offline Hub/missing optional extra |
| RCA-SMOKE | Sanity collected docker smoke / 60s cap / `smoke and not e2e` mix |

## Instructions

1. **Run Unit Tests & Mypy**:
   ```bash
   mypy src/
   mypy -p scripts.local_distillation
   pytest tests/unit -v --cov=src --cov-fail-under=85
   pytest tests/unit/scripts/local_distillation -v
   ```
   On Windows, invoke pytest as `python -m pytest ... -p no:randomly` (user-site pytest-randomly + thinc can crash collection).
2. **Run Integration Tests**:
   ```bash
   pytest tests/integration -v
   ```
3. **Run E2E and UI Tests**:
   ```bash
   E2E_DEVICES=cpu pytest tests/e2e -m "not ui" -ra
   E2E_DEVICES=cuda,cpu pytest tests/e2e -m "not ui" -ra
   pytest tests/e2e/test_ui_e2e.py
   pytest tests/e2e/test_user_journeys.py
   ```
4. **Run Status Artifact and Claim Ledger Check**:
   ```bash
   python -m src.tools.claim_ledger
   python -m src.tools.status_artifact --strict
   ```
5. **Parse Results**:
   Classify into the RCA ids above (plus typing / serialization / logger leaks when they appear) and write under `docs/reviews/`.
