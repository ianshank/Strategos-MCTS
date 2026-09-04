# E2E test program — device-agnostic end-to-end coverage for the 2026 H2 work

> **Status:** v1 draft, under adversarial review · **Owner axis:** sequenced work (this file plans
> tests; it asserts nothing about measured status — `docs/STATUS.md` owns that, and no line here
> promotes a capability grade in `docs/CLAIM_LEDGER.md`).
>
> **Provenance:** written after three independent surveys of the tree at `30a75bf` (`main`):
> the existing E2E suite and its CI gating, the work that landed since June 2026 and its test
> coverage, and every device-selection site under `src/` and `tests/`.

---

## 0. The verdict, in one paragraph

The repository's "E2E" suite is not end-to-end and does not run on pull requests. Of the 95 tests
under `tests/e2e/`, 27 reimplement UCB1 in a dictionary and import nothing from `src/`, 26 touch
`src` only through the `QueryInput` validator, and 27 need `gradio`. The one module that drives real
components (`test_user_journeys.py`) hard-codes `device="cpu"`. The only workflow that runs the
directory is `e2e_with_langsmith.yml`, which is gated on a `LANGSMITH_API_KEY` secret, so on a fork
or a secret-less clone the suite never executes and never blocks a merge. Meanwhile the work that
landed since June — the self-play convergence driver and policy-lift gate, the DDP orchestrator,
the evidence tooling, the REST hardening, the training-container healthcheck — is unit-tested to a
high standard and has **zero** tests that spawn the real entry points as a user would. Not one
test in the tree is parametrized over devices; the repository's single-GPU path is verified by
nothing but the CPU-only CI runner and the `NG-6` invariant that says it should keep working.

## 1. What "E2E" means here, stated as a contract

A test belongs in `tests/e2e/` only if all four hold:

1. **Real entry point.** It drives the component the way a user does: a console-script module via
   `python -m`, the FastAPI app through its lifespan, or a public factory — never a private
   function and never a re-implementation of the algorithm.
2. **Real components, mocked boundary.** Everything inside the process is real. The only stand-ins
   allowed are at the network boundary (LLM provider, vector store, tracker), and only through the
   paths the code already offers for that purpose (`ALLOW_MOCK_LLM_FALLBACK`, offline hub flags).
   No socket leaves the machine — `CHARTER.md` NG-8 is stated for unit tests, and this program
   applies it to E2E too because a suite whose verdict depends on the network gates nothing.
3. **Device-agnostic by construction.** Any test that moves a tensor is parametrized over every
   device the host offers, using one shared fixture built on `src/utils/device.py`. The same test
   file passes on a CPU runner, a CUDA box, and Apple Silicon; the accelerator cases are selected by
   availability, never by a hard-coded `"cuda"`.
4. **Bounded and reproducible.** Every subprocess carries an explicit timeout, every stochastic
   path is seeded through the entry point's own `--seed` / `SEED` channel, and a failure prints
   the captured stdout, stderr, and resolved device so the log alone explains it.

## 2. Device matrix design

All device logic in tests flows from three helpers in `tests/e2e/conftest.py`, which delegate to
`src.utils.device` rather than adding a fifth copy of the CUDA → MPS → CPU ladder:

| Fixture / helper | Behaviour |
|---|---|
| `available_devices()` | `["cpu"]`, plus `"cuda"` when `torch.cuda.is_available()`, plus `"mps"` when the MPS backend reports available. Overridable with the `E2E_DEVICES` environment variable (comma-separated) so an operator can pin the matrix; a requested device the host lacks becomes a **skip with a reason**, never a silent CPU substitution. |
| `device` (parametrized fixture) | Yields each entry of `available_devices()` as a plain string, because every `src/` seam (`NeuralMCTS`, `SelfPlayTrainer`, the CLIs' `--device`) takes a string. Non-CPU cases are auto-tagged with the `gpu` marker so `-m "not gpu"` deselects them. |
| `resolve_test_device(device)` | Thin wrapper over `src.utils.device.resolve_device` for the few call sites that need a `torch.device`. |
| `e2e_seed` | The seed passed to every entry point: `SEED` from the environment when set, else the documented default. Reuses the existing `Settings.SEED` channel — no new seed variable (`hygiene_determinism` AC-2). |
| `subprocess_env` | A hermetic copy of the environment: offline hub flags, tracing off, the dummy provider key the unit suite already uses, `PYTHONHASHSEED`, and `CUDA_VISIBLE_DEVICES` **passed through unchanged** so the child sees the same devices the parametrization saw. |
| `run_module(module, *args)` | Runs `python -m <module>` with the hermetic env and a timeout, logs the command and the resolved device at DEBUG, and attaches stdout/stderr to the assertion message on failure. |

A `gpu` marker is registered in both marker registries (`pyproject.toml` and `tests/conftest.py`;
`--strict-markers` is on). Existing GPU-only smoke tests keep their `FORCE_GPU_TESTS` contract —
this program adds no competing flag.

**Determinism across devices is not asserted.** CPU and CUDA kernels differ in reduction order,
so "same visit counts on both devices" is not a property the framework promises. What is asserted:
identical results on the **same** device from the same seed, valid outputs on **every** device, and
that a checkpoint written under any device loads under CPU through `src.models.checkpoints`.

## 3. Test modules (the deliverable)

| # | Module | Entry point exercised | Device-parametrized | Closes |
|---|---|---|---|---|
| B1 | `test_self_play_golden_path_e2e.py` | `python -m src.training.self_play_convergence` → `--resume` → `python -m src.benchmark.policy_lift` on the synthetic `reasoning` domain | yes | `m5_policy_lift` plumbing at the process boundary; checkpoint portability (save on accelerator, load on CPU); resume numbering; `.meta.json` sidecar; artifact provenance |
| B2 | `test_neural_mcts_device_parity_e2e.py` | `DomainRegistry` → `build_network` → `NeuralMCTS.search` on Connect Four | yes | CL-13's "search runs on every device": legal-move policy mass, visit-count budget, same-device seeded reproducibility, forward-pass agreement CPU vs accelerator within tolerance |
| B3 | `test_ddp_two_rank_cpu_e2e.py` | two subprocesses of the convergence driver under a real `gloo` process group (`RANK`/`WORLD_SIZE`/`MASTER_*` set as `torchrun` would) | CPU-only by design (`CUDA_VISIBLE_DEVICES=""` in the children) | `ddp_orchestrator` AC-3/AC-4 at the process boundary: both ranks exit 0, only rank 0 writes the checkpoint, the process group is initialised and torn down |
| B4 | `test_rest_api_lifespan_e2e.py` | the FastAPI `app` through its lifespan with `ALLOW_MOCK_LLM_FALLBACK=true` and the LLM factory refused | no (LLM path) | CL-2 / CL-10: `/health`, `/ready` reports `framework_degraded`, `/metrics`, `/graph/structure`, `/graph/mermaid`, `/query` answers through the real graph |
| B5 | `test_evidence_chain_cli_e2e.py` | `python -m` for `claim_ledger --json`, `status_artifact --stdout --strict`, `context_docs`, `action_pins`, `harness validate-spec`, `benchmark --dry-run` against the real tree | provenance assertions only | `evidence_claim_ledger` at the process boundary: exit codes, parseable JSON, and the status artifact's hardware record agreeing with what torch reports on this host |
| B6 | `test_healthcheck_contract_e2e.py` | `python healthcheck.py` as the container runs it | yes — expectation derived from `torch.cuda.is_available()` | the PR #165 exit-code contract: DEGRADED exits 0 on a CPU host, `REQUIRE_GPU=1` flips CUDA to critical, a GPU host reports the CUDA check healthy |

Explicitly **out of scope**, with the reason:

- The Hugging Face Space deploy path (`deploy-space.yml`, `space_bootstrap.py`): it force-pushes
  to an external service and polls its API. Its ruling under NG-1 is unratified (`CHARTER.md`
  §8.1). Unit tests for the assembler are a separate, spec-gated change.
- A two-rank **NCCL** run: needs two GPUs; the `gloo` CPU run proves the orchestration and is
  what a CPU-only CI can gate. A `gpu`-marked NCCL variant is a follow-up once a GPU runner exists.
- The Gradio UI: `tests/e2e/test_ui_e2e.py` and the `ui-tests` job already own it.
- Chess: `tests/games/chess/unit/` is documented bit-rot in `docs/STATUS.md`; repairing it is the
  `hygiene_chess_consolidation` spec's job, not this one's.
- Re-grading any ledger row. This program produces tests, not evidence artefacts; a grade moves
  only when the `eval-warden` referee accepts an artefact.

## 4. The one `src/` change

`src/framework/component_factory/configs.py` builds `TrainerConfig.from_settings` with
`device="cuda" if MCTS_IMPL == "neural" else "cpu"` — no availability check and no consultation of
`TORCH_DEVICE_OVERRIDE`, the single device knob `Settings` offers. On a CPU host with the neural
implementation selected, that string fails at the first `.to()`. The fix routes through the
existing helper: `TORCH_DEVICE_OVERRIDE` if set, else `get_default_device_str()`, else `"cpu"` for
the baseline implementation. Backwards compatible: a CUDA host with the neural implementation still
resolves to `cuda`. It lands in its own commit with a `No-Spec:` trailer because this branch is not
a `spec/<id>` branch (`CHARTER.md` NG-4, budget 1/2 — the trailer is the written exception).

The other duplicated ladders the survey found (`bert_controller.py`, `rnn_controller.py`,
`llm_guided/training/trainer.py`, `chess/config.py`, `neuro_symbolic/config.py`) are **not**
touched here: they are correct on CPU and only lack an MPS branch. Consolidating them is a
`hygiene_*` spec's work and would widen this diff past what the tests need.

## 5. CI and local wiring

- The `test` job in `ci.yml` gains one step after the coverage run: `pytest tests/e2e -m "e2e and
  not ui and not slow"`. It reuses the job's `.[dev,neural,api]` install, hermetic env, and
  `STRICT_OPTIONAL_DEPS`, and adds **no new `uses:`** — so the action-pin ratchet
  (`.github/action_pin_baseline.json`) is untouched. The e2e run is deliberately outside the
  `--cov` invocation: E2E coverage must not dilute or inflate the unit denominator (evidence
  program rule R3).
- `e2e_with_langsmith.yml` keeps running the directory post-merge; every new module
  `importorskip`s torch so the `.[dev]`-only install there degrades to skips, not errors.
- `Makefile` gains `test-e2e` with help text (the Makefile-documentation invariant) and `test-all`
  keeps covering it.
- `tests/README.md`'s E2E section is rewritten around the contract in §1, and its stale
  "minimum 50% coverage" line is corrected to point at `pyproject.toml`.

## 6. Acceptance criteria for this program

- AC-1: every module in §3 passes on a CPU-only host under `.[dev,neural,api]`, in under the CI job
  budget, with the `gpu` cases reported as skipped, not failed.
- AC-2: the same modules pass unchanged on a CUDA host with `cuda` in the matrix (verified by the
  maintainer on GPU hardware; this branch was developed on CPU and says so).
- AC-3: the full local gate is green: `black --check`, `ruff`, `mypy src/`, the unit suite at
  ≥85% branch coverage, `validate-context-docs`, `claim-ledger`, `harness validate-spec`, and the
  workflow-invariant tests.
- AC-4: no new hard-coded device string under `tests/e2e/`; `grep -rn '"cuda"' tests/e2e` returns
  only the marker registration and the availability probe.
- AC-5: `CHANGELOG.md` `[Unreleased]` records the suite, the CI step, and the `src/` fix.

## 7. Risks

| Risk | Mitigation |
|---|---|
| The two-rank gloo run hangs when one rank dies | Both children run under one timeout; on expiry the test kills both and fails with both logs attached. `MASTER_PORT` is a free port bound and released immediately before the spawn. |
| CUDA runs are slower than the CI budget | The CI runner has no GPU, so the accelerator cases skip there; the matrix is only widened where hardware exists. Work sizes (`--iterations 1`, `--num-simulations` in the single digits) are the same the existing integration smoke uses. |
| MPS lacks an op the network needs | The `mps` case is selected by availability; a failure there is a real finding about the `NG-6` single-accelerator promise, not a flake, and is reported as such. |
| The REST E2E reaches a real provider | The LLM factory is refused at the boundary and the test asserts `/ready` reports `framework_degraded=true`; a test that ever sees `framework_degraded=false` fails, which is the guard against silently using a live client. |
