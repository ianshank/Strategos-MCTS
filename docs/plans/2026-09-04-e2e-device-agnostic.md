# E2E test program — device-agnostic end-to-end coverage for the 2026 H2 work

> **Status:** v2, implemented. v1 was a proposal; this version records what was built,
> what two adversarial reviews cut from it, and the defects the work surfaced.
>
> **Axis:** sequenced work (what next, in what order). This file plans and records *tests*.
> It asserts no measured status — `docs/STATUS.md` owns that — and it promotes no grade in
> `docs/CLAIM_LEDGER.md`: a test is not an evidence artefact, and only the `eval-warden`
> referee moves a grade.
>
> **Provenance:** three independent surveys of the tree at `30a75bf` (`main`) — the
> existing E2E suite and its CI gating, the work landed since June 2026, and every
> device-selection site under `src/` and `tests/` — followed by two adversarial reviews of
> the v1 plan (a product/value lens and an SDLC/CI-governance lens). Both are reflected in
> §2 and §6. Every measurement below was taken on the CPU-only development host; nothing
> here is claimed for GPU hardware (§7).

---

## 0. The verdict that motivated this

The suite named `tests/e2e/` was neither end-to-end nor gating.

Of its 95 tests, 27 reimplemented UCB1 in a dictionary and imported nothing from `src/`,
26 touched `src` only through the `QueryInput` validator, and 27 needed `gradio`. The one
module that drove real components hard-coded `device="cpu"`. No PR-gating workflow ran the
directory at all: `e2e_with_langsmith.yml` is the only workflow that did, and every job in
it is gated on a `LANGSMITH_API_KEY` secret, so on a fork or a secret-less clone the whole
suite silently did not run and could not block a merge.

Meanwhile the work landed since June — the self-play convergence driver and policy-lift
gate, the DDP orchestrator, the evidence tooling, the REST hardening, the training-container
healthcheck — is unit-tested to a high standard and had **no** test that spawned its real
entry points. And not one test in the repository was parametrized over devices, so
`CHARTER.md` INV-9's promise that "CPU-only and single-GPU paths keep working" rested on CI
happening to run CPU-only.

## 1. What "E2E" means here, as a contract

A module belongs in `tests/e2e/` only if all four hold.

1. **Real entry point.** It drives the thing the way a user does — an installed console
   script, `python -m`, the FastAPI app through its lifespan — never a private function and
   never a re-implementation of the algorithm.
2. **Real components, mocked boundary.** Everything inside the process is real. The only
   stand-ins are at the network boundary, reached through the paths the code already offers
   (`ALLOW_MOCK_LLM_FALLBACK`, the offline hub flags). No socket leaves the machine;
   loopback is used by the two-rank rendezvous and nothing else. `CHARTER.md` NG-8 states
   this for unit tests, and this program applies it to E2E because a suite whose verdict
   depends on the network gates nothing.
3. **Device-agnostic by construction.** Anything that moves a tensor is parametrized over
   the device matrix from one shared helper. Accelerator selection is never hard-coded.
4. **Bounded and self-explaining.** Every subprocess has a timeout and is killed as a
   process group on expiry; every stochastic path is seeded through the entry point's own
   channel; a failure prints argv, exit code, both streams, and the resolved device.

## 2. What the adversarial reviews changed

v1 proposed six modules. Two reviews — one on product value, one on CI governance —
independently concluded it was padded, and both found defects that would have made a green
run meaningless. The plan was cut and corrected rather than defended.

| v1 module | Outcome | Why |
|---|---|---|
| B1 self-play golden path | **Kept, narrowed and re-pointed** | `tests/integration/benchmark/test_self_play_convergence_e2e.py` already drives this pipeline in-process on CPU. Only four things were genuinely new, so only those are asserted (§3 E1). Moved from the synthetic `reasoning` domain to `connect_four`, which is adversarial and is the evidence program's E3 golden path. |
| B2 neural MCTS device parity | **Kept, restructured** | The cross-device comparison needs *two* devices, so it lives in its own test with its own fixture; inside a per-device parametrization it would have self-skipped on `[cpu]` on every host. TF32 is disabled and a tolerance is stated, or the assertion would flake on Ampere. |
| B3 two-rank DDP | **Kept, promoted to first priority, and fixed** | The only module that can move a charter gate (§5 G-M1). As specified in v1 it would have **passed without ever forming a process group** — see §4. |
| B4 REST lifespan | **Kept, network defect fixed** | v1 kept a provider key in the child environment, so LLM-client creation would have *succeeded*, `framework_degraded` would have been false, and `/query` would have attempted a real call with a fake key. The refusal is now explicit and asserted before any query is issued. |
| B5 evidence-chain CLIs | **Cut, one part kept** | Every command it proposed is already run against the real tree by the CI `spec-validate` job and by unit tests. It also asserted a "hardware record" in `artifacts/status.json` that does not exist. What survived is the genuine gap: nothing invoked the eight declared console scripts *by their installed names*. |
| B6 healthcheck | **Cut to two boundary cases** | `tests/unit/test_healthcheck_exit_codes.py` and `test_healthcheck_cuda.py` already cover the branches. Worse, as specified it would have called a provider: with a key in the environment the script's LLM check makes a live request, and this suite also runs in the post-merge workflow that exports **real** secrets. Now runs with every credential stripped. |

Two v1 claims were withdrawn as overreach: "closes `ddp_orchestrator` AC-3" (a two-rank
run with identical seeds proves the group formed and that I/O is fenced, not that
gradients are averaged), and the suggestion that these tests close ledger rows.

## 3. What was built

Shared infrastructure:

- **`tests/utils/device_matrix.py`** — the single answer to "which devices does this host
  offer, and which did the operator ask for". The matrix is **static** (`cpu`, `cuda`,
  `mps`), so on a CPU-only runner the accelerator cases are reported as *skipped with a
  reason* rather than being absent. That distinction is the point: "88 passed" on a CPU
  host must not read as "the CUDA path is tested". Non-CPU cases carry the `gpu` marker, so
  `-m "not gpu"` deselects them.
- **`tests/utils/e2e_process.py`** — hermetic subprocess execution. Strips every provider
  and tracker credential before a child starts, pins the offline posture, bounds every
  child, kills whole process groups on timeout, and renders a failure as argv + exit code +
  both streams.
- **`tests/e2e/conftest.py`** — the fixtures: `device_case`/`device`, `accelerator_case`,
  `e2e_seed`, `e2e_env`, `run_module`, `run_script`. It stays importable without the
  `neural` extra, because it is imported for the whole directory including the torch-free
  modules.

The modules:

| ID | Module | Drives | Devices | What is new |
|---|---|---|---|---|
| E1 | `test_self_play_golden_path_e2e.py` | `self-play-convergence` → `--resume` → `policy-lift`, as installed console scripts on `connect_four` | matrix | the process boundary; `--device` on every device; a checkpoint written on any device loading under `policy-lift --device cpu`; fresh-process seeded reproducibility (`hygiene_determinism` AC-3's fresh-process half, asserted bitwise on CPU only) |
| E2 | `test_neural_mcts_device_e2e.py` | `DomainRegistry` → `build_network` → `NeuralMCTS.search` | matrix + accelerator | a real search returns a normalized legal policy on every device; same-device seeded reproducibility; accelerator/CPU forward-pass agreement with TF32 off |
| E3 | `test_ddp_two_rank_cpu_e2e.py` | two driver subprocesses under a real `gloo` process group | CPU by design | the first process group ever formed by a test in this repository; rank-0 I/O fencing (`ddp_orchestrator` AC-4) proven by rank 1's directory staying empty |
| E4 | `test_rest_api_e2e.py` | the FastAPI app through its own lifespan | n/a | `/graph/structure` and `/graph/mermaid` served by a real built graph; `framework_degraded` asserted before any query, which is both new coverage and the network guard |
| E5 | `test_operational_entry_points_e2e.py` | the eight declared console scripts; `healthcheck.py` as a process | matrix-aware | a broken `module:function` target in `[project.scripts]` was previously invisible until a user typed the command; the healthcheck's exit-code contract at the boundary Docker actually observes |

Measured on the CPU-only development host: **88 passed, 8 skipped, 27 deselected in 55 s**
(`pytest tests/e2e -m "not ui"`, under the CI test job's environment). The 8 skips are the
`cuda` and `mps` cases, each naming the device and how to require it.

Out of scope, with the reason: the Hugging Face Space deploy path (force-pushes to an
external service; its NG-1 ruling is unratified), a two-rank **NCCL** run (needs two GPUs),
the Gradio UI (`tests/e2e/test_ui_e2e.py` and the `ui-tests` job own it), and chess
(`tests/games/chess/unit/` is documented bit-rot, owned by `hygiene_chess_consolidation`).

## 4. Defects this work surfaced

Recorded here rather than silently worked around. None is fixed by this program except the
first.

1. **`TrainerConfig.from_settings` hard-selected `"cuda"`** from the MCTS implementation
   flag, with no availability check and no consultation of `TORCH_DEVICE_OVERRIDE` — the
   one device knob `Settings` offers. On a CPU-only or Apple-silicon host with
   `MCTS_IMPL=neural` it produced a device string that fails at the first tensor move.
   **Fixed** in `src/framework/component_factory/configs.py` via a named
   `TrainerConfig.resolve_device`, routed through `src/utils/device.py` in the same order
   `src/training/system_config.py` uses. A CUDA host resolves to `cuda` exactly as before
   (`CHARTER.md` NG-6). It lands in its own commit with a `No-Spec:` trailer, which is
   NG-4's own written-exception clause; it consumes no §8 carve-out budget, and none is
   claimed.
2. **A failed process-group init degrades silently.** `src/utils/distributed.py`'s
   `init_distributed` catches the failure and returns `False`;
   `src/training/self_play_convergence.py` ignores the return value. Verified: launching
   two ranks with the default `nccl` backend on a CPU-only host produces two independent
   single-process runs that **both** write `ckpt_iter_1.pt` and both exit 0. An operator
   who mis-set the backend would get half the data and no error. This looks like an NG-2
   ("no silent fallback") violation in the distributed path. Not fixed here: `src/training/`
   is claimed by open approved specs, and the fix is a decision about failure policy, not a
   test change. E3 pins the correct behaviour by setting `TRAINING_BACKEND=gloo` explicitly.
3. **`NeuralMCTS` search is irreproducible under a torch-only seed.** Root Dirichlet noise
   and stochastic action selection draw from the process-global NumPy RNG. Already known —
   `EVIDENCE_FIRST_PROGRAM.md` §2.5, and `specs/hygiene_determinism.SPEC.md` AC-3 (approved,
   unimplemented) specifies the injected-generator fix. E2 seeds both RNGs, which is what
   the self-play driver does, and says so at the seam so the coupling is visible.
4. **BatchNorm networks are a DDP hazard.** DDP broadcasts buffers on every forward, so with
   per-rank self-play trajectories of differing length a buffer-carrying network can
   deadlock mid-search. E3 uses the MLP-based `reasoning` domain to stay out of that path;
   it is a real question for `ddp_orchestrator` at scale.
5. **Two Makefile-target parsers disagreed on digits.** The `.PHONY` parser splits on
   whitespace; the documentation regex in the invariant test and the `make help` recipe both
   used `[a-zA-Z_-]+`, so any target with a digit in its name (`test-e2e`) was reported
   undocumented and was invisible in `make help`. Both fixed.

## 5. CI and local wiring

- The `test` job in `ci.yml` gains one step, `Run end-to-end tests`, placed after the status
  artifact so an e2e failure still leaves one behind. It reuses the job's
  `.[dev,neural,api]` install and hermetic environment. **A step, not a job**: a new job
  would need `actions/checkout` and `actions/setup-python`, and the ratchet in
  `.github/action_pin_baseline.json` permits only decreases in unpinned uses. The suite is
  selected **by directory** (`pytest tests/e2e -m "not ui"`), not by `-m e2e`, so a new
  module whose author forgets `pytestmark` still runs.
- The junit report is uploaded, because it carries the per-device skip reasons. That upload
  is **SHA-pinned** (`actions/upload-artifact` v4.6.2) rather than tag-pinned: a ninth
  unpinned use of that action would fail the ratchet, and pinning is the direction CL-33
  wants anyway.
- `tests/unit/test_ci_workflow_invariants.py` gains
  `test_the_end_to_end_suite_runs_on_pull_requests`, so the step cannot be quietly deleted.
- The e2e run is deliberately **outside** the `--cov` invocation: E2E coverage must not move
  the unit denominator in either direction (evidence-program R3). E2E coverage is therefore
  not measured, and this file says so rather than implying it is.
- `make test-e2e` runs exactly what CI runs. `E2E_DEVICES` pins the matrix.

## 6. Honest limits

- **`E2E_DEVICES` semantics.** Unset, the matrix is all three devices and unavailable ones
  skip with a reason. Set, the named devices are **required**: one the host lacks fails the
  test rather than skipping, because a pinned matrix that cannot be honoured is a broken
  host. This is a test-side variable only; nothing under `src/` reads it. It is unrelated to
  `FORCE_GPU_TESTS`, which makes `tests/deployment/test_docker_smoke.py` treat a GPU as
  present for the Docker probe.
- **Cross-device determinism is not asserted, and must not be.** CPU and accelerator kernels
  reduce in different orders. What is asserted is same-device reproducibility, valid output
  on every device, and cross-device forward-pass agreement within a stated tolerance.
- **Bitwise reproducibility is asserted on CPU only.** The driver sets neither
  `cudnn.deterministic` nor `CUBLAS_WORKSPACE_CONFIG`, so demanding it on CUDA would assert
  a property the code does not claim.
- **No GPU verification happened.** This branch was developed and measured on a CPU-only
  host. The `cuda` and `mps` cases are written and selected by availability; they are
  **unverified**. Under NG-3 that is stated rather than implied: running
  `E2E_DEVICES=cuda make test-e2e` on GPU hardware is what would verify them, and no
  document should claim the GPU path is tested until an artefact from such a run exists.

## 7. Acceptance criteria

Each is a command, not a judgement.

- AC-1: `make test-e2e` is green on a CPU-only host with the accelerator cases reported as
  skipped. *Measured: 88 passed, 8 skipped, 55 s.*
- AC-2: `E2E_DEVICES=cuda make test-e2e` on CUDA hardware is green. **Not run** — see §6. An
  operator verifying it should commit the junit report as the artefact.
- AC-3: `E2E_DEVICES=cuda make test-e2e` on a host *without* CUDA **fails** rather than
  skipping. *Verified.*
- AC-4: the full local gate is green — `black --check`, `ruff check .`, `mypy src/`, the
  unit suite at ≥85% branch coverage, `validate-context-docs`, `claim-ledger`,
  `action-pins`, `harness validate-spec`.
- AC-5: `grep -rnE "['\"](cuda|mps)" tests/e2e` returns exactly two lines, both in
  `test_operational_entry_points_e2e.py`, and both are lookups of the healthcheck report's
  `"cuda"` *check name* rather than a device string. No test selects a device by literal;
  every device comes from the matrix helper. *Verified.*
- AC-6: deleting the CI step makes `tests/unit/test_ci_workflow_invariants.py` fail.
