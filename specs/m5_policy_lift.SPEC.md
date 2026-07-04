---
id: m5_policy_lift
goal: Ship a resumable chess self-play convergence driver and record the M5 policy-lift gate run against its trained checkpoint
module: src/training/
milestone: M5
status: draft
---

# Goal

Provide the missing execution path for the M5 decision-quality milestone: a runnable,
resumable, seedable self-play convergence driver for the chess domain that produces
torch-safe checkpoints with `.meta.json` architecture sidecars, and a recorded run of
the existing `python -m src.benchmark.policy_lift` gate against the resulting
checkpoint. The gate target is a >=20% decision-quality lift over the untrained
policy, measured as the 95% confidence-interval lower bound of the win-rate-derived
lift (`lift_ci_lower_pct`) over >=100 chess games.

# Acceptance Criteria

- AC-1: `python -m src.training.self_play_convergence --domain chess --iterations <N> --checkpoint-dir <dir> --seed <s> --device <dev>` runs a SelfPlayTrainer loop end-to-end and writes at least one checkpoint plus its `.meta.json` sidecar into `<dir>`; invoking it again with `--resume` continues from the latest checkpoint (weights loaded, iteration numbering monotonic) instead of reinitializing. Falsified by a nonzero exit, a missing checkpoint/sidecar, or restarted iteration numbering on resume. Intended test: `tests/unit/training/test_self_play_convergence.py` (tiny network, tiny iteration/game/simulation budget, CPU only).
- AC-2: A checkpoint written by the driver round-trips through the policy-lift architecture resolution with no manual flags: its sidecar carries a `network` object that `src.benchmark.policy_lift.load_architecture` resolves to a network onto which the checkpoint's state_dict loads. Falsified by `load_architecture` raising, or by the state_dict failing to load onto the resolved network (`RuntimeError`). Intended test: `tests/unit/training/test_self_play_convergence.py` (sidecar round-trip case).
- AC-3: A run of `python -m src.benchmark.policy_lift --domain chess --checkpoint <driver-produced .pt> --num-games 100 --output benchmarks/results/m5_policy_lift.json` completes without error (exit code 0 or 1, never 2) and the JSON artifact is committed at `benchmarks/results/m5_policy_lift.json` with run provenance: domain `chess`, `num_games >= 100`, confidence `0.95`, target lift `20`, seed, and checkpoint identity. Intended test: `tests/integration/benchmark/test_m5_lift_artifact.py` (schema and provenance assertions on the committed artifact).
- AC-4: The committed artifact records the gate as met: `meets_target` is true — the 95% CI lower bound of the lift clears 20%. Intended test: `tests/integration/benchmark/test_m5_lift_gate.py` (asserts `meets_target` on the committed artifact; this test lands together with the `verified` flip so a not-yet-met gate never breaks CI).

# Constraints

- Checkpoints remain torch-safe `state_dict` files (no pickle); the sidecar is written via `SelfPlayTrainer.save_checkpoint(metadata=...)`.
- The driver reuses existing building blocks — `DomainRegistry.get("chess")`, `SelfPlayTrainer`, and the network builders in `src.benchmark.policy_lift` — so no parallel network-construction path can drift from the gate's resolver.
- Unit tests must pass on CPU without the compute-bound convergence run.
- No changes to `src/benchmark/` gate semantics (thresholds, CI math, exit codes).

# Invariants

- `python -m src.benchmark.policy_lift` remains the sole authority for the >=20% claim; the driver never computes or asserts lift itself.
- Reasoning/planning domains remain smoke-test-only; this spec claims nothing beyond chess.

# Out of Scope

- Meta-controller training and the LLM curriculum (`scripts/run_comprehensive_training.py`).
- GPU provisioning and run scheduling (operational, not contractual).
- The PreToolUse gate's warn->block flip and Phase 3 plugin packaging.
