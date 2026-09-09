---
id: local_distillation_contract
goal: Hygienic NeuralMCTS teacher labels under scripts/local_distillation (visit/sum π, STM z, eval+cache) with a default-off recurrent student and a promotion reject — not Unsloth, not llm_guided, not a golden_path_connect_four claim
module: scripts/local_distillation/
status: draft
---

# Goal

Keep explicit NeuralMCTS plus Connect Four terminals in charge of distillation labels. The first deliverable is honest π/z (eval mode, cache invalidation, side-to-move z, visit/sum targets), a versioned trajectory schema with game-id splits, a default-off recurrent PolicyValue-compatible student, and a promotion function that can reject a degraded checkpoint. Language/Unsloth is out of scope until CHARTER §7.

# Acceptance Criteria

- AC-1: `policy_target` equals visit/sum (`τ=1`) independent of play temperature. After `temperature_threshold`, stored π is not `N^{1/τ_final}`. Falsified if labels match `get_action_probs(temperature_final)` on a skewed visit vector. Intended test: `tests/unit/scripts/local_distillation/test_pi_train.py` and `tests/unit/scripts/local_distillation/test_collector.py`.
- AC-2: `HygienicCollector.play_game` runs `network.eval()` for every search forward; `train()` is restored afterward and used only inside `HygienicTrainer.train_step`. Falsified if BatchNorm/Dropout forwards during search see `training=True`. Intended test: `tests/unit/scripts/local_distillation/test_eval_mode_search.py`.
- AC-3: `HygienicTrainer.train_step` calls `mcts.clear_cache()` after the optimizer step (not `len(cache) > 10000`). Falsified if a pre-step cache entry survives. Intended test: `tests/unit/scripts/local_distillation/test_cache_invalidation.py`.
- AC-4: `value_target = terminal.get_reward(player=row.current_player)` including a midgame root with `current_player=-1`. Falsified if z follows an independent counter that starts at 1. Intended test: `tests/unit/scripts/local_distillation/test_stm_value_targets.py`.
- AC-5: The package does not import `llm_guided`, Unsloth, Hugging Face Hub, `HRMAgent`, or `TRMAgent`. Intended test: `tests/unit/scripts/local_distillation/test_no_hub_network.py`.
- AC-6: Splits are grouped by `game_id`. Overlapping state hashes across train/val/test fail. Unvisited legal actions remain 0 in π (rows are kept); all-zero/NaN rows are rejected; empty legal yields empty π. Intended test: `tests/unit/scripts/local_distillation/test_splits.py`.
- AC-7: Checkpoint sidecars pin `type=resnet`, `input_channels=3`, `board_rows=6`, `board_cols=7`, `action_size=7`, plus `num_res_blocks` and `num_channels`. Missing `board_rows` is an error; `chess_default_architecture` is forbidden. Intended test: `tests/unit/scripts/local_distillation/test_sidecar.py`.
- AC-8: `build_student` defaults to `PolicyValueNetwork`. With `LOCAL_DISTILLATION_RECURRENT_ENABLED=true` it returns `RecurrentPolicyValue` whose `forward(x) -> (log_probs, value)` duck-types `PolicyValueNetwork` (log-softmax + tanh). It does not wrap `HRMAgent`/`TRMAgent`. Student outputs are finite. Intended test: `tests/unit/scripts/local_distillation/test_recurrent_student.py`.
- AC-9: `compare_search_vs_no_search` exposes a 0-expansion greedy-net arm and a NeuralMCTS arm at a declared simulation count, each with wall-clock and an `EVIDENCE_PROVENANCES` label, plus a wall-clock-matched no-search repeat count. Intended test: `tests/unit/scripts/local_distillation/test_eval_arms.py`.
- AC-10: `decide_promotion` rejects a synthetic degraded checkpoint (`candidate < incumbent + min_delta`). Intended test: `tests/unit/scripts/local_distillation/test_eval_arms.py`.
- AC-11: The primary experiment endpoint is declared as `search_minus_no_search` before any holdout run. Committed results, if an experiment PR lands later, belong at `benchmarks/results/local_distillation_c4.json` — not `artifacts/`, not `self-play-convergence` plumbing, not a 2-sim Connect Four `policy-lift`. This spec does not commit that JSON. Intended test: `tests/unit/scripts/local_distillation/test_eval_arms.py`.

# Constraints

- Teacher engine is `NeuralMCTS` only (`single_agent=False` on Connect Four). Core / parallel / progressive-widening engines are forbidden teachers.
- Knobs come from `DistillationSettings` (`LOCAL_DISTILLATION_*`). No new seed env var; reuse `Settings.SEED` / `new_rng`.
- Unit tests are CPU-only, no real network/API/Hub calls, no 2-sim Connect Four plumbing numbers treated as lift.
- Student `forward` returns log-probabilities, matching `AlphaZeroLoss`. Do not change `evaluate_state`'s `(policy, value)` signature.
- `MCTSExample` is not required to grow a `game_id`; the scripts schema carries lineage.

# Invariants

- π_train is visit/sum. Play may use `temperature_init` / `temperature_final`. Passing `τ=1` into `search()` is not a substitute unless play is sampled from `root.get_action_probs(τ_play)`.
- Connect Four `get_reward` is valid AlphaZero z, not an independent promotion oracle.
- `artifacts/` cannot support a `PROVEN` claim. Experiment endpoints, if added later, follow `m5_policy_lift` (committed JSON, primary endpoint declared first).
- `self-play-convergence` remains the M5/E3 CLI; this spec does not claim its labels are hygienic until a No-Spec `SelfPlayTrainer` patch lands.

# Out of Scope

- `golden_path_connect_four`, Pareto search-vs-no-search as an Evidence-First E4 milestone, and a live GPU promotion rejection (E5).
- Unsloth, Qwen3 QLoRA, Ollama, HRM-Text, published TRM checkpoints, TRM-Planner teacher-cache, wrapping project `HRMAgent`/`TRMAgent`.
- Extending `src.framework.mcts.llm_guided.DistillationTrainer`.
- Language-track DAgger / frozen-teacher relabel. If that track ever starts it is after LLMHRM/LLMTRM cost is measured and CHARTER §7 ratifies Unsloth (or the work stays uncommitted under PEP 723). Serve only via existing `LMStudioClient`. `QUALITY_*` structural scores are not labels.
- Literature 25% latency / 2pp success figures as measured results.
- Chess M5 ≥20% lift reused as a Connect Four result.
- Pooling GPUs; P40 as an Unsloth learner.
