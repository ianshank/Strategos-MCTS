---
id: hygiene_mcts_value_semantics
goal: Fix the proven MCTS value-perspective bugs (PUCT double-division; negamax selection sign in parallel and progressive-widening engines; unconditional backup negation in those same engines and absent negation in core)
module: src/framework/mcts/
status: implemented
---

# Goal

select_child_puct divides an already-averaged Q by visits again (neural_policies.py), and the
parallel/progressive-widening engines flip value sign per level during backpropagation but
select on the un-negated child value, choosing the opponent-best move. The backup side is
symmetrically broken and was previously understated here: `parallel_mcts.py:535-539` and
`progressive_widening.py:472` negate unconditionally, ignoring the very `two_player` flag their
selection paths consult (and which `progressive_widening.py:293` documents as controlling backprop),
while `core.py:377-393` never negates and exposes no flag at all. Adopt the proven
negate_child_value pattern from neural_mcts.py, with the two-player perspective as an explicit
config field so single-agent search gets a coherent non-negating pair.

# Acceptance Criteria

- AC-1: A regression suite ported from the executable proofs shows all three engines select the minimax-optimal child on a seeded 2-ply tree.
- AC-2: Cross-engine parity: core, parallel, and progressive-widening engines agree on the root action for seeded small states in single-agent mode.
- AC-3: select_child_puct agrees with the canonical puct() on 1,000 seeded random inputs.
- AC-4: Fixed selection paths emit per-child DEBUG structured logs (visits, mean value, exploration term) via the project logger.
- AC-5: Affected benchmark baselines are re-run and re-recorded (or explicitly flagged for re-validation where environment-bound); MIGRATION_NOTES documents the intentional absence of an escape hatch to the broken behavior.
- AC-6: The two-player perspective flag is honoured on the **backup** path as well as the selection path. `parallel_mcts` and `progressive_widening` must not negate when the flag is disabled, and `core.MCTSEngine.backpropagate` must negate when it is enabled. A regression test asserts, for each of the four engines, that a single-agent backup accumulates a monotone value and a two-player backup alternates sign.
- AC-7: Cross-engine backup parity is asserted directly, not inferred from root-action agreement: for a seeded fixed tree and a fixed leaf value, all four engines produce identical per-node value sums in both flag settings.
- AC-8: Parent AMAF/RAVE is parent side-to-move after two-player backup. `select_child_rave` must not negate that table again; with β dominating, RAVE agrees with negated UCB on the same child. The regression must read the **parent** RAVE table (not stuffed child dicts).
- AC-9: Parallel virtual loss deters under `negate_child_value=True`. Adding VL to the UCB-best child must not make that child win; the same tree with `negate_child_value=False` still deters.
- AC-10: `MAX_VALUE`, `ROBUST_CHILD`, and `action_stats["value"]` (consumed by `ValueCandidateScorer`) use parent-perspective Q when `two_player` is set. A two-player root with child A Q=+0.9 and child B Q=+0.1 must return B.
- AC-11: `NeuralMCTS()` without `single_agent` binds `single_agent=not Settings.MCTS_TWO_PLAYER`. `RootParallelMCTSEngine` and `create_parallel_mcts(..., strategy="root")` forward `two_player` into worker `MCTSEngine` instances.

# Constraints

- No symbol renames or moves: open approved specs cite this module.
- Land on branch `spec/hygiene_mcts_value_semantics`. `harness spec-trace` matches the branch suffix to this spec; it does not check module uniqueness. Do not add a `No-Spec:` trailer — a trailer short-circuits the approved→implemented flip. Cite module overlap with `strategos_risk_averse_subgoal_scorer` in the PR body. Must land before that spec's implementation begins.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.

# Out of Scope

- Engine/config consolidation (hygiene_mcts_policies/engines/config).
