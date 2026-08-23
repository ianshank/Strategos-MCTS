---
id: hygiene_mcts_value_semantics
goal: Fix the proven MCTS value-perspective bugs (PUCT double-division; negamax selection sign in parallel and progressive-widening engines; unconditional backup negation in those same engines and absent negation in core)
module: src/framework/mcts/
status: approved
---

# Goal

select_child_puct divides an already-averaged Q by visits again (neural_policies.py), and the
parallel/progressive-widening engines flip value sign per level during backpropagation but
select on the un-negated child value, choosing the opponent-best move. The backup side is
symmetrically broken and was previously understated here: `parallel_mcts.py:535-539` and
`progressive_widening.py:470-471` negate unconditionally, ignoring the very `two_player` flag their
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
- AC-6: The two-player perspective flag is honoured on the **backup** path as well as the selection path. `parallel_mcts` (currently negating unconditionally) and `progressive_widening` (likewise) must not negate when the flag is disabled, and `core.MCTSNode.backpropagate` (currently never negating) must negate when it is enabled. A regression test asserts, for each of the four engines, that a single-agent backup accumulates a monotone value and a two-player backup alternates sign; the current tree fails this test for three of the four.
- AC-7: Cross-engine backup parity is asserted directly, not inferred from root-action agreement: for a seeded fixed tree and a fixed leaf value, all four engines produce identical per-node value sums in both flag settings.

# Constraints

- No symbol renames or moves: open approved specs cite this module.
- Implemented under a human-approved No-Spec exception (module overlaps the open strategos_risk_averse_subgoal_scorer spec); must land before that spec's implementation begins.
- Backward compatible; no hardcoded values (tunables via src/config/settings.py or constants modules).
- Full local quality gate green before push (black 120 / ruff / mypy src/ / pytest --cov-fail-under=85 / secret grep).
- No real network or API calls in unit tests; mock all I/O.
- CHANGELOG [Unreleased] entry; MIGRATION_NOTES entry for any behavior change.

# Out of Scope

- Engine/config consolidation (hygiene_mcts_policies/engines/config).
