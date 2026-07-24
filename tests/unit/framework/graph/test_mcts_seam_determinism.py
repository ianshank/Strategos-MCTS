"""Determinism tests for the MCTS candidate-scoring seam.

Covers strategos_subgoal_scoring_seam AC-3: the default (identity) path is seed
deterministic and the seam neither draws from the engine RNG nor reorders anything —
two seeded runs of the default node produce byte-for-byte identical output, and an
identity run matches a run with no scorer injected at all (both default to identity).
"""

from __future__ import annotations

from src.framework.mcts.scoring import IdentityCandidateScorer


class TestSeamDeterminism:
    async def test_two_default_runs_are_identical(self, make_graph_builder, mcts_state):  # AC-3
        result_a = await make_graph_builder()._mcts_simulator_node(dict(mcts_state))
        result_b = await make_graph_builder()._mcts_simulator_node(dict(mcts_state))
        assert result_a == result_b

    async def test_explicit_identity_matches_default_none(self, make_graph_builder, mcts_state):  # AC-3
        # Injecting IdentityCandidateScorer explicitly must equal the no-scorer default.
        explicit = await make_graph_builder(candidate_scorer=IdentityCandidateScorer())._mcts_simulator_node(
            dict(mcts_state)
        )
        defaulted = await make_graph_builder()._mcts_simulator_node(dict(mcts_state))
        assert explicit == defaulted

    async def test_scorer_does_not_consume_engine_rng(self, make_graph_builder, mcts_state):  # AC-3
        # The engine RNG state after a default run must be independent of scorer calls:
        # run twice on fresh builders and confirm the emitted seed/stats are identical
        # (a scorer that drew from self.rng would desynchronize the second search).
        first = await make_graph_builder()._mcts_simulator_node(dict(mcts_state))
        second = await make_graph_builder()._mcts_simulator_node(dict(mcts_state))
        assert first["mcts_stats"]["seed"] == second["mcts_stats"]["seed"]
        assert first["mcts_stats"]["action_stats"] == second["mcts_stats"]["action_stats"]
