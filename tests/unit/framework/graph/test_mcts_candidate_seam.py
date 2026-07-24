"""Node-level tests for the MCTS candidate-scoring seam in ``_mcts_simulator_node``.

Covers strategos_subgoal_scoring_seam AC-1 (identity default emits the engine's own
selection unchanged) and AC-2 (an overriding scorer redirects the emitted action and
its summary/confidence). asyncio_mode is "auto", so async tests run directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock

from src.framework.mcts.scoring import CandidateRecord, IdentityCandidateScorer


class _SpyIdentityScorer:
    """Identity scorer that records the engine choice it was handed."""

    name = "spy-identity"

    def __init__(self) -> None:
        self.seen_engine_choice: str | None = "UNSET"
        self.seen_candidate_ids: list[str] = []

    def select_best(self, candidates: Sequence[CandidateRecord], *, engine_choice: str | None) -> str | None:
        self.seen_engine_choice = engine_choice
        self.seen_candidate_ids = [c.candidate_id for c in candidates]
        return engine_choice


class _FirstDifferentScorer:
    """Overriding scorer that always picks the first candidate != engine_choice."""

    name = "first-different"

    def __init__(self) -> None:
        self.engine_choice: str | None = None

    def select_best(self, candidates: Sequence[CandidateRecord], *, engine_choice: str | None) -> str | None:
        self.engine_choice = engine_choice
        for candidate in candidates:
            if candidate.candidate_id != engine_choice:
                return candidate.candidate_id
        return engine_choice


class TestIdentityDefault:
    def test_default_scorer_is_identity(self, make_graph_builder):  # AC-1
        builder = make_graph_builder()
        assert isinstance(builder.candidate_scorer, IdentityCandidateScorer)

    async def test_emits_engine_choice_unchanged(self, make_graph_builder, mcts_state):  # AC-1
        spy = _SpyIdentityScorer()
        builder = make_graph_builder(candidate_scorer=spy)

        result = await builder._mcts_simulator_node(mcts_state)

        # The node must emit exactly the engine's own selection under an identity pass-through.
        assert result["mcts_best_action"] is not None
        assert result["mcts_best_action"] == spy.seen_engine_choice
        # The seam saw the full candidate set drawn from action_stats.
        assert set(spy.seen_candidate_ids) == set(result["mcts_stats"]["action_stats"].keys())

    async def test_summary_and_confidence_track_engine_stats(self, make_graph_builder, mcts_state):  # AC-1
        builder = make_graph_builder(candidate_scorer=IdentityCandidateScorer())

        result = await builder._mcts_simulator_node(mcts_state)
        stats = result["mcts_stats"]
        output = result["agent_outputs"][0]

        # On the identity path, the emitted visit/value come straight from the engine stats.
        assert f"visits={stats['best_action_visits']}" in output["response"]
        expected_conf = min(
            stats["best_action_visits"] / stats["iterations"] if stats["iterations"] > 0 else 0.5,
            1.0,
        )
        assert output["confidence"] == expected_conf


class TestMinimalStatsRobustness:
    async def test_missing_action_stats_preserves_engine_choice(self, make_graph_builder, mcts_state):  # AC-1
        # A stubbed/minimal stats dict without 'action_stats' must not break the seam:
        # no candidates -> identity returns the engine's choice, output unchanged.
        builder = make_graph_builder()  # identity default
        builder.mcts_engine.search = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                "A",
                {
                    "iterations": 1,
                    "cache_hit_rate": 0.0,
                    "best_action_visits": 1,
                    "best_action_value": 0.5,
                },
            )
        )
        result = await builder._mcts_simulator_node(dict(mcts_state))
        assert result["mcts_best_action"] == "A"
        assert "Recommended action: A" in result["agent_outputs"][0]["response"]

    async def test_none_action_stats_preserves_engine_choice(self, make_graph_builder, mcts_state):  # AC-1
        # A non-mapping/None action_stats must not break candidate construction.
        builder = make_graph_builder()
        builder.mcts_engine.search = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                "A",
                {
                    "iterations": 1,
                    "cache_hit_rate": 0.0,
                    "best_action_visits": 1,
                    "best_action_value": 0.5,
                    "action_stats": None,
                },
            )
        )
        result = await builder._mcts_simulator_node(dict(mcts_state))
        assert result["mcts_best_action"] == "A"


class _UnknownIdScorer:
    """A misbehaving scorer that returns a candidate_id not in the candidate set."""

    name = "unknown-id"

    def select_best(self, candidates, *, engine_choice):
        return "does-not-exist"


class TestOverridingScorer:
    async def test_unknown_id_override_is_ignored(self, make_graph_builder, mcts_state):  # AC-1
        # A scorer returning an id not among the candidates must not desync the emitted
        # action from its stats; the node keeps the engine's own choice.
        builder = make_graph_builder(candidate_scorer=_UnknownIdScorer())
        result = await builder._mcts_simulator_node(mcts_state)
        assert result["mcts_best_action"] in result["mcts_stats"]["action_stats"]
        assert result["mcts_best_action"] != "does-not-exist"

    async def test_override_redirects_action_and_summary(self, make_graph_builder, mcts_state):  # AC-2
        scorer = _FirstDifferentScorer()
        builder = make_graph_builder(candidate_scorer=scorer)

        result = await builder._mcts_simulator_node(mcts_state)
        action_stats = result["mcts_stats"]["action_stats"]

        assert len(action_stats) >= 2  # sanity: the seam had a real choice to make
        chosen = result["mcts_best_action"]
        # The scorer forced a candidate other than the engine's choice.
        assert chosen != scorer.engine_choice
        assert chosen in action_stats
        # Summary + confidence must describe the *chosen* candidate, not the engine's.
        chosen_visits = action_stats[chosen]["visits"]
        chosen_value = action_stats[chosen]["value"]
        output = result["agent_outputs"][0]
        assert f"visits={chosen_visits}" in output["response"]
        assert f"value={chosen_value:.3f}" in output["response"]
        expected_conf = min(
            chosen_visits / result["mcts_stats"]["iterations"] if result["mcts_stats"]["iterations"] > 0 else 0.5,
            1.0,
        )
        assert output["confidence"] == expected_conf

    async def test_override_makes_emitted_stats_consistent(self, make_graph_builder, mcts_state):  # AC-2
        # After an override, mcts_stats must describe the *chosen* action so downstream consumers
        # (synthesis value blend, experiment tracker) don't read the engine's stale pick.
        scorer = _FirstDifferentScorer()
        builder = make_graph_builder(candidate_scorer=scorer)
        result = await builder._mcts_simulator_node(mcts_state)
        stats = result["mcts_stats"]
        chosen = result["mcts_best_action"]
        assert stats["best_action"] == chosen
        assert stats["best_action_visits"] == stats["action_stats"][chosen]["visits"]
        assert stats["best_action_value"] == stats["action_stats"][chosen]["value"]
