"""
Tests for :mod:`src.ui.agent_routes`.

The UI previously answered every query with ``asyncio.sleep()`` plus an f-string,
returning hardcoded confidences of 0.85/0.80/0.88 and a fixed list of reasoning
prose that described the architecture rather than the run that just happened.
These tests pin the replacement: the routing table is the single source of truth,
and the reasoning trace is derived from what the framework actually reported.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any

import pytest

from src.ui.agent_routes import (
    AGENT_ROUTES,
    AgentRouteSpec,
    default_route,
    derive_reasoning_steps,
)

pytestmark = pytest.mark.unit


@dataclass
class _Result:
    """Stand-in for framework_service.QueryResult."""

    agents_used: list[str] = field(default_factory=list)
    mcts_stats: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TestRoutingTable:
    def test_covers_the_three_controller_outputs(self) -> None:
        assert set(AGENT_ROUTES) == {"hrm", "trm", "mcts"}

    def test_only_the_mcts_route_enables_tree_search(self) -> None:
        assert AGENT_ROUTES["mcts"].use_mcts is True
        assert AGENT_ROUTES["hrm"].use_mcts is False
        assert AGENT_ROUTES["trm"].use_mcts is False

    def test_every_route_key_matches_its_dict_key(self) -> None:
        for key, spec in AGENT_ROUTES.items():
            assert spec.key == key

    def test_every_route_has_a_human_label(self) -> None:
        for spec in AGENT_ROUTES.values():
            assert spec.label.strip()

    def test_default_route_is_a_member_of_the_table(self) -> None:
        """An unrecognized controller output must not route somewhere unlisted."""
        assert default_route() in AGENT_ROUTES.values()

    def test_specs_are_immutable(self) -> None:
        """Routes are shared module state; a caller must not be able to mutate them."""
        with pytest.raises(FrozenInstanceError):
            AGENT_ROUTES["hrm"].use_mcts = True  # type: ignore[misc]

    def test_spec_is_constructible_for_future_routes(self) -> None:
        spec = AgentRouteSpec(key="custom", label="Custom", use_mcts=True)

        assert spec.key == "custom"
        assert spec.use_mcts


class TestReasoningStepDerivation:
    def test_framework_reported_steps_win(self) -> None:
        result = _Result(metadata={"reasoning_steps": ["Decompose", "Solve", "Verify"]})

        assert derive_reasoning_steps(result) == ["Decompose", "Solve", "Verify"]

    def test_falls_back_to_agents_used(self) -> None:
        result = _Result(agents_used=["hrm", "trm"])

        steps = derive_reasoning_steps(result)

        assert any("hrm" in s and "trm" in s for s in steps)

    def test_mcts_statistics_appear_in_the_trace(self) -> None:
        result = _Result(mcts_stats={"iterations": 100, "nodes_explored": 42, "best_action": "expand"})

        steps = derive_reasoning_steps(result)
        joined = " ".join(steps)

        assert "100" in joined
        assert "42" in joined
        assert "expand" in joined

    @pytest.mark.parametrize(
        ("stats", "expected_fragment", "absent_fragment"),
        [
            ({"iterations": 7}, "MCTS iterations: 7", "Nodes explored"),
            ({"nodes_explored": 9}, "Nodes explored: 9", "MCTS iterations"),
            ({"best_action": "hold"}, "Best action: hold", "Nodes explored"),
        ],
    )
    def test_partial_mcts_stats_report_only_what_is_present(
        self, stats: dict, expected_fragment: str, absent_fragment: str
    ) -> None:
        """
        A framework reporting some MCTS keys but not others must not invent the rest.

        Each key is independently optional; covering only the all-keys-present case
        left the absent-key branches untested, which is where a KeyError or a
        fabricated "Nodes explored: None" line would hide.
        """
        steps = derive_reasoning_steps(_Result(mcts_stats=stats))
        joined = " ".join(steps)

        assert expected_fragment in joined
        assert absent_fragment not in joined

    def test_total_nodes_is_accepted_as_an_alias(self) -> None:
        """The reader falls back to `total_nodes` when `nodes_explored` is absent."""
        steps = derive_reasoning_steps(_Result(mcts_stats={"total_nodes": 11}))

        assert any("11" in s for s in steps)

    def test_empty_mcts_stats_adds_nothing(self) -> None:
        assert derive_reasoning_steps(_Result(mcts_stats={})) == []

    def test_empty_result_yields_no_invented_steps(self) -> None:
        """The old code returned four lines of prose regardless of what ran."""
        assert derive_reasoning_steps(_Result()) == []

    def test_rag_usage_is_reported(self) -> None:
        result = _Result(metadata={"rag_context_used": True})

        assert any("RAG" in step for step in derive_reasoning_steps(result))

    def test_missing_attributes_do_not_raise(self) -> None:
        class _Bare:
            pass

        assert derive_reasoning_steps(_Bare()) == []  # type: ignore[arg-type]

    def test_non_list_reported_steps_are_ignored(self) -> None:
        result = _Result(agents_used=["hrm"], metadata={"reasoning_steps": "not a list"})

        steps = derive_reasoning_steps(result)

        assert steps and all(isinstance(s, str) for s in steps)
        assert "not a list" not in steps
