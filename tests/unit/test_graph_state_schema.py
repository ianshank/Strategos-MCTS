"""Unit tests for construction-time graph state/topology validation.

Covers ``src/framework/graph/schema.py`` — the pure-stdlib validators that make a
malformed state schema or graph topology fail at construction time, and a malformed
initial state fail before the graph runs. Maps to spec ``strategos_langgraph_hardening``
AC-1.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, Required, TypedDict

import pytest

# Importing the schema module pulls the graph package __init__ (builder -> mcts -> numpy).
pytest.importorskip("numpy", reason="graph package import chain requires numpy")

from src.framework.graph.schema import (  # noqa: E402
    END_SENTINEL,
    GraphConstructionError,
    StateValidationError,
    required_keys,
    validate_graph_topology,
    validate_initial_state,
    validate_state_schema,
)
from src.framework.graph.state import AgentState  # noqa: E402


def _valid_state() -> dict[str, Any]:
    return {
        "query": "hello",
        "use_rag": True,
        "use_mcts": False,
        "iteration": 0,
        "max_iterations": 3,
        "agent_outputs": [],
    }


class TestRequiredKeys:
    """required_keys() derives required-ness from resolved hints, not __required_keys__."""

    def test_agent_state_required_keys(self):
        # PEP 563 (from __future__ import annotations) breaks AgentState.__required_keys__;
        # required_keys() must still return only the genuinely-required fields.
        assert required_keys(AgentState) == {
            "query",
            "use_rag",
            "use_mcts",
            "iteration",
            "max_iterations",
            "agent_outputs",
        }

    def test_not_required_excluded_required_included(self):
        class Schema(TypedDict):
            a: int
            b: NotRequired[str]
            c: Required[float]

        assert required_keys(Schema) == {"a", "c"}


class TestValidateStateSchema:
    def test_agent_state_passes(self):
        validate_state_schema(AgentState)  # does not raise

    def test_non_typeddict_rejected(self):
        class NotATypedDict:
            pass

        with pytest.raises(GraphConstructionError, match="not a TypedDict"):
            validate_state_schema(NotATypedDict)

    def test_empty_schema_rejected(self):
        class Empty(TypedDict):
            pass

        with pytest.raises(GraphConstructionError, match="no fields"):
            validate_state_schema(Empty)

    def test_reducer_without_callable_rejected(self):
        class BadReducer(TypedDict):
            channel: Annotated[list, "not-a-callable"]

        with pytest.raises(GraphConstructionError, match="no callable reducer"):
            validate_state_schema(BadReducer)

    def test_reducer_with_callable_passes(self):
        class GoodReducer(TypedDict):
            channel: Annotated[list, operator.add]

        validate_state_schema(GoodReducer)  # does not raise

    def test_unresolvable_annotation_rejected(self):
        class BadRef(TypedDict):
            x: ThisTypeDoesNotExistAnywhere  # noqa: F821

        with pytest.raises(GraphConstructionError, match="Cannot resolve annotations"):
            validate_state_schema(BadRef)


class TestValidateGraphTopology:
    def test_valid_topology_passes(self):
        validate_graph_topology(
            nodes={"a", "b"},
            edges=[("a", "b"), ("b", END_SENTINEL)],
            conditional_targets=["a"],
            entry_point="a",
        )

    def test_end_sentinel_destination_allowed(self):
        validate_graph_topology(
            nodes={"a"},
            edges=[("a", END_SENTINEL)],
            conditional_targets=[],
            entry_point="a",
        )

    def test_dangling_edge_destination_rejected(self):
        with pytest.raises(GraphConstructionError, match="destination 'ghost'"):
            validate_graph_topology(nodes={"a"}, edges=[("a", "ghost")], conditional_targets=[], entry_point="a")

    def test_unregistered_edge_source_rejected(self):
        with pytest.raises(GraphConstructionError, match="source 'ghost'"):
            validate_graph_topology(nodes={"a"}, edges=[("ghost", "a")], conditional_targets=[], entry_point="a")

    def test_dangling_conditional_target_rejected(self):
        with pytest.raises(GraphConstructionError, match="routing target 'ghost'"):
            validate_graph_topology(nodes={"a"}, edges=[], conditional_targets=["ghost"], entry_point="a")

    def test_bad_entry_point_rejected(self):
        with pytest.raises(GraphConstructionError, match="Entry point 'missing'"):
            validate_graph_topology(nodes={"a"}, edges=[], conditional_targets=[], entry_point="missing")

    def test_custom_terminal_sentinel(self):
        validate_graph_topology(
            nodes={"a"},
            edges=[("a", "STOP")],
            conditional_targets=["STOP"],
            entry_point="a",
            terminal="STOP",
        )


class TestValidateInitialState:
    def test_valid_state_passes(self):
        validate_initial_state(_valid_state())

    def test_defaults_to_agent_state(self):
        # schema omitted -> AgentState used
        validate_initial_state(_valid_state(), None)

    def test_missing_required_rejected(self):
        state = _valid_state()
        del state["query"]
        with pytest.raises(StateValidationError, match="missing required key"):
            validate_initial_state(state)

    def test_unknown_key_rejected(self):
        with pytest.raises(StateValidationError, match="unknown key"):
            validate_initial_state({**_valid_state(), "bogus": 1})

    def test_unknown_key_allowed_with_flag(self):
        validate_initial_state({**_valid_state(), "bogus": 1}, allow_extra_keys=True)

    def test_type_mismatch_rejected(self):
        with pytest.raises(StateValidationError, match="expected type"):
            validate_initial_state({**_valid_state(), "query": 123})

    def test_bool_accepted_where_int_expected(self):
        # bool is a subclass of int; iteration: int accepts a bool.
        validate_initial_state({**_valid_state(), "iteration": True})

    def test_int_rejected_where_bool_expected(self):
        # use_rag: bool must reject a plain int.
        with pytest.raises(StateValidationError):
            validate_initial_state({**_valid_state(), "use_rag": 1})

    def test_optional_keys_accepted_when_present(self):
        validate_initial_state({**_valid_state(), "rag_context": "ctx", "mcts_iterations": 5})

    def test_optional_key_wrong_type_rejected(self):
        with pytest.raises(StateValidationError):
            validate_initial_state({**_valid_state(), "rag_context": 123})

    def test_container_origin_checked(self):
        # retrieved_docs: NotRequired[list[dict]] -> a dict value is rejected at container level.
        with pytest.raises(StateValidationError):
            validate_initial_state({**_valid_state(), "retrieved_docs": {"not": "a list"}})

    def test_any_typed_key_accepts_anything(self):
        # mcts_root: NotRequired[Any] accepts arbitrary values.
        validate_initial_state({**_valid_state(), "mcts_root": object()})

    def test_non_mapping_rejected(self):
        with pytest.raises(StateValidationError, match="must be a mapping"):
            validate_initial_state(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_union_type_accepts_either_member(self):
        class UnionSchema(TypedDict):
            v: NotRequired[str | None]

        validate_initial_state({"v": None}, UnionSchema)
        validate_initial_state({"v": "x"}, UnionSchema)
        with pytest.raises(StateValidationError):
            validate_initial_state({"v": 123}, UnionSchema)
