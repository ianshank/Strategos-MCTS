"""
Unit tests for src/neuro_symbolic/reasoning.py.

Tests Predicate, Rule, ProofStep, ProofTree, Proof, LogicEngine,
SymbolicReasoner, and SymbolicReasoningAgent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.neuro_symbolic.config import LogicEngineConfig, NeuroSymbolicConfig
from src.neuro_symbolic.reasoning import (
    LogicEngine,
    Predicate,
    Proof,
    ProofStatus,
    ProofStep,
    ProofTree,
    Rule,
    SymbolicReasoner,
    SymbolicReasoningAgent,
)
from src.neuro_symbolic.state import Fact, NeuroSymbolicState, SymbolicFactType

# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredicate:
    def test_to_string(self):
        p = Predicate(name="parent", arguments=("john", "mary"))
        assert p.to_string() == "parent(john, mary)"

    def test_to_string_negated(self):
        p = Predicate(name="likes", arguments=("a",), negated=True)
        assert p.to_string() == "\\+likes(a)"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Predicate name cannot be empty"):
            Predicate(name="", arguments=("x",))

    def test_is_ground_true(self):
        p = Predicate(name="age", arguments=("alice", 30))
        assert p.is_ground() is True

    def test_is_ground_false(self):
        p = Predicate(name="age", arguments=("?X", 30))
        assert p.is_ground() is False

    def test_get_variables(self):
        p = Predicate(name="rel", arguments=("?X", "const", "?Y"))
        assert p.get_variables() == {"X", "Y"}

    def test_get_variables_empty(self):
        p = Predicate(name="rel", arguments=("a", "b"))
        assert p.get_variables() == set()

    def test_substitute(self):
        p = Predicate(name="parent", arguments=("?X", "?Y"))
        result = p.substitute({"X": "alice"})
        assert result.arguments == ("alice", "?Y")
        assert result.name == "parent"

    def test_substitute_no_matching_vars(self):
        p = Predicate(name="f", arguments=("a", "b"))
        result = p.substitute({"X": "z"})
        assert result.arguments == ("a", "b")

    def test_to_fact_ground(self):
        p = Predicate(name="color", arguments=("sky", "blue"))
        fact = p.to_fact()
        assert fact.name == "color"
        assert fact.arguments == ("sky", "blue")
        assert fact.fact_type == SymbolicFactType.PREDICATE

    def test_to_fact_with_variable_raises(self):
        p = Predicate(name="color", arguments=("?X", "blue"))
        with pytest.raises(ValueError, match="variables"):
            p.to_fact()


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRule:
    def test_to_string(self):
        head = Predicate(name="grandparent", arguments=("?X", "?Z"))
        body = [
            Predicate(name="parent", arguments=("?X", "?Y")),
            Predicate(name="parent", arguments=("?Y", "?Z")),
        ]
        rule = Rule(rule_id="r1", head=head, body=body)
        s = rule.to_string()
        assert "grandparent(?X, ?Z)" in s
        assert "parent(?X, ?Y)" in s
        assert "parent(?Y, ?Z)" in s

    def test_get_all_variables(self):
        head = Predicate(name="gp", arguments=("?A", "?C"))
        body = [Predicate(name="p", arguments=("?A", "?B")), Predicate(name="p", arguments=("?B", "?C"))]
        rule = Rule(rule_id="r1", head=head, body=body)
        assert rule.get_all_variables() == {"A", "B", "C"}

    def test_confidence_default(self):
        rule = Rule(rule_id="r1", head=Predicate(name="x", arguments=()), body=[])
        assert rule.confidence == 1.0


# ---------------------------------------------------------------------------
# ProofStep and ProofTree
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProofStepAndTree:
    def test_proof_step_to_dict(self):
        step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="f", arguments=("a",)),
            rule_applied=None,
            bindings={"X": "a"},
            success=True,
            explanation="matched fact",
        )
        d = step.to_dict()
        assert d["step_id"] == "s1"
        assert d["rule_applied"] is None
        assert d["success"] is True

    def test_proof_step_to_dict_with_rule(self):
        rule = Rule(rule_id="r1", head=Predicate(name="h", arguments=()), body=[])
        step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="h", arguments=()),
            rule_applied=rule,
            bindings={},
        )
        d = step.to_dict()
        assert d["rule_applied"] == "r1"

    def test_proof_tree_to_dict(self):
        root_step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="q", arguments=("a",)),
            rule_applied=None,
            bindings={},
        )
        tree = ProofTree(
            root=root_step,
            query=Predicate(name="q", arguments=("a",)),
            status=ProofStatus.SUCCESS,
            bindings=[{"X": "a"}],
            depth=1,
            node_count=1,
            search_time_ms=5.0,
        )
        d = tree.to_dict()
        assert d["status"] == "SUCCESS"
        assert d["depth"] == 1

    def test_generate_explanation_failure(self):
        root_step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="q", arguments=("a",)),
            rule_applied=None,
            bindings={},
            success=False,
        )
        tree = ProofTree(
            root=root_step,
            query=Predicate(name="q", arguments=("a",)),
            status=ProofStatus.FAILURE,
            bindings=[],
        )
        explanation = tree.generate_explanation()
        assert "FAILURE" in explanation

    def test_generate_explanation_success_verbosity_1(self):
        root_step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="q", arguments=("a",)),
            rule_applied=None,
            bindings={},
            success=True,
        )
        tree = ProofTree(
            root=root_step,
            query=Predicate(name="q", arguments=("a",)),
            status=ProofStatus.SUCCESS,
            bindings=[],
        )
        explanation = tree.generate_explanation(verbosity=1)
        assert "Proved" in explanation

    def test_generate_explanation_success_verbosity_2(self):
        root_step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="q", arguments=("a",)),
            rule_applied=None,
            bindings={"X": "a"},
            success=True,
        )
        tree = ProofTree(
            root=root_step,
            query=Predicate(name="q", arguments=("a",)),
            status=ProofStatus.SUCCESS,
            bindings=[{"X": "a"}],
        )
        explanation = tree.generate_explanation(verbosity=2)
        assert "Variable bindings" in explanation

    def test_generate_explanation_success_verbosity_3(self):
        child_step = ProofStep(
            step_id="s2",
            predicate=Predicate(name="p", arguments=("a",)),
            rule_applied=None,
            bindings={},
            success=True,
        )
        rule = Rule(rule_id="r1", head=Predicate(name="q", arguments=("a",)), body=[])
        root_step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="q", arguments=("a",)),
            rule_applied=rule,
            bindings={"X": "a"},
            children=[child_step],
            success=True,
        )
        tree = ProofTree(
            root=root_step,
            query=Predicate(name="q", arguments=("a",)),
            status=ProofStatus.SUCCESS,
            bindings=[{"X": "a"}],
        )
        explanation = tree.generate_explanation(verbosity=3)
        assert "Proof steps" in explanation
        assert "r1" in explanation


# ---------------------------------------------------------------------------
# LogicEngine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogicEngine:
    def _make_engine(self, **kwargs) -> LogicEngine:
        config = LogicEngineConfig(**kwargs)
        return LogicEngine(config)

    def test_add_and_clear_rules(self):
        engine = self._make_engine()
        rule = Rule(
            rule_id="r1",
            head=Predicate(name="mortal", arguments=("?X",)),
            body=[Predicate(name="human", arguments=("?X",))],
        )
        engine.add_rule(rule)
        assert len(engine._rules) == 1
        engine.clear_rules()
        assert len(engine._rules) == 0

    def test_add_rules_bulk(self):
        engine = self._make_engine()
        rules = [
            Rule(rule_id="r1", head=Predicate(name="a", arguments=()), body=[]),
            Rule(rule_id="r2", head=Predicate(name="b", arguments=()), body=[]),
        ]
        engine.add_rules(rules)
        assert len(engine._rules) == 2

    def test_unify_same_name_and_args(self):
        engine = self._make_engine()
        p1 = Predicate(name="f", arguments=("?X",))
        p2 = Predicate(name="f", arguments=("alice",))
        result = engine._unify(p1, p2, {})
        assert result is not None
        assert result["X"] == "alice"

    def test_unify_different_names(self):
        engine = self._make_engine()
        p1 = Predicate(name="f", arguments=("a",))
        p2 = Predicate(name="g", arguments=("a",))
        assert engine._unify(p1, p2, {}) is None

    def test_unify_different_arity(self):
        engine = self._make_engine()
        p1 = Predicate(name="f", arguments=("a",))
        p2 = Predicate(name="f", arguments=("a", "b"))
        assert engine._unify(p1, p2, {}) is None

    def test_unify_negation_mismatch(self):
        engine = self._make_engine()
        p1 = Predicate(name="f", arguments=("a",), negated=False)
        p2 = Predicate(name="f", arguments=("a",), negated=True)
        assert engine._unify(p1, p2, {}) is None

    def test_unify_conflicting_constants(self):
        engine = self._make_engine()
        p1 = Predicate(name="f", arguments=("a",))
        p2 = Predicate(name="f", arguments=("b",))
        assert engine._unify(p1, p2, {}) is None

    def test_rename_variables(self):
        engine = self._make_engine()
        p = Predicate(name="f", arguments=("?X", "const", "?Y"))
        renamed = engine._rename_variables(p, 3)
        assert renamed.arguments == ("?X_3", "const", "?Y_3")

    def test_count_depth_single(self):
        engine = self._make_engine()
        step = ProofStep(
            step_id="s1",
            predicate=Predicate(name="f", arguments=()),
            rule_applied=None,
            bindings={},
        )
        assert engine._count_depth(step) == 1

    def test_count_depth_nested(self):
        engine = self._make_engine()
        child = ProofStep(
            step_id="s2",
            predicate=Predicate(name="g", arguments=()),
            rule_applied=None,
            bindings={},
        )
        parent = ProofStep(
            step_id="s1",
            predicate=Predicate(name="f", arguments=()),
            rule_applied=None,
            bindings={},
            children=[child],
        )
        assert engine._count_depth(parent) == 2

    @pytest.mark.asyncio
    async def test_query_fact_matching(self):
        engine = self._make_engine()
        goal = Predicate(name="parent", arguments=("alice", "bob"))
        fact = Fact(name="parent", arguments=("alice", "bob"))
        state = NeuroSymbolicState(state_id="test", facts=frozenset([fact]))
        tree = await engine.query(goal, state)
        assert tree.status == ProofStatus.SUCCESS
        assert len(tree.bindings) >= 1

    @pytest.mark.asyncio
    async def test_query_no_match(self):
        engine = self._make_engine()
        goal = Predicate(name="parent", arguments=("alice", "charlie"))
        fact = Fact(name="parent", arguments=("alice", "bob"))
        state = NeuroSymbolicState(state_id="test", facts=frozenset([fact]))
        tree = await engine.query(goal, state)
        assert tree.status == ProofStatus.FAILURE

    @pytest.mark.asyncio
    async def test_query_with_rule(self):
        engine = self._make_engine()
        engine.add_rule(
            Rule(
                rule_id="mortal_rule",
                head=Predicate(name="mortal", arguments=("?X",)),
                body=[Predicate(name="human", arguments=("?X",))],
            )
        )
        goal = Predicate(name="mortal", arguments=("socrates",))
        fact = Fact(name="human", arguments=("socrates",))
        state = NeuroSymbolicState(state_id="test", facts=frozenset([fact]))
        tree = await engine.query(goal, state)
        assert tree.status == ProofStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_query_memoization(self):
        engine = self._make_engine(enable_memoization=True)
        goal = Predicate(name="f", arguments=("a",))
        fact = Fact(name="f", arguments=("a",))
        state = NeuroSymbolicState(state_id="test", facts=frozenset([fact]))

        tree1 = await engine.query(goal, state)
        tree2 = await engine.query(goal, state)
        # Second call should return cached result
        assert tree1 is tree2

    def test_get_cache_key_deterministic(self):
        engine = self._make_engine()
        goal = Predicate(name="f", arguments=("a",))
        facts = frozenset([Fact(name="f", arguments=("a",))])
        key1 = engine._get_cache_key(goal, facts)
        key2 = engine._get_cache_key(goal, facts)
        assert key1 == key2


# ---------------------------------------------------------------------------
# SymbolicReasoner
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSymbolicReasoner:
    def _make_reasoner(self) -> SymbolicReasoner:
        config = NeuroSymbolicConfig()
        return SymbolicReasoner(config)

    def test_add_rule(self):
        reasoner = self._make_reasoner()
        reasoner.add_rule(
            rule_id="r1",
            head=("mortal", ("?X",)),
            body=[("human", ("?X",))],
        )
        assert len(reasoner.logic_engine._rules) == 1

    @pytest.mark.asyncio
    async def test_prove_success(self):
        reasoner = self._make_reasoner()
        fact = Fact(name="human", arguments=("socrates",))
        state = NeuroSymbolicState(state_id="test", facts=frozenset([fact]))
        reasoner.add_rule("mortal_rule", ("mortal", ("?X",)), [("human", ("?X",))])

        proof = await reasoner.prove(("mortal", ("socrates",)), state)
        assert proof.success is True
        assert proof.confidence > 0

    @pytest.mark.asyncio
    async def test_prove_failure(self):
        reasoner = self._make_reasoner()
        state = NeuroSymbolicState(state_id="test", facts=frozenset())
        proof = await reasoner.prove(("unknown", ("x",)), state)
        assert proof.success is False

    def test_parse_query(self):
        reasoner = self._make_reasoner()
        pred = reasoner._parse_query("parent(?X, bob)")
        assert pred.name == "parent"
        assert pred.arguments == ("?X", "bob")

    def test_parse_query_invalid(self):
        reasoner = self._make_reasoner()
        with pytest.raises(ValueError, match="Invalid query format"):
            reasoner._parse_query("not a valid query")

    def test_parse_query_numeric(self):
        reasoner = self._make_reasoner()
        pred = reasoner._parse_query("age(alice, 30)")
        assert pred.arguments == ("alice", 30)

    @pytest.mark.asyncio
    async def test_ask(self):
        reasoner = self._make_reasoner()
        fact = Fact(name="parent", arguments=("alice", "bob"))
        state = NeuroSymbolicState(state_id="test", facts=frozenset([fact]))
        results = await reasoner.ask("parent(alice, bob)", state)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# SymbolicReasoningAgent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSymbolicReasoningAgent:
    def _make_agent(self, neural_fallback=None) -> SymbolicReasoningAgent:
        config = NeuroSymbolicConfig()
        return SymbolicReasoningAgent(config=config, neural_fallback=neural_fallback)

    def test_init(self):
        agent = self._make_agent()
        assert agent._query_count == 0
        assert agent._success_count == 0
        assert agent._fallback_count == 0

    def test_get_statistics_initial(self):
        agent = self._make_agent()
        stats = agent.get_statistics()
        assert stats["total_queries"] == 0
        assert stats["success_rate"] == 0.0

    def test_query_to_goal_isa(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("is alice a human?")
        assert goal == ("isa", ("alice", "human"))

    def test_query_to_goal_has(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("does alice have wings?")
        assert goal == ("has", ("alice", "wings"))

    def test_query_to_goal_can(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("can birds do fly?")
        assert goal == ("can", ("birds", "fly"))

    def test_query_to_goal_prolog_style(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("parent(alice, bob)")
        assert goal == ("parent", ("alice", "bob"))

    def test_query_to_goal_unknown(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("what is the meaning of life")
        assert goal is None

    def test_extract_facts_from_context(self):
        agent = self._make_agent()
        state = NeuroSymbolicState(state_id="test", facts=frozenset())
        context = "Alice is a human. Bob has wings."
        new_state = agent._extract_facts_from_context(state, context)
        fact_names = {f.name for f in new_state.facts}
        assert "isa" in fact_names
        assert "has" in fact_names

    def test_format_response_with_bindings(self):
        agent = self._make_agent()
        proof = Proof(
            success=True,
            bindings={"X": "alice"},
            explanation="Proved by fact",
        )
        resp = agent._format_response("query", proof)
        assert "X=alice" in resp

    def test_format_response_no_bindings(self):
        agent = self._make_agent()
        proof = Proof(success=True, explanation="Proved directly")
        resp = agent._format_response("query", proof)
        assert "Proved directly" in resp

    @pytest.mark.asyncio
    async def test_process_no_match_no_fallback(self):
        agent = self._make_agent()
        result = await agent.process("what is the meaning of life")
        assert "Could not determine" in result["response"]
        assert result["metadata"]["proof_found"] is False
        assert agent._query_count == 1

    @pytest.mark.asyncio
    async def test_process_with_neural_fallback(self):
        fallback = MagicMock(return_value="Neural answer")
        config = NeuroSymbolicConfig()
        config.agent.fallback_to_neural = True
        agent = SymbolicReasoningAgent(config=config, neural_fallback=fallback)

        result = await agent.process("what is the meaning of life")
        assert result["metadata"]["agent"] == "symbolic_with_neural_fallback"
        assert agent._fallback_count == 1

    def test_add_knowledge(self):
        agent = self._make_agent()
        agent.add_knowledge(
            facts=[],
            rules=[("r1", ("mortal", ("?X",)), [("human", ("?X",))])],
        )
        assert len(agent.reasoner.logic_engine._rules) == 1


# ---------------------------------------------------------------------------
# ProofStatus enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProofStatus:
    def test_all_values(self):
        assert ProofStatus.SUCCESS is not None
        assert ProofStatus.FAILURE is not None
        assert ProofStatus.TIMEOUT is not None
        assert ProofStatus.UNKNOWN is not None
        assert ProofStatus.PARTIAL is not None
