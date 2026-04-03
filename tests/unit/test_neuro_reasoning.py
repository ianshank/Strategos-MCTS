"""Unit tests for neuro-symbolic reasoning module."""

import asyncio
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
# Predicate tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPredicate:
    """Test Predicate data class."""

    def test_create_predicate(self):
        p = Predicate(name="parent", arguments=("john", "mary"))
        assert p.name == "parent"
        assert p.arguments == ("john", "mary")
        assert p.negated is False

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Predicate(name="", arguments=("a",))

    def test_to_string(self):
        p = Predicate(name="likes", arguments=("alice", "bob"))
        assert p.to_string() == "likes(alice, bob)"

    def test_to_string_negated(self):
        p = Predicate(name="likes", arguments=("alice", "bob"), negated=True)
        assert p.to_string() == "\\+likes(alice, bob)"

    def test_is_ground_true(self):
        p = Predicate(name="parent", arguments=("john", "mary"))
        assert p.is_ground() is True

    def test_is_ground_false(self):
        p = Predicate(name="parent", arguments=("?X", "mary"))
        assert p.is_ground() is False

    def test_get_variables(self):
        p = Predicate(name="parent", arguments=("?X", "mary", "?Y"))
        assert p.get_variables() == {"X", "Y"}

    def test_get_variables_none(self):
        p = Predicate(name="parent", arguments=("john", "mary"))
        assert p.get_variables() == set()

    def test_substitute(self):
        p = Predicate(name="parent", arguments=("?X", "mary"))
        result = p.substitute({"X": "john"})
        assert result.arguments == ("john", "mary")
        assert result.name == "parent"

    def test_substitute_partial(self):
        p = Predicate(name="f", arguments=("?X", "?Y"))
        result = p.substitute({"X": "a"})
        assert result.arguments == ("a", "?Y")

    def test_to_fact(self):
        p = Predicate(name="parent", arguments=("john", "mary"))
        fact = p.to_fact()
        assert fact.name == "parent"
        assert fact.arguments == ("john", "mary")
        assert fact.fact_type == SymbolicFactType.PREDICATE

    def test_to_fact_with_variable_raises(self):
        p = Predicate(name="parent", arguments=("?X", "mary"))
        with pytest.raises(ValueError, match="variables"):
            p.to_fact()


# ---------------------------------------------------------------------------
# Rule tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRule:
    """Test Rule data class."""

    def test_create_rule(self):
        head = Predicate(name="grandparent", arguments=("?X", "?Z"))
        body = [
            Predicate(name="parent", arguments=("?X", "?Y")),
            Predicate(name="parent", arguments=("?Y", "?Z")),
        ]
        rule = Rule(rule_id="r1", head=head, body=body)
        assert rule.rule_id == "r1"
        assert rule.confidence == 1.0

    def test_to_string(self):
        head = Predicate(name="mortal", arguments=("?X",))
        body = [Predicate(name="human", arguments=("?X",))]
        rule = Rule(rule_id="r1", head=head, body=body)
        assert rule.to_string() == "mortal(?X) :- human(?X)."

    def test_get_all_variables(self):
        head = Predicate(name="gp", arguments=("?X", "?Z"))
        body = [
            Predicate(name="p", arguments=("?X", "?Y")),
            Predicate(name="p", arguments=("?Y", "?Z")),
        ]
        rule = Rule(rule_id="r1", head=head, body=body)
        assert rule.get_all_variables() == {"X", "Y", "Z"}


# ---------------------------------------------------------------------------
# ProofTree tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProofTree:
    """Test ProofTree and ProofStep."""

    def _make_success_tree(self) -> ProofTree:
        root = ProofStep(
            step_id="s1",
            predicate=Predicate(name="mortal", arguments=("socrates",)),
            rule_applied=Rule(
                rule_id="r1",
                head=Predicate(name="mortal", arguments=("?X",)),
                body=[Predicate(name="human", arguments=("?X",))],
            ),
            bindings={"X": "socrates"},
            children=[
                ProofStep(
                    step_id="s2",
                    predicate=Predicate(name="human", arguments=("socrates",)),
                    rule_applied=None,
                    bindings={"X": "socrates"},
                    success=True,
                    explanation="matched fact",
                )
            ],
            success=True,
        )
        return ProofTree(
            root=root,
            query=Predicate(name="mortal", arguments=("socrates",)),
            status=ProofStatus.SUCCESS,
            bindings=[{"X": "socrates"}],
            depth=2,
            node_count=2,
        )

    def test_to_dict(self):
        tree = self._make_success_tree()
        d = tree.to_dict()
        assert d["status"] == "SUCCESS"
        assert d["query"] == "mortal(socrates)"
        assert d["bindings"] == [{"X": "socrates"}]
        assert "proof_tree" in d

    def test_generate_explanation_success_minimal(self):
        tree = self._make_success_tree()
        exp = tree.generate_explanation(verbosity=1)
        assert "Proved: mortal(socrates)" in exp

    def test_generate_explanation_success_standard(self):
        tree = self._make_success_tree()
        exp = tree.generate_explanation(verbosity=2)
        assert "Variable bindings:" in exp
        assert "X=socrates" in exp

    def test_generate_explanation_success_detailed(self):
        tree = self._make_success_tree()
        exp = tree.generate_explanation(verbosity=3)
        assert "Proof steps:" in exp

    def test_generate_explanation_failure(self):
        root = ProofStep(
            step_id="s1",
            predicate=Predicate(name="fly", arguments=("bob",)),
            rule_applied=None,
            bindings={},
            success=False,
        )
        tree = ProofTree(
            root=root,
            query=Predicate(name="fly", arguments=("bob",)),
            status=ProofStatus.FAILURE,
            bindings=[],
        )
        exp = tree.generate_explanation()
        assert "Could not prove" in exp
        assert "FAILURE" in exp


# ---------------------------------------------------------------------------
# LogicEngine tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLogicEngine:
    """Test LogicEngine backward chaining."""

    def _make_engine(self, **kwargs) -> LogicEngine:
        config = LogicEngineConfig(
            solver_timeout_ms=kwargs.get("timeout_ms", 5000),
            max_proof_depth=kwargs.get("max_depth", 50),
            enable_memoization=kwargs.get("memoization", False),
        )
        return LogicEngine(config)

    def _make_state(self, facts: list[Fact]) -> NeuroSymbolicState:
        return NeuroSymbolicState(state_id="test", facts=frozenset(facts))

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

    def test_add_rules_batch(self):
        engine = self._make_engine()
        rules = [
            Rule(rule_id="r1", head=Predicate(name="a", arguments=()), body=[]),
            Rule(rule_id="r2", head=Predicate(name="b", arguments=()), body=[]),
        ]
        engine.add_rules(rules)
        assert len(engine._rules) == 2

    def test_query_fact_match(self):
        """Engine finds a direct fact match."""
        engine = self._make_engine()
        state = self._make_state([
            Fact(name="human", arguments=("socrates",)),
        ])
        goal = Predicate(name="human", arguments=("socrates",))
        result = asyncio.run(engine.query(goal, state))
        assert result.status == ProofStatus.SUCCESS
        assert len(result.bindings) > 0

    def test_query_no_match(self):
        """Engine returns FAILURE when no fact or rule matches."""
        engine = self._make_engine()
        state = self._make_state([
            Fact(name="human", arguments=("socrates",)),
        ])
        goal = Predicate(name="fly", arguments=("socrates",))
        result = asyncio.run(engine.query(goal, state))
        assert result.status == ProofStatus.FAILURE

    def test_query_with_rule(self):
        """Engine proves goal through a rule."""
        engine = self._make_engine()
        engine.add_rule(Rule(
            rule_id="mortal_rule",
            head=Predicate(name="mortal", arguments=("?X",)),
            body=[Predicate(name="human", arguments=("?X",))],
        ))
        state = self._make_state([
            Fact(name="human", arguments=("socrates",)),
        ])
        goal = Predicate(name="mortal", arguments=("socrates",))
        result = asyncio.run(engine.query(goal, state))
        assert result.status == ProofStatus.SUCCESS

    def test_query_with_variable(self):
        """Engine resolves variables via unification."""
        engine = self._make_engine()
        engine.add_rule(Rule(
            rule_id="mortal_rule",
            head=Predicate(name="mortal", arguments=("?X",)),
            body=[Predicate(name="human", arguments=("?X",))],
        ))
        state = self._make_state([
            Fact(name="human", arguments=("socrates",)),
        ])
        goal = Predicate(name="mortal", arguments=("?X",))
        result = asyncio.run(engine.query(goal, state))
        assert result.status == ProofStatus.SUCCESS
        # Should bind X to socrates (possibly renamed)
        assert len(result.bindings) > 0

    def test_query_chained_rules(self):
        """Engine handles multi-step rule chains."""
        engine = self._make_engine()
        engine.add_rule(Rule(
            rule_id="grandparent",
            head=Predicate(name="grandparent", arguments=("?X", "?Z")),
            body=[
                Predicate(name="parent", arguments=("?X", "?Y")),
                Predicate(name="parent", arguments=("?Y", "?Z")),
            ],
        ))
        state = self._make_state([
            Fact(name="parent", arguments=("alice", "bob")),
            Fact(name="parent", arguments=("bob", "charlie")),
        ])
        goal = Predicate(name="grandparent", arguments=("alice", "charlie"))
        result = asyncio.run(engine.query(goal, state))
        assert result.status == ProofStatus.SUCCESS

    def test_depth_limit(self):
        """Engine respects max proof depth."""
        engine = self._make_engine(max_depth=1)
        # Create a rule that requires depth > 1
        engine.add_rule(Rule(
            rule_id="r1",
            head=Predicate(name="a", arguments=("?X",)),
            body=[Predicate(name="b", arguments=("?X",))],
        ))
        engine.add_rule(Rule(
            rule_id="r2",
            head=Predicate(name="b", arguments=("?X",)),
            body=[Predicate(name="c", arguments=("?X",))],
        ))
        state = self._make_state([Fact(name="c", arguments=("val",))])
        goal = Predicate(name="a", arguments=("val",))
        result = asyncio.run(engine.query(goal, state))
        # With depth limit of 1, should fail to prove a multi-step chain
        assert result.status == ProofStatus.FAILURE

    def test_memoization(self):
        """Engine caches results when memoization is enabled."""
        engine = self._make_engine(memoization=True)
        state = self._make_state([Fact(name="human", arguments=("socrates",))])
        goal = Predicate(name="human", arguments=("socrates",))

        result1 = asyncio.run(engine.query(goal, state))
        assert result1.status == ProofStatus.SUCCESS
        assert len(engine._cache) == 1

        # Second query should hit cache
        result2 = asyncio.run(engine.query(goal, state))
        assert result2.status == ProofStatus.SUCCESS

    def test_unify_different_names(self):
        """Unification fails for different predicate names."""
        engine = self._make_engine()
        p1 = Predicate(name="a", arguments=("x",))
        p2 = Predicate(name="b", arguments=("x",))
        assert engine._unify(p1, p2, {}) is None

    def test_unify_different_arity(self):
        """Unification fails for different argument counts."""
        engine = self._make_engine()
        p1 = Predicate(name="a", arguments=("x",))
        p2 = Predicate(name="a", arguments=("x", "y"))
        assert engine._unify(p1, p2, {}) is None

    def test_unify_negation_mismatch(self):
        """Unification fails when negation differs."""
        engine = self._make_engine()
        p1 = Predicate(name="a", arguments=("x",), negated=False)
        p2 = Predicate(name="a", arguments=("x",), negated=True)
        assert engine._unify(p1, p2, {}) is None

    def test_unify_success(self):
        """Unification succeeds with variable binding."""
        engine = self._make_engine()
        p1 = Predicate(name="parent", arguments=("?X", "mary"))
        p2 = Predicate(name="parent", arguments=("john", "mary"))
        result = engine._unify(p1, p2, {})
        assert result is not None
        assert result["X"] == "john"

    def test_unify_conflict(self):
        """Unification fails on conflicting ground terms."""
        engine = self._make_engine()
        p1 = Predicate(name="a", arguments=("x",))
        p2 = Predicate(name="a", arguments=("y",))
        assert engine._unify(p1, p2, {}) is None


# ---------------------------------------------------------------------------
# SymbolicReasoner tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSymbolicReasoner:
    """Test high-level SymbolicReasoner."""

    def _make_reasoner(self) -> SymbolicReasoner:
        config = NeuroSymbolicConfig()
        config.logic_engine.enable_memoization = False
        return SymbolicReasoner(config)

    def _make_state(self, facts: list[Fact]) -> NeuroSymbolicState:
        return NeuroSymbolicState(state_id="test", facts=frozenset(facts))

    def test_add_rule(self):
        reasoner = self._make_reasoner()
        reasoner.add_rule("r1", ("mortal", ("?X",)), [("human", ("?X",))])
        assert len(reasoner.logic_engine._rules) == 1

    def test_prove_success(self):
        reasoner = self._make_reasoner()
        reasoner.add_rule("r1", ("mortal", ("?X",)), [("human", ("?X",))])
        state = self._make_state([Fact(name="human", arguments=("socrates",))])

        proof = asyncio.run(
            reasoner.prove(("mortal", ("socrates",)), state)
        )
        assert proof.success is True
        assert proof.confidence > 0.0
        assert proof.explanation != ""

    def test_prove_failure(self):
        reasoner = self._make_reasoner()
        state = self._make_state([])

        proof = asyncio.run(
            reasoner.prove(("fly", ("bob",)), state)
        )
        assert proof.success is False
        assert "Could not prove" in proof.explanation

    def test_ask_success(self):
        reasoner = self._make_reasoner()
        state = self._make_state([Fact(name="parent", arguments=("alice", "bob"))])

        results = asyncio.run(
            reasoner.ask("parent(alice, bob)", state)
        )
        assert len(results) > 0

    def test_ask_no_results(self):
        reasoner = self._make_reasoner()
        state = self._make_state([])

        results = asyncio.run(
            reasoner.ask("parent(alice, bob)", state)
        )
        assert results == []

    def test_parse_query_valid(self):
        reasoner = self._make_reasoner()
        pred = reasoner._parse_query("parent(?X, bob)")
        assert pred.name == "parent"
        assert pred.arguments == ("?X", "bob")

    def test_parse_query_invalid(self):
        reasoner = self._make_reasoner()
        with pytest.raises(ValueError, match="Invalid query"):
            reasoner._parse_query("not a valid query")

    def test_parse_query_numbers(self):
        reasoner = self._make_reasoner()
        pred = reasoner._parse_query("add(1, 2)")
        assert pred.arguments == (1, 2)

    def test_calculate_proof_confidence_fact_only(self):
        """Confidence is 1.0 for fact-only proof."""
        reasoner = self._make_reasoner()
        root = ProofStep(
            step_id="s1",
            predicate=Predicate(name="a", arguments=("x",)),
            rule_applied=None,
            bindings={},
            success=True,
        )
        tree = ProofTree(
            root=root,
            query=Predicate(name="a", arguments=("x",)),
            status=ProofStatus.SUCCESS,
            bindings=[{}],
        )
        assert reasoner._calculate_proof_confidence(tree) == 1.0

    def test_calculate_proof_confidence_with_rule(self):
        """Confidence reflects rule confidence."""
        reasoner = self._make_reasoner()
        child = ProofStep(
            step_id="s2",
            predicate=Predicate(name="b", arguments=("x",)),
            rule_applied=None,
            bindings={},
            success=True,
        )
        root = ProofStep(
            step_id="s1",
            predicate=Predicate(name="a", arguments=("x",)),
            rule_applied=Rule(
                rule_id="r1",
                head=Predicate(name="a", arguments=("?X",)),
                body=[Predicate(name="b", arguments=("?X",))],
                confidence=0.8,
            ),
            bindings={},
            children=[child],
            success=True,
        )
        tree = ProofTree(
            root=root,
            query=Predicate(name="a", arguments=("x",)),
            status=ProofStatus.SUCCESS,
            bindings=[{}],
        )
        conf = reasoner._calculate_proof_confidence(tree)
        assert conf == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# SymbolicReasoningAgent tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSymbolicReasoningAgent:
    """Test SymbolicReasoningAgent integration."""

    def _make_agent(self, **kwargs) -> SymbolicReasoningAgent:
        config = NeuroSymbolicConfig()
        config.logic_engine.enable_memoization = False
        return SymbolicReasoningAgent(config=config, **kwargs)

    def test_init(self):
        agent = self._make_agent()
        assert agent._query_count == 0
        assert agent._success_count == 0
        assert agent._fallback_count == 0

    def test_statistics_initial(self):
        agent = self._make_agent()
        stats = agent.get_statistics()
        assert stats["total_queries"] == 0
        assert stats["success_rate"] == 0.0

    def test_process_with_prolog_query(self):
        """Agent processes Prolog-style query successfully."""
        agent = self._make_agent()
        agent.reasoner.add_rule("r1", ("mortal", ("?X",)), [("human", ("?X",))])

        state = NeuroSymbolicState(
            state_id="test",
            facts=frozenset([Fact(name="human", arguments=("socrates",))]),
        )
        result = asyncio.run(
            agent.process("mortal(socrates)", state=state)
        )
        assert result["metadata"]["proof_found"] is True
        assert result["metadata"]["agent"] == "symbolic"
        assert result["metadata"]["confidence"] > 0.0

    def test_process_natural_language_isa(self):
        """Agent handles 'is X a Y?' pattern."""
        agent = self._make_agent()
        state = NeuroSymbolicState(
            state_id="test",
            facts=frozenset([Fact(name="isa", arguments=("socrates", "human"))]),
        )
        result = asyncio.run(
            agent.process("is socrates a human?", state=state)
        )
        assert result["metadata"]["proof_found"] is True

    def test_process_no_proof_no_fallback(self):
        """Agent returns failure when no proof and no fallback."""
        config = NeuroSymbolicConfig()
        config.agent.fallback_to_neural = False
        config.logic_engine.enable_memoization = False
        agent = SymbolicReasoningAgent(config=config)

        result = asyncio.run(
            agent.process("is bob a fish?")
        )
        assert result["metadata"]["proof_found"] is False
        assert "Could not determine" in result["response"]

    def test_process_with_neural_fallback(self):
        """Agent falls back to neural when symbolic fails."""
        fallback = MagicMock(return_value="Neural answer")
        agent = self._make_agent(neural_fallback=fallback)

        result = asyncio.run(
            agent.process("is bob a fish?")
        )
        assert result["metadata"]["agent"] == "symbolic_with_neural_fallback"
        assert result["response"] == "Neural answer"
        assert agent._fallback_count == 1

    def test_process_with_async_neural_fallback(self):
        """Agent handles async neural fallback."""
        async def async_fallback(query, state):
            return "Async neural answer"

        agent = self._make_agent(neural_fallback=async_fallback)

        result = asyncio.run(
            agent.process("is bob a fish?")
        )
        assert result["response"] == "Async neural answer"

    def test_process_with_rag_context(self):
        """Agent extracts facts from RAG context."""
        agent = self._make_agent()
        result = asyncio.run(
            agent.process("is socrates a human?", rag_context="socrates is a human")
        )
        assert result["metadata"]["proof_found"] is True

    def test_query_to_goal_isa(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("is socrates a human?")
        assert goal == ("isa", ("socrates", "human"))

    def test_query_to_goal_has(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("does alice have wings?")
        assert goal == ("has", ("alice", "wings"))

    def test_query_to_goal_can(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("can birds do fly?")
        assert goal == ("can", ("birds", "fly"))

    def test_query_to_goal_prolog(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("parent(alice, bob)")
        assert goal == ("parent", ("alice", "bob"))

    def test_query_to_goal_unknown(self):
        agent = self._make_agent()
        goal = agent._query_to_goal("hello world")
        assert goal is None

    def test_add_knowledge(self):
        agent = self._make_agent()
        agent.add_knowledge(
            facts=[],
            rules=[("r1", ("mortal", ("?X",)), [("human", ("?X",))])],
        )
        assert len(agent.reasoner.logic_engine._rules) == 1

    def test_extract_facts_from_context(self):
        agent = self._make_agent()
        state = NeuroSymbolicState(state_id="test", facts=frozenset())
        new_state = agent._extract_facts_from_context(state, "alice is a human. bob has wings")
        fact_strings = {f.to_string() for f in new_state.facts}
        assert "isa(alice, human)" in fact_strings
        assert "has(bob, wings)" in fact_strings

    def test_format_response_with_bindings(self):
        agent = self._make_agent()
        proof = Proof(success=True, bindings={"X": "socrates"}, explanation="Proved it")
        resp = agent._format_response("test", proof)
        assert "Yes" in resp
        assert "X=socrates" in resp

    def test_format_response_without_bindings(self):
        agent = self._make_agent()
        proof = Proof(success=True, bindings={}, explanation="Proved it")
        resp = agent._format_response("test", proof)
        assert "Yes" in resp
        assert "Proved it" in resp
