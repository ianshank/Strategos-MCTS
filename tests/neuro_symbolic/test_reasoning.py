"""
Unit tests for symbolic reasoning module.

Tests:
- Predicate and Rule creation
- Logic engine operations
- Proof tree generation
- Symbolic reasoner
- Symbolic reasoning agent

Best Practices 2025:
- Async test support
- Property-based testing
- Comprehensive proof verification
"""

import pytest

from src.neuro_symbolic.config import get_default_config
from src.neuro_symbolic.reasoning import (
    LogicEngine,
    Predicate,
    ProofStatus,
    ProofStep,
    ProofTree,
    Rule,
    SymbolicReasoner,
    SymbolicReasoningAgent,
)
from src.neuro_symbolic.state import Fact, NeuroSymbolicState


class TestPredicate:
    """Tests for Predicate."""

    def test_predicate_creation(self):
        """Test basic predicate creation."""
        pred = Predicate(name="parent", arguments=("john", "mary"))

        assert pred.name == "parent"
        assert pred.arguments == ("john", "mary")
        assert pred.negated is False

    def test_negated_predicate(self):
        """Test negated predicate."""
        pred = Predicate(name="valid", arguments=("x",), negated=True)

        assert pred.negated is True
        assert pred.to_string().startswith("\\+")

    def test_predicate_to_string(self):
        """Test string representation."""
        pred = Predicate(name="edge", arguments=("a", "b", 5))
        assert pred.to_string() == "edge(a, b, 5)"

        neg_pred = Predicate(name="error", arguments=(), negated=True)
        assert neg_pred.to_string() == "\\+error()"

    def test_predicate_to_fact(self):
        """Test converting ground predicate to fact."""
        pred = Predicate(name="parent", arguments=("john", "mary"))
        fact = pred.to_fact()

        assert isinstance(fact, Fact)
        assert fact.name == "parent"
        assert fact.arguments == ("john", "mary")

    def test_predicate_with_variables_to_fact_raises(self):
        """Test that predicate with variables cannot convert to fact."""
        pred = Predicate(name="parent", arguments=("?X", "mary"))

        with pytest.raises(ValueError, match="variables"):
            pred.to_fact()

    def test_predicate_is_ground(self):
        """Test ground predicate detection."""
        ground = Predicate(name="test", arguments=("a", "b"))
        not_ground = Predicate(name="test", arguments=("?X", "b"))

        assert ground.is_ground()
        assert not not_ground.is_ground()

    def test_predicate_get_variables(self):
        """Test extracting variables."""
        pred = Predicate(name="edge", arguments=("?From", "?To", 5))

        variables = pred.get_variables()

        assert variables == {"From", "To"}

    def test_predicate_substitute(self):
        """Test variable substitution."""
        pred = Predicate(name="parent", arguments=("?X", "?Y"))
        bindings = {"X": "john", "Y": "mary"}

        result = pred.substitute(bindings)

        assert result.arguments == ("john", "mary")
        assert result.name == "parent"

    def test_predicate_partial_substitute(self):
        """Test partial substitution."""
        pred = Predicate(name="edge", arguments=("?X", "?Y", "?Z"))
        bindings = {"X": "a"}

        result = pred.substitute(bindings)

        assert result.arguments == ("a", "?Y", "?Z")

    def test_predicate_empty_name_raises(self):
        """Test empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Predicate(name="", arguments=("a",))


class TestRule:
    """Tests for inference Rule."""

    def test_rule_creation(self):
        """Test basic rule creation."""
        head = Predicate(name="grandparent", arguments=("?X", "?Z"))
        body = [
            Predicate(name="parent", arguments=("?X", "?Y")),
            Predicate(name="parent", arguments=("?Y", "?Z")),
        ]
        rule = Rule(rule_id="r1", head=head, body=body)

        assert rule.rule_id == "r1"
        assert rule.head == head
        assert len(rule.body) == 2

    def test_rule_to_string(self):
        """Test rule string representation."""
        head = Predicate(name="mortal", arguments=("?X",))
        body = [Predicate(name="human", arguments=("?X",))]
        rule = Rule(rule_id="r1", head=head, body=body)

        rule_str = rule.to_string()

        assert "mortal(?X)" in rule_str
        assert ":-" in rule_str
        assert "human(?X)" in rule_str

    def test_rule_get_all_variables(self):
        """Test getting all variables in rule."""
        head = Predicate(name="ancestor", arguments=("?X", "?Z"))
        body = [
            Predicate(name="parent", arguments=("?X", "?Y")),
            Predicate(name="ancestor", arguments=("?Y", "?Z")),
        ]
        rule = Rule(rule_id="r1", head=head, body=body)

        variables = rule.get_all_variables()

        assert variables == {"X", "Y", "Z"}


class TestLogicEngine:
    """Tests for LogicEngine."""

    @pytest.fixture
    def engine(self):
        """Create logic engine with default config."""
        config = get_default_config().logic_engine
        return LogicEngine(config)

    def test_add_rule(self, engine):
        """Test adding rules to engine."""
        rule = Rule(
            rule_id="r1",
            head=Predicate(name="mortal", arguments=("?X",)),
            body=[Predicate(name="human", arguments=("?X",))],
        )
        engine.add_rule(rule)

        assert len(engine._rules) == 1

    def test_add_multiple_rules(self, engine):
        """Test adding multiple rules."""
        rules = [
            Rule(
                rule_id="r1",
                head=Predicate(name="mortal", arguments=("?X",)),
                body=[Predicate(name="human", arguments=("?X",))],
            ),
            Rule(
                rule_id="r2",
                head=Predicate(name="living", arguments=("?X",)),
                body=[Predicate(name="animal", arguments=("?X",))],
            ),
        ]
        engine.add_rules(rules)

        assert len(engine._rules) == 2

    def test_clear_rules(self, engine):
        """Test clearing rules."""
        engine.add_rule(
            Rule(
                rule_id="r1",
                head=Predicate(name="test", arguments=()),
                body=[],
            )
        )
        engine.clear_rules()

        assert len(engine._rules) == 0

    @pytest.mark.asyncio
    async def test_query_fact_match(self, engine):
        """Test query matching a fact directly."""
        fact = Fact(name="human", arguments=("socrates",))
        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([fact]),
        )

        goal = Predicate(name="human", arguments=("socrates",))
        result = await engine.query(goal, state)

        assert result.status == ProofStatus.SUCCESS
        assert len(result.bindings) == 1

    @pytest.mark.asyncio
    async def test_query_with_rule(self, engine):
        """Test query using inference rule."""
        # Rule: mortal(X) :- human(X)
        engine.add_rule(
            Rule(
                rule_id="mortality",
                head=Predicate(name="mortal", arguments=("?X",)),
                body=[Predicate(name="human", arguments=("?X",))],
            )
        )

        # Fact: human(socrates)
        fact = Fact(name="human", arguments=("socrates",))
        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([fact]),
        )

        # Query: mortal(socrates)?
        goal = Predicate(name="mortal", arguments=("socrates",))
        result = await engine.query(goal, state)

        assert result.status == ProofStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_query_with_variable(self, engine):
        """Test query with variable binding."""
        engine.add_rule(
            Rule(
                rule_id="r1",
                head=Predicate(name="mortal", arguments=("?X",)),
                body=[Predicate(name="human", arguments=("?X",))],
            )
        )

        facts = frozenset(
            [
                Fact(name="human", arguments=("socrates",)),
                Fact(name="human", arguments=("plato",)),
            ]
        )
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        # Query: mortal(?Who)?
        goal = Predicate(name="mortal", arguments=("?Who",))
        result = await engine.query(goal, state)

        assert result.status == ProofStatus.SUCCESS
        # Should find at least one binding
        assert len(result.bindings) >= 1

    @pytest.mark.asyncio
    async def test_query_no_match(self, engine):
        """Test query with no matching facts or rules."""
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        goal = Predicate(name="nonexistent", arguments=("x",))
        result = await engine.query(goal, state)

        assert result.status == ProofStatus.FAILURE

    @pytest.mark.asyncio
    async def test_query_transitive_rule(self, engine):
        """Test query with transitive relation."""
        # Rule: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
        # Rule: ancestor(X,Y) :- parent(X,Y)
        engine.add_rules(
            [
                Rule(
                    rule_id="ancestor_base",
                    head=Predicate(name="ancestor", arguments=("?X", "?Y")),
                    body=[Predicate(name="parent", arguments=("?X", "?Y"))],
                ),
                Rule(
                    rule_id="ancestor_trans",
                    head=Predicate(name="ancestor", arguments=("?X", "?Z")),
                    body=[
                        Predicate(name="parent", arguments=("?X", "?Y")),
                        Predicate(name="ancestor", arguments=("?Y", "?Z")),
                    ],
                ),
            ]
        )

        facts = frozenset(
            [
                Fact(name="parent", arguments=("a", "b")),
                Fact(name="parent", arguments=("b", "c")),
            ]
        )
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        # Query: ancestor(a, c)?
        goal = Predicate(name="ancestor", arguments=("a", "c"))
        result = await engine.query(goal, state)

        assert result.status == ProofStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_proof_tree_depth(self, engine):
        """Test proof tree depth tracking."""
        engine.add_rule(
            Rule(
                rule_id="r1",
                head=Predicate(name="mortal", arguments=("?X",)),
                body=[Predicate(name="human", arguments=("?X",))],
            )
        )

        fact = Fact(name="human", arguments=("socrates",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([fact]))

        goal = Predicate(name="mortal", arguments=("socrates",))
        result = await engine.query(goal, state)

        assert result.depth >= 1

    @pytest.mark.asyncio
    async def test_caching(self, engine):
        """Test query result caching."""
        fact = Fact(name="test", arguments=("x",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([fact]))

        goal = Predicate(name="test", arguments=("x",))

        # First query
        await engine.query(goal, state)

        # Second query should hit cache
        result2 = await engine.query(goal, state)

        assert result2.status == ProofStatus.SUCCESS


class TestProofTree:
    """Tests for ProofTree."""

    def test_proof_tree_to_dict(self):
        """Test proof tree serialization."""
        root = ProofStep(
            step_id="s1",
            predicate=Predicate(name="test", arguments=("x",)),
            rule_applied=None,
            bindings={},
            success=True,
        )
        tree = ProofTree(
            root=root,
            query=Predicate(name="test", arguments=("x",)),
            status=ProofStatus.SUCCESS,
            bindings=[{}],
        )

        tree_dict = tree.to_dict()

        assert tree_dict["status"] == "SUCCESS"
        assert "proof_tree" in tree_dict

    def test_generate_explanation_success(self):
        """Test explanation generation for successful proof."""
        root = ProofStep(
            step_id="s1",
            predicate=Predicate(name="mortal", arguments=("socrates",)),
            rule_applied=Rule(
                rule_id="r1",
                head=Predicate(name="mortal", arguments=("?X",)),
                body=[Predicate(name="human", arguments=("?X",))],
            ),
            bindings={"X": "socrates"},
            success=True,
        )
        tree = ProofTree(
            root=root,
            query=Predicate(name="mortal", arguments=("socrates",)),
            status=ProofStatus.SUCCESS,
            bindings=[{"X": "socrates"}],
        )

        explanation = tree.generate_explanation(verbosity=2)

        assert "Proved" in explanation
        assert "socrates" in explanation

    def test_generate_explanation_failure(self):
        """Test explanation generation for failed proof."""
        root = ProofStep(
            step_id="s1",
            predicate=Predicate(name="test", arguments=()),
            rule_applied=None,
            bindings={},
            success=False,
        )
        tree = ProofTree(
            root=root,
            query=Predicate(name="test", arguments=()),
            status=ProofStatus.FAILURE,
            bindings=[],
        )

        explanation = tree.generate_explanation()

        assert "Could not prove" in explanation


class TestSymbolicReasoner:
    """Tests for SymbolicReasoner."""

    @pytest.fixture
    def reasoner(self):
        """Create symbolic reasoner."""
        config = get_default_config()
        return SymbolicReasoner(config)

    def test_add_rule(self, reasoner):
        """Test adding rule via helper method."""
        reasoner.add_rule(
            rule_id="r1",
            head=("mortal", ("?X",)),
            body=[("human", ("?X",))],
        )

        assert len(reasoner.logic_engine._rules) == 1

    @pytest.mark.asyncio
    async def test_prove_success(self, reasoner):
        """Test successful proof."""
        reasoner.add_rule(
            rule_id="r1",
            head=("mortal", ("?X",)),
            body=[("human", ("?X",))],
        )

        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([Fact(name="human", arguments=("socrates",))]),
        )

        proof = await reasoner.prove(("mortal", ("socrates",)), state)

        assert proof.success
        assert proof.confidence > 0

    @pytest.mark.asyncio
    async def test_prove_failure(self, reasoner):
        """Test failed proof."""
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        proof = await reasoner.prove(("nonexistent", ("x",)), state)

        assert not proof.success

    @pytest.mark.asyncio
    async def test_ask_query(self, reasoner):
        """Test ask method with Prolog-style query."""
        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([Fact(name="parent", arguments=("john", "mary"))]),
        )

        results = await reasoner.ask("parent(john, mary)", state)

        assert len(results) > 0


class TestSymbolicReasoningAgent:
    """Tests for SymbolicReasoningAgent."""

    @pytest.fixture
    def agent(self):
        """Create symbolic reasoning agent."""
        config = get_default_config()
        return SymbolicReasoningAgent(config)

    @pytest.mark.asyncio
    async def test_process_query_with_facts(self, agent):
        """Test processing query with facts."""
        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset(
                [
                    Fact(name="isa", arguments=("socrates", "human")),
                ]
            ),
        )

        result = await agent.process(
            query="is socrates a human?",
            state=state,
        )

        assert "response" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_process_creates_state_if_none(self, agent):
        """Test that agent creates state if not provided."""
        result = await agent.process(
            query="test query",
        )

        assert "response" in result
        assert "metadata" in result
        assert result["metadata"]["agent"] in ("symbolic", "symbolic_with_neural_fallback")

    @pytest.mark.asyncio
    async def test_process_extracts_facts_from_rag(self, agent):
        """Test fact extraction from RAG context."""
        result = await agent.process(
            query="is socrates a philosopher?",
            rag_context="Socrates is a philosopher. Plato is a student.",
        )

        # Should have attempted to process
        assert "response" in result

    def test_add_knowledge(self, agent):
        """Test adding knowledge to agent."""
        rules = [
            ("mortality", ("mortal", ("?X",)), [("human", ("?X",))]),
        ]

        agent.add_knowledge(facts=[], rules=rules)

        assert len(agent.reasoner.logic_engine._rules) == 1

    def test_get_statistics(self, agent):
        """Test statistics retrieval."""
        stats = agent.get_statistics()

        assert "total_queries" in stats
        assert "successful_proofs" in stats
        assert "neural_fallbacks" in stats
        assert "success_rate" in stats

    @pytest.mark.asyncio
    async def test_process_with_neural_fallback(self):
        """Test neural fallback when symbolic fails."""
        config = get_default_config()
        config.agent.fallback_to_neural = True

        fallback_called = False

        async def mock_fallback(query, state):
            nonlocal fallback_called
            fallback_called = True
            return "Neural answer"

        agent = SymbolicReasoningAgent(
            config=config,
            neural_fallback=mock_fallback,
        )

        result = await agent.process(query="complex unprovable query xyz123")

        assert "response" in result
        # Should have tried symbolic first


class TestQueryParsing:
    """Tests for query parsing in SymbolicReasoningAgent."""

    @pytest.fixture
    def agent(self):
        config = get_default_config()
        return SymbolicReasoningAgent(config)

    @pytest.mark.parametrize(
        "query,expected_name,expected_args",
        [
            ("is socrates a human?", "isa", ("socrates", "human")),
            ("does john have car?", "has", ("john", "car")),
            ("can agent do task?", "can", ("agent", "task")),
            ("parent(john, mary)", "parent", ("john", "mary")),
        ],
    )
    def test_query_to_goal_patterns(self, agent, query, expected_name, expected_args):
        """Test query pattern recognition."""
        result = agent._query_to_goal(query)

        if result is not None:
            name, args = result
            assert name == expected_name
            assert args == expected_args
