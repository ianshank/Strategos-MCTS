"""
Tests for neuro-symbolic state module.

Tests Fact, NeuroSymbolicState, StateTransition, and SimpleStateEncoder.
"""

import pytest

from src.neuro_symbolic.state import (
    Fact,
    NeuroSymbolicState,
    StateTransition,
    SymbolicFactType,
)


@pytest.mark.unit
class TestFact:
    """Tests for Fact dataclass."""

    def test_basic(self):
        f = Fact(name="parent", arguments=("john", "mary"))
        assert f.name == "parent"
        assert f.arguments == ("john", "mary")
        assert f.fact_type == SymbolicFactType.PREDICATE
        assert f.confidence == 1.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Fact(name="", arguments=())

    def test_confidence_clamp(self):
        f = Fact(name="test", arguments=(), confidence=1.5)
        assert f.confidence <= 1.0

    def test_to_string(self):
        f = Fact(name="parent", arguments=("john", "mary"))
        assert f.to_string() == "parent(john, mary)"

    def test_matches(self):
        f1 = Fact(name="parent", arguments=("john", "mary"))
        f2 = Fact(name="parent", arguments=("john", "mary"))
        f3 = Fact(name="parent", arguments=("john", "bob"))
        assert f1.matches(f2)
        assert not f1.matches(f3)

    def test_unify_simple(self):
        f1 = Fact(name="parent", arguments=("john", "mary"))
        template = Fact(name="parent", arguments=("?x", "?y"))
        bindings = f1.unify(template)
        assert bindings == {"x": "john", "y": "mary"}

    def test_unify_fails_name_mismatch(self):
        f1 = Fact(name="parent", arguments=("john",))
        template = Fact(name="child", arguments=("?x",))
        assert f1.unify(template) is None

    def test_unify_fails_arity_mismatch(self):
        f1 = Fact(name="parent", arguments=("john",))
        template = Fact(name="parent", arguments=("?x", "?y"))
        assert f1.unify(template) is None

    def test_unify_conflicting_bindings(self):
        f1 = Fact(name="same", arguments=("a", "b"))
        template = Fact(name="same", arguments=("?x", "?x"))
        assert f1.unify(template) is None

    def test_substitute(self):
        f = Fact(name="parent", arguments=("?x", "?y"))
        result = f.substitute({"x": "john", "y": "mary"})
        assert result.arguments == ("john", "mary")

    def test_to_hash_key(self):
        f = Fact(name="parent", arguments=("john",))
        key = f.to_hash_key()
        assert isinstance(key, str)
        assert len(key) == 16


@pytest.mark.unit
class TestNeuroSymbolicState:
    """Tests for NeuroSymbolicState."""

    def test_basic(self):
        state = NeuroSymbolicState(state_id="s1")
        assert state.state_id == "s1"
        assert len(state.facts) == 0
        assert state.confidence == 1.0

    def test_confidence_clamp(self):
        state = NeuroSymbolicState(state_id="s1", confidence=1.5)
        assert state.confidence <= 1.0

    def test_hash_key(self):
        state = NeuroSymbolicState(state_id="s1")
        key = state.hash_key
        assert isinstance(key, str)
        assert state.hash_key == key  # cached

    def test_fact_index(self):
        facts = frozenset([
            Fact(name="a", arguments=(1,)),
            Fact(name="a", arguments=(2,)),
            Fact(name="b", arguments=(3,)),
        ])
        state = NeuroSymbolicState(state_id="s1", facts=facts)
        idx = state.fact_index
        assert len(idx["a"]) == 2
        assert len(idx["b"]) == 1

    def test_add_fact(self):
        state = NeuroSymbolicState(state_id="s1")
        fact = Fact(name="ready", arguments=("agent",))
        new_state = state.add_fact(fact)
        assert len(new_state.facts) == 1
        assert len(state.facts) == 0  # immutable

    def test_remove_fact(self):
        fact = Fact(name="ready", arguments=("agent",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([fact]))
        new_state = state.remove_fact(fact)
        assert len(new_state.facts) == 0

    def test_add_constraint(self):
        state = NeuroSymbolicState(state_id="s1")
        new_state = state.add_constraint("x < 10")
        assert "x < 10" in new_state.constraints

    def test_has_fact(self):
        fact = Fact(name="ready", arguments=("agent",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([fact]))
        assert state.has_fact("ready", "agent")
        assert not state.has_fact("ready", "other")

    def test_query_facts(self):
        facts = frozenset([
            Fact(name="parent", arguments=("john", "mary")),
            Fact(name="parent", arguments=("john", "bob")),
            Fact(name="child", arguments=("mary", "john")),
        ])
        state = NeuroSymbolicState(state_id="s1", facts=facts)
        results = state.query_facts("parent")
        assert len(results) == 2

    def test_query_facts_with_filter(self):
        facts = frozenset([
            Fact(name="parent", arguments=("john", "mary")),
            Fact(name="parent", arguments=("john", "bob")),
        ])
        state = NeuroSymbolicState(state_id="s1", facts=facts)
        results = state.query_facts("parent", arg1="mary")
        assert len(results) == 1

    def test_similarity_symbolic_only(self):
        f1 = Fact(name="a", arguments=(1,))
        f2 = Fact(name="b", arguments=(2,))
        s1 = NeuroSymbolicState(state_id="s1", facts=frozenset([f1, f2]))
        s2 = NeuroSymbolicState(state_id="s2", facts=frozenset([f1]))
        sim = s1.similarity(s2)
        assert 0.0 < sim < 1.0

    def test_similarity_identical(self):
        f = Fact(name="a", arguments=(1,))
        s1 = NeuroSymbolicState(state_id="s1", facts=frozenset([f]))
        s2 = NeuroSymbolicState(state_id="s2", facts=frozenset([f]))
        assert s1.similarity(s2) == 1.0

    def test_similarity_empty(self):
        s1 = NeuroSymbolicState(state_id="s1")
        s2 = NeuroSymbolicState(state_id="s2")
        assert s1.similarity(s2) == 1.0  # Both empty

    def test_to_dict(self):
        fact = Fact(name="ready", arguments=("x",))
        state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([fact]),
            constraints=frozenset(["c1"]),
            metadata={"key": "val"},
        )
        d = state.to_dict()
        assert d["state_id"] == "s1"
        assert len(d["facts"]) == 1
        assert "c1" in d["constraints"]

    def test_from_dict(self):
        d = {
            "state_id": "s1",
            "facts": [{"name": "ready", "arguments": ["x"], "type": "PREDICATE"}],
            "constraints": ["c1"],
            "confidence": 0.9,
            "metadata": {"key": "val"},
        }
        state = NeuroSymbolicState.from_dict(d)
        assert state.state_id == "s1"
        assert len(state.facts) == 1
        assert state.confidence == 0.9

    def test_hash_and_eq(self):
        f = Fact(name="a", arguments=(1,))
        s1 = NeuroSymbolicState(state_id="s1", facts=frozenset([f]))
        s2 = NeuroSymbolicState(state_id="s2", facts=frozenset([f]))
        assert s1 == s2  # Same facts
        # Hash based on state_id
        assert hash(s1) != hash(s2)


@pytest.mark.unit
class TestStateTransition:
    """Tests for StateTransition."""

    def test_valid_transition(self):
        pre = Fact(name="ready", arguments=("agent",))
        post = Fact(name="done", arguments=("agent",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([pre]))
        t = StateTransition(
            from_state=state,
            to_state=state,  # placeholder
            action="process",
            preconditions=frozenset([pre]),
            postconditions=frozenset([post]),
        )
        assert t.is_valid()
        new_state = t.apply()
        assert new_state.has_fact("done", "agent")
        assert new_state.has_fact("ready", "agent")

    def test_invalid_transition(self):
        pre = Fact(name="ready", arguments=("agent",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())
        t = StateTransition(
            from_state=state,
            to_state=state,
            action="process",
            preconditions=frozenset([pre]),
        )
        assert not t.is_valid()
        with pytest.raises(ValueError, match="preconditions not satisfied"):
            t.apply()

    def test_negation_in_postcondition(self):
        ready = Fact(name="ready", arguments=("agent",))
        not_ready = Fact(name="not_ready", arguments=("agent",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([ready]))
        t = StateTransition(
            from_state=state,
            to_state=state,
            action="shutdown",
            preconditions=frozenset(),
            postconditions=frozenset([not_ready]),
        )
        new_state = t.apply()
        assert not new_state.has_fact("ready", "agent")

    def test_transition_metadata(self):
        state = NeuroSymbolicState(state_id="s1", facts=frozenset())
        t = StateTransition(
            from_state=state,
            to_state=state,
            action="step",
            cost=0.5,
            probability=0.9,
        )
        new_state = t.apply()
        assert new_state.metadata["last_action"] == "step"
        assert new_state.metadata["transition_cost"] == 0.5
        assert new_state.confidence == pytest.approx(0.9)


@pytest.mark.unit
class TestSimpleStateEncoder:
    """Tests for SimpleStateEncoder (requires torch)."""

    def test_encode(self):
        torch = pytest.importorskip("torch")
        from src.neuro_symbolic.state import SimpleStateEncoder

        encoder = SimpleStateEncoder(embedding_dim=32, vocab_size=100)
        fact = Fact(name="test", arguments=("x",))
        state = NeuroSymbolicState(state_id="s1", facts=frozenset([fact]))
        embedding = encoder.encode(state)
        assert embedding.shape == (32,)

    def test_encode_empty_state(self):
        torch = pytest.importorskip("torch")
        from src.neuro_symbolic.state import SimpleStateEncoder

        encoder = SimpleStateEncoder(embedding_dim=32)
        state = NeuroSymbolicState(state_id="s1")
        embedding = encoder.encode(state)
        assert embedding.shape == (32,)

    def test_decode(self):
        torch = pytest.importorskip("torch")
        from src.neuro_symbolic.state import SimpleStateEncoder

        encoder = SimpleStateEncoder(embedding_dim=32)
        emb = torch.randn(32)
        result = encoder.decode(emb)
        assert "embedding_norm" in result
        assert "embedding_dim" in result
