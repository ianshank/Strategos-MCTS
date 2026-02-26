"""
Unit tests for neuro-symbolic state representation.

Tests:
- Fact creation and operations
- State creation and manipulation
- State transitions
- Hash consistency
- Encoding/decoding

Best Practices 2025:
- Property-based testing
- Immutability verification
- Edge case coverage
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

from src.neuro_symbolic.state import (
    Fact,
    NeuroSymbolicState,
    SimpleStateEncoder,
    StateTransition,
    SymbolicFactType,
)

# Skip marker for tests requiring torch
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")


class TestFact:
    """Tests for Fact dataclass."""

    def test_fact_creation(self):
        """Test basic fact creation."""
        fact = Fact(name="parent", arguments=("john", "mary"))

        assert fact.name == "parent"
        assert fact.arguments == ("john", "mary")
        assert fact.fact_type == SymbolicFactType.PREDICATE
        assert fact.confidence == 1.0

    def test_fact_with_type_and_confidence(self):
        """Test fact with custom type and confidence."""
        fact = Fact(
            name="height",
            arguments=(180,),
            fact_type=SymbolicFactType.ATTRIBUTE,
            confidence=0.95,
        )

        assert fact.fact_type == SymbolicFactType.ATTRIBUTE
        assert fact.confidence == 0.95

    def test_fact_immutability(self):
        """Test that facts are immutable (frozen dataclass)."""
        fact = Fact(name="test", arguments=("a",))

        with pytest.raises(AttributeError):
            fact.name = "modified"  # type: ignore

    def test_fact_to_string(self):
        """Test Prolog-style string representation."""
        fact = Fact(name="loves", arguments=("romeo", "juliet"))
        assert fact.to_string() == "loves(romeo, juliet)"

        fact_single = Fact(name="valid", arguments=("x",))
        assert fact_single.to_string() == "valid(x)"

    def test_fact_matches(self):
        """Test fact matching."""
        fact1 = Fact(name="parent", arguments=("john", "mary"))
        fact2 = Fact(name="parent", arguments=("john", "mary"))
        fact3 = Fact(name="parent", arguments=("jane", "mary"))

        assert fact1.matches(fact2)
        assert not fact1.matches(fact3)

    def test_fact_unify_with_variables(self):
        """Test unification with variable patterns."""
        fact = Fact(name="parent", arguments=("john", "mary"))
        template = Fact(name="parent", arguments=("?X", "mary"))

        bindings = fact.unify(template)

        assert bindings is not None
        assert bindings["X"] == "john"

    def test_fact_unify_multiple_variables(self):
        """Test unification with multiple variables."""
        fact = Fact(name="edge", arguments=("a", "b", 5))
        template = Fact(name="edge", arguments=("?From", "?To", "?Weight"))

        bindings = fact.unify(template)

        assert bindings is not None
        assert bindings["From"] == "a"
        assert bindings["To"] == "b"
        assert bindings["Weight"] == 5

    def test_fact_unify_failure(self):
        """Test unification failure."""
        fact = Fact(name="parent", arguments=("john", "mary"))
        template = Fact(name="child", arguments=("?X", "mary"))  # Different name

        bindings = fact.unify(template)
        assert bindings is None

    def test_fact_substitute(self):
        """Test variable substitution."""
        template = Fact(name="parent", arguments=("?X", "?Y"))
        bindings = {"X": "john", "Y": "mary"}

        result = template.substitute(bindings)

        assert result.arguments == ("john", "mary")
        assert result.name == "parent"

    def test_fact_hash_key_determinism(self):
        """Test hash key is deterministic."""
        fact = Fact(name="test", arguments=("a", "b", "c"))

        hash1 = fact.to_hash_key()
        hash2 = fact.to_hash_key()

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA-256 truncated

    def test_fact_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        fact_high = Fact(name="test", arguments=(), confidence=1.5)
        fact_low = Fact(name="test", arguments=(), confidence=-0.5)

        assert fact_high.confidence == 1.0
        assert fact_low.confidence == 0.0

    def test_fact_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Fact(name="", arguments=("a",))

    @pytest.mark.property
    @given(
        name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L",))),
        args=st.tuples(st.integers(), st.integers()),
    )
    @settings(max_examples=50)
    def test_property_fact_to_string_roundtrip(self, name: str, args: tuple):
        """Property: Fact string should contain name and all args."""
        fact = Fact(name=name, arguments=args)
        fact_str = fact.to_string()

        assert name in fact_str
        for arg in args:
            assert str(arg) in fact_str


class TestNeuroSymbolicState:
    """Tests for NeuroSymbolicState."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = NeuroSymbolicState(state_id="test_state")

        assert state.state_id == "test_state"
        assert len(state.facts) == 0
        assert len(state.constraints) == 0
        assert state.confidence == 1.0

    def test_state_with_facts(self):
        """Test state creation with facts."""
        facts = frozenset(
            [
                Fact(name="a", arguments=("x",)),
                Fact(name="b", arguments=("y",)),
            ]
        )
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        assert len(state.facts) == 2

    def test_state_add_fact(self):
        """Test adding a fact creates new state."""
        state1 = NeuroSymbolicState(state_id="s1")
        fact = Fact(name="new", arguments=("data",))

        state2 = state1.add_fact(fact)

        assert len(state1.facts) == 0  # Original unchanged
        assert len(state2.facts) == 1
        assert fact in state2.facts

    def test_state_remove_fact(self):
        """Test removing a fact creates new state."""
        fact = Fact(name="removable", arguments=("x",))
        state1 = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([fact]),
        )

        state2 = state1.remove_fact(fact)

        assert len(state1.facts) == 1  # Original unchanged
        assert len(state2.facts) == 0

    def test_state_add_constraint(self):
        """Test adding a constraint creates new state."""
        state1 = NeuroSymbolicState(state_id="s1")
        state2 = state1.add_constraint("x < 10")

        assert len(state1.constraints) == 0
        assert "x < 10" in state2.constraints

    def test_state_query_facts_by_name(self):
        """Test querying facts by name."""
        facts = frozenset(
            [
                Fact(name="edge", arguments=("a", "b")),
                Fact(name="edge", arguments=("b", "c")),
                Fact(name="node", arguments=("a",)),
            ]
        )
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        edge_facts = state.query_facts("edge")
        node_facts = state.query_facts("node")

        assert len(edge_facts) == 2
        assert len(node_facts) == 1

    def test_state_has_fact(self):
        """Test checking if state has a fact."""
        facts = frozenset(
            [
                Fact(name="parent", arguments=("john", "mary")),
            ]
        )
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        assert state.has_fact("parent", "john", "mary")
        assert not state.has_fact("parent", "jane", "mary")
        assert not state.has_fact("child", "john", "mary")

    def test_state_hash_key_consistency(self):
        """Test hash key is consistent for same facts."""
        facts = frozenset(
            [
                Fact(name="a", arguments=(1,)),
                Fact(name="b", arguments=(2,)),
            ]
        )
        state1 = NeuroSymbolicState(state_id="s1", facts=facts)
        state2 = NeuroSymbolicState(state_id="s1", facts=facts)

        assert state1.hash_key == state2.hash_key

    def test_state_hash_key_different_for_different_facts(self):
        """Test hash key differs for different facts."""
        state1 = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([Fact(name="a", arguments=(1,))]),
        )
        state2 = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([Fact(name="b", arguments=(2,))]),
        )

        assert state1.hash_key != state2.hash_key

    def test_state_to_dict_serialization(self):
        """Test state can be serialized to dict."""
        facts = frozenset(
            [
                Fact(name="test", arguments=("a", "b")),
            ]
        )
        state = NeuroSymbolicState(
            state_id="s1",
            facts=facts,
            constraints=frozenset(["x < 10"]),
            confidence=0.9,
            metadata={"key": "value"},
        )

        state_dict = state.to_dict()

        assert state_dict["state_id"] == "s1"
        assert len(state_dict["facts"]) == 1
        assert "x < 10" in state_dict["constraints"]
        assert state_dict["confidence"] == 0.9
        assert state_dict["metadata"]["key"] == "value"

    def test_state_from_dict_deserialization(self):
        """Test state can be deserialized from dict."""
        data = {
            "state_id": "s1",
            "facts": [
                {"name": "test", "arguments": ["a", "b"], "type": "PREDICATE", "confidence": 0.9, "source": "test"},
            ],
            "constraints": ["x < 10"],
            "confidence": 0.95,
            "metadata": {"key": "value"},
        }

        state = NeuroSymbolicState.from_dict(data)

        assert state.state_id == "s1"
        assert len(state.facts) == 1
        assert state.confidence == 0.95

    def test_state_confidence_normalization(self):
        """Test confidence is normalized to [0, 1]."""
        state_high = NeuroSymbolicState(state_id="s1", confidence=1.5)
        state_low = NeuroSymbolicState(state_id="s2", confidence=-0.5)

        assert state_high.confidence == 1.0
        assert state_low.confidence == 0.0

    def test_state_fact_index_caching(self):
        """Test fact index is cached after first access."""
        facts = frozenset(
            [
                Fact(name="a", arguments=(1,)),
                Fact(name="a", arguments=(2,)),
                Fact(name="b", arguments=(3,)),
            ]
        )
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        # First access builds index
        index1 = state.fact_index
        # Second access returns cached
        index2 = state.fact_index

        assert index1 is index2
        assert len(index1["a"]) == 2
        assert len(index1["b"]) == 1


class TestStateTransition:
    """Tests for StateTransition."""

    def test_transition_creation(self):
        """Test basic transition creation."""
        from_state = NeuroSymbolicState(state_id="s1")
        to_state = NeuroSymbolicState(state_id="s2")

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            action="move",
        )

        assert transition.action == "move"
        assert transition.probability == 1.0
        assert transition.cost == 0.0

    def test_transition_with_preconditions(self):
        """Test transition with preconditions."""
        fact = Fact(name="ready", arguments=("agent",))
        from_state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([fact]),
        )
        to_state = NeuroSymbolicState(state_id="s2")

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            action="start",
            preconditions=frozenset([fact]),
        )

        assert transition.is_valid()

    def test_transition_invalid_preconditions(self):
        """Test transition with unsatisfied preconditions."""
        required_fact = Fact(name="ready", arguments=("agent",))
        from_state = NeuroSymbolicState(state_id="s1", facts=frozenset())

        transition = StateTransition(
            from_state=from_state,
            to_state=NeuroSymbolicState(state_id="s2"),
            action="start",
            preconditions=frozenset([required_fact]),
        )

        assert not transition.is_valid()

    def test_transition_apply_adds_postconditions(self):
        """Test applying transition adds postconditions."""
        from_state = NeuroSymbolicState(state_id="s1")
        to_state = NeuroSymbolicState(state_id="s2")
        post_fact = Fact(name="done", arguments=("task",))

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            action="complete",
            postconditions=frozenset([post_fact]),
        )

        result = transition.apply()

        assert result.has_fact("done", "task")
        assert "complete" in result.state_id

    def test_transition_apply_with_negation(self):
        """Test applying transition with negated postconditions."""
        existing_fact = Fact(name="active", arguments=("task",))
        from_state = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([existing_fact]),
        )
        to_state = NeuroSymbolicState(state_id="s2")
        negation = Fact(name="not_active", arguments=("task",))

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            action="stop",
            postconditions=frozenset([negation]),
        )

        result = transition.apply()

        assert not result.has_fact("active", "task")

    def test_transition_apply_invalid_raises(self):
        """Test applying invalid transition raises ValueError."""
        required_fact = Fact(name="ready", arguments=())
        from_state = NeuroSymbolicState(state_id="s1")

        transition = StateTransition(
            from_state=from_state,
            to_state=NeuroSymbolicState(state_id="s2"),
            action="start",
            preconditions=frozenset([required_fact]),
        )

        with pytest.raises(ValueError, match="preconditions not satisfied"):
            transition.apply()


@requires_torch
class TestSimpleStateEncoder:
    """Tests for SimpleStateEncoder (requires torch)."""

    def test_encoder_creation(self):
        """Test encoder initialization."""
        encoder = SimpleStateEncoder(embedding_dim=128, vocab_size=5000, seed=42)

        assert encoder.embedding_dim == 128
        assert encoder.vocab_size == 5000

    def test_encode_empty_state(self):
        """Test encoding empty state."""
        encoder = SimpleStateEncoder(embedding_dim=64)
        state = NeuroSymbolicState(state_id="empty")

        embedding = encoder.encode(state)

        assert embedding.shape == (64,)
        # Empty state should have zero embedding
        assert torch.all(embedding == 0)

    def test_encode_state_with_facts(self):
        """Test encoding state with facts."""
        encoder = SimpleStateEncoder(embedding_dim=64)
        facts = frozenset(
            [
                Fact(name="a", arguments=(1,)),
                Fact(name="b", arguments=(2,)),
            ]
        )
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        embedding = encoder.encode(state)

        assert embedding.shape == (64,)
        # Non-empty state should have non-zero embedding
        assert torch.norm(embedding) > 0

    def test_encode_determinism(self):
        """Test encoding is deterministic with same seed."""
        facts = frozenset([Fact(name="test", arguments=(1,))])
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        encoder1 = SimpleStateEncoder(embedding_dim=64, seed=42)
        encoder2 = SimpleStateEncoder(embedding_dim=64, seed=42)

        emb1 = encoder1.encode(state)
        emb2 = encoder2.encode(state)

        assert torch.allclose(emb1, emb2)

    def test_encode_different_states_different_embeddings(self):
        """Test different states produce different embeddings."""
        encoder = SimpleStateEncoder(embedding_dim=64)

        state1 = NeuroSymbolicState(
            state_id="s1",
            facts=frozenset([Fact(name="a", arguments=(1,))]),
        )
        state2 = NeuroSymbolicState(
            state_id="s2",
            facts=frozenset([Fact(name="b", arguments=(2,))]),
        )

        emb1 = encoder.encode(state1)
        emb2 = encoder.encode(state2)

        # Embeddings should be different
        assert not torch.allclose(emb1, emb2)

    def test_decode_returns_metadata(self):
        """Test decode returns embedding metadata."""
        encoder = SimpleStateEncoder(embedding_dim=64)
        embedding = torch.randn(64)

        metadata = encoder.decode(embedding)

        assert "embedding_norm" in metadata
        assert "embedding_dim" in metadata
        assert metadata["embedding_dim"] == 64

    @pytest.mark.property
    @given(
        embedding_dim=st.integers(min_value=16, max_value=512),
    )
    @settings(max_examples=10)
    def test_property_encoded_embedding_normalized(self, embedding_dim: int):
        """Property: Non-empty state embeddings should be normalized."""
        encoder = SimpleStateEncoder(embedding_dim=embedding_dim)
        facts = frozenset([Fact(name="test", arguments=(1,))])
        state = NeuroSymbolicState(state_id="s1", facts=facts)

        embedding = encoder.encode(state)
        norm = torch.norm(embedding)

        # Should be approximately 1.0 (normalized)
        assert abs(norm - 1.0) < 1e-5
