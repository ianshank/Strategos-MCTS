"""
Tests for enterprise base use case module.

Tests BaseDomainState, exception hierarchy, DomainAgentProtocol,
and BaseUseCase patterns.
"""

import copy

import pytest

from src.enterprise.base.use_case import (
    AgentProcessingError,
    BaseDomainState,
    EnterpriseUseCaseError,
    MCTSSearchError,
    StateValidationError,
)


@pytest.mark.unit
class TestBaseDomainState:
    """Tests for BaseDomainState dataclass."""

    def test_init_minimal(self):
        """Test state creation with required fields only."""
        state = BaseDomainState(state_id="test-1", domain="test_domain")
        assert state.state_id == "test-1"
        assert state.domain == "test_domain"
        assert state.features == {}
        assert state.metadata == {}

    def test_init_with_features(self):
        """Test state creation with features."""
        features = {"risk_score": 0.7, "compliance_level": "high"}
        state = BaseDomainState(
            state_id="test-2",
            domain="compliance",
            features=features,
        )
        assert state.features["risk_score"] == 0.7
        assert state.features["compliance_level"] == "high"

    def test_to_mcts_state(self):
        """Test conversion to MCTS state."""
        state = BaseDomainState(
            state_id="mcts-1",
            domain="due_diligence",
            features={"phase": "analysis", "score": 0.8},
        )
        mcts_state = state.to_mcts_state()
        assert mcts_state.state_id == "mcts-1"
        assert mcts_state.features["domain"] == "due_diligence"
        assert mcts_state.features["phase"] == "analysis"
        assert mcts_state.features["score"] == 0.8

    def test_to_hash_key_deterministic(self):
        """Test hash key generation is deterministic."""
        state = BaseDomainState(
            state_id="hash-1",
            domain="test",
            features={"a": 1, "b": 2},
        )
        key1 = state.to_hash_key()
        key2 = state.to_hash_key()
        assert key1 == key2
        assert len(key1) == 16  # SHA256 truncated to 16 chars

    def test_to_hash_key_different_for_different_features(self):
        """Test different features produce different hash keys."""
        state1 = BaseDomainState(state_id="hash-1", domain="test", features={"a": 1})
        state2 = BaseDomainState(state_id="hash-1", domain="test", features={"a": 2})
        assert state1.to_hash_key() != state2.to_hash_key()

    def test_to_hash_key_same_features_different_order(self):
        """Test feature order doesn't affect hash (sorted keys)."""
        state1 = BaseDomainState(
            state_id="hash-1",
            domain="test",
            features={"a": 1, "b": 2},
        )
        state2 = BaseDomainState(
            state_id="hash-1",
            domain="test",
            features={"b": 2, "a": 1},
        )
        assert state1.to_hash_key() == state2.to_hash_key()

    def test_copy_creates_deep_copy(self):
        """Test copy creates independent state."""
        original = BaseDomainState(
            state_id="copy-1",
            domain="test",
            features={"nested": {"value": 42}},
            metadata={"tag": "original"},
        )
        copied = original.copy()

        # Verify values are equal
        assert copied.state_id == original.state_id
        assert copied.features == original.features

        # Verify independence
        copied.features["nested"]["value"] = 99
        assert original.features["nested"]["value"] == 42

    def test_copy_preserves_type(self):
        """Test copy preserves the state type."""
        state = BaseDomainState(state_id="type-1", domain="test")
        copied = state.copy()
        assert type(copied) is BaseDomainState


@pytest.mark.unit
class TestEnterpriseExceptions:
    """Tests for enterprise exception hierarchy."""

    def test_base_exception(self):
        """Test EnterpriseUseCaseError is proper exception."""
        with pytest.raises(EnterpriseUseCaseError):
            raise EnterpriseUseCaseError("test error")

    def test_mcts_search_error_inherits(self):
        """Test MCTSSearchError inherits from base."""
        err = MCTSSearchError("search failed")
        assert isinstance(err, EnterpriseUseCaseError)
        assert isinstance(err, Exception)

    def test_agent_processing_error(self):
        """Test AgentProcessingError captures agent name and original error."""
        original = ValueError("bad input")
        err = AgentProcessingError("hrm_agent", original)
        assert err.agent_name == "hrm_agent"
        assert err.original_error is original
        assert "hrm_agent" in str(err)
        assert isinstance(err, EnterpriseUseCaseError)

    def test_state_validation_error(self):
        """Test StateValidationError inherits from base."""
        err = StateValidationError("invalid state")
        assert isinstance(err, EnterpriseUseCaseError)

    @pytest.mark.parametrize(
        "error_class",
        [EnterpriseUseCaseError, MCTSSearchError, StateValidationError],
    )
    def test_exception_can_be_caught_as_base(self, error_class):
        """Test all exceptions can be caught as EnterpriseUseCaseError."""
        with pytest.raises(EnterpriseUseCaseError):
            raise error_class("test")


@pytest.mark.unit
class TestDomainStateEdgeCases:
    """Tests for edge cases in BaseDomainState."""

    def test_empty_features_hash(self):
        """Test hash works with empty features."""
        state = BaseDomainState(state_id="empty", domain="test")
        key = state.to_hash_key()
        assert isinstance(key, str)
        assert len(key) == 16

    def test_complex_features_hash(self):
        """Test hash works with complex nested features."""
        state = BaseDomainState(
            state_id="complex",
            domain="test",
            features={
                "list": [1, 2, 3],
                "nested": {"deep": {"value": True}},
                "none": None,
            },
        )
        key = state.to_hash_key()
        assert isinstance(key, str)

    def test_metadata_not_in_hash(self):
        """Test metadata doesn't affect hash key."""
        state1 = BaseDomainState(
            state_id="meta-1",
            domain="test",
            features={"a": 1},
            metadata={"tag": "v1"},
        )
        state2 = BaseDomainState(
            state_id="meta-1",
            domain="test",
            features={"a": 1},
            metadata={"tag": "v2"},
        )
        assert state1.to_hash_key() == state2.to_hash_key()
