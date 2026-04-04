"""Unit tests for enterprise base use case module (src/enterprise/base/use_case.py).

Tests cover:
- BaseDomainState (to_mcts_state, to_hash_key, copy)
- Custom exception classes
- BaseUseCase (initialization, properties, lifecycle methods)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.enterprise.base.use_case import (
    AgentProcessingError,
    BaseDomainState,
    BaseUseCase,
    EnterpriseUseCaseError,
    MCTSSearchError,
    StateValidationError,
)

# ---------- Test domain state ----------


@pytest.mark.unit
class TestBaseDomainState:
    """Tests for BaseDomainState."""

    def test_create_state(self):
        state = BaseDomainState(state_id="test_1", domain="test_domain")
        assert state.state_id == "test_1"
        assert state.domain == "test_domain"
        assert state.features == {}
        assert state.metadata == {}

    def test_create_state_with_features(self):
        features = {"key": "value", "count": 42}
        state = BaseDomainState(state_id="s1", domain="d1", features=features)
        assert state.features["key"] == "value"
        assert state.features["count"] == 42

    def test_to_mcts_state(self):
        state = BaseDomainState(
            state_id="mcts_test",
            domain="finance",
            features={"risk_level": "high"},
        )
        mcts_state = state.to_mcts_state()
        assert mcts_state.state_id == "mcts_test"
        assert mcts_state.features["domain"] == "finance"
        assert mcts_state.features["risk_level"] == "high"

    def test_to_hash_key(self):
        state = BaseDomainState(state_id="hash_test", domain="test")
        key = state.to_hash_key()
        assert isinstance(key, str)
        assert len(key) == 16  # truncated SHA256

    def test_to_hash_key_deterministic(self):
        state1 = BaseDomainState(state_id="same", domain="d", features={"a": 1})
        state2 = BaseDomainState(state_id="same", domain="d", features={"a": 1})
        assert state1.to_hash_key() == state2.to_hash_key()

    def test_to_hash_key_different_features(self):
        state1 = BaseDomainState(state_id="s", domain="d", features={"a": 1})
        state2 = BaseDomainState(state_id="s", domain="d", features={"a": 2})
        assert state1.to_hash_key() != state2.to_hash_key()

    def test_copy(self):
        state = BaseDomainState(
            state_id="orig",
            domain="test",
            features={"data": [1, 2, 3]},
        )
        copied = state.copy()
        assert copied.state_id == "orig"
        assert copied.features["data"] == [1, 2, 3]
        # Deep copy - mutating copy shouldn't affect original
        copied.features["data"].append(4)
        assert len(state.features["data"]) == 3


# ---------- Test exceptions ----------


@pytest.mark.unit
class TestEnterpriseExceptions:
    """Tests for enterprise exception classes."""

    def test_enterprise_use_case_error(self):
        err = EnterpriseUseCaseError("test error")
        assert str(err) == "test error"
        assert isinstance(err, Exception)

    def test_mcts_search_error(self):
        err = MCTSSearchError("search failed")
        assert isinstance(err, EnterpriseUseCaseError)
        assert str(err) == "search failed"

    def test_agent_processing_error(self):
        original = ValueError("bad input")
        err = AgentProcessingError("analysis_agent", original)
        assert err.agent_name == "analysis_agent"
        assert err.original_error is original
        assert "analysis_agent" in str(err)
        assert isinstance(err, EnterpriseUseCaseError)

    def test_state_validation_error(self):
        err = StateValidationError("invalid state")
        assert isinstance(err, EnterpriseUseCaseError)
        assert str(err) == "invalid state"


# ---------- Test BaseUseCase ----------


@dataclass
class _TestState(BaseDomainState):
    """Test domain state for testing BaseUseCase."""

    domain: str = "test"
    custom_field: str = ""
    actions_applied: list[str] = field(default_factory=list)


class _TestUseCase(BaseUseCase[_TestState]):
    """Concrete test implementation of BaseUseCase."""

    @property
    def name(self) -> str:
        return "test_use_case"

    @property
    def domain(self) -> str:
        return "testing"

    def get_initial_state(self, query: str, context: dict[str, Any]) -> _TestState:
        return _TestState(
            state_id="init_1",
            domain="testing",
            custom_field=query,
            features={"query": query},
        )

    def get_available_actions(self, state: _TestState) -> list[str]:
        return ["action_a", "action_b", "action_c"]

    def apply_action(self, state: _TestState, action: str) -> _TestState:
        import copy

        new_state = copy.deepcopy(state)
        new_state.actions_applied.append(action)
        return new_state


@pytest.mark.unit
class TestBaseUseCase:
    """Tests for BaseUseCase base class."""

    def test_init_defaults(self):
        config = MagicMock()
        uc = _TestUseCase(config=config)
        assert uc.name == "test_use_case"
        assert uc.domain == "testing"
        assert uc.config is config
        assert uc.is_initialized is False
        assert uc._llm_client is None

    def test_init_with_deps(self):
        config = MagicMock()
        llm = MagicMock()
        logger = logging.getLogger("test")
        uc = _TestUseCase(config=config, llm_client=llm, logger=logger)
        assert uc._llm_client is llm
        assert uc._logger is logger

    def test_get_initial_state(self):
        uc = _TestUseCase(config=MagicMock())
        state = uc.get_initial_state("test query", {})
        assert state.custom_field == "test query"
        assert state.features["query"] == "test query"

    def test_get_available_actions(self):
        uc = _TestUseCase(config=MagicMock())
        state = _TestState(state_id="s1")
        actions = uc.get_available_actions(state)
        assert len(actions) == 3
        assert "action_a" in actions

    def test_apply_action(self):
        uc = _TestUseCase(config=MagicMock())
        state = _TestState(state_id="s1")
        new_state = uc.apply_action(state, "action_a")
        assert "action_a" in new_state.actions_applied
        assert len(state.actions_applied) == 0  # Original unchanged

    def test_initialize(self):
        uc = _TestUseCase(config=MagicMock())
        assert not uc.is_initialized
        uc.initialize()
        assert uc.is_initialized

    def test_config_property(self):
        config = MagicMock(max_retries=3)
        uc = _TestUseCase(config=config)
        assert uc.config.max_retries == 3
