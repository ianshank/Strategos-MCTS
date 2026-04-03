"""
Tests for enterprise base use case module.

Tests BaseDomainState, exception hierarchy, BaseUseCase patterns,
and result synthesis.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.enterprise.base.use_case import (
    AgentProcessingError,
    BaseDomainState,
    BaseUseCase,
    EnterpriseUseCaseError,
    MCTSSearchError,
    StateValidationError,
)


@pytest.mark.unit
class TestExceptions:
    """Tests for exception hierarchy."""

    def test_enterprise_use_case_error(self):
        with pytest.raises(EnterpriseUseCaseError):
            raise EnterpriseUseCaseError("test")

    def test_mcts_search_error_is_enterprise(self):
        assert issubclass(MCTSSearchError, EnterpriseUseCaseError)

    def test_agent_processing_error(self):
        orig = ValueError("bad value")
        err = AgentProcessingError("hrm", orig)
        assert err.agent_name == "hrm"
        assert err.original_error is orig
        assert "hrm" in str(err)
        assert issubclass(AgentProcessingError, EnterpriseUseCaseError)

    def test_state_validation_error(self):
        assert issubclass(StateValidationError, EnterpriseUseCaseError)


@pytest.mark.unit
class TestBaseDomainState:
    """Tests for BaseDomainState dataclass."""

    def test_init_minimal(self):
        state = BaseDomainState(state_id="s1", domain="test")
        assert state.state_id == "s1"
        assert state.domain == "test"
        assert state.features == {}
        assert state.metadata == {}

    def test_init_with_features(self):
        features = {"risk": 0.7, "score": 0.9}
        state = BaseDomainState(state_id="s2", domain="finance", features=features)
        assert state.features["risk"] == 0.7

    def test_to_hash_key(self):
        state = BaseDomainState(state_id="s1", domain="test", features={"a": 1})
        key = state.to_hash_key()
        assert isinstance(key, str)
        assert len(key) == 16
        # Deterministic
        assert state.to_hash_key() == key

    def test_copy(self):
        state = BaseDomainState(
            state_id="s1", domain="test",
            features={"nested": {"a": 1}},
        )
        copied = state.copy()
        assert copied.state_id == state.state_id
        # Deep copy
        copied.features["nested"]["a"] = 2
        assert state.features["nested"]["a"] == 1

    def test_to_mcts_state(self):
        state = BaseDomainState(
            state_id="s1", domain="finance",
            features={"risk": 0.5},
        )
        mcts_state = state.to_mcts_state()
        assert mcts_state.state_id == "s1"
        assert mcts_state.features["domain"] == "finance"
        assert mcts_state.features["risk"] == 0.5


@pytest.mark.unit
class TestBaseUseCase:
    """Tests for BaseUseCase abstract base class."""

    def _make_concrete_use_case(self, **kwargs):
        """Create a concrete subclass for testing."""

        class ConcreteUseCase(BaseUseCase):
            @property
            def name(self):
                return "test_use_case"

            @property
            def domain(self):
                return "test_domain"

            def get_initial_state(self, query, context):
                return BaseDomainState(
                    state_id="init",
                    domain="test_domain",
                    features={"query": query},
                )

            def get_available_actions(self, state):
                return ["analyze", "summarize", "conclude"]

            def apply_action(self, state, action):
                new_features = {**state.features, "last_action": action}
                return BaseDomainState(
                    state_id=f"{state.state_id}_{action}",
                    domain=state.domain,
                    features=new_features,
                )

        config = MagicMock()
        config.enabled = kwargs.get("enabled", False)
        return ConcreteUseCase(config=config, **{k: v for k, v in kwargs.items() if k != "enabled"})

    def test_init(self):
        uc = self._make_concrete_use_case()
        assert uc.name == "test_use_case"
        assert uc.domain == "test_domain"
        assert not uc.is_initialized

    def test_config_property(self):
        uc = self._make_concrete_use_case()
        assert uc.config is not None

    def test_initialize(self):
        uc = self._make_concrete_use_case()
        uc.initialize()
        assert uc.is_initialized
        # Second call is no-op
        uc.initialize()
        assert uc.is_initialized

    def test_get_initial_state(self):
        uc = self._make_concrete_use_case()
        state = uc.get_initial_state("test query", {})
        assert state.state_id == "init"
        assert state.features["query"] == "test query"

    def test_get_available_actions(self):
        uc = self._make_concrete_use_case()
        state = BaseDomainState(state_id="s1", domain="test")
        actions = uc.get_available_actions(state)
        assert "analyze" in actions
        assert len(actions) == 3

    def test_apply_action(self):
        uc = self._make_concrete_use_case()
        state = BaseDomainState(state_id="s1", domain="test", features={})
        new_state = uc.apply_action(state, "analyze")
        assert new_state.state_id == "s1_analyze"
        assert new_state.features["last_action"] == "analyze"

    def test_get_reward_function_not_initialized(self):
        uc = self._make_concrete_use_case()
        with pytest.raises(RuntimeError, match="not initialized"):
            uc.get_reward_function()

    def test_get_domain_agents_triggers_init(self):
        uc = self._make_concrete_use_case()
        agents = uc.get_domain_agents()
        assert uc.is_initialized
        assert isinstance(agents, dict)

    def test_custom_logger(self):
        logger = logging.getLogger("test_uc")
        uc = self._make_concrete_use_case(logger=logger)
        assert uc._logger is logger

    def test_synthesize_results(self):
        uc = self._make_concrete_use_case()
        agent_results = {
            "agent1": {"response": "Analysis done", "confidence": 0.9},
            "agent2": {"response": "Summary done", "confidence": 0.8},
            "agent3": {"error": "timeout", "confidence": 0.0},
        }
        mcts_result = {"best_action": "analyze", "stats": {}}
        result = uc._synthesize_results(agent_results, mcts_result)
        assert result["confidence"] == pytest.approx(0.85)
        assert result["agent_count"] == 2
        assert "Analysis done" in result["response"]
        assert result["mcts_action"] == "analyze"

    def test_synthesize_results_no_valid_agents(self):
        uc = self._make_concrete_use_case()
        agent_results = {
            "agent1": {"error": "fail"},
        }
        result = uc._synthesize_results(agent_results, {})
        assert result["confidence"] == 0.0
        assert result["agent_count"] == 0

    @pytest.mark.asyncio
    async def test_process_without_mcts(self):
        uc = self._make_concrete_use_case(enabled=False)
        result = await uc.process("test query", use_mcts=False)
        assert result["use_case"] == "test_use_case"
        assert result["domain"] == "test_domain"
        assert isinstance(result["agent_results"], dict)

    @pytest.mark.asyncio
    async def test_process_with_agents(self):
        uc = self._make_concrete_use_case(enabled=False)
        mock_agent = AsyncMock()
        mock_agent.process.return_value = {"response": "done", "confidence": 0.9}
        uc._agents = {"test_agent": mock_agent}
        uc._initialized = True

        result = await uc.process("query", use_mcts=False)
        assert "test_agent" in result["agent_results"]

    @pytest.mark.asyncio
    async def test_process_agent_error_handling(self):
        uc = self._make_concrete_use_case(enabled=False)
        mock_agent = AsyncMock()
        mock_agent.process.side_effect = ValueError("bad input")
        uc._agents = {"bad_agent": mock_agent}
        uc._initialized = True

        result = await uc.process("query", use_mcts=False)
        assert "error" in result["agent_results"]["bad_agent"]
        assert result["agent_results"]["bad_agent"]["error_type"] == "ValueError"
