"""
Unit tests for Hybrid LLM-Neural Agent.

Tests auto/neural_only/llm_only/adaptive routing modes, cost tracking,
and blending of neural and LLM estimates.

Based on: NEXT_STEPS_PLAN.md Phase 2.2
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
    pytest.mark.neural,
]

# Check if torch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

skip_without_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_policy_network():
    """Create a mock policy network."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    @dataclass
    class MockActionSelection:
        action: int
        confidence: float
        entropy: float
        log_prob: float

    policy = MagicMock()
    policy.eval.return_value = None
    policy.select_action.return_value = MockActionSelection(
        action=5,
        confidence=0.85,
        entropy=0.5,
        log_prob=-0.16,
    )
    return policy


@pytest.fixture
def mock_value_network():
    """Create a mock value network."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    value = MagicMock()
    value.eval.return_value = None
    value.evaluate.return_value = 0.7
    value.get_confidence.return_value = 0.8
    return value


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    client.generate.return_value = {"text": "3"}
    return client


@pytest.fixture
def hybrid_config():
    """Create a test hybrid configuration."""
    from src.agents.hybrid_agent import HybridConfig

    return HybridConfig(
        policy_confidence_threshold=0.8,
        value_confidence_threshold=0.7,
        mode="auto",
        track_costs=True,
        neural_cost_per_call=0.000001,
        llm_cost_per_1k_tokens=0.03,
        log_decisions=True,
        prometheus_enabled=False,
    )


@pytest.fixture
def hybrid_agent(mock_policy_network, mock_value_network, mock_llm_client, hybrid_config):
    """Create a hybrid agent for testing."""
    from src.agents.hybrid_agent import HybridAgent

    return HybridAgent(
        policy_net=mock_policy_network,
        value_net=mock_value_network,
        llm_client=mock_llm_client,
        config=hybrid_config,
    )


@pytest.fixture
def sample_state():
    """Create a sample torch state tensor."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    return torch.randn(1, 64)


# =============================================================================
# Initialization Tests
# =============================================================================


@skip_without_torch
class TestHybridAgentInitialization:
    """Tests for HybridAgent initialization."""

    def test_agent_initializes_with_all_components(
        self, mock_policy_network, mock_value_network, mock_llm_client, hybrid_config
    ):
        """Test agent initializes with all components."""
        from src.agents.hybrid_agent import HybridAgent

        agent = HybridAgent(
            policy_net=mock_policy_network,
            value_net=mock_value_network,
            llm_client=mock_llm_client,
            config=hybrid_config,
        )

        assert agent.policy_net is not None
        assert agent.value_net is not None
        assert agent.llm_client is not None

    def test_agent_initializes_with_none_components(self):
        """Test agent initializes with None components."""
        from src.agents.hybrid_agent import HybridAgent

        agent = HybridAgent()

        assert agent.policy_net is None
        assert agent.value_net is None
        assert agent.llm_client is None

    def test_agent_initializes_statistics(self, hybrid_agent):
        """Test agent initializes statistics tracking."""
        assert hybrid_agent.stats["neural_policy_calls"] == 0
        assert hybrid_agent.stats["neural_value_calls"] == 0
        assert hybrid_agent.stats["llm_calls"] == 0
        assert hybrid_agent.stats["total_neural_cost"] == 0.0
        assert hybrid_agent.stats["total_llm_cost"] == 0.0

    def test_agent_sets_networks_to_eval_mode(
        self, mock_policy_network, mock_value_network, mock_llm_client, hybrid_config
    ):
        """Test agent sets networks to eval mode."""
        from src.agents.hybrid_agent import HybridAgent

        HybridAgent(
            policy_net=mock_policy_network,
            value_net=mock_value_network,
            llm_client=mock_llm_client,
            config=hybrid_config,
        )

        mock_policy_network.eval.assert_called_once()
        mock_value_network.eval.assert_called_once()

    def test_agent_uses_default_config_when_none(self, mock_policy_network):
        """Test agent uses default config when None provided."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        agent = HybridAgent(policy_net=mock_policy_network)

        assert isinstance(agent.config, HybridConfig)


# =============================================================================
# Configuration Tests
# =============================================================================


@skip_without_torch
class TestHybridConfig:
    """Tests for HybridConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        from src.agents.hybrid_agent import HybridConfig

        config = HybridConfig()

        assert config.policy_confidence_threshold == 0.8
        assert config.value_confidence_threshold == 0.7
        assert config.mode == "auto"
        assert config.track_costs is True
        assert config.neural_cost_per_call == 0.000001

    def test_config_accepts_custom_values(self):
        """Test config accepts custom values."""
        from src.agents.hybrid_agent import HybridConfig

        config = HybridConfig(
            policy_confidence_threshold=0.9,
            mode="neural_only",
        )

        assert config.policy_confidence_threshold == 0.9
        assert config.mode == "neural_only"

    def test_config_blend_weights_default(self):
        """Test default blend weights."""
        from src.agents.hybrid_agent import HybridConfig

        config = HybridConfig()

        assert config.blend_weights["neural"] == 0.3
        assert config.blend_weights["llm"] == 0.7


# =============================================================================
# Neural Only Mode Tests
# =============================================================================


@skip_without_torch
class TestNeuralOnlyMode:
    """Tests for neural_only routing mode."""

    @pytest.mark.asyncio
    async def test_neural_only_uses_policy_network(self, mock_policy_network, mock_llm_client, sample_state):
        """Test neural_only mode uses policy network."""
        from src.agents.hybrid_agent import DecisionSource, HybridAgent, HybridConfig

        config = HybridConfig(mode="neural_only")
        agent = HybridAgent(
            policy_net=mock_policy_network,
            llm_client=mock_llm_client,
            config=config,
        )

        action, metadata = await agent.select_action(sample_state)

        assert action == 5  # From mock
        assert metadata.source == DecisionSource.POLICY_NETWORK
        assert metadata.confidence == 0.85
        mock_policy_network.select_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_neural_only_does_not_call_llm(self, mock_policy_network, mock_llm_client, sample_state):
        """Test neural_only mode doesn't call LLM."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(mode="neural_only")
        agent = HybridAgent(
            policy_net=mock_policy_network,
            llm_client=mock_llm_client,
            config=config,
        )

        await agent.select_action(sample_state)

        mock_llm_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_neural_only_tracks_statistics(self, mock_policy_network, sample_state):
        """Test neural_only mode tracks statistics."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(mode="neural_only")
        agent = HybridAgent(
            policy_net=mock_policy_network,
            config=config,
        )

        await agent.select_action(sample_state)

        assert agent.stats["neural_policy_calls"] == 1
        assert agent.stats["llm_calls"] == 0


# =============================================================================
# LLM Only Mode Tests
# =============================================================================


@skip_without_torch
class TestLLMOnlyMode:
    """Tests for llm_only routing mode."""

    @pytest.mark.asyncio
    async def test_llm_only_uses_llm_client(self, mock_policy_network, mock_llm_client, sample_state):
        """Test llm_only mode uses LLM client."""
        from src.agents.hybrid_agent import DecisionSource, HybridAgent, HybridConfig

        config = HybridConfig(mode="llm_only")
        agent = HybridAgent(
            policy_net=mock_policy_network,
            llm_client=mock_llm_client,
            config=config,
        )

        action, metadata = await agent.select_action(sample_state)

        assert metadata.source == DecisionSource.LLM
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_only_does_not_use_neural(self, mock_policy_network, mock_llm_client, sample_state):
        """Test llm_only mode doesn't use neural network."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(mode="llm_only")
        agent = HybridAgent(
            policy_net=mock_policy_network,
            llm_client=mock_llm_client,
            config=config,
        )

        await agent.select_action(sample_state)

        mock_policy_network.select_action.assert_not_called()


# =============================================================================
# Auto Mode Tests
# =============================================================================


@skip_without_torch
class TestAutoMode:
    """Tests for auto routing mode."""

    @pytest.mark.asyncio
    async def test_auto_uses_neural_when_confident(self, mock_policy_network, mock_llm_client, sample_state):
        """Test auto mode uses neural when confidence is high."""
        from src.agents.hybrid_agent import DecisionSource, HybridAgent, HybridConfig

        # confidence=0.85, threshold=0.8 -> should use neural
        config = HybridConfig(
            mode="auto",
            policy_confidence_threshold=0.8,
        )
        agent = HybridAgent(
            policy_net=mock_policy_network,
            llm_client=mock_llm_client,
            config=config,
        )

        action, metadata = await agent.select_action(sample_state)

        assert metadata.source == DecisionSource.POLICY_NETWORK
        mock_llm_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_falls_back_to_llm_when_uncertain(self, mock_policy_network, mock_llm_client, sample_state):
        """Test auto mode falls back to LLM when neural is uncertain."""
        from src.agents.hybrid_agent import DecisionSource, HybridAgent, HybridConfig

        # Mock low confidence
        @dataclass
        class MockActionSelection:
            action: int
            confidence: float
            entropy: float
            log_prob: float

        mock_policy_network.select_action.return_value = MockActionSelection(
            action=5,
            confidence=0.5,  # Below threshold
            entropy=0.5,
            log_prob=-0.16,
        )

        config = HybridConfig(
            mode="auto",
            policy_confidence_threshold=0.8,
        )
        agent = HybridAgent(
            policy_net=mock_policy_network,
            llm_client=mock_llm_client,
            config=config,
        )

        action, metadata = await agent.select_action(sample_state)

        assert metadata.source == DecisionSource.LLM_FALLBACK
        assert "neural_confidence" in metadata.additional_info
        mock_llm_client.generate.assert_called_once()


# =============================================================================
# Adaptive Mode Tests
# =============================================================================


@skip_without_torch
class TestAdaptiveMode:
    """Tests for adaptive routing mode."""

    @pytest.mark.asyncio
    async def test_adaptive_tracks_confidences(self, mock_policy_network, sample_state):
        """Test adaptive mode tracks recent confidences."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(
            mode="adaptive",
            adaptive_threshold_window=5,
        )
        agent = HybridAgent(
            policy_net=mock_policy_network,
            config=config,
        )

        # Make multiple calls
        for _ in range(3):
            await agent.select_action(sample_state)

        assert len(agent.recent_confidences) == 3
        assert all(c == 0.85 for c in agent.recent_confidences)

    @pytest.mark.asyncio
    async def test_adaptive_respects_window_limit(self, mock_policy_network, sample_state):
        """Test adaptive mode respects window size limit."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(
            mode="adaptive",
            adaptive_threshold_window=3,
        )
        agent = HybridAgent(
            policy_net=mock_policy_network,
            config=config,
        )

        # Make more calls than window size
        for _ in range(5):
            await agent.select_action(sample_state)

        assert len(agent.recent_confidences) == 3

    def test_adaptive_threshold_calculation(self, mock_policy_network):
        """Test adaptive threshold is calculated correctly."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(
            mode="adaptive",
            adaptive_threshold_window=5,
            adaptive_min_threshold=0.5,
            adaptive_max_threshold=0.95,
        )
        agent = HybridAgent(
            policy_net=mock_policy_network,
            config=config,
        )

        # Populate confidences
        agent.recent_confidences = [0.7, 0.75, 0.8, 0.85, 0.9, 0.8, 0.85, 0.75, 0.8, 0.78, 0.82]

        threshold = agent._get_confidence_threshold()

        # Threshold should be within bounds
        assert config.adaptive_min_threshold <= threshold <= config.adaptive_max_threshold


# =============================================================================
# Cost Tracking Tests
# =============================================================================


@skip_without_torch
class TestCostTracking:
    """Tests for cost tracking functionality."""

    @pytest.mark.asyncio
    async def test_neural_cost_tracked(self, mock_policy_network, sample_state):
        """Test neural network costs are tracked."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(
            mode="neural_only",
            neural_cost_per_call=0.000005,
        )
        agent = HybridAgent(
            policy_net=mock_policy_network,
            config=config,
        )

        await agent.select_action(sample_state)
        await agent.select_action(sample_state)

        assert agent.stats["total_neural_cost"] == pytest.approx(0.00001)

    @pytest.mark.asyncio
    async def test_llm_cost_tracked(self, mock_llm_client, sample_state):
        """Test LLM costs are tracked."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(
            mode="llm_only",
            llm_cost_per_1k_tokens=0.03,
        )
        agent = HybridAgent(
            llm_client=mock_llm_client,
            config=config,
        )

        await agent.select_action(sample_state)

        assert agent.stats["total_llm_cost"] > 0

    def test_cost_savings_calculation(self, hybrid_agent):
        """Test cost savings calculation."""
        from src.agents.hybrid_agent import CostSavings

        # Simulate some calls
        hybrid_agent.stats["neural_policy_calls"] = 100
        hybrid_agent.stats["neural_value_calls"] = 50
        hybrid_agent.stats["llm_calls"] = 10
        hybrid_agent.stats["total_neural_cost"] = 0.00015
        hybrid_agent.stats["total_llm_cost"] = 0.5

        savings = hybrid_agent.get_cost_savings()

        assert isinstance(savings, CostSavings)
        assert savings.total_calls == 160
        assert savings.actual_cost == pytest.approx(0.50015)
        assert savings.savings > 0  # Should show savings vs pure LLM

    def test_cost_savings_percentage(self, hybrid_agent):
        """Test cost savings percentage is calculated correctly."""
        hybrid_agent.stats["neural_policy_calls"] = 100
        hybrid_agent.stats["llm_calls"] = 0
        hybrid_agent.stats["total_neural_cost"] = 0.0001
        hybrid_agent.stats["total_llm_cost"] = 0.0

        savings = hybrid_agent.get_cost_savings()

        assert savings.neural_percentage == 100.0

    def test_cost_savings_zero_calls(self, hybrid_agent):
        """Test cost savings with zero calls."""
        savings = hybrid_agent.get_cost_savings()

        assert savings.total_calls == 0
        assert savings.savings_percentage == 0.0


# =============================================================================
# Value Evaluation Tests
# =============================================================================


@skip_without_torch
class TestValueEvaluation:
    """Tests for position evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_uses_value_network(self, mock_value_network, sample_state):
        """Test evaluate_position uses value network."""
        from src.agents.hybrid_agent import DecisionSource, HybridAgent

        agent = HybridAgent(value_net=mock_value_network)

        value, metadata = await agent.evaluate_position(sample_state)

        assert value == 0.7  # From mock
        assert metadata.source == DecisionSource.VALUE_NETWORK
        assert metadata.confidence == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_blends_when_uncertain(self, mock_value_network, mock_llm_client, sample_state):
        """Test evaluate_position blends neural and LLM when uncertain."""
        from src.agents.hybrid_agent import DecisionSource, HybridAgent, HybridConfig

        # Mock low confidence for value network
        mock_value_network.get_confidence.return_value = 0.4  # Below threshold

        # Mock LLM response
        mock_llm_client.generate.return_value = {"text": "0.5"}

        config = HybridConfig(
            value_confidence_threshold=0.7,
            blend_weights={"neural": 0.3, "llm": 0.7},
        )
        agent = HybridAgent(
            value_net=mock_value_network,
            llm_client=mock_llm_client,
            config=config,
        )

        value, metadata = await agent.evaluate_position(sample_state)

        assert metadata.source == DecisionSource.BLENDED
        assert "llm_value" in metadata.additional_info
        assert "neural_value" in metadata.additional_info

    @pytest.mark.asyncio
    async def test_evaluate_uses_llm_when_no_value_network(self, mock_llm_client, sample_state):
        """Test evaluate_position uses LLM when no value network."""
        from src.agents.hybrid_agent import DecisionSource, HybridAgent

        agent = HybridAgent(llm_client=mock_llm_client)

        mock_llm_client.generate.return_value = {"text": "0.5"}

        value, metadata = await agent.evaluate_position(sample_state)

        assert metadata.source == DecisionSource.LLM
        mock_llm_client.generate.assert_called_once()


# =============================================================================
# Decision Metadata Tests
# =============================================================================


@skip_without_torch
class TestDecisionMetadata:
    """Tests for decision metadata."""

    @pytest.mark.asyncio
    async def test_metadata_includes_latency(self, mock_policy_network, sample_state):
        """Test metadata includes latency measurement."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(mode="neural_only")
        agent = HybridAgent(policy_net=mock_policy_network, config=config)

        _, metadata = await agent.select_action(sample_state)

        assert metadata.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_metadata_includes_cost(self, mock_policy_network, sample_state):
        """Test metadata includes cost."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(
            mode="neural_only",
            neural_cost_per_call=0.000001,
        )
        agent = HybridAgent(policy_net=mock_policy_network, config=config)

        _, metadata = await agent.select_action(sample_state)

        assert metadata.cost == 0.000001

    @pytest.mark.asyncio
    async def test_decision_history_logged(self, mock_policy_network, sample_state):
        """Test decisions are logged to history."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(
            mode="neural_only",
            log_decisions=True,
        )
        agent = HybridAgent(policy_net=mock_policy_network, config=config)

        await agent.select_action(sample_state)
        await agent.select_action(sample_state)

        assert len(agent.stats["decision_history"]) == 2


# =============================================================================
# Statistics Tests
# =============================================================================


@skip_without_torch
class TestStatistics:
    """Tests for statistics reporting."""

    def test_get_statistics_returns_comprehensive_data(self, hybrid_agent):
        """Test get_statistics returns comprehensive data."""
        hybrid_agent.stats["neural_policy_calls"] = 50
        hybrid_agent.stats["llm_calls"] = 5
        hybrid_agent.stats["total_neural_cost"] = 0.00005
        hybrid_agent.stats["total_llm_cost"] = 0.25

        stats = hybrid_agent.get_statistics()

        assert "calls" in stats
        assert "costs" in stats
        assert "cost_savings" in stats

        assert stats["calls"]["neural_policy"] == 50
        assert stats["calls"]["llm"] == 5


# =============================================================================
# Error Handling Tests
# =============================================================================


@skip_without_torch
class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_raises_error_when_policy_net_none(self, sample_state):
        """Test raises error when policy_net is None in neural mode."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(mode="neural_only")
        agent = HybridAgent(config=config)

        with pytest.raises(ValueError, match="Policy network not initialized"):
            await agent.select_action(sample_state)

    @pytest.mark.asyncio
    async def test_raises_error_when_llm_client_none(self, sample_state):
        """Test raises error when llm_client is None in LLM mode."""
        from src.agents.hybrid_agent import HybridAgent, HybridConfig

        config = HybridConfig(mode="llm_only")
        agent = HybridAgent(config=config)

        with pytest.raises(ValueError, match="LLM client not initialized"):
            await agent.select_action(sample_state)

    @pytest.mark.asyncio
    async def test_raises_error_when_both_missing_for_evaluate(self, sample_state):
        """Test raises error when both value_net and llm_client are None."""
        from src.agents.hybrid_agent import HybridAgent

        agent = HybridAgent()

        with pytest.raises(ValueError, match="not initialized"):
            await agent.evaluate_position(sample_state)


# =============================================================================
# Prompt Generation Tests
# =============================================================================


@skip_without_torch
class TestPromptGeneration:
    """Tests for prompt generation utilities."""

    def test_state_to_prompt_includes_state(self, hybrid_agent, sample_state):
        """Test state_to_prompt includes state representation."""
        prompt = hybrid_agent._state_to_prompt(sample_state)

        assert "State:" in prompt
        assert "Action:" in prompt

    def test_state_to_prompt_includes_context(self, hybrid_agent, sample_state):
        """Test state_to_prompt includes context when provided."""
        context = {"game": "chess", "turn": 5}
        prompt = hybrid_agent._state_to_prompt(sample_state, context)

        assert "Context:" in prompt

    def test_evaluation_prompt_format(self, hybrid_agent, sample_state):
        """Test evaluation prompt format."""
        prompt = hybrid_agent._state_to_evaluation_prompt(sample_state)

        assert "Evaluate" in prompt
        assert "Value:" in prompt


# =============================================================================
# Response Parsing Tests
# =============================================================================


@skip_without_torch
class TestResponseParsing:
    """Tests for response parsing."""

    def test_parse_action_from_valid_response(self, hybrid_agent):
        """Test parsing action from valid response."""
        response = {"text": "42 is the best action"}

        action = hybrid_agent._parse_action(response)

        assert action == 42

    def test_parse_action_from_invalid_response(self, hybrid_agent):
        """Test parsing action from invalid response returns 0."""
        response = {"text": "invalid response"}

        action = hybrid_agent._parse_action(response)

        assert action == 0

    def test_parse_value_from_valid_response(self, hybrid_agent):
        """Test parsing value from valid response."""
        response = {"text": "0.75 is the evaluation"}

        value = hybrid_agent._parse_value(response)

        assert value == pytest.approx(0.75)

    def test_parse_value_from_invalid_response(self, hybrid_agent):
        """Test parsing value from invalid response returns 0.0."""
        response = {"text": "invalid"}

        value = hybrid_agent._parse_value(response)

        assert value == 0.0
