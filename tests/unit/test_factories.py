"""
Tests for framework factories module.

Tests LLMClientFactory, AgentFactory, MCTSEngineFactory,
MetaControllerFactory, HybridAgentFactory, and FrameworkFactory.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.framework.factories import (
    AgentFactory,
    FrameworkFactory,
    HybridAgentFactory,
    LLMClientFactory,
    MCTSEngineFactory,
    MetaControllerFactory,
)


def _mock_settings():
    """Create a mock Settings instance."""
    s = MagicMock()
    s.LLM_PROVIDER = "openai"
    s.HTTP_TIMEOUT_SECONDS = 60.0
    s.HTTP_MAX_RETRIES = 3
    s.SEED = 42
    s.MCTS_C = 1.414
    s.MCTS_ENABLED = True
    s.HRM_H_DIM = 256
    s.HRM_L_DIM = 128
    s.HRM_NUM_H_LAYERS = 3
    s.HRM_NUM_L_LAYERS = 2
    s.HRM_MAX_OUTER_STEPS = 10
    s.HRM_HALT_THRESHOLD = 0.01
    s.TRM_LATENT_DIM = 256
    s.TRM_HIDDEN_DIM = 512
    s.TRM_NUM_RECURSIONS = 5
    s.TRM_CONVERGENCE_THRESHOLD = 0.001
    s.META_CONTROLLER_TYPE = "rnn"
    s.META_CONTROLLER_INPUT_DIM = 10
    s.META_CONTROLLER_HIDDEN_DIM = 64
    s.META_CONTROLLER_NUM_LAYERS = 2
    s.META_CONTROLLER_NUM_AGENTS = 3
    s.META_CONTROLLER_DROPOUT = 0.1
    s.HYBRID_MODE = "auto"
    s.HYBRID_POLICY_CONFIDENCE_THRESHOLD = 0.7
    s.HYBRID_VALUE_CONFIDENCE_THRESHOLD = 0.6
    s.HYBRID_NEURAL_COST_PER_CALL = 0.0001
    s.HYBRID_LLM_COST_PER_1K_TOKENS = 0.03
    return s


@pytest.mark.unit
class TestLLMClientFactory:
    """Tests for LLMClientFactory."""

    def test_init_with_settings(self):
        settings = _mock_settings()
        factory = LLMClientFactory(settings=settings)
        assert factory.settings is settings

    @patch("src.framework.factories.get_settings")
    def test_init_default_settings(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        factory = LLMClientFactory()
        assert factory.settings is mock_get_settings.return_value

    def test_get_default_model_openai(self):
        factory = LLMClientFactory(settings=_mock_settings())
        assert factory._get_default_model("openai") == "gpt-4-turbo-preview"

    def test_get_default_model_anthropic(self):
        factory = LLMClientFactory(settings=_mock_settings())
        assert factory._get_default_model("anthropic") == "claude-3-sonnet-20240229"

    def test_get_default_model_lmstudio(self):
        factory = LLMClientFactory(settings=_mock_settings())
        assert factory._get_default_model("lmstudio") == "local-model"

    def test_get_default_model_unknown(self):
        factory = LLMClientFactory(settings=_mock_settings())
        assert factory._get_default_model("unknown") == "gpt-4-turbo-preview"

    @patch("src.framework.factories.LLMClientFactory.create")
    def test_create_from_settings(self, mock_create):
        mock_create.return_value = MagicMock()
        factory = LLMClientFactory(settings=_mock_settings())
        factory.create_from_settings()
        mock_create.assert_called_once()

    @patch("src.adapters.llm.create_client")
    def test_create_with_provider(self, mock_create_client):
        mock_create_client.return_value = MagicMock()
        factory = LLMClientFactory(settings=_mock_settings())
        factory.create(provider="openai", model="gpt-4")
        mock_create_client.assert_called_once()
        call_kwargs = mock_create_client.call_args
        assert call_kwargs.kwargs["provider"] == "openai"
        assert call_kwargs.kwargs["model"] == "gpt-4"


@pytest.mark.unit
class TestAgentFactory:
    """Tests for AgentFactory."""

    def test_init(self):
        llm_client = MagicMock()
        settings = _mock_settings()
        factory = AgentFactory(llm_client=llm_client, settings=settings)
        assert factory.llm_client is llm_client

    @patch("src.agents.hrm_agent.create_hrm_agent")
    @patch("src.training.system_config.HRMConfig")
    def test_create_hrm_agent(self, mock_config_cls, mock_create):
        mock_create.return_value = MagicMock()
        mock_config_cls.return_value = MagicMock(h_dim=256, l_dim=128)
        mock_config_cls.__dataclass_fields__ = {}

        factory = AgentFactory(llm_client=MagicMock(), settings=_mock_settings())
        factory.create_hrm_agent(h_dim=256, l_dim=128)
        mock_create.assert_called_once()

    @patch("src.agents.trm_agent.create_trm_agent")
    @patch("src.training.system_config.TRMConfig")
    def test_create_trm_agent(self, mock_config_cls, mock_create):
        mock_create.return_value = MagicMock()
        mock_config_cls.return_value = MagicMock(latent_dim=256, num_recursions=5)
        mock_config_cls.__dataclass_fields__ = {}

        factory = AgentFactory(llm_client=MagicMock(), settings=_mock_settings())
        factory.create_trm_agent(latent_dim=256, output_dim=10)
        mock_create.assert_called_once()


@pytest.mark.unit
class TestMCTSEngineFactory:
    """Tests for MCTSEngineFactory."""

    def test_init(self):
        settings = _mock_settings()
        factory = MCTSEngineFactory(settings=settings)
        assert factory.settings is settings

    @patch("src.framework.mcts.core.MCTSEngine")
    def test_create_with_defaults(self, mock_engine):
        mock_engine.return_value = MagicMock()
        factory = MCTSEngineFactory(settings=_mock_settings())
        factory.create()
        mock_engine.assert_called_once()
        call_kwargs = mock_engine.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["exploration_weight"] == 1.414

    @patch("src.framework.mcts.core.MCTSEngine")
    def test_create_with_custom_seed(self, mock_engine):
        mock_engine.return_value = MagicMock()
        factory = MCTSEngineFactory(settings=_mock_settings())
        factory.create(seed=99, exploration_weight=2.0)
        call_kwargs = mock_engine.call_args.kwargs
        assert call_kwargs["seed"] == 99
        assert call_kwargs["exploration_weight"] == 2.0

    @patch("src.framework.mcts.config.FAST_CONFIG")
    @patch("src.framework.mcts.core.MCTSEngine")
    def test_create_with_preset(self, mock_engine, mock_fast):
        mock_fast.__dict__ = {"num_iterations": 50}
        mock_engine.return_value = MagicMock()
        factory = MCTSEngineFactory(settings=_mock_settings())
        factory.create(config_preset="fast")
        mock_engine.assert_called_once()

    def test_get_preset_config_unknown(self):
        factory = MCTSEngineFactory(settings=_mock_settings())
        with pytest.raises(ValueError, match="Unknown preset"):
            factory._get_preset_config("nonexistent")


@pytest.mark.unit
class TestMetaControllerFactory:
    """Tests for MetaControllerFactory."""

    def test_init(self):
        factory = MetaControllerFactory(settings=_mock_settings())
        assert factory.settings is not None

    @patch("src.agents.meta_controller.rnn_controller.RNNMetaController")
    def test_create_rnn(self, mock_rnn):
        mock_rnn.return_value = MagicMock()
        factory = MetaControllerFactory(settings=_mock_settings())
        factory.create(controller_type="rnn")
        mock_rnn.assert_called_once()

    @patch("src.agents.meta_controller.bert_controller.BERTMetaController")
    def test_create_bert(self, mock_bert):
        mock_bert.return_value = MagicMock()
        factory = MetaControllerFactory(settings=_mock_settings())
        factory.create(controller_type="bert")
        mock_bert.assert_called_once()

    @patch("src.agents.meta_controller.rnn_controller.RNNMetaController")
    @patch("src.agents.meta_controller.hybrid_controller.HybridMetaController")
    def test_create_hybrid(self, mock_hybrid, mock_rnn):
        mock_rnn.return_value = MagicMock()
        mock_hybrid.return_value = MagicMock()
        factory = MetaControllerFactory(settings=_mock_settings())
        factory.create(controller_type="hybrid")
        mock_hybrid.assert_called_once()

    @patch("src.agents.meta_controller.assembly_router.AssemblyRouter")
    def test_create_assembly(self, mock_assembly):
        mock_assembly.return_value = MagicMock()
        factory = MetaControllerFactory(settings=_mock_settings())
        factory.create(controller_type="assembly")
        mock_assembly.assert_called_once()

    def test_create_unknown_type(self):
        factory = MetaControllerFactory(settings=_mock_settings())
        with pytest.raises(ValueError, match="Unknown controller type"):
            factory.create(controller_type="unknown")


@pytest.mark.unit
class TestHybridAgentFactory:
    """Tests for HybridAgentFactory."""

    def test_init(self):
        factory = HybridAgentFactory(llm_client=MagicMock(), settings=_mock_settings())
        assert factory._llm_client is not None

    @patch("src.agents.hybrid_agent.HybridAgent")
    @patch("src.agents.hybrid_agent.HybridConfig")
    def test_create(self, mock_config, mock_agent):
        mock_config.return_value = MagicMock(mode="auto", policy_confidence_threshold=0.7)
        mock_agent.return_value = MagicMock()

        factory = HybridAgentFactory(llm_client=MagicMock(), settings=_mock_settings())
        factory.create(mode="auto")
        mock_agent.assert_called_once()

    @patch("src.agents.hybrid_agent.HybridAgent")
    @patch("src.agents.hybrid_agent.HybridConfig")
    def test_create_invalid_mode_defaults_to_auto(self, mock_config, mock_agent):
        mock_config.return_value = MagicMock()
        mock_agent.return_value = MagicMock()

        factory = HybridAgentFactory(llm_client=MagicMock(), settings=_mock_settings())
        factory.create(mode="invalid_mode")
        # Should default to "auto" without error
        mock_config.assert_called_once()


@pytest.mark.unit
class TestFrameworkFactory:
    """Tests for FrameworkFactory."""

    def test_init(self):
        factory = FrameworkFactory(settings=_mock_settings())
        assert factory.llm_factory is not None
        assert factory.mcts_factory is not None
        assert factory.meta_controller_factory is not None

    @patch("src.agents.meta_controller.rnn_controller.RNNMetaController")
    @patch("src.framework.mcts.core.MCTSEngine")
    @patch("src.adapters.llm.create_client")
    def test_create_framework(self, mock_llm, mock_mcts, mock_rnn):
        mock_llm.return_value = MagicMock()
        mock_mcts.return_value = MagicMock()
        mock_rnn.return_value = MagicMock()

        factory = FrameworkFactory(settings=_mock_settings())
        result = factory.create_framework(mcts_enabled=True, mcts_seed=42)

        assert "llm_client" in result
        assert "mcts_engine" in result
        assert result["mcts_engine"] is not None

    @patch("src.adapters.llm.create_client")
    def test_create_framework_mcts_disabled(self, mock_llm):
        mock_llm.return_value = MagicMock()

        factory = FrameworkFactory(settings=_mock_settings())
        result = factory.create_framework(mcts_enabled=False, meta_controller_enabled=False)

        assert result["mcts_engine"] is None
        assert result["meta_controller"] is None

    @patch("src.agents.meta_controller.rnn_controller.RNNMetaController")
    def test_create_meta_controller(self, mock_rnn):
        mock_rnn.return_value = MagicMock()
        factory = FrameworkFactory(settings=_mock_settings())
        factory.create_meta_controller(controller_type="rnn")
        mock_rnn.assert_called_once()
