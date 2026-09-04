from typing import Any

from src.framework.graph.state import AgentState
from src.observability.logging import get_structured_logger

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
except ImportError:
    StateGraph = None
    END = "__end__"
    MemorySaver = None

try:
    from src.agents.meta_controller.base import AbstractMetaController, MetaControllerFeatures
    from src.agents.meta_controller.bert_controller import BERTMetaController
    from src.agents.meta_controller.config_loader import MetaControllerConfig, MetaControllerConfigLoader
    from src.agents.meta_controller.rnn_controller import RNNMetaController
    _META_CONTROLLER_AVAILABLE = True
except ImportError:
    _META_CONTROLLER_AVAILABLE = False
    AbstractMetaController = None
    MetaControllerFeatures = None
    RNNMetaController = None
    BERTMetaController = None
    MetaControllerConfig = None
    MetaControllerConfigLoader = None

try:
    from src.neuro_symbolic import (
        ConstraintSystem,
        NeuroSymbolicConfig,
        NeuroSymbolicMCTSConfig,
        NeuroSymbolicMCTSIntegration,
        SymbolicAgentGraphExtension,
        SymbolicAgentNodeConfig,
        SymbolicReasoningAgent,
    )
    from src.neuro_symbolic.config import ConstraintConfig
    _NEURO_SYMBOLIC_AVAILABLE = True
except ImportError:
    _NEURO_SYMBOLIC_AVAILABLE = False
    NeuroSymbolicConfig = None
    SymbolicReasoningAgent = None
    SymbolicAgentGraphExtension = None
    SymbolicAgentNodeConfig = None
    NeuroSymbolicMCTSIntegration = None
    NeuroSymbolicMCTSConfig = None
    ConstraintSystem = None
    ConstraintConfig = None

logger = get_structured_logger(__name__)

class MetaControllerNodesMixin:

    def _init_meta_controller(self, config: Any) -> None:
        """
        Initialize the neural meta-controller based on configuration.

        Args:
            config: MetaControllerConfig or dict with configuration
        """
        if not _META_CONTROLLER_AVAILABLE:
            self.logger.warning('Meta-controller modules not available. Falling back to rule-based routing.')
            return
        try:
            mc_config = MetaControllerConfigLoader.load_from_dict(config) if isinstance(config, dict) else config
            if not mc_config.enabled:
                self.logger.info('Neural meta-controller disabled in config')
                return
            if mc_config.type == 'rnn':
                self.meta_controller = RNNMetaController(name='GraphBuilder_RNN', seed=mc_config.inference.seed, hidden_dim=mc_config.rnn.hidden_dim, num_layers=mc_config.rnn.num_layers, dropout=mc_config.rnn.dropout, device=mc_config.inference.device)
                if mc_config.rnn.model_path:
                    self.meta_controller.load_model(mc_config.rnn.model_path)
                    self.logger.info(f'Loaded RNN model from {mc_config.rnn.model_path}')
            elif mc_config.type == 'bert':
                self.meta_controller = BERTMetaController(name='GraphBuilder_BERT', seed=mc_config.inference.seed, model_name=mc_config.bert.model_name, lora_r=mc_config.bert.lora_r, lora_alpha=mc_config.bert.lora_alpha, lora_dropout=mc_config.bert.lora_dropout, device=mc_config.inference.device, use_lora=mc_config.bert.use_lora)
                if mc_config.bert.model_path:
                    self.meta_controller.load_model(mc_config.bert.model_path)
                    self.logger.info(f'Loaded BERT model from {mc_config.bert.model_path}')
            else:
                raise ValueError(f'Unknown meta-controller type: {mc_config.type}')
            self.use_neural_routing = True
            self.logger.info(f'Initialized {mc_config.type.upper()} neural meta-controller')
        except Exception as e:
            self.logger.error(f'Failed to initialize meta-controller: {e}')
            if hasattr(config, 'fallback_to_rule_based') and config.fallback_to_rule_based:
                self.logger.warning('Falling back to rule-based routing')
            else:
                raise

    def _init_neuro_symbolic(self, config: Any) -> None:
        """
        Initialize neuro-symbolic reasoning components.

        Args:
            config: NeuroSymbolicConfig or dict with configuration
        """
        if not _NEURO_SYMBOLIC_AVAILABLE:
            self.logger.warning('Neuro-symbolic modules not available. Skipping initialization.')
            return
        try:
            if isinstance(config, dict):
                ns_config = NeuroSymbolicConfig.from_dict(config)
            else:
                ns_config = config
            self.symbolic_agent = SymbolicReasoningAgent(config=ns_config, neural_fallback=self._neural_fallback_for_symbolic, logger=self.logger)
            self.symbolic_extension = SymbolicAgentGraphExtension(reasoning_agent=self.symbolic_agent, config=SymbolicAgentNodeConfig(), logger=self.logger)
            mcts_ns_config = NeuroSymbolicMCTSConfig(neural_weight=ns_config.agent.neural_confidence_weight, symbolic_weight=ns_config.agent.symbolic_confidence_weight)
            self.neuro_symbolic_mcts = NeuroSymbolicMCTSIntegration(config=mcts_ns_config, reasoning_agent=self.symbolic_agent, logger=self.logger)
            self.use_symbolic_reasoning = True
            self.logger.info('Initialized neuro-symbolic reasoning components')
        except Exception as e:
            self.logger.error(f'Failed to initialize neuro-symbolic components: {e}')
            self.use_symbolic_reasoning = False

    def _extract_meta_controller_features(self, state: AgentState) -> Any:
        """
        Extract features from AgentState for meta-controller prediction.

        Args:
            state: Current agent state

        Returns:
            MetaControllerFeatures instance
        """
        if not _META_CONTROLLER_AVAILABLE or MetaControllerFeatures is None:
            return None
        hrm_conf = 0.0
        if 'hrm_results' in state:
            hrm_conf = state['hrm_results'].get('metadata', {}).get('decomposition_quality_score', 0.5)
        trm_conf = 0.0
        if 'trm_results' in state:
            trm_conf = state['trm_results'].get('metadata', {}).get('final_quality_score', 0.5)
        mcts_val = 0.0
        if 'mcts_stats' in state:
            mcts_val = state['mcts_stats'].get('best_action_value', 0.5)
        consensus = state.get('consensus_score', 0.0)
        last_agent = state.get('last_routed_agent', 'none')
        iteration = state.get('iteration', 0)
        query_length = len(state.get('query', ''))
        has_rag = bool(state.get('rag_context', ''))
        return MetaControllerFeatures(hrm_confidence=hrm_conf, trm_confidence=trm_conf, mcts_value=mcts_val, consensus_score=consensus, last_agent=last_agent, iteration=iteration, query_length=query_length, has_rag_context=has_rag)
