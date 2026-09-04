from __future__ import annotations

from src.framework.graph.state import AgentState

try:
    from src.agents.meta_controller.feature_extractor import FeatureExtractor, MetaControllerFeatures
    _META_CONTROLLER_AVAILABLE = True
except ImportError:
    _META_CONTROLLER_AVAILABLE = False
    MetaControllerFeatures = None

try:
    from src.agents.neuro_symbolic.agent import SymbolicReasoningAgent
    from src.agents.neuro_symbolic.config import SymbolicAgentNodeConfig
    from src.agents.neuro_symbolic.graph_extension import SymbolicAgentGraphExtension
    from src.agents.neuro_symbolic.mcts_integration import NeuroSymbolicMCTSConfig, NeuroSymbolicMCTSIntegration
    _NEURO_SYMBOLIC_AVAILABLE = True
except ImportError:
    _NEURO_SYMBOLIC_AVAILABLE = False
    SymbolicReasoningAgent = None
    SymbolicAgentNodeConfig = None
    SymbolicAgentGraphExtension = None
    NeuroSymbolicMCTSConfig = None
    NeuroSymbolicMCTSIntegration = None

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)

class ConsensusNodesMixin:

    def _aggregate_results_node(self, state: AgentState) -> dict:
        """Aggregate results from all agents."""
        self.logger.info('Aggregating agent results')
        agent_outputs = state.get('agent_outputs', [])
        confidence_scores = {output['agent']: output['confidence'] for output in agent_outputs}
        return {'confidence_scores': confidence_scores}

    def _evaluate_consensus_node(self, state: AgentState) -> dict:
        """Evaluate consensus among agents and increment iteration counter.

        The iteration counter is incremented here to ensure proper loop termination
        when consensus is not reached and max_iterations is checked in _check_consensus.
        """
        agent_outputs = state.get('agent_outputs', [])
        current_iteration = state.get('iteration', 0)
        next_iteration = current_iteration + 1
        if len(agent_outputs) < 2:
            self.logger.debug(f'Single agent output, auto-consensus (iteration={next_iteration})')
            return {'consensus_reached': True, 'consensus_score': 1.0, 'iteration': next_iteration}
        avg_confidence = sum(o['confidence'] for o in agent_outputs) / len(agent_outputs)
        consensus_reached = avg_confidence >= self.consensus_threshold
        self.logger.info(f"Consensus: {consensus_reached} (score={avg_confidence:.2f}, iteration={next_iteration}/{state.get('max_iterations', self.max_iterations)})")
        return {'consensus_reached': consensus_reached, 'consensus_score': avg_confidence, 'iteration': next_iteration}

    def _check_consensus(self, state: AgentState) -> str:
        """Check if consensus reached or need more iterations.

        Returns:
            'synthesize' if consensus reached or max iterations exceeded
            'iterate' if more iterations needed
        """
        current_iteration = state.get('iteration', 0)
        max_iter = state.get('max_iterations', self.max_iterations)
        if state.get('consensus_reached', False):
            self.logger.info(f'Consensus reached at iteration {current_iteration}')
            return 'synthesize'
        if current_iteration >= max_iter:
            self.logger.warning(f'Max iterations ({max_iter}) reached without consensus, proceeding to synthesis')
            return 'synthesize'
        self.logger.debug(f'Continuing iteration loop (current={current_iteration}, max={max_iter})')
        return 'iterate'
