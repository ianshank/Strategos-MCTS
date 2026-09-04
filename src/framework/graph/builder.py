"""
LangGraph Integration Module - Extract graph building with new MCTS core integration.

Provides:
- Graph building extracted from LangGraphMultiAgentFramework
- Integration with new deterministic MCTS core
- Backward compatibility with original process() signature
- Support for parallel HRM/TRM execution
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from .builder_components.consensusnodes_mixin import ConsensusNodesMixin
from .builder_components.corenodes_mixin import CoreNodesMixin
from .builder_components.metacontrollernodes_mixin import MetaControllerNodesMixin
from .builder_components.routingnodes_mixin import RoutingNodesMixin

if TYPE_CHECKING:
    StateGraph: Any = None
    MemorySaver: Any = None
    END = '__end__'
else:
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph
    except ImportError:
        StateGraph = None
        END = '__end__'
        MemorySaver = None
from ..mcts.config import ConfigPreset, MCTSConfig, create_preset_config
from ..mcts.core import MCTSEngine, MCTSNode, MCTSState
from ..mcts.experiments import ExperimentTracker
from ..mcts.policies import HybridRolloutPolicy
from ..mcts.scoring import CandidateScorer, IdentityCandidateScorer, candidates_from_action_stats

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
from src.observability.logging import get_logger

from .retry import NodeRetryPolicy, set_node_attempts
from .schema import GraphConstructionError, validate_graph_topology, validate_state_schema
from .state import AgentState
from .tracing import GraphTraceRecorder

logger = get_logger(__name__)

class GraphBuilder(MetaControllerNodesMixin, RoutingNodesMixin, ConsensusNodesMixin, CoreNodesMixin):
    """
    Builds and configures the LangGraph state machine for multi-agent orchestration.

    Extracts graph building logic from LangGraphMultiAgentFramework for modularity.
    """

    def __init__(self, hrm_agent, trm_agent, model_adapter, logger, vector_store=None, mcts_config: MCTSConfig | None=None, top_k_retrieval: int=5, max_iterations: int=3, consensus_threshold: float=0.75, enable_parallel_agents: bool=True, meta_controller_config: Any | None=None, adk_agents: dict[str, Any] | None=None, neuro_symbolic_config: Any | None=None, synthesis_temperature: float=0.5, retry_policy: NodeRetryPolicy | None=None, trace_recorder: GraphTraceRecorder | None=None, candidate_scorer: CandidateScorer | None=None):
        """
        Initialize graph builder.

        Args:
            hrm_agent: HRM agent instance
            trm_agent: TRM agent instance
            model_adapter: Model adapter for LLM calls
            logger: Logger instance
            vector_store: Optional vector store for RAG
            mcts_config: MCTS configuration (uses balanced preset if None)
            top_k_retrieval: Number of documents for RAG
            max_iterations: Maximum agent iterations
            consensus_threshold: Threshold for consensus
            enable_parallel_agents: Run HRM/TRM in parallel
            meta_controller_config: Optional neural meta-controller configuration
            adk_agents: Dictionary of ADK agent instances
            neuro_symbolic_config: Optional neuro-symbolic configuration
        """
        self.hrm_agent = hrm_agent
        self.trm_agent = trm_agent
        self.model_adapter = model_adapter
        self.logger = logger
        self.vector_store = vector_store
        self.top_k_retrieval = top_k_retrieval
        self.max_iterations = max_iterations
        self.consensus_threshold = consensus_threshold
        self.enable_parallel_agents = enable_parallel_agents
        self.adk_agents = adk_agents or {}
        self.synthesis_temperature = synthesis_temperature
        self.retry_policy = retry_policy or NodeRetryPolicy(enabled=False)
        self.trace_recorder = trace_recorder
        self.candidate_scorer: CandidateScorer = candidate_scorer or IdentityCandidateScorer()
        self.mcts_config = mcts_config or create_preset_config(ConfigPreset.BALANCED)
        self.mcts_engine = MCTSEngine(seed=self.mcts_config.seed, exploration_weight=self.mcts_config.exploration_weight, progressive_widening_k=self.mcts_config.progressive_widening_k, progressive_widening_alpha=self.mcts_config.progressive_widening_alpha, max_parallel_rollouts=self.mcts_config.max_parallel_rollouts, cache_size_limit=self.mcts_config.cache_size_limit)
        self.experiment_tracker = ExperimentTracker(name='langgraph_mcts')
        self.meta_controller: Any | None = None
        self.meta_controller_config = meta_controller_config
        self.use_neural_routing = False
        if meta_controller_config is not None:
            self._init_meta_controller(meta_controller_config)
        self.symbolic_agent: Any | None = None
        self.symbolic_extension: Any | None = None
        self.neuro_symbolic_mcts: Any | None = None
        self.use_symbolic_reasoning = False
        if neuro_symbolic_config is not None:
            self._init_neuro_symbolic(neuro_symbolic_config)
        logger.debug('GraphBuilder initialized: max_iterations=%d, consensus_threshold=%.2f, parallel_agents=%s, mcts_seed=%d', self.max_iterations, self.consensus_threshold, self.enable_parallel_agents, self.mcts_config.seed)

    def build_graph(self) -> StateGraph:
        """
        Build LangGraph state machine.

        Returns:
            Configured StateGraph
        """
        if StateGraph is None:
            raise ImportError('LangGraph not installed. Install with: pip install langgraph')
        logger.info('Building LangGraph state machine')
        validate_state_schema(AgentState)
        workflow = StateGraph(AgentState)
        node_names: set[str] = set()
        static_edges: list[tuple[str, str]] = []
        conditional_targets: list[str] = []

        def _add_node(name: str, handler: Any) -> None:
            if name in node_names:
                raise GraphConstructionError(f"Duplicate node name: '{name}'")
            workflow.add_node(name, self._wrap_node(handler, name))
            node_names.add(name)

        def _add_edge(source: str, destination: str) -> None:
            workflow.add_edge(source, destination)
            static_edges.append((source, destination))
        _add_node('entry', self._entry_node)
        _add_node('retrieve_context', self._retrieve_context_node)
        _add_node('route_decision', self._route_decision_node)
        _add_node('parallel_agents', self._parallel_agents_node)
        _add_node('hrm_agent', self._hrm_agent_node)
        _add_node('trm_agent', self._trm_agent_node)
        _add_node('mcts_simulator', self._mcts_simulator_node)
        for name, agent in self.adk_agents.items():
            _add_node(f'adk_{name}', self._create_adk_node_handler(name, agent))
        if self.use_symbolic_reasoning and self.symbolic_extension:
            _add_node('symbolic_agent', self._symbolic_agent_node)
        _add_node('aggregate_results', self._aggregate_results_node)
        _add_node('evaluate_consensus', self._evaluate_consensus_node)
        _add_node('synthesize', self._synthesize_node)
        workflow.set_entry_point('entry')
        _add_edge('entry', 'retrieve_context')
        _add_edge('retrieve_context', 'route_decision')
        routing_map = {'parallel': 'parallel_agents', 'hrm': 'hrm_agent', 'trm': 'trm_agent', 'mcts': 'mcts_simulator', 'aggregate': 'aggregate_results'}
        if self.use_symbolic_reasoning:
            routing_map['symbolic'] = 'symbolic_agent'
        for name in self.adk_agents:
            routing_map[f'adk_{name}'] = f'adk_{name}'
        workflow.add_conditional_edges('route_decision', self._route_to_agents, routing_map)
        conditional_targets.extend(routing_map.values())
        _add_edge('parallel_agents', 'aggregate_results')
        _add_edge('hrm_agent', 'aggregate_results')
        _add_edge('trm_agent', 'aggregate_results')
        _add_edge('mcts_simulator', 'aggregate_results')
        if self.use_symbolic_reasoning:
            _add_edge('symbolic_agent', 'aggregate_results')
        for name in self.adk_agents:
            _add_edge(f'adk_{name}', 'aggregate_results')
        _add_edge('aggregate_results', 'evaluate_consensus')
        consensus_map = {'synthesize': 'synthesize', 'iterate': 'route_decision'}
        workflow.add_conditional_edges('evaluate_consensus', self._check_consensus, consensus_map)
        conditional_targets.extend(consensus_map.values())
        _add_edge('synthesize', END)
        validate_graph_topology(nodes=node_names, edges=static_edges, conditional_targets=conditional_targets, entry_point='entry', terminal=END)
        return workflow

    async def _neural_fallback_for_symbolic(self, query: str, _state: Any) -> str:
        """Neural fallback when symbolic reasoning fails."""
        try:
            response = await self.model_adapter.generate(prompt=f'Answer this question: {query}', temperature=self.synthesis_temperature)
            return str(response.text)
        except Exception as e:
            self.logger.error(f'Neural fallback failed: {e}')
            return f'Could not determine answer for: {query}'

    async def _parallel_agents_node(self, state: AgentState) -> dict:
        """Execute HRM and TRM agents in parallel."""
        self.logger.info('Executing HRM and TRM agents in parallel')
        attempts_box = [1]

        def _aggregate_attempts(_exc: Exception, attempt: int) -> None:
            attempts_box[0] = max(attempts_box[0], attempt + 1)

        async def _hrm_call() -> Any:
            return await self.hrm_agent.process(query=state['query'], rag_context=state.get('rag_context'))

        async def _trm_call() -> Any:
            return await self.trm_agent.process(query=state['query'], rag_context=state.get('rag_context'))
        hrm_task = asyncio.create_task(self._node_retry('parallel_agents', _hrm_call, on_retry=_aggregate_attempts)())
        trm_task = asyncio.create_task(self._node_retry('parallel_agents', _trm_call, on_retry=_aggregate_attempts)())
        try:
            hrm_result, trm_result = await asyncio.gather(hrm_task, trm_task)
        finally:
            set_node_attempts(attempts_box[0])
        return {'hrm_results': {'response': hrm_result['response'], 'metadata': hrm_result['metadata']}, 'trm_results': {'response': trm_result['response'], 'metadata': trm_result['metadata']}, 'agent_outputs': [{'agent': 'hrm', 'response': hrm_result['response'], 'confidence': hrm_result['metadata'].get('decomposition_quality_score', 0.7)}, {'agent': 'trm', 'response': trm_result['response'], 'confidence': trm_result['metadata'].get('final_quality_score', 0.7)}]}

    async def _hrm_agent_node(self, state: AgentState) -> dict:
        """Execute HRM agent."""
        self.logger.info('Executing HRM agent')

        async def _call() -> Any:
            return await self.hrm_agent.process(query=state['query'], rag_context=state.get('rag_context'))
        result = await self._node_retry('hrm_agent', _call)()
        return {'hrm_results': {'response': result['response'], 'metadata': result['metadata']}, 'agent_outputs': [{'agent': 'hrm', 'response': result['response'], 'confidence': result['metadata'].get('decomposition_quality_score', 0.7)}]}

    async def _trm_agent_node(self, state: AgentState) -> dict:
        """Execute TRM agent."""
        self.logger.info('Executing TRM agent')

        async def _call() -> Any:
            return await self.trm_agent.process(query=state['query'], rag_context=state.get('rag_context'))
        result = await self._node_retry('trm_agent', _call)()
        return {'trm_results': {'response': result['response'], 'metadata': result['metadata']}, 'agent_outputs': [{'agent': 'trm', 'response': result['response'], 'confidence': result['metadata'].get('final_quality_score', 0.7)}]}

    async def _symbolic_agent_node(self, state: AgentState) -> dict:
        """Execute symbolic reasoning agent."""
        self.logger.info('Executing symbolic reasoning agent')
        if not self.symbolic_extension:
            return {'agent_outputs': [{'agent': 'symbolic', 'response': 'Symbolic reasoning not available', 'confidence': 0.0}]}
        extension = self.symbolic_extension

        async def _handle() -> Any:
            return await extension.handle_symbolic_node(state)
        result = await self._node_retry('symbolic_agent', _handle)()
        proof_tree = None
        if 'symbolic_results' in result:
            metadata = result['symbolic_results'].get('metadata', {})
            proof_tree = metadata.get('proof_tree')
        return {'symbolic_results': result.get('symbolic_results', {}), 'symbolic_proof_tree': proof_tree, 'agent_outputs': result.get('agent_outputs', [])}

    async def _mcts_simulator_node(self, state: AgentState) -> dict:
        """Execute MCTS simulation using new deterministic engine."""
        self.logger.info('Executing MCTS simulation with deterministic engine')
        start_time = time.perf_counter()
        self.mcts_engine.clear_cache()
        root_state = MCTSState(state_id='root', features={'query': state['query'][:100], 'has_hrm': 'hrm_results' in state, 'has_trm': 'trm_results' in state})
        root = MCTSNode(state=root_state, rng=self.mcts_engine.rng)

        def action_generator(mcts_state: MCTSState) -> list[str]:
            """Generate available actions for state."""
            depth = len(mcts_state.state_id.split('_')) - 1
            if depth == 0:
                return ['action_A', 'action_B', 'action_C', 'action_D']
            elif depth < self.mcts_config.max_tree_depth:
                return ['continue', 'refine', 'fallback', 'escalate']
            else:
                return []

        def state_transition(mcts_state: MCTSState, action: str) -> MCTSState:
            """Compute next state from action."""
            new_id = f'{mcts_state.state_id}_{action}'
            new_features = mcts_state.features.copy()
            new_features['last_action'] = action
            new_features['depth'] = len(new_id.split('_')) - 1
            return MCTSState(state_id=new_id, features=new_features)

        def heuristic_fn(_mcts_state: MCTSState) -> float:
            """Evaluate state using agent confidence."""
            base = 0.5
            if state.get('hrm_results'):
                hrm_conf = state['hrm_results']['metadata'].get('decomposition_quality_score', 0.5)
                base += hrm_conf * 0.2
            if state.get('trm_results'):
                trm_conf = state['trm_results']['metadata'].get('final_quality_score', 0.5)
                base += trm_conf * 0.2
            return min(base, 1.0)
        rollout_policy = HybridRolloutPolicy(heuristic_fn=heuristic_fn, heuristic_weight=0.7, random_weight=0.3)
        early_stop_threshold = self.mcts_config.early_stop_threshold if self.mcts_config.enable_early_termination else 0.0
        best_action, stats = await self.mcts_engine.search(root=root, num_iterations=self.mcts_config.num_iterations, action_generator=action_generator, state_transition=state_transition, rollout_policy=rollout_policy, max_rollout_depth=self.mcts_config.max_rollout_depth, selection_policy=self.mcts_config.selection_policy, early_termination_threshold=self.mcts_config.early_termination_threshold, min_iterations_before_termination=self.mcts_config.min_iterations_before_termination, early_stop_threshold=early_stop_threshold, early_stop_patience=self.mcts_config.early_stop_patience)
        if stats.get('early_stopped') or stats.get('termination_reason'):
            self.logger.debug('MCTS search terminated early', extra={'termination_reason': stats.get('termination_reason'), 'iterations_run': stats.get('iterations_run'), 'early_stopped': stats.get('early_stopped')})
        best_action_visits = stats['best_action_visits']
        best_action_value = stats['best_action_value']
        raw_action_stats = stats.get('action_stats')
        action_stats = raw_action_stats if isinstance(raw_action_stats, dict) else {}
        scored_action = self.candidate_scorer.select_best(candidates_from_action_stats(action_stats), engine_choice=best_action)
        if scored_action is not None and scored_action != best_action and (scored_action in action_stats):
            chosen = action_stats[scored_action]
            self.logger.debug('Candidate scorer re-ranked MCTS selection', extra={'scorer': self.candidate_scorer.name, 'engine_choice': best_action, 'scored_action': scored_action})
            best_action = scored_action
            best_action_visits = chosen['visits']
            best_action_value = chosen['value']
            stats = {**stats, 'best_action': best_action, 'best_action_visits': best_action_visits, 'best_action_value': best_action_value}
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        tree_depth = self.mcts_engine.get_tree_depth(root)
        tree_node_count = self.mcts_engine.count_nodes(root)
        self.experiment_tracker.create_result(experiment_id=f'mcts_{int(time.time())}', config=self.mcts_config, mcts_stats=stats, execution_time_ms=execution_time_ms, tree_depth=tree_depth, tree_node_count=tree_node_count, metadata={'query': state['query'][:100], 'has_rag': state.get('use_rag', False)})
        self.logger.info(f"MCTS complete: best_action={best_action}, iterations={stats['iterations']}, cache_hit_rate={stats['cache_hit_rate']:.2%}")
        return {'mcts_root': {'state_id': root.state.state_id, 'tree_depth': tree_depth, 'tree_node_count': tree_node_count}, 'mcts_best_action': best_action, 'mcts_stats': stats, 'agent_outputs': [{'agent': 'mcts', 'response': f"Simulated {stats['iterations']} scenarios with seed {self.mcts_config.seed}. Recommended action: {best_action} (visits={best_action_visits}, value={best_action_value:.3f})", 'confidence': min(best_action_visits / stats['iterations'] if stats['iterations'] > 0 else 0.5, 1.0)}]}

    async def _synthesize_node(self, state: AgentState) -> dict:
        """Synthesize final response from agent outputs."""
        self.logger.info('Synthesizing final response')
        agent_outputs = state.get('agent_outputs', [])
        synthesis_prompt = f"Query: {state['query']}\n\nAgent Outputs:\n"
        for output in agent_outputs:
            synthesis_prompt += f"\n{output['agent'].upper()} (confidence={output['confidence']:.2f}):\n{output['response']}\n\n"
        synthesis_prompt += '\nSynthesize these outputs into a comprehensive final response.\nPrioritize higher-confidence outputs. Integrate insights from all agents.\n\nFinal Response:'
        try:

            async def _generate() -> Any:
                return await self.model_adapter.generate(prompt=synthesis_prompt, temperature=self.synthesis_temperature)
            response = await self._node_retry('synthesize', _generate)()
            final_response = response.text
        except Exception as e:
            self.logger.error(f'Synthesis failed: {e}')
            best_output = max(agent_outputs, key=lambda o: o['confidence'])
            final_response = best_output['response']
        metadata = {'agents_used': [o['agent'] for o in agent_outputs], 'confidence_scores': state.get('confidence_scores', {}), 'consensus_score': state.get('consensus_score', 0.0), 'iterations': state.get('iteration', 0), 'mcts_config': state.get('mcts_config', {})}
        if state.get('mcts_stats'):
            metadata['mcts_stats'] = state['mcts_stats']
        return {'final_response': final_response, 'metadata': metadata}
