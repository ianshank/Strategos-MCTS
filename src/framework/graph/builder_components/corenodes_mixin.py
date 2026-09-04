from collections.abc import Callable
from typing import Any

from src.framework.graph.retry import with_node_retry
from src.framework.graph.state import AgentState
from src.framework.graph.tracing import make_traced_node
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

class CoreNodesMixin:

    def _wrap_node(self, handler: Any, name: str) -> Any:
        """Return the registered form of a node ``handler``.

        Single wrapping seam applied to every node at registration time: when a trace
        recorder is configured, every node (deterministic ones included, so the full
        execution path is reconstructable) is wrapped to emit a structured transition event.
        """
        recorder = getattr(self, 'trace_recorder', None)
        if recorder is None:
            return handler
        return make_traced_node(recorder, handler, name)

    def _node_retry(self, node_name: str, fn: Callable[[], Any], on_retry: Callable[[Exception, int], None] | None=None) -> Callable[[], Any]:
        """Wrap a zero-arg node I/O callable with the configured retry policy.

        Returns ``fn`` unchanged when retries are disabled or the node is not retryable, so
        deterministic nodes and the default (injected-disabled) path carry no overhead.
        Tolerates a builder constructed via ``__new__`` (no policy attribute) by not retrying.
        ``on_retry`` lets concurrent nodes aggregate attempt counts (see ``_parallel_agents_node``).
        """
        policy = getattr(self, 'retry_policy', None)
        if policy is None:
            return fn
        return with_node_retry(policy, node_name, fn, on_retry=on_retry)

    def _entry_node(self, state: AgentState) -> dict:
        """Initialize state and parse query with validation."""
        query = state.get('query', '')
        if not query or not isinstance(query, str):
            raise ValueError('Query must be a non-empty string')
        query = query.strip()
        if not query:
            raise ValueError('Query cannot be empty or whitespace only')
        self.logger.info(f"Entry node: {query[:100]}{('...' if len(query) > 100 else '')}")
        return {'iteration': 0, 'agent_outputs': [], 'mcts_config': self.mcts_config.to_dict()}

    def _retrieve_context_node(self, state: AgentState) -> dict:
        """Retrieve context from vector store using RAG with error handling."""
        if not state.get('use_rag', True) or not self.vector_store:
            return {'rag_context': '', 'retrieved_docs': []}
        query = state.get('query', '')
        if not query:
            self.logger.warning('Empty query in retrieve_context_node')
            return {'rag_context': '', 'retrieved_docs': []}
        try:

            def _search() -> Any:
                return self.vector_store.similarity_search(query, k=self.top_k_retrieval)
            docs = self._node_retry('retrieve_context', _search)()
            context = '\n\n'.join([doc.page_content for doc in docs])
            self.logger.info(f'Retrieved {len(docs)} documents')
            return {'rag_context': context, 'retrieved_docs': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]}
        except Exception:
            self.logger.exception('RAG retrieval failed')
            return {'rag_context': '', 'retrieved_docs': []}

    def _create_adk_node_handler(self, name: str, agent: Any):
        """Create a handler function for an ADK agent node."""

        async def handler(state: AgentState) -> dict:
            self.logger.info(f'Executing ADK agent: {name}')
            if hasattr(agent, 'initialize'):
                await agent.initialize()
            try:

                async def _invoke() -> Any:
                    if hasattr(agent, 'process_query'):
                        return await agent.process_query(state['query'])
                    if hasattr(agent, 'run'):
                        return await agent.run(state['query'])
                    if hasattr(agent, 'process'):
                        return await agent.process(state['query'])
                    return {'response': f'Processed by {name}', 'confidence': 0.8}
                response = await self._node_retry(f'adk_{name}', _invoke)()
                if isinstance(response, dict):
                    content = response.get('response', str(response))
                    confidence = response.get('confidence', 0.8)
                    metadata = response.get('metadata', {})
                else:
                    content = str(response)
                    confidence = 0.8
                    metadata = {}
                return {'adk_results': {name: {'response': content, 'metadata': metadata}}, 'agent_outputs': [{'agent': f'adk_{name}', 'response': content, 'confidence': confidence}]}
            except Exception as e:
                self.logger.error(f'ADK agent {name} failed: {e}')
                return {'agent_outputs': [{'agent': f'adk_{name}', 'response': f'Error executing {name}: {e}', 'confidence': 0.0}]}
        return handler
