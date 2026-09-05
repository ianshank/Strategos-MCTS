"""Structural typing for GraphBuilder mixin hosts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphBuilderHost(Protocol):
    """Attrs mixins expect on the composed GraphBuilder instance."""

    logger: Any
    hrm_agent: Any
    trm_agent: Any
    model_adapter: Any
    vector_store: Any
    top_k_retrieval: int
    max_iterations: int
    consensus_threshold: float
    enable_parallel_agents: bool
    adk_agents: dict[str, Any]
    synthesis_temperature: float
    retry_policy: Any
    trace_recorder: Any
    candidate_scorer: Any
    mcts_config: Any
    mcts_engine: Any
    experiment_tracker: Any
    meta_controller: Any
    meta_controller_config: Any
    use_neural_routing: bool
    symbolic_agent: Any
    symbolic_extension: Any
    neuro_symbolic_mcts: Any
    use_symbolic_reasoning: bool

    def _extract_meta_controller_features(self, state: Any) -> Any: ...
    def _wrap_node(self, handler: Any, name: str) -> Any: ...
    def _node_retry(self, name: str, call: Any, on_retry: Any = None) -> Any: ...
    def _create_adk_node_handler(self, name: str, agent: Any) -> Any: ...
    def _init_meta_controller(self, config: Any) -> None: ...
    def _init_neuro_symbolic(self, config: Any) -> None: ...
