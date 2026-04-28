"""Shared LangGraph AgentState TypedDict."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict


class AgentState(TypedDict):
    """Shared state for LangGraph agent framework."""

    # Input
    query: str
    use_mcts: bool
    use_rag: bool

    # RAG context
    rag_context: NotRequired[str]
    retrieved_docs: NotRequired[list[dict]]

    # Agent results
    hrm_results: NotRequired[dict]
    trm_results: NotRequired[dict]
    adk_results: NotRequired[dict[str, Any]]
    agent_outputs: Annotated[list[dict], operator.add]

    # MCTS simulation (updated for new core)
    mcts_root: NotRequired[Any]  # MCTSNode
    mcts_iterations: NotRequired[int]
    mcts_best_action: NotRequired[str]
    mcts_stats: NotRequired[dict]
    mcts_config: NotRequired[dict]

    # Evaluation
    confidence_scores: NotRequired[dict[str, float]]
    consensus_reached: NotRequired[bool]
    consensus_score: NotRequired[float]

    # Control flow
    iteration: int
    max_iterations: int

    # Neural Meta-Controller (optional)
    routing_history: NotRequired[list[dict]]
    meta_controller_predictions: NotRequired[list[dict]]
    last_routed_agent: NotRequired[str]

    # Neuro-Symbolic Agent (optional)
    symbolic_results: NotRequired[dict]
    symbolic_proof_tree: NotRequired[dict]

    # Output
    final_response: NotRequired[str]
    metadata: NotRequired[dict]
