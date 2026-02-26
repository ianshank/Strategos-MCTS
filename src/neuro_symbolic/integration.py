"""
Neuro-Symbolic Integration Module.

Integrates neuro-symbolic components with the existing framework:
- MCTS integration with symbolic constraint pruning
- Graph builder extension for symbolic agent routing
- Hybrid confidence scoring
- State conversion utilities

Best Practices 2025:
- Minimal changes to existing code
- Extension over modification
- Clear interface boundaries
- Comprehensive logging
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import ConstraintConfig, NeuroSymbolicConfig
from .constraints import ConstraintSystem
from .reasoning import SymbolicReasoningAgent
from .state import Fact, NeuroSymbolicState, SymbolicFactType


@dataclass
class NeuroSymbolicMCTSConfig:
    """Configuration for neuro-symbolic MCTS integration."""

    # Enable/disable symbolic features
    enable_constraint_pruning: bool = True
    enable_symbolic_heuristics: bool = True
    enable_proof_caching: bool = True

    # Constraint handling
    constraint_check_frequency: int = 1  # Check every N expansions
    max_constraints_per_node: int = 50

    # Hybrid scoring
    neural_weight: float = 0.6
    symbolic_weight: float = 0.4

    # Performance
    async_constraint_checking: bool = True
    constraint_timeout_ms: int = 100

    def __post_init__(self):
        """Validate and normalize configuration."""
        total_weight = self.neural_weight + self.symbolic_weight
        if abs(total_weight - 1.0) > 1e-6:
            self.neural_weight /= total_weight
            self.symbolic_weight /= total_weight


class NeuroSymbolicMCTSIntegration:
    """
    Integration layer between neuro-symbolic system and MCTS.

    Provides:
    - Constraint-based action pruning during expansion
    - Symbolic heuristics for rollout policy
    - Hybrid value estimation
    """

    def __init__(
        self,
        config: NeuroSymbolicMCTSConfig,
        constraint_system: ConstraintSystem | None = None,
        reasoning_agent: SymbolicReasoningAgent | None = None,
        logger: Any | None = None,
    ):
        self.config = config
        self.constraint_system = constraint_system or ConstraintSystem(ConstraintConfig())
        self.reasoning_agent = reasoning_agent
        self.logger = logger

        # Statistics
        self._expansions_checked = 0
        self._actions_pruned = 0
        self._constraint_check_time_ms = 0.0

    def convert_mcts_state(
        self,
        mcts_state: Any,
        action_history: list[str] | None = None,
    ) -> NeuroSymbolicState:
        """
        Convert MCTS state to NeuroSymbolicState.

        Args:
            mcts_state: MCTS state object (MCTSState)
            action_history: List of actions taken to reach this state

        Returns:
            NeuroSymbolicState representation
        """
        # Extract state ID
        state_id = getattr(mcts_state, "state_id", str(hash(mcts_state)))

        # Extract features as facts
        facts: set[Fact] = set()
        features = getattr(mcts_state, "features", {})

        for key, value in features.items():
            facts.add(
                Fact(
                    name="has_feature",
                    arguments=(key, value),
                    fact_type=SymbolicFactType.ATTRIBUTE,
                )
            )

        # Add action history as facts
        action_history = action_history or []
        for i, action in enumerate(action_history):
            facts.add(
                Fact(
                    name="action_at",
                    arguments=(i, action),
                    fact_type=SymbolicFactType.PREDICATE,
                )
            )

        # Add depth fact
        facts.add(
            Fact(
                name="depth",
                arguments=(len(action_history),),
                fact_type=SymbolicFactType.ATTRIBUTE,
            )
        )

        return NeuroSymbolicState(
            state_id=state_id,
            facts=frozenset(facts),
            metadata={
                "action_history": action_history,
                "original_features": features,
            },
        )

    async def filter_valid_actions(
        self,
        mcts_state: Any,
        candidate_actions: list[str],
        action_history: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Filter candidate actions using symbolic constraints.

        Args:
            mcts_state: Current MCTS state
            candidate_actions: Actions to evaluate
            action_history: Previous actions

        Returns:
            List of (action, validity_score) tuples for valid actions
        """
        if not self.config.enable_constraint_pruning:
            return [(a, 1.0) for a in candidate_actions]

        self._expansions_checked += 1
        start_time = time.perf_counter()

        # Convert to neuro-symbolic state
        ns_state = self.convert_mcts_state(mcts_state, action_history)

        # Validate actions
        valid_actions = self.constraint_system.validate_expansion(
            ns_state,
            candidate_actions,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._constraint_check_time_ms += elapsed_ms

        # Track pruned actions
        pruned_count = len(candidate_actions) - len(valid_actions)
        self._actions_pruned += pruned_count

        if self.logger and pruned_count > 0:
            self.logger.debug(
                f"Constraint pruning: {pruned_count}/{len(candidate_actions)} actions pruned in {elapsed_ms:.2f}ms"
            )

        return valid_actions

    def get_symbolic_heuristic(
        self,
        mcts_state: Any,
        action_history: list[str] | None = None,
    ) -> float:
        """
        Get symbolic heuristic value for state.

        Used to bias rollout policy toward symbolically favorable states.

        Args:
            mcts_state: Current MCTS state
            action_history: Previous actions

        Returns:
            Heuristic value in [0, 1]
        """
        if not self.config.enable_symbolic_heuristics:
            return 0.5

        ns_state = self.convert_mcts_state(mcts_state, action_history)

        # Calculate heuristic based on:
        # 1. Constraint satisfaction ratio
        # 2. Goal-relevant facts present
        # 3. Action sequence quality

        satisfaction_score = self._compute_satisfaction_score(ns_state)
        progress_score = self._compute_progress_score(ns_state)

        return 0.6 * satisfaction_score + 0.4 * progress_score

    def _compute_satisfaction_score(self, state: NeuroSymbolicState) -> float:
        """Compute constraint satisfaction score."""
        is_valid, results = self.constraint_system.validator.validate(state)

        if is_valid:
            total_penalty = sum(r.penalty for r in results)
            return max(0.0, 1.0 - total_penalty)
        return 0.0

    def _compute_progress_score(self, state: NeuroSymbolicState) -> float:
        """Compute goal progress score based on facts."""
        # Simple heuristic: more facts = more progress
        # In practice, this should be domain-specific
        num_facts = len(state.facts)
        max_expected_facts = 20  # Configurable
        return min(num_facts / max_expected_facts, 1.0)

    def compute_hybrid_value(
        self,
        neural_value: float,
        mcts_state: Any,
        action_history: list[str] | None = None,
    ) -> float:
        """
        Compute hybrid value combining neural and symbolic assessments.

        Args:
            neural_value: Value from neural network
            mcts_state: Current MCTS state
            action_history: Previous actions

        Returns:
            Combined value
        """
        symbolic_value = self.get_symbolic_heuristic(mcts_state, action_history)

        return self.config.neural_weight * neural_value + self.config.symbolic_weight * symbolic_value

    def get_statistics(self) -> dict[str, Any]:
        """Get integration statistics."""
        return {
            "expansions_checked": self._expansions_checked,
            "actions_pruned": self._actions_pruned,
            "prune_rate": (self._actions_pruned / max(self._expansions_checked, 1)),
            "avg_constraint_check_time_ms": (self._constraint_check_time_ms / max(self._expansions_checked, 1)),
            "constraint_stats": self.constraint_system.get_statistics(),
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._expansions_checked = 0
        self._actions_pruned = 0
        self._constraint_check_time_ms = 0.0


@dataclass
class SymbolicAgentNodeConfig:
    """Configuration for symbolic agent graph node."""

    enabled: bool = True
    priority: int = 0  # Higher = checked first for routing
    keywords: list[str] = field(default_factory=lambda: ["prove", "logic", "rule", "constraint", "why"])
    min_confidence_for_routing: float = 0.5


class SymbolicAgentGraphExtension:
    """
    Extension for GraphBuilder to include symbolic reasoning agent.

    Provides:
    - Symbolic agent node handler
    - Routing logic for symbolic queries
    - Result aggregation with symbolic outputs
    """

    def __init__(
        self,
        reasoning_agent: SymbolicReasoningAgent,
        config: SymbolicAgentNodeConfig | None = None,
        logger: Any | None = None,
    ):
        self.reasoning_agent = reasoning_agent
        self.config = config or SymbolicAgentNodeConfig()
        self.logger = logger

    def should_route_to_symbolic(
        self,
        query: str,
        state: dict[str, Any],
    ) -> bool:
        """
        Determine if query should be routed to symbolic agent.

        Args:
            query: User query
            state: Current agent state

        Returns:
            True if symbolic agent should handle this query
        """
        if not self.config.enabled:
            return False

        # Check if symbolic agent already ran
        if "symbolic_results" in state:
            return False

        # Keyword matching
        query_lower = query.lower()
        for keyword in self.config.keywords:
            if keyword in query_lower:
                return True

        # Check for formal query patterns (Prolog-style)
        import re

        return bool(re.search(r"\w+\([^)]+\)\??", query))

    async def handle_symbolic_node(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle symbolic agent node in the graph.

        Args:
            state: Current agent state (AgentState)

        Returns:
            State updates
        """
        if self.logger:
            self.logger.info("Executing symbolic reasoning agent")

        query = state.get("query", "")
        rag_context = state.get("rag_context", "")

        result = await self.reasoning_agent.process(
            query=query,
            rag_context=rag_context,
        )

        return {
            "symbolic_results": {
                "response": result["response"],
                "metadata": result["metadata"],
            },
            "agent_outputs": [
                {
                    "agent": "symbolic",
                    "response": result["response"],
                    "confidence": result["metadata"].get("confidence", 0.0),
                    "proof_found": result["metadata"].get("proof_found", False),
                }
            ],
        }

    def get_routing_key(self) -> str:
        """Get the routing key for this agent."""
        return "symbolic"

    def get_node_name(self) -> str:
        """Get the node name for the graph."""
        return "symbolic_agent"


class HybridConfidenceAggregator:
    """
    Aggregates confidence scores from neural and symbolic agents.

    Uses weighted combination with consistency checks.
    """

    def __init__(
        self,
        neural_weight: float = 0.5,
        symbolic_weight: float = 0.5,
        consistency_bonus: float = 0.1,
    ):
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        self.consistency_bonus = consistency_bonus

        # Normalize weights
        total = self.neural_weight + self.symbolic_weight
        self.neural_weight /= total
        self.symbolic_weight /= total

    def aggregate(
        self,
        agent_outputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Aggregate agent outputs with hybrid confidence scoring.

        Args:
            agent_outputs: List of agent output dictionaries

        Returns:
            Aggregation result with combined confidence
        """
        if not agent_outputs:
            return {
                "combined_confidence": 0.0,
                "consistency_score": 0.0,
                "agent_contributions": {},
            }

        # Separate neural and symbolic outputs
        neural_outputs = []
        symbolic_outputs = []

        for output in agent_outputs:
            agent = output.get("agent", "unknown")
            if agent == "symbolic":
                symbolic_outputs.append(output)
            else:
                neural_outputs.append(output)

        # Calculate weighted confidences
        neural_conf = 0.0
        if neural_outputs:
            neural_conf = sum(o.get("confidence", 0.0) for o in neural_outputs)
            neural_conf /= len(neural_outputs)

        symbolic_conf = 0.0
        if symbolic_outputs:
            symbolic_conf = sum(o.get("confidence", 0.0) for o in symbolic_outputs)
            symbolic_conf /= len(symbolic_outputs)

        # Check consistency between neural and symbolic
        consistency_score = 1.0 - abs(neural_conf - symbolic_conf)

        # Combine with consistency bonus
        combined_confidence = (
            self.neural_weight * neural_conf
            + self.symbolic_weight * symbolic_conf
            + self.consistency_bonus * consistency_score
        )

        # Cap at 1.0
        combined_confidence = min(combined_confidence, 1.0)

        return {
            "combined_confidence": combined_confidence,
            "consistency_score": consistency_score,
            "neural_confidence": neural_conf,
            "symbolic_confidence": symbolic_conf,
            "agent_contributions": {
                "neural": self.neural_weight,
                "symbolic": self.symbolic_weight,
            },
        }


def create_neuro_symbolic_extension(
    config: NeuroSymbolicConfig,
    graph_builder: Any,
    logger: Any | None = None,
) -> tuple[SymbolicReasoningAgent, NeuroSymbolicMCTSIntegration, SymbolicAgentGraphExtension]:
    """
    Factory function to create and integrate neuro-symbolic components.

    Args:
        config: Neuro-symbolic configuration
        graph_builder: Existing GraphBuilder instance
        logger: Optional logger

    Returns:
        Tuple of (reasoning_agent, mcts_integration, graph_extension)
    """
    # Create reasoning agent
    reasoning_agent = SymbolicReasoningAgent(
        config=config,
        neural_fallback=None,  # Will be set up separately
        logger=logger,
    )

    # Create MCTS integration
    mcts_config = NeuroSymbolicMCTSConfig(
        neural_weight=config.agent.neural_confidence_weight,
        symbolic_weight=config.agent.symbolic_confidence_weight,
    )
    mcts_integration = NeuroSymbolicMCTSIntegration(
        config=mcts_config,
        reasoning_agent=reasoning_agent,
        logger=logger,
    )

    # Create graph extension
    graph_extension = SymbolicAgentGraphExtension(
        reasoning_agent=reasoning_agent,
        logger=logger,
    )

    return reasoning_agent, mcts_integration, graph_extension


def extend_graph_builder(
    graph_builder: Any,
    extension: SymbolicAgentGraphExtension,
) -> None:
    """
    Extend an existing GraphBuilder with symbolic agent support.

    This modifies the graph builder in place to add the symbolic agent node
    and update routing logic.

    Args:
        graph_builder: GraphBuilder instance to extend
        extension: SymbolicAgentGraphExtension instance
    """
    # Store extension reference
    graph_builder._symbolic_extension = extension

    # Store original route decision method
    original_rule_based_route = graph_builder._rule_based_route_decision

    def extended_route_decision(state: dict[str, Any]) -> str:
        """Extended routing that includes symbolic agent."""
        # Check if symbolic agent should handle this
        if extension.should_route_to_symbolic(state.get("query", ""), state):
            return extension.get_routing_key()

        # Fall back to original routing
        return str(original_rule_based_route(state))

    # Replace routing method
    graph_builder._rule_based_route_decision = extended_route_decision
