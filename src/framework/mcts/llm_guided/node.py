"""
Enhanced MCTS Node for LLM-Guided Search.

Provides:
- LLMGuidedMCTSNode: MCTS node with data collection hooks
- NodeState: State representation for code generation
- Training data collection fields for neural network distillation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import time
from typing import Any

import numpy as np

from . import constants as C


class NodeStatus(Enum):
    """Status of a node in the search tree."""

    UNEXPANDED = "unexpanded"
    EXPANDED = "expanded"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILURE = "terminal_failure"


@dataclass
class NodeState:
    """
    State representation for code generation MCTS.

    Hashable state that captures the current code and context.
    """

    code: str
    """Current code state."""

    problem: str
    """Original problem description."""

    test_cases: list[str] = field(default_factory=list)
    """Test cases for the problem."""

    errors: list[str] = field(default_factory=list)
    """Errors from previous execution attempts."""

    attempt_history: list[str] = field(default_factory=list)
    """History of previous code attempts."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_hash_key(self) -> str:
        """Generate a hashable key for this state."""
        # Combine code and problem for unique identification
        combined = f"{self.code}:{self.problem}:{len(self.attempt_history)}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def with_new_code(self, new_code: str, errors: list[str] | None = None) -> NodeState:
        """Create new state with updated code."""
        new_history = self.attempt_history.copy()
        if self.code:  # Don't add empty initial code
            new_history.append(self.code)

        return NodeState(
            code=new_code,
            problem=self.problem,
            test_cases=self.test_cases.copy(),
            errors=errors or [],
            attempt_history=new_history,
            metadata=self.metadata.copy(),
        )

    def __repr__(self) -> str:
        code_preview = self.code[:50] + "..." if len(self.code) > 50 else self.code
        return f"NodeState(code={code_preview!r}, attempts={len(self.attempt_history)})"


@dataclass
class LLMGuidedMCTSNode:
    """
    MCTS Node with data collection hooks for neural network training.

    This node extends standard MCTS with fields needed for collecting
    training data to distill LLM behavior into neural networks.

    Standard MCTS fields:
    - state: The code state this node represents
    - parent: Parent node (None for root)
    - action: Action (code variant) taken to reach this node
    - children: List of child nodes
    - visits: Number of times this node has been visited
    - value_sum: Total accumulated value from simulations

    Training data collection fields:
    - llm_action_probs: LLM's action probability distribution
    - llm_value_estimate: LLM's value estimate for this state
    - mcts_action_probs: Improved policy from MCTS visit counts
    - episode_id: Unique identifier for the episode
    - timestamp: When this node was created
    - test_results: Results from code execution
    """

    # Standard MCTS fields
    state: NodeState
    """The code state this node represents."""

    parent: LLMGuidedMCTSNode | None = None
    """Parent node (None for root)."""

    action: str | None = None
    """Action (variant identifier) taken to reach this node."""

    children: list[LLMGuidedMCTSNode] = field(default_factory=list)
    """Child nodes."""

    visits: int = 0
    """Number of times this node has been visited."""

    value_sum: float = 0.0
    """Total accumulated value from simulations."""

    depth: int = 0
    """Depth in the tree (0 for root)."""

    status: NodeStatus = NodeStatus.UNEXPANDED
    """Current status of the node."""

    # Training data collection fields
    llm_action_probs: dict[str, float] = field(default_factory=dict)
    """LLM's action probability distribution (teacher labels)."""

    llm_value_estimate: float = 0.0
    """LLM's value estimate for this state."""

    mcts_action_probs: dict[str, float] = field(default_factory=dict)
    """Improved policy from MCTS visit counts."""

    episode_id: str = ""
    """Unique identifier for the episode."""

    timestamp: float = field(default_factory=time.time)
    """When this node was created."""

    test_results: dict[str, Any] | None = None
    """Results from code execution."""

    # Code generation specific
    generated_variants: list[dict[str, Any]] = field(default_factory=list)
    """Generated code variants with confidence scores."""

    reflection: str = ""
    """LLM's reflection on this code."""

    # Random number generator
    _rng: np.random.Generator | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize depth and RNG."""
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        if self._rng is None:
            self._rng = np.random.default_rng()

    @property
    def q_value(self) -> float:
        """Average value (exploitation term)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.status in (NodeStatus.TERMINAL_SUCCESS, NodeStatus.TERMINAL_FAILURE)

    @property
    def is_solution(self) -> bool:
        """Check if this node represents a successful solution."""
        return self.status == NodeStatus.TERMINAL_SUCCESS

    @property
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return self.status != NodeStatus.UNEXPANDED

    def ucb1(self, c: float = C.UCB1_EXPLORATION_CONSTANT) -> float:
        """
        Calculate UCB1 score for node selection.

        Args:
            c: Exploration constant

        Returns:
            UCB1 score
        """
        if self.visits == 0:
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.q_value

        exploitation = self.q_value
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self, c: float = C.UCB1_EXPLORATION_CONSTANT) -> LLMGuidedMCTSNode | None:
        """
        Select best child using UCB1.

        Args:
            c: Exploration constant

        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None

        best_child = None
        best_score = float("-inf")

        for child in self.children:
            score = child.ucb1(c)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def add_child(
        self,
        state: NodeState,
        action: str,
        llm_action_probs: dict[str, float] | None = None,
        episode_id: str = "",
    ) -> LLMGuidedMCTSNode:
        """
        Add a child node.

        Args:
            state: Child state
            action: Action taken to reach child
            llm_action_probs: LLM's action distribution
            episode_id: Episode identifier

        Returns:
            Newly created child node
        """
        child = LLMGuidedMCTSNode(
            state=state,
            parent=self,
            action=action,
            llm_action_probs=llm_action_probs or {},
            episode_id=episode_id or self.episode_id,
            _rng=self._rng,
        )
        self.children.append(child)
        return child

    def backpropagate(self, reward: float) -> None:
        """
        Propagate reward up to root.

        Args:
            reward: Reward value to propagate
        """
        node: LLMGuidedMCTSNode | None = self
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent

    def compute_mcts_policy(self) -> dict[str, float]:
        """
        Compute improved policy from MCTS visit counts.

        This is used as the training target for the policy network,
        as it incorporates the search results.

        Returns:
            Action probability distribution based on visit counts
        """
        if not self.children:
            return {}

        total_visits = sum(child.visits for child in self.children)
        if total_visits == 0:
            return {}

        policy = {}
        for child in self.children:
            if child.action:
                policy[child.action] = child.visits / total_visits

        self.mcts_action_probs = policy
        return policy

    def get_best_child(self) -> LLMGuidedMCTSNode | None:
        """
        Get child with highest visit count (most robust selection).

        Returns:
            Best child node or None
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)

    def get_path_to_root(self) -> list[LLMGuidedMCTSNode]:
        """
        Get path from this node to root.

        Returns:
            List of nodes from root to this node
        """
        path = []
        node: LLMGuidedMCTSNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def to_training_dict(self) -> dict[str, Any]:
        """
        Convert node to dictionary for training data export.

        Returns:
            Dictionary with all training-relevant fields
        """
        return {
            "state": {
                "code": self.state.code,
                "problem": self.state.problem,
                "test_cases": self.state.test_cases,
                "errors": self.state.errors,
                "attempt_history": self.state.attempt_history,
            },
            "action": self.action,
            "depth": self.depth,
            "visits": self.visits,
            "q_value": self.q_value,
            "llm_action_probs": self.llm_action_probs,
            "llm_value_estimate": self.llm_value_estimate,
            "mcts_action_probs": self.mcts_action_probs,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "test_results": self.test_results,
            "is_terminal": self.is_terminal,
            "is_solution": self.is_solution,
        }

    def __repr__(self) -> str:
        return (
            f"LLMGuidedMCTSNode(depth={self.depth}, "
            f"visits={self.visits}, q={self.q_value:.3f}, "
            f"children={len(self.children)}, status={self.status.value})"
        )


def create_root_node(
    problem: str,
    initial_code: str = "",
    test_cases: list[str] | None = None,
    episode_id: str = "",
    seed: int | None = None,
) -> LLMGuidedMCTSNode:
    """
    Create a root node for LLM-guided MCTS.

    Args:
        problem: Problem description
        initial_code: Initial code (empty for new problems)
        test_cases: Test cases for the problem
        episode_id: Episode identifier
        seed: Random seed for reproducibility

    Returns:
        Root node
    """
    state = NodeState(
        code=initial_code,
        problem=problem,
        test_cases=test_cases or [],
    )

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    return LLMGuidedMCTSNode(
        state=state,
        episode_id=episode_id,
        _rng=rng,
    )
