"""
Enhanced MCTS Node with Reasoning Traces and PRM Scores.

Extends the base MCTSNode with:
- Reasoning trace storage for each node
- PRM score tracking
- Extended thinking results
- Dual-agent architecture support (Reasoner + Actor)

This enables MuZero-inspired architecture for LLM reasoning:
- Policy model proposes actions (expansion)
- PRM evaluates states (value estimation)
- MCTS orchestrates search (planning)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from .core import MCTSNode, MCTSState
from .extended_thinking import ThinkingResult
from .process_reward_model import PRMScore, ReasoningStep, ReasoningTrajectory


@dataclass
class ReasoningMetadata:
    """
    Metadata for reasoning at an MCTS node.

    Stores all reasoning-related information including
    thinking traces, PRM scores, and agent attributions.
    """

    # Reasoning traces
    thinking_trace: str = ""
    """Extended thinking trace from evaluation"""

    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    """Individual reasoning steps at this node"""

    # PRM scores
    prm_score: float | None = None
    """Process Reward Model score for this node"""

    prm_scores_history: list[PRMScore] = field(default_factory=list)
    """History of PRM scores (if multiple evaluations)"""

    cumulative_prm_score: float = 0.0
    """Cumulative PRM score along path from root"""

    # Thinking results
    thinking_result: ThinkingResult | None = None
    """Full extended thinking evaluation result"""

    thinking_tokens_used: int = 0
    """Total thinking tokens used at this node"""

    # Agent attribution
    source_agent: str = "unknown"
    """Which agent generated this node (reasoner, actor, hybrid)"""

    agent_confidence: float = 0.0
    """Agent's confidence in the action leading to this node"""

    # Verification
    verified: bool = False
    """Whether this node has been verified"""

    verification_result: dict[str, Any] = field(default_factory=dict)
    """Results from verification (if performed)"""


class ReasoningMCTSNode(MCTSNode):
    """
    Enhanced MCTS node with reasoning capabilities.

    Extends MCTSNode with:
    - Reasoning trace storage
    - PRM score tracking
    - Extended thinking integration
    - Dual-agent support
    """

    def __init__(
        self,
        state: MCTSState,
        parent: ReasoningMCTSNode | None = None,
        action: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(state, parent, action, rng)

        # Reasoning metadata
        self.reasoning = ReasoningMetadata()

        # Enhanced tracking
        self._prm_weighted_value_sum: float = 0.0
        self._prm_visits: int = 0

    @property
    def prm_value(self) -> float:
        """Average PRM-weighted value of this node."""
        if self._prm_visits == 0:
            return 0.0
        return self._prm_weighted_value_sum / self._prm_visits

    def set_thinking_result(self, result: ThinkingResult) -> None:
        """
        Set the extended thinking result for this node.

        Args:
            result: ThinkingResult from evaluation
        """
        self.reasoning.thinking_result = result
        self.reasoning.thinking_trace = result.thinking_trace
        self.reasoning.thinking_tokens_used = result.tokens_used

    def set_prm_score(self, score: PRMScore) -> None:
        """
        Set the PRM score for this node.

        Args:
            score: PRMScore from process reward model
        """
        self.reasoning.prm_score = score.step_score
        self.reasoning.prm_scores_history.append(score)

        # Update cumulative score
        if self.parent and isinstance(self.parent, ReasoningMCTSNode):
            self.reasoning.cumulative_prm_score = (
                self.parent.reasoning.cumulative_prm_score * 0.95 + score.step_score * 0.05
            )
        else:
            self.reasoning.cumulative_prm_score = score.step_score

    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """
        Add a reasoning step to this node.

        Args:
            step: ReasoningStep to add
        """
        step.step_index = len(self.reasoning.reasoning_steps)
        self.reasoning.reasoning_steps.append(step)

    def update_with_prm(self, value: float, prm_weight: float = 0.5) -> None:
        """
        Update node with PRM-weighted value.

        Args:
            value: Base value from simulation
            prm_weight: Weight for PRM score in combined value
        """
        if self.reasoning.prm_score is not None:
            combined_value = (
                (1 - prm_weight) * value + prm_weight * self.reasoning.prm_score
            )
        else:
            combined_value = value

        self._prm_weighted_value_sum += combined_value
        self._prm_visits += 1

        # Also update base statistics
        self.visits += 1
        self.value_sum += value

    def get_trajectory(self) -> ReasoningTrajectory:
        """
        Build a reasoning trajectory from root to this node.

        Returns:
            ReasoningTrajectory containing all steps along the path
        """
        trajectory = ReasoningTrajectory()

        # Collect path from root
        path = []
        node: ReasoningMCTSNode | MCTSNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent

        # Reverse to get root-to-leaf order
        path.reverse()

        # Build trajectory from path
        for node in path:
            if isinstance(node, ReasoningMCTSNode):
                for step in node.reasoning.reasoning_steps:
                    trajectory.add_step(step)

        return trajectory

    def select_child_with_prm(
        self,
        exploration_weight: float = 1.414,
        prm_weight: float = 0.3,
    ) -> ReasoningMCTSNode:
        """
        Select best child using PRM-enhanced UCB.

        Args:
            exploration_weight: UCB exploration constant
            prm_weight: Weight for PRM scores in selection

        Returns:
            Best child node
        """
        if not self.children:
            raise ValueError("No children to select from")

        best_child = None
        best_score = float("-inf")

        for child in self.children:
            if not isinstance(child, ReasoningMCTSNode):
                # Fall back to base UCB for non-reasoning nodes
                from .policies import ucb1

                score = ucb1(
                    value_sum=child.value_sum,
                    visits=child.visits,
                    parent_visits=self.visits,
                    c=exploration_weight,
                )
            else:
                score = self._compute_prm_enhanced_ucb(
                    child, exploration_weight, prm_weight
                )

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _compute_prm_enhanced_ucb(
        self,
        child: ReasoningMCTSNode,
        c: float,
        prm_weight: float,
    ) -> float:
        """Compute PRM-enhanced UCB score."""
        import math

        if child.visits == 0:
            return float("inf")

        # Base UCB
        exploitation = child.value / child.visits if child.visits > 0 else 0
        exploration = c * math.sqrt(math.log(self.visits) / child.visits)
        base_ucb = exploitation + exploration

        # PRM component
        if child.reasoning.prm_score is not None:
            prm_score = child.reasoning.prm_score
        else:
            prm_score = 0.5  # Default neutral

        # Combine
        return (1 - prm_weight) * base_ucb + prm_weight * prm_score

    def add_reasoning_child(
        self,
        action: str,
        child_state: MCTSState,
        reasoning_step: ReasoningStep | None = None,
    ) -> ReasoningMCTSNode:
        """
        Add a child node with reasoning metadata.

        Args:
            action: Action taken to reach child
            child_state: State of child node
            reasoning_step: Optional reasoning step for the action

        Returns:
            New ReasoningMCTSNode
        """
        child = ReasoningMCTSNode(
            state=child_state,
            parent=self,
            action=action,
            rng=self._rng,
        )

        if reasoning_step:
            child.add_reasoning_step(reasoning_step)

        self.children.append(child)
        self.expanded_actions.add(action)

        return child

    def __repr__(self) -> str:
        prm_str = f", prm={self.reasoning.prm_score:.3f}" if self.reasoning.prm_score else ""
        return (
            f"ReasoningMCTSNode(state={self.state.state_id}, "
            f"visits={self.visits}, value={self.value:.3f}"
            f"{prm_str}, children={len(self.children)})"
        )


# ============================================================================
# Dual-Agent Architecture
# ============================================================================


@dataclass
class AgentAction:
    """
    An action proposed by an agent.

    Contains the action itself plus metadata about how it was generated.
    """

    action: str
    """The action to take"""

    confidence: float
    """Agent's confidence in this action"""

    reasoning: str = ""
    """Reasoning behind the action"""

    thinking_trace: str = ""
    """Extended thinking trace (if applicable)"""

    source_agent: str = "unknown"
    """Which agent proposed this action"""

    estimated_value: float = 0.0
    """Agent's estimate of the value of this action"""


class ReasonerAgent:
    """
    Reasoning-only agent for strategy and analysis.

    Uses extended thinking with no tool access.
    Focuses on proposing strategies and analyzing states.
    """

    def __init__(
        self,
        model_fn: Callable[[str, int], Any],
        default_thinking_tokens: int = 16384,
    ):
        """
        Initialize reasoner agent.

        Args:
            model_fn: Function to call model with (prompt, tokens) -> response
            default_thinking_tokens: Default thinking budget
        """
        self.model_fn = model_fn
        self.default_thinking_tokens = default_thinking_tokens

        # Statistics
        self.total_proposals = 0
        self.total_tokens_used = 0

    async def propose_strategies(
        self,
        state: MCTSState | ReasoningMCTSNode,
        context: str,
        n_strategies: int = 5,
        thinking_tokens: int | None = None,
    ) -> list[AgentAction]:
        """
        Propose multiple strategies for the current state.

        Args:
            state: Current state or node
            context: Additional context (query, history, etc.)
            n_strategies: Number of strategies to propose
            thinking_tokens: Thinking budget (uses default if None)

        Returns:
            List of proposed AgentActions
        """
        tokens = thinking_tokens or self.default_thinking_tokens

        prompt = self._build_strategy_prompt(state, context, n_strategies)

        response = await self.model_fn(prompt, tokens)

        strategies = self._parse_strategies(response)

        self.total_proposals += len(strategies)
        self.total_tokens_used += tokens

        return strategies

    async def evaluate_state(
        self,
        state: MCTSState | ReasoningMCTSNode,
        context: str,
        thinking_tokens: int | None = None,
    ) -> ThinkingResult:
        """
        Evaluate a state through extended thinking.

        Args:
            state: State to evaluate
            context: Evaluation context
            thinking_tokens: Thinking budget

        Returns:
            ThinkingResult with evaluation
        """
        tokens = thinking_tokens or self.default_thinking_tokens

        prompt = self._build_evaluation_prompt(state, context)

        response = await self.model_fn(prompt, tokens)

        return self._parse_evaluation(response)

    def _build_strategy_prompt(
        self,
        state: MCTSState | ReasoningMCTSNode,
        context: str,
        n_strategies: int,
    ) -> str:
        """Build prompt for strategy generation."""
        state_str = self._format_state(state)

        return f"""You are a strategic reasoning agent. Analyze the current state and propose {n_strategies} distinct strategies.

Context:
{context}

Current State:
{state_str}

Think carefully about:
1. What are the key factors to consider?
2. What are the possible approaches?
3. What are the trade-offs of each approach?

Propose {n_strategies} strategies, each with:
- STRATEGY: [concise description]
- REASONING: [why this strategy could work]
- CONFIDENCE: [0.0-1.0]
- ESTIMATED_VALUE: [0.0-1.0]

Think through this thoroughly before proposing strategies."""

    def _build_evaluation_prompt(
        self,
        state: MCTSState | ReasoningMCTSNode,
        context: str,
    ) -> str:
        """Build prompt for state evaluation."""
        state_str = self._format_state(state)

        return f"""Evaluate the quality of the following state in the context of tree search.

Context:
{context}

State:
{state_str}

Analyze:
1. Is this state on a promising path?
2. What is the likelihood of reaching a good solution from here?
3. Are there any issues or concerns with this state?

Provide:
SCORE: [0.0-1.0]
ANALYSIS: [detailed analysis]
IS_TERMINAL: [yes/no]
CONFIDENCE: [0.0-1.0]"""

    def _format_state(self, state: MCTSState | ReasoningMCTSNode) -> str:
        """Format state for prompt."""
        if isinstance(state, ReasoningMCTSNode):
            return f"Node: {state.state.state_id}\nAction: {state.action}\nValue: {state.value:.3f}\nVisits: {state.visits}"
        else:
            return f"State ID: {state.state_id}\nFeatures: {state.features}"

    def _parse_strategies(self, response: Any) -> list[AgentAction]:
        """Parse strategies from model response."""
        text = self._extract_text(response)
        strategies = []

        # Split by "STRATEGY:" markers
        parts = text.split("STRATEGY:")
        for part in parts[1:]:  # Skip first empty part
            strategy = self._parse_single_strategy(part)
            if strategy:
                strategies.append(strategy)

        return strategies

    def _parse_single_strategy(self, text: str) -> AgentAction | None:
        """Parse a single strategy from text."""
        lines = text.strip().split("\n")
        if not lines:
            return None

        strategy = lines[0].strip()
        reasoning = ""
        confidence = 0.5
        estimated_value = 0.5

        for line in lines[1:]:
            if line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("ESTIMATED_VALUE:"):
                try:
                    estimated_value = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        return AgentAction(
            action=strategy,
            confidence=confidence,
            reasoning=reasoning,
            source_agent="reasoner",
            estimated_value=estimated_value,
        )

    def _parse_evaluation(self, response: Any) -> ThinkingResult:
        """Parse evaluation from model response."""
        text = self._extract_text(response)

        score = 0.5
        analysis = ""
        is_terminal = False
        confidence = 0.5

        for line in text.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("ANALYSIS:"):
                analysis = line.split(":", 1)[1].strip()
            elif line.startswith("IS_TERMINAL:"):
                is_terminal = "yes" in line.lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        return ThinkingResult(
            score=score,
            thinking_trace="",  # Would be extracted if available
            analysis=analysis,
            is_terminal=is_terminal,
            confidence=confidence,
        )

    def _extract_text(self, response: Any) -> str:
        """Extract text from model response."""
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "content"):
            # Handle anthropic-style response
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    return block.text
        return str(response)


class ActorAgent:
    """
    Action execution agent with tool access.

    Fast model for routine execution following reasoner's strategies.
    """

    def __init__(
        self,
        model_fn: Callable[[str], Any],
        tools: list[Any] | None = None,
    ):
        """
        Initialize actor agent.

        Args:
            model_fn: Function to call model with prompt -> response
            tools: Available tools for execution
        """
        self.model_fn = model_fn
        self.tools = tools or []

        # Statistics
        self.total_executions = 0
        self.successful_executions = 0

    async def execute_strategy(
        self,
        strategy: AgentAction,
        state: MCTSState | ReasoningMCTSNode,
        context: str,
    ) -> tuple[MCTSState, dict[str, Any]]:
        """
        Execute a strategy to produce a new state.

        Args:
            strategy: Strategy from reasoner
            state: Current state
            context: Execution context

        Returns:
            Tuple of (new_state, execution_metadata)
        """
        self.total_executions += 1

        prompt = self._build_execution_prompt(strategy, state, context)

        response = await self.model_fn(prompt)

        new_state, metadata = self._process_execution(response, state, strategy)

        if metadata.get("success", False):
            self.successful_executions += 1

        return new_state, metadata

    def _build_execution_prompt(
        self,
        strategy: AgentAction,
        state: MCTSState | ReasoningMCTSNode,
        context: str,
    ) -> str:
        """Build prompt for strategy execution."""
        state_id = state.state.state_id if isinstance(state, ReasoningMCTSNode) else state.state_id

        tools_str = "\n".join([f"- {t}" for t in self.tools]) if self.tools else "(None)"

        return f"""Execute the following strategy efficiently.

Strategy: {strategy.action}
Reasoning: {strategy.reasoning}

Current State: {state_id}
Context: {context}

Available Tools:
{tools_str}

Execute the strategy and report the result."""

    def _process_execution(
        self,
        response: Any,
        original_state: MCTSState | ReasoningMCTSNode,
        strategy: AgentAction,
    ) -> tuple[MCTSState, dict[str, Any]]:
        """Process execution response and create new state."""
        text = self._extract_text(response)

        # Create new state based on execution
        if isinstance(original_state, ReasoningMCTSNode):
            base_state = original_state.state
        else:
            base_state = original_state

        new_state = MCTSState(
            state_id=f"{base_state.state_id}_{strategy.action[:10]}",
            features={
                **base_state.features,
                "last_action": strategy.action,
                "execution_result": text[:200],
            },
        )

        metadata = {
            "success": True,  # Would be determined from actual execution
            "execution_output": text,
            "strategy": strategy.action,
        }

        return new_state, metadata

    def _extract_text(self, response: Any) -> str:
        """Extract text from model response."""
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return response.text
        return str(response)


class DualAgentMCTSController:
    """
    Controller for dual-agent MCTS architecture.

    Coordinates between reasoner (strategy proposal) and actor (execution)
    agents for more efficient search.
    """

    def __init__(
        self,
        reasoner: ReasonerAgent,
        actor: ActorAgent,
        prm: Any | None = None,
    ):
        """
        Initialize dual-agent controller.

        Args:
            reasoner: Strategy-proposing agent
            actor: Strategy-executing agent
            prm: Optional Process Reward Model
        """
        self.reasoner = reasoner
        self.actor = actor
        self.prm = prm

    async def expand_with_reasoning(
        self,
        node: ReasoningMCTSNode,
        context: str,
        n_strategies: int = 5,
    ) -> list[ReasoningMCTSNode]:
        """
        Expand node using dual-agent architecture.

        Args:
            node: Node to expand
            context: Expansion context
            n_strategies: Number of strategies to propose

        Returns:
            List of new child nodes
        """
        # Reasoner proposes strategies
        strategies = await self.reasoner.propose_strategies(
            node, context, n_strategies
        )

        children = []
        for strategy in strategies:
            # Actor executes strategy
            new_state, metadata = await self.actor.execute_strategy(
                strategy, node, context
            )

            # Create reasoning step
            step = ReasoningStep(
                content=strategy.action,
                step_index=0,
                step_type="action",
                confidence=strategy.confidence,
                metadata={"reasoning": strategy.reasoning},
            )

            # Create child node
            child = node.add_reasoning_child(
                action=strategy.action,
                child_state=new_state,
                reasoning_step=step,
            )

            # Store agent attribution
            child.reasoning.source_agent = "dual"
            child.reasoning.agent_confidence = strategy.confidence

            # Score with PRM if available
            if self.prm:
                trajectory = child.get_trajectory()
                scores = await self.prm.score_trajectory(trajectory)
                if scores:
                    child.set_prm_score(scores[-1])

            children.append(child)

        return children

    async def evaluate_with_reasoning(
        self,
        node: ReasoningMCTSNode,
        context: str,
    ) -> ThinkingResult:
        """
        Evaluate node using reasoner's extended thinking.

        Args:
            node: Node to evaluate
            context: Evaluation context

        Returns:
            ThinkingResult from evaluation
        """
        result = await self.reasoner.evaluate_state(node, context)
        node.set_thinking_result(result)
        return result
