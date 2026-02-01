"""
Example: Reasoning-Enhanced MCTS with Modern AI Techniques

This example demonstrates how to use the integrated reasoning features:
1. Process Reward Models for step-level evaluation
2. Extended thinking with adaptive token budgets
3. Hybrid search combining parallel and serial scaling
4. Dual-agent architecture (Reasoner + Actor)

Requirements:
- numpy
- langgraph (optional, for full graph integration)
- anthropic (optional, for Claude extended thinking)
"""

import asyncio
from typing import Any

# Import reasoning components
from src.framework.mcts import (
    # Core MCTS
    MCTSEngine,
    MCTSState,
    # Process Reward Model
    LLMProcessRewardModel,
    HeuristicProcessRewardModel,
    EnsemblePRM,
    PRMMCTSIntegration,
    PRMEnhancedMCTSConfig,
    ReasoningStep,
    ReasoningTrajectory,
    # Extended Thinking
    ThinkingBudget,
    ThinkingMode,
    ThinkingResult,
    # Hybrid Search
    HybridMCTSSearch,
    HybridSearchConfig,
    # Reasoning Node
    ReasoningMCTSNode,
    AgentAction,
    ReasonerAgent,
    ActorAgent,
    DualAgentMCTSController,
)
from src.framework.mcts.policies import HybridRolloutPolicy


# ============================================================================
# Example 1: Process Reward Model for Step Evaluation
# ============================================================================


async def prm_example():
    """Demonstrate PRM for scoring reasoning steps."""
    print("\n" + "=" * 60)
    print("Example 1: Process Reward Model")
    print("=" * 60)

    # Create a mock LLM evaluate function
    async def mock_llm_evaluate(prompt: str) -> dict:
        """Simulated LLM evaluation."""
        # In production, this would call a real LLM
        return {
            "text": "SCORE: 0.85\nREASONING: Good logical progression"
        }

    # Create PRM
    llm_prm = LLMProcessRewardModel(
        evaluate_fn=mock_llm_evaluate,
        cache_size=100,
    )

    # Create a reasoning trajectory
    trajectory = ReasoningTrajectory(query="What is the derivative of x^2?")

    step1 = ReasoningStep(
        content="Apply the power rule: d/dx[x^n] = n*x^(n-1)",
        step_index=0,
        step_type="reasoning",
        confidence=0.9,
    )
    trajectory.add_step(step1)

    step2 = ReasoningStep(
        content="For x^2: n=2, so derivative = 2*x^(2-1) = 2x",
        step_index=1,
        step_type="reasoning",
        confidence=0.95,
    )
    trajectory.add_step(step2)

    trajectory.final_answer = "2x"

    # Score the trajectory
    print(f"\nQuery: {trajectory.query}")
    print(f"\nTrajectory ({len(trajectory.steps)} steps):")

    scores = await llm_prm.score_trajectory(trajectory)

    for step, score in zip(trajectory.steps, scores):
        print(f"  Step {step.step_index + 1}: {step.content[:50]}...")
        print(f"    PRM Score: {score.step_score:.2f}")

    # Compute trajectory value
    trajectory_value = await llm_prm.estimate_trajectory_value(
        trajectory, aggregation="product"
    )
    print(f"\nTrajectory Value (product): {trajectory_value:.4f}")


# ============================================================================
# Example 2: Heuristic PRM for Domain-Specific Evaluation
# ============================================================================


async def heuristic_prm_example():
    """Demonstrate heuristic-based PRM for fast evaluation."""
    print("\n" + "=" * 60)
    print("Example 2: Heuristic Process Reward Model")
    print("=" * 60)

    # Define domain-specific heuristics
    def length_heuristic(step: ReasoningStep, traj: ReasoningTrajectory) -> float:
        """Prefer steps with adequate detail (not too short, not too long)."""
        length = len(step.content)
        if length < 20:
            return 0.3  # Too brief
        elif length < 100:
            return 0.9  # Good length
        elif length < 200:
            return 0.7  # Getting verbose
        else:
            return 0.5  # Too long

    def confidence_heuristic(step: ReasoningStep, traj: ReasoningTrajectory) -> float:
        """Use model's self-reported confidence."""
        return step.confidence

    def progress_heuristic(step: ReasoningStep, traj: ReasoningTrajectory) -> float:
        """Reward steps that show progress toward solution."""
        keywords = ["therefore", "thus", "=", "result", "answer", "conclude"]
        content_lower = step.content.lower()
        matches = sum(1 for kw in keywords if kw in content_lower)
        return min(0.5 + matches * 0.15, 1.0)

    # Create heuristic PRM
    heuristic_prm = HeuristicProcessRewardModel(
        heuristics=[length_heuristic, confidence_heuristic, progress_heuristic],
        weights=[0.3, 0.4, 0.3],
    )

    # Test on a step
    step = ReasoningStep(
        content="Therefore, by substituting x=5 into the equation, we get the result 2*5 = 10",
        step_index=0,
        step_type="reasoning",
        confidence=0.85,
    )

    trajectory = ReasoningTrajectory(query="Calculate 2*5")
    trajectory.add_step(step)

    score = await heuristic_prm.score_step(step, trajectory)

    print(f"\nStep: {step.content}")
    print(f"Confidence: {step.confidence}")
    print(f"Heuristic PRM Score: {score.step_score:.2f}")


# ============================================================================
# Example 3: Extended Thinking Budget Computation
# ============================================================================


def extended_thinking_example():
    """Demonstrate adaptive thinking budget computation."""
    print("\n" + "=" * 60)
    print("Example 3: Extended Thinking Budget")
    print("=" * 60)

    budget = ThinkingBudget(
        min_tokens=1024,
        max_tokens=65536,
        default_tokens=8192,
        depth_multiplier=1.2,
        uncertainty_multiplier=1.5,
        critical_threshold=0.8,
    )

    # Test scenarios
    scenarios = [
        {"depth": 0, "visits": 0, "ucb_score": 0.5, "uncertainty": 0.3},
        {"depth": 3, "visits": 5, "ucb_score": 0.5, "uncertainty": 0.5},
        {"depth": 5, "visits": 2, "ucb_score": 0.85, "uncertainty": 0.7},
        {"depth": 8, "visits": 1, "ucb_score": 0.9, "uncertainty": 0.9},
    ]

    print("\nAdaptive Token Budget Allocation:")
    print("-" * 70)
    print(f"{'Depth':<6} {'Visits':<7} {'UCB':<6} {'Uncert.':<8} {'Tokens':<10} {'Mode':<12}")
    print("-" * 70)

    for s in scenarios:
        tokens = budget.compute_budget(
            depth=s["depth"],
            visits=s["visits"],
            ucb_score=s["ucb_score"],
            uncertainty=s["uncertainty"],
        )
        mode = budget.get_mode(tokens)

        print(f"{s['depth']:<6} {s['visits']:<7} {s['ucb_score']:<6} {s['uncertainty']:<8} "
              f"{tokens:<10} {mode.value:<12}")

    print("\nKey insight: Deeper nodes with high uncertainty get more thinking tokens")


# ============================================================================
# Example 4: Reasoning-Enhanced MCTS Node
# ============================================================================


async def reasoning_node_example():
    """Demonstrate ReasoningMCTSNode with PRM integration."""
    print("\n" + "=" * 60)
    print("Example 4: Reasoning-Enhanced MCTS Node")
    print("=" * 60)

    # Create root state
    root_state = MCTSState(
        state_id="problem_root",
        features={"query": "Solve: x^2 - 5x + 6 = 0"},
    )

    # Create reasoning node
    root = ReasoningMCTSNode(state=root_state)

    # Add initial reasoning step
    root.add_reasoning_step(ReasoningStep(
        content="Factor the quadratic: look for two numbers that multiply to 6 and add to -5",
        step_type="reasoning",
        confidence=0.9,
    ))

    # Create child node for "factor" strategy
    factor_state = MCTSState(
        state_id="factor_strategy",
        features={"approach": "factoring"},
    )

    factor_child = root.add_reasoning_child(
        action="factor",
        child_state=factor_state,
        reasoning_step=ReasoningStep(
            content="Numbers are -2 and -3: (x-2)(x-3) = 0",
            step_type="reasoning",
            confidence=0.95,
        ),
    )

    # Create child node for "quadratic formula" strategy
    formula_state = MCTSState(
        state_id="formula_strategy",
        features={"approach": "quadratic_formula"},
    )

    formula_child = root.add_reasoning_child(
        action="quadratic_formula",
        child_state=formula_state,
        reasoning_step=ReasoningStep(
            content="Apply x = (-b ± √(b²-4ac)) / 2a",
            step_type="reasoning",
            confidence=0.8,
        ),
    )

    # Simulate visits and values
    factor_child.visits = 15
    factor_child.value_sum = 13.5
    formula_child.visits = 10
    formula_child.value_sum = 7.0

    print(f"\nRoot: {root}")
    print("\nChildren:")
    for child in root.children:
        print(f"  {child}")
        print(f"    Reasoning: {child.reasoning.reasoning_steps[0].content[:60]}...")

    # Get trajectory from child
    trajectory = factor_child.get_trajectory()
    print(f"\nTrajectory from factor_child:")
    print(trajectory.to_text())


# ============================================================================
# Example 5: Dual-Agent Architecture
# ============================================================================


async def dual_agent_example():
    """Demonstrate Reasoner + Actor dual-agent architecture."""
    print("\n" + "=" * 60)
    print("Example 5: Dual-Agent Architecture")
    print("=" * 60)

    # Mock model functions
    async def reasoner_model(prompt: str, tokens: int) -> str:
        """Simulated reasoner with extended thinking."""
        return """
STRATEGY: Decompose the problem
REASONING: Breaking complex problems into smaller parts makes them tractable
CONFIDENCE: 0.85
ESTIMATED_VALUE: 0.8

STRATEGY: Try direct calculation
REASONING: Sometimes the direct approach is most efficient
CONFIDENCE: 0.7
ESTIMATED_VALUE: 0.65

STRATEGY: Search for similar problems
REASONING: Finding analogous problems can provide insight
CONFIDENCE: 0.6
ESTIMATED_VALUE: 0.5
"""

    async def actor_model(prompt: str) -> str:
        """Simulated fast actor."""
        return "Executed strategy successfully with result: [solution]"

    # Create agents
    reasoner = ReasonerAgent(
        model_fn=reasoner_model,
        default_thinking_tokens=16384,
    )

    actor = ActorAgent(
        model_fn=actor_model,
        tools=["calculator", "search", "code_executor"],
    )

    # Propose strategies
    state = MCTSState(state_id="test", features={"query": "Optimize algorithm"})
    strategies = await reasoner.propose_strategies(
        state=state,
        context="Performance optimization problem",
        n_strategies=3,
    )

    print("\nReasonser proposed strategies:")
    for i, strategy in enumerate(strategies):
        print(f"  {i+1}. {strategy.action}")
        print(f"     Reasoning: {strategy.reasoning}")
        print(f"     Confidence: {strategy.confidence:.2f}")

    # Execute best strategy with actor
    if strategies:
        best = max(strategies, key=lambda s: s.confidence)
        new_state, metadata = await actor.execute_strategy(
            strategy=best,
            state=state,
            context="Execute the chosen strategy",
        )
        print(f"\nActor executed: {best.action}")
        print(f"New state: {new_state.state_id}")


# ============================================================================
# Example 6: Hybrid Search Strategy
# ============================================================================


async def hybrid_search_example():
    """Demonstrate hybrid search with parallel + serial scaling."""
    print("\n" + "=" * 60)
    print("Example 6: Hybrid Search Strategy")
    print("=" * 60)

    # Create MCTS engine
    engine = MCTSEngine(
        seed=42,
        exploration_weight=1.414,
        progressive_widening_k=1.0,
        progressive_widening_alpha=0.5,
    )

    # Create hybrid search (without PRM/thinking for demo)
    config = HybridSearchConfig(
        num_parallel_candidates=4,
        prm_top_k=2,
        prm_threshold=0.3,
        early_terminate_confidence=0.95,
    )

    search = HybridMCTSSearch(
        mcts_engine=engine,
        config=config,
    )

    # Create root
    root_state = MCTSState(state_id="root", features={"query": "test"})
    root = ReasoningMCTSNode(state=root_state, rng=engine.rng)

    # Define action/transition functions
    def action_generator(state: MCTSState) -> list[str]:
        depth = len(state.state_id.split("_")) - 1
        if depth < 5:
            return ["analyze", "decompose", "solve", "verify"]
        return []

    def state_transition(state: MCTSState, action: str) -> MCTSState:
        return MCTSState(
            state_id=f"{state.state_id}_{action}",
            features={**state.features, "last_action": action},
        )

    rollout_policy = HybridRolloutPolicy(
        heuristic_fn=lambda s: 0.5 + 0.1 * len(s.state_id.split("_")),
    )

    # Run search
    result = await search.search(
        root=root,
        action_generator=action_generator,
        state_transition=state_transition,
        rollout_policy=rollout_policy,
        query="Find the optimal solution",
    )

    print("\nHybrid Search Results:")
    print(f"  Best action: {result.best_action}")
    print(f"  Total candidates: {len(result.all_candidates)}")
    print(f"  Phases completed: {[p.value for p in result.phases_completed]}")
    print(f"  Statistics: {result.statistics}")


# ============================================================================
# Example 7: PRM-MCTS Integration
# ============================================================================


async def prm_mcts_integration_example():
    """Demonstrate full PRM-MCTS integration."""
    print("\n" + "=" * 60)
    print("Example 7: PRM-MCTS Integration")
    print("=" * 60)

    # Create mock LLM PRM
    async def evaluate_fn(prompt: str) -> dict:
        # Simulate scoring based on content
        if "factor" in prompt.lower():
            return {"text": "SCORE: 0.9\nREASONING: Good approach"}
        return {"text": "SCORE: 0.7\nREASONING: Acceptable"}

    prm = LLMProcessRewardModel(evaluate_fn=evaluate_fn)

    config = PRMEnhancedMCTSConfig(
        prm_selection_weight=0.4,
        prm_expansion_threshold=0.5,
        prm_expansion_top_k=3,
    )

    integration = PRMMCTSIntegration(prm, config)

    # Create test trajectory
    trajectory = ReasoningTrajectory(query="Factor x^2 - 1")
    trajectory.add_step(ReasoningStep(
        content="Recognize as difference of squares",
        step_type="reasoning",
    ))
    trajectory.add_step(ReasoningStep(
        content="Factor as (x+1)(x-1)",
        step_type="action",
    ))

    # Compute enhanced UCT
    step = trajectory.steps[-1]
    enhanced_score = await integration.enhanced_uct_score(
        node_value=8.0,
        node_visits=10,
        parent_visits=50,
        step=step,
        trajectory=trajectory,
        exploration_weight=1.414,
    )

    print(f"\nTrajectory: {trajectory.query}")
    print(f"  Steps: {len(trajectory.steps)}")
    print(f"  Enhanced UCT Score: {enhanced_score:.4f}")

    # Filter candidates
    candidates = [
        ReasoningStep(content="Factor the expression", step_index=0),
        ReasoningStep(content="Use quadratic formula", step_index=0),
        ReasoningStep(content="Complete the square", step_index=0),
    ]

    filtered = await integration.filter_expansion_candidates(candidates, trajectory)

    print(f"\nFiltered candidates (top {config.prm_expansion_top_k}):")
    for step, score in filtered:
        print(f"  {step.content}: {score:.2f}")


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Reasoning-Enhanced MCTS Examples")
    print("=" * 60)

    # Run examples
    await prm_example()
    await heuristic_prm_example()
    extended_thinking_example()
    await reasoning_node_example()
    await dual_agent_example()
    await hybrid_search_example()
    await prm_mcts_integration_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
