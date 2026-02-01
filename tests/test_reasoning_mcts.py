"""
Comprehensive tests for reasoning-enhanced MCTS integration.

Tests cover:
- Process Reward Models (PRMs)
- Extended Thinking evaluation
- Hybrid search strategies
- Reasoning nodes and dual-agent architecture
- LangGraph integration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.framework.mcts.core import MCTSEngine, MCTSState
from src.framework.mcts.extended_thinking import (
    AdaptiveThinkingRouter,
    ParallelThinkingEvaluator,
    TaskComplexity,
    ThinkingBudget,
    ThinkingMode,
    ThinkingResult,
)
from src.framework.mcts.hybrid_search import (
    HybridMCTSSearch,
    HybridSearchConfig,
    SearchCandidate,
    SearchPhase,
)
from src.framework.mcts.policies import HybridRolloutPolicy
from src.framework.mcts.process_reward_model import (
    EnsemblePRM,
    HeuristicProcessRewardModel,
    LLMProcessRewardModel,
    PRMEnhancedMCTSConfig,
    PRMMCTSIntegration,
    PRMScore,
    PRMTrainingCollector,
    PRMTrainingExample,
    ReasoningStep,
    ReasoningTrajectory,
)
from src.framework.mcts.reasoning_node import (
    ActorAgent,
    AgentAction,
    DualAgentMCTSController,
    ReasonerAgent,
    ReasoningMCTSNode,
    ReasoningMetadata,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mcts_engine():
    """Create a basic MCTS engine for testing."""
    return MCTSEngine(
        seed=42,
        exploration_weight=1.414,
        progressive_widening_k=1.0,
        progressive_widening_alpha=0.5,
    )


@pytest.fixture
def simple_state():
    """Create a simple MCTS state for testing."""
    return MCTSState(
        state_id="test_root",
        features={"query": "Test query", "depth": 0},
    )


@pytest.fixture
def reasoning_trajectory():
    """Create a sample reasoning trajectory."""
    trajectory = ReasoningTrajectory(query="What is 2 + 2?")

    step1 = ReasoningStep(
        content="First, identify the operands: 2 and 2",
        step_index=0,
        step_type="reasoning",
        confidence=0.9,
    )
    trajectory.add_step(step1)

    step2 = ReasoningStep(
        content="Apply addition: 2 + 2 = 4",
        step_index=1,
        step_type="reasoning",
        confidence=0.95,
    )
    trajectory.add_step(step2)

    trajectory.final_answer = "4"
    trajectory.is_correct = True

    return trajectory


@pytest.fixture
def mock_llm_evaluate():
    """Create a mock LLM evaluate function."""
    async def evaluate(prompt: str) -> dict:
        return {
            "text": "SCORE: 0.85\nREASONING: Good reasoning step"
        }
    return evaluate


# ============================================================================
# Process Reward Model Tests
# ============================================================================


class TestReasoningStep:
    """Tests for ReasoningStep."""

    def test_create_step(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(
            content="This is a test step",
            step_index=0,
            step_type="reasoning",
            confidence=0.8,
        )

        assert step.content == "This is a test step"
        assert step.step_index == 0
        assert step.step_type == "reasoning"
        assert step.confidence == 0.8

    def test_step_hash(self):
        """Test step hashing is deterministic."""
        step1 = ReasoningStep(content="test", step_index=0)
        step2 = ReasoningStep(content="test", step_index=0)

        assert step1.to_hash_key() == step2.to_hash_key()

    def test_different_steps_different_hash(self):
        """Test different steps have different hashes."""
        step1 = ReasoningStep(content="test1", step_index=0)
        step2 = ReasoningStep(content="test2", step_index=0)

        assert step1.to_hash_key() != step2.to_hash_key()


class TestReasoningTrajectory:
    """Tests for ReasoningTrajectory."""

    def test_create_trajectory(self, reasoning_trajectory):
        """Test creating a trajectory."""
        assert reasoning_trajectory.query == "What is 2 + 2?"
        assert len(reasoning_trajectory.steps) == 2
        assert reasoning_trajectory.final_answer == "4"

    def test_add_step(self):
        """Test adding steps to trajectory."""
        trajectory = ReasoningTrajectory(query="test")

        step = ReasoningStep(content="step 1", step_index=0)
        trajectory.add_step(step)

        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].step_index == 0

    def test_get_prefix(self, reasoning_trajectory):
        """Test getting trajectory prefix."""
        prefix = reasoning_trajectory.get_prefix(1)

        assert len(prefix.steps) == 1
        assert prefix.steps[0].content == "First, identify the operands: 2 and 2"

    def test_to_text(self, reasoning_trajectory):
        """Test trajectory text representation."""
        text = reasoning_trajectory.to_text()

        assert "What is 2 + 2?" in text
        assert "Step 1" in text
        assert "Step 2" in text
        assert "Final Answer: 4" in text


class TestLLMProcessRewardModel:
    """Tests for LLM-based PRM."""

    @pytest.mark.asyncio
    async def test_score_step(self, mock_llm_evaluate, reasoning_trajectory):
        """Test scoring a step with LLM PRM."""
        prm = LLMProcessRewardModel(
            evaluate_fn=mock_llm_evaluate,
            cache_size=100,
        )

        step = reasoning_trajectory.steps[0]
        score = await prm.score_step(step, reasoning_trajectory)

        assert isinstance(score, PRMScore)
        assert 0.0 <= score.step_score <= 1.0
        assert score.step_score == 0.85

    @pytest.mark.asyncio
    async def test_score_trajectory(self, mock_llm_evaluate, reasoning_trajectory):
        """Test scoring full trajectory."""
        prm = LLMProcessRewardModel(
            evaluate_fn=mock_llm_evaluate,
            cache_size=100,
        )

        scores = await prm.score_trajectory(reasoning_trajectory)

        assert len(scores) == 2
        for score in scores:
            assert isinstance(score, PRMScore)

    @pytest.mark.asyncio
    async def test_caching(self, mock_llm_evaluate, reasoning_trajectory):
        """Test that PRM caches evaluations."""
        prm = LLMProcessRewardModel(
            evaluate_fn=mock_llm_evaluate,
            cache_size=100,
        )

        step = reasoning_trajectory.steps[0]

        # First call
        await prm.score_step(step, reasoning_trajectory)
        assert prm.cache_misses == 1

        # Second call should hit cache
        await prm.score_step(step, reasoning_trajectory)
        assert prm.cache_hits == 1


class TestHeuristicPRM:
    """Tests for heuristic-based PRM."""

    @pytest.mark.asyncio
    async def test_heuristic_scoring(self, reasoning_trajectory):
        """Test heuristic PRM scoring."""
        # Simple heuristics
        def length_heuristic(step, traj):
            return min(len(step.content) / 100, 1.0)

        def confidence_heuristic(step, traj):
            return step.confidence

        prm = HeuristicProcessRewardModel(
            heuristics=[length_heuristic, confidence_heuristic],
            weights=[0.3, 0.7],
        )

        step = reasoning_trajectory.steps[0]
        score = await prm.score_step(step, reasoning_trajectory)

        assert isinstance(score, PRMScore)
        assert 0.0 <= score.step_score <= 1.0


class TestEnsemblePRM:
    """Tests for ensemble PRM."""

    @pytest.mark.asyncio
    async def test_ensemble_scoring(self, mock_llm_evaluate, reasoning_trajectory):
        """Test ensemble PRM combines multiple models."""
        # Create two PRMs
        prm1 = LLMProcessRewardModel(evaluate_fn=mock_llm_evaluate)

        def simple_heuristic(step, traj):
            return 0.7

        prm2 = HeuristicProcessRewardModel(heuristics=[simple_heuristic])

        ensemble = EnsemblePRM(
            models=[prm1, prm2],
            weights=[0.6, 0.4],
            aggregation="weighted_mean",
        )

        step = reasoning_trajectory.steps[0]
        score = await ensemble.score_step(step, reasoning_trajectory)

        assert isinstance(score, PRMScore)
        # Should be weighted combination of 0.85 and 0.7
        expected = 0.6 * 0.85 + 0.4 * 0.7
        assert abs(score.step_score - expected) < 0.01


class TestPRMMCTSIntegration:
    """Tests for PRM-MCTS integration."""

    @pytest.mark.asyncio
    async def test_enhanced_uct(self, mock_llm_evaluate, reasoning_trajectory):
        """Test PRM-enhanced UCT score computation."""
        prm = LLMProcessRewardModel(evaluate_fn=mock_llm_evaluate)
        integration = PRMMCTSIntegration(prm)

        step = reasoning_trajectory.steps[0]

        score = await integration.enhanced_uct_score(
            node_value=5.0,
            node_visits=10,
            parent_visits=100,
            step=step,
            trajectory=reasoning_trajectory,
            exploration_weight=1.414,
        )

        assert isinstance(score, float)
        assert score > 0

    @pytest.mark.asyncio
    async def test_filter_candidates(self, mock_llm_evaluate, reasoning_trajectory):
        """Test PRM-based candidate filtering."""
        prm = LLMProcessRewardModel(evaluate_fn=mock_llm_evaluate)
        integration = PRMMCTSIntegration(
            prm,
            config=PRMEnhancedMCTSConfig(
                prm_expansion_threshold=0.5,
                prm_expansion_top_k=2,
            ),
        )

        candidates = [
            ReasoningStep(content=f"candidate {i}", step_index=i)
            for i in range(5)
        ]

        filtered = await integration.filter_expansion_candidates(
            candidates, reasoning_trajectory
        )

        assert len(filtered) <= 2
        for step, score in filtered:
            assert score >= 0.5


class TestPRMTrainingCollector:
    """Tests for PRM training data collection."""

    def test_record_trajectory(self, reasoning_trajectory):
        """Test recording trajectories for training."""
        def verify_fn(query, answer):
            return answer == "4"

        collector = PRMTrainingCollector(verify_fn=verify_fn)
        collector.record_trajectory(reasoning_trajectory)

        assert len(collector.collected_examples) == 2

    def test_export_training_data(self, reasoning_trajectory):
        """Test exporting training data."""
        def verify_fn(query, answer):
            return answer == "4"

        collector = PRMTrainingCollector(verify_fn=verify_fn)
        collector.record_trajectory(reasoning_trajectory)

        data = collector.export_training_data()

        assert len(data) == 2
        assert all("prefix" in ex for ex in data)
        assert all("step" in ex for ex in data)


# ============================================================================
# Extended Thinking Tests
# ============================================================================


class TestThinkingBudget:
    """Tests for thinking budget computation."""

    def test_compute_budget_basic(self):
        """Test basic budget computation."""
        budget = ThinkingBudget(
            min_tokens=1024,
            max_tokens=65536,
            default_tokens=8192,
        )

        computed = budget.compute_budget(depth=2, visits=5, ucb_score=0.5)

        assert budget.min_tokens <= computed <= budget.max_tokens

    def test_budget_scales_with_depth(self):
        """Test budget increases with depth."""
        budget = ThinkingBudget(depth_multiplier=1.2)

        shallow = budget.compute_budget(depth=1, visits=5, ucb_score=0.5)
        deep = budget.compute_budget(depth=5, visits=5, ucb_score=0.5)

        assert deep > shallow

    def test_budget_scales_with_uncertainty(self):
        """Test budget increases with uncertainty."""
        budget = ThinkingBudget(uncertainty_multiplier=1.5)

        certain = budget.compute_budget(depth=2, visits=5, ucb_score=0.5, uncertainty=0.2)
        uncertain = budget.compute_budget(depth=2, visits=5, ucb_score=0.5, uncertainty=0.9)

        assert uncertain > certain

    def test_get_mode(self):
        """Test mode determination from token count."""
        budget = ThinkingBudget()

        assert budget.get_mode(0) == ThinkingMode.NONE
        assert budget.get_mode(2000) == ThinkingMode.MINIMAL
        assert budget.get_mode(10000) == ThinkingMode.STANDARD
        assert budget.get_mode(40000) == ThinkingMode.EXTENDED
        assert budget.get_mode(100000) == ThinkingMode.DEEP


class TestThinkingResult:
    """Tests for thinking results."""

    def test_create_result(self):
        """Test creating a thinking result."""
        result = ThinkingResult(
            score=0.85,
            thinking_trace="I think through this carefully...",
            analysis="Good solution",
            is_terminal=False,
            confidence=0.9,
            tokens_used=5000,
            mode=ThinkingMode.STANDARD,
        )

        assert result.score == 0.85
        assert result.confidence == 0.9
        assert result.mode == ThinkingMode.STANDARD


class TestTaskComplexity:
    """Tests for task complexity assessment."""

    def test_create_complexity(self):
        """Test creating complexity assessment."""
        complexity = TaskComplexity(
            complexity_score=0.7,
            reasoning_required=True,
            domain="math",
            requires_verification=True,
            estimated_steps=5,
            overthinking_risk=0.2,
        )

        assert complexity.complexity_score == 0.7
        assert complexity.reasoning_required is True
        assert complexity.domain == "math"


# ============================================================================
# Hybrid Search Tests
# ============================================================================


class TestHybridSearchConfig:
    """Tests for hybrid search configuration."""

    def test_validate_config(self):
        """Test config validation."""
        config = HybridSearchConfig(
            parallel_budget_ratio=0.25,
            prm_budget_ratio=0.25,
            extended_budget_ratio=0.50,
        )

        config.validate()  # Should not raise

    def test_invalid_config(self):
        """Test invalid config raises error."""
        config = HybridSearchConfig(
            parallel_budget_ratio=0.5,
            prm_budget_ratio=0.5,
            extended_budget_ratio=0.5,  # Sum > 1
        )

        with pytest.raises(ValueError):
            config.validate()


class TestHybridMCTSSearch:
    """Tests for hybrid MCTS search."""

    @pytest.mark.asyncio
    async def test_search_basic(self, mcts_engine, simple_state):
        """Test basic hybrid search execution."""
        search = HybridMCTSSearch(
            mcts_engine=mcts_engine,
            config=HybridSearchConfig(
                num_parallel_candidates=3,
                prm_top_k=2,
            ),
        )

        root = ReasoningMCTSNode(state=simple_state, rng=mcts_engine.rng)

        def action_generator(state):
            return ["action_A", "action_B", "action_C"]

        def state_transition(state, action):
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={**state.features, "action": action},
            )

        rollout_policy = HybridRolloutPolicy(
            heuristic_fn=lambda s: 0.5,
        )

        result = await search.search(
            root=root,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            query="test query",
        )

        assert isinstance(result.best_action, str) or result.best_action is None
        assert SearchPhase.PARALLEL_GENERATION in result.phases_completed


class TestSearchCandidate:
    """Tests for search candidates."""

    def test_create_candidate(self, simple_state):
        """Test creating a search candidate."""
        node = ReasoningMCTSNode(state=simple_state)
        trajectory = ReasoningTrajectory(query="test")

        candidate = SearchCandidate(
            node=node,
            path=[node],
            trajectory=trajectory,
            prm_score=0.8,
            confidence=0.75,
        )

        assert candidate.prm_score == 0.8
        assert candidate.confidence == 0.75


# ============================================================================
# Reasoning Node Tests
# ============================================================================


class TestReasoningMCTSNode:
    """Tests for reasoning-enhanced MCTS nodes."""

    def test_create_node(self, simple_state):
        """Test creating a reasoning node."""
        node = ReasoningMCTSNode(state=simple_state)

        assert node.state == simple_state
        assert isinstance(node.reasoning, ReasoningMetadata)
        assert node.visits == 0

    def test_set_thinking_result(self, simple_state):
        """Test setting thinking result on node."""
        node = ReasoningMCTSNode(state=simple_state)

        result = ThinkingResult(
            score=0.8,
            thinking_trace="Deep analysis...",
            analysis="Good",
            tokens_used=5000,
        )

        node.set_thinking_result(result)

        assert node.reasoning.thinking_result == result
        assert node.reasoning.thinking_trace == "Deep analysis..."
        assert node.reasoning.thinking_tokens_used == 5000

    def test_set_prm_score(self, simple_state):
        """Test setting PRM score on node."""
        node = ReasoningMCTSNode(state=simple_state)

        score = PRMScore(
            step_score=0.85,
            cumulative_score=0.85,
            confidence=0.9,
        )

        node.set_prm_score(score)

        assert node.reasoning.prm_score == 0.85
        assert len(node.reasoning.prm_scores_history) == 1

    def test_add_reasoning_step(self, simple_state):
        """Test adding reasoning step to node."""
        node = ReasoningMCTSNode(state=simple_state)

        step = ReasoningStep(
            content="Analysis step",
            step_index=0,
            step_type="reasoning",
        )

        node.add_reasoning_step(step)

        assert len(node.reasoning.reasoning_steps) == 1

    def test_get_trajectory(self, simple_state):
        """Test getting trajectory from node."""
        # Create parent-child chain
        parent = ReasoningMCTSNode(state=simple_state)
        parent.add_reasoning_step(ReasoningStep(
            content="Step 1", step_index=0
        ))

        child_state = MCTSState(state_id="child", features={})
        child = parent.add_reasoning_child(
            action="test_action",
            child_state=child_state,
            reasoning_step=ReasoningStep(content="Step 2", step_index=0),
        )

        trajectory = child.get_trajectory()

        assert len(trajectory.steps) == 2

    def test_add_reasoning_child(self, simple_state):
        """Test adding reasoning child to node."""
        parent = ReasoningMCTSNode(state=simple_state)

        child_state = MCTSState(state_id="child", features={})
        step = ReasoningStep(content="action step", step_index=0)

        child = parent.add_reasoning_child(
            action="test_action",
            child_state=child_state,
            reasoning_step=step,
        )

        assert isinstance(child, ReasoningMCTSNode)
        assert len(child.reasoning.reasoning_steps) == 1
        assert child.action == "test_action"


class TestAgentAction:
    """Tests for agent actions."""

    def test_create_action(self):
        """Test creating an agent action."""
        action = AgentAction(
            action="analyze the problem",
            confidence=0.9,
            reasoning="This is a good first step",
            source_agent="reasoner",
            estimated_value=0.8,
        )

        assert action.action == "analyze the problem"
        assert action.confidence == 0.9
        assert action.source_agent == "reasoner"


class TestReasonerAgent:
    """Tests for reasoner agent."""

    @pytest.mark.asyncio
    async def test_propose_strategies(self):
        """Test strategy proposal."""
        async def model_fn(prompt, tokens):
            return """
STRATEGY: Analyze the problem structure
REASONING: Understanding structure helps find solutions
CONFIDENCE: 0.85
ESTIMATED_VALUE: 0.8

STRATEGY: Try a different approach
REASONING: Alternative methods can be more efficient
CONFIDENCE: 0.7
ESTIMATED_VALUE: 0.75
"""

        reasoner = ReasonerAgent(
            model_fn=model_fn,
            default_thinking_tokens=8000,
        )

        state = MCTSState(state_id="test", features={})
        strategies = await reasoner.propose_strategies(
            state=state,
            context="Test context",
            n_strategies=2,
        )

        assert len(strategies) == 2
        assert all(isinstance(s, AgentAction) for s in strategies)
        assert strategies[0].source_agent == "reasoner"

    @pytest.mark.asyncio
    async def test_evaluate_state(self):
        """Test state evaluation."""
        async def model_fn(prompt, tokens):
            return """
SCORE: 0.75
ANALYSIS: Good progress toward solution
IS_TERMINAL: no
CONFIDENCE: 0.8
"""

        reasoner = ReasonerAgent(model_fn=model_fn)

        state = MCTSState(state_id="test", features={})
        result = await reasoner.evaluate_state(state, "context")

        assert isinstance(result, ThinkingResult)
        assert result.score == 0.75
        assert result.is_terminal is False


class TestActorAgent:
    """Tests for actor agent."""

    @pytest.mark.asyncio
    async def test_execute_strategy(self):
        """Test strategy execution."""
        async def model_fn(prompt):
            return "Executed strategy successfully"

        actor = ActorAgent(
            model_fn=model_fn,
            tools=["search", "calculate"],
        )

        strategy = AgentAction(
            action="analyze data",
            confidence=0.8,
            reasoning="Need to understand the data",
        )

        state = MCTSState(state_id="test", features={})
        new_state, metadata = await actor.execute_strategy(
            strategy=strategy,
            state=state,
            context="test context",
        )

        assert isinstance(new_state, MCTSState)
        assert "analyze" in new_state.state_id


class TestDualAgentController:
    """Tests for dual-agent controller."""

    @pytest.mark.asyncio
    async def test_expand_with_reasoning(self, simple_state):
        """Test dual-agent expansion."""
        async def reasoner_model(prompt, tokens):
            return """
STRATEGY: Step 1
REASONING: First step reasoning
CONFIDENCE: 0.8
ESTIMATED_VALUE: 0.7
"""

        async def actor_model(prompt):
            return "Executed"

        reasoner = ReasonerAgent(model_fn=reasoner_model)
        actor = ActorAgent(model_fn=actor_model)

        controller = DualAgentMCTSController(
            reasoner=reasoner,
            actor=actor,
        )

        node = ReasoningMCTSNode(state=simple_state)

        children = await controller.expand_with_reasoning(
            node=node,
            context="test",
            n_strategies=1,
        )

        assert len(children) >= 1
        assert all(isinstance(c, ReasoningMCTSNode) for c in children)


# ============================================================================
# Integration Tests
# ============================================================================


class TestFullIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_prm_mcts_workflow(self, mcts_engine, mock_llm_evaluate):
        """Test full PRM-enhanced MCTS workflow."""
        # Create components
        prm = LLMProcessRewardModel(evaluate_fn=mock_llm_evaluate)

        state = MCTSState(
            state_id="root",
            features={"query": "Test problem"},
        )

        root = ReasoningMCTSNode(state=state, rng=mcts_engine.rng)
        root.available_actions = ["a", "b", "c"]

        # Add children
        for action in ["a", "b"]:
            child_state = MCTSState(
                state_id=f"root_{action}",
                features={"action": action},
            )
            step = ReasoningStep(content=action, step_index=0)
            root.add_reasoning_child(action, child_state, step)

        # Score trajectories
        for child in root.children:
            if isinstance(child, ReasoningMCTSNode):
                trajectory = child.get_trajectory()
                scores = await prm.score_trajectory(trajectory)
                if scores:
                    child.set_prm_score(scores[-1])

        # Select using PRM-enhanced UCB
        selected = root.select_child_with_prm(
            exploration_weight=1.414,
            prm_weight=0.3,
        )

        assert selected is not None
        assert isinstance(selected, ReasoningMCTSNode)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_prm(self, mcts_engine, mock_llm_evaluate):
        """Test hybrid search with PRM filtering."""
        prm = LLMProcessRewardModel(evaluate_fn=mock_llm_evaluate)

        search = HybridMCTSSearch(
            mcts_engine=mcts_engine,
            prm=prm,
            config=HybridSearchConfig(
                num_parallel_candidates=4,
                prm_top_k=2,
                prm_threshold=0.3,
            ),
        )

        state = MCTSState(state_id="root", features={})
        root = ReasoningMCTSNode(state=state, rng=mcts_engine.rng)

        result = await search.search(
            root=root,
            action_generator=lambda s: ["a", "b", "c"],
            state_transition=lambda s, a: MCTSState(
                state_id=f"{s.state_id}_{a}",
                features={},
            ),
            rollout_policy=HybridRolloutPolicy(heuristic_fn=lambda s: 0.5),
            query="test",
        )

        assert SearchPhase.PARALLEL_GENERATION in result.phases_completed
        assert SearchPhase.PRM_FILTERING in result.phases_completed


# ============================================================================
# Run tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
