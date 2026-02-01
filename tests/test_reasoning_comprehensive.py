#!/usr/bin/env python3
"""
Comprehensive Test Suite for Reasoning-Enhanced MCTS Integration.

This test suite covers:
- Unit tests for all components
- Integration tests for component interactions
- End-to-end tests for full workflows
- User journey tests for common use cases

Can run without pytest using Python's unittest module.
"""

import asyncio
import sys
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, "/home/user/langgraph_multi_agent_mcts")


# ============================================================================
# Test Utilities
# ============================================================================


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class MockResponse:
    """Mock response object for LLM calls."""

    def __init__(self, text: str):
        self.text = text


# ============================================================================
# UNIT TESTS: Process Reward Model Components
# ============================================================================


class TestReasoningStepUnit(unittest.TestCase):
    """Unit tests for ReasoningStep class."""

    def test_create_step_with_defaults(self):
        """Test creating a step with default values."""
        from src.framework.mcts.process_reward_model import ReasoningStep

        step = ReasoningStep(content="Test content", step_index=0)

        self.assertEqual(step.content, "Test content")
        self.assertEqual(step.step_index, 0)
        self.assertEqual(step.step_type, "reasoning")
        self.assertEqual(step.confidence, 0.0)
        self.assertEqual(step.thinking_tokens, 0)
        self.assertEqual(step.metadata, {})

    def test_create_step_with_all_fields(self):
        """Test creating a step with all fields specified."""
        from src.framework.mcts.process_reward_model import ReasoningStep

        step = ReasoningStep(
            content="Complex reasoning step",
            step_index=5,
            step_type="action",
            confidence=0.95,
            thinking_tokens=5000,
            metadata={"tool": "calculator"},
        )

        self.assertEqual(step.step_type, "action")
        self.assertEqual(step.confidence, 0.95)
        self.assertEqual(step.thinking_tokens, 5000)
        self.assertEqual(step.metadata["tool"], "calculator")

    def test_hash_determinism(self):
        """Test that hashing is deterministic."""
        from src.framework.mcts.process_reward_model import ReasoningStep

        step1 = ReasoningStep(content="same content", step_index=1)
        step2 = ReasoningStep(content="same content", step_index=1)

        self.assertEqual(step1.to_hash_key(), step2.to_hash_key())

    def test_hash_uniqueness(self):
        """Test that different steps produce different hashes."""
        from src.framework.mcts.process_reward_model import ReasoningStep

        step1 = ReasoningStep(content="content A", step_index=0)
        step2 = ReasoningStep(content="content B", step_index=0)
        step3 = ReasoningStep(content="content A", step_index=1)

        self.assertNotEqual(step1.to_hash_key(), step2.to_hash_key())
        self.assertNotEqual(step1.to_hash_key(), step3.to_hash_key())


class TestReasoningTrajectoryUnit(unittest.TestCase):
    """Unit tests for ReasoningTrajectory class."""

    def test_create_empty_trajectory(self):
        """Test creating an empty trajectory."""
        from src.framework.mcts.process_reward_model import ReasoningTrajectory

        traj = ReasoningTrajectory(query="Test query")

        self.assertEqual(traj.query, "Test query")
        self.assertEqual(len(traj.steps), 0)
        self.assertIsNone(traj.final_answer)
        self.assertIsNone(traj.is_correct)

    def test_add_step_updates_index(self):
        """Test that adding steps updates step indices correctly."""
        from src.framework.mcts.process_reward_model import (
            ReasoningStep,
            ReasoningTrajectory,
        )

        traj = ReasoningTrajectory(query="test")

        step1 = ReasoningStep(content="step 1", step_index=99)  # Wrong index
        traj.add_step(step1)
        self.assertEqual(traj.steps[0].step_index, 0)

        step2 = ReasoningStep(content="step 2", step_index=99)
        traj.add_step(step2)
        self.assertEqual(traj.steps[1].step_index, 1)

    def test_get_prefix(self):
        """Test getting trajectory prefix."""
        from src.framework.mcts.process_reward_model import (
            ReasoningStep,
            ReasoningTrajectory,
        )

        traj = ReasoningTrajectory(query="test")
        for i in range(5):
            traj.add_step(ReasoningStep(content=f"step {i}", step_index=i))

        prefix = traj.get_prefix(3)

        self.assertEqual(len(prefix.steps), 3)
        self.assertEqual(prefix.query, "test")
        self.assertEqual(prefix.steps[2].content, "step 2")

    def test_get_prefix_empty(self):
        """Test getting prefix of length 0."""
        from src.framework.mcts.process_reward_model import (
            ReasoningStep,
            ReasoningTrajectory,
        )

        traj = ReasoningTrajectory(query="test")
        traj.add_step(ReasoningStep(content="step", step_index=0))

        prefix = traj.get_prefix(0)
        self.assertEqual(len(prefix.steps), 0)

    def test_to_text_formatting(self):
        """Test text representation formatting."""
        from src.framework.mcts.process_reward_model import (
            ReasoningStep,
            ReasoningTrajectory,
        )

        traj = ReasoningTrajectory(query="Solve x + 1 = 2")
        traj.add_step(ReasoningStep(content="Subtract 1", step_index=0, step_type="reasoning"))
        traj.add_step(ReasoningStep(content="x = 1", step_index=0, step_type="action"))
        traj.final_answer = "x = 1"

        text = traj.to_text()

        self.assertIn("Query:", text)
        self.assertIn("Solve x + 1 = 2", text)
        self.assertIn("Step 1", text)
        self.assertIn("Step 2", text)
        self.assertIn("Final Answer:", text)

    def test_trajectory_hash(self):
        """Test trajectory hashing."""
        from src.framework.mcts.process_reward_model import (
            ReasoningStep,
            ReasoningTrajectory,
        )

        traj1 = ReasoningTrajectory(query="test")
        traj1.add_step(ReasoningStep(content="step", step_index=0))

        traj2 = ReasoningTrajectory(query="test")
        traj2.add_step(ReasoningStep(content="step", step_index=0))

        self.assertEqual(traj1.to_hash_key(), traj2.to_hash_key())


class TestPRMScoreUnit(unittest.TestCase):
    """Unit tests for PRMScore class."""

    def test_create_score(self):
        """Test creating a PRM score."""
        from src.framework.mcts.process_reward_model import PRMScore

        score = PRMScore(
            step_score=0.85,
            cumulative_score=0.72,
            confidence=0.9,
            reasoning="Good step",
        )

        self.assertEqual(score.step_score, 0.85)
        self.assertEqual(score.cumulative_score, 0.72)
        self.assertEqual(score.confidence, 0.9)
        self.assertEqual(score.reasoning, "Good step")


# ============================================================================
# UNIT TESTS: Extended Thinking Components
# ============================================================================


class TestThinkingBudgetUnit(unittest.TestCase):
    """Unit tests for ThinkingBudget class."""

    def test_default_budget(self):
        """Test default budget values."""
        from src.framework.mcts.extended_thinking import ThinkingBudget

        budget = ThinkingBudget()

        self.assertEqual(budget.min_tokens, 1024)
        self.assertEqual(budget.max_tokens, 65536)
        self.assertEqual(budget.default_tokens, 8192)

    def test_compute_budget_respects_bounds(self):
        """Test that computed budget stays within bounds."""
        from src.framework.mcts.extended_thinking import ThinkingBudget

        budget = ThinkingBudget(min_tokens=1000, max_tokens=10000)

        # Test various scenarios
        for depth in range(20):
            for visits in range(10):
                computed = budget.compute_budget(
                    depth=depth, visits=visits, ucb_score=0.5
                )
                self.assertGreaterEqual(computed, budget.min_tokens)
                self.assertLessEqual(computed, budget.max_tokens)

    def test_depth_scaling(self):
        """Test that budget increases with depth."""
        from src.framework.mcts.extended_thinking import ThinkingBudget

        budget = ThinkingBudget(depth_multiplier=1.5)

        shallow = budget.compute_budget(depth=1, visits=5, ucb_score=0.5)
        deep = budget.compute_budget(depth=5, visits=5, ucb_score=0.5)

        self.assertGreater(deep, shallow)

    def test_uncertainty_scaling(self):
        """Test that budget increases with uncertainty."""
        from src.framework.mcts.extended_thinking import ThinkingBudget

        budget = ThinkingBudget(uncertainty_multiplier=2.0)

        certain = budget.compute_budget(
            depth=2, visits=5, ucb_score=0.5, uncertainty=0.1
        )
        uncertain = budget.compute_budget(
            depth=2, visits=5, ucb_score=0.5, uncertainty=0.9
        )

        self.assertGreater(uncertain, certain)

    def test_critical_node_bonus(self):
        """Test that critical nodes get extra budget."""
        from src.framework.mcts.extended_thinking import ThinkingBudget

        budget = ThinkingBudget(critical_threshold=0.7)

        normal = budget.compute_budget(depth=2, visits=5, ucb_score=0.5)
        critical = budget.compute_budget(depth=2, visits=5, ucb_score=0.9)

        self.assertGreater(critical, normal)

    def test_mode_thresholds(self):
        """Test thinking mode determination."""
        from src.framework.mcts.extended_thinking import ThinkingBudget, ThinkingMode

        budget = ThinkingBudget()

        self.assertEqual(budget.get_mode(0), ThinkingMode.NONE)
        self.assertEqual(budget.get_mode(2000), ThinkingMode.MINIMAL)
        self.assertEqual(budget.get_mode(10000), ThinkingMode.STANDARD)
        self.assertEqual(budget.get_mode(40000), ThinkingMode.EXTENDED)
        self.assertEqual(budget.get_mode(100000), ThinkingMode.DEEP)


class TestThinkingResultUnit(unittest.TestCase):
    """Unit tests for ThinkingResult class."""

    def test_create_result(self):
        """Test creating a thinking result."""
        from src.framework.mcts.extended_thinking import ThinkingMode, ThinkingResult

        result = ThinkingResult(
            score=0.85,
            thinking_trace="I analyzed this carefully...",
            analysis="Strong solution",
            is_terminal=False,
            confidence=0.9,
            tokens_used=8000,
            mode=ThinkingMode.STANDARD,
        )

        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.tokens_used, 8000)
        self.assertEqual(result.mode, ThinkingMode.STANDARD)
        self.assertFalse(result.is_terminal)


class TestTaskComplexityUnit(unittest.TestCase):
    """Unit tests for TaskComplexity class."""

    def test_create_complexity(self):
        """Test creating task complexity assessment."""
        from src.framework.mcts.extended_thinking import TaskComplexity

        complexity = TaskComplexity(
            complexity_score=0.8,
            reasoning_required=True,
            domain="math",
            requires_verification=True,
            estimated_steps=7,
            overthinking_risk=0.3,
        )

        self.assertEqual(complexity.complexity_score, 0.8)
        self.assertTrue(complexity.reasoning_required)
        self.assertEqual(complexity.domain, "math")
        self.assertEqual(complexity.estimated_steps, 7)


# ============================================================================
# UNIT TESTS: Hybrid Search Components
# ============================================================================


class TestHybridSearchConfigUnit(unittest.TestCase):
    """Unit tests for HybridSearchConfig class."""

    def test_valid_config(self):
        """Test valid configuration."""
        from src.framework.mcts.hybrid_search import HybridSearchConfig

        config = HybridSearchConfig(
            parallel_budget_ratio=0.25,
            prm_budget_ratio=0.25,
            extended_budget_ratio=0.50,
        )

        # Should not raise
        config.validate()

    def test_invalid_config_sum_greater(self):
        """Test validation fails when ratios sum > 1."""
        from src.framework.mcts.hybrid_search import HybridSearchConfig

        config = HybridSearchConfig(
            parallel_budget_ratio=0.4,
            prm_budget_ratio=0.4,
            extended_budget_ratio=0.4,
        )

        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_config_sum_less(self):
        """Test validation fails when ratios sum < 1."""
        from src.framework.mcts.hybrid_search import HybridSearchConfig

        config = HybridSearchConfig(
            parallel_budget_ratio=0.1,
            prm_budget_ratio=0.1,
            extended_budget_ratio=0.1,
        )

        with self.assertRaises(ValueError):
            config.validate()


class TestSearchCandidateUnit(unittest.TestCase):
    """Unit tests for SearchCandidate class."""

    def test_create_candidate(self):
        """Test creating a search candidate."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.hybrid_search import SearchCandidate
        from src.framework.mcts.process_reward_model import ReasoningTrajectory
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        state = MCTSState(state_id="test", features={})
        node = ReasoningMCTSNode(state=state)
        traj = ReasoningTrajectory(query="test")

        candidate = SearchCandidate(
            node=node,
            path=[node],
            trajectory=traj,
            prm_score=0.8,
            confidence=0.75,
        )

        self.assertEqual(candidate.prm_score, 0.8)
        self.assertEqual(candidate.confidence, 0.75)
        self.assertEqual(len(candidate.path), 1)


# ============================================================================
# UNIT TESTS: Reasoning Node Components
# ============================================================================


class TestReasoningMetadataUnit(unittest.TestCase):
    """Unit tests for ReasoningMetadata class."""

    def test_default_metadata(self):
        """Test default metadata values."""
        from src.framework.mcts.reasoning_node import ReasoningMetadata

        metadata = ReasoningMetadata()

        self.assertEqual(metadata.thinking_trace, "")
        self.assertEqual(len(metadata.reasoning_steps), 0)
        self.assertIsNone(metadata.prm_score)
        self.assertEqual(metadata.thinking_tokens_used, 0)
        self.assertFalse(metadata.verified)


class TestReasoningMCTSNodeUnit(unittest.TestCase):
    """Unit tests for ReasoningMCTSNode class."""

    def test_create_node(self):
        """Test creating a reasoning node."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.reasoning_node import (
            ReasoningMCTSNode,
            ReasoningMetadata,
        )

        state = MCTSState(state_id="test", features={"key": "value"})
        node = ReasoningMCTSNode(state=state)

        self.assertEqual(node.state, state)
        self.assertIsInstance(node.reasoning, ReasoningMetadata)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(len(node.children), 0)

    def test_set_thinking_result(self):
        """Test setting thinking result."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.extended_thinking import ThinkingResult
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        state = MCTSState(state_id="test", features={})
        node = ReasoningMCTSNode(state=state)

        result = ThinkingResult(
            score=0.8,
            thinking_trace="Deep analysis here...",
            analysis="Good",
            tokens_used=5000,
        )

        node.set_thinking_result(result)

        self.assertEqual(node.reasoning.thinking_result, result)
        self.assertEqual(node.reasoning.thinking_trace, "Deep analysis here...")
        self.assertEqual(node.reasoning.thinking_tokens_used, 5000)

    def test_set_prm_score(self):
        """Test setting PRM score."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.process_reward_model import PRMScore
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        state = MCTSState(state_id="test", features={})
        node = ReasoningMCTSNode(state=state)

        score = PRMScore(step_score=0.85, cumulative_score=0.85, confidence=0.9)
        node.set_prm_score(score)

        self.assertEqual(node.reasoning.prm_score, 0.85)
        self.assertEqual(len(node.reasoning.prm_scores_history), 1)

    def test_add_reasoning_step(self):
        """Test adding reasoning step."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.process_reward_model import ReasoningStep
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        state = MCTSState(state_id="test", features={})
        node = ReasoningMCTSNode(state=state)

        step = ReasoningStep(content="Analyze problem", step_index=99)
        node.add_reasoning_step(step)

        self.assertEqual(len(node.reasoning.reasoning_steps), 1)
        # Index should be updated to 0
        self.assertEqual(node.reasoning.reasoning_steps[0].step_index, 0)

    def test_add_reasoning_child(self):
        """Test adding a reasoning child."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.process_reward_model import ReasoningStep
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        parent_state = MCTSState(state_id="parent", features={})
        parent = ReasoningMCTSNode(state=parent_state)

        child_state = MCTSState(state_id="child", features={})
        step = ReasoningStep(content="Action step", step_index=0)

        child = parent.add_reasoning_child(
            action="test_action", child_state=child_state, reasoning_step=step
        )

        self.assertIsInstance(child, ReasoningMCTSNode)
        self.assertEqual(child.action, "test_action")
        self.assertEqual(child.parent, parent)
        self.assertEqual(len(child.reasoning.reasoning_steps), 1)
        self.assertIn(child, parent.children)

    def test_get_trajectory(self):
        """Test getting trajectory from node chain."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.process_reward_model import ReasoningStep
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        # Create chain: root -> child1 -> child2
        root = ReasoningMCTSNode(state=MCTSState(state_id="root", features={}))
        root.add_reasoning_step(ReasoningStep(content="Step 0", step_index=0))

        child1 = root.add_reasoning_child(
            action="a1",
            child_state=MCTSState(state_id="c1", features={}),
            reasoning_step=ReasoningStep(content="Step 1", step_index=0),
        )

        child2 = child1.add_reasoning_child(
            action="a2",
            child_state=MCTSState(state_id="c2", features={}),
            reasoning_step=ReasoningStep(content="Step 2", step_index=0),
        )

        trajectory = child2.get_trajectory()

        self.assertEqual(len(trajectory.steps), 3)


class TestAgentActionUnit(unittest.TestCase):
    """Unit tests for AgentAction class."""

    def test_create_action(self):
        """Test creating an agent action."""
        from src.framework.mcts.reasoning_node import AgentAction

        action = AgentAction(
            action="decompose_problem",
            confidence=0.9,
            reasoning="Breaking down helps solve",
            source_agent="reasoner",
            estimated_value=0.8,
        )

        self.assertEqual(action.action, "decompose_problem")
        self.assertEqual(action.confidence, 0.9)
        self.assertEqual(action.source_agent, "reasoner")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestPRMIntegration(unittest.TestCase):
    """Integration tests for PRM with MCTS components."""

    def test_prm_with_trajectory_scoring(self):
        """Test PRM scoring a complete trajectory."""

        async def run_test():
            from src.framework.mcts.process_reward_model import (
                HeuristicProcessRewardModel,
                ReasoningStep,
                ReasoningTrajectory,
            )

            # Define heuristics
            def quality_heuristic(step, traj):
                return min(len(step.content) / 50, 1.0)

            prm = HeuristicProcessRewardModel(heuristics=[quality_heuristic])

            # Create trajectory
            traj = ReasoningTrajectory(query="Compute 5 factorial")
            traj.add_step(
                ReasoningStep(content="5! means 5 × 4 × 3 × 2 × 1", step_index=0)
            )
            traj.add_step(
                ReasoningStep(content="5 × 4 = 20", step_index=1)
            )
            traj.add_step(
                ReasoningStep(content="20 × 3 = 60", step_index=2)
            )
            traj.add_step(
                ReasoningStep(content="60 × 2 = 120", step_index=3)
            )
            traj.add_step(
                ReasoningStep(content="120 × 1 = 120, so 5! = 120", step_index=4)
            )

            scores = await prm.score_trajectory(traj)

            self.assertEqual(len(scores), 5)
            for score in scores:
                self.assertGreaterEqual(score.step_score, 0)
                self.assertLessEqual(score.step_score, 1)

        run_async(run_test())

    def test_prm_integration_with_reasoning_node(self):
        """Test PRM integration with ReasoningMCTSNode."""

        async def run_test():
            from src.framework.mcts.core import MCTSState
            from src.framework.mcts.process_reward_model import (
                HeuristicProcessRewardModel,
                PRMScore,
                ReasoningStep,
            )
            from src.framework.mcts.reasoning_node import ReasoningMCTSNode

            def simple_heuristic(step, traj):
                return 0.8

            prm = HeuristicProcessRewardModel(heuristics=[simple_heuristic])

            # Create node with steps
            root = ReasoningMCTSNode(
                state=MCTSState(state_id="root", features={})
            )
            root.add_reasoning_step(
                ReasoningStep(content="Analysis step", step_index=0)
            )

            # Get trajectory and score
            trajectory = root.get_trajectory()
            scores = await prm.score_trajectory(trajectory)

            # Set score on node
            root.set_prm_score(scores[0])

            self.assertEqual(root.reasoning.prm_score, 0.8)

        run_async(run_test())


class TestDualAgentIntegration(unittest.TestCase):
    """Integration tests for dual-agent architecture."""

    def test_reasoner_actor_coordination(self):
        """Test reasoner and actor working together."""

        async def run_test():
            from src.framework.mcts.core import MCTSState
            from src.framework.mcts.reasoning_node import (
                ActorAgent,
                DualAgentMCTSController,
                ReasonerAgent,
                ReasoningMCTSNode,
            )

            # Mock reasoner model
            async def reasoner_model(prompt, tokens):
                return """
STRATEGY: Analyze structure
REASONING: Understanding helps
CONFIDENCE: 0.85
ESTIMATED_VALUE: 0.8
"""

            # Mock actor model
            async def actor_model(prompt):
                return "Executed successfully"

            reasoner = ReasonerAgent(model_fn=reasoner_model)
            actor = ActorAgent(model_fn=actor_model)
            controller = DualAgentMCTSController(reasoner=reasoner, actor=actor)

            # Create root node
            root = ReasoningMCTSNode(
                state=MCTSState(state_id="root", features={"problem": "test"})
            )

            # Expand with reasoning
            children = await controller.expand_with_reasoning(
                node=root, context="Test problem", n_strategies=1
            )

            self.assertGreaterEqual(len(children), 1)
            for child in children:
                self.assertIsInstance(child, ReasoningMCTSNode)
                self.assertEqual(child.reasoning.source_agent, "dual")

        run_async(run_test())


class TestHybridSearchIntegration(unittest.TestCase):
    """Integration tests for hybrid search."""

    def test_hybrid_search_phases(self):
        """Test that hybrid search completes all phases."""

        async def run_test():
            from src.framework.mcts.core import MCTSEngine, MCTSState
            from src.framework.mcts.hybrid_search import (
                HybridMCTSSearch,
                HybridSearchConfig,
                SearchPhase,
            )
            from src.framework.mcts.policies import HybridRolloutPolicy
            from src.framework.mcts.reasoning_node import ReasoningMCTSNode

            engine = MCTSEngine(seed=42)
            config = HybridSearchConfig(
                num_parallel_candidates=3,
                prm_top_k=2,
            )

            search = HybridMCTSSearch(mcts_engine=engine, config=config)

            state = MCTSState(state_id="root", features={})
            root = ReasoningMCTSNode(state=state, rng=engine.rng)

            def action_gen(s):
                return ["a", "b", "c"]

            def state_trans(s, a):
                return MCTSState(state_id=f"{s.state_id}_{a}", features={})

            rollout = HybridRolloutPolicy(heuristic_fn=lambda s: 0.5)

            result = await search.search(
                root=root,
                action_generator=action_gen,
                state_transition=state_trans,
                rollout_policy=rollout,
                query="test",
            )

            self.assertIn(SearchPhase.PARALLEL_GENERATION, result.phases_completed)
            self.assertGreater(len(result.all_candidates), 0)

        run_async(run_test())


# ============================================================================
# END-TO-END TESTS
# ============================================================================


class TestE2EReasoningWorkflow(unittest.TestCase):
    """End-to-end tests for complete reasoning workflows."""

    def test_complete_mcts_reasoning_workflow(self):
        """Test complete MCTS workflow with all reasoning components."""

        async def run_test():
            from src.framework.mcts.core import MCTSEngine, MCTSState
            from src.framework.mcts.extended_thinking import ThinkingBudget
            from src.framework.mcts.policies import HybridRolloutPolicy
            from src.framework.mcts.process_reward_model import (
                HeuristicProcessRewardModel,
                PRMMCTSIntegration,
                ReasoningStep,
            )
            from src.framework.mcts.reasoning_node import ReasoningMCTSNode

            # Setup components
            engine = MCTSEngine(seed=42)
            budget = ThinkingBudget()

            def quality_heuristic(step, traj):
                return 0.8

            prm = HeuristicProcessRewardModel(heuristics=[quality_heuristic])
            prm_integration = PRMMCTSIntegration(prm)

            # Create root
            root_state = MCTSState(
                state_id="root",
                features={"query": "Solve x^2 = 4"},
            )
            root = ReasoningMCTSNode(state=root_state, rng=engine.rng)
            root.available_actions = ["factor", "sqrt", "graph"]

            # Expand with reasoning steps
            for action in ["factor", "sqrt"]:
                child_state = MCTSState(
                    state_id=f"root_{action}",
                    features={"approach": action},
                )
                step = ReasoningStep(content=f"Apply {action} method", step_index=0)
                child = root.add_reasoning_child(action, child_state, step)

                # Score with PRM
                trajectory = child.get_trajectory()
                scores = await prm.score_trajectory(trajectory)
                if scores:
                    child.set_prm_score(scores[-1])

            # Simulate some visits
            rollout = HybridRolloutPolicy(heuristic_fn=lambda s: 0.6)

            for child in root.children:
                value = await engine.simulate(child, rollout, max_depth=5)
                engine.backpropagate(child, value)

            # Select best with PRM
            best = root.select_child_with_prm(
                exploration_weight=1.414, prm_weight=0.3
            )

            self.assertIsNotNone(best)
            self.assertIsInstance(best, ReasoningMCTSNode)
            self.assertIsNotNone(best.reasoning.prm_score)

        run_async(run_test())

    def test_multi_step_reasoning_chain(self):
        """Test building a multi-step reasoning chain."""

        async def run_test():
            from src.framework.mcts.core import MCTSState
            from src.framework.mcts.process_reward_model import (
                HeuristicProcessRewardModel,
                ReasoningStep,
            )
            from src.framework.mcts.reasoning_node import ReasoningMCTSNode

            def length_heuristic(step, traj):
                return min(len(step.content) / 100, 1.0)

            prm = HeuristicProcessRewardModel(heuristics=[length_heuristic])

            # Build a chain of reasoning
            root = ReasoningMCTSNode(
                state=MCTSState(state_id="start", features={"problem": "integral"})
            )

            current = root
            steps = [
                "Identify the integrand as x^2",
                "Apply power rule: integral of x^n is x^(n+1)/(n+1)",
                "Compute: x^3/3",
                "Add constant of integration: x^3/3 + C",
            ]

            for i, step_content in enumerate(steps):
                child_state = MCTSState(
                    state_id=f"step_{i}",
                    features={"step": i},
                )
                step = ReasoningStep(content=step_content, step_index=i)
                child = current.add_reasoning_child(
                    action=f"step_{i}",
                    child_state=child_state,
                    reasoning_step=step,
                )
                current = child

            # Get full trajectory from leaf
            trajectory = current.get_trajectory()
            self.assertEqual(len(trajectory.steps), 4)

            # Score trajectory
            scores = await prm.score_trajectory(trajectory)
            self.assertEqual(len(scores), 4)

            # All scores should be positive
            for score in scores:
                self.assertGreater(score.step_score, 0)

        run_async(run_test())


# ============================================================================
# USER JOURNEY TESTS
# ============================================================================


class TestUserJourneyMathProblem(unittest.TestCase):
    """User journey test: Solving a math problem."""

    def test_math_problem_solving_journey(self):
        """Simulate a user solving a math problem with reasoning MCTS."""

        async def run_test():
            from src.framework.mcts.core import MCTSEngine, MCTSState
            from src.framework.mcts.policies import HybridRolloutPolicy
            from src.framework.mcts.process_reward_model import (
                HeuristicProcessRewardModel,
                ReasoningStep,
                ReasoningTrajectory,
            )
            from src.framework.mcts.reasoning_node import ReasoningMCTSNode

            # User's problem
            problem = "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"

            # Setup system
            engine = MCTSEngine(seed=42)

            # Quality heuristic based on math keywords
            def math_quality(step, traj):
                keywords = ["derivative", "d/dx", "power rule", "=", "+", "-"]
                content = step.content.lower()
                matches = sum(1 for kw in keywords if kw in content)
                return min(0.3 + 0.15 * matches, 1.0)

            prm = HeuristicProcessRewardModel(heuristics=[math_quality])

            # Create root
            root = ReasoningMCTSNode(
                state=MCTSState(state_id="problem", features={"query": problem})
            )

            # Possible approaches
            approaches = [
                ("power_rule", "Apply power rule to each term"),
                ("term_by_term", "Differentiate term by term"),
                ("product_rule", "Check if product rule needed"),
            ]

            for action, description in approaches:
                child_state = MCTSState(
                    state_id=f"approach_{action}",
                    features={"approach": action},
                )
                step = ReasoningStep(content=description, step_index=0)
                root.add_reasoning_child(action, child_state, step)

            # Simulate exploration
            rollout = HybridRolloutPolicy(heuristic_fn=lambda s: 0.6)

            for _ in range(10):
                # Select
                if root.children:
                    leaf = root.select_child_with_prm(
                        exploration_weight=1.414, prm_weight=0.3
                    )
                else:
                    leaf = root

                # Simulate
                value = await engine.simulate(leaf, rollout, max_depth=5)

                # Backpropagate
                engine.backpropagate(leaf, value)

            # User gets best approach
            best_child = max(root.children, key=lambda c: c.visits)

            self.assertIsNotNone(best_child)
            self.assertGreater(best_child.visits, 0)

        run_async(run_test())


class TestUserJourneyCodeDebugging(unittest.TestCase):
    """User journey test: Debugging code with reasoning."""

    def test_code_debugging_journey(self):
        """Simulate debugging a code issue with reasoning MCTS."""

        async def run_test():
            from src.framework.mcts.core import MCTSState
            from src.framework.mcts.process_reward_model import (
                HeuristicProcessRewardModel,
                ReasoningStep,
            )
            from src.framework.mcts.reasoning_node import (
                ActorAgent,
                DualAgentMCTSController,
                ReasonerAgent,
                ReasoningMCTSNode,
            )

            # User's code issue
            issue = "Function returns None instead of expected value"

            # Setup dual-agent system
            async def reasoner_fn(prompt, tokens):
                return """
STRATEGY: Check return statements
REASONING: Missing returns are common bugs
CONFIDENCE: 0.9
ESTIMATED_VALUE: 0.85

STRATEGY: Trace execution path
REASONING: Follow the code flow
CONFIDENCE: 0.8
ESTIMATED_VALUE: 0.75
"""

            async def actor_fn(prompt):
                return "Found: return statement missing in else branch"

            reasoner = ReasonerAgent(model_fn=reasoner_fn)
            actor = ActorAgent(model_fn=actor_fn, tools=["code_search", "debugger"])

            controller = DualAgentMCTSController(reasoner=reasoner, actor=actor)

            # Create debugging session
            root = ReasoningMCTSNode(
                state=MCTSState(
                    state_id="debug_start",
                    features={"issue": issue, "language": "python"},
                )
            )

            # Expand with debugging strategies
            children = await controller.expand_with_reasoning(
                node=root, context=f"Debug issue: {issue}", n_strategies=2
            )

            # User selects best strategy
            best = max(children, key=lambda c: c.reasoning.agent_confidence)

            self.assertIsNotNone(best)
            self.assertGreater(best.reasoning.agent_confidence, 0)

        run_async(run_test())


class TestUserJourneyDocumentAnalysis(unittest.TestCase):
    """User journey test: Analyzing a document with reasoning."""

    def test_document_analysis_journey(self):
        """Simulate document analysis with extended thinking."""

        async def run_test():
            from src.framework.mcts.core import MCTSState
            from src.framework.mcts.extended_thinking import (
                ThinkingBudget,
                ThinkingMode,
                ThinkingResult,
            )
            from src.framework.mcts.process_reward_model import ReasoningStep
            from src.framework.mcts.reasoning_node import ReasoningMCTSNode

            # Document to analyze
            document = "Legal contract with multiple clauses..."

            # Compute thinking budget based on complexity
            budget = ThinkingBudget(
                min_tokens=2000,
                max_tokens=32000,
                default_tokens=8000,
            )

            # Document analysis is complex
            tokens = budget.compute_budget(
                depth=0,
                visits=0,
                ucb_score=0.5,
                uncertainty=0.8,  # High uncertainty for legal docs
            )

            mode = budget.get_mode(tokens)

            # Should use extended thinking for complex analysis
            self.assertIn(
                mode, [ThinkingMode.STANDARD, ThinkingMode.EXTENDED, ThinkingMode.DEEP]
            )

            # Create analysis node
            root = ReasoningMCTSNode(
                state=MCTSState(
                    state_id="analyze",
                    features={"document": document[:100], "type": "legal"},
                )
            )

            # Add analysis steps
            steps = [
                "Identify key parties in the contract",
                "Extract main obligations and rights",
                "Find liability and indemnification clauses",
                "Summarize key terms and conditions",
            ]

            for i, step_content in enumerate(steps):
                root.add_reasoning_step(
                    ReasoningStep(content=step_content, step_index=i)
                )

            self.assertEqual(len(root.reasoning.reasoning_steps), 4)

        run_async(run_test())


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_empty_trajectory(self):
        """Test handling of empty trajectory."""
        from src.framework.mcts.process_reward_model import ReasoningTrajectory

        traj = ReasoningTrajectory(query="test")
        text = traj.to_text()

        self.assertIn("test", text)
        self.assertNotIn("Step", text)

    def test_single_step_trajectory(self):
        """Test trajectory with single step."""

        async def run_test():
            from src.framework.mcts.process_reward_model import (
                HeuristicProcessRewardModel,
                ReasoningStep,
                ReasoningTrajectory,
            )

            traj = ReasoningTrajectory(query="simple")
            traj.add_step(ReasoningStep(content="only step", step_index=0))

            prm = HeuristicProcessRewardModel(
                heuristics=[lambda s, t: 0.5]
            )

            scores = await prm.score_trajectory(traj)
            self.assertEqual(len(scores), 1)

        run_async(run_test())

    def test_deep_tree_trajectory(self):
        """Test getting trajectory from very deep tree."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.process_reward_model import ReasoningStep
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        # Build deep tree
        root = ReasoningMCTSNode(
            state=MCTSState(state_id="root", features={})
        )

        current = root
        depth = 50  # Deep tree

        for i in range(depth):
            child_state = MCTSState(state_id=f"level_{i}", features={})
            step = ReasoningStep(content=f"Step at level {i}", step_index=0)
            child = current.add_reasoning_child(
                action=f"action_{i}",
                child_state=child_state,
                reasoning_step=step,
            )
            current = child

        # Get trajectory from deepest node
        trajectory = current.get_trajectory()
        self.assertEqual(len(trajectory.steps), depth)

    def test_zero_budget(self):
        """Test thinking budget edge cases."""
        from src.framework.mcts.extended_thinking import ThinkingBudget, ThinkingMode

        budget = ThinkingBudget(min_tokens=0, max_tokens=100)

        # Should handle zero/minimal cases
        mode = budget.get_mode(0)
        self.assertEqual(mode, ThinkingMode.NONE)

    def test_node_without_parent(self):
        """Test reasoning node without parent."""
        from src.framework.mcts.core import MCTSState
        from src.framework.mcts.reasoning_node import ReasoningMCTSNode

        node = ReasoningMCTSNode(state=MCTSState(state_id="orphan", features={}))

        self.assertIsNone(node.parent)
        self.assertEqual(node.depth, 0)

        trajectory = node.get_trajectory()
        self.assertEqual(len(trajectory.steps), 0)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance(unittest.TestCase):
    """Performance-related tests."""

    def test_prm_caching_effectiveness(self):
        """Test that PRM caching reduces redundant evaluations."""

        async def run_test():
            from src.framework.mcts.process_reward_model import (
                LLMProcessRewardModel,
                ReasoningStep,
                ReasoningTrajectory,
            )

            call_count = 0

            async def counting_evaluate(prompt):
                nonlocal call_count
                call_count += 1
                return {"text": "SCORE: 0.8\nREASONING: Good"}

            prm = LLMProcessRewardModel(
                evaluate_fn=counting_evaluate, cache_size=100
            )

            traj = ReasoningTrajectory(query="test")
            step = ReasoningStep(content="same step", step_index=0)
            traj.add_step(step)

            # Score same step multiple times
            for _ in range(5):
                await prm.score_step(step, traj)

            # Should only have called LLM once due to caching
            self.assertEqual(call_count, 1)
            self.assertEqual(prm.cache_hits, 4)
            self.assertEqual(prm.cache_misses, 1)

        run_async(run_test())

    def test_trajectory_hash_consistency(self):
        """Test that trajectory hashing is consistent."""
        from src.framework.mcts.process_reward_model import (
            ReasoningStep,
            ReasoningTrajectory,
        )

        # Create same trajectory twice
        hashes = []
        for _ in range(2):
            traj = ReasoningTrajectory(query="consistent test")
            traj.add_step(ReasoningStep(content="step 1", step_index=0))
            traj.add_step(ReasoningStep(content="step 2", step_index=1))
            hashes.append(traj.to_hash_key())

        self.assertEqual(hashes[0], hashes[1])


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_all_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        # Unit tests
        TestReasoningStepUnit,
        TestReasoningTrajectoryUnit,
        TestPRMScoreUnit,
        TestThinkingBudgetUnit,
        TestThinkingResultUnit,
        TestTaskComplexityUnit,
        TestHybridSearchConfigUnit,
        TestSearchCandidateUnit,
        TestReasoningMetadataUnit,
        TestReasoningMCTSNodeUnit,
        TestAgentActionUnit,
        # Integration tests
        TestPRMIntegration,
        TestDualAgentIntegration,
        TestHybridSearchIntegration,
        # E2E tests
        TestE2EReasoningWorkflow,
        # User journey tests
        TestUserJourneyMathProblem,
        TestUserJourneyCodeDebugging,
        TestUserJourneyDocumentAnalysis,
        # Edge cases
        TestEdgeCases,
        # Performance tests
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nTests with errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASSED' if success else 'FAILED'}")

    return result


if __name__ == "__main__":
    # Create event loop for async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        run_all_tests()
    finally:
        loop.close()
