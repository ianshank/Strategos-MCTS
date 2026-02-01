"""
MCTS (Monte Carlo Tree Search) module for multi-agent framework.

Provides deterministic, testable MCTS with:
- Progressive widening for controlled branching
- Simulation result caching
- Configurable selection and rollout policies
- Experiment tracking and analysis

Modern Reasoning Model Integration:
- Process Reward Models (PRMs) for step-level evaluation
- Extended Thinking with adaptive token budgets
- Hybrid search combining parallel and serial scaling
- Dual-agent architecture (Reasoner + Actor)
- Enhanced LangGraph integration

References:
- "Let's Verify Step by Step" (OpenAI)
- "ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search"
- "Scaling Test-Time Compute Optimally"
"""

from .config import MCTSConfig, create_preset_config
from .core import MCTSEngine, MCTSNode, MCTSState
from .experiments import ExperimentResult, ExperimentTracker
from .policies import RolloutPolicy, SelectionPolicy, ucb1

# Process Reward Model components
from .process_reward_model import (
    EnsemblePRM,
    HeuristicProcessRewardModel,
    LLMProcessRewardModel,
    MonteCarloProcessRewardModel,
    PRMEnhancedMCTSConfig,
    PRMMCTSIntegration,
    PRMScore,
    PRMTrainingCollector,
    PRMTrainingExample,
    ProcessRewardModel,
    ReasoningStep,
    ReasoningTrajectory,
)

# Extended Thinking components
from .extended_thinking import (
    AdaptiveThinkingRouter,
    ClaudeExtendedThinkingEvaluator,
    ExtendedThinkingEvaluator,
    ParallelThinkingEvaluator,
    TaskComplexity,
    ThinkingBudget,
    ThinkingEnhancedMCTSEvaluator,
    ThinkingMode,
    ThinkingResult,
)

# Hybrid Search components
from .hybrid_search import (
    HybridMCTSSearch,
    HybridSearchConfig,
    HybridSearchResult,
    SearchCandidate,
    SearchPhase,
    VerifiedHybridSearch,
)

# Reasoning Node and Dual-Agent components
from .reasoning_node import (
    ActorAgent,
    AgentAction,
    DualAgentMCTSController,
    ReasonerAgent,
    ReasoningMCTSNode,
    ReasoningMetadata,
)

# LangGraph Integration
from .reasoning_graph import (
    EnhancedTreeState,
    ReasoningGraphBuilder,
    ReasoningGraphConfig,
    create_reasoning_graph,
    run_reasoning_search,
)

__all__ = [
    # Core MCTS
    "MCTSNode",
    "MCTSState",
    "MCTSEngine",
    "ucb1",
    "RolloutPolicy",
    "SelectionPolicy",
    "MCTSConfig",
    "create_preset_config",
    "ExperimentTracker",
    "ExperimentResult",
    # Process Reward Model
    "ProcessRewardModel",
    "LLMProcessRewardModel",
    "MonteCarloProcessRewardModel",
    "HeuristicProcessRewardModel",
    "EnsemblePRM",
    "PRMScore",
    "PRMMCTSIntegration",
    "PRMEnhancedMCTSConfig",
    "PRMTrainingCollector",
    "PRMTrainingExample",
    "ReasoningStep",
    "ReasoningTrajectory",
    # Extended Thinking
    "ThinkingMode",
    "ThinkingBudget",
    "ThinkingResult",
    "TaskComplexity",
    "ExtendedThinkingEvaluator",
    "ClaudeExtendedThinkingEvaluator",
    "ParallelThinkingEvaluator",
    "AdaptiveThinkingRouter",
    "ThinkingEnhancedMCTSEvaluator",
    # Hybrid Search
    "HybridSearchConfig",
    "HybridSearchResult",
    "SearchCandidate",
    "SearchPhase",
    "HybridMCTSSearch",
    "VerifiedHybridSearch",
    # Reasoning Node and Dual-Agent
    "ReasoningMCTSNode",
    "ReasoningMetadata",
    "AgentAction",
    "ReasonerAgent",
    "ActorAgent",
    "DualAgentMCTSController",
    # LangGraph Integration
    "EnhancedTreeState",
    "ReasoningGraphConfig",
    "ReasoningGraphBuilder",
    "create_reasoning_graph",
    "run_reasoning_search",
]
