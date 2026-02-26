"""
Assembly-Aware Routing Logic (Story 2.2).

Provides rule-based routing decisions based on assembly theory features,
complementing neural meta-controllers with interpretable heuristics.
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.framework.assembly import (
    AssemblyConfig,
    AssemblyFeatureExtractor,
    AssemblyFeatures,
)

from .base import MetaControllerPrediction

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """
    Result of assembly-based routing decision.

    Attributes:
        agent: Selected agent name ('hrm', 'trm', 'mcts')
        confidence: Confidence in this decision (0.0-1.0)
        reasoning: Human-readable explanation of routing decision
        assembly_features: Assembly features that informed the decision
    """

    agent: str
    confidence: float
    reasoning: str
    assembly_features: AssemblyFeatures


class AssemblyRouter:
    """
    Rule-based routing using Assembly Theory features.

    Routing Rules (default thresholds):
    - assembly_index < 3 AND copy_number > 5 → TRM (simple, reusable patterns)
    - assembly_index < 7 OR decomposability > 0.7 → HRM (decomposable)
    - assembly_index >= 7 → MCTS (complex, needs search)

    Additional heuristics:
    - Very high copy_number (>10) → TRM (lots of reusable patterns)
    - Very low decomposability (<0.3) → MCTS (hard to decompose hierarchically)
    - High technical complexity + medium assembly index → HRM (structured domain knowledge)
    """

    def __init__(
        self,
        config: AssemblyConfig | None = None,
        domain: str = "general",
    ):
        """
        Initialize assembly router.

        Args:
            config: Assembly configuration with routing thresholds
            domain: Domain for concept extraction
        """
        self.config = config or AssemblyConfig()
        self.domain = domain

        # Initialize feature extractor
        self.feature_extractor = AssemblyFeatureExtractor(
            config=self.config,
            domain=domain,
        )

        # Routing thresholds from config
        self.simple_threshold = self.config.routing_simple_threshold
        self.medium_threshold = self.config.routing_medium_threshold

        # Statistics
        self._routing_stats = {
            "total_routes": 0,
            "trm_routes": 0,
            "hrm_routes": 0,
            "mcts_routes": 0,
        }

    def route(
        self,
        query: str,
        features: AssemblyFeatures | None = None,
    ) -> RoutingDecision:
        """
        Route query to appropriate agent based on assembly features.

        Args:
            query: Input query text
            features: Pre-computed assembly features (optional, will extract if None)

        Returns:
            RoutingDecision with selected agent and reasoning

        Example:
            >>> router = AssemblyRouter()
            >>> decision = router.route("What is 2+2?")
            >>> decision.agent
            'trm'
            >>> decision.reasoning
            'Simple query (assembly index: 2) with high reusability → TRM'
        """
        # Extract features if not provided
        if features is None:
            features = self.feature_extractor.extract(query)

        self._routing_stats["total_routes"] += 1

        # Apply routing rules
        agent, confidence, reasoning = self._apply_routing_rules(features)

        # Update stats
        self._routing_stats[f"{agent}_routes"] += 1

        logger.debug(f"Assembly routing: {query[:50]}... → {agent} (confidence: {confidence:.2f})")

        return RoutingDecision(
            agent=agent,
            confidence=confidence,
            reasoning=reasoning,
            assembly_features=features,
        )

    def _apply_routing_rules(
        self,
        features: AssemblyFeatures,
    ) -> tuple[str, float, str]:
        """
        Apply routing heuristics based on assembly features.

        Args:
            features: Assembly features

        Returns:
            Tuple of (agent, confidence, reasoning)
        """
        assembly_idx = features.assembly_index
        copy_num = features.copy_number
        decomp = features.decomposability_score
        tech_comp = features.technical_complexity

        # Rule 1: Very simple queries with high reusability → TRM
        if assembly_idx < self.simple_threshold and copy_num > 5:
            return (
                "trm",
                0.9,
                f"Simple query (assembly index: {assembly_idx:.1f}) with high reusability (copy number: {copy_num:.1f}) → TRM for pattern-based reasoning",
            )

        # Rule 2: Very high reusability (lots of known patterns) → TRM
        if copy_num > 10:
            return (
                "trm",
                0.85,
                f"Very high pattern reusability (copy number: {copy_num:.1f}) → TRM can leverage existing patterns",
            )

        # Rule 3: Simple query → TRM
        if assembly_idx < self.simple_threshold:
            return ("trm", 0.8, f"Simple query (assembly index: {assembly_idx:.1f}) → TRM for direct reasoning")

        # Rule 4: Highly decomposable → HRM
        if decomp > 0.7:
            return (
                "hrm",
                0.9,
                f"Highly decomposable query (score: {decomp:.2f}) with {features.graph_depth} dependency layers → HRM for hierarchical breakdown",
            )

        # Rule 5: Medium complexity with good decomposability → HRM
        if assembly_idx < self.medium_threshold and decomp > 0.4:
            return (
                "hrm",
                0.85,
                f"Medium complexity (assembly index: {assembly_idx:.1f}) with moderate decomposability ({decomp:.2f}) → HRM",
            )

        # Rule 6: High technical complexity with structured dependencies → HRM
        if tech_comp > 0.5 and assembly_idx < self.medium_threshold and features.graph_depth > 3:
            return (
                "hrm",
                0.8,
                f"Technical query (complexity: {tech_comp:.2f}) with structured {features.graph_depth}-layer hierarchy → HRM",
            )

        # Rule 7: Very low decomposability (hard to break down) → MCTS
        if decomp < 0.3:
            return (
                "mcts",
                0.85,
                f"Low decomposability ({decomp:.2f}) - complex interconnected structure → MCTS for exploratory search",
            )

        # Rule 8: High complexity → MCTS
        if assembly_idx >= self.medium_threshold:
            return (
                "mcts",
                0.9,
                f"High complexity (assembly index: {assembly_idx:.1f}) → MCTS for systematic exploration",
            )

        # Default: Medium complexity → HRM (safe default for structured reasoning)
        return (
            "hrm",
            0.6,
            f"Medium complexity (assembly index: {assembly_idx:.1f}) → HRM (default for structured queries)",
        )

    def to_prediction(self, decision: RoutingDecision) -> MetaControllerPrediction:
        """
        Convert RoutingDecision to MetaControllerPrediction format.

        Args:
            decision: Routing decision

        Returns:
            MetaControllerPrediction compatible with existing meta-controllers
        """
        # Create probability distribution
        # High confidence → sharp distribution, low confidence → more uniform
        if decision.confidence > 0.8:
            # Very confident - concentrate probability
            base_prob = (1.0 - decision.confidence) / 2.0
        else:
            # Less confident - more distributed
            base_prob = (1.0 - decision.confidence) / 2.0

        probabilities = {
            "hrm": base_prob,
            "trm": base_prob,
            "mcts": base_prob,
        }
        probabilities[decision.agent] = decision.confidence

        return MetaControllerPrediction(
            agent=decision.agent,
            confidence=decision.confidence,
            probabilities=probabilities,
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary of routing statistics
        """
        stats: dict[str, Any] = dict(self._routing_stats)

        if stats["total_routes"] > 0:
            stats["trm_rate"] = stats["trm_routes"] / stats["total_routes"]
            stats["hrm_rate"] = stats["hrm_routes"] / stats["total_routes"]
            stats["mcts_rate"] = stats["mcts_routes"] / stats["total_routes"]
        else:
            stats["trm_rate"] = 0.0
            stats["hrm_rate"] = 0.0
            stats["mcts_rate"] = 0.0

        return stats

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self._routing_stats = {
            "total_routes": 0,
            "trm_routes": 0,
            "hrm_routes": 0,
            "mcts_routes": 0,
        }

    def explain_routing(self, query: str) -> str:
        """
        Generate detailed explanation of routing decision.

        Args:
            query: Input query

        Returns:
            Multi-line explanation string
        """
        features = self.feature_extractor.extract(query)
        decision = self.route(query, features)

        explanation = [
            "=" * 60,
            "Assembly-Based Routing Analysis",
            "=" * 60,
            "",
            f"Query: {query}",
            "",
            "Assembly Features:",
            f"  - Assembly Index: {features.assembly_index:.1f}",
            f"  - Copy Number: {features.copy_number:.1f}",
            f"  - Decomposability: {features.decomposability_score:.2f}",
            f"  - Graph Depth: {features.graph_depth} layers",
            f"  - Constraint Count: {features.constraint_count}",
            f"  - Concept Count: {features.concept_count}",
            f"  - Technical Complexity: {features.technical_complexity:.2f}",
            "",
            "Routing Decision:",
            f"  - Selected Agent: {decision.agent.upper()}",
            f"  - Confidence: {decision.confidence:.2%}",
            f"  - Reasoning: {decision.reasoning}",
            "",
            "Feature Explanation:",
            self.feature_extractor.explain_features(features),
            "=" * 60,
        ]

        return "\n".join(explanation)
