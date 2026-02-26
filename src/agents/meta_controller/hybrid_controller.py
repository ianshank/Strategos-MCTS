"""
Hybrid Meta-Controller combining Neural and Assembly routing (Story 2.3).

Provides a weighted ensemble of neural meta-controller predictions and
assembly-based routing heuristics for improved decision-making.
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.framework.assembly import AssemblyConfig, AssemblyFeatures

from .assembly_router import AssemblyRouter, RoutingDecision
from .base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridPrediction:
    """
    Extended prediction with both neural and assembly components.

    Attributes:
        agent: Final selected agent
        confidence: Combined confidence score
        probabilities: Probability distribution over agents
        neural_prediction: Prediction from neural controller
        assembly_decision: Decision from assembly router
        neural_weight: Weight given to neural prediction
        assembly_weight: Weight given to assembly routing
        explanation: Detailed explanation of decision
    """

    agent: str
    confidence: float
    probabilities: dict[str, float]
    neural_prediction: MetaControllerPrediction | None = None
    assembly_decision: RoutingDecision | None = None
    neural_weight: float = 0.6
    assembly_weight: float = 0.4
    explanation: str = ""


class HybridMetaController(AbstractMetaController):
    """
    Hybrid meta-controller combining neural predictions with assembly heuristics.

    Weighted Ensemble:
        final_prob = neural_weight * neural_prob + assembly_weight * assembly_prob

    Default weights: 60% neural, 40% assembly (configurable)

    Benefits:
    - Interpretable: Assembly routing provides clear reasoning
    - Robust: Falls back to assembly rules when neural model uncertain
    - Adaptive: Weights can be tuned based on domain/performance
    - Complementary: Neural learns patterns, assembly provides structure
    """

    def __init__(
        self,
        neural_controller: AbstractMetaController | None = None,
        assembly_config: AssemblyConfig | None = None,
        neural_weight: float = 0.6,
        assembly_weight: float = 0.4,
        name: str = "hybrid",
        seed: int = 42,
        domain: str = "general",
    ):
        """
        Initialize hybrid meta-controller.

        Args:
            neural_controller: Underlying neural meta-controller (RNN/BERT)
            assembly_config: Configuration for assembly routing
            neural_weight: Weight for neural predictions (0.0-1.0)
            assembly_weight: Weight for assembly routing (0.0-1.0)
            name: Controller name
            seed: Random seed
            domain: Domain for assembly concept extraction
        """
        super().__init__(name, seed)

        self.neural_controller = neural_controller
        self.assembly_config = assembly_config or AssemblyConfig()

        # Validate and normalize weights
        total_weight = neural_weight + assembly_weight
        self.neural_weight = neural_weight / total_weight
        self.assembly_weight = assembly_weight / total_weight

        # Initialize assembly router
        self.assembly_router = AssemblyRouter(
            config=self.assembly_config,
            domain=domain,
        )

        # Track current query for assembly feature extraction
        self._current_query: str | None = None
        self._current_assembly_features: AssemblyFeatures | None = None

        # Statistics
        self._stats = {
            "total_predictions": 0,
            "neural_dominant": 0,  # Neural and assembly agreed, neural had higher confidence
            "assembly_dominant": 0,  # Neural and assembly agreed, assembly had higher confidence
            "neural_override": 0,  # Neural overrode assembly (disagreement, neural won)
            "assembly_override": 0,  # Assembly overrode neural (disagreement, assembly won)
            "agreement_rate": 0.0,
        }

        logger.info(
            f"Initialized HybridMetaController: "
            f"neural_weight={self.neural_weight:.2f}, "
            f"assembly_weight={self.assembly_weight:.2f}"
        )

    def set_query_context(self, query: str) -> None:
        """
        Set query context for assembly feature extraction.

        Must be called before predict() to enable assembly routing.

        Args:
            query: Input query text
        """
        self._current_query = query
        self._current_assembly_features = None  # Will be computed on demand

    def predict(  # type: ignore[override]
        self,
        features: MetaControllerFeatures,
        query: str | None = None,
    ) -> HybridPrediction:
        """
        Predict agent using hybrid neural + assembly approach.

        Args:
            features: Meta-controller features (for neural prediction)
            query: Optional query text (for assembly routing)

        Returns:
            HybridPrediction with combined decision and explanation

        Example:
            >>> controller = HybridMetaController(neural_controller)
            >>> controller.set_query_context("Design a microservices architecture")
            >>> prediction = controller.predict(meta_features)
            >>> print(prediction.agent, prediction.explanation)
        """
        self._stats["total_predictions"] += 1

        # Get query (from context or parameter)
        if query is not None:
            self.set_query_context(query)

        # Neural prediction (if controller available)
        neural_pred = None
        if self.neural_controller is not None:
            try:
                neural_pred = self.neural_controller.predict(features)
                logger.debug(f"Neural prediction: {neural_pred.agent} (confidence: {neural_pred.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Neural prediction failed: {e}")

        # Assembly routing (if query available)
        assembly_decision = None
        if self._current_query is not None:
            try:
                assembly_decision = self.assembly_router.route(
                    self._current_query,
                    self._current_assembly_features,
                )
                logger.debug(
                    f"Assembly routing: {assembly_decision.agent} (confidence: {assembly_decision.confidence:.2f})"
                )
            except Exception as e:
                logger.warning(f"Assembly routing failed: {e}")

        # Combine predictions
        if neural_pred is None and assembly_decision is None:
            # Fallback: no information available
            return self._fallback_prediction()
        elif neural_pred is None:
            # Only assembly available
            assert assembly_decision is not None
            return self._assembly_only_prediction(assembly_decision)
        elif assembly_decision is None:
            # Only neural available
            return self._neural_only_prediction(neural_pred)
        else:
            # Both available - weighted ensemble
            return self._ensemble_prediction(neural_pred, assembly_decision)

    def _ensemble_prediction(
        self,
        neural_pred: MetaControllerPrediction,
        assembly_decision: RoutingDecision,
    ) -> HybridPrediction:
        """
        Combine neural and assembly predictions via weighted ensemble.

        Args:
            neural_pred: Neural controller prediction
            assembly_decision: Assembly router decision

        Returns:
            HybridPrediction with combined result
        """
        # Combine probability distributions
        combined_probs = {}

        for agent in self.AGENT_NAMES:
            neural_prob = neural_pred.probabilities.get(agent, 1.0 / len(self.AGENT_NAMES))

            # Get assembly probability (convert decision to distribution)
            assembly_pred = self.assembly_router.to_prediction(assembly_decision)
            assembly_prob = assembly_pred.probabilities.get(agent, 1.0 / len(self.AGENT_NAMES))

            # Weighted combination
            combined_probs[agent] = self.neural_weight * neural_prob + self.assembly_weight * assembly_prob

        # Select agent with highest combined probability
        selected_agent = max(combined_probs, key=lambda k: combined_probs[k])
        confidence = combined_probs[selected_agent]

        # Track agreement/disagreement
        if neural_pred.agent == assembly_decision.agent:
            # Agreement
            if neural_pred.confidence > assembly_decision.confidence:
                self._stats["neural_dominant"] += 1
            else:
                self._stats["assembly_dominant"] += 1
        else:
            # Disagreement
            if selected_agent == neural_pred.agent:
                self._stats["neural_override"] += 1
            else:
                self._stats["assembly_override"] += 1

        # Generate explanation
        explanation = self._generate_explanation(
            selected_agent,
            confidence,
            neural_pred,
            assembly_decision,
        )

        return HybridPrediction(
            agent=selected_agent,
            confidence=confidence,
            probabilities=combined_probs,
            neural_prediction=neural_pred,
            assembly_decision=assembly_decision,
            neural_weight=self.neural_weight,
            assembly_weight=self.assembly_weight,
            explanation=explanation,
        )

    def _neural_only_prediction(
        self,
        neural_pred: MetaControllerPrediction,
    ) -> HybridPrediction:
        """Create prediction from neural controller only."""
        return HybridPrediction(
            agent=neural_pred.agent,
            confidence=neural_pred.confidence,
            probabilities=neural_pred.probabilities,
            neural_prediction=neural_pred,
            assembly_decision=None,
            neural_weight=1.0,
            assembly_weight=0.0,
            explanation=f"Neural-only prediction: {neural_pred.agent} (assembly routing unavailable)",
        )

    def _assembly_only_prediction(
        self,
        assembly_decision: RoutingDecision,
    ) -> HybridPrediction:
        """Create prediction from assembly router only."""
        assembly_pred = self.assembly_router.to_prediction(assembly_decision)

        return HybridPrediction(
            agent=assembly_decision.agent,
            confidence=assembly_decision.confidence,
            probabilities=assembly_pred.probabilities,
            neural_prediction=None,
            assembly_decision=assembly_decision,
            neural_weight=0.0,
            assembly_weight=1.0,
            explanation=f"Assembly-only routing: {assembly_decision.agent} - {assembly_decision.reasoning}",
        )

    def _fallback_prediction(self) -> HybridPrediction:
        """Fallback when neither neural nor assembly available."""
        # Default to HRM (safest general choice)
        logger.warning("Both neural and assembly unavailable - using fallback")

        return HybridPrediction(
            agent="hrm",
            confidence=0.5,
            probabilities={"hrm": 0.5, "trm": 0.25, "mcts": 0.25},
            neural_prediction=None,
            assembly_decision=None,
            neural_weight=0.0,
            assembly_weight=0.0,
            explanation="Fallback to HRM (no prediction sources available)",
        )

    def _generate_explanation(
        self,
        selected_agent: str,
        confidence: float,
        neural_pred: MetaControllerPrediction,
        assembly_decision: RoutingDecision,
    ) -> str:
        """Generate human-readable explanation of hybrid decision."""
        lines = []

        # Decision summary
        lines.append(f"Selected: {selected_agent.upper()} (confidence: {confidence:.2%})")
        lines.append("")

        # Neural component
        lines.append(f"Neural Prediction ({self.neural_weight:.0%} weight):")
        lines.append(f"  Agent: {neural_pred.agent}")
        lines.append(f"  Confidence: {neural_pred.confidence:.2%}")
        neural_probs = ", ".join([f"{k}:{v:.2%}" for k, v in neural_pred.probabilities.items()])
        lines.append(f"  Probabilities: {neural_probs}")
        lines.append("")

        # Assembly component
        lines.append(f"Assembly Routing ({self.assembly_weight:.0%} weight):")
        lines.append(f"  Agent: {assembly_decision.agent}")
        lines.append(f"  Confidence: {assembly_decision.confidence:.2%}")
        lines.append(f"  Reasoning: {assembly_decision.reasoning}")
        lines.append("")

        # Agreement analysis
        if neural_pred.agent == assembly_decision.agent:
            lines.append(f"✓ Agreement: Both selected {neural_pred.agent}")
        else:
            lines.append(
                f"⚠ Disagreement: Neural→{neural_pred.agent}, Assembly→{assembly_decision.agent}, "
                f"Hybrid→{selected_agent}"
            )

        return "\n".join(lines)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get hybrid controller statistics.

        Returns:
            Dictionary of statistics including agreement rates
        """
        stats: dict[str, Any] = dict(self._stats)

        if stats["total_predictions"] > 0:
            total = stats["total_predictions"]
            agreements = stats["neural_dominant"] + stats["assembly_dominant"]
            stats["agreement_rate"] = agreements / total

            stats["neural_win_rate"] = (stats["neural_dominant"] + stats["neural_override"]) / total
            stats["assembly_win_rate"] = (stats["assembly_dominant"] + stats["assembly_override"]) / total

        # Include assembly router stats
        stats["assembly_router"] = self.assembly_router.get_statistics()

        return stats

    def load_model(self, path: str) -> None:
        """
        Load neural controller model.

        Args:
            path: Path to neural model
        """
        if self.neural_controller is not None:
            self.neural_controller.load_model(path)
            logger.info(f"Loaded neural controller model from {path}")
        else:
            logger.warning("No neural controller configured - load_model has no effect")

    def save_model(self, path: str) -> None:
        """
        Save neural controller model.

        Args:
            path: Path to save neural model
        """
        if self.neural_controller is not None:
            self.neural_controller.save_model(path)
            logger.info(f"Saved neural controller model to {path}")
        else:
            logger.warning("No neural controller configured - save_model has no effect")

    def adjust_weights(self, neural_weight: float, assembly_weight: float) -> None:
        """
        Adjust ensemble weights dynamically.

        Args:
            neural_weight: New neural weight
            assembly_weight: New assembly weight
        """
        total = neural_weight + assembly_weight
        self.neural_weight = neural_weight / total
        self.assembly_weight = assembly_weight / total

        logger.info(f"Adjusted weights: neural={self.neural_weight:.2f}, assembly={self.assembly_weight:.2f}")

    def explain_decision(self, verbose: bool = False) -> str:
        """
        Get detailed explanation of last prediction.

        Args:
            verbose: Include assembly feature details

        Returns:
            Formatted explanation string
        """
        if not hasattr(self, "_last_prediction") or self._last_prediction is None:
            return "No predictions made yet"

        pred = self._last_prediction
        explanation = [pred.explanation]

        if verbose and pred.assembly_decision is not None:
            explanation.append("")
            explanation.append("Assembly Feature Details:")
            features = pred.assembly_decision.assembly_features
            explanation.append(self.assembly_router.feature_extractor.explain_features(features))

        return "\n".join(explanation)
