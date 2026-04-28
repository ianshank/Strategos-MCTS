"""
Hybrid LLM-Neural Agent.

Combines LLM reasoning with neural network efficiency, using neural networks
for routine decisions and LLM for complex reasoning tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch

from src.observability.logging import get_logger

from ..models.policy_network import PolicyNetwork
from ..models.value_network import ValueNetwork

logger = get_logger(__name__)


class DecisionSource(Enum):
    """Source of decision in hybrid agent."""

    POLICY_NETWORK = "policy_network"
    VALUE_NETWORK = "value_network"
    LLM = "llm"
    LLM_FALLBACK = "llm_fallback"
    BLENDED = "blended"


@dataclass
class HybridConfig:
    """Configuration for hybrid LLM-neural agent."""

    # Model selection thresholds
    policy_confidence_threshold: float = 0.8
    value_confidence_threshold: float = 0.7

    # Mode selection
    mode: Literal["auto", "neural_only", "llm_only", "adaptive"] = "auto"

    # Cost tracking
    track_costs: bool = True
    neural_cost_per_call: float = 0.000001  # $1e-6 per inference
    llm_cost_per_1k_tokens: float = 0.03  # GPT-4 pricing

    # Performance monitoring
    log_decisions: bool = True
    langsmith_project: str | None = None
    prometheus_enabled: bool = False

    # Adaptive thresholds
    adaptive_threshold_window: int = 100
    adaptive_min_threshold: float = 0.5
    adaptive_max_threshold: float = 0.95

    # Blending
    blend_weights: dict[str, float] = field(default_factory=lambda: {"neural": 0.3, "llm": 0.7})


@dataclass
class DecisionMetadata:
    """Metadata about a decision."""

    source: DecisionSource
    confidence: float | None
    cost: float
    latency_ms: float
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostSavings:
    """Cost savings analysis."""

    actual_cost: float
    hypothetical_llm_cost: float
    savings: float
    savings_percentage: float
    neural_percentage: float
    total_calls: int


class HybridAgent:
    """
    Hybrid agent combining LLM reasoning with neural network efficiency.

    Uses neural networks for routine decisions and LLM for complex reasoning,
    achieving optimal cost-performance tradeoff.

    Args:
        policy_net: Policy network for action selection
        value_net: Value network for position evaluation
        llm_client: LLM client for complex reasoning
        config: Hybrid configuration
    """

    def __init__(
        self,
        policy_net: PolicyNetwork | None = None,
        value_net: ValueNetwork | None = None,
        llm_client: Any | None = None,
        config: HybridConfig | None = None,
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.llm_client = llm_client
        self.config = config or HybridConfig()

        # Move networks to eval mode
        if self.policy_net is not None:
            self.policy_net.eval()
        if self.value_net is not None:
            self.value_net.eval()

        # Statistics tracking
        self.stats: dict[str, Any] = {
            "neural_policy_calls": 0,
            "neural_value_calls": 0,
            "llm_calls": 0,
            "total_neural_cost": 0.0,
            "total_llm_cost": 0.0,
            "neural_failures": 0,
            "decision_history": [],
        }

        # Adaptive threshold tracking
        self.recent_confidences: list[float] = []

        # Setup monitoring
        if self.config.prometheus_enabled:
            self._setup_prometheus()

    def _setup_prometheus(self) -> None:
        """Setup Prometheus metrics."""
        try:
            from prometheus_client import Counter, Histogram

            self.neural_predictions = Counter(
                "hybrid_agent_neural_predictions_total", "Total neural network predictions", ["model_type"]
            )

            self.llm_calls_metric = Counter("hybrid_agent_llm_calls_total", "Total LLM API calls", ["reason"])

            self.prediction_latency = Histogram(
                "hybrid_agent_prediction_latency_seconds",
                "Prediction latency",
                ["source"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            )

            self.model_confidence = Histogram(
                "hybrid_agent_model_confidence",
                "Model confidence scores",
                ["model_type"],
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            )

        except ImportError:
            logger.warning("prometheus_client not installed, disabling Prometheus metrics")

    async def select_action(
        self, state: torch.Tensor, context: dict[str, Any] | None = None, temperature: float = 1.0
    ) -> tuple[int, DecisionMetadata]:
        """
        Select action using hybrid approach.

        Args:
            state: Current state representation
            context: Optional context (conversation history, metadata, etc.)
            temperature: Temperature for action selection

        Returns:
            action: Selected action
            metadata: Decision metadata (source, confidence, cost, etc.)
        """
        if self.config.mode == "llm_only":
            return await self._llm_select_action(state, context)

        if self.config.mode == "neural_only":
            return self._neural_select_action(state, temperature)

        # Auto or adaptive mode: try neural first, fall back to LLM if uncertain
        action, metadata = self._neural_select_action(state, temperature)

        # Determine threshold (adaptive or fixed)
        threshold = self._get_confidence_threshold()

        if metadata.confidence is not None and metadata.confidence < threshold:
            # Neural network is uncertain, query LLM
            logger.info(
                f"Neural confidence {metadata.confidence:.3f} below threshold {threshold:.3f}, falling back to LLM"
            )

            action, llm_metadata = await self._llm_select_action(state, context)
            llm_metadata.source = DecisionSource.LLM_FALLBACK
            llm_metadata.additional_info["neural_confidence"] = metadata.confidence

            return action, llm_metadata

        return action, metadata

    def _neural_select_action(self, state: torch.Tensor, temperature: float = 1.0) -> tuple[int, DecisionMetadata]:
        """Select action using policy network."""
        if self.policy_net is None:
            raise ValueError("Policy network not initialized")

        start_time = time.time()

        # Get action from policy network
        action_selection = self.policy_net.select_action(state, temperature=temperature)
        action = action_selection.action
        confidence = action_selection.confidence

        latency_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.stats["neural_policy_calls"] += 1
        self.stats["total_neural_cost"] += self.config.neural_cost_per_call

        # Track confidence for adaptive thresholds
        if self.config.mode == "adaptive":
            self.recent_confidences.append(confidence)
            if len(self.recent_confidences) > self.config.adaptive_threshold_window:
                self.recent_confidences.pop(0)

        # Log to Prometheus if enabled
        if self.config.prometheus_enabled:
            self.neural_predictions.labels(model_type="policy").inc()
            self.prediction_latency.labels(source="policy_network").observe(latency_ms / 1000)
            self.model_confidence.labels(model_type="policy").observe(confidence)

        metadata = DecisionMetadata(
            source=DecisionSource.POLICY_NETWORK,
            confidence=confidence,
            cost=self.config.neural_cost_per_call,
            latency_ms=latency_ms,
            additional_info={"entropy": action_selection.entropy, "log_prob": action_selection.log_prob},
        )

        if self.config.log_decisions:
            self.stats["decision_history"].append(metadata)

        return action, metadata

    async def _llm_select_action(
        self, state: torch.Tensor, context: dict[str, Any] | None = None
    ) -> tuple[int, DecisionMetadata]:
        """Select action using LLM."""
        if self.llm_client is None:
            raise ValueError("LLM client not initialized")

        start_time = time.time()

        # Convert state to prompt
        prompt = self._state_to_prompt(state, context)

        # Query LLM
        response = await self.llm_client.generate(prompt)

        latency_ms = (time.time() - start_time) * 1000

        # Parse action from response
        action = self._parse_action(response)

        # Estimate cost
        num_tokens = len(prompt.split()) + len(response.get("text", "").split())
        cost = (num_tokens / 1000) * self.config.llm_cost_per_1k_tokens

        # Update statistics
        self.stats["llm_calls"] += 1
        self.stats["total_llm_cost"] += cost

        # Log to Prometheus if enabled
        if self.config.prometheus_enabled:
            self.llm_calls_metric.labels(reason="primary").inc()
            self.prediction_latency.labels(source="llm").observe(latency_ms / 1000)

        metadata = DecisionMetadata(
            source=DecisionSource.LLM,
            confidence=None,  # LLMs don't provide calibrated confidence
            cost=cost,
            latency_ms=latency_ms,
            additional_info={"response": response, "num_tokens": num_tokens},
        )

        if self.config.log_decisions:
            self.stats["decision_history"].append(metadata)

        return action, metadata

    async def evaluate_position(
        self, state: torch.Tensor, use_llm_if_uncertain: bool = True
    ) -> tuple[float, DecisionMetadata]:
        """
        Evaluate position using hybrid approach.

        Args:
            state: Current state
            use_llm_if_uncertain: Whether to fall back to LLM if uncertain

        Returns:
            value: Position evaluation
            metadata: Source and confidence information
        """
        if self.value_net is None:
            if self.llm_client is not None and use_llm_if_uncertain:
                return await self._llm_evaluate(state)
            else:
                raise ValueError("Value network and LLM client not initialized")

        start_time = time.time()

        # Get neural network evaluation
        value = self.value_net.evaluate(state)
        confidence = self.value_net.get_confidence(state)

        latency_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.stats["neural_value_calls"] += 1
        self.stats["total_neural_cost"] += self.config.neural_cost_per_call

        # Log to Prometheus if enabled
        if self.config.prometheus_enabled:
            self.neural_predictions.labels(model_type="value").inc()
            self.prediction_latency.labels(source="value_network").observe(latency_ms / 1000)
            self.model_confidence.labels(model_type="value").observe(confidence)

        metadata = DecisionMetadata(
            source=DecisionSource.VALUE_NETWORK,
            confidence=confidence,
            cost=self.config.neural_cost_per_call,
            latency_ms=latency_ms,
        )

        # Fall back to LLM if uncertain
        if use_llm_if_uncertain and confidence < self.config.value_confidence_threshold:
            llm_value, llm_meta = await self._llm_evaluate(state)

            # Blend neural and LLM estimates
            neural_weight = self.config.blend_weights.get("neural", 0.3)
            llm_weight = self.config.blend_weights.get("llm", 0.7)

            blended_value = neural_weight * value + llm_weight * llm_value

            metadata.source = DecisionSource.BLENDED
            metadata.additional_info = {
                "llm_value": llm_value,
                "neural_value": value,
                "neural_confidence": confidence,
                "blend_weights": {"neural": neural_weight, "llm": llm_weight},
            }
            metadata.cost += llm_meta.cost

            return blended_value, metadata

        return value, metadata

    async def _llm_evaluate(self, state: torch.Tensor) -> tuple[float, DecisionMetadata]:
        """Evaluate position using LLM."""
        if self.llm_client is None:
            raise ValueError("LLM client not initialized")

        start_time = time.time()

        # Convert state to prompt
        prompt = self._state_to_evaluation_prompt(state)

        # Query LLM
        response = await self.llm_client.generate(prompt)

        latency_ms = (time.time() - start_time) * 1000

        # Parse value from response
        value = self._parse_value(response)

        # Estimate cost
        num_tokens = len(prompt.split()) + len(response.get("text", "").split())
        cost = (num_tokens / 1000) * self.config.llm_cost_per_1k_tokens

        # Update statistics
        self.stats["llm_calls"] += 1
        self.stats["total_llm_cost"] += cost

        metadata = DecisionMetadata(source=DecisionSource.LLM, confidence=None, cost=cost, latency_ms=latency_ms)

        return value, metadata

    def _get_confidence_threshold(self) -> float:
        """Get current confidence threshold (adaptive or fixed)."""
        if self.config.mode == "adaptive" and len(self.recent_confidences) > 10:
            # Adaptive threshold based on recent performance
            avg_confidence = sum(self.recent_confidences) / len(self.recent_confidences)

            # Adjust threshold to maintain good coverage
            # If average confidence is high, increase threshold
            # If average confidence is low, decrease threshold
            threshold = min(
                self.config.adaptive_max_threshold, max(self.config.adaptive_min_threshold, avg_confidence * 1.1)
            )

            logger.debug(f"Adaptive threshold: {threshold:.3f} (avg confidence: {avg_confidence:.3f})")
            return threshold
        else:
            return self.config.policy_confidence_threshold

    def _state_to_prompt(self, state: torch.Tensor, context: dict[str, Any] | None = None) -> str:
        """Convert state to LLM prompt for action selection."""
        # This is a placeholder - should be implemented based on domain
        state_desc = f"State: {state.tolist()}"
        if context:
            state_desc += f"\nContext: {context}"
        return f"Given the following state, select the best action:\n{state_desc}\n\nAction:"

    def _state_to_evaluation_prompt(self, state: torch.Tensor) -> str:
        """Convert state to LLM prompt for position evaluation."""
        state_desc = f"State: {state.tolist()}"
        return f"Evaluate the following position (return value between -1 and 1):\n{state_desc}\n\nValue:"

    def _parse_action(self, response: dict[str, Any]) -> int:
        """Parse action from LLM response."""
        # Placeholder - should be implemented based on LLM response format
        text = response.get("text", "0")
        try:
            return int(text.strip().split()[0])
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse action from response: {text}")
            return 0

    def _parse_value(self, response: dict[str, Any]) -> float:
        """Parse value from LLM response."""
        # Placeholder - should be implemented based on LLM response format
        text = response.get("text", "0.0")
        try:
            return float(text.strip().split()[0])
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse value from response: {text}")
            return 0.0

    def get_cost_savings(self) -> CostSavings:
        """
        Calculate cost savings from hybrid approach.

        Compares actual costs to hypothetical pure LLM approach.
        """
        total_calls = self.stats["neural_policy_calls"] + self.stats["neural_value_calls"] + self.stats["llm_calls"]

        actual_cost = self.stats["total_neural_cost"] + self.stats["total_llm_cost"]

        # Estimate cost if all calls were LLM
        avg_llm_cost = (
            self.stats["total_llm_cost"] / self.stats["llm_calls"]
            if self.stats["llm_calls"] > 0
            else 0.05  # Default estimate
        )

        hypothetical_llm_cost = total_calls * avg_llm_cost

        savings = hypothetical_llm_cost - actual_cost
        savings_pct = (savings / hypothetical_llm_cost * 100) if hypothetical_llm_cost > 0 else 0

        neural_calls = self.stats["neural_policy_calls"] + self.stats["neural_value_calls"]
        neural_pct = (neural_calls / total_calls * 100) if total_calls > 0 else 0

        return CostSavings(
            actual_cost=actual_cost,
            hypothetical_llm_cost=hypothetical_llm_cost,
            savings=savings,
            savings_percentage=savings_pct,
            neural_percentage=neural_pct,
            total_calls=total_calls,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        cost_savings = self.get_cost_savings()

        return {
            "calls": {
                "neural_policy": self.stats["neural_policy_calls"],
                "neural_value": self.stats["neural_value_calls"],
                "llm": self.stats["llm_calls"],
                "total": cost_savings.total_calls,
            },
            "costs": {
                "neural": self.stats["total_neural_cost"],
                "llm": self.stats["total_llm_cost"],
                "total": cost_savings.actual_cost,
            },
            "cost_savings": {
                "actual_cost": cost_savings.actual_cost,
                "hypothetical_llm_cost": cost_savings.hypothetical_llm_cost,
                "savings": cost_savings.savings,
                "savings_percentage": cost_savings.savings_percentage,
            },
            "usage": {
                "neural_percentage": cost_savings.neural_percentage,
                "llm_percentage": 100 - cost_savings.neural_percentage,
            },
            "failures": {"neural": self.stats["neural_failures"]},
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = {
            "neural_policy_calls": 0,
            "neural_value_calls": 0,
            "llm_calls": 0,
            "total_neural_cost": 0.0,
            "total_llm_cost": 0.0,
            "neural_failures": 0,
            "decision_history": [],
        }
        self.recent_confidences = []


def create_hybrid_agent(
    policy_net: PolicyNetwork | None = None,
    value_net: ValueNetwork | None = None,
    llm_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> HybridAgent:
    """
    Factory function to create hybrid agent.

    Args:
        policy_net: Policy network
        value_net: Value network
        llm_client: LLM client
        config: Configuration dict

    Returns:
        HybridAgent instance
    """
    hybrid_config = HybridConfig(**config) if config else HybridConfig()
    return HybridAgent(policy_net=policy_net, value_net=value_net, llm_client=llm_client, config=hybrid_config)
