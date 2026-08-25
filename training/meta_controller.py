"""
Meta-Controller Training Module

Trains neural routers and ensemble aggregators for optimal multi-agent coordination.
Collects execution traces and learns routing policies from agent performance data.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTrace:
    """Record of a single agent execution."""

    trace_id: str
    task_id: str
    agent_type: str  # hrm, trm, mcts
    task_features: dict[str, float]
    agent_confidence: float
    iteration_count: int
    consensus_score: float
    latency_ms: float
    memory_mb: float
    success: bool
    output_quality: float
    timestamp: float = field(default_factory=lambda: 0.0)


@dataclass
class RoutingDecision:
    """Routing decision made by meta-controller."""

    task_id: str
    selected_agent: str
    confidence: float
    alternative_agents: list[tuple[str, float]]  # (agent, probability)
    reasoning: str


class ExecutionTraceCollector:
    """Collect and manage execution traces from multi-agent system."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize trace collector.

        Args:
            config: Meta-controller configuration
        """
        self.config = config
        self.buffer_size = config.get("buffer_size", 50000)
        self.sample_rate = config.get("sample_rate", 1.0)

        self.traces = deque(maxlen=self.buffer_size)
        self.trace_file = Path("training/models/traces.jsonl")

        logger.info(f"ExecutionTraceCollector initialized with buffer size {self.buffer_size}")

    def add_trace(self, trace: ExecutionTrace) -> None:
        """
        Add new execution trace to buffer.

        Args:
            trace: Execution trace to add
        """
        # Sample rate filtering
        if np.random.random() > self.sample_rate:
            return

        self.traces.append(trace)

        # Persist to disk periodically
        if len(self.traces) % 1000 == 0:
            self._persist_traces()

    def _persist_traces(self) -> None:
        """Write traces to disk."""
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.trace_file, "a") as f:
            for trace in list(self.traces)[-100:]:  # Write last 100
                trace_dict = {
                    "trace_id": trace.trace_id,
                    "task_id": trace.task_id,
                    "agent_type": trace.agent_type,
                    "task_features": trace.task_features,
                    "agent_confidence": trace.agent_confidence,
                    "iteration_count": trace.iteration_count,
                    "consensus_score": trace.consensus_score,
                    "latency_ms": trace.latency_ms,
                    "memory_mb": trace.memory_mb,
                    "success": trace.success,
                    "output_quality": trace.output_quality,
                    "timestamp": trace.timestamp,
                }
                f.write(json.dumps(trace_dict) + "\n")

    def get_training_data(self) -> list[dict[str, Any]]:
        """
        Convert traces to training data format.

        Returns:
            List of training samples
        """
        training_data = []
        agent_map = {"hrm": 0, "trm": 1, "mcts": 2}

        for trace in self.traces:
            features = self._extract_features(trace)
            label = agent_map.get(trace.agent_type, 0)

            # Weight by success and quality
            weight = trace.output_quality if trace.success else 0.1

            training_data.append({"features": features, "label": label, "weight": weight})

        return training_data

    def _extract_features(self, trace: ExecutionTrace) -> np.ndarray:
        """
        Extract feature vector from trace.

        Args:
            trace: Execution trace

        Returns:
            Feature vector
        """
        # Task complexity features
        task_features = trace.task_features
        complexity = task_features.get("complexity", 0.5)
        num_steps = task_features.get("num_steps", 1)
        has_dependencies = task_features.get("has_dependencies", 0)

        # Agent performance features
        agent_confidence = trace.agent_confidence
        iteration_count = trace.iteration_count / 10.0  # Normalize
        consensus_score = trace.consensus_score

        # Resource features
        latency_norm = min(trace.latency_ms / 10000, 1.0)  # Normalize
        memory_norm = min(trace.memory_mb / 1000, 1.0)

        # Success and quality
        success = float(trace.success)
        quality = trace.output_quality

        features = np.array(
            [
                complexity,
                num_steps / 10.0,
                has_dependencies,
                agent_confidence,
                iteration_count,
                consensus_score,
                latency_norm,
                memory_norm,
                success,
                quality,
                # Derived features
                complexity * num_steps,  # Interaction
                agent_confidence * quality,  # Calibration
            ],
            dtype=np.float32,
        )

        return features

    def generate_synthetic_traces(self, num_traces: int = 10000) -> None:
        """
        Generate synthetic traces for initial training.

        Args:
            num_traces: Number of synthetic traces to generate
        """
        logger.info(f"Generating {num_traces} synthetic traces")

        agent_types = ["hrm", "trm", "mcts"]

        for i in range(num_traces):
            # Generate task characteristics
            complexity = np.random.uniform(0.1, 1.0)
            num_steps = np.random.randint(1, 10)

            # Select agent based on task characteristics (ground truth policy)
            if complexity < 0.3:
                best_agent = "hrm"  # Simple decomposition
            elif num_steps > 5:
                best_agent = "mcts"  # Complex search
            else:
                best_agent = "trm"  # Iterative refinement

            # Simulate performance
            if np.random.random() < 0.8:  # 80% follow optimal policy
                agent_type = best_agent
                success = np.random.random() < 0.9
                quality = np.random.uniform(0.7, 1.0) if success else np.random.uniform(0.2, 0.5)
            else:
                agent_type = np.random.choice(agent_types)
                success = np.random.random() < 0.6
                quality = np.random.uniform(0.4, 0.8) if success else np.random.uniform(0.1, 0.4)

            trace = ExecutionTrace(
                trace_id=f"synthetic_{i}",
                task_id=f"task_{i}",
                agent_type=agent_type,
                task_features={
                    "complexity": complexity,
                    "num_steps": num_steps,
                    "has_dependencies": float(np.random.random() < 0.3),
                },
                agent_confidence=np.random.uniform(0.5, 1.0),
                iteration_count=np.random.randint(1, 10),
                consensus_score=np.random.uniform(0.6, 1.0),
                latency_ms=np.random.uniform(100, 5000),
                memory_mb=np.random.uniform(50, 500),
                success=success,
                output_quality=quality,
                timestamp=float(i),
            )

            self.add_trace(trace)

        logger.info(f"Generated {len(self.traces)} synthetic traces")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about collected traces."""
        if not self.traces:
            return {"total_traces": 0}

        agent_counts = {"hrm": 0, "trm": 0, "mcts": 0}
        success_rates = {"hrm": [], "trm": [], "mcts": []}
        qualities = {"hrm": [], "trm": [], "mcts": []}

        for trace in self.traces:
            agent_counts[trace.agent_type] = agent_counts.get(trace.agent_type, 0) + 1
            success_rates[trace.agent_type].append(float(trace.success))
            qualities[trace.agent_type].append(trace.output_quality)

        stats = {
            "total_traces": len(self.traces),
            "agent_distribution": agent_counts,
            "success_rates": {agent: np.mean(rates) if rates else 0 for agent, rates in success_rates.items()},
            "avg_quality": {agent: np.mean(quals) if quals else 0 for agent, quals in qualities.items()},
        }

        return stats


class NeuralRouter(nn.Module):
    """Neural network for agent routing decisions."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize neural router.

        Args:
            config: Router configuration
        """
        super().__init__()

        self.input_features = config.get("input_features", 12)
        hidden_layers = config.get("hidden_layers", [128, 64, 32])
        self.num_agents = config.get("num_agents", 3)
        dropout = config.get("dropout", 0.2)

        # Build network
        layers = []
        prev_dim = self.input_features

        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.num_agents))

        self.network = nn.Sequential(*layers)

        # Confidence head (separate)
        self.confidence_head = nn.Sequential(nn.Linear(prev_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        # Store last hidden for confidence
        self._last_hidden = None

        logger.info(f"NeuralRouter: {self.input_features} -> {hidden_layers} -> {self.num_agents}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through router.

        Args:
            x: Input features (batch_size, input_features)

        Returns:
            Dictionary with logits and confidence
        """
        # Pass through hidden layers
        hidden = x
        for i in range(len(self.network) - 1):
            hidden = self.network[i](hidden)

        # Store for confidence
        self._last_hidden = hidden

        # Final layer for agent probabilities
        logits = self.network[-1](hidden)

        # Confidence score
        confidence = self.confidence_head(hidden)

        return {"logits": logits, "probabilities": F.softmax(logits, dim=-1), "confidence": confidence.squeeze(-1)}

    def route(self, features: torch.Tensor) -> RoutingDecision:
        """
        Make routing decision for a task.

        Args:
            features: Task features

        Returns:
            Routing decision
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features.unsqueeze(0))
            probs = outputs["probabilities"].squeeze()
            confidence = outputs["confidence"].item()

        agent_names = ["hrm", "trm", "mcts"]
        selected_idx = probs.argmax().item()
        selected_agent = agent_names[selected_idx]

        alternatives = [(agent_names[i], probs[i].item()) for i in range(len(agent_names)) if i != selected_idx]
        alternatives.sort(key=lambda x: x[1], reverse=True)

        reasoning = f"Selected {selected_agent} with confidence {confidence:.3f}"

        return RoutingDecision(
            task_id="",
            selected_agent=selected_agent,
            confidence=confidence,
            alternative_agents=alternatives,
            reasoning=reasoning,
        )


class EnsembleAggregator(nn.Module):
    """Aggregate outputs from multiple agents."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize ensemble aggregator.

        Args:
            config: Aggregator configuration
        """
        super().__init__()

        self.method = config.get("method", "weighted_voting")
        # Ensure numeric values are floats (YAML may parse them as strings)
        self.confidence_threshold = float(config.get("confidence_threshold", 0.7))
        self.num_agents = 3

        if self.method == "attention":
            # Attention-based aggregation
            self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1)
            self.output_projection = nn.Linear(256, 256)
        elif self.method == "ensemble":
            # Learned ensemble weights
            self.agent_weights = nn.Parameter(torch.ones(self.num_agents))

        logger.info(f"EnsembleAggregator using {self.method} method")

    def forward(self, agent_outputs: list[torch.Tensor], agent_confidences: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Aggregate agent outputs.

        Args:
            agent_outputs: List of outputs from each agent
            agent_confidences: Confidence scores for each agent

        Returns:
            Aggregated output and metadata
        """
        if self.method == "weighted_voting":
            return self._weighted_voting(agent_outputs, agent_confidences)
        elif self.method == "attention":
            return self._attention_aggregation(agent_outputs, agent_confidences)
        else:  # ensemble
            return self._learned_ensemble(agent_outputs, agent_confidences)

    def _weighted_voting(self, outputs: list[torch.Tensor], confidences: torch.Tensor) -> dict[str, torch.Tensor]:
        """Weighted voting based on confidence."""
        weights = F.softmax(confidences, dim=-1)

        # Stack outputs
        stacked = torch.stack(outputs, dim=0)  # (num_agents, ...)

        # Handle batch dimensions properly
        # If confidences has shape [num_agents], weights has shape [num_agents]
        # If confidences has shape [batch_size, num_agents], weights has shape [batch_size, num_agents]
        if weights.dim() == 1:
            # Single sample: weights shape [num_agents]
            # Reshape to broadcast over stacked dims (num_agents, batch?, features...)
            weight_shape = [-1] + [1] * (stacked.dim() - 1)
            reshaped_weights = weights.view(*weight_shape)
        else:
            # Batched: weights shape [batch_size, num_agents]
            # Transpose to [num_agents, batch_size] then add dims for features
            reshaped_weights = weights.t().unsqueeze(-1)  # [num_agents, batch_size, 1]
            # Expand to match stacked dimensions if needed
            if stacked.dim() > 2:
                reshaped_weights = reshaped_weights.view(self.num_agents, weights.size(0), *([1] * (stacked.dim() - 2)))

        aggregated = (stacked * reshaped_weights).sum(dim=0)

        return {"output": aggregated, "weights": weights, "disagreement": self._compute_disagreement(outputs)}

    def _attention_aggregation(self, outputs: list[torch.Tensor], confidences: torch.Tensor) -> dict[str, torch.Tensor]:
        """Attention-based aggregation."""
        # Reshape for attention
        stacked = torch.stack(outputs, dim=0)  # (num_agents, batch, features)

        # Self-attention
        attn_out, attn_weights = self.attention(stacked, stacked, stacked)

        # Project and aggregate
        aggregated = self.output_projection(attn_out.mean(dim=0))

        return {
            "output": aggregated,
            "attention_weights": attn_weights,
            "disagreement": self._compute_disagreement(outputs),
        }

    def _learned_ensemble(self, outputs: list[torch.Tensor], confidences: torch.Tensor) -> dict[str, torch.Tensor]:
        """Learned ensemble weights."""
        weights = F.softmax(self.agent_weights, dim=0)
        stacked = torch.stack(outputs, dim=0)

        aggregated = (stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)

        return {"output": aggregated, "weights": weights, "disagreement": self._compute_disagreement(outputs)}

    def _compute_disagreement(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Compute disagreement between agent outputs."""
        if len(outputs) < 2:
            return torch.tensor(0.0)

        # Pairwise differences
        disagreements = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diff = (outputs[i] - outputs[j]).abs().mean()
                disagreements.append(diff)

        return torch.stack(disagreements).mean()


class MetaControllerTrainer:
    """Train meta-controller components (router and aggregator)."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize meta-controller trainer.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.mc_config = self.config["meta_controller"]
        # Ensure numeric values are floats (YAML may parse them as strings)
        self.mc_config["router"]["learning_rate"] = float(self.mc_config["router"]["learning_rate"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize components
        self.trace_collector = ExecutionTraceCollector(self.mc_config["trace_collection"])
        self.router = NeuralRouter(self.mc_config["router"]).to(self.device)
        self.aggregator = EnsembleAggregator(self.mc_config["aggregator"]).to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            list(self.router.parameters()) + list(self.aggregator.parameters()),
            lr=self.mc_config["router"]["learning_rate"],
        )

        self.checkpoint_path = Path("training/models/checkpoints/meta_controller.pt")

        logger.info("MetaControllerTrainer initialized")

    def train_router(self, num_epochs: int = 10) -> dict[str, list[float]]:
        """
        Train the neural router on collected traces.

        Args:
            num_epochs: Number of training epochs

        Returns:
            Training history
        """
        logger.info("Training neural router...")

        # Get training data
        training_data = self.trace_collector.get_training_data()

        if not training_data:
            logger.warning("No training data available, generating synthetic traces")
            self.trace_collector.generate_synthetic_traces(10000)
            training_data = self.trace_collector.get_training_data()

        # Create dataset
        dataset = RouterDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        history = {"loss": [], "accuracy": []}

        for epoch in range(num_epochs):
            self.router.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in dataloader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                weights = batch["weights"].to(self.device)

                # Forward pass
                outputs = self.router(features)
                logits = outputs["logits"]

                # Weighted cross-entropy loss
                loss = F.cross_entropy(logits, labels, reduction="none")
                loss = (loss * weights).mean()

                # Add confidence calibration loss
                confidences = outputs["confidence"]
                accuracy_per_sample = (logits.argmax(dim=-1) == labels).float()
                calibration_loss = F.mse_loss(confidences, accuracy_per_sample)
                loss = loss + 0.1 * calibration_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Accuracy
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total

            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            logger.info(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        return history

    def evaluate_router(self, test_dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate router performance.

        Args:
            test_dataloader: Test data loader

        Returns:
            Evaluation metrics
        """
        self.router.eval()

        correct = 0
        total = 0
        confidences = []
        calibration_errors = []

        with torch.no_grad():
            for batch in test_dataloader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.router(features)
                predictions = outputs["logits"].argmax(dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Confidence calibration
                conf = outputs["confidence"]
                acc = (predictions == labels).float()
                calibration_errors.extend((conf - acc).abs().cpu().tolist())
                confidences.extend(conf.cpu().tolist())

        metrics = {
            "accuracy": correct / total,
            "avg_confidence": np.mean(confidences),
            "expected_calibration_error": np.mean(calibration_errors),
            "confidence_std": np.std(confidences),
        }

        logger.info(f"Router evaluation: {metrics}")
        return metrics

    def train_aggregator_end_to_end(
        self, agent_outputs_dataset: Dataset, num_epochs: int = 5
    ) -> dict[str, list[float]]:
        """
        Train aggregator on real agent outputs.

        Args:
            agent_outputs_dataset: Dataset of agent outputs
            num_epochs: Number of epochs

        Returns:
            Training history
        """
        logger.info("Training ensemble aggregator...")

        dataloader = DataLoader(agent_outputs_dataset, batch_size=32, shuffle=True)
        history = {"loss": []}

        for epoch in range(num_epochs):
            self.aggregator.train()
            epoch_loss = 0.0

            for batch in dataloader:
                # Assume batch contains agent outputs and ground truth
                agent_outputs = [batch[f"agent_{i}"].to(self.device) for i in range(3)]
                confidences = batch["confidences"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Aggregate
                result = self.aggregator(agent_outputs, confidences)
                aggregated = result["output"]

                # Loss (MSE for regression, CE for classification)
                loss = F.mse_loss(aggregated, targets)

                # Penalize high disagreement when wrong
                disagreement = result["disagreement"]
                loss = loss + 0.05 * disagreement

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history["loss"].append(avg_loss)

            logger.info(f"Aggregator Epoch {epoch + 1}: Loss={avg_loss:.4f}")

        return history

    def save_checkpoint(self) -> None:
        """Save meta-controller checkpoint."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "router_state": self.router.state_dict(),
            "aggregator_state": self.aggregator.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "trace_statistics": self.trace_collector.get_statistics(),
        }

        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Meta-controller checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self) -> None:
        """Load meta-controller checkpoint."""
        if not self.checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {self.checkpoint_path}")
            return

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)

        self.router.load_state_dict(checkpoint["router_state"])
        self.aggregator.load_state_dict(checkpoint["aggregator_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        logger.info(f"Meta-controller loaded from {self.checkpoint_path}")

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get statistics about routing performance."""
        trace_stats = self.trace_collector.get_statistics()

        # Test routing on sample data
        if trace_stats["total_traces"] > 0:
            training_data = self.trace_collector.get_training_data()[:100]

            self.router.eval()
            routing_correct = 0

            with torch.no_grad():
                for sample in training_data:
                    features = torch.tensor(sample["features"]).to(self.device)
                    outputs = self.router(features.unsqueeze(0))
                    predicted = outputs["logits"].argmax().item()

                    if predicted == sample["label"]:
                        routing_correct += 1

            routing_accuracy = routing_correct / len(training_data) if training_data else 0
        else:
            routing_accuracy = 0

        stats = {**trace_stats, "routing_accuracy": routing_accuracy}

        return stats


class RouterDataset(Dataset):
    """Dataset for router training."""

    def __init__(self, training_data: list[dict[str, Any]]):
        self.data = training_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.data[idx]
        return {
            "features": torch.tensor(sample["features"], dtype=torch.float32),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
            "weights": torch.tensor(sample["weight"], dtype=torch.float32),
        }


if __name__ == "__main__":
    # Test meta-controller training
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Meta-Controller Training Module")

    # Initialize trainer
    trainer = MetaControllerTrainer()

    # Generate synthetic traces
    trainer.trace_collector.generate_synthetic_traces(5000)

    # Get statistics
    stats = trainer.trace_collector.get_statistics()
    logger.info(f"Trace statistics: {stats}")

    # Train router
    history = trainer.train_router(num_epochs=3)
    logger.info(f"Training history: {history}")

    # Test routing decision
    test_features = torch.randn(12)
    decision = trainer.router.route(test_features)
    logger.info(f"Routing decision: {decision}")

    # Test aggregation
    agent_outputs = [torch.randn(256) for _ in range(3)]
    confidences = torch.tensor([0.8, 0.6, 0.9])
    aggregated = trainer.aggregator(agent_outputs, confidences)
    logger.info(f"Aggregation disagreement: {aggregated['disagreement']:.4f}")

    # Save checkpoint
    trainer.save_checkpoint()

    logger.info("Meta-Controller Training Module test complete")
