"""Meta-controller learning loop: routing-decision collection + train/validate (Phase 5.3).

Collects labeled routing decisions (which agent to use, and how that choice fared) during
self-play / inference, vectorizes the meta-controller features deterministically, and
provides a reproducible supervised train+validate loop that fits any logits-producing
controller model and reports routing accuracy against a majority-class baseline.

The module is decoupled from any specific controller architecture: ``train_and_validate``
accepts any ``nn.Module`` mapping ``(batch, FEATURE_DIM) -> (batch, num_agents)`` logits
(e.g. the RNN/BERT controllers in :mod:`src.agents.meta_controller`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.agents.meta_controller.base import MetaControllerFeatures
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Routing label space (the agents the controller chooses between) and the last-agent vocab.
AGENT_LABELS: tuple[str, ...] = ("hrm", "trm", "mcts")
_LAST_AGENT_VOCAB: tuple[str, ...] = ("none", "hrm", "trm", "mcts")

# Normalization constant for query length (chars) — keeps the feature in ~[0, 1].
_QUERY_LENGTH_NORM = 1000.0

# Deterministic feature width: 9 scalar features + one-hot(last_agent).
FEATURE_DIM = 9 + len(_LAST_AGENT_VOCAB)


def features_to_vector(features: MetaControllerFeatures) -> np.ndarray:
    """Deterministically vectorize MetaControllerFeatures into a fixed-width float array."""
    last_agent_onehot = [1.0 if features.last_agent == name else 0.0 for name in _LAST_AGENT_VOCAB]
    vector = [
        float(features.hrm_confidence),
        float(features.trm_confidence),
        float(features.mcts_value),
        float(features.consensus_score),
        float(features.iteration),
        min(1.0, float(features.query_length) / _QUERY_LENGTH_NORM),
        1.0 if features.has_rag_context else 0.0,
        float(features.rag_relevance_score),
        1.0 if features.is_technical_query else 0.0,
        *last_agent_onehot,
    ]
    return np.asarray(vector, dtype=np.float32)


def agent_to_label(agent: str) -> int:
    """Map an agent name to its class index (raises on unknown agent)."""
    return AGENT_LABELS.index(agent)


@dataclass
class RoutingExample:
    """A single labeled routing decision."""

    features: np.ndarray  # shape (FEATURE_DIM,)
    label: int  # index into AGENT_LABELS
    outcome: float  # realized quality of the decision (e.g. terminal reward)


@dataclass
class MetaControllerTrainingReport:
    """Result of a train+validate run."""

    train_examples: int
    val_examples: int
    epochs: int
    final_train_loss: float
    val_accuracy: float
    baseline_accuracy: float
    accuracy_lift: float  # val_accuracy - baseline_accuracy (absolute)


class MetaControllerDataCollector:
    """Accumulate labeled routing decisions for supervised meta-controller training."""

    def __init__(self) -> None:
        self._examples: list[RoutingExample] = []

    def __len__(self) -> int:
        return len(self._examples)

    def record(self, features: np.ndarray, agent: str, outcome: float) -> None:
        """Record a routing decision from an already-vectorized feature array."""
        vec = np.asarray(features, dtype=np.float32)
        if vec.shape != (FEATURE_DIM,):
            raise ValueError(f"features must have shape ({FEATURE_DIM},), got {vec.shape}")
        self._examples.append(RoutingExample(features=vec, label=agent_to_label(agent), outcome=float(outcome)))

    def record_features(self, features: MetaControllerFeatures, agent: str, outcome: float) -> None:
        """Record a routing decision from MetaControllerFeatures (vectorized internally)."""
        self.record(features_to_vector(features), agent, outcome)

    def to_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X[float32, N x FEATURE_DIM], y[long, N])."""
        if not self._examples:
            raise ValueError("no routing examples collected")
        x = torch.tensor(np.stack([e.features for e in self._examples]), dtype=torch.float32)
        y = torch.tensor([e.label for e in self._examples], dtype=torch.long)
        return x, y


def train_and_validate(
    model: nn.Module,
    collector: MetaControllerDataCollector,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-2,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> MetaControllerTrainingReport:
    """Fit ``model`` on collected routing decisions and validate on a held-out split.

    ``model`` must map ``(batch, FEATURE_DIM) -> (batch, len(AGENT_LABELS))`` logits.
    Reproducible under ``seed``. Reports validation accuracy vs a majority-class baseline.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y = collector.to_dataset()
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least 2 examples to train/validate")

    # Deterministic shuffle + split.
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    x, y = x[perm], y[perm]
    n_val = max(1, int(n * val_fraction))
    x_train, y_train = x[n_val:], y[n_val:]
    x_val, y_val = x[:n_val], y[:n_val]
    if x_train.shape[0] == 0:  # tiny datasets: fall back to training on all rows
        x_train, y_train = x, y

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    # Validation accuracy.
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val).argmax(dim=1)
        val_accuracy = float((val_pred == y_val).float().mean().item())

    # Majority-class baseline (over the training labels) evaluated on the val split.
    majority_label = int(torch.bincount(y_train, minlength=len(AGENT_LABELS)).argmax().item())
    baseline_accuracy = float((y_val == majority_label).float().mean().item())

    report = MetaControllerTrainingReport(
        train_examples=int(x_train.shape[0]),
        val_examples=int(x_val.shape[0]),
        epochs=epochs,
        final_train_loss=final_loss,
        val_accuracy=val_accuracy,
        baseline_accuracy=baseline_accuracy,
        accuracy_lift=val_accuracy - baseline_accuracy,
    )
    logger.info(
        "Meta-controller train/validate complete",
        extra={
            "train_examples": report.train_examples,
            "val_examples": report.val_examples,
            "val_accuracy": report.val_accuracy,
            "baseline_accuracy": report.baseline_accuracy,
            "accuracy_lift": report.accuracy_lift,
        },
    )
    return report


__all__ = [
    "AGENT_LABELS",
    "FEATURE_DIM",
    "MetaControllerDataCollector",
    "MetaControllerTrainingReport",
    "RoutingExample",
    "agent_to_label",
    "features_to_vector",
    "train_and_validate",
]
