"""Place student/teacher tensors on DistillationSettings.device."""

from __future__ import annotations

from typing import Protocol

from torch import nn

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


class SupportsNetworkDevice(Protocol):
    """Search tree that exposes ``device`` and ``network`` but is not an ``nn.Module``."""

    device: str
    network: nn.Module


def place_network(
    network: nn.Module,
    device: str,
    *,
    mcts: SupportsNetworkDevice | None = None,
) -> nn.Module:
    """Move ``network`` onto ``device`` and optionally bind a NeuralMCTS occupant."""
    placed = network.to(device)
    if mcts is not None:
        mcts.device = device
        mcts.network = placed
    logger.debug("distillation network placed", device=device)
    return placed
