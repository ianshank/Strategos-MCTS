"""Self-play train loop that keeps search in eval() and invalidates the eval cache."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.local_distillation.collector import HygienicCollector, assert_neural_mcts_teacher
from scripts.local_distillation.schema import TrajectoryRow
from scripts.local_distillation.settings import DistillationSettings
from scripts.local_distillation.sidecar import c4_network_architecture
from src.framework.mcts.neural_mcts import GameState
from src.models.policy_value_net import AlphaZeroLoss
from src.observability.logging import get_logger

logger = get_logger(__name__)


class HygienicTrainer:
    """Search under eval(); ``train()`` only inside ``train_step``; cache cleared after."""

    def __init__(
        self,
        network: nn.Module,
        collector: HygienicCollector,
        settings: DistillationSettings,
        rng: np.random.Generator,
        *,
        device: str = "cpu",
        buffer_capacity: int = 10_000,
    ) -> None:
        assert_neural_mcts_teacher(collector.mcts)
        self.network = network
        self.collector = collector
        self.settings = settings
        self.rng = rng
        self.device = device
        self.buffer: deque[TrajectoryRow] = deque(maxlen=buffer_capacity)
        self.loss_fn = AlphaZeroLoss(value_loss_weight=settings.value_loss_weight)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)

    async def generate(self, num_games: int, initial_state_fn: Callable[[], GameState]) -> int:
        rows = await self.collector.generate_batch(num_games, initial_state_fn, self.rng)
        self.buffer.extend(rows)
        return len(rows)

    def train_step(self) -> dict[str, float] | None:
        if not self.buffer:
            return None
        batch_size = min(self.settings.batch_size, len(self.buffer))
        idxs = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[int(i)] for i in idxs]
        states = torch.stack([row.state.float() for row in batch]).to(self.device)
        target_policy = torch.tensor(np.stack([row.policy_target for row in batch]), dtype=torch.float32)
        target_value = torch.tensor([row.value_target for row in batch], dtype=torch.float32)
        target_policy = target_policy.to(self.device)
        target_value = target_value.to(self.device)

        self.network.train()
        self.optimizer.zero_grad()
        log_probs, value = self.network(states)
        total_loss, loss_dict = self.loss_fn(log_probs, value, target_policy, target_value)
        total_loss.backward()
        if self.settings.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.settings.grad_clip)
        self.optimizer.step()
        self.network.eval()
        self.collector.mcts.clear_cache()
        return {key: float(val) for key, val in loss_dict.items()}

    def save_checkpoint(self, path: Path, *, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)
        meta = {
            "network": c4_network_architecture(self.settings),
            "schema_version": self.settings.schema_version,
            **(extra or {}),
        }
        sidecar = path.with_name(path.name + ".meta.json")
        sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        logger.info("distillation checkpoint saved", extra={"path": str(path)})
