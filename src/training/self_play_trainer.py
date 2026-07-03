"""Generalized self-play trainer (Phase 5.1).

A domain-agnostic AlphaZero-style self-play → buffer → train loop that composes the
existing neural-MCTS building blocks:

- :class:`~src.framework.mcts.neural_mcts.NeuralMCTS` (+ :class:`SelfPlayCollector`) for
  search and example generation,
- :class:`~src.models.policy_value_net.AlphaZeroLoss` for the policy+value objective,
- torch-safe checkpoints (``state_dict`` only — no pickle).

Domain behavior enters solely via an injected ``initial_state_fn`` returning a
:class:`~src.framework.mcts.neural_mcts.GameState`; no game-specific logic lives here.

**Single-agent support.** Set ``single_agent=True`` for non-adversarial domains (e.g.
reasoning/planning). This propagates to :class:`NeuralMCTS`/``SelfPlayCollector`` so the
two-player negamax assumptions (per-ply value negation, player alternation, sign-flipped
value targets) are bypassed. Defaults to ``False`` (two-player zero-sum, unchanged).
"""

from __future__ import annotations

import collections
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.framework.mcts.neural_mcts import GameState, MCTSExample, NeuralMCTS, SelfPlayCollector
from src.models.policy_value_net import AlphaZeroLoss
from src.observability.logging import get_logger

# NeuralMCTS uses the neural-specific MCTSConfig (num_simulations / c_puct / temperature_*),
# which is distinct from the baseline src.framework.mcts.config.MCTSConfig.
from src.training.system_config import MCTSConfig

logger = get_logger(__name__)


@dataclass
class SelfPlayConfig:
    """Tunables for the self-play loop (named fields — no inline magic numbers)."""

    num_games_per_iteration: int = 10
    batch_size: int = 32
    buffer_capacity: int = 10_000
    learning_rate: float = 1e-3
    value_loss_weight: float = 1.0
    grad_clip: float = 1.0
    train_steps_per_iteration: int = 1


@dataclass
class SelfPlayIterationMetrics:
    """Metrics returned per training iteration."""

    games_played: int
    examples_collected: int
    buffer_size: int
    train_steps: int
    total_loss: float
    policy_loss: float
    value_loss: float


class SelfPlayTrainer:
    """Composable self-play trainer parameterized by domain + network."""

    def __init__(
        self,
        network: nn.Module,
        initial_state_fn: Callable[[], GameState],
        action_space_size: int,
        *,
        mcts_config: MCTSConfig | None = None,
        config: SelfPlayConfig | None = None,
        single_agent: bool = False,
        device: str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or SelfPlayConfig()
        self.mcts_config = mcts_config or MCTSConfig()
        self.device = device
        self.single_agent = single_agent
        self.action_space_size = action_space_size
        self.initial_state_fn = initial_state_fn

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.mcts = NeuralMCTS(self.network, self.mcts_config, device=device, single_agent=single_agent)
        self.collector = SelfPlayCollector(self.mcts, self.mcts_config, action_space_size=action_space_size)
        self.loss_fn = AlphaZeroLoss(value_loss_weight=self.config.value_loss_weight)
        self.optimizer = optimizer or torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.buffer: collections.deque[MCTSExample] = collections.deque(maxlen=self.config.buffer_capacity)

        logger.info(
            "SelfPlayTrainer initialized",
            extra={
                "single_agent": single_agent,
                "action_space_size": action_space_size,
                "device": device,
                "buffer_capacity": self.config.buffer_capacity,
            },
        )

    async def generate_self_play(self, num_games: int | None = None) -> int:
        """Play ``num_games`` self-play games and append examples to the buffer.

        Returns the number of examples collected.
        """
        num_games = num_games if num_games is not None else self.config.num_games_per_iteration
        examples = await self.collector.generate_batch(num_games, self.initial_state_fn)
        self.buffer.extend(examples)
        logger.debug(
            "Self-play games generated",
            extra={"games": num_games, "examples": len(examples), "buffer_size": len(self.buffer)},
        )
        return len(examples)

    def train_step(self) -> dict[str, float] | None:
        """Sample one batch from the buffer and take a single optimizer step.

        Returns the loss dict, or None when the buffer is empty.
        """
        if not self.buffer:
            return None

        batch_size = min(self.config.batch_size, len(self.buffer))
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]

        states = torch.stack([self._as_tensor(ex.state) for ex in batch]).to(self.device)
        target_policy = torch.tensor(np.stack([ex.policy_target for ex in batch]), dtype=torch.float32).to(self.device)
        target_value = torch.tensor([ex.value_target for ex in batch], dtype=torch.float32).to(self.device)

        self.network.train()
        self.optimizer.zero_grad()
        policy_logits, value = self.network(states)
        total_loss, loss_dict = self.loss_fn(policy_logits, value, target_policy, target_value)
        total_loss.backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return {key: float(val) for key, val in loss_dict.items()}

    async def train_iteration(self, num_games: int | None = None) -> SelfPlayIterationMetrics:
        """Run one self-play → train iteration and return metrics."""
        games = num_games if num_games is not None else self.config.num_games_per_iteration
        collected = await self.generate_self_play(games)

        last: dict[str, float] = {"total": 0.0, "policy": 0.0, "value": 0.0}
        steps = 0
        for _ in range(max(1, self.config.train_steps_per_iteration)):
            result = self.train_step()
            if result is None:
                break
            last = result
            steps += 1

        metrics = SelfPlayIterationMetrics(
            games_played=games,
            examples_collected=collected,
            buffer_size=len(self.buffer),
            train_steps=steps,
            total_loss=last["total"],
            policy_loss=last["policy"],
            value_loss=last["value"],
        )
        logger.info(
            "Self-play iteration complete",
            extra={
                "games": metrics.games_played,
                "examples": metrics.examples_collected,
                "train_steps": metrics.train_steps,
                "total_loss": metrics.total_loss,
            },
        )
        return metrics

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        """Save the network weights in the torch-safe (``state_dict``) format.

        When ``metadata`` is provided (e.g. a network-architecture spec), it is written
        to a ``<path>.meta.json`` sidecar so tools like the ``policy-lift`` CLI can
        reconstruct the network without guessing the architecture.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)
        if metadata is not None:
            sidecar = path.with_name(path.name + ".meta.json")
            sidecar.write_text(json.dumps({"schema_version": 1, **metadata}, indent=2, sort_keys=True) + "\n")
        logger.info("Checkpoint saved", extra={"path": str(path), "sidecar": metadata is not None})

    def load_checkpoint(self, path: str | Path) -> None:
        """Load network weights saved by :meth:`save_checkpoint` (``weights_only=True``)."""
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.network.load_state_dict(state)
        logger.info("Checkpoint loaded", extra={"path": str(path)})

    @staticmethod
    def _as_tensor(state_repr: Any) -> torch.Tensor:
        """Coerce a stored state representation into a float tensor."""
        if isinstance(state_repr, torch.Tensor):
            return state_repr.float()
        return torch.as_tensor(np.asarray(state_repr), dtype=torch.float32)


__all__ = ["SelfPlayConfig", "SelfPlayIterationMetrics", "SelfPlayTrainer"]
