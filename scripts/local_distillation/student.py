"""Default-off recurrent policy/value student. Separate module from project latent agents.

Feed-forward control is :class:`~src.models.policy_value_net.PolicyValueNetwork`.
The recurrent student duck-types ``forward(x) -> (log_probs, value)`` with shared-weight
residual steps. It does not wrap ``HRMAgent`` / ``TRMAgent``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from scripts.local_distillation.settings import DistillationSettings
from src.models.policy_value_net import PolicyHead, PolicyValueNetwork, ResidualBlock, ValueHead
from src.training.system_config import NeuralNetworkConfig


def c4_network_config(settings: DistillationSettings) -> NeuralNetworkConfig:
    """Smoke-sized C4 ResNet config from distillation settings (not chess/Go defaults)."""
    return NeuralNetworkConfig(
        num_res_blocks=settings.num_res_blocks,
        num_channels=settings.num_channels,
        input_channels=settings.input_channels,
        action_size=settings.action_size,
    )


def build_policy_value_network(settings: DistillationSettings) -> PolicyValueNetwork:
    """Feed-forward C4 leaf net. Sidecar shape is 3×6×7, action_size=7."""
    config = c4_network_config(settings)
    return PolicyValueNetwork(
        config,
        board_size=max(settings.board_rows, settings.board_cols),
        board_rows=settings.board_rows,
        board_cols=settings.board_cols,
    )


class RecurrentPolicyValue(nn.Module):
    """Shared-weight residual recurrence inside ``forward``; one (log_probs, value) out."""

    def __init__(self, settings: DistillationSettings) -> None:
        super().__init__()
        if settings.recurrences < 1:
            raise ValueError("recurrences must be >= 1")
        config = c4_network_config(settings)
        self.recurrences = settings.recurrences
        self.conv_input = nn.Conv2d(
            config.input_channels,
            config.num_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn_input = nn.BatchNorm2d(config.num_channels) if config.use_batch_norm else nn.Identity()
        self.shared_block = ResidualBlock(config.num_channels, config.use_batch_norm)
        board_size = max(settings.board_rows, settings.board_cols)
        self.policy_head = PolicyHead(
            input_channels=config.num_channels,
            policy_conv_channels=config.policy_conv_channels,
            action_size=config.action_size,
            board_size=board_size,
            board_rows=settings.board_rows,
            board_cols=settings.board_cols,
        )
        self.value_head = ValueHead(
            input_channels=config.num_channels,
            value_conv_channels=config.value_conv_channels,
            value_fc_hidden=config.value_fc_hidden,
            board_size=board_size,
            board_rows=settings.board_rows,
            board_cols=settings.board_cols,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn_input(self.conv_input(x)))
        for _ in range(self.recurrences):
            out = self.shared_block(out)
        log_probs = self.policy_head(out)
        value = self.value_head(out)
        return log_probs, value

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_student(settings: DistillationSettings) -> nn.Module:
    """FF ``PolicyValueNetwork`` unless ``recurrent_enabled`` is set (default off)."""
    if settings.recurrent_enabled:
        return RecurrentPolicyValue(settings)
    return build_policy_value_network(settings)
