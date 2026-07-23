"""
Policy-Value Network using ResNet Architecture.

Implements the dual-head neural network used in AlphaZero:
- Policy Head: Outputs action probabilities
- Value Head: Outputs state value estimation

Based on:
- "Mastering Chess and Shogi by Self-Play with a General RL Algorithm" (AlphaZero)
- Deep Residual Learning for Image Recognition (ResNet)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.observability.logging import get_structured_logger

from ..training.system_config import NeuralNetworkConfig

logger = get_structured_logger(__name__)


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and skip connections.

    Architecture:
        Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """

    def __init__(self, channels: int, use_batch_norm: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block transformation."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        out = out + residual
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """
    Policy head for outputting action probabilities.

    Architecture:
        Conv -> BN -> ReLU -> FC -> LogSoftmax
    """

    def __init__(
        self,
        input_channels: int,
        policy_conv_channels: int,
        action_size: int,
        board_size: int = 19,
        board_rows: int | None = None,
        board_cols: int | None = None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, policy_conv_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(policy_conv_channels)

        rows = board_rows if board_rows is not None else board_size
        cols = board_cols if board_cols is not None else board_size
        fc_input_size = policy_conv_channels * rows * cols

        self.fc = nn.Linear(fc_input_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute policy (action probabilities).

        Args:
            x: [batch, channels, height, width]

        Returns:
            Log probabilities: [batch, action_size]
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Flatten spatial dimensions
        out = out.flatten(1)

        # Fully connected layer
        out = self.fc(out)

        # Log probabilities for numerical stability
        return F.log_softmax(out, dim=1)


class ValueHead(nn.Module):
    """
    Value head for estimating state value.

    Architecture:
        Conv -> BN -> ReLU -> FC -> ReLU -> FC -> Tanh
    """

    def __init__(
        self,
        input_channels: int,
        value_conv_channels: int,
        value_fc_hidden: int,
        board_size: int = 19,
        board_rows: int | None = None,
        board_cols: int | None = None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, value_conv_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(value_conv_channels)

        rows = board_rows if board_rows is not None else board_size
        cols = board_cols if board_cols is not None else board_size
        fc_input_size = value_conv_channels * rows * cols

        self.fc1 = nn.Linear(fc_input_size, value_fc_hidden)
        self.fc2 = nn.Linear(value_fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimation.

        Args:
            x: [batch, channels, height, width]

        Returns:
            Value: [batch, 1] in range [-1, 1]
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Flatten spatial dimensions
        out = out.flatten(1)

        # Fully connected layers
        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)

        # Tanh to bound value in [-1, 1]
        return torch.tanh(out)


class PolicyValueNetwork(nn.Module):
    """
    Combined policy-value network with ResNet backbone.

    This is the core neural network used in AlphaZero-style learning.
    """

    def __init__(
        self,
        config: NeuralNetworkConfig,
        board_size: int = 19,
        board_rows: int | None = None,
        board_cols: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.board_size = board_size
        self.board_rows = board_rows or board_size
        self.board_cols = board_cols or board_size

        logger.debug(
            "PolicyValueNetwork initialized",
            input_channels=config.input_channels,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
            action_size=config.action_size,
            board_size=board_size,
            board_rows=self.board_rows,
            board_cols=self.board_cols,
        )

        # Initial convolution
        self.conv_input = nn.Conv2d(
            config.input_channels,
            config.num_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn_input = nn.BatchNorm2d(config.num_channels) if config.use_batch_norm else nn.Identity()

        # Residual blocks (shared feature extractor)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(config.num_channels, config.use_batch_norm) for _ in range(config.num_res_blocks)]
        )

        # Policy head
        self.policy_head = PolicyHead(
            input_channels=config.num_channels,
            policy_conv_channels=config.policy_conv_channels,
            action_size=config.action_size,
            board_size=board_size,
            board_rows=self.board_rows,
            board_cols=self.board_cols,
        )

        # Value head
        self.value_head = ValueHead(
            input_channels=config.num_channels,
            value_conv_channels=config.value_conv_channels,
            value_fc_hidden=config.value_fc_hidden,
            board_size=board_size,
            board_rows=self.board_rows,
            board_cols=self.board_cols,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input state [batch, channels, height, width]

        Returns:
            (policy_logits, value) tuple
            - policy_logits: [batch, action_size] log probabilities
            - value: [batch, 1] state value in [-1, 1]
        """
        # Initial convolution
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)

        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)

        # Split into policy and value heads
        policy = self.policy_head(out)
        value = self.value_head(out)

        return policy, value

    def predict(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode prediction.

        Args:
            state: Input state tensor

        Returns:
            (policy_probs, value) tuple with probabilities (not log)
        """
        with torch.no_grad():
            policy_log_probs, value = self.forward(state)
            policy_probs = torch.exp(policy_log_probs)
            return policy_probs, value

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AlphaZeroLoss(nn.Module):
    """
    Combined loss function for AlphaZero training.

    Loss = (z - v)^2 - π^T log(p) + c||θ||^2

    Where:
    - z: actual game outcome
    - v: value prediction
    - π: MCTS visit count distribution
    - p: policy prediction
    - c: L2 regularization coefficient
    """

    def __init__(self, value_loss_weight: float = 1.0):
        super().__init__()
        self.value_loss_weight = value_loss_weight

    def forward(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute AlphaZero loss.

        Args:
            policy_logits: Predicted policy log probabilities [batch, action_size]
            value: Predicted values [batch, 1]
            target_policy: Target policy from MCTS [batch, action_size]
            target_value: Target value from game outcome [batch, 1]

        Returns:
            (total_loss, loss_dict) tuple
        """
        # Value loss: MSE between predicted and actual outcome
        value_loss = F.mse_loss(value.view_as(target_value), target_value)

        # Policy loss: Cross-entropy between MCTS policy and network policy
        # Target policy is already normalized, policy_logits are log probabilities
        # Prevent 0.0 * -inf = nan when illegal actions are masked with -inf logits
        masked_logits = torch.where(target_policy > 0, policy_logits, torch.zeros_like(policy_logits))
        policy_loss = -torch.sum(target_policy * masked_logits, dim=1).mean()

        # Combined loss
        total_loss = self.value_loss_weight * value_loss + policy_loss

        loss_dict = {
            "total": total_loss.item(),
            "value": value_loss.item(),
            "policy": policy_loss.item(),
        }

        return total_loss, loss_dict


def create_policy_value_network(
    config: NeuralNetworkConfig,
    board_size: int = 19,
    device: str = "cpu",
    board_rows: int | None = None,
    board_cols: int | None = None,
) -> PolicyValueNetwork:
    """
    Factory function to create and initialize policy-value network.

    Args:
        config: Network configuration
        board_size: Board/grid size (for games)
        device: Device to place model on
        board_rows: Optional row count for rectangular boards
        board_cols: Optional col count for rectangular boards

    Returns:
        Initialized PolicyValueNetwork
    """
    network = PolicyValueNetwork(config, board_size=board_size, board_rows=board_rows, board_cols=board_cols)

    # He initialization for convolutional layers
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    network.apply(init_weights)
    network_on_device: PolicyValueNetwork = network.to(device)

    return network_on_device


# Example: Simpler MLP-based policy-value network for non-spatial tasks
class MLPPolicyValueNetwork(nn.Module):
    """
    MLP-based policy-value network for non-spatial state representations.

    Useful for tasks where state is not naturally represented as an image.
    """

    def __init__(
        self,
        state_dim: int,
        action_size: int,
        hidden_dims: list[int] | None = None,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_size = action_size

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Shared feature extractor
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input state [batch, state_dim]

        Returns:
            (policy_log_probs, value) tuple
        """
        # Shared features
        features = self.shared_network(x)

        # Policy
        policy_logits = self.policy_head(features)
        policy_log_probs = F.log_softmax(policy_logits, dim=1)

        # Value
        value = self.value_head(features)

        return policy_log_probs, value

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
