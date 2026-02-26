"""
Value Network for Position Evaluation in MCTS.

Implements value networks that estimate expected outcomes from states,
enabling efficient position evaluation without full tree search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ValueOutput:
    """Output from value network forward pass."""

    value: torch.Tensor  # [batch, 1] estimated value
    features: torch.Tensor | None = None  # [batch, hidden_dim] learned features
    uncertainty: torch.Tensor | None = None  # [batch, 1] epistemic uncertainty estimate


class ValueNetwork(nn.Module):
    """
    Value network for position evaluation in MCTS.

    Estimates expected reward from current state, enabling faster tree search
    with fewer simulations. Can output values in different ranges based on
    the output activation function.

    Architecture:
        state -> feature_extractor -> value_head -> value

    Args:
        state_dim: Dimension of state representation
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        output_activation: Output activation ('tanh' for [-1,1], 'sigmoid' for [0,1], None for R)
        use_batch_norm: Whether to use batch normalization
        estimate_uncertainty: Whether to estimate epistemic uncertainty
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        output_activation: str | None = "tanh",
        use_batch_norm: bool = True,
        estimate_uncertainty: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.dropout = dropout
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.estimate_uncertainty = estimate_uncertainty

        # Build feature extraction layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Value head: single scalar output
        self.value_head = nn.Sequential(nn.Linear(prev_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

        # Uncertainty head (optional): estimates epistemic uncertainty
        if estimate_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus(),  # Ensure positive
            )
        else:
            self.uncertainty_head = None

        # Initialize weights
        self.apply(self._init_weights)

        logger.debug(
            "ValueNetwork initialized: state_dim=%d, hidden_dims=%s, output_activation=%s, estimate_uncertainty=%s",
            state_dim,
            self.hidden_dims,
            output_activation,
            estimate_uncertainty,
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, return_features: bool = False) -> ValueOutput:
        """
        Forward pass through value network.

        Args:
            state: [batch, state_dim] state tensor
            return_features: Whether to return learned features

        Returns:
            ValueOutput containing value estimate and optional features/uncertainty
        """
        # Extract features
        features = self.feature_extractor(state)

        # Compute value
        value = self.value_head(features)

        # Apply output activation
        if self.output_activation == "tanh":
            value = torch.tanh(value)
        elif self.output_activation == "sigmoid":
            value = torch.sigmoid(value)
        elif self.output_activation is None:
            pass  # No activation
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}")

        # Compute uncertainty if enabled
        uncertainty = None
        if self.estimate_uncertainty and self.uncertainty_head is not None:
            uncertainty = self.uncertainty_head(features)

        return ValueOutput(value=value, features=features if return_features else None, uncertainty=uncertainty)

    def evaluate(self, state: torch.Tensor) -> float:
        """
        Evaluate a single state.

        Args:
            state: [state_dim] or [1, state_dim] state tensor

        Returns:
            value: Scalar value estimate
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            output = self.forward(state)
            result = output.value.item()

            if was_training:
                self.train()

            return float(result)

    def evaluate_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a batch of states.

        Args:
            states: [batch, state_dim] state tensor

        Returns:
            values: [batch] value estimates
        """
        with torch.no_grad():
            output = self.forward(states)
            return output.value.squeeze(-1)

    def get_confidence(self, state: torch.Tensor) -> float:
        """
        Get confidence in value prediction.

        For values in [0, 1], confidence is based on distance from 0.5.
        For values in [-1, 1], confidence is based on absolute value.

        Args:
            state: [state_dim] state tensor

        Returns:
            confidence: Confidence score in [0, 1]
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            output = self.forward(state)
            value = output.value.item()

            if self.output_activation == "sigmoid":
                # Values in [0, 1]: confidence is distance from 0.5
                confidence = abs(value - 0.5) * 2
            elif self.output_activation == "tanh":
                # Values in [-1, 1]: confidence is absolute value
                confidence = abs(value)
            else:
                # No activation: use uncertainty if available
                if output.uncertainty is not None:
                    # Lower uncertainty = higher confidence
                    uncertainty = output.uncertainty.item()
                    confidence = 1.0 / (1.0 + uncertainty)
                else:
                    # Default to moderate confidence
                    confidence = 0.5

            if was_training:
                self.train()

            return float(confidence)

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ValueLoss(nn.Module):
    """
    Combined loss for value network training.

    Supports multiple loss types:
        - MSE: Standard mean squared error
        - Huber: Robust to outliers
        - Quantile: For distributional value estimation

    Args:
        loss_type: Type of loss ('mse', 'huber', 'quantile')
        huber_delta: Delta parameter for Huber loss
        quantile_tau: Tau parameter for quantile loss
        l2_weight: Weight for L2 regularization
        uncertainty_weight: Weight for uncertainty regularization
    """

    def __init__(
        self,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        quantile_tau: float = 0.5,
        l2_weight: float = 0.0001,
        uncertainty_weight: float = 0.01,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.quantile_tau = quantile_tau
        self.l2_weight = l2_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute value loss.

        Args:
            predictions: [batch, 1] predicted values
            targets: [batch, 1] or [batch] target values
            uncertainty: [batch, 1] uncertainty estimates (optional)
            model: Model for L2 regularization (optional)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Ensure targets have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        # Compute main value loss
        if self.loss_type == "mse":
            value_loss = F.mse_loss(predictions, targets)
        elif self.loss_type == "huber":
            value_loss = F.smooth_l1_loss(predictions, targets, beta=self.huber_delta)
        elif self.loss_type == "quantile":
            errors = targets - predictions
            quantile_loss = torch.max(self.quantile_tau * errors, (self.quantile_tau - 1) * errors)
            value_loss = quantile_loss.mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss_dict = {self.loss_type: value_loss.item()}

        # Uncertainty regularization
        uncertainty_loss = torch.tensor(0.0, device=predictions.device)
        if uncertainty is not None:
            # Negative log likelihood loss with uncertainty
            # Assumes Gaussian: -log N(target | pred, uncertainty^2)
            nll = 0.5 * torch.log(2 * torch.pi * uncertainty.pow(2)) + (predictions - targets).pow(2) / (
                2 * uncertainty.pow(2)
            )
            uncertainty_loss = nll.mean()
            loss_dict["uncertainty"] = uncertainty_loss.item()

        # L2 regularization
        l2_reg = torch.tensor(0.0, device=predictions.device)
        if model is not None and self.l2_weight > 0:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
            loss_dict["l2"] = l2_reg.item()

        # Combine losses
        total_loss = value_loss + self.uncertainty_weight * uncertainty_loss + self.l2_weight * l2_reg

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


class TemporalDifferenceLoss(nn.Module):
    """
    Temporal Difference (TD) loss for value network training.

    Implements TD(λ) learning with bootstrapping from next state values.

    Args:
        gamma: Discount factor
        lambda_: TD(λ) parameter (0 = TD(0), 1 = Monte Carlo)
        loss_type: Base loss type ('mse' or 'huber')
    """

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95, loss_type: str = "mse"):
        super().__init__()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.loss_type = loss_type

    def forward(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute TD loss.

        Args:
            values: [batch] current state values
            rewards: [batch] immediate rewards
            next_values: [batch] next state values
            dones: [batch] episode done flags

        Returns:
            loss: TD loss
            loss_dict: Loss components
        """
        # Compute TD target
        td_target = rewards + self.gamma * next_values * (1 - dones.float())

        # Compute TD error
        td_error = td_target.detach() - values

        # Apply loss function
        if self.loss_type == "mse":
            loss = F.mse_loss(values, td_target.detach())
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(values, td_target.detach())
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss_dict = {"td_loss": loss.item(), "mean_td_error": td_error.abs().mean().item()}

        return loss, loss_dict


def create_value_network(state_dim: int, config: dict[str, Any] | None = None, device: str = "cpu") -> ValueNetwork:
    """
    Factory function to create and initialize value network.

    Args:
        state_dim: State dimension
        config: Optional configuration dict
        device: Device to place model on

    Returns:
        Initialized ValueNetwork
    """
    config = config or {}

    network = ValueNetwork(
        state_dim=state_dim,
        hidden_dims=config.get("hidden_dims", [512, 256, 128]),
        dropout=config.get("dropout", 0.1),
        output_activation=config.get("output_activation", "tanh"),
        use_batch_norm=config.get("use_batch_norm", True),
        estimate_uncertainty=config.get("estimate_uncertainty", False),
    )

    network.to(device)
    return network


class EnsembleValueNetwork(nn.Module):
    """
    Ensemble of value networks for uncertainty estimation.

    Uses multiple value networks and aggregates predictions to
    estimate both aleatoric and epistemic uncertainty.

    Args:
        state_dim: State dimension
        num_networks: Number of networks in ensemble
        network_config: Configuration for individual networks
    """

    def __init__(self, state_dim: int, num_networks: int = 5, network_config: dict[str, Any] | None = None):
        super().__init__()
        self.num_networks = num_networks

        # Create ensemble of networks
        self.networks = nn.ModuleList([create_value_network(state_dim, network_config) for _ in range(num_networks)])

    def forward(self, state: torch.Tensor) -> ValueOutput:
        """
        Forward pass through ensemble.

        Args:
            state: [batch, state_dim] state tensor

        Returns:
            ValueOutput with mean prediction and uncertainty estimate
        """
        # Get predictions from all networks
        predictions = []
        for network in self.networks:
            output = network(state)
            predictions.append(output.value)

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_networks, batch, 1]

        # Compute mean and std
        mean_value = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return ValueOutput(value=mean_value, uncertainty=uncertainty)

    def evaluate(self, state: torch.Tensor) -> tuple[float, float]:
        """
        Evaluate state with uncertainty.

        Args:
            state: [state_dim] state tensor

        Returns:
            value: Mean value estimate
            uncertainty: Standard deviation across ensemble
        """
        # Set all networks to eval
        was_training = [net.training for net in self.networks]
        for net in self.networks:
            net.eval()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            output = self.forward(state)
            value_item = output.value.item() if output.value is not None else 0.0
            uncertainty_item = output.uncertainty.item() if output.uncertainty is not None else 0.0
            result = (value_item, uncertainty_item)

            # Restore training modes
            for net, was_train in zip(self.networks, was_training, strict=True):
                if was_train:
                    net.train()

            return result
