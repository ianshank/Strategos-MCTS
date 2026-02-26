"""
Policy Network for MCTS Action Selection.

Implements policy networks that learn to select actions directly from states,
enabling fast action selection without expensive LLM calls for routine decisions.
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
class PolicyOutput:
    """Output from policy network forward pass."""

    policy_logits: torch.Tensor  # [batch, action_dim] unnormalized action scores
    state_value: torch.Tensor  # [batch, 1] estimated value of state
    action_probs: torch.Tensor | None = None  # [batch, action_dim] action probabilities
    entropy: torch.Tensor | None = None  # [batch] policy entropy


@dataclass
class ActionSelection:
    """Result of action selection."""

    action: int  # Selected action index
    log_prob: float  # Log probability of selected action
    confidence: float  # Confidence in selection (max probability)
    entropy: float  # Policy entropy (exploration measure)


class PolicyNetwork(nn.Module):
    """
    Policy network for action selection in MCTS.

    Maps state representations to action probabilities, enabling fast action
    selection without LLM calls. Also includes a value head for state evaluation.

    Architecture:
        state -> feature_extractor -> policy_head -> action_logits
                                   -> value_head -> state_value

    Args:
        state_dim: Dimension of state representation
        action_dim: Number of possible actions
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'gelu', 'tanh')
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256, 128]
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Select activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build feature extraction layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Policy head: outputs action logits
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2), self.activation_fn, nn.Linear(prev_dim // 2, action_dim)
        )

        # Value head: estimates state value
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64), self.activation_fn, nn.Dropout(dropout), nn.Linear(64, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.debug(
            "PolicyNetwork initialized: state_dim=%d, action_dim=%d, hidden_dims=%s, dropout=%.2f",
            state_dim,
            action_dim,
            self.hidden_dims,
            dropout,
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, return_probs: bool = False) -> PolicyOutput:
        """
        Forward pass through policy network.

        Args:
            state: [batch, state_dim] state tensor
            return_probs: Whether to compute action probabilities

        Returns:
            PolicyOutput containing policy logits, state value, and optional probs
        """
        # Extract features
        features = self.feature_extractor(state)

        # Compute policy logits and state value
        policy_logits = self.policy_head(features)
        state_value = self.value_head(features)

        # Optionally compute probabilities and entropy
        action_probs = None
        entropy = None

        if return_probs:
            action_probs = F.softmax(policy_logits, dim=-1)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            entropy = -(action_probs * log_probs).sum(dim=-1)

        return PolicyOutput(
            policy_logits=policy_logits, state_value=state_value, action_probs=action_probs, entropy=entropy
        )

    def select_action(
        self, state: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, deterministic: bool = False
    ) -> ActionSelection:
        """
        Select action using the policy network.

        Args:
            state: [state_dim] or [1, state_dim] state tensor
            temperature: Exploration parameter (lower = more greedy)
            top_k: If set, sample from top-k actions only
            deterministic: If True, select argmax action

        Returns:
            ActionSelection with action, log_prob, confidence, entropy
        """
        # Set to eval mode for inference
        was_training = self.training
        self.eval()

        with torch.no_grad():
            # Ensure batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)

            # Forward pass
            output = self.forward(state, return_probs=True)
            policy_logits = output.policy_logits / temperature

            # Apply top-k filtering if requested
            if top_k is not None and top_k < self.action_dim:
                top_k_logits, top_k_indices = torch.topk(policy_logits, top_k, dim=-1)
                # Create mask for top-k
                mask = torch.full_like(policy_logits, float("-inf"))
                mask.scatter_(-1, top_k_indices, top_k_logits)
                policy_logits = mask

            # Compute probabilities
            probs = F.softmax(policy_logits, dim=-1)
            log_probs = F.log_softmax(policy_logits, dim=-1)

            # Select action
            action = torch.argmax(probs, dim=-1).item() if deterministic else torch.multinomial(probs, 1).item()

            # Get log probability and confidence
            log_prob = log_probs[0, action].item()
            confidence = probs[0, action].item()

            # Compute entropy
            entropy = -(probs * log_probs).sum(dim=-1).item()

            # Restore training mode if needed
            if was_training:
                self.train()

            return ActionSelection(action=action, log_prob=log_prob, confidence=confidence, entropy=entropy)

    def get_action_probs(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get probability distribution over actions.

        Args:
            state: [state_dim] or [batch, state_dim] state tensor
            temperature: Temperature for softmax

        Returns:
            [batch, action_dim] action probabilities
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)

            output = self.forward(state)
            policy_logits = output.policy_logits / temperature
            probs = F.softmax(policy_logits, dim=-1)

            if was_training:
                self.train()

            return probs

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities of given actions (for training).

        Args:
            state: [batch, state_dim] state tensor
            actions: [batch] action indices

        Returns:
            log_probs: [batch] log probabilities of actions
            entropy: [batch] policy entropy
        """
        output = self.forward(state, return_probs=True)
        log_probs = F.log_softmax(output.policy_logits, dim=-1)

        # Gather log probs for selected actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        return action_log_probs, output.entropy

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PolicyLoss(nn.Module):
    """
    Combined loss for policy network training.

    Includes:
        - Policy loss (cross-entropy or policy gradient)
        - Value loss (MSE)
        - Entropy regularization (encourages exploration)
        - L2 regularization

    Args:
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss
        entropy_weight: Weight for entropy regularization
        l2_weight: Weight for L2 regularization
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        l2_weight: float = 0.0001,
    ):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.l2_weight = l2_weight

    def forward(
        self,
        policy_output: PolicyOutput,
        target_policy: torch.Tensor | None = None,
        target_actions: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined loss.

        Args:
            policy_output: Output from PolicyNetwork
            target_policy: [batch, action_dim] target policy distribution (for supervised)
            target_actions: [batch] target action indices (for supervised)
            target_values: [batch] target values
            advantages: [batch] advantage estimates (for RL)
            model: Model for L2 regularization

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}

        # Policy loss
        if target_policy is not None:
            # Supervised learning with target distribution
            log_probs = F.log_softmax(policy_output.policy_logits, dim=-1)
            policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
            loss_dict["policy_supervised"] = policy_loss.item()
        elif target_actions is not None:
            if advantages is not None:
                # Policy gradient with advantages
                log_probs = F.log_softmax(policy_output.policy_logits, dim=-1)
                action_log_probs = log_probs.gather(1, target_actions.unsqueeze(-1)).squeeze(-1)
                policy_loss = -(action_log_probs * advantages).mean()
                loss_dict["policy_gradient"] = policy_loss.item()
            else:
                # Supervised learning with action labels
                policy_loss = F.cross_entropy(policy_output.policy_logits, target_actions)
                loss_dict["policy_ce"] = policy_loss.item()
        else:
            policy_loss = torch.tensor(0.0, device=policy_output.policy_logits.device)

        # Value loss
        if target_values is not None:
            value_loss = F.mse_loss(policy_output.state_value.squeeze(-1), target_values)
            loss_dict["value_mse"] = value_loss.item()
        else:
            value_loss = torch.tensor(0.0, device=policy_output.policy_logits.device)

        # Entropy regularization (encourage exploration)
        if policy_output.entropy is not None:
            entropy_loss = -policy_output.entropy.mean()
            loss_dict["entropy"] = -entropy_loss.item()  # Log actual entropy, not loss
        else:
            # Compute entropy manually
            probs = F.softmax(policy_output.policy_logits, dim=-1)
            log_probs = F.log_softmax(policy_output.policy_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            entropy_loss = -entropy
            loss_dict["entropy"] = entropy.item()

        # L2 regularization
        if model is not None and self.l2_weight > 0:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
            loss_dict["l2"] = l2_reg.item()
        else:
            l2_reg = torch.tensor(0.0, device=policy_output.policy_logits.device)

        # Combine losses
        total_loss = (
            self.policy_weight * policy_loss
            + self.value_weight * value_loss
            + self.entropy_weight * entropy_loss
            + self.l2_weight * l2_reg
        )

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


def create_policy_network(
    state_dim: int, action_dim: int, config: dict[str, Any] | None = None, device: str = "cpu"
) -> PolicyNetwork:
    """
    Factory function to create and initialize policy network.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        config: Optional configuration dict
        device: Device to place model on

    Returns:
        Initialized PolicyNetwork
    """
    config = config or {}

    network = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.get("hidden_dims", [256, 256, 128]),
        dropout=config.get("dropout", 0.1),
        use_batch_norm=config.get("use_batch_norm", True),
        activation=config.get("activation", "relu"),
    )

    network.to(device)
    return network
