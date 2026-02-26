"""
Hierarchical Reasoning Model (HRM) Agent.

Implements the HRM architecture with:
- H-Module: High-level planning and decomposition
- L-Module: Low-level execution and refinement
- Adaptive Computation Time (ACT) for dynamic depth
- Halting mechanism based on confidence thresholds

Based on: "Hierarchical Reasoning for Compositional Generalization"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.system_config import HRMConfig

logger = logging.getLogger(__name__)


@dataclass
class SubProblem:
    """Represents a decomposed subproblem in the hierarchy."""

    level: int  # Hierarchy level (0 = root, higher = more abstract)
    description: str  # Natural language description
    state: torch.Tensor  # Latent state representation
    parent_id: int | None = None  # Parent subproblem ID
    confidence: float = 0.0  # Confidence in this decomposition


@dataclass
class HRMOutput:
    """Output from HRM processing."""

    final_state: torch.Tensor  # Final processed state
    subproblems: list[SubProblem]  # Hierarchical decomposition
    halt_step: int  # Step at which halting occurred
    total_ponder_cost: float  # Total computation cost (for training)
    convergence_path: list[float]  # Confidence at each step


class AdaptiveComputationTime(nn.Module):
    """
    Adaptive Computation Time (ACT) mechanism for dynamic depth.

    Allows the model to "ponder" longer on difficult problems by
    dynamically adjusting the number of processing steps.
    """

    def __init__(self, hidden_dim: int, epsilon: float = 0.01):
        super().__init__()
        self.epsilon = epsilon

        # Halting unit: predicts probability of halting
        self.halt_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Compute halting probabilities.

        Args:
            hidden_states: [batch, seq, hidden_dim]

        Returns:
            halt_probs: [batch, seq] probability of halting
            ponder_cost: Scalar cost for training
        """
        # Compute halting probabilities
        halt_logits = self.halt_fc(hidden_states)  # [batch, seq, 1]
        halt_probs = halt_logits.squeeze(-1)  # [batch, seq]

        # Ponder cost is the expected number of steps
        ponder_cost = halt_probs.sum(dim=-1).mean()

        return halt_probs, ponder_cost


class HModule(nn.Module):
    """
    H-Module: High-level planning and abstract reasoning.

    Responsible for:
    - Decomposing problems into subproblems
    - Abstract planning and strategy
    - Coordinating L-module executions
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        # Multi-head self-attention for relational reasoning
        self.attention = nn.MultiheadAttention(
            embed_dim=config.h_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.h_dim, config.h_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.h_dim * 4, config.h_dim),
            nn.Dropout(config.dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.h_dim)
        self.norm2 = nn.LayerNorm(config.h_dim)

        # Decomposition head: outputs subproblem structure
        self.decompose_head = nn.Sequential(
            nn.Linear(config.h_dim, config.h_dim),
            nn.ReLU(),
            nn.Linear(config.h_dim, config.h_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through high-level reasoning.

        Args:
            x: [batch, seq, h_dim] input tensor

        Returns:
            [batch, seq, h_dim] processed tensor
        """
        # Self-attention for relational reasoning
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward processing
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

    def decompose(self, x: torch.Tensor) -> torch.Tensor:
        """Generate subproblem representations."""
        return self.decompose_head(x)


class LModule(nn.Module):
    """
    L-Module: Low-level execution and concrete operations.

    Responsible for:
    - Executing concrete operations
    - Processing individual subproblems
    - Generating intermediate results
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        # Projection from H-module to L-module dimension
        self.h_to_l = nn.Linear(config.h_dim, config.l_dim)

        # GRU for sequential processing
        self.gru = nn.GRU(
            input_size=config.l_dim,
            hidden_size=config.l_dim,
            num_layers=config.num_l_layers,
            dropout=config.dropout if config.num_l_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.l_dim, config.l_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.l_dim * 2, config.l_dim),
        )

        # Back-projection to H-module dimension
        self.l_to_h = nn.Linear(config.l_dim, config.h_dim)

    def forward(self, x: torch.Tensor, h_context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute low-level processing.

        Args:
            x: [batch, seq, h_dim] input from H-module
            h_context: Optional hidden state

        Returns:
            output: [batch, seq, l_dim] processed output
            l_to_h: [batch, seq, h_dim] back-projection to H-module
        """
        # Project to L-module dimension
        x_l = self.h_to_l(x)

        # Sequential processing
        gru_out, _ = self.gru(x_l, h_context)

        # Output processing
        output = self.output_proj(gru_out)

        # Back-project to H-module dimension for feedback
        feedback = self.l_to_h(output)

        return output, feedback


class HRMAgent(nn.Module):
    """
    Complete Hierarchical Reasoning Model agent.

    Combines H-module and L-module with ACT for adaptive computation.
    """

    def __init__(self, config: HRMConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        logger.debug(
            "HRM agent initialized: h_dim=%d, l_dim=%d, num_h_layers=%d, num_l_layers=%d, device=%s",
            config.h_dim,
            config.l_dim,
            config.num_h_layers,
            config.num_l_layers,
            device,
        )

        # Input embedding
        self.input_proj = nn.Linear(config.h_dim, config.h_dim)

        # Core modules
        self.h_module = nn.ModuleList([HModule(config) for _ in range(config.num_h_layers)])

        self.l_module = LModule(config)

        # Adaptive computation time
        self.act = AdaptiveComputationTime(config.h_dim, config.ponder_epsilon)

        # State integration
        self.integrate = nn.Sequential(
            nn.Linear(config.h_dim * 2, config.h_dim),
            nn.LayerNorm(config.h_dim),
            nn.GELU(),
        )

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        max_steps: int | None = None,
        return_decomposition: bool = False,
    ) -> HRMOutput:
        """
        Process input through hierarchical reasoning.

        Args:
            x: [batch, seq, h_dim] input tensor
            max_steps: Maximum outer loop steps (defaults to config)
            return_decomposition: Whether to return subproblem decomposition

        Returns:
            HRMOutput containing final state and optional decomposition
        """
        batch_size, seq_len, _ = x.shape
        max_steps = max_steps or self.config.max_outer_steps

        logger.debug("HRM forward pass: input_shape=%s, max_steps=%d", x.shape, max_steps)

        # Initial projection
        h_state = self.input_proj(x)

        # Tracking
        subproblems = []
        convergence_path = []
        total_ponder_cost = 0.0

        # Outer loop: iterative refinement
        for step in range(max_steps):
            # H-module: high-level planning
            for h_layer in self.h_module:
                h_state = h_layer(h_state)

            # Check halting condition
            halt_probs, ponder_cost = self.act(h_state)
            total_ponder_cost += ponder_cost

            # Average halting probability across sequence
            avg_halt_prob = halt_probs.mean().item()
            convergence_path.append(avg_halt_prob)

            # Generate subproblem decomposition if requested
            if return_decomposition:
                subproblem_repr = self.h_module[0].decompose(h_state)
                # Create subproblem entries (simplified)
                for i in range(min(3, seq_len)):  # Top 3 subproblems
                    subproblems.append(
                        SubProblem(
                            level=step,
                            description=f"Subproblem at step {step}, position {i}",
                            state=subproblem_repr[:, i, :].detach(),
                            confidence=halt_probs[:, i].mean().item(),
                        )
                    )

            # Halt if confident enough
            if avg_halt_prob >= self.config.halt_threshold:
                logger.debug(
                    "HRM halting at step %d: avg_halt_prob=%.4f, threshold=%.4f, ponder_cost=%.4f",
                    step,
                    avg_halt_prob,
                    self.config.halt_threshold,
                    total_ponder_cost,
                )
                break

            # L-module: low-level execution
            l_output, l_feedback = self.l_module(h_state)

            # Integrate L-module feedback
            h_state = self.integrate(torch.cat([h_state, l_feedback], dim=-1))

        return HRMOutput(
            final_state=h_state,
            subproblems=subproblems,
            halt_step=step + 1,
            total_ponder_cost=total_ponder_cost,
            convergence_path=convergence_path,
        )

    async def decompose_problem(self, query: str, state: torch.Tensor) -> list[SubProblem]:
        """
        Decompose a problem into hierarchical subproblems.

        Args:
            query: Natural language problem description
            state: Initial state representation

        Returns:
            List of subproblems in hierarchical order
        """
        # Ensure state is batched
        if state.dim() == 2:
            state = state.unsqueeze(0)  # [1, seq, dim]

        # Forward pass with decomposition
        output = self.forward(state, return_decomposition=True)

        # Add query context to subproblems
        for i, sp in enumerate(output.subproblems):
            sp.description = f"{query} -> Level {sp.level} Subproblem {i}"

        return output.subproblems

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training utilities
class HRMLoss(nn.Module):
    """
    Combined loss for HRM training.

    Includes:
    - Task loss (e.g., cross-entropy for classification)
    - Ponder cost regularization (encourages efficiency)
    - Consistency loss (encourages stable convergence)
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        ponder_weight: float = 0.01,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.task_weight = task_weight
        self.ponder_weight = ponder_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        hrm_output: HRMOutput,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_loss_fn: nn.Module,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            hrm_output: Output from HRM forward pass
            predictions: Model predictions
            targets: Ground truth targets
            task_loss_fn: Loss function for the task

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Task loss
        task_loss = task_loss_fn(predictions, targets)

        # Ponder cost (encourages efficiency)
        ponder_loss = hrm_output.total_ponder_cost

        # Consistency loss (encourages monotonic convergence)
        if len(hrm_output.convergence_path) > 1:
            conv_tensor = torch.tensor(hrm_output.convergence_path)
            # Penalize non-monotonic increases
            diffs = conv_tensor[1:] - conv_tensor[:-1]
            consistency_loss = F.relu(-diffs).mean()  # Penalize decreases
        else:
            consistency_loss = torch.tensor(0.0)

        # Combine losses
        total_loss = (
            self.task_weight * task_loss + self.ponder_weight * ponder_loss + self.consistency_weight * consistency_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "ponder": ponder_loss,
            "consistency": consistency_loss.item(),
            "halt_step": hrm_output.halt_step,
        }

        return total_loss, loss_dict


def create_hrm_agent(config: HRMConfig, device: str = "cpu") -> HRMAgent:
    """
    Factory function to create and initialize HRM agent.

    Args:
        config: HRM configuration
        device: Device to place model on

    Returns:
        Initialized HRMAgent
    """
    agent = HRMAgent(config, device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    agent.apply(init_weights)

    return agent
