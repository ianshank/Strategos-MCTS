"""
Tiny Recursive Model (TRM) Agent.

Implements recursive refinement with:
- Deep supervision at all recursion levels
- Convergence detection
- Memory-efficient recursion
- Iterative improvement mechanism

Based on principles from:
- "Recursive Refinement Networks"
- "Deep Supervision for Neural Networks"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..training.system_config import TRMConfig

logger = logging.getLogger(__name__)


@dataclass
class TRMOutput:
    """Output from TRM recursive processing."""

    final_prediction: torch.Tensor  # Final refined output
    intermediate_predictions: list[torch.Tensor]  # Predictions at each recursion
    recursion_depth: int  # Actual depth used
    converged: bool  # Whether convergence was achieved
    convergence_step: int  # Step at which convergence occurred
    residual_norms: list[float]  # L2 norms of residuals at each step


class RecursiveBlock(nn.Module):
    """
    Core recursive processing block.

    Applies the same transformation repeatedly, with residual connections.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Main processing pathway
        self.transform = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim) if config.use_layer_norm else nn.Identity(),
        )

        # Residual scaling (learned)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, iteration: int = 0) -> torch.Tensor:  # noqa: ARG002
        """
        Apply recursive transformation.

        Args:
            x: Input tensor [batch, ..., latent_dim]
            iteration: Current recursion iteration (reserved for future iteration-dependent behavior)

        Returns:
            Refined tensor [batch, ..., latent_dim]
        """
        # Residual connection with learned scaling
        residual = self.transform(x)
        result: torch.Tensor = x + self.residual_scale * residual
        return result


class DeepSupervisionHead(nn.Module):
    """
    Supervision head for intermediate predictions.

    Enables training signal at each recursion level.
    """

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate prediction from latent state."""
        result: torch.Tensor = self.head(x)
        return result


class TRMAgent(nn.Module):
    """
    Tiny Recursive Model for iterative refinement.

    Features:
    - Shared weights across recursions (parameter efficiency)
    - Deep supervision at all levels
    - Automatic convergence detection
    - Residual connections for stable gradients
    """

    def __init__(self, config: TRMConfig, output_dim: int | None = None, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.output_dim = output_dim or config.latent_dim

        logger.debug(
            "TRM agent initialized: latent_dim=%d, hidden_dim=%d, num_recursions=%d, device=%s",
            config.latent_dim,
            config.hidden_dim,
            config.num_recursions,
            device,
        )

        # Initial encoding
        self.encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim) if config.use_layer_norm else nn.Identity(),
        )

        # Shared recursive block
        self.recursive_block = RecursiveBlock(config)

        # Deep supervision heads (one per recursion level)
        if config.deep_supervision:
            self.supervision_heads = nn.ModuleList(
                [DeepSupervisionHead(config.latent_dim, self.output_dim) for _ in range(config.num_recursions)]
            )
        else:
            # Single output head
            self.output_head = DeepSupervisionHead(config.latent_dim, self.output_dim)

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        num_recursions: int | None = None,
        check_convergence: bool = True,
    ) -> TRMOutput:
        """
        Process input through recursive refinement.

        Args:
            x: Input tensor [batch, ..., latent_dim]
            num_recursions: Number of recursions (defaults to config)
            check_convergence: Whether to check for early convergence

        Returns:
            TRMOutput with final and intermediate predictions
        """
        num_recursions = num_recursions or self.config.num_recursions

        logger.debug("TRM forward: input_shape=%s, num_recursions=%d", x.shape, num_recursions)

        # Initial encoding
        latent = self.encoder(x)
        previous_latent = latent.clone()

        # Tracking
        intermediate_predictions = []
        residual_norms = []
        converged = False
        convergence_step = num_recursions

        # Recursive refinement
        for i in range(num_recursions):
            # Apply recursive transformation
            latent = self.recursive_block(latent, iteration=i)

            # Generate intermediate prediction
            if self.config.deep_supervision and i < len(self.supervision_heads):
                pred = self.supervision_heads[i](latent)
            else:
                pred = self.output_head(latent)

            intermediate_predictions.append(pred)

            # Check convergence
            if check_convergence and i >= self.config.min_recursions:
                residual = latent - previous_latent
                residual_norm = torch.norm(residual, p=2, dim=-1).mean().item()
                residual_norms.append(residual_norm)

                if residual_norm < self.config.convergence_threshold:
                    converged = True
                    convergence_step = i + 1
                    logger.debug(
                        "TRM converged at recursion %d: residual_norm=%.6f, threshold=%.6f",
                        i,
                        residual_norm,
                        self.config.convergence_threshold,
                    )
                    break

            previous_latent = latent.clone()

        # Final prediction
        final_pred = intermediate_predictions[-1]

        return TRMOutput(
            final_prediction=final_pred,
            intermediate_predictions=intermediate_predictions,
            recursion_depth=len(intermediate_predictions),
            converged=converged,
            convergence_step=convergence_step,
            residual_norms=residual_norms,
        )

    async def refine_solution(
        self,
        initial_prediction: torch.Tensor,
        num_recursions: int | None = None,
        convergence_threshold: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Refine an initial prediction through recursive processing.

        Args:
            initial_prediction: Initial solution [batch, ..., latent_dim]
            num_recursions: Maximum recursions (optional)
            convergence_threshold: Convergence threshold (optional)

        Returns:
            refined_solution: Final refined prediction
            info: Dictionary with refinement metadata
        """
        # Temporarily override convergence threshold if provided
        original_threshold = self.config.convergence_threshold
        if convergence_threshold is not None:
            self.config.convergence_threshold = convergence_threshold

        # Process
        output = self.forward(
            initial_prediction,
            num_recursions=num_recursions,
            check_convergence=True,
        )

        # Restore original threshold
        self.config.convergence_threshold = original_threshold

        info = {
            "converged": output.converged,
            "convergence_step": output.convergence_step,
            "total_recursions": output.recursion_depth,
            "final_residual": output.residual_norms[-1] if output.residual_norms else None,
            "refinement_path": output.residual_norms,
        }

        return output.final_prediction, info

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TRMLoss(nn.Module):
    """
    Deep supervision loss for TRM.

    Applies weighted supervision at all recursion levels,
    with exponential decay for deeper levels.
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        supervision_weight_decay: float = 0.5,
        final_weight: float = 1.0,
    ):
        """
        Initialize TRM loss.

        Args:
            task_loss_fn: Base loss function (e.g., MSE, CrossEntropy)
            supervision_weight_decay: Decay factor for intermediate losses
            final_weight: Weight for final prediction loss
        """
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.supervision_weight_decay = supervision_weight_decay
        self.final_weight = final_weight

    def forward(self, trm_output: TRMOutput, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute deep supervision loss.

        Args:
            trm_output: Output from TRM forward pass
            targets: Ground truth targets

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        # Final prediction loss (highest weight)
        final_loss = self.task_loss_fn(trm_output.final_prediction, targets)
        total_loss = self.final_weight * final_loss

        # Intermediate supervision losses
        intermediate_losses = []
        num_intermediate = len(trm_output.intermediate_predictions) - 1

        for i, pred in enumerate(trm_output.intermediate_predictions[:-1]):
            # Exponential decay: earlier predictions get lower weight
            weight = self.supervision_weight_decay ** (num_intermediate - i)
            loss = self.task_loss_fn(pred, targets)
            intermediate_losses.append(loss.item())
            total_loss = total_loss + weight * loss

        loss_dict = {
            "total": total_loss.item(),
            "final": final_loss.item(),
            "intermediate_mean": (sum(intermediate_losses) / len(intermediate_losses) if intermediate_losses else 0.0),
            "recursion_depth": trm_output.recursion_depth,
            "converged": trm_output.converged,
            "convergence_step": trm_output.convergence_step,
        }

        return total_loss, loss_dict


def create_trm_agent(config: TRMConfig, output_dim: int | None = None, device: str = "cpu") -> TRMAgent:
    """
    Factory function to create and initialize TRM agent.

    Args:
        config: TRM configuration
        output_dim: Output dimension (defaults to latent_dim)
        device: Device to place model on

    Returns:
        Initialized TRMAgent
    """
    agent = TRMAgent(config, output_dim, device)

    # Initialize weights with Xavier/He initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    agent.apply(init_weights)

    return agent


# Utility functions for integration
class TRMRefinementWrapper:
    """
    Wrapper for using TRM as a refinement step in pipelines.

    Provides a clean interface for integrating TRM into larger systems.
    """

    def __init__(self, trm_agent: TRMAgent, device: str = "cpu"):
        self.trm_agent = trm_agent
        self.device = device
        self.trm_agent.eval()

    @torch.no_grad()
    async def refine(
        self,
        predictions: torch.Tensor,
        num_iterations: int = 10,
        return_path: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Refine predictions using TRM.

        Args:
            predictions: Initial predictions to refine
            num_iterations: Number of refinement iterations
            return_path: Whether to return intermediate predictions

        Returns:
            refined_predictions or (refined_predictions, refinement_path)
        """
        # Ensure predictions are on correct device
        predictions = predictions.to(self.device)

        # Run TRM
        output = self.trm_agent(predictions, num_recursions=num_iterations, check_convergence=True)

        if return_path:
            result: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] = (
                output.final_prediction,
                output.intermediate_predictions,
            )
            return result
        final: torch.Tensor = output.final_prediction
        return final

    def get_refinement_stats(self, predictions: torch.Tensor) -> dict:
        """Get statistics about the refinement process."""
        with torch.no_grad():
            output = self.trm_agent(predictions, check_convergence=True)

        return {
            "converged": output.converged,
            "steps_to_convergence": output.convergence_step,
            "final_residual": (output.residual_norms[-1] if output.residual_norms else None),
            "total_refinement_iterations": output.recursion_depth,
        }
