"""
RNN-based Meta-Controller for dynamic agent selection.

This module provides a GRU-based recurrent neural network meta-controller
that learns to select the optimal agent (HRM, TRM, or MCTS) based on
sequential patterns in the agent state features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.meta_controller.base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)
from src.agents.meta_controller.utils import features_to_tensor
from src.observability.logging import get_logger

logger = get_logger(__name__)


class RNNMetaControllerModel(nn.Module):
    """
    GRU-based neural network model for meta-controller predictions.

    This model uses a Gated Recurrent Unit (GRU) to capture sequential
    patterns in agent state features and predict which agent should be
    selected next.

    Architecture:
        - GRU layer for sequence processing
        - Dropout for regularization
        - Linear layer for classification

    Attributes:
        gru: GRU recurrent layer for processing sequences.
        dropout: Dropout layer for regularization.
        fc: Fully connected output layer.
        hidden_dim: Dimension of the hidden state.
        num_layers: Number of GRU layers.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_agents: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the RNN meta-controller model.

        Args:
            input_dim: Dimension of input features. Defaults to 10.
            hidden_dim: Dimension of GRU hidden state. Defaults to 64.
            num_layers: Number of stacked GRU layers. Defaults to 1.
            num_agents: Number of agents to choose from. Defaults to 3.
            dropout: Dropout probability for regularization. Defaults to 0.1.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layer for sequence processing
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Linear output layer for classification
        self.fc = nn.Linear(hidden_dim, num_agents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Processes input features through GRU and produces agent selection logits.

        Args:
            x: Input tensor of shape (batch_size, features) or
               (batch_size, seq_len, features).

        Returns:
            Logits tensor of shape (batch_size, num_agents).
            Note: Returns raw logits, NOT softmax probabilities.

        Example:
            >>> model = RNNMetaControllerModel()
            >>> x = torch.randn(4, 10)  # batch of 4, 10 features
            >>> logits = model(x)
            >>> logits.shape
            torch.Size([4, 3])
        """
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            # Shape: (batch_size, features) -> (batch_size, 1, features)
            x = x.unsqueeze(1)

        # Pass through GRU
        # output shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        output, hidden = self.gru(x)

        # Take the final hidden state from the last layer
        # Shape: (batch_size, hidden_dim)
        final_hidden = hidden[-1] if self.num_layers > 1 else hidden.squeeze(0)

        # Apply dropout
        dropped = self.dropout(final_hidden)

        # Apply linear layer to get logits
        logits: torch.Tensor = self.fc(dropped)

        return logits


class RNNMetaController(AbstractMetaController):
    """
    RNN-based meta-controller using GRU for agent selection.

    This controller uses a recurrent neural network to learn patterns in
    agent state sequences and predict the optimal agent for the current
    situation. It supports both CPU and GPU execution.

    Attributes:
        device: PyTorch device (CPU or CUDA) for tensor operations.
        hidden_dim: Dimension of GRU hidden state.
        num_layers: Number of GRU layers.
        dropout: Dropout probability.
        model: The underlying RNNMetaControllerModel.
        hidden_state: Optional hidden state for sequence tracking.

    Example:
        >>> controller = RNNMetaController(name="RNNController", seed=42)
        >>> features = MetaControllerFeatures(
        ...     hrm_confidence=0.8,
        ...     trm_confidence=0.6,
        ...     mcts_value=0.75,
        ...     consensus_score=0.7,
        ...     last_agent='hrm',
        ...     iteration=2,
        ...     query_length=150,
        ...     has_rag_context=True
        ... )
        >>> prediction = controller.predict(features)
        >>> prediction.agent in ['hrm', 'trm', 'mcts']
        True
        >>> 0.0 <= prediction.confidence <= 1.0
        True
    """

    def __init__(
        self,
        name: str = "RNNMetaController",
        seed: int = 42,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        device: str | None = None,
    ) -> None:
        """
        Initialize the RNN meta-controller.

        Args:
            name: Name identifier for this controller. Defaults to "RNNMetaController".
            seed: Random seed for reproducibility. Defaults to 42.
            hidden_dim: Dimension of GRU hidden state. Defaults to 64.
            num_layers: Number of GRU layers. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.1.
            device: Device to run model on ('cpu', 'cuda', 'mps', etc.).
                   If None, auto-detects best available device.
        """
        super().__init__(name=name, seed=seed)

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Store configuration
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize model
        self.model = RNNMetaControllerModel(
            input_dim=10,  # Fixed based on features_to_tensor output
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_agents=len(self.AGENT_NAMES),
            dropout=dropout,
        )

        # Move model to device
        self.model = self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Initialize hidden state for sequence tracking
        self.hidden_state: torch.Tensor | None = None

        logger.info(
            "Initialized RNNMetaController",
            extra={
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "device": str(self.device),
            },
        )

    def predict(self, features: MetaControllerFeatures) -> MetaControllerPrediction:
        """
        Predict which agent should handle the current query.

        Converts features to tensor format, runs through the GRU model,
        and returns a prediction with confidence scores.

        Args:
            features: Features extracted from the current agent state.

        Returns:
            Prediction containing the selected agent, confidence score,
            and probability distribution over all agents.

        Example:
            >>> controller = RNNMetaController()
            >>> features = MetaControllerFeatures(
            ...     hrm_confidence=0.9,
            ...     trm_confidence=0.3,
            ...     mcts_value=0.5,
            ...     consensus_score=0.8,
            ...     last_agent='none',
            ...     iteration=0,
            ...     query_length=100,
            ...     has_rag_context=False
            ... )
            >>> pred = controller.predict(features)
            >>> isinstance(pred.agent, str)
            True
            >>> isinstance(pred.confidence, float)
            True
            >>> len(pred.probabilities) == 3
            True
        """
        # Convert features to tensor
        feature_tensor = features_to_tensor(features)

        # Add batch dimension: (10,) -> (1, 10)
        feature_tensor = feature_tensor.unsqueeze(0)

        # Move to device
        feature_tensor = feature_tensor.to(self.device)

        # Perform inference without gradient tracking
        with torch.no_grad():
            # Get logits from model
            logits = self.model(feature_tensor)

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Get predicted agent index (argmax)
            predicted_idx: int = int(torch.argmax(probabilities, dim=-1).item())

            # Extract confidence for selected agent
            confidence = probabilities[0, predicted_idx].item()

            # Create probability dictionary
            prob_dict: dict[str, float] = {}
            for i, agent_name in enumerate(self.AGENT_NAMES):
                prob_dict[agent_name] = probabilities[0, i].item()

        # Get agent name
        selected_agent = self.AGENT_NAMES[predicted_idx]

        logger.debug(
            "RNN meta-controller routing decision",
            extra={"selected_agent": selected_agent, "confidence": float(confidence), "probabilities": prob_dict},
        )

        return MetaControllerPrediction(
            agent=selected_agent,
            confidence=float(confidence),
            probabilities=prob_dict,
        )

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Loads the model state dictionary from the specified path and
        sets the model to evaluation mode.

        Args:
            path: Path to the saved model file (.pt or .pth).

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the state dict is incompatible with the model.

        Example:
            >>> controller = RNNMetaController()
            >>> controller.load_model("/path/to/model.pt")
        """
        # Load state dict with appropriate device mapping
        state_dict = torch.load(path, map_location=self.device, weights_only=True)

        # Load into model
        self.model.load_state_dict(state_dict)

        # Ensure model is in evaluation mode
        self.model.eval()
        logger.info("Loaded RNN meta-controller from %s", path)

    def save_model(self, path: str) -> None:
        """
        Save the current model to disk.

        Saves the model state dictionary to the specified path.

        Args:
            path: Path where the model should be saved (.pt or .pth).

        Example:
            >>> controller = RNNMetaController()
            >>> controller.save_model("/path/to/model.pt")
        """
        torch.save(self.model.state_dict(), path)
        logger.info("Saved RNN meta-controller to %s", path)

    def reset_hidden_state(self) -> None:
        """
        Reset the hidden state for sequence tracking.

        This method clears any accumulated hidden state, useful when
        starting a new conversation or resetting the controller state.

        Example:
            >>> controller = RNNMetaController()
            >>> controller.reset_hidden_state()
            >>> controller.hidden_state is None
            True
        """
        self.hidden_state = None
