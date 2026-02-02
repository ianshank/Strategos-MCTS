"""
Domain Adapters for MCTS - State conversion and action filtering.

Provides domain-specific implementations of the StateAdapter protocol
for converting MCTS states to neural network inputs.

Available Adapters:
- GridStateAdapter: For grid-based domains (Tic-Tac-Toe, Connect Four)
- TextStateAdapter: For text/NLP-based domains
- FeatureStateAdapter: For feature vector states

All adapters follow the StateAdapter protocol defined in neural_policies.py
and can be used with TorchNeuralRolloutPolicy.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .core import MCTSState

# Optional PyTorch support
_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment,unused-ignore]

_logger = logging.getLogger(__name__)


# =============================================================================
# Base Domain Adapter
# =============================================================================


class BaseDomainAdapter(ABC):
    """
    Abstract base class for domain adapters.

    Provides common functionality for state conversion and action mapping.
    Subclasses implement domain-specific tensor conversion.
    """

    def __init__(
        self,
        action_space_size: int,
        state_dim: int | tuple[int, ...],
        device: str = "cpu",
    ):
        """
        Initialize domain adapter.

        Args:
            action_space_size: Total number of possible actions
            state_dim: Dimensions of state representation
            device: Device for tensor operations
        """
        self.action_space_size = action_space_size
        self.state_dim = state_dim if isinstance(state_dim, tuple) else (state_dim,)
        self.device = device
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Action mapping caches
        self._action_to_index: dict[str, int] = {}
        self._index_to_action: dict[int, str] = {}

    @abstractmethod
    def state_to_tensor(self, state: MCTSState) -> Any:
        """Convert MCTS state to tensor."""
        pass

    @abstractmethod
    def _extract_legal_actions(self, state: MCTSState) -> list[str]:
        """Extract legal actions from state."""
        pass

    def get_action_mask(self, state: MCTSState) -> Any:
        """
        Get boolean mask for legal actions.

        Args:
            state: MCTS state

        Returns:
            Boolean array/tensor where True = legal action
        """
        legal_actions = self._extract_legal_actions(state)

        if not legal_actions:
            # All actions legal if no specific restrictions
            return None

        mask = np.zeros(self.action_space_size, dtype=bool)
        for action in legal_actions:
            idx = self.action_to_index(action)
            if 0 <= idx < self.action_space_size:
                mask[idx] = True

        if _TORCH_AVAILABLE:
            return torch.tensor(mask, dtype=torch.bool, device=self.device)
        return mask

    def action_to_index(self, action: str) -> int:
        """
        Convert action string to index.

        Args:
            action: Action string

        Returns:
            Index in action space
        """
        if action in self._action_to_index:
            return self._action_to_index[action]

        # Default: try to parse as integer
        try:
            idx = int(action)
            self._action_to_index[action] = idx
            self._index_to_action[idx] = action
            return idx
        except ValueError:
            # Hash-based index for string actions
            idx = int(hashlib.md5(action.encode()).hexdigest()[:8], 16) % self.action_space_size
            self._action_to_index[action] = idx
            self._index_to_action[idx] = action
            return idx

    def index_to_action(self, index: int) -> str:
        """
        Convert index to action string.

        Args:
            index: Index in action space

        Returns:
            Action string
        """
        return self._index_to_action.get(index, str(index))

    def tensor_to_action_priors(
        self,
        policy_output: Any,
        state: MCTSState,
    ) -> dict[str, float]:
        """
        Convert policy output to action priors dictionary.

        Args:
            policy_output: Raw policy network output (numpy array or tensor)
            state: Current state (for action mapping)

        Returns:
            Dictionary mapping action strings to probabilities
        """
        # Convert to numpy if needed
        if _TORCH_AVAILABLE and isinstance(policy_output, torch.Tensor):
            policy_output = policy_output.cpu().numpy()

        policy_output = np.asarray(policy_output).flatten()

        # Get legal actions
        legal_actions = self._extract_legal_actions(state)

        if not legal_actions:
            # Return all actions with their probabilities
            return {self.index_to_action(i): float(p) for i, p in enumerate(policy_output)}

        # Filter to legal actions and renormalize
        priors = {}
        total = 0.0

        for action in legal_actions:
            idx = self.action_to_index(action)
            if 0 <= idx < len(policy_output):
                prob = float(policy_output[idx])
                priors[action] = prob
                total += prob

        # Normalize
        if total > 0:
            priors = {a: p / total for a, p in priors.items()}
        else:
            # Uniform if all zeros
            uniform = 1.0 / len(legal_actions) if legal_actions else 0.0
            priors = dict.fromkeys(legal_actions, uniform)

        return priors


# =============================================================================
# Grid-Based Domain Adapter
# =============================================================================


@dataclass
class GridAdapterConfig:
    """Configuration for grid-based domain adapter."""

    board_size: int = 3
    num_channels: int = 3  # e.g., player 1, player 2, empty
    include_history: bool = False
    history_length: int = 8
    player_perspective: bool = True  # Canonical form from player's view


class GridStateAdapter(BaseDomainAdapter):
    """
    Adapter for grid-based game states.

    Suitable for:
    - Tic-Tac-Toe (3x3)
    - Connect Four (6x7)
    - Othello/Reversi (8x8)
    - Go (9x9, 13x13, 19x19)

    Converts grid states to multi-channel tensor representation
    compatible with convolutional neural networks.
    """

    def __init__(
        self,
        config: GridAdapterConfig | None = None,
        device: str = "cpu",
    ):
        """
        Initialize grid adapter.

        Args:
            config: Grid configuration
            device: Device for tensors
        """
        self.grid_config = config or GridAdapterConfig()

        board_size = self.grid_config.board_size
        num_channels = self.grid_config.num_channels

        if self.grid_config.include_history:
            num_channels *= self.grid_config.history_length

        super().__init__(
            action_space_size=board_size * board_size,
            state_dim=(num_channels, board_size, board_size),
            device=device,
        )

        self._logger.info(f"Initialized GridStateAdapter: board={board_size}x{board_size}, " f"channels={num_channels}")

    def state_to_tensor(self, state: MCTSState) -> Any:
        """
        Convert grid state to tensor.

        Expected state.features format:
        - 'board': 2D array or flat list of cell values
        - 'player': Current player (1 or -1)
        - 'history' (optional): List of previous boards

        Args:
            state: MCTS state with grid data

        Returns:
            Tensor of shape (C, H, W)
        """
        features = state.features
        board_size = self.grid_config.board_size

        # Extract board
        board = features.get("board", [])
        if isinstance(board, list):
            board = np.array(board)

        # Reshape if flat
        if board.ndim == 1:
            board = board.reshape(board_size, board_size)

        # Current player
        player = features.get("player", 1)

        # Create multi-channel representation
        channels = []

        if self.grid_config.player_perspective:
            # Canonical form from current player's perspective
            channels.append((board == player).astype(np.float32))  # Current player pieces
            channels.append((board == -player).astype(np.float32))  # Opponent pieces
            channels.append((board == 0).astype(np.float32))  # Empty cells
        else:
            # Absolute representation
            channels.append((board == 1).astype(np.float32))  # Player 1
            channels.append((board == -1).astype(np.float32))  # Player 2
            channels.append((board == 0).astype(np.float32))  # Empty

        # Add history if configured
        if self.grid_config.include_history:
            history = features.get("history", [])
            for i in range(self.grid_config.history_length - 1):
                if i < len(history):
                    hist_board = np.array(history[-(i + 1)])
                    if hist_board.ndim == 1:
                        hist_board = hist_board.reshape(board_size, board_size)
                    channels.append((hist_board == player).astype(np.float32))
                    channels.append((hist_board == -player).astype(np.float32))
                else:
                    # Pad with zeros
                    channels.append(np.zeros((board_size, board_size), dtype=np.float32))
                    channels.append(np.zeros((board_size, board_size), dtype=np.float32))

        tensor = np.stack(channels, axis=0)

        if _TORCH_AVAILABLE:
            return torch.tensor(tensor, dtype=torch.float32, device=self.device)
        return tensor

    def _extract_legal_actions(self, state: MCTSState) -> list[str]:
        """Extract legal moves from grid state."""
        features = state.features

        # Check for explicit legal moves
        if "legal_moves" in features:
            return [str(m) for m in features["legal_moves"]]

        # Derive from board (empty cells are legal)
        board = features.get("board", [])
        if isinstance(board, list):
            board = np.array(board)

        if board.ndim == 1:
            board = board.reshape(self.grid_config.board_size, self.grid_config.board_size)

        legal = []
        for i in range(self.grid_config.board_size):
            for j in range(self.grid_config.board_size):
                if board[i, j] == 0:
                    legal.append(f"{i},{j}")

        return legal

    def action_to_index(self, action: str) -> int:
        """Convert grid action (row,col) to flat index."""
        if "," in action:
            row, col = map(int, action.split(","))
            return row * self.grid_config.board_size + col
        return super().action_to_index(action)

    def index_to_action(self, index: int) -> str:
        """Convert flat index to grid action."""
        row = index // self.grid_config.board_size
        col = index % self.grid_config.board_size
        return f"{row},{col}"


# =============================================================================
# Feature Vector Domain Adapter
# =============================================================================


@dataclass
class FeatureAdapterConfig:
    """Configuration for feature-based domain adapter."""

    feature_dim: int = 128
    normalize_features: bool = True
    feature_names: list[str] = field(default_factory=list)


class FeatureStateAdapter(BaseDomainAdapter):
    """
    Adapter for feature vector states.

    Suitable for:
    - Tabular domains with feature engineering
    - Embedding-based states
    - Custom feature representations

    Converts feature dictionaries to flat tensors for MLP networks.
    """

    def __init__(
        self,
        config: FeatureAdapterConfig | None = None,
        action_space_size: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize feature adapter.

        Args:
            config: Feature configuration
            action_space_size: Number of possible actions
            device: Device for tensors
        """
        self.feature_config = config or FeatureAdapterConfig()

        super().__init__(
            action_space_size=action_space_size,
            state_dim=self.feature_config.feature_dim,
            device=device,
        )

    def state_to_tensor(self, state: MCTSState) -> Any:
        """
        Convert feature state to tensor.

        Expected state.features format:
        - Can be a dict with feature names as keys
        - Or a list/array of feature values

        Args:
            state: MCTS state with feature data

        Returns:
            Tensor of shape (feature_dim,)
        """
        features = state.features

        if isinstance(features, dict):
            # Extract features by name or all values
            if self.feature_config.feature_names:
                values = [features.get(name, 0.0) for name in self.feature_config.feature_names]
            else:
                values = list(features.values())
        elif isinstance(features, (list, tuple, np.ndarray)):
            values = list(features)
        else:
            values = [0.0] * self.feature_config.feature_dim

        # Pad or truncate to feature_dim
        feature_dim = self.feature_config.feature_dim
        if len(values) < feature_dim:
            values.extend([0.0] * (feature_dim - len(values)))
        elif len(values) > feature_dim:
            values = values[:feature_dim]

        tensor = np.array(values, dtype=np.float32)

        # Normalize if configured
        if self.feature_config.normalize_features:
            norm = np.linalg.norm(tensor)
            if norm > 0:
                tensor = tensor / norm

        if _TORCH_AVAILABLE:
            return torch.tensor(tensor, dtype=torch.float32, device=self.device)
        return tensor

    def _extract_legal_actions(self, state: MCTSState) -> list[str]:
        """Extract legal actions from feature state."""
        features = state.features

        if isinstance(features, dict) and "legal_actions" in features:
            return [str(a) for a in features["legal_actions"]]

        # Default: all actions legal
        return []


# =============================================================================
# Text/NLP Domain Adapter
# =============================================================================


@dataclass
class TextAdapterConfig:
    """Configuration for text-based domain adapter."""

    max_sequence_length: int = 512
    embedding_dim: int = 768  # BERT-like
    use_tokenizer: bool = False
    tokenizer_name: str = "bert-base-uncased"


class TextStateAdapter(BaseDomainAdapter):
    """
    Adapter for text/NLP states.

    Suitable for:
    - Natural language understanding tasks
    - Code understanding/generation
    - Document analysis

    Note: For best results, use with pre-trained language model embeddings.
    """

    def __init__(
        self,
        config: TextAdapterConfig | None = None,
        action_space_size: int = 1000,
        device: str = "cpu",
    ):
        """
        Initialize text adapter.

        Args:
            config: Text configuration
            action_space_size: Vocabulary size or action count
            device: Device for tensors
        """
        self.text_config = config or TextAdapterConfig()

        super().__init__(
            action_space_size=action_space_size,
            state_dim=(self.text_config.max_sequence_length, self.text_config.embedding_dim),
            device=device,
        )

        self._tokenizer = None
        self._embedding_model = None

    def state_to_tensor(self, state: MCTSState) -> Any:
        """
        Convert text state to tensor.

        Expected state.features format:
        - 'text': String content
        - 'embedding' (optional): Pre-computed embedding

        Args:
            state: MCTS state with text data

        Returns:
            Tensor of shape (seq_len, embed_dim) or (embed_dim,)
        """
        features = state.features

        # Use pre-computed embedding if available
        if "embedding" in features:
            embedding = features["embedding"]
            if isinstance(embedding, np.ndarray):
                tensor = embedding.astype(np.float32)
            elif _TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
                return embedding.to(self.device)
            else:
                tensor = np.array(embedding, dtype=np.float32)

            if _TORCH_AVAILABLE:
                return torch.tensor(tensor, dtype=torch.float32, device=self.device)
            return tensor

        # Simple bag-of-words fallback
        text = features.get("text", "")
        if not text:
            tensor = np.zeros(self.text_config.embedding_dim, dtype=np.float32)
        else:
            # Simple character-level embedding (placeholder)
            chars = [ord(c) % 256 for c in text[: self.text_config.max_sequence_length]]
            tensor = np.zeros(self.text_config.embedding_dim, dtype=np.float32)
            for i, c in enumerate(chars[: self.text_config.embedding_dim]):
                tensor[i] = c / 255.0

        if _TORCH_AVAILABLE:
            return torch.tensor(tensor, dtype=torch.float32, device=self.device)
        return tensor

    def _extract_legal_actions(self, state: MCTSState) -> list[str]:
        """Extract legal actions from text state."""
        features = state.features

        if isinstance(features, dict) and "legal_actions" in features:
            return [str(a) for a in features["legal_actions"]]

        return []


# =============================================================================
# Factory Function
# =============================================================================


def create_domain_adapter(
    domain_type: str,
    **kwargs,
) -> BaseDomainAdapter:
    """
    Factory function to create domain adapters.

    Args:
        domain_type: Type of domain ("grid", "feature", "text")
        **kwargs: Domain-specific configuration

    Returns:
        Configured domain adapter

    Example:
        >>> adapter = create_domain_adapter("grid", board_size=8)
        >>> adapter = create_domain_adapter("feature", feature_dim=256)
    """
    adapters = {
        "grid": (GridStateAdapter, GridAdapterConfig),
        "feature": (FeatureStateAdapter, FeatureAdapterConfig),
        "text": (TextStateAdapter, TextAdapterConfig),
    }

    if domain_type not in adapters:
        raise ValueError(f"Unknown domain type: {domain_type}. Available: {list(adapters.keys())}")

    adapter_class, config_class = adapters[domain_type]

    # Extract config parameters
    config_params = {}
    other_params = {}

    for key, value in kwargs.items():
        if hasattr(config_class, key):
            config_params[key] = value
        else:
            other_params[key] = value

    config = config_class(**config_params) if config_params else None

    adapter: BaseDomainAdapter = adapter_class(config=config, **other_params)
    return adapter


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base class
    "BaseDomainAdapter",
    # Grid adapter
    "GridAdapterConfig",
    "GridStateAdapter",
    # Feature adapter
    "FeatureAdapterConfig",
    "FeatureStateAdapter",
    # Text adapter
    "TextAdapterConfig",
    "TextStateAdapter",
    # Factory
    "create_domain_adapter",
]
