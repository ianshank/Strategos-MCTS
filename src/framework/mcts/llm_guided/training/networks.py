"""
Neural Networks for MCTS Knowledge Distillation.

Provides:
- CodeEncoder: Transformer-based code and text encoder
- PolicyNetwork: Network for action selection (distilled from LLM)
- ValueNetwork: Network for state value estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.observability.logging import get_structured_logger

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

if TYPE_CHECKING:
    pass

logger = get_structured_logger(__name__)


@dataclass
class CodeEncoderConfig:
    """Configuration for code encoder."""

    # Architecture
    vocab_size: int = 50257  # GPT-2 vocab size
    hidden_dim: int = 256
    """Hidden dimension for transformer layers."""

    num_layers: int = 4
    """Number of transformer layers."""

    num_heads: int = 8
    """Number of attention heads."""

    ff_dim: int = 1024
    """Feedforward dimension."""

    max_seq_length: int = 2048
    """Maximum sequence length."""

    # Regularization
    dropout: float = 0.1
    """Dropout rate."""

    # Pooling
    pooling: str = "mean"
    """Pooling strategy: 'mean', 'cls', 'max'."""

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        if self.hidden_dim % self.num_heads != 0:
            errors.append("hidden_dim must be divisible by num_heads")
        if self.num_layers < 1:
            errors.append("num_layers must be >= 1")
        if self.dropout < 0 or self.dropout > 1:
            errors.append("dropout must be in [0, 1]")
        if self.pooling not in ("mean", "cls", "max"):
            errors.append("pooling must be 'mean', 'cls', or 'max'")

        if errors:
            raise ValueError("Invalid CodeEncoderConfig:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class PolicyNetworkConfig:
    """Configuration for policy network."""

    # Encoder
    encoder_config: CodeEncoderConfig = field(default_factory=CodeEncoderConfig)
    """Configuration for the code encoder."""

    # Policy head
    max_actions: int = 10
    """Maximum number of actions to predict."""

    hidden_dim: int = 256
    """Hidden dimension for policy head."""

    num_layers: int = 2
    """Number of layers in policy head."""

    # Regularization
    dropout: float = 0.1
    """Dropout rate."""

    # Temperature
    temperature: float = 1.0
    """Temperature for softmax."""

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        if self.max_actions < 1:
            errors.append("max_actions must be >= 1")
        if self.hidden_dim < 1:
            errors.append("hidden_dim must be >= 1")
        if self.num_layers < 1:
            errors.append("num_layers must be >= 1")
        if self.dropout < 0 or self.dropout > 1:
            errors.append("dropout must be in [0, 1]")
        if self.temperature <= 0:
            errors.append("temperature must be > 0")

        # Also validate encoder config
        self.encoder_config.validate()

        if errors:
            raise ValueError("Invalid PolicyNetworkConfig:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class ValueNetworkConfig:
    """Configuration for value network."""

    # Encoder
    encoder_config: CodeEncoderConfig = field(default_factory=CodeEncoderConfig)
    """Configuration for the code encoder."""

    # Value head
    hidden_dim: int = 256
    """Hidden dimension for value head."""

    num_layers: int = 2
    """Number of layers in value head."""

    # Regularization
    dropout: float = 0.1
    """Dropout rate."""

    # Output bounds
    min_value: float = -1.0
    """Minimum value output."""

    max_value: float = 1.0
    """Maximum value output."""

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        if self.hidden_dim < 1:
            errors.append("hidden_dim must be >= 1")
        if self.num_layers < 1:
            errors.append("num_layers must be >= 1")
        if self.dropout < 0 or self.dropout > 1:
            errors.append("dropout must be in [0, 1]")
        if self.min_value >= self.max_value:
            errors.append("min_value must be less than max_value")

        # Also validate encoder config
        self.encoder_config.validate()

        if errors:
            raise ValueError("Invalid ValueNetworkConfig:\n" + "\n".join(f"  - {e}" for e in errors))


def _check_torch() -> None:
    """Check if PyTorch is available."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")


class PositionalEncoding(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        _check_torch()
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CodeEncoder(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Transformer-based encoder for code and text.

    Encodes input sequences into fixed-size representations suitable
    for downstream policy and value prediction.
    """

    def __init__(self, config: CodeEncoderConfig | None = None):
        """
        Initialize code encoder.

        Args:
            config: Encoder configuration
        """
        _check_torch()
        super().__init__()

        self._config = config or CodeEncoderConfig()
        self._config.validate()

        # Embedding layer
        self.embedding = nn.Embedding(self._config.vocab_size, self._config.hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self._config.hidden_dim,
            max_len=self._config.max_seq_length,
            dropout=self._config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._config.hidden_dim,
            nhead=self._config.num_heads,
            dim_feedforward=self._config.ff_dim,
            dropout=self._config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self._config.num_layers,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(self._config.hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def output_dim(self) -> int:
        """Output dimension of encoder."""
        return self._config.hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode input sequence.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Encoded representation [batch_size, hidden_dim]
        """
        # Embed tokens
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert from [batch, seq] to [batch, seq, seq] for self-attention
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode with transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)

        # Pool to fixed-size representation
        if self._config.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            else:
                x = x.mean(dim=1)
        elif self._config.pooling == "cls":
            x = x[:, 0]
        elif self._config.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = x.masked_fill(mask == 0, float("-inf"))
            x = x.max(dim=1)[0]

        return x


class PolicyNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Policy network for action selection.

    Takes encoded state and predicts action probability distribution,
    distilled from LLM policy through MCTS improvement.
    """

    def __init__(self, config: PolicyNetworkConfig | None = None):
        """
        Initialize policy network.

        Args:
            config: Policy network configuration
        """
        _check_torch()
        super().__init__()

        self._config = config or PolicyNetworkConfig()
        self._config.validate()

        # Encoder for code and problem
        self.code_encoder = CodeEncoder(self._config.encoder_config)
        self.problem_encoder = CodeEncoder(self._config.encoder_config)

        # Combine encoded representations
        encoder_dim = self.code_encoder.output_dim * 2

        # Policy head
        layers = []
        in_dim = encoder_dim
        for _ in range(self._config.num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, self._config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self._config.dropout),
                ]
            )
            in_dim = self._config.hidden_dim

        layers.append(nn.Linear(in_dim, self._config.max_actions))
        self.policy_head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.policy_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        code_tokens: torch.Tensor,
        code_attention_mask: torch.Tensor,
        problem_tokens: torch.Tensor,
        problem_attention_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for policy prediction.

        Args:
            code_tokens: Code token IDs [batch_size, code_seq_len]
            code_attention_mask: Code attention mask [batch_size, code_seq_len]
            problem_tokens: Problem token IDs [batch_size, problem_seq_len]
            problem_attention_mask: Problem attention mask [batch_size, problem_seq_len]
            action_mask: Valid action mask [batch_size, max_actions]

        Returns:
            Log probabilities over actions [batch_size, max_actions]
        """
        # Encode code and problem
        code_repr = self.code_encoder(code_tokens, code_attention_mask)
        problem_repr = self.problem_encoder(problem_tokens, problem_attention_mask)

        # Combine representations
        combined = torch.cat([code_repr, problem_repr], dim=-1)

        # Predict logits
        logits = self.policy_head(combined)

        # Apply temperature
        logits = logits / self._config.temperature

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        # Log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def predict(
        self,
        code_tokens: torch.Tensor,
        code_attention_mask: torch.Tensor,
        problem_tokens: torch.Tensor,
        problem_attention_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get action probabilities (not log).

        Args:
            Same as forward()

        Returns:
            Probabilities over actions [batch_size, max_actions]
        """
        log_probs = self.forward(
            code_tokens,
            code_attention_mask,
            problem_tokens,
            problem_attention_mask,
            action_mask,
        )
        return torch.exp(log_probs)


class ValueNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Value network for state value estimation.

    Takes encoded state and predicts the expected outcome,
    distilled from LLM value estimates and MCTS Q-values.
    """

    def __init__(self, config: ValueNetworkConfig | None = None):
        """
        Initialize value network.

        Args:
            config: Value network configuration
        """
        _check_torch()
        super().__init__()

        self._config = config or ValueNetworkConfig()
        self._config.validate()

        # Encoder for code and problem
        self.code_encoder = CodeEncoder(self._config.encoder_config)
        self.problem_encoder = CodeEncoder(self._config.encoder_config)

        # Combine encoded representations
        encoder_dim = self.code_encoder.output_dim * 2

        # Value head
        layers = []
        in_dim = encoder_dim
        for _ in range(self._config.num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, self._config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self._config.dropout),
                ]
            )
            in_dim = self._config.hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.value_head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        code_tokens: torch.Tensor,
        code_attention_mask: torch.Tensor,
        problem_tokens: torch.Tensor,
        problem_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for value prediction.

        Args:
            code_tokens: Code token IDs [batch_size, code_seq_len]
            code_attention_mask: Code attention mask [batch_size, code_seq_len]
            problem_tokens: Problem token IDs [batch_size, problem_seq_len]
            problem_attention_mask: Problem attention mask [batch_size, problem_seq_len]

        Returns:
            Value predictions [batch_size]
        """
        # Encode code and problem
        code_repr = self.code_encoder(code_tokens, code_attention_mask)
        problem_repr = self.problem_encoder(problem_tokens, problem_attention_mask)

        # Combine representations
        combined = torch.cat([code_repr, problem_repr], dim=-1)

        # Predict value
        value = self.value_head(combined).squeeze(-1)

        # Bound output
        value = torch.tanh(value)  # Maps to [-1, 1]

        # Scale to config bounds
        value_range = self._config.max_value - self._config.min_value
        value = value * (value_range / 2) + (self._config.min_value + value_range / 2)

        return value


class CombinedNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Combined policy and value network with shared encoder.

    More efficient than separate networks when both predictions are needed.
    """

    def __init__(
        self,
        encoder_config: CodeEncoderConfig | None = None,
        max_actions: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        """
        Initialize combined network.

        Args:
            encoder_config: Configuration for shared encoder
            max_actions: Maximum number of actions
            hidden_dim: Hidden dimension for heads
            dropout: Dropout rate
            min_value: Minimum value output bound
            max_value: Maximum value output bound
        """
        _check_torch()
        super().__init__()

        self._encoder_config = encoder_config or CodeEncoderConfig()
        self._max_actions = max_actions
        self._hidden_dim = hidden_dim
        self._min_value = min_value
        self._max_value = max_value

        # Shared encoders
        self.code_encoder = CodeEncoder(self._encoder_config)
        self.problem_encoder = CodeEncoder(self._encoder_config)

        encoder_dim = self.code_encoder.output_dim * 2

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.policy_head, self.value_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        code_tokens: torch.Tensor,
        code_attention_mask: torch.Tensor,
        problem_tokens: torch.Tensor,
        problem_attention_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both policy and value.

        Returns:
            Tuple of (log_probs [batch, actions], values [batch])
        """
        # Encode
        code_repr = self.code_encoder(code_tokens, code_attention_mask)
        problem_repr = self.problem_encoder(problem_tokens, problem_attention_mask)
        combined = torch.cat([code_repr, problem_repr], dim=-1)

        # Policy
        logits = self.policy_head(combined)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)

        # Value - apply tanh then scale to configured bounds
        value = torch.tanh(self.value_head(combined).squeeze(-1))
        value_range = self._max_value - self._min_value
        value = value * (value_range / 2) + (self._min_value + value_range / 2)

        return log_probs, value


# Factory functions


def create_policy_network(
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    max_actions: int = 10,
    dropout: float = 0.1,
) -> PolicyNetwork:
    """
    Create a policy network with specified architecture.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_actions: Maximum actions to predict
        dropout: Dropout rate

    Returns:
        Configured PolicyNetwork
    """
    _check_torch()

    encoder_config = CodeEncoderConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=hidden_dim * 4,
        dropout=dropout,
    )

    policy_config = PolicyNetworkConfig(
        encoder_config=encoder_config,
        max_actions=max_actions,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    return PolicyNetwork(policy_config)


def create_value_network(
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> ValueNetwork:
    """
    Create a value network with specified architecture.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        Configured ValueNetwork
    """
    _check_torch()

    encoder_config = CodeEncoderConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=hidden_dim * 4,
        dropout=dropout,
    )

    value_config = ValueNetworkConfig(
        encoder_config=encoder_config,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    return ValueNetwork(value_config)
