"""
PyTorch Dataset for MCTS Training Data.

Provides:
- MCTSDataset: PyTorch Dataset for loading training examples
- MCTSDatasetConfig: Configuration for dataset behavior
- TrainingBatch: Structured batch representation
- Utilities for loading and splitting data
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.observability.logging import get_structured_logger

# Optional PyTorch imports
try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    Dataset = object
    DataLoader = None

if TYPE_CHECKING:
    pass

logger = get_structured_logger(__name__)


@dataclass
class MCTSDatasetConfig:
    """Configuration for MCTS training dataset."""

    # Data paths
    data_dir: str | Path = "./training_data"
    """Directory containing training data files."""

    # Feature extraction
    max_code_length: int = 2048
    """Maximum code sequence length."""

    max_problem_length: int = 1024
    """Maximum problem description length."""

    max_actions: int = 10
    """Maximum number of actions to consider."""

    # Tokenization
    tokenizer_name: str = "gpt2"
    """Tokenizer to use for text encoding."""

    # Augmentation
    shuffle_actions: bool = False
    """Shuffle action order during training.

    Note: This is a placeholder for future data augmentation.
    Currently not implemented - actions maintain their sorted order.
    """

    # Filtering
    min_visits: int = 1
    """Minimum node visits to include example."""

    exclude_root_nodes: bool = True
    """Exclude root nodes (no meaningful action)."""

    only_successful_episodes: bool = False
    """Only include examples from successful episodes."""

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if self.max_code_length < 1:
            errors.append("max_code_length must be >= 1")
        if self.max_problem_length < 1:
            errors.append("max_problem_length must be >= 1")
        if self.max_actions < 1:
            errors.append("max_actions must be >= 1")
        if self.min_visits < 0:
            errors.append("min_visits must be >= 0")

        if errors:
            raise ValueError("Invalid MCTSDatasetConfig:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class TrainingBatch:
    """
    A batch of training examples.

    All tensors have shape [batch_size, ...].
    """

    # Code features
    code_tokens: Any  # torch.Tensor [batch_size, seq_len]
    code_attention_mask: Any  # torch.Tensor [batch_size, seq_len]

    # Problem features
    problem_tokens: Any  # torch.Tensor [batch_size, seq_len]
    problem_attention_mask: Any  # torch.Tensor [batch_size, seq_len]

    # Policy targets
    llm_policy: Any  # torch.Tensor [batch_size, max_actions]
    mcts_policy: Any  # torch.Tensor [batch_size, max_actions]
    action_mask: Any  # torch.Tensor [batch_size, max_actions]

    # Value targets
    llm_value: Any  # torch.Tensor [batch_size]
    outcome: Any  # torch.Tensor [batch_size]
    q_value: Any  # torch.Tensor [batch_size]

    # Metadata
    episode_ids: list[str]
    depths: Any  # torch.Tensor [batch_size]
    visits: Any  # torch.Tensor [batch_size]

    def to(self, device: str | Any) -> TrainingBatch:
        """Move batch to specified device."""
        if not _TORCH_AVAILABLE:
            return self

        return TrainingBatch(
            code_tokens=self.code_tokens.to(device),
            code_attention_mask=self.code_attention_mask.to(device),
            problem_tokens=self.problem_tokens.to(device),
            problem_attention_mask=self.problem_attention_mask.to(device),
            llm_policy=self.llm_policy.to(device),
            mcts_policy=self.mcts_policy.to(device),
            action_mask=self.action_mask.to(device),
            llm_value=self.llm_value.to(device),
            outcome=self.outcome.to(device),
            q_value=self.q_value.to(device),
            episode_ids=self.episode_ids,
            depths=self.depths.to(device),
            visits=self.visits.to(device),
        )


@dataclass
class RawExample:
    """Raw training example before tensor conversion."""

    state_code: str
    state_problem: str
    state_hash: str
    depth: int
    llm_action_probs: dict[str, float]
    mcts_action_probs: dict[str, float]
    llm_value_estimate: float
    outcome: float
    episode_id: str
    visits: int
    q_value: float


class MCTSDataset(Dataset if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    PyTorch Dataset for MCTS training data.

    Loads training examples from JSONL files and converts them to tensors
    suitable for training policy and value networks.
    """

    def __init__(
        self,
        config: MCTSDatasetConfig | None = None,
        examples: list[RawExample] | None = None,
    ):
        """
        Initialize dataset.

        Args:
            config: Dataset configuration
            examples: Pre-loaded examples (if None, load from config.data_dir)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MCTSDataset. Install with: pip install torch")

        self._config = config or MCTSDatasetConfig()
        self._config.validate()

        # Load or use provided examples
        if examples is not None:
            self._examples = examples
        else:
            self._examples = self._load_examples()

        # Initialize tokenizer
        self._tokenizer = self._init_tokenizer()

        logger.info(
            "Initialized MCTSDataset",
            num_examples=len(self._examples),
            data_dir=str(self._config.data_dir),
        )

    def _init_tokenizer(self) -> Any:
        """Initialize tokenizer for text encoding.

        Falls back to a simple character-level tokenizer when transformers
        is unavailable (ImportError) or the configured tokenizer cannot be
        downloaded (OSError, e.g. offline environments).
        """
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._config.tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except ImportError:
            logger.warning("transformers not available, using simple tokenization")
            return None
        except OSError as exc:
            logger.warning(
                f"Could not load tokenizer {self._config.tokenizer_name} ({exc}); " "using simple tokenization"
            )
            return None

    def _load_examples(self) -> list[RawExample]:
        """Load examples from data directory."""
        data_dir = Path(self._config.data_dir)

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return []

        examples = []
        episode_files = list(data_dir.glob("episode_*.jsonl"))

        for filepath in episode_files:
            episode_examples = self._load_episode_file(filepath)
            examples.extend(episode_examples)

        # Also check for split files
        splits_dir = data_dir / "splits"
        if splits_dir.exists():
            for split_file in splits_dir.glob("*.jsonl"):
                examples.extend(self._load_split_file(split_file))

        return examples

    def _load_episode_file(self, filepath: Path) -> list[RawExample]:
        """Load examples from an episode file."""
        examples = []
        episode_metadata = None

        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON in {filepath}:{line_num}: {e}",
                            filepath=str(filepath),
                            line_num=line_num,
                        )
                        continue

                    if "_metadata" in data:
                        episode_metadata = data["_metadata"]
                        # Filter by success if configured
                        if self._config.only_successful_episodes and not episode_metadata.get("solution_found", False):
                            return []  # Skip entire episode
                        continue

                    # Apply filters
                    if data.get("visits", 0) < self._config.min_visits:
                        continue

                    if self._config.exclude_root_nodes and data.get("depth", 0) == 0:
                        continue

                    example = RawExample(
                        state_code=data.get("state_code", ""),
                        state_problem=data.get("state_problem", ""),
                        state_hash=data.get("state_hash", ""),
                        depth=data.get("depth", 0),
                        llm_action_probs=data.get("llm_action_probs", {}),
                        mcts_action_probs=data.get("mcts_action_probs", {}),
                        llm_value_estimate=data.get("llm_value_estimate", 0.5),
                        outcome=data.get("outcome", 0.0),
                        episode_id=data.get("episode_id", ""),
                        visits=data.get("visits", 0),
                        q_value=data.get("q_value", 0.0),
                    )
                    examples.append(example)
        except OSError as e:
            logger.error(f"Failed to read episode file {filepath}: {e}")

        return examples

    def _load_split_file(self, filepath: Path) -> list[RawExample]:
        """Load examples from a split file (train/val/test)."""
        examples = []

        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON in {filepath}:{line_num}: {e}",
                            filepath=str(filepath),
                            line_num=line_num,
                        )
                        continue

                    # Apply filters
                    if data.get("visits", 0) < self._config.min_visits:
                        continue

                    if self._config.exclude_root_nodes and data.get("depth", 0) == 0:
                        continue

                    example = RawExample(
                        state_code=data.get("state_code", ""),
                        state_problem=data.get("state_problem", ""),
                        state_hash=data.get("state_hash", ""),
                        depth=data.get("depth", 0),
                        llm_action_probs=data.get("llm_action_probs", {}),
                        mcts_action_probs=data.get("mcts_action_probs", {}),
                        llm_value_estimate=data.get("llm_value_estimate", 0.5),
                        outcome=data.get("outcome", 0.0),
                        episode_id=data.get("episode_id", ""),
                        visits=data.get("visits", 0),
                        q_value=data.get("q_value", 0.0),
                    )
                    examples.append(example)
        except OSError as e:
            logger.error(f"Failed to read split file {filepath}: {e}")

        return examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single example as a dictionary."""
        example = self._examples[idx]
        return self._convert_example(example)

    def _convert_example(self, example: RawExample) -> dict[str, Any]:
        """Convert a raw example to tensor format."""
        # Tokenize code
        code_encoded = self._tokenize(example.state_code, self._config.max_code_length)

        # Tokenize problem
        problem_encoded = self._tokenize(example.state_problem, self._config.max_problem_length)

        # Convert action probabilities to fixed-size tensors
        llm_policy, mcts_policy, action_mask = self._encode_policies(
            example.llm_action_probs,
            example.mcts_action_probs,
        )

        return {
            "code_tokens": torch.tensor(code_encoded["input_ids"], dtype=torch.long),
            "code_attention_mask": torch.tensor(code_encoded["attention_mask"], dtype=torch.long),
            "problem_tokens": torch.tensor(problem_encoded["input_ids"], dtype=torch.long),
            "problem_attention_mask": torch.tensor(problem_encoded["attention_mask"], dtype=torch.long),
            "llm_policy": torch.tensor(llm_policy, dtype=torch.float32),
            "mcts_policy": torch.tensor(mcts_policy, dtype=torch.float32),
            "action_mask": torch.tensor(action_mask, dtype=torch.float32),
            "llm_value": torch.tensor(example.llm_value_estimate, dtype=torch.float32),
            "outcome": torch.tensor(example.outcome, dtype=torch.float32),
            "q_value": torch.tensor(example.q_value, dtype=torch.float32),
            "episode_id": example.episode_id,
            "depth": torch.tensor(example.depth, dtype=torch.long),
            "visits": torch.tensor(example.visits, dtype=torch.long),
        }

    def _tokenize(self, text: str, max_length: int) -> dict[str, list[int]]:
        """Tokenize text to fixed length."""
        if self._tokenizer is not None:
            encoded = self._tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        else:
            # Simple character-level encoding as fallback
            tokens = [ord(c) % 256 for c in text[:max_length]]
            padding = [0] * (max_length - len(tokens))
            attention = [1] * len(tokens) + [0] * len(padding)
            return {
                "input_ids": tokens + padding,
                "attention_mask": attention,
            }

    def _encode_policies(
        self,
        llm_probs: dict[str, float],
        mcts_probs: dict[str, float],
    ) -> tuple[list[float], list[float], list[float]]:
        """Encode action probabilities to fixed-size arrays."""
        max_actions = self._config.max_actions

        # Get all unique actions
        all_actions = sorted(set(llm_probs.keys()) | set(mcts_probs.keys()))

        llm_policy = []
        mcts_policy = []
        action_mask = []

        for i in range(max_actions):
            if i < len(all_actions):
                action = all_actions[i]
                llm_policy.append(llm_probs.get(action, 0.0))
                mcts_policy.append(mcts_probs.get(action, 0.0))
                action_mask.append(1.0)
            else:
                llm_policy.append(0.0)
                mcts_policy.append(0.0)
                action_mask.append(0.0)

        # Normalize
        llm_sum = sum(llm_policy)
        mcts_sum = sum(mcts_policy)

        if llm_sum > 0:
            llm_policy = [p / llm_sum for p in llm_policy]
        if mcts_sum > 0:
            mcts_policy = [p / mcts_sum for p in mcts_policy]

        return llm_policy, mcts_policy, action_mask

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        if not self._examples:
            return {"num_examples": 0}

        depths = [e.depth for e in self._examples]
        visits = [e.visits for e in self._examples]
        outcomes = [e.outcome for e in self._examples]
        q_values = [e.q_value for e in self._examples]
        episode_ids = {e.episode_id for e in self._examples}

        return {
            "num_examples": len(self._examples),
            "num_episodes": len(episode_ids),
            "depth_stats": {
                "mean": np.mean(depths),
                "std": np.std(depths),
                "min": min(depths),
                "max": max(depths),
            },
            "visits_stats": {
                "mean": np.mean(visits),
                "std": np.std(visits),
                "min": min(visits),
                "max": max(visits),
            },
            "outcome_stats": {
                "mean": np.mean(outcomes),
                "positive_rate": sum(1 for o in outcomes if o > 0) / len(outcomes),
            },
            "q_value_stats": {
                "mean": np.mean(q_values),
                "std": np.std(q_values),
                "min": min(q_values),
                "max": max(q_values),
            },
        }


def collate_fn(batch: list[dict[str, Any]]) -> TrainingBatch:
    """Collate function for DataLoader."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    return TrainingBatch(
        code_tokens=torch.stack([b["code_tokens"] for b in batch]),
        code_attention_mask=torch.stack([b["code_attention_mask"] for b in batch]),
        problem_tokens=torch.stack([b["problem_tokens"] for b in batch]),
        problem_attention_mask=torch.stack([b["problem_attention_mask"] for b in batch]),
        llm_policy=torch.stack([b["llm_policy"] for b in batch]),
        mcts_policy=torch.stack([b["mcts_policy"] for b in batch]),
        action_mask=torch.stack([b["action_mask"] for b in batch]),
        llm_value=torch.stack([b["llm_value"] for b in batch]),
        outcome=torch.stack([b["outcome"] for b in batch]),
        q_value=torch.stack([b["q_value"] for b in batch]),
        episode_ids=[b["episode_id"] for b in batch],
        depths=torch.stack([b["depth"] for b in batch]),
        visits=torch.stack([b["visits"] for b in batch]),
    )


def load_training_data(
    data_dir: str | Path,
    config: MCTSDatasetConfig | None = None,
) -> list[RawExample]:
    """
    Load training examples from a directory.

    Args:
        data_dir: Directory containing training data
        config: Optional configuration for filtering

    Returns:
        List of raw training examples
    """
    if config is None:
        config = MCTSDatasetConfig(data_dir=data_dir)
    else:
        config.data_dir = data_dir

    dataset = MCTSDataset(config=config)
    return dataset._examples


def create_dataloaders(
    data_dir: str | Path,
    config: MCTSDatasetConfig | None = None,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from training data.

    Args:
        data_dir: Directory containing training data
        config: Dataset configuration
        batch_size: Batch size for DataLoader
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of worker processes
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader. Install with: pip install torch")

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")

    # Load all examples
    if config is None:
        config = MCTSDatasetConfig(data_dir=data_dir)
    else:
        config.data_dir = data_dir

    # Create dataset to load examples
    full_dataset = MCTSDataset(config=config)
    all_examples = full_dataset._examples

    if not all_examples:
        raise ValueError(f"No training examples found in {data_dir}")

    # Shuffle and split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_examples))

    n = len(all_examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_examples = [all_examples[i] for i in train_indices]
    val_examples = [all_examples[i] for i in val_indices]
    test_examples = [all_examples[i] for i in test_indices]

    # Create datasets
    train_dataset = MCTSDataset(config=config, examples=train_examples)
    val_dataset = MCTSDataset(config=config, examples=val_examples)
    test_dataset = MCTSDataset(config=config, examples=test_examples)

    # Create DataLoaders
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
    }

    logger.info(
        "Created DataLoaders",
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        batch_size=batch_size,
    )

    return loaders
