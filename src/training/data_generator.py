"""
Synthetic data generator for training Neural Meta-Controllers.

Provides functionality to generate synthetic training data for meta-controllers
that learn to select the optimal agent (HRM, TRM, or MCTS) based on system state.
"""

import json
import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from src.agents.meta_controller.base import MetaControllerFeatures
from src.agents.meta_controller.utils import features_to_text, normalize_features

logger = logging.getLogger(__name__)


class MetaControllerDataGenerator:
    """
    Synthetic data generator for training neural meta-controllers.

    Generates labeled training data by creating random feature vectors
    and determining the optimal agent based on weighted scoring rules.
    The generator supports balanced and unbalanced datasets, multiple
    output formats (tensors, text), and dataset persistence.

    Attributes:
        seed: Random seed for reproducibility.
        rng: NumPy random number generator instance.
        AGENT_NAMES: List of valid agent names.
        LABEL_TO_INDEX: Mapping from agent names to numeric indices.
        INDEX_TO_LABEL: Mapping from numeric indices to agent names.
    """

    AGENT_NAMES = ["hrm", "trm", "mcts"]
    LABEL_TO_INDEX = {"hrm": 0, "trm": 1, "mcts": 2}
    INDEX_TO_LABEL = {0: "hrm", 1: "trm", 2: "mcts"}

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the data generator with a random seed.

        Args:
            seed: Random seed for reproducibility. Defaults to 42.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> generator.seed
            42
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.debug("MetaControllerDataGenerator initialized: seed=%d", seed)

    def generate_single_sample(self) -> tuple[MetaControllerFeatures, str]:
        """
        Generate a single training sample with features and optimal agent label.

        Creates random features and determines the optimal agent based on
        weighted scoring rules:
        - If hrm_confidence > 0.7 and highest: select "hrm"
        - If trm_confidence > 0.7 and highest: select "trm"
        - If mcts_value > 0.6 and iteration > 3: select "mcts"
        - Otherwise: select agent with highest score

        Returns:
            Tuple of (MetaControllerFeatures, optimal_agent_label).

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, label = generator.generate_single_sample()
            >>> isinstance(features, MetaControllerFeatures)
            True
            >>> label in ['hrm', 'trm', 'mcts']
            True
        """
        # Generate random features
        hrm_confidence = float(self.rng.uniform(0, 1))
        trm_confidence = float(self.rng.uniform(0, 1))
        mcts_value = float(self.rng.uniform(0, 1))

        # Consensus score is average of confidences plus noise
        avg_confidence = (hrm_confidence + trm_confidence + mcts_value) / 3.0
        noise = float(self.rng.uniform(-0.1, 0.1))
        consensus_score = float(np.clip(avg_confidence + noise, 0.0, 1.0))

        # Random categorical and discrete features
        last_agent = str(self.rng.choice(["none", "hrm", "trm", "mcts"]))
        iteration = int(self.rng.integers(0, 11))  # [0, 10] inclusive
        query_length = int(self.rng.integers(10, 5001))  # [10, 5000] inclusive
        has_rag_context = bool(self.rng.choice([True, False]))

        features = MetaControllerFeatures(
            hrm_confidence=hrm_confidence,
            trm_confidence=trm_confidence,
            mcts_value=mcts_value,
            consensus_score=consensus_score,
            last_agent=last_agent,
            iteration=iteration,
            query_length=query_length,
            has_rag_context=has_rag_context,
        )

        # Determine optimal agent based on weighted scoring
        optimal_agent = self._determine_optimal_agent(features)

        return features, optimal_agent

    def _determine_optimal_agent(self, features: MetaControllerFeatures) -> str:
        """
        Determine the optimal agent based on weighted scoring rules.

        Args:
            features: MetaControllerFeatures to evaluate.

        Returns:
            Name of the optimal agent ('hrm', 'trm', or 'mcts').
        """
        hrm_conf = features.hrm_confidence
        trm_conf = features.trm_confidence
        mcts_val = features.mcts_value

        # Check if HRM should be selected (high confidence and highest)
        if hrm_conf > 0.7 and hrm_conf > trm_conf and hrm_conf > mcts_val:
            return "hrm"

        # Check if TRM should be selected (high confidence and highest)
        if trm_conf > 0.7 and trm_conf > hrm_conf and trm_conf > mcts_val:
            return "trm"

        # Check if MCTS should be selected (good value and enough iterations)
        if mcts_val > 0.6 and features.iteration > 3:
            return "mcts"

        # Default: select agent with highest score
        scores = {"hrm": hrm_conf, "trm": trm_conf, "mcts": mcts_val}
        return max(scores, key=lambda k: scores[k])

    def generate_dataset(self, num_samples: int = 1000) -> tuple[list[MetaControllerFeatures], list[str]]:
        """
        Generate a dataset with the specified number of samples.

        Creates an unbalanced dataset where the distribution of labels
        depends on the random feature generation and scoring rules.

        Args:
            num_samples: Number of samples to generate. Defaults to 1000.

        Returns:
            Tuple of (features_list, labels_list).

        Raises:
            ValueError: If num_samples is not positive.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, labels = generator.generate_dataset(100)
            >>> len(features)
            100
            >>> len(labels)
            100
            >>> all(isinstance(f, MetaControllerFeatures) for f in features)
            True
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        logger.info("Generating dataset with %d samples", num_samples)

        features_list: list[MetaControllerFeatures] = []
        labels_list: list[str] = []

        for _ in range(num_samples):
            features, label = self.generate_single_sample()
            features_list.append(features)
            labels_list.append(label)

        logger.info("Dataset generation complete: %d samples generated", len(features_list))
        return features_list, labels_list

    def generate_balanced_dataset(
        self, num_samples_per_class: int = 500
    ) -> tuple[list[MetaControllerFeatures], list[str]]:
        """
        Generate a balanced dataset with equal samples per agent class.

        Creates samples biased toward each agent class to ensure balanced
        representation. This is useful for training when class imbalance
        would otherwise affect model performance.

        Args:
            num_samples_per_class: Number of samples per agent class.
                Defaults to 500.

        Returns:
            Tuple of (features_list, labels_list) with balanced classes.

        Raises:
            ValueError: If num_samples_per_class is not positive.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, labels = generator.generate_balanced_dataset(10)
            >>> labels.count('hrm')
            10
            >>> labels.count('trm')
            10
            >>> labels.count('mcts')
            10
        """
        if num_samples_per_class <= 0:
            raise ValueError(f"num_samples_per_class must be positive, got {num_samples_per_class}")

        features_list: list[MetaControllerFeatures] = []
        labels_list: list[str] = []

        # Generate samples for each class
        for target_agent in self.AGENT_NAMES:
            count = 0
            max_attempts = num_samples_per_class * 100  # Prevent infinite loop

            attempts = 0
            while count < num_samples_per_class and attempts < max_attempts:
                attempts += 1
                features = self._generate_biased_features(target_agent)
                label = self._determine_optimal_agent(features)

                if label == target_agent:
                    features_list.append(features)
                    labels_list.append(label)
                    count += 1

            # If we couldn't generate enough samples, force generate the rest
            while count < num_samples_per_class:
                features = self._generate_forced_features(target_agent)
                features_list.append(features)
                labels_list.append(target_agent)
                count += 1

        return features_list, labels_list

    def _generate_biased_features(self, target_agent: str) -> MetaControllerFeatures:
        """
        Generate features biased toward selecting a specific agent.

        Args:
            target_agent: The agent to bias toward ('hrm', 'trm', or 'mcts').

        Returns:
            MetaControllerFeatures biased toward the target agent.
        """
        if target_agent == "hrm":
            # Bias toward high HRM confidence
            hrm_confidence = float(self.rng.uniform(0.7, 1.0))
            trm_confidence = float(self.rng.uniform(0, hrm_confidence - 0.1))
            mcts_value = float(self.rng.uniform(0, hrm_confidence - 0.1))
        elif target_agent == "trm":
            # Bias toward high TRM confidence
            trm_confidence = float(self.rng.uniform(0.7, 1.0))
            hrm_confidence = float(self.rng.uniform(0, trm_confidence - 0.1))
            mcts_value = float(self.rng.uniform(0, trm_confidence - 0.1))
        else:  # mcts
            # Bias toward high MCTS value with enough iterations
            mcts_value = float(self.rng.uniform(0.6, 1.0))
            hrm_confidence = float(self.rng.uniform(0, 0.7))
            trm_confidence = float(self.rng.uniform(0, 0.7))

        # Ensure valid ranges
        hrm_confidence = float(np.clip(hrm_confidence, 0.0, 1.0))
        trm_confidence = float(np.clip(trm_confidence, 0.0, 1.0))
        mcts_value = float(np.clip(mcts_value, 0.0, 1.0))

        avg_confidence = (hrm_confidence + trm_confidence + mcts_value) / 3.0
        noise = float(self.rng.uniform(-0.1, 0.1))
        consensus_score = float(np.clip(avg_confidence + noise, 0.0, 1.0))

        last_agent = str(self.rng.choice(["none", "hrm", "trm", "mcts"]))

        # For MCTS, bias iteration to be > 3
        iteration = int(self.rng.integers(4, 11)) if target_agent == "mcts" else int(self.rng.integers(0, 11))

        query_length = int(self.rng.integers(10, 5001))
        has_rag_context = bool(self.rng.choice([True, False]))

        return MetaControllerFeatures(
            hrm_confidence=hrm_confidence,
            trm_confidence=trm_confidence,
            mcts_value=mcts_value,
            consensus_score=consensus_score,
            last_agent=last_agent,
            iteration=iteration,
            query_length=query_length,
            has_rag_context=has_rag_context,
        )

    def _generate_forced_features(self, target_agent: str) -> MetaControllerFeatures:
        """
        Generate features that will definitely select a specific agent.

        Args:
            target_agent: The agent to force selection of.

        Returns:
            MetaControllerFeatures that will result in target_agent selection.
        """
        if target_agent == "hrm":
            hrm_confidence = 0.85
            trm_confidence = 0.3
            mcts_value = 0.3
            iteration = 2
        elif target_agent == "trm":
            hrm_confidence = 0.3
            trm_confidence = 0.85
            mcts_value = 0.3
            iteration = 2
        else:  # mcts
            hrm_confidence = 0.5
            trm_confidence = 0.5
            mcts_value = 0.75
            iteration = 5

        avg_confidence = (hrm_confidence + trm_confidence + mcts_value) / 3.0
        noise = float(self.rng.uniform(-0.05, 0.05))
        consensus_score = float(np.clip(avg_confidence + noise, 0.0, 1.0))

        return MetaControllerFeatures(
            hrm_confidence=hrm_confidence,
            trm_confidence=trm_confidence,
            mcts_value=mcts_value,
            consensus_score=consensus_score,
            last_agent=str(self.rng.choice(["none", "hrm", "trm", "mcts"])),
            iteration=iteration,
            query_length=int(self.rng.integers(10, 5001)),
            has_rag_context=bool(self.rng.choice([True, False])),
        )

    def to_tensor_dataset(
        self, features_list: list[MetaControllerFeatures], labels_list: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert features and labels to PyTorch tensors.

        Uses normalize_features to convert each feature set to a 10-dimensional
        vector, and converts string labels to numeric indices.

        Args:
            features_list: List of MetaControllerFeatures instances.
            labels_list: List of agent name strings ('hrm', 'trm', 'mcts').

        Returns:
            Tuple of (X tensor shape (N, 10), y tensor shape (N,)).
            X contains normalized features as float32.
            y contains label indices as int64.

        Raises:
            ValueError: If lists have different lengths or are empty.
            KeyError: If labels_list contains invalid agent names.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, labels = generator.generate_dataset(10)
            >>> X, y = generator.to_tensor_dataset(features, labels)
            >>> X.shape
            torch.Size([10, 10])
            >>> y.shape
            torch.Size([10])
            >>> X.dtype
            torch.float32
            >>> y.dtype
            torch.int64
        """
        if len(features_list) != len(labels_list):
            raise ValueError(
                f"features_list and labels_list must have same length, got {len(features_list)} and {len(labels_list)}"
            )

        if len(features_list) == 0:
            raise ValueError("Cannot convert empty dataset to tensors")

        # Convert features to normalized vectors
        X_list = [normalize_features(f) for f in features_list]
        X = torch.tensor(X_list, dtype=torch.float32)

        # Convert labels to indices
        try:
            y_list = [self.LABEL_TO_INDEX[label] for label in labels_list]
        except KeyError as e:
            raise KeyError(f"Invalid agent label: {e}. Valid labels: {self.AGENT_NAMES}") from e
        y = torch.tensor(y_list, dtype=torch.int64)

        return X, y

    def to_text_dataset(
        self, features_list: list[MetaControllerFeatures], labels_list: list[str]
    ) -> tuple[list[str], list[int]]:
        """
        Convert features to text format and labels to indices.

        Uses features_to_text to create human-readable text representations
        suitable for text-based models like BERT.

        Args:
            features_list: List of MetaControllerFeatures instances.
            labels_list: List of agent name strings ('hrm', 'trm', 'mcts').

        Returns:
            Tuple of (text_list, label_indices).
            text_list contains structured text representations.
            label_indices contains integer indices for each label.

        Raises:
            ValueError: If lists have different lengths.
            KeyError: If labels_list contains invalid agent names.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, labels = generator.generate_dataset(10)
            >>> texts, indices = generator.to_text_dataset(features, labels)
            >>> len(texts)
            10
            >>> all(isinstance(t, str) for t in texts)
            True
            >>> all(i in [0, 1, 2] for i in indices)
            True
        """
        if len(features_list) != len(labels_list):
            raise ValueError(
                f"features_list and labels_list must have same length, got {len(features_list)} and {len(labels_list)}"
            )

        # Convert features to text
        text_list = [features_to_text(f) for f in features_list]

        # Convert labels to indices
        try:
            label_indices = [self.LABEL_TO_INDEX[label] for label in labels_list]
        except KeyError as e:
            raise KeyError(f"Invalid agent label: {e}. Valid labels: {self.AGENT_NAMES}") from e

        return text_list, label_indices

    def split_dataset(
        self,
        X: Any,
        y: Any,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> dict[str, Any]:
        """
        Split dataset into train, validation, and test sets.

        Shuffles the data and splits it according to the specified ratios.
        The test ratio is automatically calculated as (1 - train_ratio - val_ratio).

        Args:
            X: Feature data (tensor, array, or list).
            y: Label data (tensor, array, or list).
            train_ratio: Proportion for training set. Defaults to 0.7.
            val_ratio: Proportion for validation set. Defaults to 0.15.

        Returns:
            Dictionary with keys:
            - 'X_train': Training features
            - 'y_train': Training labels
            - 'X_val': Validation features
            - 'y_val': Validation labels
            - 'X_test': Test features
            - 'y_test': Test labels

        Raises:
            ValueError: If ratios are invalid or data sizes don't match.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, labels = generator.generate_dataset(100)
            >>> X, y = generator.to_tensor_dataset(features, labels)
            >>> splits = generator.split_dataset(X, y, 0.7, 0.15)
            >>> 'X_train' in splits
            True
            >>> splits['X_train'].shape[0] == 70
            True
        """
        # Validate ratios
        if not (0 < train_ratio < 1):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if not (0 < val_ratio < 1):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        if train_ratio + val_ratio >= 1:
            raise ValueError(f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}")

        # Get dataset size
        n_samples = X.shape[0] if isinstance(X, (torch.Tensor, np.ndarray)) else len(X)

        n_labels = y.shape[0] if isinstance(y, (torch.Tensor, np.ndarray)) else len(y)

        if n_samples != n_labels:
            raise ValueError(f"X and y must have same number of samples, got {n_samples} and {n_labels}")

        if n_samples == 0:
            raise ValueError("Cannot split empty dataset")

        # Generate shuffled indices
        indices = self.rng.permutation(n_samples)

        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Split data based on type
        if isinstance(X, (torch.Tensor, np.ndarray)):
            X_train = X[train_indices]
            X_val = X[val_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_val = y[val_indices]
            y_test = y[test_indices]
        else:
            # Assume list-like
            X_train = [X[i] for i in train_indices]
            X_val = [X[i] for i in val_indices]
            X_test = [X[i] for i in test_indices]
            y_train = [y[i] for i in train_indices]
            y_val = [y[i] for i in val_indices]
            y_test = [y[i] for i in test_indices]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def save_dataset(
        self,
        features_list: list[MetaControllerFeatures],
        labels_list: list[str],
        path: str,
    ) -> None:
        """
        Save dataset to a JSON file.

        Converts MetaControllerFeatures to dictionaries for JSON serialization.

        Args:
            features_list: List of MetaControllerFeatures instances.
            labels_list: List of agent name strings.
            path: Path to save the JSON file.

        Raises:
            ValueError: If lists have different lengths.
            IOError: If file cannot be written.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, labels = generator.generate_dataset(10)
            >>> generator.save_dataset(features, labels, 'dataset.json')
        """
        if len(features_list) != len(labels_list):
            raise ValueError(
                f"features_list and labels_list must have same length, got {len(features_list)} and {len(labels_list)}"
            )

        # Convert to serializable format
        data = {
            "seed": self.seed,
            "num_samples": len(features_list),
            "samples": [
                {"features": asdict(f), "label": label} for f, label in zip(features_list, labels_list, strict=False)
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Dataset saved to %s (%d samples)", path, len(features_list))

    def load_dataset(self, path: str) -> tuple[list[MetaControllerFeatures], list[str]]:
        """
        Load dataset from a JSON file.

        Reconstructs MetaControllerFeatures from saved dictionaries.

        Args:
            path: Path to the JSON file to load.

        Returns:
            Tuple of (features_list, labels_list).

        Raises:
            IOError: If file cannot be read.
            KeyError: If JSON structure is invalid.
            TypeError: If data types are incorrect.

        Example:
            >>> generator = MetaControllerDataGenerator(seed=42)
            >>> features, labels = generator.load_dataset('dataset.json')
            >>> isinstance(features[0], MetaControllerFeatures)
            True
        """
        logger.info("Loading dataset from %s", path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        features_list: list[MetaControllerFeatures] = []
        labels_list: list[str] = []

        for sample in data["samples"]:
            features_dict = sample["features"]
            features = MetaControllerFeatures(
                hrm_confidence=float(features_dict["hrm_confidence"]),
                trm_confidence=float(features_dict["trm_confidence"]),
                mcts_value=float(features_dict["mcts_value"]),
                consensus_score=float(features_dict["consensus_score"]),
                last_agent=str(features_dict["last_agent"]),
                iteration=int(features_dict["iteration"]),
                query_length=int(features_dict["query_length"]),
                has_rag_context=bool(features_dict["has_rag_context"]),
            )
            features_list.append(features)
            labels_list.append(str(sample["label"]))

        return features_list, labels_list
