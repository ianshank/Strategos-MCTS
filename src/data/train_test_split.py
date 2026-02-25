"""
Data Splitting Module for Training Pipeline.

Provides utilities for:
- Train/validation/test splitting
- Stratified sampling by domain or difficulty
- Cross-validation fold creation
- Reproducible splits with seeding
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .dataset_loader import DatasetSample

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Result of dataset splitting."""

    train: list[DatasetSample]
    validation: list[DatasetSample]
    test: list[DatasetSample]
    split_info: dict[str, Any]


@dataclass
class CrossValidationFold:
    """Single fold for cross-validation."""

    fold_id: int
    train: list[DatasetSample]
    validation: list[DatasetSample]


class DataSplitter:
    """
    Basic dataset splitter with random sampling.

    Provides reproducible train/validation/test splits
    with configurable ratios.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        import random

        self.rng = random.Random(seed)

    def split(
        self,
        samples: list[DatasetSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
    ) -> DataSplit:
        """
        Split dataset into train/validation/test sets.

        Args:
            samples: List of all samples
            train_ratio: Proportion for training (default 0.7)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)
            shuffle: Whether to shuffle before splitting

        Returns:
            DataSplit with train, validation, and test sets
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")

        if not samples:
            raise ValueError("Cannot split empty sample list")

        # Copy and optionally shuffle
        all_samples = list(samples)
        if shuffle:
            self.rng.shuffle(all_samples)

        n = len(all_samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_samples = all_samples[:train_end]
        val_samples = all_samples[train_end:val_end]
        test_samples = all_samples[val_end:]

        split_info = {
            "total_samples": n,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "train_ratio": len(train_samples) / n,
            "val_ratio": len(val_samples) / n,
            "test_ratio": len(test_samples) / n,
            "seed": self.seed,
            "shuffled": shuffle,
        }

        logger.info(f"Split {n} samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

        return DataSplit(
            train=train_samples,
            validation=val_samples,
            test=test_samples,
            split_info=split_info,
        )

    def create_k_folds(
        self,
        samples: list[DatasetSample],
        k: int = 5,
        shuffle: bool = True,
    ) -> list[CrossValidationFold]:
        """
        Create k-fold cross-validation splits.

        Args:
            samples: List of all samples
            k: Number of folds
            shuffle: Whether to shuffle before splitting

        Returns:
            List of CrossValidationFold objects
        """
        if k < 2:
            raise ValueError("k must be at least 2")

        if len(samples) < k:
            raise ValueError(f"Need at least {k} samples for {k}-fold CV")

        # Copy and optionally shuffle
        all_samples = list(samples)
        if shuffle:
            self.rng.shuffle(all_samples)

        # Calculate fold sizes
        fold_size = len(all_samples) // k
        folds = []

        for fold_id in range(k):
            # Validation is the current fold
            val_start = fold_id * fold_size
            val_end = len(all_samples) if fold_id == k - 1 else val_start + fold_size  # noqa: SIM108

            val_samples = all_samples[val_start:val_end]
            train_samples = all_samples[:val_start] + all_samples[val_end:]

            folds.append(
                CrossValidationFold(
                    fold_id=fold_id,
                    train=train_samples,
                    validation=val_samples,
                )
            )

        logger.info(f"Created {k}-fold cross-validation splits")
        return folds


class StratifiedSplitter(DataSplitter):
    """
    Stratified dataset splitter.

    Ensures proportional representation of categories
    (domain, difficulty, etc.) across splits.
    """

    def __init__(self, seed: int = 42, stratify_by: str = "domain"):
        """
        Initialize stratified splitter.

        Args:
            seed: Random seed for reproducibility
            stratify_by: Attribute to stratify on ('domain', 'difficulty', 'labels')
        """
        super().__init__(seed)
        self.stratify_by = stratify_by

    def split(
        self,
        samples: list[DatasetSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
    ) -> DataSplit:
        """
        Stratified split maintaining category proportions.

        Args:
            samples: List of all samples
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            shuffle: Whether to shuffle before splitting

        Returns:
            DataSplit with stratified train, validation, and test sets
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")

        if not samples:
            raise ValueError("Cannot split empty sample list")

        # Group samples by stratification key
        groups = defaultdict(list)
        for sample in samples:
            key = self._get_stratify_key(sample)
            groups[key].append(sample)

        # Split each group proportionally
        train_samples = []
        val_samples = []
        test_samples = []

        for _key, group_samples in groups.items():
            if shuffle:
                self.rng.shuffle(group_samples)

            n = len(group_samples)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_samples.extend(group_samples[:train_end])
            val_samples.extend(group_samples[train_end:val_end])
            test_samples.extend(group_samples[val_end:])

        # Final shuffle of combined sets
        if shuffle:
            self.rng.shuffle(train_samples)
            self.rng.shuffle(val_samples)
            self.rng.shuffle(test_samples)

        # Verify stratification
        stratify_info = self._verify_stratification(train_samples, val_samples, test_samples)

        split_info = {
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "train_ratio": len(train_samples) / len(samples),
            "val_ratio": len(val_samples) / len(samples),
            "test_ratio": len(test_samples) / len(samples),
            "stratify_by": self.stratify_by,
            "stratification_info": stratify_info,
            "seed": self.seed,
            "shuffled": shuffle,
        }

        logger.info(
            f"Stratified split ({self.stratify_by}): "
            f"train={len(train_samples)}, val={len(val_samples)}, "
            f"test={len(test_samples)}"
        )

        return DataSplit(
            train=train_samples,
            validation=val_samples,
            test=test_samples,
            split_info=split_info,
        )

    def _get_stratify_key(self, sample: DatasetSample) -> str:
        """Get stratification key for a sample."""
        if self.stratify_by == "domain":
            return sample.domain or "unknown"
        elif self.stratify_by == "difficulty":
            return sample.difficulty or "unknown"
        elif self.stratify_by == "labels":
            return ",".join(sorted(sample.labels)) if sample.labels else "unknown"
        else:
            return str(getattr(sample, self.stratify_by, "unknown"))

    def _verify_stratification(
        self,
        train: list[DatasetSample],
        val: list[DatasetSample],
        test: list[DatasetSample],
    ) -> dict[str, dict[str, float]]:
        """
        Verify that stratification was successful.

        Returns dictionary showing distribution of stratification key
        across train/val/test splits.
        """

        def get_distribution(samples: list[DatasetSample]) -> dict[str, float]:
            if not samples:
                return {}
            counts: dict[str, int] = defaultdict(int)
            for sample in samples:
                key = self._get_stratify_key(sample)
                counts[key] += 1
            total = len(samples)
            return {k: v / total for k, v in counts.items()}

        return {
            "train": get_distribution(train),
            "validation": get_distribution(val),
            "test": get_distribution(test),
        }

    def create_stratified_k_folds(
        self,
        samples: list[DatasetSample],
        k: int = 5,
        shuffle: bool = True,
    ) -> list[CrossValidationFold]:
        """
        Create stratified k-fold cross-validation splits.

        Args:
            samples: List of all samples
            k: Number of folds
            shuffle: Whether to shuffle before splitting

        Returns:
            List of CrossValidationFold objects with stratification
        """
        if k < 2:
            raise ValueError("k must be at least 2")

        # Group samples by stratification key
        groups = defaultdict(list)
        for sample in samples:
            key = self._get_stratify_key(sample)
            groups[key].append(sample)

        # Initialize folds
        folds_data: list[dict[str, list[DatasetSample]]] = [{"train": [], "val": []} for _ in range(k)]

        # Distribute each group across folds
        for _key, group_samples in groups.items():
            if shuffle:
                self.rng.shuffle(group_samples)

            # Assign samples to folds
            fold_size = len(group_samples) // k
            for fold_id in range(k):
                val_start = fold_id * fold_size
                val_end = len(group_samples) if fold_id == k - 1 else val_start + fold_size

                for i, sample in enumerate(group_samples):
                    if val_start <= i < val_end:
                        folds_data[fold_id]["val"].append(sample)
                    else:
                        folds_data[fold_id]["train"].append(sample)

        # Create fold objects
        folds = [
            CrossValidationFold(
                fold_id=i,
                train=data["train"],
                validation=data["val"],
            )
            for i, data in enumerate(folds_data)
        ]

        logger.info(f"Created stratified {k}-fold cross-validation splits")
        return folds


class BalancedSampler:
    """
    Balanced sampling for imbalanced datasets.

    Provides utilities for:
    - Oversampling minority classes
    - Undersampling majority classes
    - SMOTE-like synthetic sampling (for numerical features)
    """

    def __init__(self, seed: int = 42):
        """Initialize balanced sampler."""
        self.seed = seed
        import random

        self.rng = random.Random(seed)

    def oversample_minority(
        self,
        samples: list[DatasetSample],
        target_key: str = "domain",
        target_ratio: float = 1.0,
    ) -> list[DatasetSample]:
        """
        Oversample minority classes to balance dataset.

        Args:
            samples: Original samples
            target_key: Attribute to balance on
            target_ratio: Target ratio relative to majority (1.0 = equal)

        Returns:
            Balanced sample list (originals + oversampled)
        """
        # Group by target key
        groups = defaultdict(list)
        for sample in samples:
            key = getattr(sample, target_key, "unknown") or "unknown"
            groups[key].append(sample)

        # Find majority class size
        max_count = max(len(g) for g in groups.values())
        target_count = int(max_count * target_ratio)

        # Oversample minority classes
        balanced = []
        for _key, group in groups.items():
            balanced.extend(group)

            # Oversample if needed
            if len(group) < target_count:
                num_to_add = target_count - len(group)
                for _ in range(num_to_add):
                    # Randomly duplicate from group
                    original = self.rng.choice(group)
                    duplicate = DatasetSample(
                        id=f"{original.id}_oversample_{self.rng.randint(0, 999999)}",
                        text=original.text,
                        metadata={**original.metadata, "oversampled": True},
                        labels=original.labels,
                        difficulty=original.difficulty,
                        domain=original.domain,
                        reasoning_steps=original.reasoning_steps,
                    )
                    balanced.append(duplicate)

        logger.info(f"Oversampled from {len(samples)} to {len(balanced)} samples")
        return balanced

    def undersample_majority(
        self,
        samples: list[DatasetSample],
        target_key: str = "domain",
        target_ratio: float = 1.0,
    ) -> list[DatasetSample]:
        """
        Undersample majority classes to balance dataset.

        Args:
            samples: Original samples
            target_key: Attribute to balance on
            target_ratio: Target ratio relative to minority (1.0 = equal)

        Returns:
            Balanced sample list (subset of originals)
        """
        # Group by target key
        groups = defaultdict(list)
        for sample in samples:
            key = getattr(sample, target_key, "unknown") or "unknown"
            groups[key].append(sample)

        # Find minority class size
        min_count = min(len(g) for g in groups.values())
        target_count = int(min_count * target_ratio)

        # Undersample majority classes
        balanced = []
        for _key, group in groups.items():
            if len(group) > target_count:
                # Randomly select target_count samples
                balanced.extend(self.rng.sample(group, target_count))
            else:
                balanced.extend(group)

        logger.info(f"Undersampled from {len(samples)} to {len(balanced)} samples")
        return balanced

    def get_class_distribution(
        self,
        samples: list[DatasetSample],
        target_key: str = "domain",
    ) -> dict[str, int]:
        """
        Get distribution of classes.

        Args:
            samples: Sample list
            target_key: Attribute to analyze

        Returns:
            Dictionary of class counts
        """
        distribution: dict[str, int] = defaultdict(int)
        for sample in samples:
            key = getattr(sample, target_key, "unknown") or "unknown"
            distribution[key] += 1
        return dict(distribution)
