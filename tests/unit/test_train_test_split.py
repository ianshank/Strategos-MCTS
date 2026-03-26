"""Unit tests for src/data/train_test_split module."""

import pytest

from src.data.dataset_loader import DatasetSample
from src.data.train_test_split import (
    BalancedSampler,
    CrossValidationFold,
    DataSplit,
    DataSplitter,
    StratifiedSplitter,
)


def _make_sample(id: str, domain: str = "math", difficulty: str = "easy", labels: list[str] | None = None) -> DatasetSample:
    """Helper to create a DatasetSample."""
    return DatasetSample(
        id=id,
        text=f"sample text for {id}",
        domain=domain,
        difficulty=difficulty,
        labels=labels,
    )


def _make_samples(n: int, domain: str = "math", difficulty: str = "easy") -> list[DatasetSample]:
    """Helper to create N samples."""
    return [_make_sample(f"s{i}", domain=domain, difficulty=difficulty) for i in range(n)]


@pytest.mark.unit
class TestDataSplitter:
    """Tests for DataSplitter."""

    def test_init_default_seed(self):
        splitter = DataSplitter()
        assert splitter.seed == 42

    def test_init_custom_seed(self):
        splitter = DataSplitter(seed=123)
        assert splitter.seed == 123

    def test_split_default_ratios(self):
        samples = _make_samples(100)
        splitter = DataSplitter(seed=42)
        result = splitter.split(samples)

        assert isinstance(result, DataSplit)
        assert len(result.train) == 70
        assert len(result.validation) == 15
        assert len(result.test) == 15
        assert len(result.train) + len(result.validation) + len(result.test) == 100

    def test_split_custom_ratios(self):
        samples = _make_samples(100)
        splitter = DataSplitter(seed=42)
        result = splitter.split(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        assert len(result.train) == 80
        assert len(result.validation) == 10
        assert len(result.test) == 10

    def test_split_invalid_ratios(self):
        samples = _make_samples(10)
        splitter = DataSplitter()
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            splitter.split(samples, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_split_empty_samples(self):
        splitter = DataSplitter()
        with pytest.raises(ValueError, match="Cannot split empty sample list"):
            splitter.split([])

    def test_split_no_shuffle(self):
        samples = _make_samples(10)
        splitter = DataSplitter(seed=42)
        result = splitter.split(samples, shuffle=False)
        # Without shuffle, first 7 should be train
        assert result.train == samples[:7]

    def test_split_reproducibility(self):
        samples = _make_samples(50)
        s1 = DataSplitter(seed=42).split(samples)
        s2 = DataSplitter(seed=42).split(samples)
        assert [s.id for s in s1.train] == [s.id for s in s2.train]
        assert [s.id for s in s1.validation] == [s.id for s in s2.validation]
        assert [s.id for s in s1.test] == [s.id for s in s2.test]

    def test_split_different_seeds_differ(self):
        samples = _make_samples(50)
        s1 = DataSplitter(seed=42).split(samples)
        s2 = DataSplitter(seed=99).split(samples)
        # With different seeds, the order should differ (extremely likely with 50 samples)
        assert [s.id for s in s1.train] != [s.id for s in s2.train]

    def test_split_info(self):
        samples = _make_samples(20)
        splitter = DataSplitter(seed=42)
        result = splitter.split(samples)

        info = result.split_info
        assert info["total_samples"] == 20
        assert info["seed"] == 42
        assert info["shuffled"] is True
        assert "train_samples" in info
        assert "val_samples" in info
        assert "test_samples" in info

    def test_split_does_not_modify_original(self):
        samples = _make_samples(20)
        original_ids = [s.id for s in samples]
        splitter = DataSplitter(seed=42)
        splitter.split(samples)
        assert [s.id for s in samples] == original_ids

    def test_split_small_dataset(self):
        samples = _make_samples(3)
        splitter = DataSplitter(seed=42)
        result = splitter.split(samples)
        total = len(result.train) + len(result.validation) + len(result.test)
        assert total == 3


@pytest.mark.unit
class TestDataSplitterKFolds:
    """Tests for DataSplitter.create_k_folds."""

    def test_create_k_folds_default(self):
        samples = _make_samples(20)
        splitter = DataSplitter(seed=42)
        folds = splitter.create_k_folds(samples)

        assert len(folds) == 5
        for fold in folds:
            assert isinstance(fold, CrossValidationFold)
            assert len(fold.train) + len(fold.validation) == 20

    def test_create_k_folds_custom_k(self):
        samples = _make_samples(30)
        splitter = DataSplitter(seed=42)
        folds = splitter.create_k_folds(samples, k=3)
        assert len(folds) == 3

    def test_create_k_folds_k_less_than_2(self):
        samples = _make_samples(10)
        splitter = DataSplitter()
        with pytest.raises(ValueError, match="k must be at least 2"):
            splitter.create_k_folds(samples, k=1)

    def test_create_k_folds_too_few_samples(self):
        samples = _make_samples(3)
        splitter = DataSplitter()
        with pytest.raises(ValueError, match="Need at least 5 samples"):
            splitter.create_k_folds(samples, k=5)

    def test_create_k_folds_fold_ids(self):
        samples = _make_samples(10)
        splitter = DataSplitter(seed=42)
        folds = splitter.create_k_folds(samples, k=5)
        assert [f.fold_id for f in folds] == [0, 1, 2, 3, 4]

    def test_create_k_folds_no_shuffle(self):
        samples = _make_samples(10)
        splitter = DataSplitter(seed=42)
        folds = splitter.create_k_folds(samples, shuffle=False)
        # First fold validation should be the first 2 samples
        assert len(folds[0].validation) == 2

    def test_create_k_folds_all_samples_used(self):
        samples = _make_samples(10)
        splitter = DataSplitter(seed=42)
        folds = splitter.create_k_folds(samples, k=2, shuffle=False)
        # Each sample should appear in validation exactly once across all folds
        all_val_ids = []
        for fold in folds:
            all_val_ids.extend(s.id for s in fold.validation)
        assert sorted(all_val_ids) == sorted(s.id for s in samples)


@pytest.mark.unit
class TestStratifiedSplitter:
    """Tests for StratifiedSplitter."""

    def test_init_defaults(self):
        splitter = StratifiedSplitter()
        assert splitter.seed == 42
        assert splitter.stratify_by == "domain"

    def test_init_custom(self):
        splitter = StratifiedSplitter(seed=10, stratify_by="difficulty")
        assert splitter.seed == 10
        assert splitter.stratify_by == "difficulty"

    def test_stratified_split_by_domain(self):
        # Create samples with two domains, each with 50 samples
        samples = _make_samples(50, domain="math") + _make_samples(50, domain="science")
        # Fix IDs so they're unique
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        splitter = StratifiedSplitter(seed=42, stratify_by="domain")
        result = splitter.split(samples)

        # All samples should be accounted for
        assert len(result.train) + len(result.validation) + len(result.test) == 100

        # Both domains should appear in each split
        train_domains = {s.domain for s in result.train}
        val_domains = {s.domain for s in result.validation}
        test_domains = {s.domain for s in result.test}
        assert "math" in train_domains
        assert "science" in train_domains
        assert "math" in val_domains or "science" in val_domains

    def test_stratified_split_empty(self):
        splitter = StratifiedSplitter()
        with pytest.raises(ValueError, match="Cannot split empty sample list"):
            splitter.split([])

    def test_stratified_split_invalid_ratios(self):
        samples = _make_samples(10)
        splitter = StratifiedSplitter()
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            splitter.split(samples, train_ratio=0.9, val_ratio=0.9, test_ratio=0.1)

    def test_stratified_split_info_contains_stratification(self):
        samples = _make_samples(20, domain="math") + _make_samples(20, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        splitter = StratifiedSplitter(seed=42, stratify_by="domain")
        result = splitter.split(samples)

        assert result.split_info["stratify_by"] == "domain"
        assert "stratification_info" in result.split_info

    def test_stratified_split_by_difficulty(self):
        samples = (
            _make_samples(30, difficulty="easy")
            + _make_samples(30, difficulty="medium")
            + _make_samples(30, difficulty="hard")
        )
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        splitter = StratifiedSplitter(seed=42, stratify_by="difficulty")
        result = splitter.split(samples)
        assert len(result.train) + len(result.validation) + len(result.test) == 90

    def test_stratified_split_by_labels(self):
        samples = []
        for i in range(20):
            labels = ["cat"] if i < 10 else ["dog"]
            samples.append(_make_sample(f"s{i}", labels=labels))

        splitter = StratifiedSplitter(seed=42, stratify_by="labels")
        result = splitter.split(samples)
        assert len(result.train) + len(result.validation) + len(result.test) == 20

    def test_get_stratify_key_unknown_attr(self):
        splitter = StratifiedSplitter(stratify_by="nonexistent_field")
        sample = _make_sample("s0")
        key = splitter._get_stratify_key(sample)
        assert key == "unknown"

    def test_get_stratify_key_none_domain(self):
        splitter = StratifiedSplitter(stratify_by="domain")
        sample = _make_sample("s0")
        sample.domain = None
        key = splitter._get_stratify_key(sample)
        assert key == "unknown"


@pytest.mark.unit
class TestStratifiedKFolds:
    """Tests for StratifiedSplitter.create_stratified_k_folds."""

    def test_create_stratified_k_folds(self):
        samples = _make_samples(20, domain="math") + _make_samples(20, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        splitter = StratifiedSplitter(seed=42, stratify_by="domain")
        folds = splitter.create_stratified_k_folds(samples, k=4)

        assert len(folds) == 4
        for fold in folds:
            assert isinstance(fold, CrossValidationFold)
            assert len(fold.train) + len(fold.validation) == 40

    def test_create_stratified_k_folds_k_less_than_2(self):
        samples = _make_samples(10)
        splitter = StratifiedSplitter()
        with pytest.raises(ValueError, match="k must be at least 2"):
            splitter.create_stratified_k_folds(samples, k=1)


@pytest.mark.unit
class TestBalancedSampler:
    """Tests for BalancedSampler."""

    def test_init(self):
        sampler = BalancedSampler(seed=99)
        assert sampler.seed == 99

    def test_oversample_minority(self):
        # 20 math, 5 science -> should oversample science to 20
        samples = _make_samples(20, domain="math") + _make_samples(5, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        sampler = BalancedSampler(seed=42)
        balanced = sampler.oversample_minority(samples, target_key="domain", target_ratio=1.0)

        # Should have 20 math + 20 science = 40
        assert len(balanced) == 40

    def test_oversample_minority_already_balanced(self):
        samples = _make_samples(10, domain="math") + _make_samples(10, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        sampler = BalancedSampler(seed=42)
        balanced = sampler.oversample_minority(samples, target_key="domain")
        assert len(balanced) == 20  # no oversampling needed

    def test_oversample_marks_metadata(self):
        samples = _make_samples(10, domain="math") + _make_samples(2, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        sampler = BalancedSampler(seed=42)
        balanced = sampler.oversample_minority(samples, target_key="domain")

        oversampled = [s for s in balanced if s.metadata.get("oversampled")]
        assert len(oversampled) > 0
        # All oversampled items should be science domain
        for s in oversampled:
            assert s.domain == "science"

    def test_undersample_majority(self):
        # 20 math, 5 science -> should undersample math to 5
        samples = _make_samples(20, domain="math") + _make_samples(5, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        sampler = BalancedSampler(seed=42)
        balanced = sampler.undersample_majority(samples, target_key="domain", target_ratio=1.0)

        assert len(balanced) == 10  # 5 math + 5 science

    def test_undersample_majority_already_balanced(self):
        samples = _make_samples(10, domain="math") + _make_samples(10, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        sampler = BalancedSampler(seed=42)
        balanced = sampler.undersample_majority(samples, target_key="domain")
        assert len(balanced) == 20

    def test_get_class_distribution(self):
        samples = _make_samples(15, domain="math") + _make_samples(5, domain="science")
        for i, s in enumerate(samples):
            s.id = f"s{i}"

        sampler = BalancedSampler()
        dist = sampler.get_class_distribution(samples, target_key="domain")

        assert dist["math"] == 15
        assert dist["science"] == 5

    def test_get_class_distribution_none_domain(self):
        samples = [_make_sample("s0")]
        samples[0].domain = None

        sampler = BalancedSampler()
        dist = sampler.get_class_distribution(samples, target_key="domain")
        assert dist["unknown"] == 1

    def test_get_class_distribution_empty(self):
        sampler = BalancedSampler()
        dist = sampler.get_class_distribution([], target_key="domain")
        assert dist == {}
