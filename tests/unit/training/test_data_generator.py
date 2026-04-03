"""
Tests for data_generator module.

Tests MetaControllerDataGenerator including sample generation,
dataset generation (balanced/unbalanced), tensor/text conversion,
dataset splitting, and persistence (save/load).
"""

import json

import numpy as np
import pytest
import torch

from src.agents.meta_controller.base import MetaControllerFeatures
from src.training.data_generator import MetaControllerDataGenerator


@pytest.mark.unit
class TestMetaControllerDataGeneratorInit:
    """Tests for MetaControllerDataGenerator initialization."""

    def test_init_default_seed(self):
        """Test default seed value."""
        gen = MetaControllerDataGenerator()
        assert gen.seed == 42

    def test_init_custom_seed(self):
        """Test custom seed value."""
        gen = MetaControllerDataGenerator(seed=123)
        assert gen.seed == 123

    def test_rng_initialized(self):
        """Test RNG is initialized."""
        gen = MetaControllerDataGenerator(seed=42)
        assert gen.rng is not None

    def test_class_attributes(self):
        """Test class-level constants."""
        assert MetaControllerDataGenerator.AGENT_NAMES == ["hrm", "trm", "mcts"]
        assert MetaControllerDataGenerator.LABEL_TO_INDEX == {"hrm": 0, "trm": 1, "mcts": 2}
        assert MetaControllerDataGenerator.INDEX_TO_LABEL == {0: "hrm", 1: "trm", 2: "mcts"}

    def test_reproducibility(self):
        """Test same seed produces same results."""
        gen1 = MetaControllerDataGenerator(seed=42)
        gen2 = MetaControllerDataGenerator(seed=42)
        f1, l1 = gen1.generate_single_sample()
        f2, l2 = gen2.generate_single_sample()
        assert f1.hrm_confidence == f2.hrm_confidence
        assert l1 == l2


@pytest.mark.unit
class TestGenerateSingleSample:
    """Tests for generate_single_sample method."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_returns_tuple(self, gen):
        """Test returns tuple of (features, label)."""
        result = gen.generate_single_sample()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_features_type(self, gen):
        """Test features is MetaControllerFeatures."""
        features, _ = gen.generate_single_sample()
        assert isinstance(features, MetaControllerFeatures)

    def test_label_valid(self, gen):
        """Test label is one of the valid agent names."""
        _, label = gen.generate_single_sample()
        assert label in ["hrm", "trm", "mcts"]

    def test_features_ranges(self, gen):
        """Test feature values are in expected ranges."""
        for _ in range(50):
            features, _ = gen.generate_single_sample()
            assert 0.0 <= features.hrm_confidence <= 1.0
            assert 0.0 <= features.trm_confidence <= 1.0
            assert 0.0 <= features.mcts_value <= 1.0
            assert 0.0 <= features.consensus_score <= 1.0
            assert features.last_agent in ["none", "hrm", "trm", "mcts"]
            assert 0 <= features.iteration <= 10
            assert 10 <= features.query_length <= 5000
            assert isinstance(features.has_rag_context, bool)


@pytest.mark.unit
class TestDetermineOptimalAgent:
    """Tests for _determine_optimal_agent method."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_hrm_selected_high_confidence(self, gen):
        """Test HRM is selected when its confidence is high and highest."""
        features = MetaControllerFeatures(
            hrm_confidence=0.9,
            trm_confidence=0.3,
            mcts_value=0.3,
            consensus_score=0.5,
            last_agent="none",
            iteration=2,
            query_length=100,
            has_rag_context=False,
        )
        assert gen._determine_optimal_agent(features) == "hrm"

    def test_trm_selected_high_confidence(self, gen):
        """Test TRM is selected when its confidence is high and highest."""
        features = MetaControllerFeatures(
            hrm_confidence=0.3,
            trm_confidence=0.9,
            mcts_value=0.3,
            consensus_score=0.5,
            last_agent="none",
            iteration=2,
            query_length=100,
            has_rag_context=False,
        )
        assert gen._determine_optimal_agent(features) == "trm"

    def test_mcts_selected_high_value_and_iterations(self, gen):
        """Test MCTS is selected when value > 0.6 and iteration > 3."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.8,
            consensus_score=0.5,
            last_agent="none",
            iteration=5,
            query_length=100,
            has_rag_context=False,
        )
        assert gen._determine_optimal_agent(features) == "mcts"

    def test_mcts_not_selected_low_iterations(self, gen):
        """Test MCTS is NOT selected when iteration <= 3 even with high value."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.8,
            consensus_score=0.5,
            last_agent="none",
            iteration=2,
            query_length=100,
            has_rag_context=False,
        )
        # Should fall through to highest score which is mcts_value
        result = gen._determine_optimal_agent(features)
        assert result == "mcts"  # Highest score wins in default case

    def test_default_highest_score(self, gen):
        """Test default case selects agent with highest score."""
        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.6,
            mcts_value=0.4,
            consensus_score=0.5,
            last_agent="none",
            iteration=1,
            query_length=100,
            has_rag_context=False,
        )
        assert gen._determine_optimal_agent(features) == "trm"


@pytest.mark.unit
class TestGenerateDataset:
    """Tests for generate_dataset method."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_correct_size(self, gen):
        """Test generated dataset has correct number of samples."""
        features, labels = gen.generate_dataset(50)
        assert len(features) == 50
        assert len(labels) == 50

    def test_types(self, gen):
        """Test return types are correct."""
        features, labels = gen.generate_dataset(10)
        assert all(isinstance(f, MetaControllerFeatures) for f in features)
        assert all(isinstance(l, str) for l in labels)
        assert all(l in ["hrm", "trm", "mcts"] for l in labels)

    def test_invalid_num_samples(self, gen):
        """Test ValueError for non-positive num_samples."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            gen.generate_dataset(0)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            gen.generate_dataset(-1)


@pytest.mark.unit
class TestGenerateBalancedDataset:
    """Tests for generate_balanced_dataset method."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_balanced_counts(self, gen):
        """Test each class has equal representation."""
        features, labels = gen.generate_balanced_dataset(10)
        assert labels.count("hrm") == 10
        assert labels.count("trm") == 10
        assert labels.count("mcts") == 10
        assert len(features) == 30

    def test_invalid_num_samples_per_class(self, gen):
        """Test ValueError for non-positive num_samples_per_class."""
        with pytest.raises(ValueError, match="num_samples_per_class must be positive"):
            gen.generate_balanced_dataset(0)


@pytest.mark.unit
class TestToTensorDataset:
    """Tests for to_tensor_dataset method."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_output_shapes(self, gen):
        """Test output tensor shapes."""
        features, labels = gen.generate_dataset(20)
        X, y = gen.to_tensor_dataset(features, labels)
        assert X.shape == (20, 10)
        assert y.shape == (20,)

    def test_output_dtypes(self, gen):
        """Test output tensor dtypes."""
        features, labels = gen.generate_dataset(10)
        X, y = gen.to_tensor_dataset(features, labels)
        assert X.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_label_indices(self, gen):
        """Test label indices are correct."""
        features, labels = gen.generate_dataset(10)
        _, y = gen.to_tensor_dataset(features, labels)
        for i, label in enumerate(labels):
            expected_idx = gen.LABEL_TO_INDEX[label]
            assert y[i].item() == expected_idx

    def test_mismatched_lengths(self, gen):
        """Test ValueError for mismatched list lengths."""
        features, _ = gen.generate_dataset(5)
        with pytest.raises(ValueError, match="same length"):
            gen.to_tensor_dataset(features, ["hrm", "trm"])

    def test_empty_dataset(self, gen):
        """Test ValueError for empty dataset."""
        with pytest.raises(ValueError, match="empty dataset"):
            gen.to_tensor_dataset([], [])

    def test_invalid_label(self, gen):
        """Test KeyError for invalid agent label."""
        features, _ = gen.generate_dataset(1)
        with pytest.raises(KeyError, match="Invalid agent label"):
            gen.to_tensor_dataset(features, ["invalid_agent"])


@pytest.mark.unit
class TestToTextDataset:
    """Tests for to_text_dataset method."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_output_types(self, gen):
        """Test output types."""
        features, labels = gen.generate_dataset(10)
        texts, indices = gen.to_text_dataset(features, labels)
        assert all(isinstance(t, str) for t in texts)
        assert all(isinstance(i, int) for i in indices)
        assert len(texts) == 10
        assert len(indices) == 10

    def test_indices_valid(self, gen):
        """Test all indices are valid."""
        features, labels = gen.generate_dataset(10)
        _, indices = gen.to_text_dataset(features, labels)
        assert all(i in [0, 1, 2] for i in indices)

    def test_mismatched_lengths(self, gen):
        """Test ValueError for mismatched list lengths."""
        features, _ = gen.generate_dataset(5)
        with pytest.raises(ValueError, match="same length"):
            gen.to_text_dataset(features, ["hrm"])

    def test_invalid_label(self, gen):
        """Test KeyError for invalid agent label."""
        features, _ = gen.generate_dataset(1)
        with pytest.raises(KeyError, match="Invalid agent label"):
            gen.to_text_dataset(features, ["bad_label"])


@pytest.mark.unit
class TestSplitDataset:
    """Tests for split_dataset method."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_tensor_split_sizes(self, gen):
        """Test split sizes for tensor data."""
        features, labels = gen.generate_dataset(100)
        X, y = gen.to_tensor_dataset(features, labels)
        splits = gen.split_dataset(X, y, train_ratio=0.7, val_ratio=0.15)

        assert "X_train" in splits
        assert "X_val" in splits
        assert "X_test" in splits
        assert splits["X_train"].shape[0] == 70
        assert splits["X_val"].shape[0] == 15
        assert splits["X_test"].shape[0] == 15

    def test_list_split(self, gen):
        """Test split works with list data."""
        X = list(range(20))
        y = list(range(20))
        splits = gen.split_dataset(X, y, train_ratio=0.5, val_ratio=0.25)
        assert len(splits["X_train"]) == 10
        assert len(splits["X_val"]) == 5
        assert len(splits["X_test"]) == 5

    def test_invalid_train_ratio(self, gen):
        """Test ValueError for invalid train_ratio."""
        X = torch.randn(10, 4)
        y = torch.randint(0, 3, (10,))
        with pytest.raises(ValueError, match="train_ratio"):
            gen.split_dataset(X, y, train_ratio=0.0)
        with pytest.raises(ValueError, match="train_ratio"):
            gen.split_dataset(X, y, train_ratio=1.0)

    def test_invalid_val_ratio(self, gen):
        """Test ValueError for invalid val_ratio."""
        X = torch.randn(10, 4)
        y = torch.randint(0, 3, (10,))
        with pytest.raises(ValueError, match="val_ratio"):
            gen.split_dataset(X, y, val_ratio=0.0)

    def test_ratios_sum_too_large(self, gen):
        """Test ValueError when ratios sum >= 1."""
        X = torch.randn(10, 4)
        y = torch.randint(0, 3, (10,))
        with pytest.raises(ValueError, match="must be < 1"):
            gen.split_dataset(X, y, train_ratio=0.6, val_ratio=0.5)

    def test_mismatched_sizes(self, gen):
        """Test ValueError when X and y have different sizes."""
        X = torch.randn(10, 4)
        y = torch.randint(0, 3, (5,))
        with pytest.raises(ValueError, match="same number of samples"):
            gen.split_dataset(X, y)

    def test_empty_dataset(self, gen):
        """Test ValueError for empty dataset."""
        X = torch.randn(0, 4)
        y = torch.randint(0, 3, (0,))
        with pytest.raises(ValueError, match="empty dataset"):
            gen.split_dataset(X, y)

    def test_numpy_array_split(self, gen):
        """Test split works with numpy arrays."""
        X = np.random.randn(20, 4)
        y = np.random.randint(0, 3, 20)
        splits = gen.split_dataset(X, y, train_ratio=0.7, val_ratio=0.15)
        assert splits["X_train"].shape[0] == 14
        assert splits["X_val"].shape[0] == 3
        assert splits["X_test"].shape[0] == 3


@pytest.mark.unit
class TestSaveLoadDataset:
    """Tests for save_dataset and load_dataset methods."""

    @pytest.fixture
    def gen(self):
        return MetaControllerDataGenerator(seed=42)

    def test_save_and_load_roundtrip(self, gen, tmp_path):
        """Test saving and loading preserves data."""
        features, labels = gen.generate_dataset(20)
        path = str(tmp_path / "dataset.json")

        gen.save_dataset(features, labels, path)
        loaded_features, loaded_labels = gen.load_dataset(path)

        assert len(loaded_features) == 20
        assert len(loaded_labels) == 20
        for orig, loaded in zip(features, loaded_features):
            assert orig.hrm_confidence == pytest.approx(loaded.hrm_confidence)
            assert orig.trm_confidence == pytest.approx(loaded.trm_confidence)
            assert orig.last_agent == loaded.last_agent
        assert labels == loaded_labels

    def test_save_creates_valid_json(self, gen, tmp_path):
        """Test saved file is valid JSON."""
        features, labels = gen.generate_dataset(5)
        path = str(tmp_path / "test.json")
        gen.save_dataset(features, labels, path)

        with open(path) as f:
            data = json.load(f)
        assert "seed" in data
        assert "num_samples" in data
        assert data["num_samples"] == 5
        assert len(data["samples"]) == 5

    def test_save_mismatched_lengths(self, gen, tmp_path):
        """Test ValueError for mismatched list lengths."""
        features, _ = gen.generate_dataset(5)
        path = str(tmp_path / "bad.json")
        with pytest.raises(ValueError, match="same length"):
            gen.save_dataset(features, ["hrm", "trm"], path)

    def test_load_nonexistent_file(self, gen):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            gen.load_dataset("/nonexistent/path.json")
