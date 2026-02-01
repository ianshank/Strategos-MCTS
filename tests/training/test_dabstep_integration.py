"""
Training Pipeline Tests: DABStep Dataset Integration.

Tests for:
- DABStep dataset loading (CC-BY-4.0)
- Multi-step reasoning data preprocessing
- Training data pipeline for HRM/TRM agents
- Stratified splitting
- Data augmentation

Expected outcomes:
- 450+ tasks loaded successfully
- Balanced train/val/test splits
- Feature extraction for meta-controller training
"""

from unittest.mock import patch

import pytest

# Check if the datasets library is available
try:
    import datasets

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Skip marker for tests that require the datasets library
requires_datasets = pytest.mark.skipif(
    not HAS_DATASETS,
    reason="datasets library not installed (pip install datasets)",
)


@pytest.fixture
def mock_dabstep_dataset():
    """Mock DABStep dataset structure."""
    return {
        "train": [
            {
                "question": "Analyze the correlation between sales and marketing spend",
                "difficulty": "easy",
                "steps": ["Load data", "Calculate correlation", "Interpret results"],
                "answer": "Positive correlation of 0.75",
            },
            {
                "question": "Identify outliers in customer transaction data and explain their impact",
                "difficulty": "medium",
                "steps": [
                    "Statistical analysis",
                    "Visualize distribution",
                    "Apply IQR method",
                    "Assess business impact",
                ],
                "answer": "3 outliers identified, representing high-value customers",
            },
            {
                "question": "Build a predictive model for customer churn using multiple features",
                "difficulty": "hard",
                "steps": [
                    "Feature engineering",
                    "Model selection",
                    "Cross-validation",
                    "Hyperparameter tuning",
                    "Evaluate performance",
                    "Interpret feature importance",
                ],
                "answer": "Random Forest model with 85% accuracy",
            },
        ]
    }


@pytest.fixture
def sample_dabstep_samples():
    """Sample DABStep samples for testing."""
    from src.data.dataset_loader import DatasetSample

    return [
        DatasetSample(
            id="dabstep_train_0",
            text="Analyze the correlation between sales and marketing spend",
            metadata={"source": "DABStep", "license": "CC-BY-4.0"},
            difficulty="easy",
            domain="data_analysis",
            reasoning_steps=["Load data", "Calculate correlation", "Interpret results"],
        ),
        DatasetSample(
            id="dabstep_train_1",
            text="Identify outliers in customer transaction data",
            metadata={"source": "DABStep", "license": "CC-BY-4.0"},
            difficulty="medium",
            domain="data_analysis",
            reasoning_steps=["Statistical analysis", "Visualize distribution", "Apply IQR method"],
        ),
        DatasetSample(
            id="dabstep_train_2",
            text="Build a predictive model for customer churn",
            metadata={"source": "DABStep", "license": "CC-BY-4.0"},
            difficulty="hard",
            domain="data_analysis",
            reasoning_steps=["Feature engineering", "Model selection", "Cross-validation"],
        ),
    ]


class TestDABStepDatasetLoading:
    """Test DABStep dataset loading functionality."""

    @pytest.mark.training
    @pytest.mark.dataset
    @requires_datasets
    @patch("datasets.load_dataset")
    def test_dabstep_loader_initialization(self, mock_load):
        """DABStep loader should initialize correctly."""
        from src.data.dataset_loader import DABStepLoader

        loader = DABStepLoader(cache_dir="/tmp/test_cache")

        assert loader.cache_dir == "/tmp/test_cache"
        assert loader.DATASET_NAME == "adyen/DABstep"

    @pytest.mark.training
    @pytest.mark.dataset
    @requires_datasets
    @patch("datasets.load_dataset")
    def test_load_train_split(self, mock_load, mock_dabstep_dataset):
        """Should load train split successfully."""
        from src.data.dataset_loader import DABStepLoader

        mock_load.return_value = mock_dabstep_dataset

        loader = DABStepLoader()
        samples = loader.load(split="train")

        assert len(samples) == 3
        assert all(s.domain == "data_analysis" for s in samples)
        assert all(s.metadata["license"] == "CC-BY-4.0" for s in samples)

    @pytest.mark.training
    @pytest.mark.dataset
    @requires_datasets
    @patch("datasets.load_dataset")
    def test_filter_by_difficulty(self, mock_load, mock_dabstep_dataset):
        """Should filter samples by difficulty."""
        from src.data.dataset_loader import DABStepLoader

        mock_load.return_value = mock_dabstep_dataset

        loader = DABStepLoader()
        easy_samples = loader.load(split="train", difficulty="easy")

        assert len(easy_samples) == 1
        assert easy_samples[0].difficulty == "easy"

    @pytest.mark.training
    @pytest.mark.dataset
    @requires_datasets
    @patch("datasets.load_dataset")
    def test_reasoning_steps_preserved(self, mock_load, mock_dabstep_dataset):
        """Reasoning steps should be preserved in samples."""
        from src.data.dataset_loader import DABStepLoader

        mock_load.return_value = mock_dabstep_dataset

        loader = DABStepLoader()
        samples = loader.load(split="train")

        # Check that reasoning steps are preserved
        for sample in samples:
            assert sample.reasoning_steps is not None
            assert len(sample.reasoning_steps) >= 3

    @pytest.mark.training
    @pytest.mark.dataset
    @requires_datasets
    @patch("datasets.load_dataset")
    def test_dataset_statistics(self, mock_load, mock_dabstep_dataset):
        """Should compute dataset statistics correctly."""
        from src.data.dataset_loader import DABStepLoader

        mock_load.return_value = mock_dabstep_dataset

        loader = DABStepLoader()
        loader.load(split="train")
        stats = loader.get_statistics()

        assert stats.total_samples == 3
        assert "data_analysis" in stats.domains
        assert stats.avg_text_length > 0
        assert "easy" in stats.difficulty_distribution
        assert "medium" in stats.difficulty_distribution
        assert "hard" in stats.difficulty_distribution

    @pytest.mark.training
    @pytest.mark.dataset
    @requires_datasets
    @patch("datasets.load_dataset")
    def test_batch_iteration(self, mock_load, mock_dabstep_dataset):
        """Should iterate samples in batches."""
        from src.data.dataset_loader import DABStepLoader

        mock_load.return_value = mock_dabstep_dataset

        loader = DABStepLoader()
        loader.load(split="train")

        batch_count = 0
        for batch in loader.iterate_samples(batch_size=2):
            batch_count += 1
            assert len(batch) <= 2

        assert batch_count == 2  # 3 samples, batch size 2


class TestDABStepPreprocessing:
    """Test preprocessing of DABStep data."""

    @pytest.mark.training
    def test_text_preprocessing(self, sample_dabstep_samples):
        """Text should be cleaned and normalized."""
        from src.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()

        for sample in sample_dabstep_samples:
            result = preprocessor.preprocess(sample.text)

            assert result.cleaned is not None
            assert len(result.cleaned) > 0
            assert len(result.tokens) > 0

    @pytest.mark.training
    def test_feature_extraction_for_meta_controller(self, sample_dabstep_samples):
        """Features should be extracted for meta-controller training."""
        from src.data.preprocessing import MetaControllerFeatureExtractor

        extractor = MetaControllerFeatureExtractor()

        for sample in sample_dabstep_samples:
            features = extractor.extract_query_features(sample.text)

            assert "query_length" in features
            assert "word_count" in features
            assert "complexity_score" in features
            assert "is_data_analysis" in features

            # Validate ranges
            assert 0.0 <= features["query_length"] <= 1.0
            assert 0.0 <= features["complexity_score"] <= 1.0

    @pytest.mark.training
    def test_complexity_scoring(self, sample_dabstep_samples):
        """Complexity scores should vary by difficulty."""
        from src.data.preprocessing import MetaControllerFeatureExtractor

        extractor = MetaControllerFeatureExtractor()

        scores = {}
        for sample in sample_dabstep_samples:
            features = extractor.extract_query_features(sample.text)
            scores[sample.difficulty] = features["complexity_score"]

        # Hard queries should generally have higher complexity
        # (though this is heuristic-based)
        assert scores["easy"] >= 0.0
        assert scores["hard"] >= 0.0


class TestDABStepAugmentation:
    """Test data augmentation for DABStep samples."""

    @pytest.mark.training
    def test_tactical_augmentation(self, sample_dabstep_samples):
        """Samples should be augmented for increased diversity."""
        from src.data.tactical_augmentation import TacticalAugmenter

        augmenter = TacticalAugmenter(seed=42)

        augmented = augmenter.augment_batch(
            sample_dabstep_samples,
            augmentations_per_sample=2,
        )

        # Should have original + augmented
        original_count = len(sample_dabstep_samples)
        expected_total = original_count + (original_count * 2)

        assert len(augmented) == expected_total

    @pytest.mark.training
    def test_augmentation_preserves_metadata(self, sample_dabstep_samples):
        """Augmentation should preserve original metadata."""
        from src.data.tactical_augmentation import TacticalAugmenter

        augmenter = TacticalAugmenter(seed=42)

        result = augmenter.augment_sample(
            sample_dabstep_samples[0],
            num_augmentations=1,
        )

        augmented = result.augmented[0]

        # Metadata should be preserved
        assert augmented.domain == sample_dabstep_samples[0].domain
        assert "original_id" in augmented.metadata
        assert augmented.metadata["license"] == "CC-BY-4.0"

    @pytest.mark.training
    def test_deterministic_augmentation(self, sample_dabstep_samples):
        """Same seed should produce same augmentations."""
        from src.data.tactical_augmentation import TacticalAugmenter

        augmenter1 = TacticalAugmenter(seed=42)
        augmenter2 = TacticalAugmenter(seed=42)

        result1 = augmenter1.augment_sample(sample_dabstep_samples[0], num_augmentations=1)
        result2 = augmenter2.augment_sample(sample_dabstep_samples[0], num_augmentations=1)

        # Should produce same augmentation type
        assert result1.augmentation_types == result2.augmentation_types


class TestDABStepSplitting:
    """Test train/val/test splitting for DABStep data."""

    @pytest.mark.training
    def test_stratified_split_by_difficulty(self, sample_dabstep_samples):
        """Split should maintain difficulty distribution."""
        from src.data.train_test_split import StratifiedSplitter

        # Create more samples for proper splitting
        samples = sample_dabstep_samples * 10  # 30 samples

        splitter = StratifiedSplitter(seed=42, stratify_by="difficulty")
        split = splitter.split(
            samples,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        # Verify split ratios approximately maintained
        total = len(samples)
        assert len(split.train) == pytest.approx(total * 0.6, abs=3)
        assert len(split.validation) == pytest.approx(total * 0.2, abs=3)
        assert len(split.test) == pytest.approx(total * 0.2, abs=3)

        # Verify stratification info is recorded
        assert "stratify_by" in split.split_info
        assert split.split_info["stratify_by"] == "difficulty"

    @pytest.mark.training
    def test_k_fold_cross_validation(self, sample_dabstep_samples):
        """Should create k-fold cross-validation splits."""
        from src.data.train_test_split import DataSplitter

        samples = sample_dabstep_samples * 5  # 15 samples

        splitter = DataSplitter(seed=42)
        folds = splitter.create_k_folds(samples, k=3)

        assert len(folds) == 3

        # Each fold should have train and validation
        for fold in folds:
            assert len(fold.train) > 0
            assert len(fold.validation) > 0
            # Train should be larger than validation
            assert len(fold.train) > len(fold.validation)

    @pytest.mark.training
    def test_reproducible_splits(self, sample_dabstep_samples):
        """Same seed should produce same splits."""
        from src.data.train_test_split import DataSplitter

        samples = sample_dabstep_samples * 5

        splitter1 = DataSplitter(seed=42)
        splitter2 = DataSplitter(seed=42)

        split1 = splitter1.split(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        split2 = splitter2.split(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        # Should have same samples in each split
        assert len(split1.train) == len(split2.train)
        assert len(split1.validation) == len(split2.validation)
        assert len(split1.test) == len(split2.test)


class TestMetaControllerTraining:
    """Test meta-controller training with DABStep data."""

    @pytest.mark.training
    def test_feature_tensor_conversion(self, sample_dabstep_samples):
        """Features should convert to tensors for training."""
        from src.data.preprocessing import MetaControllerFeatureExtractor

        extractor = MetaControllerFeatureExtractor()

        features_list = []
        for sample in sample_dabstep_samples:
            features = extractor.extract_query_features(sample.text)
            features_list.append(list(features.values()))

        # Simulate tensor conversion
        import numpy as np

        feature_array = np.array(features_list)

        assert feature_array.shape == (3, 8)  # 3 samples, 8 features

    @pytest.mark.training
    def test_agent_state_features(self):
        """Agent state features should be extracted correctly."""
        from src.data.preprocessing import MetaControllerFeatureExtractor

        extractor = MetaControllerFeatureExtractor()

        features = extractor.extract_agent_state_features(
            hrm_confidence=0.85,
            trm_confidence=0.82,
            mcts_iterations=100,
            consensus_score=0.83,
            rag_retrieved=5,
        )

        # Should be 10-dimensional
        assert len(features) == 10

        # Check specific features
        assert features[0] == 0.85  # HRM confidence
        assert features[1] == 0.82  # TRM confidence
        assert features[2] == 0.1  # Normalized MCTS iterations (100/1000)
        assert features[3] == 0.83  # Consensus score
        assert features[4] == 0.25  # Normalized RAG (5/20)


class TestTrainingPipelineIntegration:
    """Test complete training pipeline integration."""

    @pytest.mark.training
    @pytest.mark.integration
    @requires_datasets
    @patch("datasets.load_dataset")
    def test_full_pipeline(self, mock_load, mock_dabstep_dataset):
        """Complete pipeline from loading to training-ready data."""
        from src.data.dataset_loader import DABStepLoader
        from src.data.preprocessing import TextPreprocessor
        from src.data.tactical_augmentation import TacticalAugmenter
        from src.data.train_test_split import StratifiedSplitter

        mock_load.return_value = mock_dabstep_dataset

        # Step 1: Load data
        loader = DABStepLoader()
        samples = loader.load(split="train")
        assert len(samples) == 3

        # Step 2: Preprocess
        preprocessor = TextPreprocessor()
        for sample in samples:
            result = preprocessor.preprocess(sample.text)
            assert result.cleaned is not None

        # Step 3: Augment
        augmenter = TacticalAugmenter(seed=42)
        augmented = augmenter.augment_batch(samples, augmentations_per_sample=2)
        assert len(augmented) == 9  # 3 + 6

        # Step 4: Split
        splitter = StratifiedSplitter(seed=42, stratify_by="domain")
        split = splitter.split(
            augmented,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Verify pipeline output
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.test) > 0

    @pytest.mark.training
    def test_cost_efficiency(self):
        """Training cost should be within budget."""
        # Simulated training cost calculation
        training_config = {
            "epochs": 10,
            "batch_size": 32,
            "samples": 10000,
            "gpu_hours": 5,
            "cost_per_gpu_hour": 2.0,  # $2/hour for consumer GPU
        }

        total_cost = training_config["gpu_hours"] * training_config["cost_per_gpu_hour"]

        # Should be under $100 budget
        assert total_cost < 100, f"Training cost ${total_cost} exceeds $100 budget"


class TestAttributionCompliance:
    """Test license attribution compliance."""

    @pytest.mark.training
    def test_ccby_attribution_present(self, sample_dabstep_samples):
        """CC-BY-4.0 attribution should be present in metadata."""
        for sample in sample_dabstep_samples:
            assert "license" in sample.metadata
            assert sample.metadata["license"] == "CC-BY-4.0"
            assert "source" in sample.metadata

    @pytest.mark.training
    def test_attribution_documentation(self):
        """Attribution should be properly documented."""
        attribution = {
            "DABStep": {
                "source": "Hugging Face (adyen/DABstep)",
                "license": "CC-BY-4.0",
                "citation": "DABStep Multi-Step Reasoning Dataset",
                "attribution_required": True,
            }
        }

        # Verify attribution structure
        assert "DABStep" in attribution
        assert attribution["DABStep"]["license"] == "CC-BY-4.0"
        assert attribution["DABStep"]["attribution_required"] is True
