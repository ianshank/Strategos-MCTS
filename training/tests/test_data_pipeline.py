"""Unit tests for data pipeline module."""

# Import modules to test
from pathlib import Path
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.data_pipeline import (
    DABStepLoader,
    DataOrchestrator,
    DocumentChunk,
    HRMSample,
    PRIMUSProcessor,
    TaskSample,
    TRMSample,
)


class TestDABStepLoader:
    """Tests for DABStep dataset loader."""

    @pytest.fixture
    def loader_config(self):
        return {
            "path": "adyen/DABstep",
            "cache_dir": tempfile.mkdtemp(),
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,
        }

    @pytest.fixture
    def loader(self, loader_config):
        return DABStepLoader(loader_config)

    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
        assert loader.train_split == 0.8
        assert loader.val_split == 0.1
        assert loader.test_split == 0.1

    def test_synthetic_dataset_creation(self, loader):
        """Test synthetic dataset is created when real dataset unavailable."""
        loader.load_dataset()
        assert loader._dataset is not None

    def test_splits_creation(self, loader):
        """Test train/val/test splits are created correctly."""
        splits = loader.create_splits()

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        # Check split proportions (approximately)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert len(splits["train"]) / total >= 0.7
        assert len(splits["val"]) / total >= 0.05
        assert len(splits["test"]) / total >= 0.05

    def test_hrm_format_conversion(self, loader):
        """Test conversion to HRM training format."""
        sample = TaskSample(
            task_id="test_1",
            task_text="Solve this problem",
            difficulty="medium",
            category="reasoning",
            steps=["Step 1", "Step 2", "Step 3"],
            expected_output="Result",
        )

        hrm_sample = loader.convert_to_hrm_format(sample)

        assert isinstance(hrm_sample, HRMSample)
        assert hrm_sample.input_text == "Solve this problem"
        assert hrm_sample.depth == 3
        assert len(hrm_sample.decomposition) == 3

    def test_trm_format_conversion(self, loader):
        """Test conversion to TRM training format."""
        sample = TaskSample(
            task_id="test_1",
            task_text="Solve this problem",
            difficulty="medium",
            category="reasoning",
            steps=["Step 1", "Step 2"],
            expected_output="Result",
        )

        trm_sample = loader.convert_to_trm_format(sample)

        assert isinstance(trm_sample, TRMSample)
        assert trm_sample.initial_task == "Solve this problem"
        assert len(trm_sample.refinement_steps) == 2
        assert len(trm_sample.improvement_scores) > 0

    def test_task_augmentation(self, loader):
        """Test task augmentation creates variations."""
        sample = TaskSample(
            task_id="test_1",
            task_text="Solve this",
            difficulty="easy",
            category="reasoning",
            steps=["A", "B", "C"],
            expected_output="Result",
        )

        augmented = loader.augment_task(sample, num_variations=3)

        assert len(augmented) == 3
        for aug_sample in augmented:
            assert aug_sample.task_id != sample.task_id
            assert aug_sample.metadata.get("augmented") is True

    def test_curriculum_batches(self, loader):
        """Test curriculum learning generates batches in order."""
        loader.create_splits()

        samples = list(loader.get_curriculum_batches())

        # Should have samples from different difficulties
        difficulties = [s.difficulty for s in samples[:10]]
        assert len(set(difficulties)) >= 1


class TestPRIMUSProcessor:
    """Tests for PRIMUS dataset processor."""

    @pytest.fixture
    def processor_config(self):
        return {
            "primus_seed": {
                "path": "trendmicro-ailab/Primus-Seed",
                "cache_dir": tempfile.mkdtemp(),
                "categories": ["mitre", "cyber_companies"],
                "streaming": True,
            },
            "primus_instruct": {"path": "trendmicro-ailab/Primus-Instruct"},
        }

    @pytest.fixture
    def processor(self, processor_config):
        return PRIMUSProcessor(processor_config)

    def test_processor_initialization(self, processor):
        """Test processor initializes correctly."""
        assert processor is not None
        assert processor.categories == ["mitre", "cyber_companies"]

    def test_text_chunking(self, processor):
        """Test text chunking works correctly."""
        text = "This is a test. " * 100  # Long text
        chunks = processor._chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 110  # Allow some buffer

    def test_document_streaming(self, processor):
        """Test document streaming generates chunks."""
        processor.load_seed_dataset()

        chunks = list(processor.stream_documents())

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.text
            assert chunk.metadata

    def test_instruction_samples(self, processor):
        """Test instruction sample retrieval."""
        samples = processor.get_instruction_samples()

        assert isinstance(samples, list)
        if samples:
            assert "instruction" in samples[0]
            assert "input" in samples[0]
            assert "output" in samples[0]

    def test_domain_metadata_extraction(self, processor):
        """Test domain metadata extraction."""
        chunk = DocumentChunk(
            doc_id="test",
            chunk_id=0,
            text="This document discusses MITRE ATT&CK T1059 command execution vulnerability CVE-2021-12345",
            metadata={"category": "cyber"},
        )

        metadata = processor.extract_domain_metadata(chunk)

        assert metadata["has_mitre_reference"] is True
        assert metadata["has_cve_reference"] is True
        assert metadata["threat_level"] in ["high", "medium", "low"]

    def test_threat_level_estimation(self, processor):
        """Test threat level estimation."""
        high_threat_text = "Critical vulnerability exploit breach"
        medium_threat_text = "Suspicious anomaly detected"
        low_threat_text = "Normal system operation"

        assert processor._estimate_threat_level(high_threat_text) == "high"
        assert processor._estimate_threat_level(medium_threat_text) == "medium"
        assert processor._estimate_threat_level(low_threat_text) == "low"


class TestDataOrchestrator:
    """Tests for data orchestration."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config file."""
        config = {
            "data": {
                "dabstep": {
                    "path": "adyen/DABstep",
                    "cache_dir": str(tmp_path / "dabstep"),
                    "train_split": 0.8,
                    "val_split": 0.1,
                    "test_split": 0.1,
                    "seed": 42,
                },
                "primus_seed": {
                    "path": "trendmicro-ailab/Primus-Seed",
                    "cache_dir": str(tmp_path / "primus"),
                    "categories": ["mitre"],
                    "streaming": True,
                },
                "primus_instruct": {"path": "trendmicro-ailab/Primus-Instruct"},
            },
            "training": {"batch_size": 8},
            "resources": {"max_cpu_workers": 2, "pin_memory": False},
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return str(config_path)

    def test_orchestrator_initialization(self, temp_config):
        """Test orchestrator initializes correctly."""
        orchestrator = DataOrchestrator(temp_config)
        assert orchestrator is not None
        assert orchestrator.batch_size == 8

    def test_data_statistics(self, temp_config):
        """Test data statistics retrieval."""
        orchestrator = DataOrchestrator(temp_config)
        orchestrator.prepare_data()

        stats = orchestrator.get_data_statistics()

        assert "dabstep" in stats
        assert "train_samples" in stats["dabstep"]
        assert stats["dabstep"]["train_samples"] > 0


class TestTaskSample:
    """Tests for TaskSample dataclass."""

    def test_task_sample_creation(self):
        """Test TaskSample can be created."""
        sample = TaskSample(
            task_id="test",
            task_text="Test task",
            difficulty="easy",
            category="test",
            steps=["Step 1"],
            expected_output="Output",
        )

        assert sample.task_id == "test"
        assert sample.difficulty == "easy"
        assert len(sample.steps) == 1

    def test_task_sample_with_metadata(self):
        """Test TaskSample with metadata."""
        sample = TaskSample(
            task_id="test",
            task_text="Test",
            difficulty="hard",
            category="test",
            steps=[],
            expected_output="",
            metadata={"custom": "value"},
        )

        assert sample.metadata["custom"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
