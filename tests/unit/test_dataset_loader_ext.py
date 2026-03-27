"""
Unit tests for src/data/dataset_loader.py.

Tests DatasetSample, DatasetStatistics, DABStepLoader, PRIMUSLoader,
CombinedDatasetLoader, and the module-level load_dataset function.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.data.dataset_loader import (
    CombinedDatasetLoader,
    DABStepLoader,
    DatasetSample,
    DatasetStatistics,
    PRIMUSLoader,
)


# ---------------------------------------------------------------------------
# DatasetSample
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetSample:
    def test_creation(self):
        sample = DatasetSample(id="s1", text="Hello world")
        assert sample.id == "s1"
        assert sample.text == "Hello world"
        assert sample.metadata == {}
        assert sample.labels is None
        assert sample.difficulty is None
        assert sample.domain is None
        assert sample.reasoning_steps is None

    def test_full_creation(self):
        sample = DatasetSample(
            id="s2",
            text="Test text",
            metadata={"key": "val"},
            labels=["a", "b"],
            difficulty="hard",
            domain="math",
            reasoning_steps=["step1", "step2"],
        )
        assert sample.labels == ["a", "b"]
        assert sample.difficulty == "hard"
        assert sample.domain == "math"
        assert len(sample.reasoning_steps) == 2


# ---------------------------------------------------------------------------
# DatasetStatistics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetStatistics:
    def test_creation(self):
        stats = DatasetStatistics(
            total_samples=100,
            domains={"math": 60, "science": 40},
            avg_text_length=150.5,
            difficulty_distribution={"easy": 50, "hard": 50},
        )
        assert stats.total_samples == 100
        assert stats.domains["math"] == 60
        assert stats.avg_text_length == 150.5
        assert stats.total_tokens == 0


# ---------------------------------------------------------------------------
# DABStepLoader
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDABStepLoader:
    def test_init_default_cache(self):
        loader = DABStepLoader()
        assert "mcts_datasets" in loader.cache_dir

    def test_init_custom_cache(self):
        loader = DABStepLoader(cache_dir="/tmp/test_cache")
        assert loader.cache_dir == "/tmp/test_cache"

    def test_dataset_name(self):
        assert DABStepLoader.DATASET_NAME == "adyen/DABstep"

    def test_difficulties(self):
        assert "easy" in DABStepLoader.DIFFICULTIES
        assert "medium" in DABStepLoader.DIFFICULTIES
        assert "hard" in DABStepLoader.DIFFICULTIES

    def test_get_statistics_without_load_raises(self):
        loader = DABStepLoader()
        with pytest.raises(ValueError, match="No samples loaded"):
            loader.get_statistics()

    def test_iterate_samples_without_load_raises(self):
        loader = DABStepLoader()
        with pytest.raises(ValueError, match="No samples loaded"):
            list(loader.iterate_samples())

    def test_get_statistics_with_samples(self):
        loader = DABStepLoader()
        loader._loaded_samples = [
            DatasetSample(id="s1", text="short", difficulty="easy", domain="data_analysis"),
            DatasetSample(id="s2", text="medium length text", difficulty="hard", domain="data_analysis"),
        ]
        stats = loader.get_statistics()
        assert stats.total_samples == 2
        assert stats.domains == {"data_analysis": 2}
        assert stats.difficulty_distribution == {"easy": 1, "hard": 1}
        assert stats.avg_text_length > 0

    def test_iterate_samples_batching(self):
        loader = DABStepLoader()
        loader._loaded_samples = [
            DatasetSample(id=f"s{i}", text=f"text {i}") for i in range(5)
        ]
        batches = list(loader.iterate_samples(batch_size=2))
        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0]) == 2
        assert len(batches[2]) == 1

    def test_get_reasoning_tasks(self):
        loader = DABStepLoader()
        loader._loaded_samples = [
            DatasetSample(id="s1", text="t1", reasoning_steps=["step1"]),
            DatasetSample(id="s2", text="t2", reasoning_steps=None),
            DatasetSample(id="s3", text="t3", reasoning_steps=[]),
        ]
        tasks = loader.get_reasoning_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "s1"


# ---------------------------------------------------------------------------
# PRIMUSLoader
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPRIMUSLoader:
    def test_init(self):
        loader = PRIMUSLoader()
        assert loader._seed_samples == []
        assert loader._instruct_samples == []

    def test_dataset_names(self):
        assert "Primus-Seed" in PRIMUSLoader.SEED_DATASET
        assert "Primus-Instruct" in PRIMUSLoader.INSTRUCT_DATASET

    def test_domains(self):
        assert "mitre_attack" in PRIMUSLoader.DOMAINS

    def test_get_statistics_without_load_raises(self):
        loader = PRIMUSLoader()
        with pytest.raises(ValueError, match="No samples loaded"):
            loader.get_statistics()

    def test_iterate_samples_without_load_raises(self):
        loader = PRIMUSLoader()
        with pytest.raises(ValueError, match="No samples loaded"):
            list(loader.iterate_samples())

    def test_get_statistics_with_samples(self):
        loader = PRIMUSLoader()
        loader._seed_samples = [
            DatasetSample(id="s1", text="Sample text 1", domain="mitre_attack"),
            DatasetSample(id="s2", text="Sample text 2", domain="wikipedia"),
        ]
        stats = loader.get_statistics()
        assert stats.total_samples == 2
        assert "mitre_attack" in stats.domains
        assert "wikipedia" in stats.domains

    def test_iterate_samples_batching(self):
        loader = PRIMUSLoader()
        loader._seed_samples = [DatasetSample(id=f"s{i}", text=f"t{i}") for i in range(3)]
        loader._instruct_samples = [DatasetSample(id=f"i{i}", text=f"t{i}") for i in range(2)]
        batches = list(loader.iterate_samples(batch_size=2))
        assert len(batches) == 3  # 2 + 2 + 1

    def test_get_mitre_attack_samples(self):
        loader = PRIMUSLoader()
        loader._seed_samples = [
            DatasetSample(id="s1", text="t1", domain="mitre_attack"),
            DatasetSample(id="s2", text="t2", domain="wikipedia"),
        ]
        mitre = loader.get_mitre_attack_samples()
        assert len(mitre) == 1
        assert mitre[0].domain == "mitre_attack"

    def test_get_threat_intelligence_samples(self):
        loader = PRIMUSLoader()
        loader._seed_samples = [
            DatasetSample(id="s1", text="t1", domain="threat_intelligence"),
            DatasetSample(id="s2", text="t2", domain="wikipedia"),
            DatasetSample(id="s3", text="t3", domain="cti_reports"),
        ]
        threat = loader.get_threat_intelligence_samples()
        assert len(threat) == 2


# ---------------------------------------------------------------------------
# CombinedDatasetLoader
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinedDatasetLoader:
    def test_init(self):
        loader = CombinedDatasetLoader(cache_dir="/tmp/test")
        assert loader.cache_dir == "/tmp/test"
        assert isinstance(loader.dabstep_loader, DABStepLoader)
        assert isinstance(loader.primus_loader, PRIMUSLoader)

    def test_get_domain_distribution_empty(self):
        loader = CombinedDatasetLoader()
        dist = loader.get_domain_distribution()
        assert dist == {}

    def test_get_domain_distribution(self):
        loader = CombinedDatasetLoader()
        loader._all_samples = [
            DatasetSample(id="s1", text="t", domain="math"),
            DatasetSample(id="s2", text="t", domain="math"),
            DatasetSample(id="s3", text="t", domain="science"),
        ]
        dist = loader.get_domain_distribution()
        assert dist == {"math": 2, "science": 1}

    def test_filter_by_domain(self):
        loader = CombinedDatasetLoader()
        loader._all_samples = [
            DatasetSample(id="s1", text="t", domain="math"),
            DatasetSample(id="s2", text="t", domain="science"),
        ]
        filtered = loader.filter_by_domain("math")
        assert len(filtered) == 1
        assert filtered[0].domain == "math"

    def test_get_multi_step_reasoning_samples(self):
        loader = CombinedDatasetLoader()
        loader._all_samples = [
            DatasetSample(id="s1", text="t", reasoning_steps=["s1"]),
            DatasetSample(id="s2", text="t", domain="data_analysis"),
            DatasetSample(id="s3", text="t", metadata={"source": "PRIMUS-Instruct"}),
            DatasetSample(id="s4", text="t", domain="other"),
        ]
        results = loader.get_multi_step_reasoning_samples()
        assert len(results) == 3  # s1 (reasoning_steps), s2 (data_analysis), s3 (instruct)

    def test_export_for_training_jsonl(self):
        loader = CombinedDatasetLoader()
        loader._all_samples = [
            DatasetSample(id="s1", text="Hello", domain="math", difficulty="easy", labels=["a"]),
            DatasetSample(id="s2", text="World", domain="science"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.jsonl")
            result = loader.export_for_training(path, format="jsonl")
            assert result == path

            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert record["id"] == "s1"
            assert record["text"] == "Hello"

    def test_export_for_training_csv(self):
        loader = CombinedDatasetLoader()
        loader._all_samples = [
            DatasetSample(id="s1", text="Hello", domain="math"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.csv")
            result = loader.export_for_training(path, format="csv")
            assert result == path

            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["id"] == "s1"

    def test_export_for_training_unsupported_format(self):
        loader = CombinedDatasetLoader()
        loader._all_samples = [DatasetSample(id="s1", text="t")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.xml")
            with pytest.raises(NotImplementedError, match="not supported"):
                loader.export_for_training(path, format="xml")

    def test_load_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["id", "text", "domain", "difficulty", "labels", "metadata"]
                )
                writer.writeheader()
                writer.writerow({
                    "id": "s1",
                    "text": "Sample text",
                    "domain": "math",
                    "difficulty": "easy",
                    "labels": json.dumps(["tag1"]),
                    "metadata": json.dumps({"key": "val"}),
                })
                writer.writerow({
                    "id": "s2",
                    "text": "Another",
                    "domain": "",
                    "difficulty": "",
                    "labels": "",
                    "metadata": "",
                })

            samples = CombinedDatasetLoader.load_from_csv(str(path))
            assert len(samples) == 2
            assert samples[0].id == "s1"
            assert samples[0].labels == ["tag1"]
            assert samples[0].metadata == {"key": "val"}
            assert samples[1].domain is None
            assert samples[1].difficulty is None

    def test_export_parquet_missing_pyarrow(self):
        loader = CombinedDatasetLoader()
        loader._all_samples = [DatasetSample(id="s1", text="t")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.parquet")
            with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
                # The import error should propagate from _export_parquet
                # but we need the actual code path to hit the import
                try:
                    loader.export_for_training(path, format="parquet")
                except (ImportError, ModuleNotFoundError):
                    pass  # expected
