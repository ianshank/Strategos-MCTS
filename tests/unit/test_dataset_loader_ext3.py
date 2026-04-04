"""Extended unit tests for src/data/dataset_loader.py - targeting uncovered lines.

Covers: DABStepLoader.load(), PRIMUSLoader.load() (seed/instruct, streaming,
gated dataset errors), CombinedDatasetLoader.load_all(), _export_csv,
_export_parquet, load_from_csv, load_from_parquet, and the module-level
load_dataset() function.
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
    PRIMUSLoader,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_dataset(items: list[dict[str, Any]], splits: list[str] | None = None):
    """Create a mock HuggingFace dataset object."""
    splits = splits or ["train"]
    mock_ds = MagicMock()
    mock_ds.keys.return_value = splits
    mock_ds.__contains__ = lambda self, key: key in splits
    for _split in splits:
        mock_ds.__getitem__ = lambda self, key, _items=items: _items if key in splits else []
    # Make it iterable per split
    mock_ds.__getitem__ = lambda self, key: items if key in splits else []
    return mock_ds


class _DictDataset(dict):
    """A real dict subclass that mimics a HuggingFace DatasetDict.

    Using a real dict avoids issues with MagicMock and `in` operator.
    """

    def __init__(self, splits: dict[str, list[dict]]):
        super().__init__(splits)


def _make_dict_dataset(items: list[dict], splits: dict[str, list[dict]] | None = None):
    """Create a dict-like mock HuggingFace dataset with proper split access."""
    if splits is None:
        splits = {"train": items}
    return _DictDataset(splits)


# ---------------------------------------------------------------------------
# DABStepLoader.load()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDABStepLoaderLoad:
    """Tests for DABStepLoader.load() covering lines 108-153."""

    @patch("src.data.dataset_loader.load_dataset")
    def test_load_success(self, mock_load_fn):
        """Mocking the module-level load_dataset used by DABStepLoader internally."""
        # We need to mock datasets.load_dataset, not the module-level wrapper
        items = [
            {"question": "What is 2+2?", "difficulty": "easy", "steps": ["add"]},
            {"question": "Analyze data", "difficulty": "hard", "steps": []},
        ]
        mock_ds = _make_dict_dataset(items, {"train": items})

        with patch("src.data.dataset_loader.DABStepLoader.load"):
            # Instead, let's test by actually calling with mocked datasets lib
            pass

        # Actually test by patching the datasets import inside the method
        loader = DABStepLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            mock_datasets_mod = sys.modules["datasets"]
            mock_datasets_mod.load_dataset.return_value = mock_ds

            samples = loader.load(split="train")
            assert len(samples) == 2
            assert samples[0].id == "dabstep_train_0"
            assert samples[0].text == "What is 2+2?"
            assert samples[0].difficulty == "easy"
            assert samples[0].domain == "data_analysis"
            assert samples[0].metadata["license"] == "CC-BY-4.0"

    def test_load_with_difficulty_filter(self):
        items = [
            {"question": "Easy q", "difficulty": "easy", "steps": []},
            {"question": "Hard q", "difficulty": "hard", "steps": []},
            {"question": "Medium q", "difficulty": "medium", "steps": []},
        ]
        mock_ds = _make_dict_dataset(items, {"train": items})

        loader = DABStepLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds
            samples = loader.load(split="train", difficulty="hard")
            assert len(samples) == 1
            assert samples[0].difficulty == "hard"

    def test_load_split_not_found_uses_first(self):
        """Line 119-121: split not found, uses first available."""
        items = [{"text": "hello", "difficulty": "easy"}]
        mock_ds = _make_dict_dataset(items, {"validation": items})

        loader = DABStepLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds
            samples = loader.load(split="train")  # train not available
            assert len(samples) == 1

    def test_load_uses_text_field_fallback(self):
        """Line 127: uses 'text' when 'question' not present."""
        items = [{"text": "fallback text"}]
        mock_ds = _make_dict_dataset(items, {"train": items})

        loader = DABStepLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds
            samples = loader.load(split="train")
            assert samples[0].text == "fallback text"

    def test_load_import_error(self):
        """Lines 148-150: ImportError raised when datasets not available."""
        loader = DABStepLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError):
                loader.load()

    def test_load_generic_exception(self):
        """Lines 151-153: generic exception re-raised."""
        loader = DABStepLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.side_effect = RuntimeError("network error")
            with pytest.raises(RuntimeError, match="network error"):
                loader.load()


# ---------------------------------------------------------------------------
# PRIMUSLoader.load()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPRIMUSLoaderLoad:
    """Tests for PRIMUSLoader.load() covering lines 239-327."""

    def test_load_seed_non_streaming(self):
        """Non-streaming seed load."""
        items = [
            {"text": "CVE-2024-1234", "domain": "vulnerability_db"},
            {"content": "MITRE info", "source": "mitre_attack"},
        ]
        mock_ds = _make_dict_dataset(items, {"train": items})

        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds

            samples = loader.load(split="train", dataset_type="seed", streaming=False)
            assert len(samples) == 2
            assert samples[0].text == "CVE-2024-1234"
            assert samples[0].domain == "vulnerability_db"
            assert samples[0].metadata["license"] == "ODC-BY"

    def test_load_seed_streaming(self):
        """Streaming seed load with max_samples (lines 249-258)."""
        items = [{"text": f"doc {i}", "domain": "wikipedia"} for i in range(10)]

        mock_ds = MagicMock()
        mock_ds.__contains__ = lambda self, key: key == "train"
        mock_ds.__getitem__ = lambda self, key: iter(items)
        mock_ds.keys.return_value = ["train"]

        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds

            samples = loader.load(dataset_type="seed", streaming=True, max_samples=3)
            assert len(samples) == 3

    def test_load_instruct(self):
        """Instruct dataset type (line 284-285)."""
        items = [
            {"instruction": "Explain XSS", "response": "XSS is..."},
        ]
        mock_ds = _make_dict_dataset(items, {"train": items})

        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds

            samples = loader.load(dataset_type="instruct", streaming=False)
            assert len(samples) == 1
            assert "Instruction: Explain XSS" in samples[0].text
            assert "Response: XSS is..." in samples[0].text
            assert samples[0] in loader._instruct_samples

    def test_load_domain_filter(self):
        """Lines 281-282: domain filter."""
        items = [
            {"text": "mitre doc", "domain": "mitre_attack"},
            {"text": "wiki doc", "domain": "wikipedia"},
        ]
        mock_ds = _make_dict_dataset(items, {"train": items})

        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds

            samples = loader.load(dataset_type="seed", domains=["mitre_attack"], streaming=False)
            assert len(samples) == 1
            assert samples[0].domain == "mitre_attack"

    def test_load_max_samples_limit(self):
        """Lines 276-277: max_samples limits results."""
        items = [{"text": f"doc {i}", "domain": "wikipedia"} for i in range(10)]
        mock_ds = _make_dict_dataset(items, {"train": items})

        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds

            samples = loader.load(dataset_type="seed", max_samples=3, streaming=False)
            assert len(samples) == 3

    def test_load_split_not_found(self):
        """Lines 265-268: split not found falls back."""
        items = [{"text": "hello"}]
        mock_ds = _make_dict_dataset(items, {"validation": items})

        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds

            samples = loader.load(split="train", dataset_type="seed", streaming=False)
            assert len(samples) == 1

    def test_load_import_error(self):
        """Lines 313-315: ImportError raised."""
        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError):
                loader.load()

    def test_load_gated_dataset_error(self):
        """Lines 317-324: gated dataset error message."""
        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.side_effect = RuntimeError(
                "This is a gated dataset and requires authentication"
            )
            with pytest.raises(RuntimeError, match="gated dataset"):
                loader.load()

    def test_load_generic_exception(self):
        """Lines 325-327: generic exception re-raised."""
        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.side_effect = ValueError("bad data")
            with pytest.raises(ValueError, match="bad data"):
                loader.load()

    def test_load_seed_convenience(self):
        """Line 331: load_seed delegates to load."""
        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.object(loader, "load", return_value=[]) as mock_load:
            loader.load_seed(max_samples=50)
            mock_load.assert_called_once_with(dataset_type="seed", max_samples=50)

    def test_load_instruct_convenience(self):
        """Line 335: load_instruct delegates to load."""
        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.object(loader, "load", return_value=[]) as mock_load:
            loader.load_instruct()
            mock_load.assert_called_once_with(dataset_type="instruct", streaming=False)

    def test_load_with_labels_and_tags(self):
        """Line 299: labels from 'labels' or 'tags' field."""
        items = [
            {"text": "doc1", "domain": "test", "labels": ["l1", "l2"]},
            {"text": "doc2", "domain": "test", "tags": ["t1"]},
        ]
        mock_ds = _make_dict_dataset(items, {"train": items})

        loader = PRIMUSLoader(cache_dir="/tmp/test")
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.return_value = mock_ds

            samples = loader.load(dataset_type="seed", streaming=False)
            assert samples[0].labels == ["l1", "l2"]
            assert samples[1].labels == ["t1"]


# ---------------------------------------------------------------------------
# CombinedDatasetLoader.load_all()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinedDatasetLoaderLoadAll:
    """Tests for CombinedDatasetLoader.load_all() covering lines 416-435."""

    def test_load_all_with_instruct(self):
        dabstep_samples = [DatasetSample(id="d1", text="dab")]
        primus_seed = [DatasetSample(id="p1", text="seed")]
        primus_instruct = [DatasetSample(id="i1", text="instruct")]

        loader = CombinedDatasetLoader(cache_dir="/tmp/test")
        with (
            patch.object(loader.dabstep_loader, "load", return_value=dabstep_samples),
            patch.object(loader.primus_loader, "load_seed", return_value=primus_seed),
            patch.object(loader.primus_loader, "load_instruct", return_value=primus_instruct),
        ):

            result = loader.load_all(include_instruct=True)
            assert len(result) == 3
            assert loader._all_samples == result

    def test_load_all_without_instruct(self):
        dabstep_samples = [DatasetSample(id="d1", text="dab")]
        primus_seed = [DatasetSample(id="p1", text="seed")]

        loader = CombinedDatasetLoader(cache_dir="/tmp/test")
        with (
            patch.object(loader.dabstep_loader, "load", return_value=dabstep_samples),
            patch.object(loader.primus_loader, "load_seed", return_value=primus_seed),
        ):

            result = loader.load_all(include_instruct=False)
            assert len(result) == 2

    def test_load_all_custom_params(self):
        loader = CombinedDatasetLoader(cache_dir="/tmp/test")
        with (
            patch.object(loader.dabstep_loader, "load", return_value=[]) as mock_dab,
            patch.object(loader.primus_loader, "load_seed", return_value=[]) as mock_seed,
        ):

            loader.load_all(
                dabstep_split="validation",
                primus_max_samples=5000,
                include_instruct=False,
            )
            mock_dab.assert_called_once_with(split="validation")
            mock_seed.assert_called_once_with(max_samples=5000)


# ---------------------------------------------------------------------------
# CombinedDatasetLoader._export_parquet / _export_csv
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinedExportFormats:
    """Tests for _export_csv and _export_parquet (lines 542-564)."""

    def test_export_csv_full(self):
        """Lines 499-526: CSV export with all field types."""
        loader = CombinedDatasetLoader()
        loader._all_samples = [
            DatasetSample(
                id="s1", text="Hello", domain="math", difficulty="easy", labels=["a", "b"], metadata={"key": "val"}
            ),
            DatasetSample(id="s2", text="World", domain=None, difficulty=None),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            loader._export_csv(path)

            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["id"] == "s1"
            assert rows[0]["domain"] == "math"
            assert json.loads(rows[0]["labels"]) == ["a", "b"]
            assert rows[1]["domain"] == ""
            assert rows[1]["difficulty"] == ""
            assert rows[1]["labels"] == ""

    def test_export_parquet_success(self):
        """Lines 546-564: Parquet export with mock pyarrow."""
        loader = CombinedDatasetLoader()
        loader._all_samples = [
            DatasetSample(id="s1", text="Hello", domain="math", difficulty="easy", labels=["a"], metadata={"k": "v"}),
        ]

        import types

        mock_pa = types.ModuleType("pyarrow")
        mock_pq = types.ModuleType("pyarrow.parquet")

        mock_table = MagicMock()
        mock_pa.Table = MagicMock()
        mock_pa.Table.from_pydict = MagicMock(return_value=mock_table)
        mock_pa.parquet = mock_pq
        mock_pq.write_table = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            import sys

            old_pa = sys.modules.get("pyarrow")
            old_pq = sys.modules.get("pyarrow.parquet")
            sys.modules["pyarrow"] = mock_pa
            sys.modules["pyarrow.parquet"] = mock_pq
            try:
                loader._export_parquet(path)
            finally:
                if old_pa is not None:
                    sys.modules["pyarrow"] = old_pa
                else:
                    sys.modules.pop("pyarrow", None)
                if old_pq is not None:
                    sys.modules["pyarrow.parquet"] = old_pq
                else:
                    sys.modules.pop("pyarrow.parquet", None)

            mock_pa.Table.from_pydict.assert_called_once()
            mock_pq.write_table.assert_called_once_with(mock_table, path, compression="snappy")

    def test_export_parquet_missing_pyarrow(self):
        """Line 543-544: ImportError when pyarrow missing."""
        loader = CombinedDatasetLoader()
        loader._all_samples = [DatasetSample(id="s1", text="t")]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    loader._export_parquet(path)

    def test_export_for_training_parquet_path(self):
        """export_for_training dispatches to _export_parquet."""
        loader = CombinedDatasetLoader()
        loader._all_samples = [DatasetSample(id="s1", text="t")]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "out.parquet")
            with patch.object(loader, "_export_parquet") as mock_ep:
                result = loader.export_for_training(path, format="parquet")
                mock_ep.assert_called_once()
                assert result == path


# ---------------------------------------------------------------------------
# CombinedDatasetLoader.load_from_csv
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadFromCSV:
    """Tests for CombinedDatasetLoader.load_from_csv() covering lines 619-643."""

    def test_load_from_csv_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "domain", "difficulty", "labels", "metadata"])
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "s1",
                        "text": "Sample text",
                        "domain": "math",
                        "difficulty": "easy",
                        "labels": json.dumps(["tag1", "tag2"]),
                        "metadata": json.dumps({"source": "test"}),
                    }
                )
                writer.writerow(
                    {
                        "id": "s2",
                        "text": "No extras",
                        "domain": "",
                        "difficulty": "",
                        "labels": "",
                        "metadata": "",
                    }
                )

            samples = CombinedDatasetLoader.load_from_csv(str(path))
            assert len(samples) == 2
            assert samples[0].labels == ["tag1", "tag2"]
            assert samples[0].metadata == {"source": "test"}
            assert samples[0].domain == "math"
            assert samples[1].domain is None
            assert samples[1].labels == []
            assert samples[1].metadata == {}

    def test_load_from_csv_path_object(self):
        """Accepts Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "domain", "difficulty", "labels", "metadata"])
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "x1",
                        "text": "hi",
                        "domain": "d",
                        "difficulty": "e",
                        "labels": "[]",
                        "metadata": "{}",
                    }
                )

            samples = CombinedDatasetLoader.load_from_csv(path)
            assert len(samples) == 1


# ---------------------------------------------------------------------------
# CombinedDatasetLoader.load_from_parquet
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadFromParquet:
    """Tests for CombinedDatasetLoader.load_from_parquet() covering lines 619-643."""

    def _setup_pyarrow_mocks(self, mock_pq_module):
        """Set up pyarrow module mocks and return cleanup info."""
        import sys
        import types

        mock_pa = types.ModuleType("pyarrow")
        mock_pa.parquet = mock_pq_module  # type: ignore
        old_pa = sys.modules.get("pyarrow")
        old_pq = sys.modules.get("pyarrow.parquet")
        sys.modules["pyarrow"] = mock_pa
        sys.modules["pyarrow.parquet"] = mock_pq_module
        return old_pa, old_pq

    def _cleanup_pyarrow_mocks(self, old_pa, old_pq):
        import sys

        if old_pa is not None:
            sys.modules["pyarrow"] = old_pa
        else:
            sys.modules.pop("pyarrow", None)
        if old_pq is not None:
            sys.modules["pyarrow.parquet"] = old_pq
        else:
            sys.modules.pop("pyarrow.parquet", None)

    def test_load_from_parquet_success(self):
        import types

        mock_pq = types.ModuleType("pyarrow.parquet")

        mock_table = MagicMock()
        mock_table.to_pydict.return_value = {
            "id": ["s1", "s2"],
            "text": ["Hello", "World"],
            "domain": ["math", ""],
            "difficulty": ["easy", ""],
            "labels": [json.dumps(["a"]), ""],
            "metadata": [json.dumps({"k": "v"}), ""],
        }
        mock_pq.read_table = MagicMock(return_value=mock_table)  # type: ignore

        old_pa, old_pq = self._setup_pyarrow_mocks(mock_pq)
        try:
            samples = CombinedDatasetLoader.load_from_parquet("/fake/path.parquet")
        finally:
            self._cleanup_pyarrow_mocks(old_pa, old_pq)

        assert len(samples) == 2
        assert samples[0].id == "s1"
        assert samples[0].labels == ["a"]
        assert samples[0].metadata == {"k": "v"}
        assert samples[0].domain == "math"
        assert samples[1].domain is None
        assert samples[1].labels == []
        assert samples[1].metadata == {}

    def test_load_from_parquet_missing_pyarrow(self):
        import sys

        old_pa = sys.modules.get("pyarrow")
        old_pq = sys.modules.get("pyarrow.parquet")
        sys.modules["pyarrow"] = None  # type: ignore
        sys.modules["pyarrow.parquet"] = None  # type: ignore
        try:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                CombinedDatasetLoader.load_from_parquet("/fake/path.parquet")
        finally:
            if old_pa is not None:
                sys.modules["pyarrow"] = old_pa
            else:
                sys.modules.pop("pyarrow", None)
            if old_pq is not None:
                sys.modules["pyarrow.parquet"] = old_pq
            else:
                sys.modules.pop("pyarrow.parquet", None)

    def test_load_from_parquet_missing_optional_columns(self):
        """Handles missing domain/difficulty/labels/metadata columns."""
        import types

        mock_pq = types.ModuleType("pyarrow.parquet")

        mock_table = MagicMock()
        mock_table.to_pydict.return_value = {
            "id": ["s1"],
            "text": ["Hello"],
        }
        mock_pq.read_table = MagicMock(return_value=mock_table)  # type: ignore

        old_pa, old_pq = self._setup_pyarrow_mocks(mock_pq)
        try:
            samples = CombinedDatasetLoader.load_from_parquet("/fake/path.parquet")
        finally:
            self._cleanup_pyarrow_mocks(old_pa, old_pq)

        assert len(samples) == 1
        assert samples[0].domain is None
        assert samples[0].difficulty is None
        assert samples[0].labels == []
        assert samples[0].metadata == {}


# ---------------------------------------------------------------------------
# Module-level load_dataset()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModuleLevelLoadDataset:
    """Tests for the module-level load_dataset() function covering lines 683-711."""

    def test_load_dataset_success(self):
        """Lines 683-702: successful load."""
        mock_hf_load = MagicMock(return_value="mock_dataset")

        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset = mock_hf_load

            result = load_dataset("adyen/DABstep", split="train")
            assert result == "mock_dataset"
            mock_hf_load.assert_called_once()
            call_kwargs = mock_hf_load.call_args
            assert call_kwargs[0][0] == "adyen/DABstep"
            assert call_kwargs[1]["split"] == "train"

    def test_load_dataset_with_cache_dir(self):
        """Line 692-693: cache_dir passed through."""
        mock_hf_load = MagicMock(return_value="mock_dataset")

        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset = mock_hf_load

            load_dataset("test/dataset", split="train", cache_dir="/tmp/cache")
            call_kwargs = mock_hf_load.call_args[1]
            assert call_kwargs["cache_dir"] == "/tmp/cache"

    def test_load_dataset_with_extra_kwargs(self):
        """Extra kwargs passed through."""
        mock_hf_load = MagicMock(return_value="mock_dataset")

        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset = mock_hf_load

            load_dataset("test/dataset", split="test", streaming=True)
            call_kwargs = mock_hf_load.call_args[1]
            assert call_kwargs["streaming"] is True
            assert call_kwargs["split"] == "test"

    def test_load_dataset_import_error(self):
        """Lines 704-708: ImportError when datasets not installed."""
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets library is required"):
                load_dataset("test/dataset")

    def test_load_dataset_generic_exception(self):
        """Lines 709-711: generic exception re-raised."""
        with patch.dict("sys.modules", {"datasets": MagicMock()}):
            import sys

            sys.modules["datasets"].load_dataset.side_effect = RuntimeError("network error")

            with pytest.raises(RuntimeError, match="network error"):
                load_dataset("test/dataset")


# ---------------------------------------------------------------------------
# PRIMUSLoader statistics/iteration edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPRIMUSLoaderStatisticsEdge:
    """Extra edge cases for PRIMUS statistics and iteration."""

    def test_get_statistics_combined_seed_and_instruct(self):
        loader = PRIMUSLoader()
        loader._seed_samples = [
            DatasetSample(id="s1", text="seed text", domain="mitre_attack"),
        ]
        loader._instruct_samples = [
            DatasetSample(id="i1", text="instruct text", domain="instruction"),
        ]
        stats = loader.get_statistics()
        assert stats.total_samples == 2
        assert "mitre_attack" in stats.domains
        assert "instruction" in stats.domains

    def test_get_statistics_unknown_domain(self):
        loader = PRIMUSLoader()
        loader._seed_samples = [
            DatasetSample(id="s1", text="text", domain=None),
        ]
        stats = loader.get_statistics()
        assert "unknown" in stats.domains

    def test_iterate_samples_combines_both(self):
        loader = PRIMUSLoader()
        loader._seed_samples = [DatasetSample(id="s1", text="t1")]
        loader._instruct_samples = [DatasetSample(id="i1", text="t2")]
        batches = list(loader.iterate_samples(batch_size=10))
        assert len(batches) == 1
        assert len(batches[0]) == 2
