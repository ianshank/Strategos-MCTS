"""
Dataset Loading Module for Open-Source Training Data.

Provides unified loading interfaces for:
- DABStep: Multi-step data analysis reasoning
- PRIMUS: Cybersecurity domain knowledge
- Custom tactical datasets

License Attribution:
- DABStep: CC-BY-4.0 (Creative Commons Attribution)
- PRIMUS: ODC-BY (Open Data Commons Attribution)
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Standardized representation of a dataset sample."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    labels: list[str] | None = None
    difficulty: str | None = None
    domain: str | None = None
    reasoning_steps: list[str] | None = None


@dataclass
class DatasetStatistics:
    """Statistics about a loaded dataset."""

    total_samples: int
    domains: dict[str, int]
    avg_text_length: float
    difficulty_distribution: dict[str, int]
    total_tokens: int = 0


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize dataset loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "mcts_datasets")
        self._dataset = None
        self._statistics = None

    @abstractmethod
    def load(self, split: str = "train") -> list[DatasetSample]:
        """Load dataset split."""
        pass

    @abstractmethod
    def get_statistics(self) -> DatasetStatistics:
        """Get dataset statistics."""
        pass

    @abstractmethod
    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        pass


class DABStepLoader(DatasetLoader):
    """
    Loader for DABStep Multi-Step Reasoning Dataset.

    DABStep contains 450+ data analysis tasks requiring sequential,
    iterative problem-solving. Perfect for training HRM/TRM agents.

    License: CC-BY-4.0 (Attribution required)
    Source: huggingface.co/datasets/adyen/DABstep
    """

    DATASET_NAME = "adyen/DABstep"
    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, cache_dir: str | None = None):
        """Initialize DABStep loader."""
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []

    def load(self, split: str = "train", difficulty: str | None = None) -> list[DatasetSample]:
        """
        Load DABStep dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            difficulty: Filter by difficulty ('easy', 'medium', 'hard')

        Returns:
            List of DatasetSample objects
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading DABStep dataset (split={split})")

            dataset = load_dataset(
                self.DATASET_NAME,
                cache_dir=self.cache_dir,
            )

            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available: {available_splits}")
                split = available_splits[0] if available_splits else "train"

            samples = []
            for idx, item in enumerate(dataset[split]):
                sample = DatasetSample(
                    id=f"dabstep_{split}_{idx}",
                    text=str(item.get("question", item.get("text", ""))),
                    metadata={
                        "source": "DABStep",
                        "license": "CC-BY-4.0",
                        "split": split,
                        "original_data": item,
                    },
                    difficulty=item.get("difficulty", "medium"),
                    domain="data_analysis",
                    reasoning_steps=item.get("steps", []),
                )

                if difficulty and sample.difficulty != difficulty:
                    continue

                samples.append(sample)

            self._loaded_samples = samples
            logger.info(f"Loaded {len(samples)} DABStep samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load DABStep: {e}")
            raise

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded DABStep data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        difficulty_dist = {}
        total_length = 0

        for sample in self._loaded_samples:
            diff = sample.difficulty or "unknown"
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
            total_length += len(sample.text)

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"data_analysis": len(self._loaded_samples)},
            avg_text_length=total_length / len(self._loaded_samples),
            difficulty_distribution=difficulty_dist,
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]

    def get_reasoning_tasks(self) -> list[DatasetSample]:
        """Get only samples with explicit reasoning steps."""
        return [s for s in self._loaded_samples if s.reasoning_steps]


class PRIMUSLoader(DatasetLoader):
    """
    Loader for PRIMUS Cybersecurity Dataset Suite.

    PRIMUS contains:
    - Seed: 674,848 cybersecurity documents (190M tokens)
    - Instruct: 835 instruction-tuning samples
    - Reasoning: Self-reflection data for reasoning

    License: ODC-BY (Open Data Commons Attribution)
    Source: huggingface.co/datasets/trendmicro-ailab/Primus-Seed
    """

    SEED_DATASET = "trendmicro-ailab/Primus-Seed"
    INSTRUCT_DATASET = "trendmicro-ailab/Primus-Instruct"

    DOMAINS = [
        "mitre_attack",
        "wikipedia",
        "company_sites",
        "threat_intelligence",
        "vulnerability_db",
    ]

    def __init__(self, cache_dir: str | None = None):
        """Initialize PRIMUS loader."""
        super().__init__(cache_dir)
        self._seed_samples: list[DatasetSample] = []
        self._instruct_samples: list[DatasetSample] = []

    def load(
        self,
        split: str = "train",
        dataset_type: str = "seed",
        domains: list[str] | None = None,
        max_samples: int | None = None,
        streaming: bool = True,
    ) -> list[DatasetSample]:
        """
        Load PRIMUS dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            dataset_type: 'seed' for knowledge base, 'instruct' for fine-tuning
            domains: Filter by specific domains
            max_samples: Limit number of samples (useful for large datasets)
            streaming: Use streaming mode for large datasets (default True)

        Returns:
            List of DatasetSample objects
        """
        try:
            from datasets import load_dataset

            dataset_name = self.SEED_DATASET if dataset_type == "seed" else self.INSTRUCT_DATASET

            logger.info(f"Loading PRIMUS {dataset_type} dataset")

            # Use streaming for large seed dataset to avoid download issues
            use_streaming = streaming and dataset_type == "seed" and max_samples is not None

            if use_streaming:
                logger.info(f"Using streaming mode (max_samples={max_samples})")
                dataset = load_dataset(
                    dataset_name,
                    "default",
                    streaming=True,
                    cache_dir=self.cache_dir,
                )
                # For streaming, iterate the first available split
                data_iter = iter(dataset["train"]) if "train" in dataset else iter(dataset[list(dataset.keys())[0]])
            else:
                dataset = load_dataset(
                    dataset_name,
                    cache_dir=self.cache_dir,
                )

                if split not in dataset:
                    available_splits = list(dataset.keys())
                    logger.warning(f"Split '{split}' not found. Using: {available_splits[0]}")
                    split = available_splits[0]

                data_iter = iter(dataset[split])

            samples = []
            count = 0

            for idx, item in enumerate(data_iter):
                if max_samples and count >= max_samples:
                    break

                domain = item.get("domain", item.get("source", "unknown"))

                if domains and domain not in domains:
                    continue

                if dataset_type == "instruct":
                    text = f"Instruction: {item.get('instruction', '')}\nResponse: {item.get('response', '')}"
                else:
                    text = str(item.get("text", item.get("content", "")))

                sample = DatasetSample(
                    id=f"primus_{dataset_type}_{split}_{idx}",
                    text=text,
                    metadata={
                        "source": f"PRIMUS-{dataset_type.capitalize()}",
                        "license": "ODC-BY",
                        "split": split,
                        "original_domain": domain,
                    },
                    domain=domain,
                    labels=item.get("labels", item.get("tags", [])),
                )

                samples.append(sample)
                count += 1

            if dataset_type == "seed":
                self._seed_samples = samples
            else:
                self._instruct_samples = samples

            logger.info(f"Loaded {len(samples)} PRIMUS {dataset_type} samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            if "gated dataset" in str(e):
                logger.error(
                    f"PRIMUS is a gated dataset. Please authenticate with HuggingFace:\n"
                    f"1. Create account at https://huggingface.co/\n"
                    f"2. Accept dataset terms at https://huggingface.co/datasets/{dataset_name}\n"
                    f"3. Create token at https://huggingface.co/settings/tokens\n"
                    f"4. Run: huggingface-cli login"
                )
            else:
                logger.error(f"Failed to load PRIMUS: {e}")
            raise

    def load_seed(self, max_samples: int | None = None) -> list[DatasetSample]:
        """Load PRIMUS-Seed knowledge base."""
        return self.load(dataset_type="seed", max_samples=max_samples)

    def load_instruct(self) -> list[DatasetSample]:
        """Load PRIMUS-Instruct fine-tuning data."""
        return self.load(dataset_type="instruct", streaming=False)

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded PRIMUS data."""
        all_samples = self._seed_samples + self._instruct_samples

        if not all_samples:
            raise ValueError("No samples loaded. Call load() first.")

        domain_dist = {}
        total_length = 0

        for sample in all_samples:
            domain = sample.domain or "unknown"
            domain_dist[domain] = domain_dist.get(domain, 0) + 1
            total_length += len(sample.text)

        return DatasetStatistics(
            total_samples=len(all_samples),
            domains=domain_dist,
            avg_text_length=total_length / len(all_samples),
            difficulty_distribution={"cybersecurity": len(all_samples)},
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over all loaded samples in batches."""
        all_samples = self._seed_samples + self._instruct_samples

        if not all_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(all_samples), batch_size):
            yield all_samples[i : i + batch_size]

    def get_mitre_attack_samples(self) -> list[DatasetSample]:
        """Get samples specifically from MITRE ATT&CK."""
        return [s for s in self._seed_samples if "mitre" in (s.domain or "").lower()]

    def get_threat_intelligence_samples(self) -> list[DatasetSample]:
        """Get threat intelligence related samples."""
        return [
            s
            for s in self._seed_samples
            if any(kw in (s.domain or "").lower() for kw in ["threat", "cti", "intelligence"])
        ]


class CombinedDatasetLoader:
    """
    Unified loader for combining multiple datasets.

    Provides a single interface for loading and managing:
    - DABStep (multi-step reasoning)
    - PRIMUS (cybersecurity knowledge)
    - Custom tactical datasets
    """

    def __init__(self, cache_dir: str | None = None):
        """Initialize combined loader."""
        self.cache_dir = cache_dir
        self.dabstep_loader = DABStepLoader(cache_dir)
        self.primus_loader = PRIMUSLoader(cache_dir)
        self._all_samples: list[DatasetSample] = []

    def load_all(
        self,
        dabstep_split: str = "train",
        primus_max_samples: int | None = 10000,
        include_instruct: bool = True,
    ) -> list[DatasetSample]:
        """
        Load all datasets.

        Args:
            dabstep_split: Split for DABStep
            primus_max_samples: Max samples from PRIMUS-Seed (None for all)
            include_instruct: Whether to include PRIMUS-Instruct

        Returns:
            Combined list of all samples
        """
        logger.info("Loading combined datasets")

        # Load DABStep
        dabstep_samples = self.dabstep_loader.load(split=dabstep_split)
        logger.info(f"DABStep: {len(dabstep_samples)} samples")

        # Load PRIMUS-Seed
        primus_seed = self.primus_loader.load_seed(max_samples=primus_max_samples)
        logger.info(f"PRIMUS-Seed: {len(primus_seed)} samples")

        # Load PRIMUS-Instruct
        primus_instruct = []
        if include_instruct:
            primus_instruct = self.primus_loader.load_instruct()
            logger.info(f"PRIMUS-Instruct: {len(primus_instruct)} samples")

        self._all_samples = dabstep_samples + primus_seed + primus_instruct
        logger.info(f"Total combined samples: {len(self._all_samples)}")

        return self._all_samples

    def get_domain_distribution(self) -> dict[str, int]:
        """Get distribution of samples across domains."""
        dist = {}
        for sample in self._all_samples:
            domain = sample.domain or "unknown"
            dist[domain] = dist.get(domain, 0) + 1
        return dist

    def filter_by_domain(self, domain: str) -> list[DatasetSample]:
        """Filter samples by domain."""
        return [s for s in self._all_samples if s.domain == domain]

    def get_multi_step_reasoning_samples(self) -> list[DatasetSample]:
        """Get samples suitable for multi-step reasoning training."""
        return [
            s
            for s in self._all_samples
            if s.reasoning_steps or s.domain == "data_analysis" or "instruct" in s.metadata.get("source", "").lower()
        ]

    def export_for_training(self, output_path: str, format: str = "jsonl") -> str:
        """
        Export dataset for training.

        Args:
            output_path: Path to save exported data
            format: Export format ('jsonl', 'csv', 'parquet')

        Returns:
            Path to exported file
        """
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in self._all_samples:
                    record = {
                        "id": sample.id,
                        "text": sample.text,
                        "domain": sample.domain,
                        "difficulty": sample.difficulty,
                        "labels": sample.labels,
                        "metadata": sample.metadata,
                    }
                    f.write(json.dumps(record) + "\n")

        elif format == "csv":
            self._export_csv(output_file)

        elif format == "parquet":
            self._export_parquet(output_file)

        else:
            supported = ["jsonl", "csv", "parquet"]
            raise NotImplementedError(f"Format {format} not supported. Supported formats: {supported}")

        logger.info(f"Exported {len(self._all_samples)} samples to {output_file}")
        return str(output_file)

    def _export_csv(self, output_file: Path) -> None:
        """
        Export dataset to CSV format.

        Args:
            output_file: Path to output file
        """
        import csv
        import json

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "text", "domain", "difficulty", "labels", "metadata"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for sample in self._all_samples:
                writer.writerow(
                    {
                        "id": sample.id,
                        "text": sample.text,
                        "domain": sample.domain or "",
                        "difficulty": sample.difficulty or "",
                        "labels": json.dumps(sample.labels) if sample.labels else "",
                        "metadata": json.dumps(sample.metadata) if sample.metadata else "",
                    }
                )

        logger.debug(f"Exported {len(self._all_samples)} samples to CSV: {output_file}")

    def _export_parquet(self, output_file: Path) -> None:
        """
        Export dataset to Parquet format.

        Requires pyarrow to be installed.

        Args:
            output_file: Path to output file

        Raises:
            ImportError: If pyarrow is not installed
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("pyarrow is required for Parquet export. " "Install with: pip install pyarrow") from e

        import json

        # Prepare data as columns
        data = {
            "id": [s.id for s in self._all_samples],
            "text": [s.text for s in self._all_samples],
            "domain": [s.domain or "" for s in self._all_samples],
            "difficulty": [s.difficulty or "" for s in self._all_samples],
            "labels": [json.dumps(s.labels) if s.labels else "" for s in self._all_samples],
            "metadata": [json.dumps(s.metadata) if s.metadata else "" for s in self._all_samples],
        }

        # Create PyArrow table
        table = pa.Table.from_pydict(data)

        # Write to Parquet with compression
        pq.write_table(table, output_file, compression="snappy")

        logger.debug(f"Exported {len(self._all_samples)} samples to Parquet: {output_file}")

    @staticmethod
    def load_from_csv(file_path: str | Path) -> list[DatasetSample]:
        """
        Load dataset samples from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of DatasetSample objects
        """
        import csv
        import json

        samples = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels = json.loads(row.get("labels", "[]")) if row.get("labels") else []
                metadata = json.loads(row.get("metadata", "{}")) if row.get("metadata") else {}

                sample = DatasetSample(
                    id=row.get("id", ""),
                    text=row.get("text", ""),
                    domain=row.get("domain") or None,
                    difficulty=row.get("difficulty") or None,
                    labels=labels,
                    metadata=metadata,
                )
                samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from CSV: {file_path}")
        return samples

    @staticmethod
    def load_from_parquet(file_path: str | Path) -> list[DatasetSample]:
        """
        Load dataset samples from Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            List of DatasetSample objects

        Raises:
            ImportError: If pyarrow is not installed
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("pyarrow is required for Parquet import. " "Install with: pip install pyarrow") from e

        import json

        table = pq.read_table(file_path)
        df_dict = table.to_pydict()

        samples = []
        for i in range(len(df_dict["id"])):
            labels_str = df_dict.get("labels", [""])[i]
            metadata_str = df_dict.get("metadata", [""])[i]

            labels = json.loads(labels_str) if labels_str else []
            metadata = json.loads(metadata_str) if metadata_str else {}

            sample = DatasetSample(
                id=df_dict["id"][i],
                text=df_dict["text"][i],
                domain=df_dict.get("domain", [None])[i] or None,
                difficulty=df_dict.get("difficulty", [None])[i] or None,
                labels=labels,
                metadata=metadata,
            )
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from Parquet: {file_path}")
        return samples


def load_dataset(
    dataset_name: str,
    split: str = "train",
    cache_dir: str | None = None,
    **kwargs,
) -> Any:
    """
    Unified interface for loading datasets from HuggingFace.

    This function provides compatibility with the standard HuggingFace datasets API.
    It wraps the underlying load_dataset function from the datasets library.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "adyen/DABstep")
        split: Dataset split to load ("train", "validation", "test")
        cache_dir: Optional directory for caching downloaded datasets
        **kwargs: Additional arguments passed to datasets.load_dataset

    Returns:
        HuggingFace Dataset object or dict of Dataset objects

    Raises:
        ImportError: If datasets library is not installed
        Exception: If dataset loading fails

    Examples:
        >>> # Load DABStep dataset
        >>> dataset = load_dataset("adyen/DABstep")
        >>> samples = dataset["train"]

        >>> # Load PRIMUS-Seed with custom cache
        >>> dataset = load_dataset("trendmicro-ailab/Primus-Seed", cache_dir="/tmp/cache")

    License Attribution:
        - DABStep: CC-BY-4.0 (Creative Commons Attribution 4.0)
        - PRIMUS: ODC-BY (Open Data Commons Attribution)
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info(f"Loading dataset: {dataset_name} (split={split})")

        load_kwargs = {
            **kwargs,
        }

        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        # Always pass split parameter - defaults to "train" if not specified
        # This ensures consistent behavior across all dataset types
        load_kwargs["split"] = split

        dataset = hf_load_dataset(dataset_name, **load_kwargs)

        logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset

    except ImportError as e:
        logger.error("datasets library not installed. Run: pip install datasets")
        raise ImportError(
            "The datasets library is required but not installed. Install it with: pip install datasets"
        ) from e
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
