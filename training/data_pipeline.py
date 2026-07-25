"""
Data Pipeline Module for Multi-Agent MCTS Training

Handles dataset loading, preprocessing, and data orchestration for:
- DABStep multi-step reasoning tasks
- PRIMUS-Seed cybersecurity documents
- PRIMUS-Instruct instruction samples
"""

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset  # noqa: F401

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logger = logging.getLogger(__name__)


@dataclass
class TaskSample:
    """Represents a single training task sample."""

    task_id: str
    task_text: str
    difficulty: str  # easy, medium, hard
    category: str
    steps: list[str]
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HRMSample:
    """Sample formatted for HRM (hierarchical decomposition) training."""

    input_text: str
    decomposition: list[str]
    depth: int
    labels: list[int]  # Token-level decomposition labels


@dataclass
class TRMSample:
    """Sample formatted for TRM (task refinement) training."""

    initial_task: str
    refinement_steps: list[str]
    final_task: str
    improvement_scores: list[float]


@dataclass
class DocumentChunk:
    """A chunk of document for RAG processing."""

    doc_id: str
    chunk_id: int
    text: str
    metadata: dict[str, Any]
    embedding: np.ndarray | None = None


class DABStepLoader:
    """Load and preprocess DABStep multi-step reasoning tasks."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize DABStep loader.

        Args:
            config: Configuration dictionary with DABStep settings
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")

        self.config = config
        self.dataset_path = config.get("path", "adyen/DABstep")
        self.cache_dir = Path(config.get("cache_dir", "./cache/dabstep"))
        self.max_samples = config.get("max_samples")
        self.streaming = config.get("streaming", False)
        self.train_split = config.get("train_split", 0.8)
        self.val_split = config.get("val_split", 0.1)
        self.test_split = config.get("test_split", 0.1)
        self.seed = config.get("seed", 42)
        self.synthetic_data_dir = Path(config.get("synthetic_data_dir", "training/synthetic_data"))

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = None
        self._splits = {}

        logger.info(f"Initialized DABStepLoader with path: {self.dataset_path}")

    def load_dataset(self) -> None:
        """Load DABStep dataset from HuggingFace."""
        logger.info(f"Loading DABStep dataset from {self.dataset_path}")

        try:
            self._dataset = load_dataset(
                self.dataset_path,
                cache_dir=str(self.cache_dir),
                streaming=self.streaming,
            )

            if self.max_samples and not self.streaming and "train" in self._dataset:
                # Limit samples if specified
                self._dataset["train"] = self._dataset["train"].select(
                    range(min(self.max_samples, len(self._dataset["train"])))
                )

            logger.info("Loaded DABStep dataset successfully")

        except Exception as e:
            logger.error(f"Failed to load DABStep dataset: {e}")
            # Create synthetic dataset for testing/development
            self._create_synthetic_dataset()

    def load_synthetic_data(self) -> list[TaskSample]:
        """
        Load synthetic training data from JSON files.

        Returns:
            List of TaskSample objects converted from synthetic data
        """
        synthetic_samples = []

        if not self.synthetic_data_dir.exists():
            logger.info(f"Synthetic data directory not found: {self.synthetic_data_dir}")
            return synthetic_samples

        json_files = list(self.synthetic_data_dir.glob("*.json"))
        if not json_files:
            logger.info(f"No synthetic data files found in {self.synthetic_data_dir}")
            return synthetic_samples

        for json_file in json_files:
            try:
                # Skip non-dataset files (like stats or checkpoints)
                if json_file.name.endswith("_stats.json") or json_file.name.endswith("_checkpoint.json"):
                    continue

                with open(json_file) as f:
                    data = json.load(f)

                # Handle list of records
                if isinstance(data, list):
                    for item in data:
                        sample = self._convert_synthetic_to_task_sample(item)
                        if sample:
                            synthetic_samples.append(sample)
                elif isinstance(data, dict):
                    sample = self._convert_synthetic_to_task_sample(data)
                    if sample:
                        synthetic_samples.append(sample)
                else:
                    logger.warning(f"Unexpected data type in {json_file}: {type(data).__name__}. Skipping file.")

            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load synthetic data file {json_file}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error loading {json_file}: {e}")

        if synthetic_samples:
            logger.info(f"Loaded {len(synthetic_samples)} samples from {len(json_files)} synthetic data files")

        return synthetic_samples

    def _convert_synthetic_to_task_sample(self, item: dict[str, Any]) -> TaskSample | None:
        """Convert synthetic Q&A item to TaskSample."""
        try:
            # Handle both LangSmith format and raw format
            if "inputs" in item and "outputs" in item:  # LangSmith format
                question = item["inputs"].get("question", "")
                answer = item["outputs"].get("ground_truth", "")
                metadata = item.get("metadata", {})
            else:  # Raw format
                question = item.get("question", "")
                answer = item.get("answer", "")
                metadata = item.get("metadata", {})

            # Extract reasoning steps if available, otherwise just answer
            steps = item.get("reasoning_paths", [])
            if not steps:
                steps = [answer]

            # Generate deterministic ID
            task_id = hashlib.md5(question.encode()).hexdigest()[:16]

            return TaskSample(
                task_id=task_id,
                task_text=question,
                difficulty=metadata.get("difficulty", "medium"),
                category=metadata.get("category", "general"),
                steps=steps,
                expected_output=answer,
                metadata={**metadata, "is_synthetic": True},
            )
        except (KeyError, TypeError) as e:
            logger.debug(f"Failed to convert synthetic item: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error converting synthetic item: {e}")
            return None

    def _create_synthetic_dataset(self) -> None:
        """Create synthetic DABStep-like dataset for development."""
        logger.warning("Creating synthetic DABStep dataset for development")

        synthetic_tasks = []
        difficulties = ["easy", "medium", "hard"]

        for i in range(100):
            difficulty = difficulties[i % 3]
            num_steps = {"easy": 2, "medium": 4, "hard": 6}[difficulty]

            task = {
                "task_id": f"synthetic_{i}",
                "question": f"Solve multi-step reasoning task {i}",
                "difficulty": difficulty,
                "category": "reasoning",
                "steps": [f"Step {j}: Intermediate computation" for j in range(num_steps)],
                "answer": f"Result for task {i}",
                "metadata": {"synthetic": True},
            }
            synthetic_tasks.append(task)

        # Convert to HuggingFace dataset format
        from datasets import Dataset as HFDataset

        self._dataset = {"train": HFDataset.from_list(synthetic_tasks)}

    def create_splits(self) -> dict[str, list[TaskSample]]:
        """
        Create train/val/test splits from the dataset.

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if self._dataset is None:
            self.load_dataset()

        # Convert to task samples
        all_samples = []

        if "train" in self._dataset:
            dataset = self._dataset["train"]
        else:
            dataset = self._dataset

        for item in dataset:
            sample = self._convert_to_task_sample(item)
            if sample:
                all_samples.append(sample)

        # Load and merge synthetic data
        synthetic_samples = self.load_synthetic_data()
        if synthetic_samples:
            all_samples.extend(synthetic_samples)
            logger.info(f"Merged {len(synthetic_samples)} synthetic samples into dataset")

        # If no samples were successfully converted, fall back to synthetic data
        if len(all_samples) == 0:
            logger.warning("No valid samples found in dataset, creating synthetic data")
            self._create_synthetic_dataset()
            return self.create_splits()

        # Shuffle with seed
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_samples)

        # Create splits
        n_total = len(all_samples)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)

        self._splits = {
            "train": all_samples[:n_train],
            "val": all_samples[n_train : n_train + n_val],
            "test": all_samples[n_train + n_val :],
        }

        logger.info(
            f"Created splits - Train: {len(self._splits['train'])}, "
            f"Val: {len(self._splits['val'])}, Test: {len(self._splits['test'])}"
        )

        return self._splits

    def _convert_to_task_sample(self, item: dict[str, Any]) -> TaskSample | None:
        """Convert raw dataset item to TaskSample."""
        try:
            # Handle case where item is not a dict
            if not isinstance(item, dict):
                logger.debug(f"Skipping non-dict item: {type(item)}")
                return None

            # Handle different dataset formats
            task_id = item.get("task_id", item.get("id", str(hash(str(item)))))
            task_text = item.get("question", item.get("task", item.get("text", "")))
            difficulty = item.get("difficulty", "medium")
            category = item.get("category", "general")
            steps = item.get("steps", [])
            expected_output = item.get("answer", item.get("output", ""))
            metadata = item.get("metadata", {})

            return TaskSample(
                task_id=str(task_id),
                task_text=task_text,
                difficulty=difficulty,
                category=category,
                steps=steps if isinstance(steps, list) else [steps],
                expected_output=expected_output,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to convert item: {e}")
            return None

    def convert_to_hrm_format(self, sample: TaskSample) -> HRMSample:
        """
        Convert task sample to HRM training format.

        Args:
            sample: TaskSample to convert

        Returns:
            HRMSample for hierarchical decomposition training
        """
        # Create hierarchical decomposition from steps
        decomposition = sample.steps.copy()
        depth = len(decomposition)

        # Generate token-level labels
        # 0: START of subtask, 1: CONTINUE, 2: END
        labels = []
        for step in decomposition:
            step_labels = [0]  # START
            step_labels.extend([1] * (len(step.split()) - 2))  # CONTINUE
            step_labels.append(2)  # END
            labels.extend(step_labels)

        return HRMSample(input_text=sample.task_text, decomposition=decomposition, depth=depth, labels=labels)

    def convert_to_trm_format(self, sample: TaskSample) -> TRMSample:
        """
        Convert task sample to TRM training format.

        Args:
            sample: TaskSample to convert

        Returns:
            TRMSample for task refinement training
        """
        # Create refinement sequence from steps
        refinement_steps = [sample.task_text]
        refinement_steps.extend(sample.steps)

        # Simulate improvement scores (increasing)
        num_steps = len(refinement_steps)
        improvement_scores = [0.5 + 0.5 * (i / num_steps) for i in range(num_steps)]

        return TRMSample(
            initial_task=sample.task_text,
            refinement_steps=refinement_steps[1:],
            final_task=sample.steps[-1] if sample.steps else sample.task_text,
            improvement_scores=improvement_scores,
        )

    def augment_task(self, sample: TaskSample, num_variations: int = 3) -> list[TaskSample]:
        """
        Generate augmented variations of a task.

        Args:
            sample: Original task sample
            num_variations: Number of variations to generate

        Returns:
            List of augmented samples
        """
        augmented = []
        rng = np.random.RandomState(hash(sample.task_id) % (2**32))

        for i in range(num_variations):
            # Create variation with permuted steps (where order doesn't matter)
            new_steps = sample.steps.copy()
            if len(new_steps) > 1:
                # Permute middle steps (keep first and last)
                middle = new_steps[1:-1] if len(new_steps) > 2 else new_steps
                rng.shuffle(middle)
                if len(new_steps) > 2:
                    new_steps = [new_steps[0]] + middle + [new_steps[-1]]

            augmented_sample = TaskSample(
                task_id=f"{sample.task_id}_aug_{i}",
                task_text=sample.task_text,
                difficulty=sample.difficulty,
                category=sample.category,
                steps=new_steps,
                expected_output=sample.expected_output,
                metadata={**sample.metadata, "augmented": True, "variation": i},
            )
            augmented.append(augmented_sample)

        return augmented

    def get_curriculum_batches(self) -> Iterator[list[TaskSample]]:
        """
        Generate batches following curriculum learning (easy → medium → hard).

        Yields:
            Batches of samples ordered by difficulty
        """
        if not self._splits:
            self.create_splits()

        train_samples = self._splits["train"]

        # Group by difficulty
        easy = [s for s in train_samples if s.difficulty == "easy"]
        medium = [s for s in train_samples if s.difficulty == "medium"]
        hard = [s for s in train_samples if s.difficulty == "hard"]

        logger.info(f"Curriculum learning - Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}")

        # Yield in curriculum order
        for difficulty_group in [easy, medium, hard]:
            yield from difficulty_group


class PRIMUSProcessor:
    """Process PRIMUS cybersecurity datasets."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize PRIMUS processor.

        Args:
            config: Configuration for PRIMUS processing
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")

        self.seed_config = config.get("primus_seed", {})
        self.instruct_config = config.get("primus_instruct", {})

        self.seed_path = self.seed_config.get("path", "trendmicro-ailab/Primus-Seed")
        self.instruct_path = self.instruct_config.get("path", "trendmicro-ailab/Primus-Instruct")

        self.cache_dir = Path(self.seed_config.get("cache_dir", "./cache/primus_seed"))
        self.categories = self.seed_config.get("categories", ["mitre", "cyber_companies", "wikipedia"])
        self.max_documents = self.seed_config.get("max_documents")
        self.streaming = self.seed_config.get("streaming", True)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._seed_dataset = None
        self._instruct_dataset = None

        logger.info(f"Initialized PRIMUSProcessor for categories: {self.categories}")

    def load_seed_dataset(self) -> None:
        """Load PRIMUS-Seed document corpus."""
        logger.info(f"Loading PRIMUS-Seed from {self.seed_path}")

        try:
            self._seed_dataset = load_dataset(self.seed_path, cache_dir=str(self.cache_dir), streaming=self.streaming)
            logger.info("Loaded PRIMUS-Seed successfully")

        except Exception as e:
            logger.error(f"Failed to load PRIMUS-Seed: {e}")
            self._create_synthetic_seed()

    def _create_synthetic_seed(self) -> None:
        """Create synthetic PRIMUS-Seed dataset for development."""
        logger.warning("Creating synthetic PRIMUS-Seed dataset")

        synthetic_docs = []
        for i in range(1000):
            category = self.categories[i % len(self.categories)]
            doc = {
                "doc_id": f"synthetic_doc_{i}",
                "text": f"This is synthetic cybersecurity document {i} about {category}. "
                f"It contains information relevant to tactical analysis and threat intelligence. "
                f"The document discusses various aspects of cybersecurity operations.",
                "category": category,
                "source": "synthetic",
                "timestamp": "2024-01-01",
            }
            synthetic_docs.append(doc)

        from datasets import Dataset as HFDataset

        self._seed_dataset = HFDataset.from_list(synthetic_docs)

    def load_instruct_dataset(self) -> None:
        """Load PRIMUS-Instruct instruction samples."""
        logger.info(f"Loading PRIMUS-Instruct from {self.instruct_path}")

        try:
            self._instruct_dataset = load_dataset(
                self.instruct_path,
                cache_dir=str(self.cache_dir / "instruct"),
                streaming=False,  # Small dataset, no need for streaming
            )
            logger.info("Loaded PRIMUS-Instruct successfully")

        except Exception as e:
            logger.error(f"Failed to load PRIMUS-Instruct: {e}")
            self._create_synthetic_instruct()

    def _create_synthetic_instruct(self) -> None:
        """Create synthetic PRIMUS-Instruct dataset for development."""
        logger.warning("Creating synthetic PRIMUS-Instruct dataset")

        synthetic_instruct = []
        for i in range(100):
            sample = {
                "instruction": f"Analyze the cybersecurity threat scenario {i}",
                "input": f"Context for scenario {i}: A potential security breach...",
                "output": "Analysis: Based on the provided context, the threat level is moderate...",
            }
            synthetic_instruct.append(sample)

        from datasets import Dataset as HFDataset

        self._instruct_dataset = {"train": HFDataset.from_list(synthetic_instruct)}

    def stream_documents(self) -> Iterator[DocumentChunk]:
        """
        Stream documents from PRIMUS-Seed for memory-efficient processing.

        Yields:
            DocumentChunk objects
        """
        if self._seed_dataset is None:
            self.load_seed_dataset()

        chunk_id = 0
        for doc in self._seed_dataset:
            # Filter by category if specified
            doc_category = doc.get("category", "general")
            if self.categories and doc_category not in self.categories:
                continue

            # Chunk the document
            text = doc.get("text", doc.get("content", ""))
            chunks = self._chunk_text(text, chunk_size=512, overlap=50)

            for chunk_text in chunks:
                yield DocumentChunk(
                    doc_id=doc.get("doc_id", str(chunk_id)),
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "category": doc_category,
                        "source": doc.get("source", "primus"),
                        "timestamp": doc.get("timestamp", ""),
                    },
                )
                chunk_id += 1

                if self.max_documents and chunk_id >= self.max_documents:
                    return

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """
        Chunk text into smaller pieces with overlap.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for boundary in [". ", "! ", "? ", "\n"]:
                    last_boundary = text.rfind(boundary, start, end)
                    if last_boundary > start + chunk_size // 2:
                        end = last_boundary + len(boundary)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def get_instruction_samples(self) -> list[dict[str, str]]:
        """
        Get PRIMUS-Instruct samples for instruction fine-tuning.

        Returns:
            List of instruction samples
        """
        if self._instruct_dataset is None:
            self.load_instruct_dataset()

        samples = []
        dataset = self._instruct_dataset.get("train", self._instruct_dataset)

        for item in dataset:
            sample = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
            }
            samples.append(sample)

        logger.info(f"Retrieved {len(samples)} instruction samples")
        return samples

    def extract_domain_metadata(self, chunk: DocumentChunk) -> dict[str, Any]:
        """
        Extract domain-specific metadata from document chunk.

        Args:
            chunk: Document chunk to analyze

        Returns:
            Dictionary of extracted metadata
        """
        text = chunk.text.lower()

        metadata = {
            "has_mitre_reference": any(term in text for term in ["att&ck", "mitre", "t1"]),
            "has_cve_reference": "cve-" in text,
            "threat_level": self._estimate_threat_level(text),
            "domain": chunk.metadata.get("category", "general"),
            "text_length": len(chunk.text),
            "word_count": len(chunk.text.split()),
        }

        return metadata

    def _estimate_threat_level(self, text: str) -> str:
        """Estimate threat level based on keywords."""
        high_threat = ["critical", "severe", "exploit", "vulnerability", "breach"]
        medium_threat = ["warning", "suspicious", "anomaly", "risk"]

        text_lower = text.lower()

        if any(term in text_lower for term in high_threat):
            return "high"
        elif any(term in text_lower for term in medium_threat):
            return "medium"
        else:
            return "low"


class DataOrchestrator:
    """Coordinate data loading and batching across all datasets."""

    def __init__(self, config_path: str = "training/config.yaml"):
        """
        Initialize data orchestrator.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.dabstep_loader = DABStepLoader(self.config["data"]["dabstep"])
        self.primus_processor = PRIMUSProcessor(self.config["data"])

        self.batch_size = self.config["training"]["batch_size"]
        self.seed = self.config["data"]["dabstep"]["seed"]

        self._prepared = False

        logger.info("Initialized DataOrchestrator")

    def prepare_data(self) -> None:
        """Load and prepare all datasets."""
        logger.info("Preparing data...")

        # Load DABStep
        self.dabstep_loader.load_dataset()
        self.dabstep_loader.create_splits()

        # Load PRIMUS datasets
        self.primus_processor.load_seed_dataset()
        self.primus_processor.load_instruct_dataset()

        self._prepared = True
        logger.info("Data preparation complete")

    def get_hrm_dataloader(self, split: str = "train") -> DataLoader:
        """
        Get DataLoader for HRM training.

        Args:
            split: Dataset split ('train', 'val', 'test')

        Returns:
            PyTorch DataLoader
        """
        if not self._prepared:
            self.prepare_data()

        samples = self.dabstep_loader._splits[split]
        hrm_samples = [self.dabstep_loader.convert_to_hrm_format(s) for s in samples]

        dataset = HRMDataset(hrm_samples)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.get("resources", {}).get("max_cpu_workers", 2),
            pin_memory=self.config.get("resources", {}).get("pin_memory", True),
            collate_fn=hrm_collate_fn,
        )

    def get_trm_dataloader(self, split: str = "train") -> DataLoader:
        """
        Get DataLoader for TRM training.

        Args:
            split: Dataset split

        Returns:
            PyTorch DataLoader
        """
        if not self._prepared:
            self.prepare_data()

        samples = self.dabstep_loader._splits[split]
        trm_samples = [self.dabstep_loader.convert_to_trm_format(s) for s in samples]

        dataset = TRMDataset(trm_samples)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.get("resources", {}).get("max_cpu_workers", 2),
            pin_memory=self.config.get("resources", {}).get("pin_memory", True),
            collate_fn=trm_collate_fn,
        )

    def get_rag_documents(self) -> Iterator[DocumentChunk]:
        """
        Get document chunks for RAG index building.

        Yields:
            DocumentChunk objects
        """
        return self.primus_processor.stream_documents()

    def get_instruction_samples(self) -> list[dict[str, str]]:
        """Get instruction tuning samples."""
        return self.primus_processor.get_instruction_samples()

    def create_balanced_batch(
        self, hrm_samples: list[HRMSample], trm_samples: list[TRMSample], instruction_samples: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Create a balanced batch across different training objectives.

        Args:
            hrm_samples: HRM training samples
            trm_samples: TRM training samples
            instruction_samples: Instruction tuning samples

        Returns:
            Dictionary containing balanced batch
        """
        batch = {
            "hrm": hrm_samples[: self.batch_size // 3],
            "trm": trm_samples[: self.batch_size // 3],
            "instruction": instruction_samples[: self.batch_size // 3],
        }
        return batch

    def get_data_statistics(self) -> dict[str, Any]:
        """
        Get statistics about loaded datasets.

        Returns:
            Dictionary with dataset statistics
        """
        if not self._prepared:
            self.prepare_data()

        stats = {
            "dabstep": {
                "train_samples": len(self.dabstep_loader._splits.get("train", [])),
                "val_samples": len(self.dabstep_loader._splits.get("val", [])),
                "test_samples": len(self.dabstep_loader._splits.get("test", [])),
                "difficulties": {},
            },
            "primus_instruct": {"total_samples": len(self.get_instruction_samples())},
        }

        # Count by difficulty
        for _split_name, samples in self.dabstep_loader._splits.items():
            for sample in samples:
                diff = sample.difficulty
                if diff not in stats["dabstep"]["difficulties"]:
                    stats["dabstep"]["difficulties"][diff] = 0
                stats["dabstep"]["difficulties"][diff] += 1

        return stats


class HRMDataset(Dataset):
    """PyTorch Dataset for HRM training."""

    def __init__(self, samples: list[HRMSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        return {
            "input_text": sample.input_text,
            "decomposition": sample.decomposition,
            "depth": sample.depth,
            "labels": torch.tensor(sample.labels[:512], dtype=torch.long),  # Truncate to max length
        }


class TRMDataset(Dataset):
    """PyTorch Dataset for TRM training."""

    def __init__(self, samples: list[TRMSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        return {
            "initial_task": sample.initial_task,
            "refinement_steps": sample.refinement_steps,
            "final_task": sample.final_task,
            "improvement_scores": torch.tensor(sample.improvement_scores, dtype=torch.float32),
        }


def hrm_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function for HRM batches that handles variable-length label tensors.

    Args:
        batch: List of samples from HRMDataset

    Returns:
        Batched dictionary with padded label tensors
    """
    # Find max label length in batch
    max_label_length = max(len(sample["labels"]) for sample in batch)

    # Pad labels to max length
    padded_labels = []
    for sample in batch:
        labels = sample["labels"]
        if len(labels) < max_label_length:
            # Pad with -100 (PyTorch's ignore_index for CrossEntropyLoss)
            padding = torch.full((max_label_length - len(labels),), -100, dtype=torch.long)
            padded_labels.append(torch.cat([labels, padding]))
        else:
            padded_labels.append(labels)

    return {
        "input_text": [sample["input_text"] for sample in batch],
        "decomposition": [sample["decomposition"] for sample in batch],
        "depth": [sample["depth"] for sample in batch],
        "labels": torch.stack(padded_labels),
    }


def trm_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function for TRM batches that handles variable-length improvement score tensors.

    Args:
        batch: List of samples from TRMDataset

    Returns:
        Batched dictionary with padded improvement score tensors
    """
    # Find max score length in batch
    max_score_length = max(len(sample["improvement_scores"]) for sample in batch)

    # Pad scores to max length
    padded_scores = []
    for sample in batch:
        scores = sample["improvement_scores"]
        if len(scores) < max_score_length:
            # Pad with 0.0
            padding = torch.zeros(max_score_length - len(scores), dtype=torch.float32)
            padded_scores.append(torch.cat([scores, padding]))
        else:
            padded_scores.append(scores)

    return {
        "initial_task": [sample["initial_task"] for sample in batch],
        "refinement_steps": [sample["refinement_steps"] for sample in batch],
        "final_task": [sample["final_task"] for sample in batch],
        "improvement_scores": torch.stack(padded_scores),
    }


class TrainingDataset(Dataset):
    """Combined dataset for multi-objective training."""

    def __init__(
        self, hrm_samples: list[HRMSample], trm_samples: list[TRMSample], instruction_samples: list[dict[str, str]]
    ):
        self.hrm_samples = hrm_samples
        self.trm_samples = trm_samples
        self.instruction_samples = instruction_samples

        # Create unified index
        self.sample_map = []
        for i, _ in enumerate(hrm_samples):
            self.sample_map.append(("hrm", i))
        for i, _ in enumerate(trm_samples):
            self.sample_map.append(("trm", i))
        for i, _ in enumerate(instruction_samples):
            self.sample_map.append(("instruction", i))

    def __len__(self) -> int:
        return len(self.sample_map)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_type, sample_idx = self.sample_map[idx]

        if sample_type == "hrm":
            sample = self.hrm_samples[sample_idx]
            return {
                "type": "hrm",
                "input_text": sample.input_text,
                "decomposition": sample.decomposition,
                "depth": sample.depth,
                "labels": torch.tensor(sample.labels[:512], dtype=torch.long),
            }
        elif sample_type == "trm":
            sample = self.trm_samples[sample_idx]
            return {
                "type": "trm",
                "initial_task": sample.initial_task,
                "refinement_steps": sample.refinement_steps,
                "final_task": sample.final_task,
                "improvement_scores": torch.tensor(sample.improvement_scores, dtype=torch.float32),
            }
        else:  # instruction
            sample = self.instruction_samples[sample_idx]
            return {
                "type": "instruction",
                "instruction": sample["instruction"],
                "input": sample["input"],
                "output": sample["output"],
            }


if __name__ == "__main__":
    # Test the data pipeline
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Data Pipeline Module")

    # Test DABStep loader
    dabstep_config = {
        "path": "adyen/DABstep",
        "cache_dir": "./cache/dabstep",
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "seed": 42,
    }

    loader = DABStepLoader(dabstep_config)
    splits = loader.create_splits()

    logger.info(f"Created splits: {[(k, len(v)) for k, v in splits.items()]}")

    # Test format conversions
    if splits["train"]:
        sample = splits["train"][0]
        hrm_sample = loader.convert_to_hrm_format(sample)
        trm_sample = loader.convert_to_trm_format(sample)

        logger.info(f"HRM sample depth: {hrm_sample.depth}")
        logger.info(f"TRM refinement steps: {len(trm_sample.refinement_steps)}")

    # Test PRIMUS processor
    primus_config = {
        "primus_seed": {
            "path": "trendmicro-ailab/Primus-Seed",
            "cache_dir": "./cache/primus_seed",
            "categories": ["mitre", "cyber_companies"],
            "streaming": True,
        },
        "primus_instruct": {"path": "trendmicro-ailab/Primus-Instruct"},
    }

    processor = PRIMUSProcessor(primus_config)

    # Stream first 5 documents
    for count, chunk in enumerate(processor.stream_documents()):
        logger.info(f"Document chunk {chunk.chunk_id}: {len(chunk.text)} chars")
        if count >= 4:  # 0-indexed, so 4 means 5 documents
            break

    logger.info("Data Pipeline Module test complete")
